//
// algo_spline_knot_dp.cpp — global-optimal piecewise-linear curve via
// dynamic programming over knot positions, with curvature regularization.
//
// Goal. Find the curve y = f(x) within a given band that maximises the
// integrated edge evidence ∫ E(x, f(x)) dx. RANSAC approximates this
// problem with random sampling; this algorithm solves it exactly within
// a (knot, y-bin) discretisation, by reformulating it as a Viterbi-style
// DP whose state encodes the last two knots so that curvature can be
// penalised.
//
// Why DP works here. For a global polynomial fit y = p_θ(x), the
// objective S(θ) = ∫ E(x, p_θ(x)) dx couples every coordinate of θ to
// the entire x-range — DP cannot decompose it left-to-right. Spline /
// piecewise-linear curves with K knots make S(θ) decomposable: the
// segment between knots i and i+1 depends only on (y_i, y_{i+1}), and
// the curvature penalty between knots i-1, i, i+1 depends only on
// (y_{i-1}, y_i, y_{i+1}). State = (y_{i-1}, y_i); transitions consider
// every y_{i+1}. This is *exact* in the discretisation — no random
// restart, no local-minimum trap.
//
// Pipeline:
//   1. Compute |∂I/∂y| evidence map, restricted to a ±BAND_HALF band
//      around gt.y0.
//   2. Place K knots at evenly-spaced x. Discretise y at each knot
//      to Y_BINS values within ±Y_RANGE_PX of y0.
//   3. Pre-compute seg[k][a][b] = ∫ E along the line segment from
//      (x_k, y_a) to (x_{k+1}, y_b), sampled at unit-x spacing with
//      bilinear y interpolation.
//   4. DP over (y_{i-1}, y_i):
//        dp[i+1][b][c] = max_a {dp[i][a][b] + seg[i][b][c]
//                                - λ·(c − 2b + a)²}
//   5. Find arg-max at last knot, backtrack, emit a linear-interpolated
//      curve at every integer x.
//
// The output is the *globally optimal* knot path (within the
// discretisation grid) — no random seeds, no parameter starts.
//

#include "common.hpp"
#include "constrained_poly_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lab {

namespace {

// ---- Tunables -------------------------------------------------------------
//
// K_KNOTS       — number of control points along x. 15 fits all curve
//                 families in the lab benchmark (line / sine / cubic /
//                 arc) without aliasing. Knot spacing ≈ W/(K-1) ≈ 23 px.
// Y_BINS        — y-discretisation resolution. 80 bins over ±40 px gives
//                 1 px per bin, sub-pixel refined at the end via parabolic
//                 interpolation on the dp values.
// Y_RANGE_PX    — half-width of the y-search band, in pixels around y0.
// BAND_HALF_PX  — vertical band restriction for the evidence map (cuts
//                 the sampled image area).
// LAMBDA_CURV   — curvature penalty weight. Acts on (Δ²y_bins)² per
//                 knot-triple. Tuned so a 1-bin (≈1 px) deviation costs
//                 about 5% of a typical segment's edge integral —
//                 strong enough to suppress spike-induced kinks but
//                 weak enough to admit real curvature.
constexpr int    K_KNOTS      = 20;
constexpr int    Y_BINS       = 160;
constexpr double Y_RANGE_PX   = 40.0;
constexpr int    BAND_HALF_PX = 50;
// Scaled so the physical curvature penalty (in px²) stays constant
// across discretisations: LAMBDA = 0.6 · (1.0 / dy_per_bin)².
// M=80 → dy=1.0 → λ=0.6   (original spline_knot_dp tune)
// M=160 → dy=0.5 → λ=0.15
constexpr double LAMBDA_CURV  = 0.15;
constexpr int    MAX_DELTA    = 20;  // ≈10 px / 0.50 px-per-bin
// MAX_DELTA mirrors dijkstra's 3-neighbour rule: transitions outside
// this window are pruned both for speed (M³ → M²·(2·MAX_DELTA+1)) and
// to prevent spike-chain detours that would require unrealistic slopes.
// Scaled with Y_BINS so the physical max-slope budget stays ≈ 10 px
// across a knot spacing of W/(K-1) ≈ 17 px (max slope ≈ 0.6 px/px).

// Map y-bin index ∈ [0, Y_BINS) to absolute y in image coordinates.
inline double y_of_bin(int b, double y0) {
    return y0 - Y_RANGE_PX
         + 2.0 * Y_RANGE_PX * (double)b / (double)(Y_BINS - 1);
}

// Evidence map + bilinear sampler now live in constrained_poly_common.hpp
// (cpoly::compute_saturated_evidence, cpoly::sample_y). The 5-tap σ=1
// Gaussian-derivative kernel was tested but blurs the sub-pixel peak
// (RMS p50 worsens 0.161 → 0.206 on normal); kept as 3-tap central.

// Integrate E along a line from (xa, ya) to (xb, yb), sampled at every
// integer x between xa and xb.
double integrate_line(const cv::Mat& E, double xa, double ya,
                      double xb, double yb, int band_lo, int band_hi) {
    int x_lo = std::max(0, (int)std::ceil(xa));
    int x_hi = std::min(E.cols - 1, (int)std::floor(xb));
    if (x_lo > x_hi) return 0.0;
    double slope = (yb - ya) / std::max(1e-9, xb - xa);
    double sum = 0.0;
    for (int x = x_lo; x <= x_hi; ++x) {
        double y = ya + slope * ((double)x - xa);
        sum += cpoly::sample_y(E, x, y, band_lo, band_hi);
    }
    return sum;
}

} // anonymous namespace

std::vector<cv::Point2d> detect_spline_knot_dp(const cv::Mat& gray,
                                               const GroundTruth& gt) {
    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;

    // Evidence map + band limits.
    cv::Mat E = cpoly::compute_saturated_evidence(gray, y0, BAND_HALF_PX);
    const int band_lo = std::max(1,     (int)std::round(y0 - BAND_HALF_PX));
    const int band_hi = std::min(H - 2, (int)std::round(y0 + BAND_HALF_PX));

    // Knot x-positions, evenly spaced including both endpoints.
    std::vector<double> xs(K_KNOTS);
    for (int i = 0; i < K_KNOTS; ++i) {
        xs[i] = (double)(W - 1) * (double)i / (double)(K_KNOTS - 1);
    }

    // Flat-array layout for cache locality and clean OpenMP.
    //   seg [k][a][b]   = seg_flat [k*MM + a*M + b]   (K-1 × M × M doubles)
    //   dp  [i][a][b]   = dp_flat  [i*MM + a*M + b]   (K   × M × M doubles)
    //   pred[i][a][b]   = pred_flat[i*MM + a*M + b]   (K   × M × M int16)
    constexpr int M  = Y_BINS;
    constexpr int MM = M * M;
    std::vector<double>  seg_flat ((K_KNOTS - 1) * MM, 0.0);
    std::vector<double>  dp_flat  (K_KNOTS * MM, -1e30);
    std::vector<int16_t> pred_flat(K_KNOTS * MM, -1);

    // Segment precompute, parallel over k. Each line integral is pure
    // (reads E only), so threads do not race.
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < K_KNOTS - 1; ++k) {
        double* sk = &seg_flat[(size_t)k * MM];
        double xa = xs[k], xb = xs[k+1];
        for (int a = 0; a < M; ++a) {
            double ya = y_of_bin(a, y0);
            double* row = sk + (size_t)a * M;
            int b_lo = std::max(0, a - MAX_DELTA);
            int b_hi = std::min(M - 1, a + MAX_DELTA);
            for (int b = b_lo; b <= b_hi; ++b) {
                row[b] = integrate_line(E, xa, ya,
                                        xb, y_of_bin(b, y0),
                                        band_lo, band_hi);
            }
        }
    }

    constexpr double NEG_INF = -1e30;

    // Seed at i=1: dp[1][a][b] = seg[0][a][b] (no curvature term yet —
    // need three consecutive knots for a curvature triple). Skip
    // (a, b) pairs that exceed the per-knot Δy budget.
    {
        const double* s0 = &seg_flat[0];
        double*       d1 = &dp_flat[(size_t)1 * MM];
        for (int a = 0; a < M; ++a) {
            int b_lo = std::max(0, a - MAX_DELTA);
            int b_hi = std::min(M - 1, a + MAX_DELTA);
            for (int b = b_lo; b <= b_hi; ++b)
                d1[a * M + b] = s0[a * M + b];
        }
    }

    // Forward pass over flat arrays.  Loops restructured with **b
    // outermost** so each thread owns disjoint dp[i+1][b][·] slices —
    // safe to parallelise with OpenMP without atomics.  The loop
    // structure is:
    //
    //   for b in [0, M):                       // parallel-friendly
    //     for a in [b-Δ, b+Δ]:                 // collect candidates
    //       for c in [b-Δ, b+Δ]:               // emit transitions
    //         dp[i+1][b][c] = max(..., base + seg[i][b][c] - λ·curv²)
    //
    // Per-thread cache footprint:  dp[i][·][b] (1 column),
    //   seg[i][b][·] (1 row), dp[i+1][b][·] (1 row).  All contiguous
    //   along the inner index → vectoriser-friendly.
    for (int i = 1; i < K_KNOTS - 1; ++i) {
        const double* dpi   = &dp_flat[(size_t)i * MM];
        const double* segi  = &seg_flat[(size_t)i * MM];
        double*       dpi1  = &dp_flat[(size_t)(i + 1) * MM];
        int16_t*      predi = &pred_flat[(size_t)(i + 1) * MM];
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < M; ++b) {
            const double* segi_b = segi + (size_t)b * M;
            double*       dpi1_b = dpi1 + (size_t)b * M;
            int16_t*      pred_b = predi + (size_t)b * M;
            int a_lo = std::max(0, b - MAX_DELTA);
            int a_hi = std::min(M - 1, b + MAX_DELTA);
            int c_lo = std::max(0, b - MAX_DELTA);
            int c_hi = std::min(M - 1, b + MAX_DELTA);
            for (int a = a_lo; a <= a_hi; ++a) {
                double base = dpi[(size_t)a * M + b];
                if (base <= NEG_INF / 2) continue;
                int two_b_minus_a = 2 * b - a;
                for (int c = c_lo; c <= c_hi; ++c) {
                    double curv  = (double)(c - two_b_minus_a);
                    double score = base + segi_b[c] - LAMBDA_CURV * curv * curv;
                    if (score > dpi1_b[c]) {
                        dpi1_b[c] = score;
                        pred_b[c] = (int16_t)a;
                    }
                }
            }
        }
    }

    // Find best (a, b) at the final knot, using the flat dp array.
    auto dp_at = [&](int i, int a, int b) -> double& {
        return dp_flat[(size_t)i * MM + (size_t)a * M + b];
    };
    auto pred_at = [&](int i, int a, int b) -> int16_t& {
        return pred_flat[(size_t)i * MM + (size_t)a * M + b];
    };
    double best = NEG_INF;
    int best_a = 0, best_b = 0;
    for (int a = 0; a < M; ++a) {
        for (int b = 0; b < M; ++b) {
            double v = dp_at(K_KNOTS - 1, a, b);
            if (v > best) { best = v; best_a = a; best_b = b; }
        }
    }

    // Backtrack to recover the K-knot path.
    std::vector<int> path(K_KNOTS, 0);
    path[K_KNOTS - 1] = best_b;
    path[K_KNOTS - 2] = best_a;
    for (int i = K_KNOTS - 1; i >= 2; --i) {
        path[i - 2] = pred_at(i, path[i - 1], path[i]);
    }

    // Sub-pixel refinement.  Two passes of Gauss-Seidel on the K
    // knot positions: with neighbours held fixed, sample the
    // evidence integral at y_path[i] − dy, y_path[i], y_path[i] + dy
    // (where dy = dy_per_bin), parabolic-interpolate to find the
    // local maximum, and update y_path[i].  Two sweeps converge to
    // sub-bin precision because adjacent updates re-couple through
    // the next knot's neighbour-fixed score.  Replaces the earlier
    // dp-value-based parabolic interpolation, which conflated
    // accumulated curvature penalty with local evidence.
    std::vector<double> y_path(K_KNOTS);
    for (int i = 0; i < K_KNOTS; ++i) y_path[i] = y_of_bin(path[i], y0);
    const double dy_per_bin = 2.0 * Y_RANGE_PX / (double)(M - 1);

    auto score_at_y = [&](int i, double yi) -> double {
        double s = 0.0;
        if (i >= 1) {
            s += integrate_line(E, xs[i-1], y_path[i-1], xs[i], yi,
                                band_lo, band_hi);
        }
        if (i < K_KNOTS - 1) {
            s += integrate_line(E, xs[i], yi, xs[i+1], y_path[i+1],
                                band_lo, band_hi);
        }
        return s;
    };
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (int i = 0; i < K_KNOTS; ++i) {
            double yc = y_path[i];
            double sm = score_at_y(i, yc - dy_per_bin);
            double s0 = score_at_y(i, yc);
            double sp = score_at_y(i, yc + dy_per_bin);
            double denom = sm - 2.0 * s0 + sp;
            if (std::abs(denom) > 1e-12) {
                double off = 0.5 * (sm - sp) / denom;
                if (off > -1.0 && off < 1.0) {
                    y_path[i] = yc + off * dy_per_bin;
                }
            }
        }
    }

    // Emit dense output: linear interpolation between adjacent knots.
    // Output: linear interpolation between adjacent knots — matches
    // the segment evidence integral the DP actually optimised.
    //
    // Tested natural cubic spline post-fit through the K knots: although
    // C² and visually smoother, the cubic *overshoots between knots*
    // when knot y-values carry sub-pixel noise, and the overshoot is
    // off-curve evidence the DP never accounted for. RMS p50 worsens
    // 0.161 → 0.189 on normal noise. To use a cubic output coherently
    // would require recomputing segment evidence under the cubic too —
    // which couples non-adjacent knots and breaks the local DP.
    std::vector<cv::Point2d> out;
    out.reserve(W);
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        int k = (int)std::floor(xd * (K_KNOTS - 1) / (double)(W - 1));
        if (k > K_KNOTS - 2) k = K_KNOTS - 2;
        if (k < 0) k = 0;
        double t = (xd - xs[k]) / std::max(1e-9, xs[k+1] - xs[k]);
        double y = (1.0 - t) * y_path[k] + t * y_path[k+1];
        out.emplace_back(xd, y);
    }
    return out;
}

} // namespace lab
