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

// Polarity-aware, *saturating* evidence map. Pipeline:
//   1. signed ∂I/∂y per pixel (3-tap central; tested 5-tap σ=1.0
//      Gaussian-derivative, but 5-tap blurs the sub-pixel peak in
//      y so RMS p50 worsens 0.161 → 0.206 on normal noise — only
//      worth it if you specifically need spike-tolerance over
//      sub-pixel precision);
//   2. polarity ReLU using the band's dominant sign;
//   3. saturating normalisation E ← E / (E + κ_E) so a single
//      bright pixel cannot dwarf a long stretch of moderate
//      evidence. This mimics the 1/evidence cost dijkstra uses:
//      spike chains earn high evidence only at sparse peaks and
//      can no longer outscore a continuous moderately-strong curve.
cv::Mat compute_evidence(const cv::Mat& gray, double y0, int band_half) {
    const int H = gray.rows, W = gray.cols;
    const int y_lo = std::max(1,     (int)std::round(y0 - band_half));
    const int y_hi = std::min(H - 2, (int)std::round(y0 + band_half));

    long long top_sum = 0, bot_sum = 0;
    for (int x = 0; x < W; ++x) {
        top_sum += gray.ptr<uint8_t>(y_lo)[x];
        bot_sum += gray.ptr<uint8_t>(y_hi)[x];
    }
    const double pol = (bot_sum >= top_sum) ? +1.0 : -1.0;

    cv::Mat raw(H, W, CV_64F, cv::Scalar(0.0));
    double e_max = 0.0;
    for (int y = y_lo; y <= y_hi; ++y) {
        const uint8_t* rm = gray.ptr<uint8_t>(y - 1);
        const uint8_t* rp = gray.ptr<uint8_t>(y + 1);
        double* dst = raw.ptr<double>(y);
        for (int x = 0; x < W; ++x) {
            double g = pol * 0.5 * ((double)rp[x] - (double)rm[x]);
            double v = (g > 0.0) ? g : 0.0;
            dst[x] = v;
            if (v > e_max) e_max = v;
        }
    }
    // κ chosen as 0.20 × max — makes a single peak max-sample to ~0.83
    // and a moderate curve sample (e ≈ 0.5 max) to 0.71, so curve and
    // spike contributions stay within ~20% of each other.
    const double kappa = std::max(1.0, 0.20 * e_max);

    cv::Mat E(H, W, CV_64F, cv::Scalar(0.0));
    for (int y = y_lo; y <= y_hi; ++y) {
        const double* src = raw.ptr<double>(y);
        double* dst = E.ptr<double>(y);
        for (int x = 0; x < W; ++x) {
            dst[x] = src[x] / (src[x] + kappa);
        }
    }
    return E;
}

// Bilinear sample of a CV_64F image at (x, y). x is integer (we always
// sample at unit-x); y is sub-pixel. Returns 0 outside the y-band.
inline double sample_y(const cv::Mat& E, int x, double y, int band_lo, int band_hi) {
    int yi = (int)std::floor(y);
    if (yi < band_lo || yi >= band_hi) return 0.0;
    double dy = y - (double)yi;
    const double* row0 = E.ptr<double>(yi);
    const double* row1 = E.ptr<double>(yi + 1);
    return row0[x] * (1.0 - dy) + row1[x] * dy;
}

// Integrate E along a line from (xa, ya) to (xb, yb), sampled at every
// integer x between xa and xb. Endpoints get half-weight (trapezoid).
double integrate_line(const cv::Mat& E, double xa, double ya,
                      double xb, double yb, int band_lo, int band_hi) {
    int x_lo = std::max(0, (int)std::ceil(xa));
    int x_hi = std::min(E.cols - 1, (int)std::floor(xb));
    if (x_lo > x_hi) return 0.0;
    double slope = (yb - ya) / std::max(1e-9, xb - xa);
    double sum = 0.0;
    for (int x = x_lo; x <= x_hi; ++x) {
        double y = ya + slope * ((double)x - xa);
        sum += sample_y(E, x, y, band_lo, band_hi);
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
    cv::Mat E = compute_evidence(gray, y0, BAND_HALF_PX);
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

    // Forward pass over flat arrays. For each (i, b), we collect the
    // best (a → b → c) triple — but inverting the inner loops to fix
    // (b, c) and scan a is more cache-friendly because seg[i][b][c]
    // becomes a single load and dp[i][a][b] is contiguous in a along
    // a slice of constant b. Also pre-tabulate the curvature penalty
    // p[a, c] = LAMBDA · (c - 2b + a)² as a function of (a) for fixed
    // (b, c) — actually faster to inline since the squared term is
    // cheap and we avoid an extra table.
    for (int i = 1; i < K_KNOTS - 1; ++i) {
        const double* dpi   = &dp_flat[(size_t)i * MM];
        const double* segi  = &seg_flat[(size_t)i * MM];
        double*       dpi1  = &dp_flat[(size_t)(i + 1) * MM];
        int16_t*      predi = &pred_flat[(size_t)(i + 1) * MM];
        for (int a = 0; a < M; ++a) {
            const double* dpi_a = dpi + (size_t)a * M;
            int b_lo_outer = std::max(0, a - MAX_DELTA);
            int b_hi_outer = std::min(M - 1, a + MAX_DELTA);
            for (int b = b_lo_outer; b <= b_hi_outer; ++b) {
                double base = dpi_a[b];
                if (base <= NEG_INF / 2) continue;
                const double* segi_b = segi + (size_t)b * M;
                double*       dpi1_b = dpi1 + (size_t)b * M;
                int16_t*      pred_b = predi + (size_t)b * M;
                int c_lo = std::max(0, b - MAX_DELTA);
                int c_hi = std::min(M - 1, b + MAX_DELTA);
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

    // Sub-pixel refinement: parabolic interpolation across y-bins of
    // the dp value at the recovered triple.
    auto refine_bin = [](double d_prev, double d_curr, double d_next) {
        double denom = d_prev - 2.0 * d_curr + d_next;
        if (std::abs(denom) < 1e-12) return 0.0;
        double off = 0.5 * (d_prev - d_next) / denom;
        if (off < -1.0 || off > 1.0) return 0.0;
        return off;
    };
    std::vector<double> y_path(K_KNOTS);
    for (int i = 0; i < K_KNOTS; ++i) {
        int b = path[i];
        double off = 0.0;
        if (i >= 1 && b > 0 && b < M - 1) {
            int a = path[i - 1];
            off = refine_bin(dp_at(i, a, b - 1),
                             dp_at(i, a, b),
                             dp_at(i, a, b + 1));
        }
        y_path[i] = y_of_bin(b, y0) + off
                  * (2.0 * Y_RANGE_PX / (double)(M - 1));
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
