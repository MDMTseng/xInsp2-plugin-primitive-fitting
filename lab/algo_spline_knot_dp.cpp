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
constexpr int    K_KNOTS      = 25;
constexpr int    Y_BINS       = 80;
constexpr double Y_RANGE_PX   = 40.0;
constexpr int    BAND_HALF_PX = 50;
constexpr double LAMBDA_CURV  = 0.6;
// Maximum |Δy| (in bins) allowed between adjacent knots. Mirrors
// dijkstra's 3-neighbour rule: transitions outside this window are
// pruned both for speed (M³ → M²·(2·MAX_DELTA+1)) and to prevent
// spike-chain detours that would require unrealistic slopes.
// 10 bins ≈ 10 px over a knot spacing of W/(K-1) ≈ 13 px, i.e.
// max slope ≈ 0.77 px/px — safely above any curve produced by the
// scene generator (amp 28, period ≥ 320 ⇒ max slope ≈ 0.55 px/px).
constexpr int    MAX_DELTA    = 10;

// Map y-bin index ∈ [0, Y_BINS) to absolute y in image coordinates.
inline double y_of_bin(int b, double y0) {
    return y0 - Y_RANGE_PX
         + 2.0 * Y_RANGE_PX * (double)b / (double)(Y_BINS - 1);
}

// Polarity-aware, *saturating* evidence map. Pipeline:
//   1. signed ∂I/∂y per pixel (3-tap central);
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

    // First pass — signed-positive raw gradient + global stats for κ.
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

    // Pre-compute segment integrals seg[k][a][b].
    std::vector<std::vector<std::vector<double>>> seg(
        K_KNOTS - 1,
        std::vector<std::vector<double>>(Y_BINS, std::vector<double>(Y_BINS, 0.0)));
    for (int k = 0; k < K_KNOTS - 1; ++k) {
        for (int a = 0; a < Y_BINS; ++a) {
            double ya = y_of_bin(a, y0);
            for (int b = 0; b < Y_BINS; ++b) {
                double yb = y_of_bin(b, y0);
                seg[k][a][b] = integrate_line(E, xs[k], ya, xs[k+1], yb,
                                              band_lo, band_hi);
            }
        }
    }

    // DP table dp[i][a][b] = best score reaching knot i with
    //   y_{i-1} = bin a, y_i = bin b.
    // Predecessor pred[i][a][b] = bin index of y_{i-2}.
    constexpr double NEG_INF = -1e30;
    std::vector<std::vector<std::vector<double>>> dp(
        K_KNOTS,
        std::vector<std::vector<double>>(Y_BINS, std::vector<double>(Y_BINS, NEG_INF)));
    std::vector<std::vector<std::vector<int16_t>>> pred(
        K_KNOTS,
        std::vector<std::vector<int16_t>>(Y_BINS, std::vector<int16_t>(Y_BINS, -1)));

    // Seed at i=1: dp[1][a][b] = seg[0][a][b] (no curvature term yet —
    // need three consecutive knots for a curvature triple). Skip
    // (a, b) pairs that exceed the per-knot Δy budget.
    for (int a = 0; a < Y_BINS; ++a) {
        int b_lo = std::max(0, a - MAX_DELTA);
        int b_hi = std::min(Y_BINS - 1, a + MAX_DELTA);
        for (int b = b_lo; b <= b_hi; ++b)
            dp[1][a][b] = seg[0][a][b];
    }

    // Forward pass: i = 1 .. K_KNOTS-2 transitions to i+1.
    // Transition c is restricted to |c − b| ≤ MAX_DELTA so the jump
    // between adjacent knots respects the maximum realistic slope.
    for (int i = 1; i < K_KNOTS - 1; ++i) {
        for (int a = 0; a < Y_BINS; ++a) {
            for (int b = 0; b < Y_BINS; ++b) {
                double base = dp[i][a][b];
                if (base <= NEG_INF / 2) continue;
                int c_lo = std::max(0, b - MAX_DELTA);
                int c_hi = std::min(Y_BINS - 1, b + MAX_DELTA);
                for (int c = c_lo; c <= c_hi; ++c) {
                    double curv = (double)(c - 2 * b + a);
                    double score = base + seg[i][b][c]
                                 - LAMBDA_CURV * curv * curv;
                    if (score > dp[i+1][b][c]) {
                        dp[i+1][b][c] = score;
                        pred[i+1][b][c] = (int16_t)a;
                    }
                }
            }
        }
    }

    // Find best (a, b) at the final knot.
    double best = NEG_INF;
    int best_a = 0, best_b = 0;
    for (int a = 0; a < Y_BINS; ++a) {
        for (int b = 0; b < Y_BINS; ++b) {
            if (dp[K_KNOTS - 1][a][b] > best) {
                best   = dp[K_KNOTS - 1][a][b];
                best_a = a; best_b = b;
            }
        }
    }

    // Backtrack to recover the K-knot path.
    std::vector<int> path(K_KNOTS, 0);
    path[K_KNOTS - 1] = best_b;
    path[K_KNOTS - 2] = best_a;
    for (int i = K_KNOTS - 1; i >= 2; --i) {
        path[i - 2] = pred[i][path[i - 1]][path[i]];
    }

    // Sub-pixel refinement at each knot via parabolic interpolation on
    // the dp values across y-bins. We refine y_i using the row dp[i][·][b]
    // marginal — at the recovered (a, b, c) triple, the local quadratic
    // around bin b in dp[i][a][·] gives a sub-pixel offset.
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
        if (b > 0 && b < Y_BINS - 1) {
            // Use dp[i][·][b] (last-axis fixed) marginalised over prev a:
            // simpler to just use the chosen a from path[i-1].
            int a = (i >= 1) ? path[i - 1] : path[i];
            int next_a = a;
            if (i >= 1) {
                off = refine_bin(dp[i][next_a][b - 1],
                                 dp[i][next_a][b],
                                 dp[i][next_a][b + 1]);
            }
        }
        y_path[i] = y_of_bin(b, y0) + off
                  * (2.0 * Y_RANGE_PX / (double)(Y_BINS - 1));
    }

    // Emit dense output: linear interpolation between adjacent knots.
    std::vector<cv::Point2d> out;
    out.reserve(W);
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        // Knot interval k: xs[k] ≤ xd ≤ xs[k+1].
        int k = (int)std::floor((double)x * (K_KNOTS - 1) / (double)(W - 1));
        if (k > K_KNOTS - 2) k = K_KNOTS - 2;
        if (k < 0) k = 0;
        double t = (xd - xs[k]) / std::max(1e-9, xs[k+1] - xs[k]);
        double y = (1.0 - t) * y_path[k] + t * y_path[k+1];
        out.emplace_back(xd, y);
    }
    return out;
}

} // namespace lab
