//
// algo_constrained_knot_dp.cpp — knot DP with HARD derivative box
// constraints, the "structured search" counterpart to constrained_grid
// and constrained_ransac.
//
// Construction. Same skeleton as algo_spline_knot_dp (K knots, M y-bins,
// piecewise-linear interpolation, Viterbi DP over (y_{i-1}, y_i)) but
// with two changes:
//
//  1. The curvature penalty `λ·(c-2b+a)²` is replaced by a HARD box
//     constraint:  |c-2b+a|·curv_scale ≤ curv_max_pxxx.  Transitions
//     that violate it are pruned outright.
//  2. The Δy step constraint MAX_DELTA already mirrored slope_max in
//     spline_knot_dp; here we derive it from the user's slope bound
//     directly (slope_max_pxx · knot_spacing_x → bins).
//
// What this answers: when constraints are hard (rather than soft via
// penalty), does DP still converge to the same optimum as grid search?
// Comparison metric is the residual sum of saturated evidence — the
// same scalar constrained_grid optimises.
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

constexpr int    K_KNOTS      = 20;
constexpr int    M_BINS       = 80;
constexpr double Y_RANGE_PX   = 40.0;
constexpr int    BAND_HALF_PX = 50;

inline double y_of_bin(int b, double y0) {
    return y0 - Y_RANGE_PX
         + 2.0 * Y_RANGE_PX * (double)b / (double)(M_BINS - 1);
}

// Pre-tabulate segment integrals seg[k][a][b] = ∫ E(line(y_a → y_b))
// along the line from xs[k] to xs[k+1].
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

} // namespace

std::vector<cv::Point2d> detect_constrained_knot_dp(const cv::Mat& gray,
                                                    const GroundTruth& gt) {
    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;
    const cpoly::PolyConstraints con = cpoly::default_constraints();

    cv::Mat E = cpoly::compute_saturated_evidence(gray, y0, BAND_HALF_PX);
    const int band_lo = std::max(1,     (int)std::round(y0 - BAND_HALF_PX));
    const int band_hi = std::min(H - 2, (int)std::round(y0 + BAND_HALF_PX));

    // Knot x-positions and per-bin scales.
    std::vector<double> xs(K_KNOTS);
    for (int i = 0; i < K_KNOTS; ++i)
        xs[i] = (double)(W - 1) * (double)i / (double)(K_KNOTS - 1);
    const double dx_knot = xs[1] - xs[0];
    const double dy_per_bin = 2.0 * Y_RANGE_PX / (double)(M_BINS - 1);

    // Hard box constraints in bin units:
    //   |Δy|/dx ≤ slope_max_pxx              ⇒ |Δb| ≤ slope_max·dx/dy_per_bin
    //   |Δ²y|/dx² ≤ curv_max_pxxx            ⇒ |Δ²b| ≤ curv_max·dx²/dy_per_bin
    const int max_delta = std::max(1,
        (int)std::ceil(con.slope_max_pxx * dx_knot / dy_per_bin));
    const int max_curv  = std::max(1,
        (int)std::ceil(con.curv_max_pxxx * dx_knot * dx_knot / dy_per_bin));

    constexpr int M  = M_BINS;
    constexpr int MM = M * M;

    // Pre-compute segment integrals (parallel; pure read of E).
    std::vector<double> seg_flat((K_KNOTS - 1) * MM, 0.0);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < K_KNOTS - 1; ++k) {
        double* sk = &seg_flat[(size_t)k * MM];
        double xa = xs[k], xb = xs[k+1];
        for (int a = 0; a < M; ++a) {
            double ya = y_of_bin(a, y0);
            double* row = sk + (size_t)a * M;
            int b_lo = std::max(0, a - max_delta);
            int b_hi = std::min(M - 1, a + max_delta);
            for (int b = b_lo; b <= b_hi; ++b) {
                row[b] = integrate_line(E, xa, ya, xb, y_of_bin(b, y0),
                                        band_lo, band_hi);
            }
        }
    }

    constexpr double NEG_INF = -1e30;
    std::vector<double>  dp_flat (K_KNOTS * MM, NEG_INF);
    std::vector<int16_t> pred_flat(K_KNOTS * MM, -1);

    // Seed at i=1 with slope-box but no curvature box yet (need 3 knots).
    {
        const double* s0 = &seg_flat[0];
        double*       d1 = &dp_flat[(size_t)1 * MM];
        for (int a = 0; a < M; ++a) {
            int b_lo = std::max(0, a - max_delta);
            int b_hi = std::min(M - 1, a + max_delta);
            for (int b = b_lo; b <= b_hi; ++b)
                d1[a * M + b] = s0[a * M + b];
        }
    }

    // Forward pass with HARD curvature box (no soft penalty).
    for (int i = 1; i < K_KNOTS - 1; ++i) {
        const double* dpi  = &dp_flat[(size_t)i * MM];
        const double* segi = &seg_flat[(size_t)i * MM];
        double*       dpi1 = &dp_flat[(size_t)(i + 1) * MM];
        int16_t*      predi = &pred_flat[(size_t)(i + 1) * MM];
        for (int a = 0; a < M; ++a) {
            const double* dpi_a = dpi + (size_t)a * M;
            int b_lo = std::max(0, a - max_delta);
            int b_hi = std::min(M - 1, a + max_delta);
            for (int b = b_lo; b <= b_hi; ++b) {
                double base = dpi_a[b];
                if (base <= NEG_INF / 2) continue;
                const double* segi_b = segi + (size_t)b * M;
                double*       dpi1_b = dpi1 + (size_t)b * M;
                int16_t*      pred_b = predi + (size_t)b * M;
                int two_b_minus_a = 2 * b - a;
                int c_lo = std::max(0, b - max_delta);
                int c_hi = std::min(M - 1, b + max_delta);
                for (int c = c_lo; c <= c_hi; ++c) {
                    int curv_bin = c - two_b_minus_a;
                    if (curv_bin > max_curv || curv_bin < -max_curv) continue;
                    double score = base + segi_b[c];
                    if (score > dpi1_b[c]) {
                        dpi1_b[c] = score;
                        pred_b[c] = (int16_t)a;
                    }
                }
            }
        }
    }

    // Find best at last knot, backtrack.
    auto dp_at = [&](int i, int a, int b) -> double {
        return dp_flat[(size_t)i * MM + (size_t)a * M + b];
    };
    auto pred_at = [&](int i, int a, int b) -> int16_t {
        return pred_flat[(size_t)i * MM + (size_t)a * M + b];
    };
    double best = NEG_INF;
    int best_a = 0, best_b = 0;
    for (int a = 0; a < M; ++a)
        for (int b = 0; b < M; ++b) {
            double v = dp_at(K_KNOTS - 1, a, b);
            if (v > best) { best = v; best_a = a; best_b = b; }
        }

    if (best <= NEG_INF / 2) {
        return cpoly::sample_dense({0.0, 0.0, 0.0, 0.0}, y0, W);
    }

    std::vector<int> path(K_KNOTS, 0);
    path[K_KNOTS - 1] = best_b;
    path[K_KNOTS - 2] = best_a;
    for (int i = K_KNOTS - 1; i >= 2; --i)
        path[i - 2] = pred_at(i, path[i - 1], path[i]);

    // Sub-pixel refinement (parabolic on dp values).
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
            off = refine_bin(dp_at(i, a, b - 1), dp_at(i, a, b), dp_at(i, a, b + 1));
        }
        y_path[i] = y_of_bin(b, y0) + off * dy_per_bin;
    }

    // Output: linear interpolation between knots.
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
