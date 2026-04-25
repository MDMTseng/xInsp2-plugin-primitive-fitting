//
// algo_constrained_ransac.cpp — RANSAC over constrained cubic
// polynomial coefficients.
//
// Same objective as algo_constrained_grid (maximise ∫E along the
// curve under PolyConstraints box) but the search method is randomised:
// each iteration samples 4 candidate edge pixels from a strength-
// ordered pool, fits the unique cubic through them, rejects violations
// of the slope/curv/jerk caps, and scores by integrated evidence.
//
// What this benchmark answers: how close does (PROSAC-flavoured) random
// sampling get to the global optimum found by grid search, and how
// fast?
//

#include "common.hpp"
#include "constrained_poly_common.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace lab {

namespace {

constexpr int    BAND_HALF_PX  = 50;
constexpr int    N_ITERS       = 400;
constexpr double E_CAND_THR    = 0.55;  // saturated-evidence cut-off for candidate pool

// Solve for the unique deg-3 polynomial v = a0 + a1·u + a2·u² + a3·u³
// passing through 4 (u, v) points. Returns false on near-singular system.
bool fit_cubic_4pts(const std::array<double, 4>& u,
                    const std::array<double, 4>& v,
                    std::array<double, 4>& coeffs) {
    // Vandermonde-like 4×4 system:  [1 u u² u³] · θ = v.
    double M[4][4];
    for (int i = 0; i < 4; ++i) {
        M[i][0] = 1.0;
        M[i][1] = u[i];
        M[i][2] = u[i] * u[i];
        M[i][3] = M[i][2] * u[i];
    }
    double rhs[4] = { v[0], v[1], v[2], v[3] };
    // Gaussian elimination with partial pivoting.
    for (int j = 0; j < 4; ++j) {
        int piv = j;
        double maxa = std::abs(M[j][j]);
        for (int i = j + 1; i < 4; ++i) {
            if (std::abs(M[i][j]) > maxa) { maxa = std::abs(M[i][j]); piv = i; }
        }
        if (maxa < 1e-9) return false;
        if (piv != j) {
            std::swap(M[j], M[piv]);
            std::swap(rhs[j], rhs[piv]);
        }
        for (int i = j + 1; i < 4; ++i) {
            double f = M[i][j] / M[j][j];
            for (int k = j; k < 4; ++k) M[i][k] -= f * M[j][k];
            rhs[i] -= f * rhs[j];
        }
    }
    for (int j = 3; j >= 0; --j) {
        double s = rhs[j];
        for (int k = j + 1; k < 4; ++k) s -= M[j][k] * coeffs[k];
        coeffs[j] = s / M[j][j];
    }
    return true;
}

} // namespace

std::vector<cv::Point2d> detect_constrained_ransac(const cv::Mat& gray,
                                                   const GroundTruth& gt) {
    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;
    const double half_w = 0.5 * W;
    const cpoly::PolyConstraints con = cpoly::default_constraints();

    cv::Mat E = cpoly::compute_saturated_evidence(gray, y0, BAND_HALF_PX);
    const int band_lo = std::max(1,     (int)std::round(y0 - BAND_HALF_PX));
    const int band_hi = std::min(H - 2, (int)std::round(y0 + BAND_HALF_PX));

    // Build a candidate pool: pixels in band with saturated evidence above
    // a threshold. Each entry stores (x, y_image, strength).
    struct Cand { double u; double v; double w; };
    std::vector<Cand> pool;
    pool.reserve(4096);
    for (int y = band_lo; y <= band_hi; ++y) {
        const double* row = E.ptr<double>(y);
        for (int x = 0; x < W; ++x) {
            if (row[x] >= E_CAND_THR) {
                pool.push_back({((double)x - half_w) / half_w,
                                (double)y - y0,
                                row[x]});
            }
        }
    }
    if (pool.size() < 4) {
        return cpoly::sample_dense({0.0, 0.0, 0.0, 0.0}, y0, W);
    }
    // Sort descending by strength → enables PROSAC-style sampling.
    std::sort(pool.begin(), pool.end(),
        [](const Cand& a, const Cand& b){ return a.w > b.w; });
    const int N = (int)pool.size();

    std::mt19937 rng(0xC0FFEE);
    double best_score = -1e30;
    std::vector<double> best_coeffs(4, 0.0);

    for (int it = 0; it < N_ITERS; ++it) {
        // PROSAC window: grow from 4 to N over the iteration budget.
        int M = 4 + (int)((double)(N - 4) * (double)it / std::max(1, N_ITERS - 1));
        M = std::clamp(M, 4, N);
        std::uniform_int_distribution<int> pick(0, M - 1);

        std::array<int, 4> idx;
        for (int k = 0; k < 4; ++k) {
            while (true) {
                int c = pick(rng);
                bool dup = false;
                for (int j = 0; j < k; ++j) if (idx[j] == c) { dup = true; break; }
                if (!dup) { idx[k] = c; break; }
            }
        }
        // Reject minimal samples with poor u-spread (ill-conditioned).
        std::array<double, 4> us, vs;
        double umin = 1e9, umax = -1e9;
        for (int k = 0; k < 4; ++k) {
            us[k] = pool[idx[k]].u;
            vs[k] = pool[idx[k]].v;
            if (us[k] < umin) umin = us[k];
            if (us[k] > umax) umax = us[k];
        }
        if (umax - umin < 0.4) continue;

        std::array<double, 4> c;
        if (!fit_cubic_4pts(us, vs, c)) continue;
        std::vector<double> coeffs(c.begin(), c.end());

        if (!cpoly::satisfies(coeffs, con, (double)W)) continue;

        double s = cpoly::score_poly(E, coeffs, y0, band_lo, band_hi);
        if (s > best_score) {
            best_score = s;
            best_coeffs = coeffs;
        }
    }

    return cpoly::sample_dense(best_coeffs, y0, W);
}

} // namespace lab
