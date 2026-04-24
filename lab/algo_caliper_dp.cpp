//
// algo_caliper_dp.cpp — caliper frontend + scanline DP backend.
//
// Architecture experiment: instead of running DP over the full image,
// first project onto the region-local frame via sparse calipers, keep
// the top-K strongest candidates per caliper, then DP over the sparse
// column grid (N_CALIPERS columns × K candidates each) with a smooth
// v(u) prior. Finally refit a low-order polynomial through the chosen
// per-caliper hits for a dense output curve.
//
// Compared to the dense `dp_scanline` algorithm this is a ~1000× data
// reduction (30 columns × 3 candidates vs 320 × 101 cells) and
// generalises to any region mode via the caliper frontend — the same
// backend works for line / arc / ellipse once the caliper sampler
// rotates the image into the region's natural axis.
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace lab {

namespace {

constexpr int    N_CALIPERS        = 60;     // columns in the u-axis grid (denser)
constexpr int    CALIPER_SPAN      = 100;    // vertical extent per caliper
constexpr int    HALF_W            = 1;      // horizontal averaging window
constexpr int    TOP_K             = 5;      // candidates kept per caliper (more)
constexpr int    MIN_SEP_Y         = 4;      // NMS separation along the caliper
constexpr int    POLY_DEGREE_MAX   = 4;      // LSQ output curve max degree

struct Candidate {
    double v;            // signed offset from midline, px
    double strength;     // |∂I/∂y| at the peak
};

struct CaliperCol {
    double                    x;          // image-space x centre of the caliper
    std::vector<Candidate>    cands;      // top-K NMS-separated peaks
};

// Compute a data-driven threshold: 70th percentile of column-wise maxima
// inside the search band. Scales with scene contrast, no magic number.
double adaptive_min_edge(const cv::Mat& absGy, double y0) {
    const int W = absGy.cols, H = absGy.rows;
    int y_lo = std::max(0,     (int)std::round(y0) - CALIPER_SPAN / 2);
    int y_hi = std::min(H - 1, (int)std::round(y0) + CALIPER_SPAN / 2);
    std::vector<double> col_max;
    col_max.reserve(W);
    for (int x = 0; x < W; ++x) {
        double best = 0;
        for (int y = y_lo; y <= y_hi; ++y) {
            double v = absGy.at<float>(y, x);
            if (v > best) best = v;
        }
        col_max.push_back(best);
    }
    std::sort(col_max.begin(), col_max.end());
    // Very permissive: half the 10th-percentile column-max, or 5,
    // whichever is bigger. The DP is responsible for picking between
    // true-curve and spike candidates, so we cast a wide net here.
    return std::max(5.0, 0.5 * col_max[(size_t)(col_max.size() * 0.10)]);
}

// Extract up to TOP_K NMS-separated sub-pixel peaks from one caliper.
// `absGy` is the pre-computed Sobel-y magnitude; we read gradient along
// the caliper directly from it, horizontally-averaged across the strip,
// so gradient units are consistent with the threshold.
CaliperCol sample_caliper(const cv::Mat& absGy, double cx, double y0,
                          double min_strength) {
    const int H = absGy.rows, W = absGy.cols;
    int x_lo = std::max(0,     (int)std::round(cx) - HALF_W);
    int x_hi = std::min(W - 1, (int)std::round(cx) + HALF_W);
    int y_lo = std::max(1,     (int)std::round(y0) - CALIPER_SPAN / 2);
    int y_hi = std::min(H - 2, (int)std::round(y0) + CALIPER_SPAN / 2);
    int n    = y_hi - y_lo + 1;
    int cols = x_hi - x_lo + 1;
    CaliperCol out;
    out.x = cx;
    if (n < 5 || cols < 1) return out;

    // 1-D |∂I/∂y| profile, averaged across the caliper strip.
    std::vector<double> grad(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        const float* row = absGy.ptr<float>(y_lo + i);
        for (int x = x_lo; x <= x_hi; ++x) sum += row[x];
        grad[i] = sum / cols;
    }

    // Local maxima above threshold (note: grad is already magnitude, so
    // we compare directly, no abs needed).
    struct Peak { int i; double mag; };
    std::vector<Peak> peaks;
    for (int i = 1; i < n - 1; ++i) {
        double g = grad[i];
        if (g < min_strength) continue;
        if (g > grad[i - 1] && g >= grad[i + 1]) {
            peaks.push_back({i, g});
        }
    }
    if (peaks.empty()) return out;
    std::sort(peaks.begin(), peaks.end(),
              [](const Peak& a, const Peak& b){ return a.mag > b.mag; });

    // NMS + top-K.
    std::vector<Peak> kept;
    kept.reserve(TOP_K);
    for (const auto& p : peaks) {
        bool ok = true;
        for (const auto& k : kept) {
            if (std::abs(p.i - k.i) < MIN_SEP_Y) { ok = false; break; }
        }
        if (!ok) continue;
        kept.push_back(p);
        if ((int)kept.size() >= TOP_K) break;
    }

    out.cands.reserve(kept.size());
    for (const auto& p : kept) {
        int i = p.i;
        double y_sub = (double)(y_lo + i);
        double gm = grad[i - 1], g0 = p.mag, gp = grad[i + 1];
        double denom = gm - 2.0 * g0 + gp;
        if (std::abs(denom) > 1e-9) {
            double off = 0.5 * (gm - gp) / denom;
            if (off > -1.0 && off < 1.0) y_sub += off;
        }
        out.cands.push_back({ y_sub - y0, p.mag });
    }
    return out;
}

// DP over sparse columns. Each column contributes one chosen candidate;
// transitions pay a quadratic-smoothness cost on the caliper-to-caliper
// Δv divided by Δu (so it's a slope penalty independent of caliper
// spacing).
std::vector<std::pair<double, double>>  // (u, v) of chosen hit per column
run_sparse_dp(const std::vector<CaliperCol>& cols, double alpha) {
    std::vector<std::pair<double, double>> out;
    if (cols.empty()) return out;

    // Flatten: cost[i][k], prev[i][k]. K ≤ TOP_K.
    const int N = (int)cols.size();
    const double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::array<double, TOP_K>> cost(N);
    std::vector<std::array<int,    TOP_K>> prev(N);
    for (auto& row : cost) row.fill(kInf);
    for (auto& row : prev) row.fill(-1);

    // Boundary.
    for (size_t k = 0; k < cols[0].cands.size(); ++k) {
        cost[0][k] = -cols[0].cands[k].strength;
    }

    for (int i = 1; i < N; ++i) {
        double du = cols[i].x - cols[i - 1].x;
        double du2 = std::max(1.0, du * du);
        for (size_t k = 0; k < cols[i].cands.size(); ++k) {
            double v  = cols[i].cands[k].v;
            double best = kInf;
            int    best_k = -1;
            for (size_t kp = 0; kp < cols[i - 1].cands.size(); ++kp) {
                double vp = cols[i - 1].cands[kp].v;
                double dv = v - vp;
                double c  = cost[i - 1][kp] + alpha * (dv * dv) / du2;
                if (c < best) { best = c; best_k = (int)kp; }
            }
            if (best_k < 0) continue;
            cost[i][k] = best - cols[i].cands[k].strength;
            prev[i][k] = best_k;
        }
    }

    // Backtrack.
    int last = N - 1;
    int best_k = -1;
    double best_c = kInf;
    for (size_t k = 0; k < cols[last].cands.size(); ++k) {
        if (cost[last][k] < best_c) { best_c = cost[last][k]; best_k = (int)k; }
    }
    if (best_k < 0) return out;

    std::vector<int> pick(N, -1);
    pick[last] = best_k;
    for (int i = last; i > 0; --i) pick[i - 1] = prev[i][pick[i]];

    out.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (pick[i] < 0 || pick[i] >= (int)cols[i].cands.size()) continue;
        out.emplace_back(cols[i].x, cols[i].cands[pick[i]].v);
    }
    return out;
}

// Least-squares polynomial fit of v vs u through the DP-selected hits,
// returning coefficients and the LSQ residual RMS. Adaptive degree via
// MDL-style penalty.
bool fit_poly(const std::vector<std::pair<double, double>>& hits,
              int degree, std::vector<double>& coeffs_out) {
    int n = (int)hits.size(), m = degree + 1;
    if (n < m) return false;
    cv::Mat A(n, m, CV_64F), b(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        double u = hits[i].first;
        double p = 1.0;
        for (int k = 0; k < m; ++k) { A.at<double>(i, k) = p; p *= u; }
        b.at<double>(i, 0) = hits[i].second;
    }
    cv::Mat x;
    if (!cv::solve(A, b, x, cv::DECOMP_NORMAL)) return false;
    coeffs_out.assign(m, 0.0);
    for (int k = 0; k < m; ++k) coeffs_out[k] = x.at<double>(k, 0);
    return true;
}

double poly_eval(const std::vector<double>& c, double u) {
    double v = 0, p = 1;
    for (double a : c) { v += a * p; p *= u; }
    return v;
}
double poly_residual_ss(const std::vector<std::pair<double, double>>& hits,
                        const std::vector<double>& c) {
    double ss = 0;
    for (const auto& h : hits) {
        double e = h.second - poly_eval(c, h.first);
        ss += e * e;
    }
    return ss;
}

} // anon

std::vector<cv::Point2d> detect_caliper_dp(const cv::Mat& gray, const GroundTruth& gt) {
    const int W = gray.cols;
    const double y0 = gt.y0;
    const double halfW = 0.5 * W;

    // 1. Evidence map.
    cv::Mat gy;
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat absGy = cv::abs(gy);

    // 2. Caliper frontend — uniform spacing across x.
    std::vector<CaliperCol> cols;
    cols.reserve(N_CALIPERS);
    const double min_edge = adaptive_min_edge(absGy, y0);
    for (int i = 0; i < N_CALIPERS; ++i) {
        double cx = (i + 0.5) * (double)W / N_CALIPERS;
        auto col = sample_caliper(absGy, cx, y0, min_edge);
        if (!col.cands.empty()) cols.push_back(col);
    }
    if (cols.size() < 4) {
        // Fallback: emit the midline for every column.
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // 3. Sparse DP — α tied to the strength distribution so smoothness
    //    scales with scene contrast.
    // For α we want: strength gain from a spike (say Δstrength) must be
    // overwhelmed by the slope cost of jumping away. Using the difference
    // between the *top* and *bottom* of each caliper's candidates as a
    // proxy for "how much a spike could tempt the path", we set
    //     α = median(cand_max − cand_min)  per caliper
    // divided by Δu² to put it in slope²·strength units.
    double alpha = 30.0;   // default
    {
        std::vector<double> gap;
        gap.reserve(cols.size());
        for (const auto& c : cols) {
            if (c.cands.size() < 2) continue;
            double hi = c.cands.front().strength, lo = c.cands.back().strength;
            gap.push_back(hi - lo);
        }
        if (!gap.empty()) {
            std::sort(gap.begin(), gap.end());
            alpha = std::max(10.0, gap[gap.size() / 2]);
        }
    }

    auto chosen = run_sparse_dp(cols, alpha);
    if (chosen.size() < 4) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // 4. Dense output via linear interpolation between DP-selected hits.
    //    No polynomial model — the DP already produced an optimal
    //    per-caliper v, and the curve shape is arbitrary (the harness
    //    tests line / arc / cubic / full-sine). Linear interp avoids
    //    forcing a global polynomial shape that may not match the
    //    underlying curve.
    (void)halfW;
    std::vector<cv::Point2d> out;
    out.reserve(W);
    size_t j = 0;
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        // Advance j so chosen[j] is the hit at or right before x.
        while (j + 1 < chosen.size() && chosen[j + 1].first <= xd) ++j;
        double v;
        if (xd <= chosen.front().first) {
            // Clamp left of first caliper.
            v = chosen.front().second;
        } else if (xd >= chosen.back().first) {
            v = chosen.back().second;
        } else {
            // Interpolate between chosen[j] and chosen[j+1].
            double x0 = chosen[j].first,  v0 = chosen[j].second;
            double x1 = chosen[j + 1].first, v1 = chosen[j + 1].second;
            double t = (x1 > x0) ? (xd - x0) / (x1 - x0) : 0.0;
            v = v0 + t * (v1 - v0);
        }
        out.emplace_back(xd, y0 + v);
    }
    return out;
}

} // namespace lab
