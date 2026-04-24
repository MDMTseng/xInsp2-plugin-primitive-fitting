//
// algo_caliper_dp_poly.cpp — caliper frontend + sparse DP +
// polynomial fit (adaptive degree, Huber-IRLS outlier rejection).
//
// Third point on the speed/accuracy Pareto curve between
//   `caliper_dp`      (DP + linear interpolation) — 0.35 ms, RMS p50 0.24
//   `caliper_ransac`  (RANSAC polynomial on raw hits) — 3.9 ms, RMS p50 0.16
//
// Rationale: RANSAC on the caliper frontend's raw top-K peaks spends
// most of its iterations trying random minimal samples that include
// spike peaks. DP's smoothness prior already picks one consistent hit
// per caliper — those 60 points are mostly inliers. A polynomial fit
// over them is therefore far cheaper than RANSAC over all ~150 raw
// peaks, and the handful of wrong-caliper DP picks can be handled by
// a small RANSAC and Huber-IRLS polish.
//
// Pipeline (self-contained, reuses design ideas from caliper_dp and
// caliper_ransac but no code dependency):
//
//  1. Sobel |∂I/∂y|, adaptive per-caliper min-strength threshold.
//  2. N_CALIPERS vertical calipers; top-K sub-pixel peaks per caliper.
//  3. Sparse DP over (caliper, candidate) with quadratic slope prior
//     (α tied to each caliper's strength range, so smoothness scales
//     with scene contrast). Output: one (x, v) hit per caliper.
//  4. Adaptive-degree RANSAC polynomial fit over the DP-picked hits.
//     Small iteration count (30 per degree) because the candidate
//     pool is mostly clean.
//  5. Huber-IRLS refinement.
//  6. Sample densely; linear extrapolation (C¹) outside inlier span.
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace lab {

namespace {

constexpr int    N_CALIPERS       = 60;
constexpr int    CALIPER_SPAN     = 100;
constexpr int    HALF_W           = 1;
constexpr int    TOP_K            = 5;
constexpr int    MIN_SEP_Y        = 4;

// Polynomial-fit tunables (smaller budget than caliper_ransac's because
// the DP-picked input is ~60 points, mostly inliers).
constexpr int    DEG_MIN          = 1;
constexpr int    DEG_MAX          = 5;
constexpr int    RANSAC_ITERS     = 30;     // per degree
constexpr double INLIER_THR       = 2.0;    // px
constexpr double DEGREE_PENALTY   = 4.0;    // MDL bias

struct Candidate { double v; double strength; };
struct CaliperCol { double x; std::vector<Candidate> cands; };

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
    return std::max(5.0, 0.5 * col_max[(size_t)(col_max.size() * 0.10)]);
}

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

    std::vector<double> grad(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        const float* row = absGy.ptr<float>(y_lo + i);
        for (int x = x_lo; x <= x_hi; ++x) sum += row[x];
        grad[i] = sum / cols;
    }

    struct Peak { int i; double mag; };
    std::vector<Peak> peaks;
    for (int i = 1; i < n - 1; ++i) {
        double g = grad[i];
        if (g < min_strength) continue;
        if (g > grad[i - 1] && g >= grad[i + 1]) peaks.push_back({i, g});
    }
    if (peaks.empty()) return out;
    std::sort(peaks.begin(), peaks.end(),
              [](const Peak& a, const Peak& b){ return a.mag > b.mag; });

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

std::vector<std::pair<double, double>>
run_sparse_dp(const std::vector<CaliperCol>& cols, double alpha) {
    std::vector<std::pair<double, double>> out;
    if (cols.empty()) return out;

    const int N = (int)cols.size();
    const double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::array<double, TOP_K>> cost(N);
    std::vector<std::array<int,    TOP_K>> prev(N);
    for (auto& row : cost) row.fill(kInf);
    for (auto& row : prev) row.fill(-1);

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

inline double poly_eval(const std::vector<double>& c, double u) {
    double v = 0.0;
    for (int i = (int)c.size() - 1; i >= 0; --i) v = v * u + c[i];
    return v;
}

bool fit_poly(const std::vector<double>& us,
              const std::vector<double>& vs,
              int degree,
              std::vector<double>& coeffs_out,
              const std::vector<double>& w = {}) {
    const int n = (int)us.size();
    const int m = degree + 1;
    if (n < m) return false;
    cv::Mat A(n, m, CV_64F);
    cv::Mat b(n, 1, CV_64F);
    const bool weighted = ((int)w.size() == n);
    for (int i = 0; i < n; ++i) {
        double u = us[i];
        double ww = weighted ? std::sqrt(std::max(1e-9, w[i])) : 1.0;
        double p = 1.0;
        double* row = A.ptr<double>(i);
        for (int k = 0; k < m; ++k) { row[k] = p * ww; p *= u; }
        b.at<double>(i, 0) = vs[i] * ww;
    }
    cv::Mat x;
    int flag = (n == m) ? cv::DECOMP_SVD : cv::DECOMP_NORMAL;
    if (!cv::solve(A, b, x, flag)) return false;
    coeffs_out.resize(m);
    for (int k = 0; k < m; ++k) coeffs_out[k] = x.at<double>(k, 0);
    return true;
}

} // anon

std::vector<cv::Point2d> detect_caliper_dp_poly(const cv::Mat& gray,
                                                 const GroundTruth& gt) {
    const int W = gray.cols;
    const double y0 = gt.y0;
    const double halfW = 0.5 * W;

    // 1-3. Frontend + sparse DP — same as caliper_dp.
    cv::Mat gy;
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat absGy = cv::abs(gy);

    std::vector<CaliperCol> cols;
    cols.reserve(N_CALIPERS);
    const double min_edge = adaptive_min_edge(absGy, y0);
    for (int i = 0; i < N_CALIPERS; ++i) {
        double cx = (i + 0.5) * (double)W / N_CALIPERS;
        auto col = sample_caliper(absGy, cx, y0, min_edge);
        if (!col.cands.empty()) cols.push_back(col);
    }
    if (cols.size() < 4) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    double alpha = 30.0;
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
    if ((int)chosen.size() < DEG_MIN + 2) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // 4. Transform to normalised (u, v) for the polynomial fit.
    const int N = (int)chosen.size();
    std::vector<double> us(N), vs(N);
    for (int i = 0; i < N; ++i) {
        us[i] = (chosen[i].first - halfW) / halfW;   // u ∈ [-1, 1]
        vs[i] = chosen[i].second;                    // already v = y - y0
    }

    // 5. Adaptive-degree RANSAC over the DP-picked hits. Tiny budget:
    //    the candidate pool is mostly inliers after DP, so each minimal
    //    sample is very likely to be clean.
    std::mt19937 rng(12345u);
    double              best_score  = -1e30;
    std::vector<double> best_coeffs;
    std::vector<int>    best_inlier_idx;
    int                 best_degree = DEG_MIN;
    const double u_slope_cap = halfW;

    for (int degree = DEG_MIN; degree <= DEG_MAX; ++degree) {
        const int need = degree + 1;
        if (N < need) continue;
        std::uniform_int_distribution<int> pick(0, N - 1);

        double              deg_best_score = -1e30;
        std::vector<double> deg_best_coeffs;
        std::vector<int>    deg_best_idx;

        for (int iter = 0; iter < RANSAC_ITERS; ++iter) {
            int idx[8];
            for (int k = 0; k < need; ++k) {
                while (true) {
                    int c = pick(rng);
                    bool dup = false;
                    for (int j = 0; j < k; ++j) if (idx[j] == c) { dup = true; break; }
                    if (!dup) { idx[k] = c; break; }
                }
            }
            std::vector<double> su(need), sv(need);
            for (int k = 0; k < need; ++k) { su[k] = us[idx[k]]; sv[k] = vs[idx[k]]; }
            double smin = 1e30, smax = -1e30;
            for (int k = 0; k < need; ++k) {
                smin = std::min(smin, su[k]); smax = std::max(smax, su[k]);
            }
            if ((smax - smin) < 0.4) continue;

            std::vector<double> coeffs;
            if (!fit_poly(su, sv, degree, coeffs)) continue;

            bool bad = false;
            for (int s = 0; s <= 10 && !bad; ++s) {
                double u = -1.0 + 0.2 * s;
                double dv = 0.0;
                for (int k = (int)coeffs.size() - 1; k >= 1; --k) {
                    dv = dv * u + k * coeffs[k];
                }
                if (std::abs(dv) > u_slope_cap) bad = true;
            }
            if (bad) continue;

            int cnt = 0;
            std::vector<int> in_idx;
            in_idx.reserve(N);
            for (int j = 0; j < N; ++j) {
                double r = vs[j] - poly_eval(coeffs, us[j]);
                if (std::abs(r) <= INLIER_THR) { ++cnt; in_idx.push_back(j); }
            }
            if (cnt < need) continue;

            // No caliper-coverage weighting here — the DP-picked input
            // has exactly one hit per caliper, so raw inlier count is
            // already equivalent to caliper coverage.
            double score = (double)cnt;
            if (score > deg_best_score) {
                deg_best_score  = score;
                deg_best_coeffs = coeffs;
                deg_best_idx    = std::move(in_idx);
            }
        }

        if (deg_best_score <= 0.0) continue;
        double score = deg_best_score - DEGREE_PENALTY * (degree - DEG_MIN);
        if (score > best_score) {
            best_score      = score;
            best_coeffs     = std::move(deg_best_coeffs);
            best_inlier_idx = std::move(deg_best_idx);
            best_degree     = degree;
        }
    }

    if (best_coeffs.empty() || best_inlier_idx.empty()) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // 6. Huber-IRLS refinement over the inlier set.
    {
        const int need = best_degree + 1;
        for (int outer = 0; outer < 2; ++outer) {
            std::vector<int> new_idx;
            new_idx.reserve(N);
            for (int j = 0; j < N; ++j) {
                double r = vs[j] - poly_eval(best_coeffs, us[j]);
                if (std::abs(r) <= INLIER_THR) new_idx.push_back(j);
            }
            if ((int)new_idx.size() < need + 1) break;
            best_inlier_idx = std::move(new_idx);

            std::vector<double> su, sv;
            su.reserve(best_inlier_idx.size());
            sv.reserve(best_inlier_idx.size());
            for (int i : best_inlier_idx) { su.push_back(us[i]); sv.push_back(vs[i]); }

            std::vector<double> refined;
            if (!fit_poly(su, sv, best_degree, refined)) break;
            best_coeffs = refined;

            const double c_huber = 0.7;
            std::vector<double> wv(su.size(), 1.0);
            for (size_t i = 0; i < su.size(); ++i) {
                double r = sv[i] - poly_eval(best_coeffs, su[i]);
                double ar = std::abs(r);
                wv[i] = (ar <= c_huber) ? 1.0 : c_huber / std::max(ar, 1e-9);
            }
            std::vector<double> refined2;
            if (fit_poly(su, sv, best_degree, refined2, wv)) {
                best_coeffs = refined2;
            }
        }
    }

    // 7. Sample + linear extrapolation outside inlier span.
    double x_min_in =  1e30, x_max_in = -1e30;
    for (int i : best_inlier_idx) {
        double x = chosen[i].first;
        if (x < x_min_in) x_min_in = x;
        if (x > x_max_in) x_max_in = x;
    }
    if (!(x_min_in < x_max_in)) { x_min_in = 0; x_max_in = W - 1; }

    auto eval_at_x = [&](double x) {
        double u = (x - halfW) / halfW;
        return poly_eval(best_coeffs, u) + y0;
    };
    auto dy_dx_at_x = [&](double x) {
        double u = (x - halfW) / halfW;
        double dv_du = 0.0;
        for (int k = (int)best_coeffs.size() - 1; k >= 1; --k) {
            dv_du = dv_du * u + k * best_coeffs[k];
        }
        return dv_du / halfW;
    };
    const double y_lo_edge = eval_at_x(x_min_in);
    const double y_hi_edge = eval_at_x(x_max_in);
    const double m_lo      = dy_dx_at_x(x_min_in);
    const double m_hi      = dy_dx_at_x(x_max_in);

    std::vector<cv::Point2d> out;
    out.reserve(W);
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        double y;
        if (xd < x_min_in)      y = y_lo_edge + m_lo * (xd - x_min_in);
        else if (xd > x_max_in) y = y_hi_edge + m_hi * (xd - x_max_in);
        else                    y = eval_at_x(xd);
        out.emplace_back(xd, y);
    }
    return out;
}

} // namespace lab
