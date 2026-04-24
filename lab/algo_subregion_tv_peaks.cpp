//
// algo_subregion_tv_peaks.cpp — tensor voting on top-K peak tokens.
//
// Architectural experiment: use the caliper "peak extraction" subregion
// decomposition as the voter set for tensor voting. Each subregion
// (vertical 1-D scan line) yields up to TOP_K sub-pixel edge peaks.
// Those ~300 points are the *only* voters + receivers — no dense image
// band participates.
//
// Expected outcome: significantly worse than dense tensor voting. Tensor
// voting's power comes from accumulating *continuous* chains of weak
// support; peak-only sparsification discards the continuity and leaves
// voters that barely vote to each other. This file exists to
// empirically demonstrate that failure mode (see algo_subregion_tv_band
// for the correct dense-band version).
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace lab {

namespace {

constexpr int    N_CALIPERS   = 60;
constexpr int    CALIPER_SPAN = 100;
constexpr int    HALF_W       = 1;
constexpr int    TOP_K        = 5;
constexpr int    MIN_SEP_Y    = 4;

struct Token {
    double x, y;            // image coords
    double tx, ty;          // unit tangent (perp to gradient)
    double strength;
    int    caliper_idx;     // which subregion this token came from
};

// Extract top-K peaks from one caliper, storing the gradient-direction
// tangent at each peak (perpendicular to the gradient = along the curve).
std::vector<Token> sample_caliper(const cv::Mat& absGy, const cv::Mat& gx_map,
                                  const cv::Mat& gy_map,
                                  double cx, double y0, double min_strength,
                                  int caliper_idx) {
    const int H = absGy.rows, W = absGy.cols;
    int x_lo = std::max(0,     (int)std::round(cx) - HALF_W);
    int x_hi = std::min(W - 1, (int)std::round(cx) + HALF_W);
    int y_lo = std::max(1,     (int)std::round(y0) - CALIPER_SPAN / 2);
    int y_hi = std::min(H - 2, (int)std::round(y0) + CALIPER_SPAN / 2);
    int n    = y_hi - y_lo + 1;
    int cols = x_hi - x_lo + 1;
    std::vector<Token> out;
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
        if (g > grad[i - 1] && g >= grad[i + 1])
            peaks.push_back({i, g});
    }
    if (peaks.empty()) return out;

    std::sort(peaks.begin(), peaks.end(),
              [](const Peak& a, const Peak& b){ return a.mag > b.mag; });
    std::vector<Peak> kept;
    for (const auto& p : peaks) {
        bool ok = true;
        for (const auto& k : kept)
            if (std::abs(p.i - k.i) < MIN_SEP_Y) { ok = false; break; }
        if (ok) kept.push_back(p);
        if ((int)kept.size() >= TOP_K) break;
    }

    int cix = (int)std::round(cx);
    for (const auto& p : kept) {
        int i = p.i;
        double y_sub = (double)(y_lo + i);
        double gm = grad[i - 1], g0 = p.mag, gp = grad[i + 1];
        double denom = gm - 2.0 * g0 + gp;
        if (std::abs(denom) > 1e-9) {
            double off = 0.5 * (gm - gp) / denom;
            if (off > -1.0 && off < 1.0) y_sub += off;
        }
        // Tangent = gradient rotated 90° + normalised; sign is irrelevant
        // (stick tensor is sign-invariant).
        int iy = (int)std::round(y_sub);
        double gxv = gx_map.at<float>(iy, cix);
        double gyv = gy_map.at<float>(iy, cix);
        double gm_ = std::hypot(gxv, gyv);
        double tx = (gm_ > 1e-6) ? (-gyv / gm_) : 1.0;
        double ty = (gm_ > 1e-6) ? ( gxv / gm_) : 0.0;
        out.push_back({(double)cix, y_sub, tx, ty, p.mag, caliper_idx});
    }
    return out;
}

} // anon

std::vector<cv::Point2d> detect_subregion_tv_peaks(const cv::Mat& gray,
                                                    const GroundTruth& gt) {
    const int W = gray.cols, H = gray.rows;
    const double y0 = gt.y0;

    // Gradient map for caliper sampling.
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat absGy = cv::abs(gy);

    // Adaptive threshold same as caliper_dp.
    double min_edge;
    {
        int yl = std::max(0,     (int)std::round(y0) - CALIPER_SPAN / 2);
        int yh = std::min(H - 1, (int)std::round(y0) + CALIPER_SPAN / 2);
        std::vector<double> col_max;
        col_max.reserve(W);
        for (int x = 0; x < W; ++x) {
            double best = 0;
            for (int y = yl; y <= yh; ++y) {
                double v = absGy.at<float>(y, x);
                if (v > best) best = v;
            }
            col_max.push_back(best);
        }
        std::sort(col_max.begin(), col_max.end());
        min_edge = std::max(5.0, 0.5 * col_max[(size_t)(col_max.size() * 0.10)]);
    }

    // Extract peak tokens across all calipers.
    std::vector<Token> tokens;
    tokens.reserve(N_CALIPERS * TOP_K);
    for (int i = 0; i < N_CALIPERS; ++i) {
        double cx = (i + 0.5) * (double)W / N_CALIPERS;
        auto ts = sample_caliper(absGy, gx, gy, cx, y0, min_edge, i);
        for (auto& t : ts) tokens.push_back(t);
    }
    if (tokens.size() < 3) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // Sparse stick-tensor voting among the peak tokens.
    const double sigma = 20.0;   // larger than dense version — fewer voters, bigger radius
    const double sig2  = sigma * sigma;
    const double c_curv = 3.0 * sig2;
    const double R     = 2.5 * sigma;
    const double R2    = R * R;
    const double cos_cone = 0.70710678118654752;

    std::vector<double> Txx(tokens.size(), 0.0);
    std::vector<double> Txy(tokens.size(), 0.0);
    std::vector<double> Tyy(tokens.size(), 0.0);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const Token& ti = tokens[i];
        for (size_t j = 0; j < tokens.size(); ++j) {
            if (i == j) continue;
            const Token& tj = tokens[j];
            double dx = ti.x - tj.x, dy = ti.y - tj.y;
            double d2 = dx*dx + dy*dy;
            if (d2 > R2 || d2 < 1e-12) continue;
            double d = std::sqrt(d2);
            double vx = dx / d, vy = dy / d;
            double dot = vx * tj.tx + vy * tj.ty;
            if (dot < 0) dot = -dot;
            if (dot < cos_cone) continue;
            double sin_t = std::sqrt(std::max(0.0, 1.0 - dot * dot));
            double s, kappa;
            if (sin_t < 1e-6) { s = d; kappa = 0.0; }
            else              { double th = std::asin(sin_t);
                                s = d * th / sin_t; kappa = 2.0 * sin_t / d; }
            double df = std::exp(-(s*s + c_curv*kappa*kappa) / sig2);
            if (df < 1e-4) continue;
            // Vote direction = reflect tj.t about v̂.
            double r_x = 2.0 * dot * vx - tj.tx;
            double r_y = 2.0 * dot * vy - tj.ty;
            Txx[i] += df * r_x * r_x;
            Txy[i] += df * r_x * r_y;
            Tyy[i] += df * r_y * r_y;
        }
    }

    // Per-caliper: pick the token with highest stick saliency (λ1 - λ2).
    std::vector<int> best_per_col(N_CALIPERS, -1);
    std::vector<double> best_sal(N_CALIPERS, -1.0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        double tr = Txx[i] + Tyy[i];
        double det = Txx[i] * Tyy[i] - Txy[i] * Txy[i];
        double disc = std::sqrt(std::max(0.0, tr*tr/4.0 - det));
        double l1 = tr/2.0 + disc;
        double l2 = tr/2.0 - disc;
        double sal = l1 - l2;
        int ci = tokens[i].caliper_idx;
        if (sal > best_sal[ci]) { best_sal[ci] = sal; best_per_col[ci] = (int)i; }
    }

    // Produce dense output via linear interp between picked caliper tokens.
    std::vector<std::pair<double, double>> picked;
    picked.reserve(N_CALIPERS);
    for (int i = 0; i < N_CALIPERS; ++i) {
        if (best_per_col[i] < 0) continue;
        picked.emplace_back(tokens[best_per_col[i]].x, tokens[best_per_col[i]].y);
    }
    if (picked.size() < 2) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    std::vector<cv::Point2d> out;
    out.reserve(W);
    size_t j = 0;
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        while (j + 1 < picked.size() && picked[j + 1].first <= xd) ++j;
        double y;
        if (xd <= picked.front().first)      y = picked.front().second;
        else if (xd >= picked.back().first)  y = picked.back().second;
        else {
            double x0 = picked[j].first, y_0 = picked[j].second;
            double x1 = picked[j + 1].first, y_1 = picked[j + 1].second;
            double t = (x1 > x0) ? (xd - x0) / (x1 - x0) : 0.0;
            y = y_0 + t * (y_1 - y_0);
        }
        out.emplace_back(xd, y);
    }
    return out;
}

} // namespace lab
