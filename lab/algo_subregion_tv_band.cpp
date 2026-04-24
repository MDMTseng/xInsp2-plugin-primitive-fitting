//
// algo_subregion_tv_band.cpp — tensor voting on a subregion-local
// dense gradient band.
//
// This is the *correct* generalisation of tensor voting to the
// subregion-frontend architecture. Unlike the peak-only variant, we
// keep the continuous gradient chain along the curve direction. The
// only sparsification is spatial decimation (every `STRIDE` pixels in
// u-v) which preserves continuity while cutting the voting cost.
//
// For a Line region the u-v frame is axis-aligned so we simply crop a
// horizontal band around y0. For Arc / Ellipse Arc regions the same
// backend would consume a `cv::warpPolar` / custom-remap output
// (identical algorithm, different frontend).
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>

namespace lab {

namespace {

constexpr int    HALF_BAND       = 40;     // ± around y0
constexpr int    STRIDE          = 2;      // sub-sample factor in both u & v
constexpr double TOKEN_FRAC      = 0.08;   // pixels above this fraction of max are tokens
constexpr double SIGMA           = 14.0;   // voting scale (sub-sampled units)
constexpr int    DF_D_BINS       = 128;
constexpr int    DF_T_BINS       = 360;

// Decay-field lookup on (d, cos θ).
struct DfLut {
    std::vector<float> tbl;
    double d_max, d_step_inv, t_step_inv;
};
DfLut build_df_lut(double sigma) {
    DfLut L;
    double sig2 = sigma * sigma;
    double c_curv = 3.0 * sig2;
    L.d_max = 3.0 * sigma;
    L.d_step_inv = (DF_D_BINS - 1) / L.d_max;
    L.t_step_inv = (DF_T_BINS - 1) / 1.0;   // cos θ ∈ [0, 1]
    L.tbl.assign((size_t)DF_D_BINS * DF_T_BINS, 0.0f);
    for (int di = 0; di < DF_D_BINS; ++di) {
        double d = di / L.d_step_inv;
        for (int ti = 0; ti < DF_T_BINS; ++ti) {
            double cos_t = ti / L.t_step_inv;
            double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
            double s, kappa;
            if (sin_t < 1e-6) { s = d; kappa = 0.0; }
            else {
                double th = std::asin(sin_t);
                s = d * th / sin_t;
                kappa = 2.0 * sin_t / (d + 1e-9);
            }
            double df = std::exp(-(s*s + c_curv*kappa*kappa) / sig2);
            L.tbl[(size_t)di * DF_T_BINS + ti] = (float)df;
        }
    }
    return L;
}
inline float df_lookup(const DfLut& L, double d, double cos_t) {
    if (d <= 0 || d >= L.d_max) return 0.0f;
    int di = std::min(DF_D_BINS - 1, (int)(d * L.d_step_inv));
    int ti = std::min(DF_T_BINS - 1, (int)(cos_t * L.t_step_inv));
    return L.tbl[(size_t)di * DF_T_BINS + ti];
}

} // anon

std::vector<cv::Point2d> detect_subregion_tv_band(const cv::Mat& gray,
                                                   const GroundTruth& gt) {
    const int W = gray.cols, H = gray.rows;
    const double y0 = gt.y0;
    const int y_lo = std::max(0, (int)std::round(y0) - HALF_BAND);
    const int y_hi = std::min(H - 1, (int)std::round(y0) + HALF_BAND);
    const int Bh   = y_hi - y_lo + 1;

    // 1. Extract the u-v band + gradients.
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    // 2. Decimated token list. Spatial stride = STRIDE; tokens kept if
    //    magnitude ≥ TOKEN_FRAC × band_max.
    //    Receivers are exactly the same decimated grid.
    std::vector<float> tok_x, tok_y, tok_tx, tok_ty;
    tok_x.reserve(2048);
    tok_y.reserve(2048);
    tok_tx.reserve(2048);
    tok_ty.reserve(2048);

    double max_mag = 0;
    for (int y = y_lo; y <= y_hi; y += STRIDE) {
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < W; x += STRIDE) {
            double m = std::hypot(gxr[x], gyr[x]);
            if (m > max_mag) max_mag = m;
        }
    }
    const double thr = TOKEN_FRAC * max_mag;
    for (int y = y_lo; y <= y_hi; y += STRIDE) {
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        for (int x = 0; x < W; x += STRIDE) {
            double m = std::hypot(gxr[x], gyr[x]);
            if (m < thr) continue;
            tok_x.push_back((float)x);
            tok_y.push_back((float)(y - y_lo));           // store in band-local y
            tok_tx.push_back((float)(-gyr[x] / m));
            tok_ty.push_back((float)( gxr[x] / m));
        }
    }
    const int Ntok = (int)tok_x.size();
    if (Ntok < 3) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // 3. Tensor accumulators — per-thread scratch on band-local grid.
    const int grid_w = (W + STRIDE - 1) / STRIDE;
    const int grid_h = (Bh + STRIDE - 1) / STRIDE;
    const int grid_N = grid_w * grid_h;

    const DfLut lut = build_df_lut(SIGMA);
    const double R  = 2.5 * SIGMA;
    const double R2 = R * R;
    const double cos_cone = 0.70710678118654752;

    int nth = 1;
    #ifdef _OPENMP
    nth = omp_get_max_threads();
    #endif
    std::vector<std::vector<double>> Txx_t(nth, std::vector<double>(grid_N, 0.0));
    std::vector<std::vector<double>> Txy_t(nth, std::vector<double>(grid_N, 0.0));
    std::vector<std::vector<double>> Tyy_t(nth, std::vector<double>(grid_N, 0.0));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < Ntok; ++i) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& Txx = Txx_t[tid];
        auto& Txy = Txy_t[tid];
        auto& Tyy = Tyy_t[tid];

        double px = tok_x[i], py = tok_y[i];
        double tx = tok_tx[i], ty = tok_ty[i];
        int ix_lo = std::max(0,          (int)std::floor((px - R) / STRIDE));
        int ix_hi = std::min(grid_w - 1, (int)std::ceil ((px + R) / STRIDE));
        int iy_lo = std::max(0,          (int)std::floor((py - R) / STRIDE));
        int iy_hi = std::min(grid_h - 1, (int)std::ceil ((py + R) / STRIDE));

        for (int yr = iy_lo; yr <= iy_hi; ++yr) {
            double ry = (double)yr * STRIDE;
            double dy = ry - py;
            for (int xr = ix_lo; xr <= ix_hi; ++xr) {
                double rx = (double)xr * STRIDE;
                double dx = rx - px;
                double d2 = dx*dx + dy*dy;
                if (d2 > R2) continue;
                if (d2 < 1e-9) {
                    int idx = yr * grid_w + xr;
                    Txx[idx] += tx*tx; Txy[idx] += tx*ty; Tyy[idx] += ty*ty;
                    continue;
                }
                double d = std::sqrt(d2);
                double inv_d = 1.0 / d;
                double vx = dx * inv_d, vy = dy * inv_d;
                double dot = vx * tx + vy * ty;
                if (dot < 0) dot = -dot;
                if (dot < cos_cone) continue;
                float df = df_lookup(lut, d, dot);
                if (df < 1e-4f) continue;
                double r_x = 2.0 * dot * vx - tx;
                double r_y = 2.0 * dot * vy - ty;
                int idx = yr * grid_w + xr;
                Txx[idx] += df * r_x * r_x;
                Txy[idx] += df * r_x * r_y;
                Tyy[idx] += df * r_y * r_y;
            }
        }
    }

    // 4. Reduce per-thread scratch into a single band.
    std::vector<double> Txx(grid_N, 0.0), Txy(grid_N, 0.0), Tyy(grid_N, 0.0);
    for (int t = 0; t < nth; ++t) {
        #pragma omp parallel for
        for (int k = 0; k < grid_N; ++k) {
            Txx[k] += Txx_t[t][k];
            Txy[k] += Txy_t[t][k];
            Tyy[k] += Tyy_t[t][k];
        }
    }

    // 5. Stick saliency on the grid, argmax along y per x column.
    //    Saliency = λ1 - λ2 of the 2×2 tensor.
    std::vector<double> col_best_sal(grid_w, -1.0);
    std::vector<int>    col_best_y  (grid_w, -1);
    for (int yr = 0; yr < grid_h; ++yr) {
        for (int xr = 0; xr < grid_w; ++xr) {
            int idx = yr * grid_w + xr;
            double tr = Txx[idx] + Tyy[idx];
            double det = Txx[idx] * Tyy[idx] - Txy[idx] * Txy[idx];
            double disc = std::sqrt(std::max(0.0, tr*tr/4.0 - det));
            double sal = (tr/2.0 + disc) - (tr/2.0 - disc);
            if (sal > col_best_sal[xr]) { col_best_sal[xr] = sal; col_best_y[xr] = yr; }
        }
    }

    // 6. Parabolic sub-pixel refine in grid-y, then un-decimate + lift
    //    back into image coords.
    std::vector<cv::Point2d> out;
    out.reserve(W);
    auto sal_at = [&](int xr, int yr) -> double {
        int idx = yr * grid_w + xr;
        double tr = Txx[idx] + Tyy[idx];
        double det = Txx[idx] * Tyy[idx] - Txy[idx] * Txy[idx];
        double disc = std::sqrt(std::max(0.0, tr*tr/4.0 - det));
        return (tr/2.0 + disc) - (tr/2.0 - disc);
    };
    for (int x = 0; x < W; ++x) {
        int xr = std::min(grid_w - 1, x / STRIDE);
        int yr = col_best_y[xr];
        if (yr < 0) { out.emplace_back((double)x, y0); continue; }
        double y_band = (double)yr * STRIDE;
        if (yr > 0 && yr < grid_h - 1) {
            double sm = sal_at(xr, yr - 1);
            double s0 = col_best_sal[xr];
            double sp = sal_at(xr, yr + 1);
            double denom = sm - 2.0 * s0 + sp;
            if (std::abs(denom) > 1e-9) {
                double off = 0.5 * (sm - sp) / denom;
                if (off > -1.0 && off < 1.0) y_band += off * STRIDE;
            }
        }
        double y_img = (double)y_lo + y_band;
        out.emplace_back((double)x, y_img);
    }
    return out;
}

} // namespace lab
