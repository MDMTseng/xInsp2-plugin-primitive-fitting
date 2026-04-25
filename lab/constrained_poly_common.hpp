#pragma once
//
// constrained_poly_common.hpp — shared types/utilities for the three
// constrained-polynomial fitters (`algo_constrained_ransac.cpp`,
// `algo_constrained_grid.cpp`, `algo_constrained_bernstein_dp.cpp`).
//
// Problem statement these three solve:
//
//   maximise  S(θ) = ∫_0^W E(x, p_θ(x)) dx        with  p_θ(x) = Σ a_k u^k
//
//   subject to                                    where  u = (x - W/2)/(W/2)
//     max_x |p'(x)|   ≤  slope_max_pxx
//     max_x |p''(x)|  ≤  curv_max_pxxx
//     max_x |p'''(x)| ≤  jerk_max_pxxxx
//
// The three algorithms differ only in *how* they search the constrained
// θ-space: RANSAC random-sampling, brute-force grid, or DP over a
// Bernstein/control-point parameterisation.
//
// All three share:
//   * Saturating evidence map  E_sat = E_raw / (E_raw + κ)  (anchors the
//     score so a long stretch of moderate evidence beats a few intense
//     spikes — same insight that powers spline_knot_dp).
//   * Polynomial helpers in the normalised u ∈ [-1, 1] frame.
//   * A constraint check that samples 21 u-points and evaluates the
//     three derivatives in pixel-space.
//

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lab {
namespace cpoly {

// User-tuneable bounds. Defaults are set so every scene curve in the lab
// generator (max amp 28 px, max slope 0.55 px/px, max 2nd deriv ~0.011
// px/px², max 3rd deriv ~0.00021 px/px³) sits comfortably under the cap.
struct PolyConstraints {
    int    degree         = 3;
    // Bounds in image-pixel x-space. Set to ∞ to disable.
    double slope_max_pxx  = 1.5;
    double curv_max_pxxx  = 0.05;
    double jerk_max_pxxxx = 0.005;
};

// Default constraints — used by all three algos for fair comparison.
inline PolyConstraints default_constraints() { return PolyConstraints{}; }

// Polynomial evaluation in u ∈ [-1, 1].
inline double poly_eval(const std::vector<double>& c, double u) {
    double v = 0.0;
    for (int i = (int)c.size() - 1; i >= 0; --i) v = v * u + c[i];
    return v;
}

// d/du p(u).  Returns the derivative *coefficients* one shorter.
inline std::vector<double> poly_deriv(const std::vector<double>& c) {
    if (c.size() <= 1) return {};
    std::vector<double> d(c.size() - 1);
    for (size_t i = 1; i < c.size(); ++i) d[i - 1] = (double)i * c[i];
    return d;
}

// Sample 21 u points in [-1, 1] and check |p'(u)|, |p''(u)|, |p'''(u)|
// (after Jacobian conversion to x-space) against the three caps.
// Returns true if every sample obeys every active bound.
//
// du/dx = 2/W ⇒ d^k y/dx^k = (2/W)^k · d^k y/du^k.
inline bool satisfies(const std::vector<double>& coeffs,
                      const PolyConstraints& con,
                      double image_width) {
    const double inv_half_w = 2.0 / std::max(1.0, image_width);
    auto d1 = poly_deriv(coeffs);
    auto d2 = poly_deriv(d1);
    auto d3 = poly_deriv(d2);
    constexpr int S = 21;
    for (int s = 0; s < S; ++s) {
        double u = -1.0 + 2.0 * (double)s / (S - 1);
        if (con.slope_max_pxx > 0) {
            double v1 = poly_eval(d1, u) * inv_half_w;
            if (std::abs(v1) > con.slope_max_pxx) return false;
        }
        if (con.curv_max_pxxx > 0) {
            double v2 = poly_eval(d2, u) * inv_half_w * inv_half_w;
            if (std::abs(v2) > con.curv_max_pxxx) return false;
        }
        if (con.jerk_max_pxxxx > 0) {
            double v3 = poly_eval(d3, u) * inv_half_w * inv_half_w * inv_half_w;
            if (std::abs(v3) > con.jerk_max_pxxxx) return false;
        }
    }
    return true;
}

// Polarity-aware, saturating evidence map. Pipeline:
//   1. signed ∂I/∂y (3-tap central);
//   2. ReLU with band's dominant polarity (lab convention: dark→bright,
//      curve has + gradient);
//   3. saturating normalisation E ← E/(E+κ).  κ = 0.20·percentile_90(E).
//      The 90th-percentile knee is robust to spike outliers — under
//      harsh noise a few spike pixels otherwise inflate max(E) by
//      5-10×, pushing κ so high that the saturating bend smothers the
//      curve's typical contribution. p90 tracks 'typical strong
//      evidence' rather than 'any extreme value', so the saturation
//      knee stays at the right scale even when extreme values exist.
inline cv::Mat compute_saturated_evidence(const cv::Mat& gray, double y0,
                                          int band_half) {
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
    std::vector<double> samples;
    samples.reserve((size_t)(y_hi - y_lo + 1) * W);
    for (int y = y_lo; y <= y_hi; ++y) {
        const uint8_t* rm = gray.ptr<uint8_t>(y - 1);
        const uint8_t* rp = gray.ptr<uint8_t>(y + 1);
        double* dst = raw.ptr<double>(y);
        for (int x = 0; x < W; ++x) {
            double g = pol * 0.5 * ((double)rp[x] - (double)rm[x]);
            double v = (g > 0.0) ? g : 0.0;
            dst[x] = v;
            if (v > 0.0) samples.push_back(v);
        }
    }
    double p90 = 0.0;
    if (!samples.empty()) {
        size_t k = (size_t)((samples.size() - 1) * 0.90);
        std::nth_element(samples.begin(), samples.begin() + k, samples.end());
        p90 = samples[k];
    }
    const double kappa = std::max(1.0, 0.20 * p90);

    cv::Mat E(H, W, CV_64F, cv::Scalar(0.0));
    for (int y = y_lo; y <= y_hi; ++y) {
        const double* src = raw.ptr<double>(y);
        double* dst = E.ptr<double>(y);
        for (int x = 0; x < W; ++x) dst[x] = src[x] / (src[x] + kappa);
    }
    return E;
}

// Bilinear sample of E at (x, y), x integer, y sub-pixel. Returns 0
// outside the [band_lo, band_hi] window.
inline double sample_y(const cv::Mat& E, int x, double y,
                       int band_lo, int band_hi) {
    int yi = (int)std::floor(y);
    if (yi < band_lo || yi >= band_hi) return 0.0;
    double dy = y - (double)yi;
    const double* row0 = E.ptr<double>(yi);
    const double* row1 = E.ptr<double>(yi + 1);
    return row0[x] * (1.0 - dy) + row1[x] * dy;
}

// Score a polynomial (coeffs in u-frame, v = y - y0) by integrating the
// evidence map along the curve at every integer x ∈ [0, W).
inline double score_poly(const cv::Mat& E,
                         const std::vector<double>& coeffs,
                         double y0, int band_lo, int band_hi) {
    const int W = E.cols;
    const double half_w = 0.5 * W;
    double sum = 0.0;
    for (int x = 0; x < W; ++x) {
        double u = ((double)x - half_w) / half_w;
        double v = poly_eval(coeffs, u);
        sum += sample_y(E, x, v + y0, band_lo, band_hi);
    }
    return sum;
}

// Sample the polynomial densely at every integer x → image (x, y).
inline std::vector<cv::Point2d> sample_dense(const std::vector<double>& coeffs,
                                             double y0, int W) {
    std::vector<cv::Point2d> out;
    out.reserve(W);
    const double half_w = 0.5 * W;
    for (int x = 0; x < W; ++x) {
        double u = ((double)x - half_w) / half_w;
        out.emplace_back((double)x, poly_eval(coeffs, u) + y0);
    }
    return out;
}

} // namespace cpoly
} // namespace lab
