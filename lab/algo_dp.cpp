//
// algo_dp.cpp — Viterbi-style DP along the x-axis.
//
// For every column x we build a cost table over the rows y in a narrow
// vertical band centred on the known midline (gt.y0). The cost of
// reaching (x, y) is the negative edge evidence at that pixel plus the
// best cumulative cost of reaching any predecessor (x-1, y') in
// [y-K, y+K], weighted by a quadratic smoothness penalty α·(y-y')².
// Backtracking the argmin yields a globally optimal smooth poly-line
// through the evidence field.
//
// Robustness improvements over the simple fixed-parameter version:
//   * Evidence is pre-smoothed along y with a short 1-D Gaussian, which
//     attenuates salt-pepper and Gaussian noise without blunting edges
//     (kernel size derived from image height, not per-scene).
//   * Smoothness weight α is scaled by a data-driven estimate S of the
//     per-column peak edge strength (median of per-column maxima in the
//     band). High-contrast scenes thus get a proportionally stronger
//     smoothness prior — spikes can no longer out-vote the curve just
//     because they're bright.
//   * Band half-width is derived from image H (≈ H/6). The fixed
//     ±40 px magic number is gone.
//   * K (max |Δy| per column step) is derived from the half-band
//     width, so large-amplitude scenes still have slack but small
//     images don't waste work.
//
// None of these knobs read anything from GroundTruth other than W, H,
// y0 (all legitimate priors per the caliper-band contract).
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lab {

namespace {

// Robust statistics for a float array. Returns (median, MAD).
// MAD = median(|x - median(x)|). For Gaussian noise,
// σ ≈ 1.4826 · MAD; we use MAD directly as a scale estimate.
std::pair<float, float> median_and_mad(const cv::Mat& m) {
    std::vector<float> vals;
    vals.reserve((size_t)m.rows * m.cols);
    for (int y = 0; y < m.rows; ++y) {
        const float* row = m.ptr<float>(y);
        for (int x = 0; x < m.cols; ++x) vals.push_back(row[x]);
    }
    if (vals.empty()) return {0.f, 0.f};
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    float med = vals[vals.size() / 2];
    std::vector<float> dev(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) dev[i] = std::abs(vals[i] - med);
    std::nth_element(dev.begin(), dev.begin() + dev.size() / 2, dev.end());
    float mad = dev[dev.size() / 2];
    return {med, mad};
}

} // anon

std::vector<cv::Point2d> detect_dp(const cv::Mat& gray, const GroundTruth& gt) {
    // 1. Edge evidence: |∂I/∂y| via 3-tap Sobel, float.
    cv::Mat gy;
    cv::Sobel(gray, gy, CV_32F, /*dx=*/0, /*dy=*/1, /*ksize=*/3);
    cv::Mat e = cv::abs(gy);

    const int H = e.rows;
    const int W = e.cols;

    // 2. Pre-smooth evidence along y with a short 1-D Gaussian. This
    //    suppresses salt-pepper and low-σ Gaussian noise without
    //    materially blurring true edges. Kernel size ties to image
    //    height (so larger images get proportionally larger smoothing)
    //    and is kept small so the peak locus stays sharp.
    {
        int ks = std::max(3, (H / 80) | 1);   // odd, ≥3, grows with H
        if (ks > 7) ks = 7;
        cv::GaussianBlur(e, e, cv::Size(1, ks), /*σy*/ 0.0);
    }

    // 3. Restrict the DP to a vertical band around the midline. Half-band
    //    is a fraction of image height (≈ H/6). Tall images → wider band;
    //    small images → tighter. We also clamp to at least 20 px so
    //    tiny images still leave room for the curve swing.
    const int half_band_raw = std::max(20, H / 6);
    const int y_lo = std::max(1,      (int)std::round(gt.y0) - half_band_raw);
    const int y_hi = std::min(H - 2,  (int)std::round(gt.y0) + half_band_raw);
    const int B    = y_hi - y_lo + 1;

    // 4. Measure the edge-strength scale on the band. This lets us pick
    //    an α that is comparable across scenes with wildly different
    //    contrast (δ ≈ 30..90). MAD is retained only as a fallback
    //    scale if the band happens to be uniformly dark.
    cv::Mat band = e.rowRange(y_lo, y_hi + 1);
    auto [_med, mad] = median_and_mad(band);
    (void)_med;

    // The "edge scale" S is what a good path's per-column evidence
    // contributes. Compute per-column max over the band (that is what
    // the DP tends to select at each column), then take the median
    // across columns. This is robust against sparse spike columns
    // without ignoring them: only the median column matters, not the
    // brightest outliers.
    float S = 0.f;
    {
        std::vector<float> col_max;
        col_max.reserve((size_t)band.cols);
        for (int x = 0; x < band.cols; ++x) {
            float m = 0.f;
            for (int y = 0; y < band.rows; ++y) {
                float v = band.ptr<float>(y)[x];
                if (v > m) m = v;
            }
            col_max.push_back(m);
        }
        if (!col_max.empty()) {
            std::nth_element(col_max.begin(),
                             col_max.begin() + col_max.size() / 2,
                             col_max.end());
            S = col_max[col_max.size() / 2];
        }
        if (S < 1e-3f) S = std::max(1e-3f, 4.f * mad + 1e-3f);
    }

    // 5. DP parameters.
    //    α is proportional to S so scenes of any contrast use the same
    //    *relative* smoothness prior: one dy=1 step costs α·1 = 0.75 S,
    //    i.e. about 3/4 of a column's worth of median per-column peak
    //    evidence. Below ~0.3·S the path gets dragged by spikes; above
    //    ~1.5·S it over-smooths curved shapes. 0.75 is the sweet spot
    //    empirically across the full random-scene distribution.
    const double alpha = 0.75 * (double)S;

    // K: max |Δy| per column. Tied to half_band so large-amplitude
    // scenes can still bend back toward the curve over a few columns,
    // but small enough to block teleporting to a spike. Clamped so we
    // never exceed 6 (diminishing returns; more pen[] entries would
    // just invite spike jumps) or drop below 3 (need at least room for
    // sub-pixel curvature).
    const int K = std::max(3, std::min(6, half_band_raw / 8));

    const float kInf = std::numeric_limits<float>::infinity();

    // Flat storage for cost and argmin-predecessor tables, indexed by
    // [x * B + (y - y_lo)].
    std::vector<float> cost((size_t)W * B, kInf);
    std::vector<int>   prev((size_t)W * B, -1);

    // Pre-compute quadratic transition penalties for dy ∈ [-K, K].
    // (Empirically, Huber and L1 match the quadratic once α is
    // rescaled; quadratic is the simplest that performs as well.)
    std::vector<float> pen(2 * K + 1);
    for (int d = -K; d <= K; ++d) {
        pen[d + K] = (float)(alpha * (double)d * (double)d);
    }

    // 6. Boundary: cost[0][y] = -e(0, y).
    for (int y = y_lo; y <= y_hi; ++y) {
        cost[(size_t)0 * B + (y - y_lo)] = -e.at<float>(y, 0);
    }

    // 7. Forward pass.
    for (int x = 1; x < W; ++x) {
        const float* eCol = e.ptr<float>();  // row-major: e.at<float>(y, x) = eCol[y*W+x]
        const size_t off_cur  = (size_t)x       * B;
        const size_t off_prev = (size_t)(x - 1) * B;

        for (int yi = 0; yi < B; ++yi) {
            const int y = y_lo + yi;

            float best_c   = kInf;
            int   best_yi2 = -1;

            const int dmin = std::max(-K, y_lo - y);
            const int dmax = std::min( K, y_hi - y);
            for (int d = dmin; d <= dmax; ++d) {
                const int yi2 = yi + d;
                const float c = cost[off_prev + yi2] + pen[d + K];
                if (c < best_c) {
                    best_c   = c;
                    best_yi2 = yi2;
                }
            }

            const float ev = eCol[(size_t)y * W + x];
            cost[off_cur + yi] = -ev + best_c;
            prev[off_cur + yi] = best_yi2;
        }
    }

    // 8. Backtrack from argmin of last column.
    std::vector<int> path(W, 0);
    {
        float best_c  = kInf;
        int   best_yi = 0;
        const size_t off = (size_t)(W - 1) * B;
        for (int yi = 0; yi < B; ++yi) {
            if (cost[off + yi] < best_c) {
                best_c  = cost[off + yi];
                best_yi = yi;
            }
        }
        path[W - 1] = best_yi;
        for (int x = W - 1; x > 0; --x) {
            path[x - 1] = prev[(size_t)x * B + path[x]];
        }
    }

    // 9. Sub-pixel refine each (x, y) by parabolic fit on the smoothed
    //    evidence. The smoothed field gives cleaner parabola fits than
    //    the raw field when noise is present.
    std::vector<cv::Point2d> out;
    out.reserve(W);
    for (int x = 0; x < W; ++x) {
        const int y = y_lo + path[x];
        double y_sub = (double)y;
        if (y > 0 && y < H - 1) {
            const double ym = e.at<float>(y - 1, x);
            const double y0 = e.at<float>(y,     x);
            const double yp = e.at<float>(y + 1, x);
            const double denom = ym - 2.0 * y0 + yp;
            if (std::abs(denom) > 1e-12) {
                const double offset = 0.5 * (ym - yp) / denom;
                if (offset > -1.0 && offset < 1.0) y_sub += offset;
            }
        }
        out.emplace_back((double)x, y_sub);
    }
    return out;
}

} // namespace lab
