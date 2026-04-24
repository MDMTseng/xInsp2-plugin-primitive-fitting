//
// algo_subregion_dp_strips.cpp — DP backend on 1-px-wide strips.
//
// Experiment: test whether the "subregion frontend + DP backend"
// architecture (validated by caliper_dp) benefits from dense-y
// candidate retention. Unlike caliper_dp's top-K per caliper, this
// variant initially kept the full |∂I/∂y| profile per strip.
//
// Finding: dense y hurts DP. Within the 2-3 px edge-width plateau,
// adjacent y bins are near-equal for real edges, so DP drifts
// between them across strips, and the quadratic penalty cannot
// distinguish drift from curve curvature. Noise spikes also get
// many neighbouring candidates instead of one peak. With α tuned
// up to reject spikes, the curve is over-smoothed; with α tuned
// down to follow the curve, spike chains win.
//
// Final config: 1-D NMS (radius 3) + top-5 peaks per strip —
// essentially caliper_dp's candidate structure over 1-px strips
// rather than 3-col caliper averaging. Outlier rate 27% still
// beats nothing; caliper_dp's 3-col averaging pre-denoises better.
//
// Conclusion: x-sparsification (caliper_dp) is the right hybrid
// with DP. Additional y-sparsification via narrow strips buys no
// accuracy and loses horizontal pre-denoising.
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

constexpr int N_STRIPS  = 64;
constexpr int HALF_BAND = 40;
constexpr int K_MAX     = 4;     // max |Δy| per strip transition

} // anon

std::vector<cv::Point2d> detect_subregion_dp_strips(const cv::Mat& gray,
                                                     const GroundTruth& gt) {
    const int W = gray.cols, H = gray.rows;
    const double y0 = gt.y0;
    const int y_lo = std::max(0,     (int)std::round(y0) - HALF_BAND);
    const int y_hi = std::min(H - 1, (int)std::round(y0) + HALF_BAND);
    const int Bh   = y_hi - y_lo + 1;

    // ---- 1. Strip x positions ------------------------------------------
    std::vector<int> strip_x(N_STRIPS);
    for (int s = 0; s < N_STRIPS; ++s) {
        strip_x[s] = std::min(W - 1,
            (int)std::round((s + 0.5) * (double)W / N_STRIPS));
    }

    // ---- 2. Per-strip evidence ------------------------------------------
    // Sobel |∂I/∂y| at strip_x[s], averaged across a 3-col local neighbourhood
    // to suppress salt-pepper. Output: evidence[s][yi] where yi ∈ [0, Bh).
    cv::Mat gy;
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::Mat absGy = cv::abs(gy);

    std::vector<std::vector<float>> evidence(N_STRIPS,
        std::vector<float>(Bh, 0.0f));
    for (int s = 0; s < N_STRIPS; ++s) {
        int sx = strip_x[s];
        int x_lo = std::max(0, sx - 1), x_hi = std::min(W - 1, sx + 1);
        int cols = x_hi - x_lo + 1;
        for (int yi = 0; yi < Bh; ++yi) {
            int y = y_lo + yi;
            double sum = 0;
            for (int x = x_lo; x <= x_hi; ++x) sum += absGy.at<float>(y, x);
            evidence[s][yi] = (float)(sum / cols);
        }
    }

    // ---- 3a. Sparsify DP evidence per strip via 1-D NMS + top-K.
    //        Dense y lets DP drift within the 2-3 px edge-width plateau
    //        (adjacent y bins are near-equal for a real edge), producing
    //        inter-strip jitter that the quadratic penalty then amplifies.
    //        NMS collapses each edge plateau to one peak.
    //        raw_evidence (kept dense) drives sub-pixel refinement later.
    constexpr int NMS_RAD = 3;
    constexpr int TOP_K   = 5;
    std::vector<std::vector<float>> raw_evidence = evidence;
    for (int s = 0; s < N_STRIPS; ++s) {
        auto& ev = evidence[s];
        const auto& orig = raw_evidence[s];
        std::fill(ev.begin(), ev.end(), 0.0f);
        std::vector<std::pair<float, int>> peaks;
        for (int yi = 0; yi < Bh; ++yi) {
            bool is_max = true;
            for (int d = -NMS_RAD; d <= NMS_RAD && is_max; ++d) {
                int y2 = yi + d;
                if (y2 < 0 || y2 >= Bh || d == 0) continue;
                if (orig[y2] > orig[yi]) is_max = false;
            }
            if (is_max && orig[yi] > 0) peaks.emplace_back(orig[yi], yi);
        }
        std::sort(peaks.begin(), peaks.end(),
            [](const auto& a, const auto& b){ return a.first > b.first; });
        int n = std::min((int)peaks.size(), TOP_K);
        for (int i = 0; i < n; ++i) ev[peaks[i].second] = peaks[i].first;
    }

    // ---- 3b. Data-driven α.
    std::vector<float> strip_max(N_STRIPS);
    for (int s = 0; s < N_STRIPS; ++s) {
        float m = 0;
        for (int yi = 0; yi < Bh; ++yi) m = std::max(m, evidence[s][yi]);
        strip_max[s] = m;
    }
    std::vector<float> sm_copy = strip_max;
    std::sort(sm_copy.begin(), sm_copy.end());
    double alpha = std::max(1.0, 0.75 * (double)sm_copy[sm_copy.size() / 2]);

    // Pre-computed quadratic penalties.
    std::array<float, 2 * K_MAX + 1> pen{};
    for (int d = -K_MAX; d <= K_MAX; ++d)
        pen[d + K_MAX] = (float)(alpha * (double)d * (double)d);

    // ---- 4. DP scan left → right ---------------------------------------
    // cost[s][yi] = best cumulative cost ending at (s, y_lo+yi).
    // Transitions: (s-1, yi+d) → (s, yi), d ∈ [-K_MAX, K_MAX].
    const float kInf = std::numeric_limits<float>::infinity();
    std::vector<std::vector<float>> cost(N_STRIPS,
        std::vector<float>(Bh, kInf));
    std::vector<std::vector<int>> prev(N_STRIPS,
        std::vector<int>(Bh, -1));

    for (int yi = 0; yi < Bh; ++yi) cost[0][yi] = -evidence[0][yi];

    for (int s = 1; s < N_STRIPS; ++s) {
        const auto& ev_cur  = evidence[s];
        const auto& cost_pr = cost[s - 1];
        auto&       cost_cu = cost[s];
        auto&       prev_cu = prev[s];
        for (int yi = 0; yi < Bh; ++yi) {
            int dmin = std::max(-K_MAX, -yi);
            int dmax = std::min( K_MAX, Bh - 1 - yi);
            float best = kInf;
            int   best_d = 0;
            for (int d = dmin; d <= dmax; ++d) {
                int yi2 = yi + d;
                float c = cost_pr[yi2] + pen[d + K_MAX];
                if (c < best) { best = c; best_d = d; }
            }
            cost_cu[yi] = -ev_cur[yi] + best;
            prev_cu[yi] = yi + best_d;
        }
    }

    // ---- 5. Backtrack ---------------------------------------------------
    std::vector<int> path(N_STRIPS, 0);
    {
        float best = kInf;
        int   best_yi = 0;
        for (int yi = 0; yi < Bh; ++yi) {
            if (cost[N_STRIPS - 1][yi] < best) {
                best = cost[N_STRIPS - 1][yi];
                best_yi = yi;
            }
        }
        path[N_STRIPS - 1] = best_yi;
        for (int s = N_STRIPS - 1; s > 0; --s) {
            path[s - 1] = prev[s][path[s]];
        }
    }

    // ---- 6. Sub-pixel refine (parabolic on raw evidence) + interp ------
    std::vector<std::pair<double, double>> picked;   // (x, y) per strip
    picked.reserve(N_STRIPS);
    for (int s = 0; s < N_STRIPS; ++s) {
        int yi = path[s];
        double y_sub = (double)(y_lo + yi);
        if (yi > 0 && yi < Bh - 1) {
            double em = raw_evidence[s][yi - 1];
            double e0 = raw_evidence[s][yi];
            double ep = raw_evidence[s][yi + 1];
            double denom = em - 2.0 * e0 + ep;
            if (std::abs(denom) > 1e-9) {
                double off = 0.5 * (em - ep) / denom;
                if (off > -1.0 && off < 1.0) y_sub += off;
            }
        }
        picked.emplace_back((double)strip_x[s], y_sub);
    }

    // ---- 7. Dense output via linear interpolation ----------------------
    std::vector<cv::Point2d> out;
    out.reserve(W);
    size_t j = 0;
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        while (j + 1 < picked.size() && picked[j + 1].first <= xd) ++j;
        double y;
        if (xd <= picked.front().first)     y = picked.front().second;
        else if (xd >= picked.back().first) y = picked.back().second;
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
