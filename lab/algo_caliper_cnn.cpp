//
// algo_caliper_cnn.cpp — caliper frontend with a learned 1-D CNN
// edge-probability filter, then sparse DP backend.
//
// Architectural shape:
//   1. Place K calipers across x.
//   2. For each, extract a 3×80 ROI.
//   3. Run the trained CNN (cv::dnn) to get an 80-element edge-prob
//      array per caliper. Batch all calipers into one forward pass.
//   4. NMS on each caliper's prob array → top-N peaks (sub-pixel via
//      parabolic refinement).
//   5. Feed the (caliper, peak) pairs into a sparse Viterbi DP
//      identical in shape to caliper_dp's, but operating on
//      learned-probability evidence instead of hand-crafted gradient.
//
// Activation. Looks for the ONNX file path in the XICAL_ONNX env var.
// If unset (or load fails), this algorithm returns an empty vector and
// the lab benchmark reports no-detect for it. The model is therefore
// optional — production builds without an ONNX file simply skip it.
//

#include "common.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <vector>

namespace lab {

namespace {

constexpr int CAL_W      = 3;     // matches dump_caliper_dataset.cpp
constexpr int CAL_H      = 80;
constexpr int N_CAL      = 16;    // calipers per image
constexpr int TOP_K_NMS  = 3;     // peaks kept per caliper
constexpr int MIN_SEP_Y  = 4;     // NMS separation in y (pixels)
constexpr double PROB_THR = 0.30; // peaks below this are ignored

constexpr int    DP_BAND_HALF = 50;
constexpr double DP_ALPHA     = 0.4;     // smoothness weight per |Δy| (px)

// Lazy-loaded model. Set on first call; null if XICAL_ONNX is unset
// or readNetFromONNX failed. Once null we stay null for the rest of
// the run — `detect_caliper_cnn` returns empty until the user
// fixes the env and restarts.
struct ModelHandle {
    cv::dnn::Net net;
    bool         loaded = false;
    bool         tried  = false;
};
static ModelHandle  g_model;
static std::mutex   g_model_mu;

ModelHandle& get_model() {
    std::lock_guard<std::mutex> lk(g_model_mu);
    if (!g_model.tried) {
        g_model.tried = true;
        const char* onnx_path = std::getenv("XICAL_ONNX");
        if (onnx_path && onnx_path[0]) {
            try {
                g_model.net = cv::dnn::readNetFromONNX(onnx_path);
                g_model.loaded = !g_model.net.empty();
            } catch (const cv::Exception&) {
                g_model.loaded = false;
            }
        }
    }
    return g_model;
}

struct Hit {
    double x;
    double y;          // sub-pixel image y
    double prob;
    int    caliper_id;
};

// Build a [B, 3, CAL_H] float32 blob from `n` caliper ROIs.
// Each ROI is read column-major into the (3, H) channels-first layout
// the model expects.
cv::Mat build_blob(const cv::Mat& gray, const std::vector<int>& cxs,
                   double y0) {
    const int H = gray.rows, W = gray.cols;
    const int B = (int)cxs.size();
    int dims[3] = {B, CAL_W, CAL_H};
    cv::Mat blob(3, dims, CV_32F);
    int y_top_image = (int)std::round(y0) - CAL_H / 2;
    for (int b = 0; b < B; ++b) {
        float* dst_b = blob.ptr<float>(b);  // points to [CAL_W * CAL_H] floats
        int cx = cxs[b];
        for (int c = 0; c < CAL_W; ++c) {
            float* dst_ch = dst_b + (size_t)c * CAL_H;
            int x = std::clamp(cx + c - CAL_W / 2, 0, W - 1);
            for (int i = 0; i < CAL_H; ++i) {
                int y = std::clamp(y_top_image + i, 0, H - 1);
                dst_ch[i] = gray.ptr<uint8_t>(y)[x] / 255.0f;
            }
        }
    }
    return blob;
}

// Sigmoid in place over `n` floats.
inline void sigmoid_inplace(float* p, int n) {
    for (int i = 0; i < n; ++i) p[i] = 1.0f / (1.0f + std::exp(-p[i]));
}

// Top-N sub-pixel peaks from an 80-element probability array.
// Returns image-coordinate y values.
std::vector<std::pair<double, double>>  // (y_image, prob)
extract_peaks(const float* probs, double y0) {
    int y_top_image = (int)std::round(y0) - CAL_H / 2;
    // Local maxima with prob > PROB_THR.
    std::vector<std::pair<int, double>> raw;
    for (int i = 1; i < CAL_H - 1; ++i) {
        double p = probs[i];
        if (p < PROB_THR) continue;
        if (p > probs[i - 1] && p >= probs[i + 1]) raw.emplace_back(i, p);
    }
    // Sort descending, NMS, keep top-K.
    std::sort(raw.begin(), raw.end(),
        [](auto& a, auto& b){ return a.second > b.second; });
    std::vector<std::pair<int, double>> kept;
    for (const auto& p : raw) {
        bool ok = true;
        for (const auto& k : kept) {
            if (std::abs(p.first - k.first) < MIN_SEP_Y) { ok = false; break; }
        }
        if (ok) kept.push_back(p);
        if ((int)kept.size() >= TOP_K_NMS) break;
    }
    // Sub-pixel refine each peak by parabolic fit on adjacent probs.
    std::vector<std::pair<double, double>> out;
    out.reserve(kept.size());
    for (const auto& p : kept) {
        int i = p.first;
        double pm = (i > 0)            ? probs[i - 1] : probs[i];
        double p0 = probs[i];
        double pp = (i < CAL_H - 1)    ? probs[i + 1] : probs[i];
        double denom = pm - 2.0 * p0 + pp;
        double off = (std::abs(denom) > 1e-9) ? (0.5 * (pm - pp) / denom) : 0.0;
        if (off < -1.0 || off > 1.0) off = 0.0;
        out.emplace_back((double)y_top_image + i + off, p0);
    }
    return out;
}

// Sparse DP smoothness-prior assignment: pick one (caliper, hit)
// chain that maximises Σ prob − DP_ALPHA · Σ |Δy|.
std::vector<int> dp_assign(const std::vector<std::vector<Hit>>& per_cal) {
    int K = (int)per_cal.size();
    std::vector<std::vector<double>> dp(K);
    std::vector<std::vector<int>>    pred(K);
    for (int i = 0; i < K; ++i) {
        dp[i].resize(per_cal[i].size(), -1e30);
        pred[i].resize(per_cal[i].size(), -1);
    }
    // Seed.
    for (size_t k = 0; k < per_cal[0].size(); ++k) dp[0][k] = per_cal[0][k].prob;
    // Forward.
    for (int i = 1; i < K; ++i) {
        for (size_t b = 0; b < per_cal[i].size(); ++b) {
            double bestv = -1e30; int besta = -1;
            for (size_t a = 0; a < per_cal[i-1].size(); ++a) {
                double dy = std::abs(per_cal[i][b].y - per_cal[i-1][a].y);
                double v  = dp[i-1][a] - DP_ALPHA * dy;
                if (v > bestv) { bestv = v; besta = (int)a; }
            }
            if (besta >= 0) {
                dp[i][b] = bestv + per_cal[i][b].prob;
                pred[i][b] = besta;
            }
        }
    }
    // Pick best at last knot, backtrack.
    int last_a = -1;
    double bestv = -1e30;
    for (size_t b = 0; b < per_cal[K-1].size(); ++b) {
        if (dp[K-1][b] > bestv) { bestv = dp[K-1][b]; last_a = (int)b; }
    }
    std::vector<int> picks(K, -1);
    for (int i = K - 1; i >= 0 && last_a >= 0; --i) {
        picks[i] = last_a;
        last_a   = (i > 0) ? pred[i][last_a] : -1;
    }
    return picks;
}

} // namespace

std::vector<cv::Point2d> detect_caliper_cnn(const cv::Mat& gray,
                                            const GroundTruth& gt) {
    auto& mh = get_model();
    if (!mh.loaded) return {};   // model unavailable → skip

    const int W = gray.cols;
    const double y0 = gt.y0;

    // 1. Caliper x-positions.
    std::vector<int> cxs(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        cxs[i] = (int)std::round((i + 0.5) * (double)W / N_CAL);
        cxs[i] = std::clamp(cxs[i], 1, W - 2);
    }

    // 2-3. Build batch blob and run CNN forward in one shot.
    cv::Mat blob = build_blob(gray, cxs, y0);
    std::lock_guard<std::mutex> lk(g_model_mu);
    mh.net.setInput(blob);
    cv::Mat out = mh.net.forward();           // [N_CAL, CAL_H] logits
    if (!out.isContinuous()) out = out.clone();
    sigmoid_inplace(out.ptr<float>(), N_CAL * CAL_H);

    // 4. Per-caliper NMS top-K peaks.
    std::vector<std::vector<Hit>> per_cal(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        const float* probs = out.ptr<float>(i);
        auto peaks = extract_peaks(probs, y0);
        for (const auto& p : peaks) {
            per_cal[i].push_back({(double)cxs[i], p.first, p.second, i});
        }
    }
    // Drop calipers with no peaks; empty hit set must not break DP.
    for (auto& v : per_cal) {
        if (v.empty()) v.push_back({0.0, y0, 0.0, -1});
    }

    // 5. Sparse DP across calipers.
    std::vector<int> picks = dp_assign(per_cal);

    // Output: linear interpolation between picked (x, y) at every integer x.
    std::vector<cv::Point2d> picks_xy;
    picks_xy.reserve(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        if (picks[i] < 0) continue;
        const Hit& h = per_cal[i][picks[i]];
        if (h.caliper_id < 0) continue;
        picks_xy.emplace_back(h.x, h.y);
    }
    if (picks_xy.size() < 2) {
        std::vector<cv::Point2d> fb;
        fb.reserve(W);
        for (int x = 0; x < W; ++x) fb.emplace_back((double)x, y0);
        return fb;
    }

    std::vector<cv::Point2d> out_dense;
    out_dense.reserve(W);
    int p = 0;
    for (int x = 0; x < W; ++x) {
        while (p + 1 < (int)picks_xy.size() && picks_xy[p + 1].x <= x) ++p;
        if (p >= (int)picks_xy.size() - 1) {
            out_dense.emplace_back((double)x, picks_xy.back().y);
        } else {
            const auto& a = picks_xy[p];
            const auto& b = picks_xy[p + 1];
            double t = (x - a.x) / std::max(1e-9, b.x - a.x);
            out_dense.emplace_back((double)x, (1.0 - t) * a.y + t * b.y);
        }
    }
    return out_dense;
}

} // namespace lab
