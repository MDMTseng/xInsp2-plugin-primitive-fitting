//
// algo_caliper_cnn_ort.cpp — caliper CNN inference via ONNX Runtime.
//
// Same model as algo_caliper_cnn / algo_caliper_cnn_cross
// (CrossCaliperEdgeNet from lab/cnn/model.py), same blob layout
// [1, N_CAL, CAL_W, CAL_H], same NMS top-K + sparse Viterbi DP +
// linear interpolation post-processing.  The only difference is the
// inference engine: instead of cv::dnn, this algo uses ONNX Runtime
// (ORT) directly via the C++ API. ORT's CPU EP typically delivers
// 2-3× faster forward than cv::dnn on the same model.
//
// Selected at run time via the XICAL_ONNX_CROSS_ORT env var (model
// path).  Build flag `XINSP_ENABLE_ORT` controls whether this TU
// participates at all; without it, detect_caliper_cnn_cross_ort()
// stubs out to "not loaded".  See lab/CMakeLists.txt for build wiring.
//

#include "common.hpp"

#ifdef XINSP_ENABLE_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace lab {

namespace {

constexpr int CAL_W      = 15;
constexpr int CAL_H      = 80;
constexpr int N_CAL      = 16;
constexpr int TOP_K_NMS  = 3;
constexpr int MIN_SEP_Y  = 4;
constexpr double PROB_THR = 0.30;
constexpr double DP_ALPHA = 0.4;

struct Hit {
    double x, y, prob;
    int    caliper_id;
};

// ── ORT model handle ────────────────────────────────────────────────
#ifdef XINSP_ENABLE_ORT
// Lazy-allocated. Static-storage `Ort::Env` triggers a Windows DLL
// loader race during program startup (segfault before main); deferring
// construction to first call avoids it.
struct OrtHandle {
    std::unique_ptr<Ort::Env>           env;
    std::unique_ptr<Ort::SessionOptions> opts;
    std::unique_ptr<Ort::Session>       session;
    std::string input_name, output_name;
    bool         loaded = false;
    bool         tried  = false;
};
static OrtHandle  g_ort;
static std::mutex g_ort_mu;

OrtHandle& get_ort() {
    std::lock_guard<std::mutex> lk(g_ort_mu);
    if (g_ort.tried) return g_ort;
    g_ort.tried = true;
    const char* path = std::getenv("XICAL_ONNX_CROSS_ORT");
    if (!path || !path[0]) return g_ort;
    try {
        g_ort.env  = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "xinsp");
        g_ort.opts = std::make_unique<Ort::SessionOptions>();
        g_ort.opts->SetIntraOpNumThreads(1);
        g_ort.opts->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::string s(path);
        std::wstring w(s.begin(), s.end());
        g_ort.session = std::make_unique<Ort::Session>(
            *g_ort.env, w.c_str(), *g_ort.opts);
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name  = g_ort.session->GetInputNameAllocated(0, alloc);
        auto out_name = g_ort.session->GetOutputNameAllocated(0, alloc);
        g_ort.input_name  = in_name.get();
        g_ort.output_name = out_name.get();
        g_ort.loaded = true;
    } catch (const Ort::Exception&) {
        g_ort.loaded = false;
    } catch (const std::exception&) {
        g_ort.loaded = false;
    }
    return g_ort;
}
#endif // XINSP_ENABLE_ORT

// ── Shared post-processing (mirrors algo_caliper_cnn.cpp) ───────────

inline void sigmoid_inplace(float* p, int n) {
    for (int i = 0; i < n; ++i) p[i] = 1.0f / (1.0f + std::exp(-p[i]));
}

std::vector<std::pair<double, double>>  // (y_image, prob)
extract_peaks(const float* probs, double y0) {
    int y_top_image = (int)std::round(y0) - CAL_H / 2;
    std::vector<std::pair<int, double>> raw;
    for (int i = 1; i < CAL_H - 1; ++i) {
        double p = probs[i];
        if (p < PROB_THR) continue;
        if (p > probs[i - 1] && p >= probs[i + 1]) raw.emplace_back(i, p);
    }
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
    std::vector<std::pair<double, double>> out;
    out.reserve(kept.size());
    for (const auto& p : kept) {
        int i = p.first;
        double pm = (i > 0)            ? probs[i - 1] : probs[i];
        double p0 = probs[i];
        double pp = (i < CAL_H - 1)    ? probs[i + 1] : probs[i];
        double denom = pm - 2.0 * p0 + pp;
        double off = (std::abs(denom) > 1e-9) ? 0.5 * (pm - pp) / denom : 0.0;
        if (off < -1.0 || off > 1.0) off = 0.0;
        out.emplace_back((double)y_top_image + i + off, p0);
    }
    return out;
}

std::vector<int> dp_assign(const std::vector<std::vector<Hit>>& per_cal) {
    int K = (int)per_cal.size();
    std::vector<std::vector<double>> dp(K);
    std::vector<std::vector<int>>    pred(K);
    for (int i = 0; i < K; ++i) {
        dp[i].resize(per_cal[i].size(), -1e30);
        pred[i].resize(per_cal[i].size(), -1);
    }
    for (size_t k = 0; k < per_cal[0].size(); ++k) dp[0][k] = per_cal[0][k].prob;
    for (int i = 1; i < K; ++i) {
        for (size_t b = 0; b < per_cal[i].size(); ++b) {
            double bv = -1e30; int ba = -1;
            for (size_t a = 0; a < per_cal[i-1].size(); ++a) {
                double dy = std::abs(per_cal[i][b].y - per_cal[i-1][a].y);
                double v  = dp[i-1][a] - DP_ALPHA * dy;
                if (v > bv) { bv = v; ba = (int)a; }
            }
            if (ba >= 0) {
                dp[i][b] = bv + per_cal[i][b].prob;
                pred[i][b] = ba;
            }
        }
    }
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

#ifdef XINSP_ENABLE_ORT

std::vector<cv::Point2d> detect_caliper_cnn_cross_ort(const cv::Mat& gray,
                                                       const GroundTruth& gt) {
    auto& mh = get_ort();
    if (!mh.loaded) return {};

    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;

    // 1. Caliper x-positions.
    std::vector<int> cxs(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        cxs[i] = (int)std::round((i + 0.5) * (double)W / N_CAL);
        cxs[i] = std::clamp(cxs[i], 1, W - 2);
    }

    // 2. Build [1, N_CAL, CAL_W, CAL_H] input directly into a flat
    //    float buffer ORT will tensor-wrap.
    constexpr int64_t shape[4] = {1, N_CAL, CAL_W, CAL_H};
    std::vector<float> input(N_CAL * CAL_W * CAL_H);
    int y_top = (int)std::round(y0) - CAL_H / 2;
    for (int b = 0; b < N_CAL; ++b) {
        float* dst_b = input.data() + (size_t)b * CAL_W * CAL_H;
        int cx = cxs[b];
        for (int c = 0; c < CAL_W; ++c) {
            float* dst_ch = dst_b + (size_t)c * CAL_H;
            int x = std::clamp(cx + c - CAL_W / 2, 0, W - 1);
            for (int i = 0; i < CAL_H; ++i) {
                int y = std::clamp(y_top + i, 0, H - 1);
                dst_ch[i] = gray.ptr<uint8_t>(y)[x] / 255.0f;
            }
        }
    }

    // 3. Run ORT.
    std::lock_guard<std::mutex> lk(g_ort_mu);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_t = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), shape, 4);
    const char* in_names[]  = { mh.input_name.c_str() };
    const char* out_names[] = { mh.output_name.c_str() };
    std::vector<Ort::Value> outputs;
    try {
        outputs = mh.session->Run(Ort::RunOptions{nullptr},
                                  in_names, &in_t, 1,
                                  out_names, 1);
    } catch (const Ort::Exception&) {
        mh.loaded = false;
        return {};
    }
    float* out_data = outputs[0].GetTensorMutableData<float>();
    sigmoid_inplace(out_data, N_CAL * CAL_H);

    // 4. NMS + DP + linear interp (same as cv::dnn algo).
    std::vector<std::vector<Hit>> per_cal(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        const float* probs = out_data + (size_t)i * CAL_H;
        auto peaks = extract_peaks(probs, y0);
        for (const auto& p : peaks) {
            per_cal[i].push_back({(double)cxs[i], p.first, p.second, i});
        }
    }
    for (auto& v : per_cal) if (v.empty()) v.push_back({0.0, y0, 0.0, -1});
    std::vector<int> picks = dp_assign(per_cal);

    std::vector<cv::Point2d> picks_xy;
    picks_xy.reserve(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        if (picks[i] < 0) continue;
        const Hit& h = per_cal[i][picks[i]];
        if (h.caliper_id < 0) continue;
        picks_xy.emplace_back(h.x, h.y);
    }
    if (picks_xy.size() < 2) {
        std::vector<cv::Point2d> fb; fb.reserve(W);
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

#else  // ORT not enabled at build time

std::vector<cv::Point2d> detect_caliper_cnn_cross_ort(const cv::Mat&,
                                                       const GroundTruth&) {
    // ORT was not linked; algo always reports nohit so the lab table
    // shows it but the row is harmless.
    return {};
}

#endif // XINSP_ENABLE_ORT

} // namespace lab
