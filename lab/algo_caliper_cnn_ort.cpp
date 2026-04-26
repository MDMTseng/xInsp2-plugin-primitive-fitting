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
#include <random>
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

#endif // XINSP_ENABLE_ORT (closing the cross_ort block)

// ── CNN + PROSAC variant ───────────────────────────────────────────
// Same ORT forward as detect_caliper_cnn_cross_ort, but the per-
// caliper NMS hits feed a deg-3 polynomial PROSAC fit (Progressive
// Sample Consensus, sampled by descending CNN probability) plus
// two Huber-IRLS rounds, instead of the sparse Viterbi DP.
// Output is the polynomial sampled at every integer x — natively
// C∞ continuous, no piecewise-linear knot artefacts.

namespace {

// Solve weighted-LSQ for poly coefficients in normalised u ∈ [-1, 1].
// Returns false on near-singular system.
bool fit_poly_w(const std::vector<double>& us,
                const std::vector<double>& vs,
                const std::vector<double>& ws,
                int degree, std::vector<double>& coeffs) {
    const int n = (int)us.size();
    const int m = degree + 1;
    if (n < m) return false;
    constexpr int M = 8;        // supports degree up to 7
    if (m > M) return false;
    double ATA[M][M] = {};
    double ATb[M] = {};
    for (int i = 0; i < n; ++i) {
        double u = us[i], v = vs[i], w = (i < (int)ws.size() ? ws[i] : 1.0);
        double pow_[M]; pow_[0] = 1.0;
        for (int k = 1; k < m; ++k) pow_[k] = pow_[k - 1] * u;
        for (int j = 0; j < m; ++j) {
            double wpj = w * pow_[j];
            ATb[j] += wpj * v;
            for (int k = j; k < m; ++k) ATA[j][k] += wpj * pow_[k];
        }
    }
    for (int j = 1; j < m; ++j)
        for (int k = 0; k < j; ++k) ATA[j][k] = ATA[k][j];
    // Cholesky.
    double L[M][M] = {};
    for (int j = 0; j < m; ++j) {
        double s = ATA[j][j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
        if (s <= 1e-15) return false;
        L[j][j] = std::sqrt(s);
        double inv_d = 1.0 / L[j][j];
        for (int i = j + 1; i < m; ++i) {
            double s2 = ATA[i][j];
            for (int k = 0; k < j; ++k) s2 -= L[i][k] * L[j][k];
            L[i][j] = s2 * inv_d;
        }
    }
    double y[M];
    for (int j = 0; j < m; ++j) {
        double s = ATb[j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * y[k];
        y[j] = s / L[j][j];
    }
    coeffs.assign(m, 0.0);
    for (int j = m - 1; j >= 0; --j) {
        double s = y[j];
        for (int k = j + 1; k < m; ++k) s -= L[k][j] * coeffs[k];
        coeffs[j] = s / L[j][j];
    }
    return true;
}

inline double poly_eval(const std::vector<double>& c, double u) {
    double v = 0.0;
    for (int i = (int)c.size() - 1; i >= 0; --i) v = v * u + c[i];
    return v;
}

// Adaptive-degree PROSAC over (x, y, prob, caliper_id) hits.  Score is
//   Σ_inlier prob × (unique_calipers_with_inlier / N_CAL)
// — the same coverage-weighted trick that drove caliper_ransac from
// 23% outlier-scene rate to 0% in earlier lab work. Sweeps degrees
// 1..3 with an MDL-style penalty, then runs two Huber-IRLS rounds
// on the winning model.
bool fit_prosac_adaptive(std::vector<Hit> hits, int W, double y0,
                         int n_cal_total, int& best_degree_out,
                         std::vector<double>& best_coeffs) {
    if (hits.size() < 2) return false;
    std::sort(hits.begin(), hits.end(),
        [](const Hit& a, const Hit& b){ return a.prob > b.prob; });
    const int N = (int)hits.size();
    const double half_w = 0.5 * W;
    std::vector<double> us(N), vs(N), ws(N);
    std::vector<int>    cid(N);
    for (int i = 0; i < N; ++i) {
        us[i] = (hits[i].x - half_w) / half_w;
        vs[i] = hits[i].y - y0;
        ws[i] = hits[i].prob;
        cid[i] = hits[i].caliper_id;
    }
    constexpr int    DEG_MIN   = 1;
    constexpr int    DEG_MAX   = 3;
    constexpr double INLIER_PX = 1.5;
    constexpr double C_HUBER   = 0.5;
    constexpr double DEG_PEN   = 0.5;     // MDL bias toward lower degree

    std::mt19937 rng(0x5EED);
    double best_overall = -1.0;
    best_coeffs.clear();
    best_degree_out = DEG_MIN;

    for (int degree = DEG_MIN; degree <= DEG_MAX; ++degree) {
        const int NEED = degree + 1;
        if (N < NEED) continue;
        const int ITERS = 30 + 15 * (degree - DEG_MIN);
        double      deg_best = -1.0;
        std::vector<double> deg_best_coeffs;
        for (int it = 0; it < ITERS; ++it) {
            int M = NEED + (int)((double)(N - NEED) * (double)it /
                                 std::max(1, ITERS - 1));
            M = std::clamp(M, NEED, N);
            std::uniform_int_distribution<int> pick(0, M - 1);
            int idx[8];
            for (int k = 0; k < NEED; ++k) {
                while (true) {
                    int c = pick(rng);
                    bool dup = false;
                    for (int j = 0; j < k; ++j) if (idx[j] == c) { dup = true; break; }
                    if (!dup) { idx[k] = c; break; }
                }
            }
            std::vector<double> su(NEED), sv(NEED), sw(NEED, 1.0);
            for (int k = 0; k < NEED; ++k) { su[k] = us[idx[k]]; sv[k] = vs[idx[k]]; }
            double umin = 1e9, umax = -1e9;
            for (int k = 0; k < NEED; ++k) { umin = std::min(umin, su[k]); umax = std::max(umax, su[k]); }
            if (umax - umin < 0.4) continue;
            std::vector<double> coeffs;
            if (!fit_poly_w(su, sv, sw, degree, coeffs)) continue;
            double sum_prob = 0.0;
            std::vector<uint8_t> hit_cal(n_cal_total, 0);
            for (int j = 0; j < N; ++j) {
                double r = vs[j] - poly_eval(coeffs, us[j]);
                if (std::abs(r) > INLIER_PX) continue;
                sum_prob += ws[j];
                if (cid[j] >= 0 && cid[j] < n_cal_total) hit_cal[cid[j]] = 1;
            }
            int cov_count = 0;
            for (int c = 0; c < n_cal_total; ++c) cov_count += hit_cal[c];
            double cov_frac = (double)cov_count / std::max(1, n_cal_total);
            double s = sum_prob * cov_frac;          // *the* score
            if (s > deg_best) {
                deg_best = s;
                deg_best_coeffs = coeffs;
            }
        }
        if (deg_best <= 0.0) continue;
        double mdl = deg_best - DEG_PEN * (double)(degree - DEG_MIN);
        if (mdl > best_overall) {
            best_overall = mdl;
            best_coeffs  = deg_best_coeffs;
            best_degree_out = degree;
        }
    }
    if (best_coeffs.empty()) return false;
    // Huber-IRLS refinement on the winning model.
    for (int round = 0; round < 2; ++round) {
        std::vector<double> in_u, in_v, in_w;
        in_u.reserve(N); in_v.reserve(N); in_w.reserve(N);
        for (int j = 0; j < N; ++j) {
            double r = vs[j] - poly_eval(best_coeffs, us[j]);
            if (std::abs(r) > INLIER_PX) continue;
            double ar = std::abs(r);
            double huber = ar <= C_HUBER ? 1.0 : C_HUBER / std::max(ar, 1e-9);
            in_u.push_back(us[j]);
            in_v.push_back(vs[j]);
            in_w.push_back(ws[j] * huber);
        }
        if ((int)in_u.size() < best_degree_out + 2) break;
        std::vector<double> refined;
        if (fit_poly_w(in_u, in_v, in_w, best_degree_out, refined))
            best_coeffs = refined;
    }
    return true;
}

} // namespace

#ifdef XINSP_ENABLE_ORT

std::vector<cv::Point2d> detect_caliper_cnn_cross_ort_prosac(const cv::Mat& gray,
                                                              const GroundTruth& gt) {
    auto& mh = get_ort();
    if (!mh.loaded) return {};

    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;

    std::vector<int> cxs(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        cxs[i] = (int)std::round((i + 0.5) * (double)W / N_CAL);
        cxs[i] = std::clamp(cxs[i], 1, W - 2);
    }
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

    // Per-caliper NMS top-K, then sparse Viterbi DP picks 1 hit per
    // caliper — same as detect_caliper_cnn_cross_ort but the picked
    // (x, y, prob) sequence then feeds an adaptive-degree polynomial
    // LSQ fit (PROSAC over the 16 DP picks with CNN probs as weights).
    // Combines DP's structural outlier rejection (spike chains cannot
    // form a smooth y(x) sequence) with poly smoothness on output.
    std::vector<std::vector<Hit>> per_cal(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        const float* probs = out_data + (size_t)i * CAL_H;
        for (const auto& p : extract_peaks(probs, y0)) {
            per_cal[i].push_back({(double)cxs[i], p.first, p.second, i});
        }
    }
    for (auto& v : per_cal) if (v.empty()) v.push_back({0.0, y0, 0.0, -1});
    std::vector<int> picks = dp_assign(per_cal);

    std::vector<Hit> dp_hits;
    dp_hits.reserve(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        if (picks[i] < 0) continue;
        const Hit& h = per_cal[i][picks[i]];
        if (h.caliper_id < 0) continue;
        dp_hits.push_back(h);
    }
    if (dp_hits.size() < 2) {
        std::vector<cv::Point2d> fb; fb.reserve(W);
        for (int x = 0; x < W; ++x) fb.emplace_back((double)x, y0);
        return fb;
    }

    // Direct weighted LSQ (no PROSAC needed — DP picks are already
    // outlier-clean) over the 16 DP picks, with degree sweep 1..3 and
    // MDL bias toward the simpler model. CNN prob is the LSQ weight.
    const double half_wc = 0.5 * W;
    std::vector<double> us_dp(dp_hits.size()), vs_dp(dp_hits.size()),
                        ws_dp(dp_hits.size());
    for (size_t i = 0; i < dp_hits.size(); ++i) {
        us_dp[i] = (dp_hits[i].x - half_wc) / half_wc;
        vs_dp[i] = dp_hits[i].y - y0;
        ws_dp[i] = std::max(0.05, dp_hits[i].prob);
    }
    int best_degree = 1;
    std::vector<double> coeffs;
    double best_score = std::numeric_limits<double>::lowest();
    constexpr double MDL_PEN = 0.08;        // each extra degree costs 0.08 px² RMS
    for (int deg = 1; deg <= 3; ++deg) {
        if ((int)us_dp.size() < deg + 2) continue;
        std::vector<double> c;
        if (!fit_poly_w(us_dp, vs_dp, ws_dp, deg, c)) continue;
        // Weighted RMS in y-pixels.
        double sse = 0.0, wsum = 0.0;
        for (size_t i = 0; i < us_dp.size(); ++i) {
            double r = vs_dp[i] - poly_eval(c, us_dp[i]);
            sse  += ws_dp[i] * r * r;
            wsum += ws_dp[i];
        }
        double wrms = std::sqrt(sse / std::max(1e-9, wsum));
        double score = -wrms - MDL_PEN * (double)(deg - 1);
        if (score > best_score) {
            best_score = score;
            coeffs = c;
            best_degree = deg;
        }
    }
    if (coeffs.empty()) {
        std::vector<cv::Point2d> fb; fb.reserve(W);
        for (int x = 0; x < W; ++x) fb.emplace_back((double)x, y0);
        return fb;
    }

    // Output: polynomial inside inlier x-span; linear extrapolation
    // (C¹-anchored at the boundary) outside. Same trick as
    // caliper_ransac §2.7 — eliminates cubic blow-up at x=0 / x=W.
    //
    // Safety net: if the polynomial deviates from the linear-interp
    // baseline by more than DEV_THR_PX anywhere within the inlier
    // span, we abandon the polynomial and emit the linear-interp
    // path instead. This catches the cubic-overshoot scenes (~5%
    // without the safeguard) without giving up the median-RMS gain
    // on well-behaved scenes.
    const double half_w = 0.5 * W;
    double x_min_in = 1e30, x_max_in = -1e30;
    for (const Hit& h : dp_hits) {
        x_min_in = std::min(x_min_in, h.x);
        x_max_in = std::max(x_max_in, h.x);
    }
    if (!(x_min_in < x_max_in)) { x_min_in = 0; x_max_in = W - 1; }

    auto eval_at = [&](double x) {
        double u = (x - half_w) / half_w;
        return poly_eval(coeffs, u) + y0;
    };
    auto slope_at = [&](double x) {
        double u = (x - half_w) / half_w;
        double dv_du = 0.0;
        for (int k = (int)coeffs.size() - 1; k >= 1; --k)
            dv_du = dv_du * u + (double)k * coeffs[k];
        return dv_du / half_w;
    };

    // Sort DP hits by x for the linear-interp baseline lookup.
    std::vector<Hit> sorted_hits = dp_hits;
    std::sort(sorted_hits.begin(), sorted_hits.end(),
        [](const Hit& a, const Hit& b){ return a.x < b.x; });
    auto linear_at = [&](double x) -> double {
        const auto& v = sorted_hits;
        if (x <= v.front().x) return v.front().y;
        if (x >= v.back().x)  return v.back().y;
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            if (x <= v[i+1].x) {
                double t = (x - v[i].x) / std::max(1e-9, v[i+1].x - v[i].x);
                return (1.0 - t) * v[i].y + t * v[i+1].y;
            }
        }
        return v.back().y;
    };

    constexpr double DEV_THR_PX = 3.0;
    bool safe = true;
    for (int x = (int)std::ceil(x_min_in); x <= (int)std::floor(x_max_in); x += 4) {
        double dev = std::abs(eval_at((double)x) - linear_at((double)x));
        if (dev > DEV_THR_PX) { safe = false; break; }
    }

    const double y_lo = eval_at(x_min_in);
    const double y_hi = eval_at(x_max_in);
    const double m_lo = slope_at(x_min_in);
    const double m_hi = slope_at(x_max_in);
    std::vector<cv::Point2d> out_dense;
    out_dense.reserve(W);
    if (!safe) {
        // Polynomial overshot — fall back to linear interp through DP picks.
        for (int x = 0; x < W; ++x) {
            out_dense.emplace_back((double)x, linear_at((double)x));
        }
        return out_dense;
    }
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        double y;
        if      (xd < x_min_in) y = y_lo + m_lo * (xd - x_min_in);
        else if (xd > x_max_in) y = y_hi + m_hi * (xd - x_max_in);
        else                    y = eval_at(xd);
        out_dense.emplace_back(xd, y);
    }
    return out_dense;
}

#endif // XINSP_ENABLE_ORT (closing the cross_ort_prosac block)

// ── CNN + natural cubic spline post-fit ───────────────────────────
// Same CNN+DP as cross_ort_prosac, but output is a natural cubic
// spline through the 16 DP picks (instead of polynomial LSQ +
// safety net). Spline traces through every knot exactly, giving
// C² continuity. With the same safety check (max-deviation from
// linear-interp baseline > 3 px ⇒ fall back to linear) for
// the rare scenes where DP picks have enough noise to make spline
// over-shoot between knots.

namespace {
// Solve natural cubic spline through K knot points (xs, ys).
// Output is a vector of (a, b, c, d) per segment; piece i covers
// xs[i] .. xs[i+1] and evaluates as
//   y(x) = a_i + b_i·t + c_i·t² + d_i·t³  where t = (x − xs[i]) / h_i.
// Returns false if K < 3 or the system is degenerate.
bool fit_natural_cubic_spline(const std::vector<double>& xs,
                              const std::vector<double>& ys,
                              std::vector<std::array<double, 4>>& seg) {
    const int n = (int)xs.size();
    if (n < 3) return false;
    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i] = xs[i + 1] - xs[i];
        if (h[i] <= 0) return false;
    }
    // Tridiagonal system for second derivatives M_i (M_0 = M_{n-1} = 0).
    std::vector<double> sub(n, 0.0), diag(n, 0.0), sup(n, 0.0), rhs(n, 0.0);
    diag[0] = 1.0; rhs[0] = 0.0;
    diag[n-1] = 1.0; rhs[n-1] = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        sub[i]  = h[i-1];
        diag[i] = 2.0 * (h[i-1] + h[i]);
        sup[i]  = h[i];
        rhs[i]  = 6.0 * ((ys[i+1] - ys[i]) / h[i] - (ys[i] - ys[i-1]) / h[i-1]);
    }
    for (int i = 1; i < n; ++i) {
        double w = sub[i] / diag[i-1];
        diag[i] -= w * sup[i-1];
        rhs[i]  -= w * rhs[i-1];
    }
    std::vector<double> M(n);
    M[n-1] = rhs[n-1] / diag[n-1];
    for (int i = n - 2; i >= 0; --i) M[i] = (rhs[i] - sup[i] * M[i+1]) / diag[i];
    seg.assign(n - 1, {0, 0, 0, 0});
    for (int i = 0; i < n - 1; ++i) {
        // y(x) = ((M_i+1 - M_i)/(6 h)) (x - xi)^3
        //      + (M_i / 2) (x - xi)^2
        //      + ((y_{i+1} - y_i)/h - h(M_{i+1} + 2 M_i)/6) (x - xi)
        //      + y_i
        // Reparametrise to t = (x - xi)/h:
        //   y(t) = a + b·t + c·t² + d·t³
        double hi = h[i];
        double a = ys[i];
        double b = (ys[i+1] - ys[i]) - hi * hi * (2.0 * M[i] + M[i+1]) / 6.0;
        double c = hi * hi * M[i] / 2.0;
        double d = hi * hi * (M[i+1] - M[i]) / 6.0;
        seg[i] = {a, b, c, d};
    }
    return true;
}
} // namespace

#ifdef XINSP_ENABLE_ORT

std::vector<cv::Point2d> detect_caliper_cnn_cross_ort_spline(const cv::Mat& gray,
                                                              const GroundTruth& gt) {
    auto& mh = get_ort();
    if (!mh.loaded) return {};
    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;

    std::vector<int> cxs(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        cxs[i] = (int)std::round((i + 0.5) * (double)W / N_CAL);
        cxs[i] = std::clamp(cxs[i], 1, W - 2);
    }
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

    std::vector<std::vector<Hit>> per_cal(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        const float* probs = out_data + (size_t)i * CAL_H;
        for (const auto& p : extract_peaks(probs, y0)) {
            per_cal[i].push_back({(double)cxs[i], p.first, p.second, i});
        }
    }
    for (auto& v : per_cal) if (v.empty()) v.push_back({0.0, y0, 0.0, -1});
    std::vector<int> picks = dp_assign(per_cal);
    std::vector<Hit> dp_hits;
    dp_hits.reserve(N_CAL);
    for (int i = 0; i < N_CAL; ++i) {
        if (picks[i] < 0) continue;
        const Hit& h = per_cal[i][picks[i]];
        if (h.caliper_id < 0) continue;
        dp_hits.push_back(h);
    }
    if (dp_hits.size() < 3) {
        std::vector<cv::Point2d> fb; fb.reserve(W);
        for (int x = 0; x < W; ++x) fb.emplace_back((double)x, y0);
        return fb;
    }
    // Sort by x for spline.
    std::sort(dp_hits.begin(), dp_hits.end(),
        [](const Hit& a, const Hit& b){ return a.x < b.x; });
    std::vector<double> xs(dp_hits.size()), ys(dp_hits.size());
    for (size_t i = 0; i < dp_hits.size(); ++i) {
        xs[i] = dp_hits[i].x;
        ys[i] = dp_hits[i].y;
    }
    std::vector<std::array<double, 4>> seg;
    if (!fit_natural_cubic_spline(xs, ys, seg)) {
        std::vector<cv::Point2d> fb; fb.reserve(W);
        for (int x = 0; x < W; ++x) fb.emplace_back((double)x, y0);
        return fb;
    }

    auto spline_at = [&](double x) -> double {
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back())  return ys.back();
        // Find segment i: xs[i] ≤ x ≤ xs[i+1].
        size_t i = 0;
        for (; i + 1 < xs.size(); ++i) if (x <= xs[i + 1]) break;
        double hi = xs[i + 1] - xs[i];
        double t = (x - xs[i]) / hi;
        const auto& s = seg[i];
        return s[0] + s[1] * t + s[2] * t * t + s[3] * t * t * t;
    };
    auto linear_at = [&](double x) -> double {
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back())  return ys.back();
        for (size_t i = 0; i + 1 < xs.size(); ++i) {
            if (x <= xs[i + 1]) {
                double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
                return (1.0 - t) * ys[i] + t * ys[i + 1];
            }
        }
        return ys.back();
    };

    // Safety check: spline should not deviate from linear baseline by
    // more than DEV_THR_PX anywhere.
    constexpr double DEV_THR_PX = 3.0;
    bool safe = true;
    for (int x = (int)std::ceil(xs.front()); x <= (int)std::floor(xs.back()); x += 4) {
        double dev = std::abs(spline_at((double)x) - linear_at((double)x));
        if (dev > DEV_THR_PX) { safe = false; break; }
    }

    std::vector<cv::Point2d> out_dense;
    out_dense.reserve(W);
    if (!safe) {
        for (int x = 0; x < W; ++x) {
            out_dense.emplace_back((double)x, linear_at((double)x));
        }
        return out_dense;
    }
    for (int x = 0; x < W; ++x) {
        double xd = (double)x;
        if      (xd < xs.front()) out_dense.emplace_back(xd, ys.front());
        else if (xd > xs.back())  out_dense.emplace_back(xd, ys.back());
        else                      out_dense.emplace_back(xd, spline_at(xd));
    }
    return out_dense;
}

#else

std::vector<cv::Point2d> detect_caliper_cnn_cross_ort(const cv::Mat&,
                                                       const GroundTruth&) {
    return {};
}
std::vector<cv::Point2d> detect_caliper_cnn_cross_ort_prosac(const cv::Mat&,
                                                              const GroundTruth&) {
    return {};
}
std::vector<cv::Point2d> detect_caliper_cnn_cross_ort_spline(const cv::Mat&,
                                                              const GroundTruth&) {
    return {};
}

#endif // XINSP_ENABLE_ORT

} // namespace lab
