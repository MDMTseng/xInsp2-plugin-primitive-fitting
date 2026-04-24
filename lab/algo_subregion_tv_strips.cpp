//
// algo_subregion_tv_strips.cpp — strip-based sparse tensor voting with
// *deferred candidate selection*.
//
// History:
//   v1: emitted tokens at every y-row above threshold in every strip, then
//       picked per-strip argmax-of-saliency (79% outlier scenes).
//   v2 ("rescued"): restricted the pre-TV pool to per-strip top-K=5 NMS peaks
//       on raw |∇I|, then TV, then sparse DP. That fixed robustness but left
//       2% outliers and 0.202/0.276 RMS — because the top-K-by-raw-magnitude
//       shortlist throws away curve-rows with low |∇I|. In spike-dense strips
//       the true curve peak can fail to enter the top-5 and is lost before TV
//       even runs.
//
//   v3 (this file): **defer candidate selection until *after* TV scoring.**
//
//   1. Strip frontend: 64 strips, 5-col averaging, scene-adaptive low
//      threshold (TOKEN_FRAC × band-global max magnitude). Tokens emitted at
//      every local-max-ish row in every strip whose magnitude exceeds the
//      threshold — ~2000 tokens total instead of ~300.
//
//   2. Polarity majority vote, using the per-strip strongest peak to
//      establish curve_sign, then filter tokens of the opposite sign.
//
//   3. Stick-tensor voting among the wider token set (same Medioni decay /
//      LUT / σ / π/4 cone as the dense baseline). O(N²) is still cheap in
//      absolute terms; strip x-pre-bucketing bounds the neighborhood so the
//      inner loop only touches voters within R in x.
//
//   4. Per-strip NMS on **saliency** (not raw magnitude) keeps the top-K
//      tokens per strip with MIN_SEP_Y=4. Curve rows that were low-magnitude
//      but earned high saliency from TV continuity now survive; spike rows
//      that were high-magnitude but earned low saliency get demoted.
//
//   5. Sparse DP across strips (unchanged): node = -sal, edge = α·(Δy/Δx)².
//
//   6. Linear interpolation (unchanged).
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <omp.h>
#include <vector>

namespace lab {

namespace {

constexpr int    N_STRIPS     = 64;     // vertical strips across the image
constexpr int    HALF_STRIP_W = 2;      // 5-column averaging window
constexpr int    HALF_BAND    = 40;     // ±40 px around y0
constexpr int    TOP_K        = 10;     // per-strip saliency-NMS shortlist
constexpr int    MIN_SEP_Y    = 4;      // NMS separation along strip
constexpr double TOKEN_FRAC   = 0.08;   // pre-TV token threshold = frac × band max
constexpr double SIGMA        = 14.0;
constexpr int    DF_D_BINS    = 128;
constexpr int    DF_T_BINS    = 256;

struct DfLut {
    std::vector<float> tbl;
    double d_max, d_step_inv, t_step_inv;
};
DfLut build_df_lut(double sigma) {
    DfLut L;
    double sig2 = sigma * sigma, c_curv = 3.0 * sig2;
    L.d_max = 3.0 * sigma;
    L.d_step_inv = (DF_D_BINS - 1) / L.d_max;
    L.t_step_inv = (DF_T_BINS - 1) / 1.0;
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

struct Token {
    float  x, y;          // position (image coords)
    float  tx, ty;        // unit tangent
    float  gy_sign;       // polarity (+1 / -1)
    float  strength;      // peak magnitude (raw gradient)
    int    strip;         // which strip this token came from
};

} // anon

std::vector<cv::Point2d> detect_subregion_tv_strips(const cv::Mat& gray,
                                                     const GroundTruth& gt) {
    const int W = gray.cols, H = gray.rows;
    const double y0 = gt.y0;
    const int y_lo = std::max(0,     (int)std::round(y0) - HALF_BAND);
    const int y_hi = std::min(H - 1, (int)std::round(y0) + HALF_BAND);

    // --- 1. Gradients ---
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    // --- 2. Strip setup. Strip x-center at round((s + 0.5) * W / N_STRIPS),
    //        5-column averaging.
    std::vector<int> strip_x(N_STRIPS);
    for (int s = 0; s < N_STRIPS; ++s) {
        strip_x[s] = std::min(W - 1,
            (int)std::round((s + 0.5) * (double)W / N_STRIPS));
    }

    // Gather strip-wise |g| profiles.
    std::vector<std::vector<double>> strip_grad(N_STRIPS);
    std::vector<std::vector<double>> strip_gx_avg(N_STRIPS);
    std::vector<std::vector<double>> strip_gy_avg(N_STRIPS);
    const int Bh = y_hi - y_lo + 1;
    double band_max_mag = 0.0;
    double strip_max_sum = 0.0;
    std::vector<double> col_max(N_STRIPS, 0.0);
    for (int s = 0; s < N_STRIPS; ++s) {
        int sx = strip_x[s];
        int x_lo = std::max(0, sx - HALF_STRIP_W);
        int x_hi = std::min(W - 1, sx + HALF_STRIP_W);
        int cols = x_hi - x_lo + 1;
        auto& G  = strip_grad[s];  G.assign(Bh, 0.0);
        auto& GX = strip_gx_avg[s]; GX.assign(Bh, 0.0);
        auto& GY = strip_gy_avg[s]; GY.assign(Bh, 0.0);
        double smax = 0;
        for (int i = 0; i < Bh; ++i) {
            int y = y_lo + i;
            const float* gxr = gx.ptr<float>(y);
            const float* gyr = gy.ptr<float>(y);
            double sgx = 0, sgy = 0;
            for (int x = x_lo; x <= x_hi; ++x) { sgx += gxr[x]; sgy += gyr[x]; }
            sgx /= cols; sgy /= cols;
            double m = std::hypot(sgx, sgy);
            GX[i] = sgx; GY[i] = sgy; G[i] = m;
            if (m > smax) smax = m;
        }
        col_max[s] = smax;
        if (smax > band_max_mag) band_max_mag = smax;
        strip_max_sum += smax;
    }

    // Adaptive low threshold for pre-TV token emission: TOKEN_FRAC of the
    // scene's band-global max magnitude (floor to a small absolute value to
    // cut deep-noise rows).
    double min_tok = std::max(3.0, TOKEN_FRAC * band_max_mag);

    // Also keep a stricter per-strip peak threshold for the polarity-seeding
    // majority vote. We reuse the v2 "50% of 10th-pct strip-max" recipe.
    std::vector<double> sorted_col_max(col_max.begin(), col_max.end());
    std::sort(sorted_col_max.begin(), sorted_col_max.end());
    double min_peak_strong =
        std::max(5.0,
                 0.5 * sorted_col_max[(size_t)(sorted_col_max.size() * 0.10)]);

    // --- 3. Token emission: every local-max-ish row per strip whose
    //        magnitude exceeds min_tok. We still apply a light NMS of 1-row
    //        (strict local max) so that the edge plateau doesn't emit 3 near-
    //        duplicate tokens at every edge — they'd just become near-zero-
    //        distance voters and waste time.
    std::vector<Token> tokens;
    tokens.reserve((size_t)N_STRIPS * 16);
    std::vector<std::vector<int>> strip_tok_idx(N_STRIPS);
    std::vector<int> strip_strongest(N_STRIPS, -1); // index into tokens

    for (int s = 0; s < N_STRIPS; ++s) {
        const auto& G  = strip_grad[s];
        const auto& GX = strip_gx_avg[s];
        const auto& GY = strip_gy_avg[s];
        double strongest_mag = 0.0;
        for (int i = 1; i < Bh - 1; ++i) {
            double g = G[i];
            if (g < min_tok) continue;
            // Local-max filter on |g|. Plateau duplicates are suppressed but
            // we still capture every edge (curve *and* spike back-edge).
            bool is_lmax = (g > G[i - 1] && g >= G[i + 1]);
            if (!is_lmax) continue;

            double y_sub = (double)(y_lo + i);
            double gm = G[i - 1], g0 = g, gp = G[i + 1];
            double denom = gm - 2.0 * g0 + gp;
            if (std::abs(denom) > 1e-9) {
                double off = 0.5 * (gm - gp) / denom;
                if (off > -1.0 && off < 1.0) y_sub += off;
            }
            double avg_gx = GX[i], avg_gy = GY[i];
            double m = std::hypot(avg_gx, avg_gy);
            if (m < 1e-6) continue;
            Token t;
            t.x  = (float)strip_x[s];
            t.y  = (float)y_sub;
            t.tx = (float)(-avg_gy / m);
            t.ty = (float)( avg_gx / m);
            t.gy_sign  = (avg_gy >= 0.0) ? 1.0f : -1.0f;
            t.strength = (float)g;
            t.strip    = s;
            int tid = (int)tokens.size();
            strip_tok_idx[s].push_back(tid);
            tokens.push_back(t);
            if (g >= min_peak_strong && g > strongest_mag) {
                strongest_mag = g;
                strip_strongest[s] = tid;
            }
        }
    }

    if (tokens.size() < 3) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // --- 4. Polarity majority vote. Only the strip's strongest *strong* peak
    //        (>= min_peak_strong) votes, so weak-background tokens don't
    //        dilute the signal.
    int pos = 0, neg = 0;
    for (int s = 0; s < N_STRIPS; ++s) {
        int tid = strip_strongest[s];
        if (tid < 0) continue;
        if (tokens[tid].gy_sign > 0) ++pos; else ++neg;
    }
    float curve_sign = (pos >= neg) ? 1.0f : -1.0f;

    // Mask out opposite-polarity tokens; keep a dense remap of indices.
    std::vector<Token> ftokens;
    ftokens.reserve(tokens.size());
    std::vector<std::vector<int>> strip_fidx(N_STRIPS);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].gy_sign != curve_sign) continue;
        strip_fidx[tokens[i].strip].push_back((int)ftokens.size());
        ftokens.push_back(tokens[i]);
    }
    if (ftokens.size() < 3) {
        // Fallback: keep all tokens if filter was too aggressive.
        ftokens = tokens;
        for (int s = 0; s < N_STRIPS; ++s) strip_fidx[s] = strip_tok_idx[s];
    }
    const int Ntok = (int)ftokens.size();

    // --- 5. Stick-tensor voting among filtered tokens. Same math as the
    //        dense baseline (LUT over (d, cosθ), π/4 cone). With ~2000
    //        tokens the O(N²) is still cheap: we pre-bucket tokens by strip
    //        index and iterate only over strips whose x is within voting
    //        radius R of the receiver's x. This bounds voter count per
    //        receiver to a few hundred rather than the full N.
    const DfLut lut = build_df_lut(SIGMA);
    const double R  = 2.5 * SIGMA;
    const double R2 = R * R;
    const double cos_cone = 0.70710678118654752;
    const double strip_dx = (double)W / (double)N_STRIPS;
    const int strip_range = (int)std::ceil(R / strip_dx) + 1;

    std::vector<double> Txx(Ntok, 0.0), Txy(Ntok, 0.0), Tyy(Ntok, 0.0);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < Ntok; ++i) {
        double px = ftokens[i].x, py = ftokens[i].y;
        int si = ftokens[i].strip;
        int s0 = std::max(0, si - strip_range);
        int s1 = std::min(N_STRIPS - 1, si + strip_range);
        double axx = 0, axy = 0, ayy = 0;
        for (int ss = s0; ss <= s1; ++ss) {
            const auto& bucket = strip_fidx[ss];
            for (int j : bucket) {
                if (j == i) continue;
                double dx = ftokens[j].x - px;
                double dy = ftokens[j].y - py;
                double d2 = dx*dx + dy*dy;
                if (d2 > R2 || d2 < 1e-9) continue;
                double d = std::sqrt(d2);
                double inv_d = 1.0 / d;
                double vx = dx * inv_d, vy = dy * inv_d;
                double tjx = ftokens[j].tx, tjy = ftokens[j].ty;
                double dot = (-vx) * tjx + (-vy) * tjy;
                double abs_dot = (dot >= 0) ? dot : -dot;
                if (abs_dot < cos_cone) continue;
                float df = df_lookup(lut, d, abs_dot);
                if (df < 1e-4f) continue;
                double sgn   = (dot >= 0) ? 1.0 : -1.0;
                double tsx   = sgn * tjx, tsy = sgn * tjy;
                double nx    = -vx, ny = -vy;
                double ra    = 2.0 * (tsx * nx + tsy * ny);
                double r_x   = ra * nx - tsx;
                double r_y   = ra * ny - tsy;
                double w     = (double)df * (double)ftokens[j].strength;
                axx += w * r_x * r_x;
                axy += w * r_x * r_y;
                ayy += w * r_y * r_y;
            }
        }
        Txx[i] = axx; Txy[i] = axy; Tyy[i] = ayy;
    }

    // --- 6. Per-token saliency = λ1 − λ2.
    std::vector<double> sal(Ntok, 0.0);
    for (int i = 0; i < Ntok; ++i) {
        double tr = Txx[i] + Tyy[i];
        double det = Txx[i] * Tyy[i] - Txy[i] * Txy[i];
        double disc = std::sqrt(std::max(0.0, tr*tr*0.25 - det));
        sal[i] = 2.0 * disc;
    }
    double sal_max = 0;
    for (double v : sal) if (v > sal_max) sal_max = v;
    if (sal_max < 1e-9) sal_max = 1.0;
    for (double& v : sal) v /= sal_max;

    // --- 7. Per-strip saliency-NMS: keep top-K candidates per strip by
    //        *saliency*, enforcing MIN_SEP_Y. Curve-tokens that got promoted
    //        by TV now compete on equal footing with spike-tokens.
    std::vector<std::vector<int>> strip_cand(N_STRIPS);  // ftoken indices
    for (int s = 0; s < N_STRIPS; ++s) {
        auto& bucket = strip_fidx[s];
        if (bucket.empty()) continue;
        // Sort by saliency descending.
        std::sort(bucket.begin(), bucket.end(),
                  [&sal](int a, int b) { return sal[a] > sal[b]; });
        auto& kept = strip_cand[s];
        kept.reserve(TOP_K);
        for (int idx : bucket) {
            bool ok = true;
            for (int kidx : kept) {
                if (std::abs(ftokens[idx].y - ftokens[kidx].y) < MIN_SEP_Y) {
                    ok = false; break;
                }
            }
            if (!ok) continue;
            kept.push_back(idx);
            if ((int)kept.size() >= TOP_K) break;
        }
    }

    // --- 8. Sparse DP across strips: node cost = −sal, edge cost =
    //        α·(Δy/Δx)². Only strips with ≥1 candidate participate.
    struct NodeCost {
        std::array<double, TOP_K> cost;
        std::array<int,    TOP_K> prev;
        std::array<int,    TOP_K> tok_idx;
        int ncand = 0;
    };

    std::vector<int> active_strips;
    active_strips.reserve(N_STRIPS);
    for (int s = 0; s < N_STRIPS; ++s) {
        if (!strip_cand[s].empty()) active_strips.push_back(s);
    }
    if (active_strips.size() < 2) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    const int Ns = (int)active_strips.size();
    std::vector<NodeCost> dp(Ns);
    for (int si = 0; si < Ns; ++si) {
        int s = active_strips[si];
        int n = std::min((int)strip_cand[s].size(), TOP_K);
        dp[si].ncand = n;
        for (int k = 0; k < TOP_K; ++k) {
            dp[si].cost[k] = std::numeric_limits<double>::infinity();
            dp[si].prev[k] = -1;
            dp[si].tok_idx[k] = -1;
        }
        for (int k = 0; k < n; ++k) {
            dp[si].tok_idx[k] = strip_cand[s][k];
        }
    }

    // Data-driven α: median strip sal-gap times 2, floored at 0.25.
    double alpha = 0.5;
    {
        std::vector<double> gaps;
        gaps.reserve(Ns);
        for (int si = 0; si < Ns; ++si) {
            if (dp[si].ncand < 2) continue;
            double hi = sal[dp[si].tok_idx[0]];
            double lo = sal[dp[si].tok_idx[dp[si].ncand - 1]];
            gaps.push_back(std::max(0.0, hi - lo));
        }
        if (!gaps.empty()) {
            std::sort(gaps.begin(), gaps.end());
            alpha = std::max(0.25, 2.0 * gaps[gaps.size() / 2]);
        }
    }

    // Boundary
    for (int k = 0; k < dp[0].ncand; ++k) {
        dp[0].cost[k] = -sal[dp[0].tok_idx[k]];
    }
    for (int si = 1; si < Ns; ++si) {
        double dx = ftokens[dp[si].tok_idx[0]].x - ftokens[dp[si - 1].tok_idx[0]].x;
        double dx2 = std::max(1.0, dx * dx);
        for (int k = 0; k < dp[si].ncand; ++k) {
            double y_k = ftokens[dp[si].tok_idx[k]].y;
            double best = std::numeric_limits<double>::infinity();
            int best_p = -1;
            for (int kp = 0; kp < dp[si - 1].ncand; ++kp) {
                double y_p = ftokens[dp[si - 1].tok_idx[kp]].y;
                double dy = y_k - y_p;
                double c  = dp[si - 1].cost[kp] + alpha * (dy * dy) / dx2;
                if (c < best) { best = c; best_p = kp; }
            }
            if (best_p < 0) continue;
            dp[si].cost[k] = best - sal[dp[si].tok_idx[k]];
            dp[si].prev[k] = best_p;
        }
    }

    // Backtrack
    int last = Ns - 1;
    int best_k = -1;
    double best_c = std::numeric_limits<double>::infinity();
    for (int k = 0; k < dp[last].ncand; ++k) {
        if (dp[last].cost[k] < best_c) { best_c = dp[last].cost[k]; best_k = k; }
    }
    if (best_k < 0) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }
    std::vector<int> pick(Ns, -1);
    pick[last] = best_k;
    for (int si = last; si > 0; --si) pick[si - 1] = dp[si].prev[pick[si]];

    std::vector<std::pair<double, double>> picked;
    picked.reserve(Ns);
    for (int si = 0; si < Ns; ++si) {
        int k = pick[si];
        if (k < 0 || k >= dp[si].ncand) continue;
        int tid = dp[si].tok_idx[k];
        picked.emplace_back(ftokens[tid].x, ftokens[tid].y);
    }
    if (picked.size() < 2) {
        std::vector<cv::Point2d> out;
        out.reserve(W);
        for (int x = 0; x < W; ++x) out.emplace_back((double)x, y0);
        return out;
    }

    // --- 9. Dense output via linear interpolation between picks.
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
            double x0 = picked[j].first,     y_0 = picked[j].second;
            double x1 = picked[j + 1].first, y_1 = picked[j + 1].second;
            double t = (x1 > x0) ? (xd - x0) / (x1 - x0) : 0.0;
            y = y_0 + t * (y_1 - y_0);
        }
        out.emplace_back(xd, y);
    }
    return out;
}

} // namespace lab
