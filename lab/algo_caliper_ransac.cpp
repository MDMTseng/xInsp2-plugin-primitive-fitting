//
// algo_caliper_ransac.cpp — caliper sampling + RANSAC polynomial fit.
//
// Pipeline (self-contained, no primitive_fitting plugin dep):
//
//  1. Caliper sampling. Calipers are placed uniformly across x plus a
//     density boost near both image edges to stabilise polynomial
//     boundary behaviour. Each caliper averages HALF_W*2+1 columns,
//     takes a central-difference gradient along y, and extracts up to
//     TOP_K strongest local peaks on |grad| (NMS with MIN_SEP-px
//     separation, sub-pixel parabolic refinement). The edge-strength
//     threshold is *data-driven* — a quantile of the image's vertical
//     |gradient| around y0 — so it adapts to scene contrast / noise.
//
//     Each hit also carries the *signed* gradient. The scene generator
//     places the curve as a dark→bright transition, while spikes are
//     bright blobs on dark background — so spike top edges have the
//     *same* sign as the curve but spike bottom edges have the opposite
//     sign. We estimate the dominant curve-edge sign from the set of
//     hits (its x-distribution is necessarily wide — one peak per
//     caliper) and drop opposite-sign peaks. That alone strips out
//     ~half of all spike peaks.
//
//  2. Adaptive-degree RANSAC polynomial fit in normalised coords
//        u = (x - W/2) / (W/2)  ∈ [-1, 1]
//        v = y - gt.y0.
//     We try degrees 1 … 5. The RANSAC score for a hypothesis is
//        inliers · coverage_fraction
//     where `coverage_fraction` is the number of distinct caliper x's
//     hit by inliers over the total number of calipers — this punishes
//     hypotheses that pile up inliers in a narrow x-band (which is what
//     spike-only fits look like). Minimal samples with near-zero u-spread
//     are rejected (ill-conditioned Vandermonde) and near-vertical fits
//     are rejected by slope-sanity. Across degrees we subtract an
//     MDL-style DEGREE_PENALTY × (degree − 1) so lower-degree models win
//     ties. The winner is refined by iterative LSQ + Huber-IRLS over all
//     hits (not only the RANSAC inliers), which recovers edge points the
//     minimal sample initially missed.
//
//  3. Sample the polynomial at every integer x ∈ [0, W). *Outside* the
//     inlier x-span (± a small geometric pad) we clamp to the
//     polynomial evaluated at the nearest inlier-x rather than
//     extrapolating further — this kills endpoint blow-up without
//     obliterating coverage.
//
// Only `gt.y0`, `gt.W`, `gt.H` are read from ground truth (legitimate
// priors per the lab contract).
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace lab {

namespace {

// ---- Tunables -------------------------------------------------------------
constexpr int    N_CALIPERS_CORE  = 32;     // uniform sampling lines
constexpr int    N_CALIPERS_EDGE  = 8;      // extra calipers inside each edge band
constexpr double EDGE_FRAC        = 0.08;   // edge band width (fraction of W)
constexpr int    CALIPER_SPAN     = 80;     // vertical extent (pixels)
constexpr int    HALF_W           = 1;      // horizontal averaging width
constexpr int    TOP_K            = 3;      // peaks kept per caliper
constexpr int    MIN_SEP          = 4;      // NMS separation in y (pixels)

// Edge-strength threshold is adaptive, with a sensible floor.
constexpr double MIN_EDGE_FLOOR   = 4.0;
// Scene gradient quantile used to set the per-run strength threshold.
constexpr double STRENGTH_QUANTILE= 0.88;

// Base number of RANSAC iterations for degree 1; we scale up per degree
// because the minimal sample is larger and the prior is weaker.
// With PROSAC (strength-ordered sampling), far fewer iterations are
// needed than under uniform RANSAC — early minimal samples from the
// top-strength window are overwhelmingly inliers.
constexpr int    RANSAC_ITERS_BASE= 45;
// USER-FACING tolerance: which hits the caller considers "in tolerance".
// In the plugin port this is a runtime parameter; in the lab it's a
// constant. It should only affect output labelling / extrapolation-span
// decisions — never the fit itself. Changing this value must not change
// the fitted polynomial.
constexpr double INLIER_THR       = 2.0;    // vertical pixels
// INTERNAL fit-precision threshold: used by RANSAC scoring and IRLS
// inlier gating. Locked to the sub-pixel-refinement scale of the
// Gaussian derivative + parabolic peaks. Does NOT follow INLIER_THR —
// loosening INLIER_THR must not degrade the fit.
constexpr double FIT_THR          = 2.0;    // vertical pixels

// Adaptive polynomial degrees to try.
constexpr int    DEG_MIN          = 1;
constexpr int    DEG_MAX          = 5;
// Inlier-equivalent penalty for each extra polynomial degree beyond
// DEG_MIN — an MDL-style bias toward simpler models.
constexpr double DEGREE_PENALTY   = 4.0;

struct Hit {
    double x;
    double y;
    double strength;   // |grad|
    int    sign;       // +1 or -1 (sign of the signed gradient)
    int    caliper_id; // index of the caliper this hit came from (for spread metric)
};

// Evaluate polynomial: coeffs[0] + coeffs[1]*u + ... + coeffs[deg]*u^deg.
inline double poly_eval(const std::vector<double>& c, double u) {
    double v = 0.0;
    for (int i = (int)c.size() - 1; i >= 0; --i) v = v * u + c[i];
    return v;
}

// ---- Stack-allocated polynomial LSQ ---------------------------------------
// Build normal equations A^T W A x = A^T W b with a stack matrix, solve via
// Cholesky. Replaces cv::solve() — no heap allocation, no cv::Mat overhead.
// Observed ~15× faster than cv::DECOMP_NORMAL on 6-point minimal samples
// because the fixed loop bounds inline cleanly and Cholesky on a 6×6 SPD
// matrix is ~60 flops vs cv::solve's ~5 μs of setup.
//
// Correctness notes:
//   * Weights (w_i) act on the normal equations directly:
//       ATA[j][k] = Σ w_i · u_i^j · u_i^k
//       ATb[j]    = Σ w_i · u_i^j · v_i
//     which is equivalent to weighted least squares with weights w_i.
//   * MAX_M = 7 supports polynomial degrees 0..6 inclusive (lab uses 1..5).
//   * Cholesky fails silently (returns false) on non-positive-definite ATA,
//     e.g. when all u-values collapse.  Caller skips the iteration.
constexpr int POLY_MAX_M = 7;

bool fit_poly_fast(const double* us, const double* vs, const double* ws,
                   int n, int degree, double* coeffs_out)
{
    if (degree < 0 || degree + 1 > POLY_MAX_M) return false;
    const int m = degree + 1;
    if (n < m) return false;

    double ATA[POLY_MAX_M][POLY_MAX_M] = {};
    double ATb[POLY_MAX_M] = {};

    // Accumulate upper triangle of A^T W A and A^T W b.
    for (int i = 0; i < n; ++i) {
        const double u = us[i];
        const double v = vs[i];
        const double w = ws ? ws[i] : 1.0;
        double pow_[POLY_MAX_M];
        pow_[0] = 1.0;
        for (int k = 1; k < m; ++k) pow_[k] = pow_[k - 1] * u;
        for (int j = 0; j < m; ++j) {
            const double wpj = w * pow_[j];
            ATb[j] += wpj * v;
            for (int k = j; k < m; ++k) ATA[j][k] += wpj * pow_[k];
        }
    }
    // Mirror upper → lower for Cholesky readability below.
    for (int j = 1; j < m; ++j)
        for (int k = 0; k < j; ++k)
            ATA[j][k] = ATA[k][j];

    // Cholesky L · L^T = ATA, in-place into L (separate matrix for clarity).
    double L[POLY_MAX_M][POLY_MAX_M] = {};
    for (int j = 0; j < m; ++j) {
        double s = ATA[j][j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
        if (s <= 1e-18) return false;
        L[j][j] = std::sqrt(s);
        const double inv_djj = 1.0 / L[j][j];
        for (int i = j + 1; i < m; ++i) {
            double s2 = ATA[i][j];
            for (int k = 0; k < j; ++k) s2 -= L[i][k] * L[j][k];
            L[i][j] = s2 * inv_djj;
        }
    }
    // Solve L y = ATb (forward substitution).
    double y[POLY_MAX_M];
    for (int j = 0; j < m; ++j) {
        double s = ATb[j];
        for (int k = 0; k < j; ++k) s -= L[j][k] * y[k];
        y[j] = s / L[j][j];
    }
    // Solve L^T x = y (back substitution).
    for (int j = m - 1; j >= 0; --j) {
        double s = y[j];
        for (int k = j + 1; k < m; ++k) s -= L[k][j] * coeffs_out[k];
        coeffs_out[j] = s / L[j][j];
    }
    return true;
}

// Thin std::vector wrapper matching the original signature.
bool fit_poly(const std::vector<double>& us,
              const std::vector<double>& vs,
              int degree,
              std::vector<double>& coeffs_out,
              const std::vector<double>& w = {})
{
    const int n = (int)us.size();
    const int m = degree + 1;
    if (n < m || m > POLY_MAX_M) return false;
    coeffs_out.assign(m, 0.0);
    const double* wp = ((int)w.size() == n) ? w.data() : nullptr;
    return fit_poly_fast(us.data(), vs.data(), wp, n, degree, coeffs_out.data());
}

// Compute an approximate quantile by nth_element (mutates input).
double quantile_inplace(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    size_t k = (size_t)std::clamp((double)v.size() * q, 0.0, (double)(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// Extract up to TOP_K sub-pixel-refined gradient peaks from a single
// vertical caliper, storing signed gradient along with strength.
void sample_caliper(const cv::Mat& gray,
                    int caliper_id,
                    int cx, double y0, int span, int half_w,
                    double min_edge,
                    std::vector<Hit>& out)
{
    const int H = gray.rows;
    const int W = gray.cols;

    int y_lo = std::max(0, (int)std::round(y0 - span * 0.5));
    int y_hi = std::min(H - 1, (int)std::round(y0 + span * 0.5));
    int x_lo = std::max(0, cx - half_w);
    int x_hi = std::min(W - 1, cx + half_w);
    int n    = y_hi - y_lo + 1;
    int cols = x_hi - x_lo + 1;
    if (n < 5 || cols < 1) return;

    std::vector<double> prof(n, 0.0);
    for (int i = 0; i < n; ++i) {
        int y = y_lo + i;
        double acc = 0.0;
        const uint8_t* row = gray.ptr<uint8_t>(y);
        for (int x = x_lo; x <= x_hi; ++x) acc += row[x];
        prof[i] = acc / (double)cols;
    }

    // Smooth + differentiate with a 1-D Gaussian derivative (σ=1.0, 5-tap),
    // normalised so the response to a unit ramp is 1.0. Compared to the
    // 3-tap central difference this averages over 5 px instead of 2 and
    // attenuates high-frequency noise, which sharpens the parabolic
    // sub-pixel fit later on.
    static constexpr double GD[5] = { -0.1179, -0.2641, 0.0, 0.2641, 0.1179 };
    std::vector<double> grad(n, 0.0);  // signed
    for (int i = 2; i < n - 2; ++i) {
        grad[i] = GD[0] * prof[i - 2] + GD[1] * prof[i - 1]
                + GD[3] * prof[i + 1] + GD[4] * prof[i + 2];
    }
    // Boundary fallback: narrower kernel near the caliper ends.
    if (n >= 3) {
        grad[1]     = 0.5 * (prof[2]     - prof[0]);
        grad[n - 2] = 0.5 * (prof[n - 1] - prof[n - 3]);
    }
    grad[0]     = prof[1]     - prof[0];
    grad[n - 1] = prof[n - 1] - prof[n - 2];

    std::vector<std::pair<int, double>> peaks;  // (i, |g|)
    for (int i = 1; i < n - 1; ++i) {
        double ag = std::abs(grad[i]);
        if (ag < min_edge) continue;
        if (ag > std::abs(grad[i - 1]) && ag > std::abs(grad[i + 1])) {
            peaks.emplace_back(i, ag);
        }
    }
    if (peaks.empty()) return;

    std::sort(peaks.begin(), peaks.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });

    std::vector<std::pair<int, double>> kept;
    for (const auto& p : peaks) {
        bool ok = true;
        for (const auto& k : kept) {
            if (std::abs(p.first - k.first) < MIN_SEP) { ok = false; break; }
        }
        if (ok) kept.push_back(p);
        if ((int)kept.size() >= TOP_K) break;
    }

    for (const auto& k : kept) {
        int i = k.first;
        double y_sub = (double)(y_lo + i);
        if (i > 0 && i < n - 1) {
            double gm = std::abs(grad[i - 1]);
            double g0 = std::abs(grad[i]);
            double gp = std::abs(grad[i + 1]);
            double denom = (gm - 2.0 * g0 + gp);
            if (std::abs(denom) > 1e-12) {
                double off = 0.5 * (gm - gp) / denom;
                if (off > -1.0 && off < 1.0) y_sub += off;
            }
        }
        int sg = (grad[i] >= 0) ? +1 : -1;
        out.push_back({(double)cx, y_sub, k.second, sg, caliper_id});
    }
}

// Estimate a data-driven edge-strength threshold from the image's
// |∂I/∂y| distribution across a mid-band of rows.
double estimate_edge_threshold(const cv::Mat& gray, double y0, int band) {
    const int H = gray.rows, W = gray.cols;
    int y_lo = std::max(1, (int)std::round(y0 - band * 0.5));
    int y_hi = std::min(H - 2, (int)std::round(y0 + band * 0.5));
    if (y_hi <= y_lo) return MIN_EDGE_FLOOR;

    std::vector<double> mags;
    mags.reserve((y_hi - y_lo + 1) * (W / 2) / 2 + 16);
    for (int y = y_lo; y <= y_hi; y += 2) {
        const uint8_t* rm = gray.ptr<uint8_t>(y - 1);
        const uint8_t* rp = gray.ptr<uint8_t>(y + 1);
        for (int x = 0; x < W; x += 2) {
            double g = 0.5 * ((double)rp[x] - (double)rm[x]);
            mags.push_back(std::abs(g));
        }
    }
    if (mags.empty()) return MIN_EDGE_FLOOR;
    double q = quantile_inplace(mags, STRENGTH_QUANTILE);
    return std::max(MIN_EDGE_FLOOR, q);
}

} // anonymous namespace

std::vector<cv::Point2d> detect_caliper_ransac(const cv::Mat& gray,
                                               const GroundTruth& gt)
{
    const int W = gray.cols;
    const double y_mid = gt.y0;
    const double half_w_img = 0.5 * W;

    // ---- 0. Adaptive edge-strength threshold ------------------------------
    const double min_edge = estimate_edge_threshold(gray, y_mid, CALIPER_SPAN);

    // ---- 1. Build caliper x-positions (uniform core + edge densification) -
    std::vector<int> cxs;
    cxs.reserve(N_CALIPERS_CORE + 2 * N_CALIPERS_EDGE);
    for (int i = 0; i < N_CALIPERS_CORE; ++i) {
        int cx = (int)std::round((i + 0.5) * (double)W / N_CALIPERS_CORE);
        cxs.push_back(std::clamp(cx, 0, W - 1));
    }
    const double edge_band = std::max(4.0, EDGE_FRAC * W);
    for (int i = 0; i < N_CALIPERS_EDGE; ++i) {
        double t = (i + 0.5) / N_CALIPERS_EDGE;
        int cxL = (int)std::round(t * edge_band);
        int cxR = (int)std::round((W - 1) - t * edge_band);
        cxs.push_back(std::clamp(cxL, 0, W - 1));
        cxs.push_back(std::clamp(cxR, 0, W - 1));
    }
    std::sort(cxs.begin(), cxs.end());
    cxs.erase(std::unique(cxs.begin(), cxs.end()), cxs.end());
    const int N_CAL = (int)cxs.size();

    std::vector<Hit> hits_all;
    hits_all.reserve(N_CAL * TOP_K);
    for (int i = 0; i < N_CAL; ++i) {
        sample_caliper(gray, i, cxs[i], y_mid, CALIPER_SPAN, HALF_W, min_edge, hits_all);
    }

    if ((int)hits_all.size() < DEG_MIN + 1) {
        std::vector<cv::Point2d> fallback;
        fallback.reserve(W);
        for (int x = 0; x < W; ++x) fallback.emplace_back((double)x, y_mid);
        return fallback;
    }

    // ---- 1b. Polarity hypothesis selection.
    //        Compute unique-caliper coverage per sign. If one sign
    //        dominates (minority < BIMODAL_RATIO × majority), keep only
    //        that sign. Otherwise run RANSAC on *both* polarities and
    //        keep the one with the higher MDL-penalised score. This
    //        protects the curve from losing to dense spike-bottom edges
    //        in scenes where the curve is weakly lit: the spike-bottom
    //        hypothesis may win on caliper coverage but will lose on
    //        the coverage-weighted inlier score because spike bottoms do
    //        not form a smooth polynomial.
    //
    //        Threshold 0.7 = truly ambiguous only. Most real scenes
    //        with a spike field still produce a lopsided coverage
    //        (curve spans every caliper; spike bottoms only some), so
    //        the dual-hypothesis path rarely triggers and the runtime
    //        tax is minimal.
    constexpr double BIMODAL_RATIO = 0.7;
    int pc = 0, nc = 0;
    {
        std::vector<uint8_t> pos_cal(N_CAL, 0), neg_cal(N_CAL, 0);
        for (const auto& h : hits_all) {
            if (h.caliper_id < 0 || h.caliper_id >= N_CAL) continue;
            if (h.sign > 0) pos_cal[h.caliper_id] = 1;
            else            neg_cal[h.caliper_id] = 1;
        }
        for (int i = 0; i < N_CAL; ++i) { pc += pos_cal[i]; nc += neg_cal[i]; }
    }
    std::vector<int> sign_list;
    const int maxc = std::max(pc, nc);
    const int minc = std::min(pc, nc);
    if (maxc > 0 && (double)minc >= BIMODAL_RATIO * (double)maxc) {
        sign_list = { +1, -1 };
    } else {
        sign_list = { (pc >= nc) ? +1 : -1 };
    }

    // Winner across the one or two polarity hypotheses.
    double              winner_score       = -1e30;
    std::vector<double> winner_coeffs;
    std::vector<int>    winner_inlier_idx;
    int                 winner_degree      = DEG_MIN;
    std::vector<Hit>    winner_hits;

    for (int dsign : sign_list) {
        std::vector<Hit> hits;
        hits.reserve(hits_all.size());
        for (const auto& h : hits_all) if (h.sign == dsign) hits.push_back(h);
        if ((int)hits.size() < DEG_MIN + 1) continue;

        // Sort hits by strength descending for PROSAC. The ranking prior
        // is |gradient|: real-edge peaks tend to have higher, more
        // consistent strength than peak-by-chance spikes, so minimal
        // samples drawn from the strongest hits converge much faster
        // than uniform sampling.
        std::sort(hits.begin(), hits.end(),
            [](const Hit& a, const Hit& b){ return a.strength > b.strength; });

        // Pre-transform to (u, v) space once.
        const int N = (int)hits.size();
        std::vector<double> us(N), vs(N), ws(N);
        for (int i = 0; i < N; ++i) {
            us[i] = (hits[i].x - half_w_img) / half_w_img;
            vs[i] = hits[i].y - y_mid;
            ws[i] = hits[i].strength;
        }

        // ---- 2. Adaptive-degree PROSAC (Progressive Sample Consensus).
        //        Iteration t draws from the top-M(t) strongest hits,
        //        where M grows from `need` to N over the iteration
        //        budget. Early iterations try strong-only hypotheses
        //        (which usually find consensus fast on the real edge);
        //        later iterations widen the pool as a fallback.
        std::mt19937 rng(12345u);

        double              best_score      = -1e30;
        std::vector<double> best_coeffs;
        std::vector<int>    best_inlier_idx;
        int                 best_degree     = DEG_MIN;

        const double u_slope_cap = half_w_img;  // tan(45°) in pixel space

        for (int degree = DEG_MIN; degree <= DEG_MAX; ++degree) {
        const int need = degree + 1;
        if (N < need) continue;

        double deg_best_score = -1e30;
        std::vector<double> deg_best_coeffs;
        std::vector<int>    deg_best_idx;

        // More iters for higher degrees (larger minimal-sample space).
        const int ransac_iters = RANSAC_ITERS_BASE + 10 * (degree - DEG_MIN);
        for (int iter = 0; iter < ransac_iters; ++iter) {
            // PROSAC window: grow linearly from `need` to N across the
            // iteration budget. A small constant floor keeps the window
            // non-degenerate for degree 1.
            int M = need + (int)((double)(N - need) * (double)iter /
                                  std::max(1, ransac_iters - 1));
            M = std::clamp(M, need + 2, N);
            std::uniform_int_distribution<int> pick(0, M - 1);

            int idx[8];  // supports up to degree 7
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
            // Require x-spread in the minimal sample: if all picks share
            // < min_u_spread in u, the fit is ill-conditioned. Reject.
            double smin = 1e30, smax = -1e30;
            for (int k = 0; k < need; ++k) { smin = std::min(smin, su[k]); smax = std::max(smax, su[k]); }
            if ((smax - smin) < 0.4) continue;

            std::vector<double> coeffs;
            if (!fit_poly(su, sv, degree, coeffs)) continue;

            // Slope sanity check.
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

            // Count inliers AND collect unique-caliper coverage.
            int cnt = 0;
            std::vector<int> in_idx;
            in_idx.reserve(N);
            std::vector<uint8_t> cal_hit(N_CAL, 0);
            for (int j = 0; j < N; ++j) {
                double r = vs[j] - poly_eval(coeffs, us[j]);
                // Score with the tight fit-precision threshold, not the
                // user tolerance — RANSAC's job is to find the *tightest*
                // consensus, regardless of how lenient the caller is.
                if (std::abs(r) <= FIT_THR) {
                    ++cnt;
                    in_idx.push_back(j);
                    int cid = hits[j].caliper_id;
                    if (cid >= 0 && cid < N_CAL) cal_hit[cid] = 1;
                }
            }
            if (cnt < need) continue;
            int cal_covered = 0;
            for (int i = 0; i < N_CAL; ++i) cal_covered += cal_hit[i];

            // Score: inliers scaled by caliper-coverage fraction. A fit
            // that only explains a narrow x-band (spikes piled in one
            // place) has low coverage → low score, so RANSAC prefers
            // fits supported by calipers across the full width.
            double cov_frac = (double)cal_covered / std::max(1, N_CAL);
            double score = (double)cnt * cov_frac;
            if (score > deg_best_score) {
                deg_best_score  = score;
                deg_best_coeffs = coeffs;
                deg_best_idx    = std::move(in_idx);
            }
        }

        if (deg_best_score <= 0.0) continue;

        // Model-selection score across degrees: reward inliers+coverage,
        // penalise complexity. A cubic needs DEGREE_PENALTY more inliers
        // than a line to win.
        double score = deg_best_score - DEGREE_PENALTY * (degree - DEG_MIN);
        if (score > best_score) {
            best_score      = score;
            best_coeffs     = std::move(deg_best_coeffs);
            best_inlier_idx = std::move(deg_best_idx);
            best_degree     = degree;
        }
    }

        if (best_coeffs.empty() || best_inlier_idx.empty()) continue;

        // ---- 2b. Huber-IRLS refinement, 2 rounds.
        //         Gates inliers at FIT_THR (the internal fit-precision
        //         threshold), not the user-facing INLIER_THR. This keeps
        //         the fit locked onto the tight sub-pixel inlier cluster
        //         even when the caller has set a loose tolerance.
        //         Huber c = 0.7 px inside FIT_THR pulls the fit toward
        //         the tightest-residual subset.
        //         No unweighted LSQ pre-step: that pre-step got biased
        //         when thresholds loosened. Compute weights from the
        //         current fit's residuals, then run weighted LSQ once.
        {
            const int need = best_degree + 1;
            constexpr double C_HUBER = 0.7;  // px — fit-precision scale
            for (int outer = 0; outer < 2; ++outer) {
                std::vector<int> new_idx;
                new_idx.reserve(N);
                for (int j = 0; j < N; ++j) {
                    double r = vs[j] - poly_eval(best_coeffs, us[j]);
                    if (std::abs(r) <= FIT_THR) new_idx.push_back(j);
                }
                if ((int)new_idx.size() < need + 1) break;
                best_inlier_idx = std::move(new_idx);

                std::vector<double> su, sv, wv;
                su.reserve(best_inlier_idx.size());
                sv.reserve(best_inlier_idx.size());
                wv.reserve(best_inlier_idx.size());
                for (int i : best_inlier_idx) {
                    su.push_back(us[i]);
                    sv.push_back(vs[i]);
                    double r  = vs[i] - poly_eval(best_coeffs, us[i]);
                    double ar = std::abs(r);
                    wv.push_back(ar <= C_HUBER ? 1.0 : C_HUBER / std::max(ar, 1e-9));
                }
                std::vector<double> refined;
                if (fit_poly(su, sv, best_degree, refined, wv)) {
                    best_coeffs = refined;
                }
            }
        }

        if (best_score > winner_score) {
            winner_score      = best_score;
            winner_coeffs     = std::move(best_coeffs);
            winner_inlier_idx = std::move(best_inlier_idx);
            winner_degree     = best_degree;
            winner_hits       = std::move(hits);
        }
    }  // end for (int dsign : sign_list)

    if (winner_coeffs.empty() || winner_inlier_idx.empty()) {
        std::vector<cv::Point2d> fallback;
        fallback.reserve(W);
        for (int x = 0; x < W; ++x) fallback.emplace_back((double)x, y_mid);
        return fallback;
    }

    // ---- 2c. Per-caliper dedup: if a caliper contributed multiple
    //         inliers (e.g. a real edge and a nearby noise peak both
    //         within FIT_THR of the converged fit), keep only the
    //         one whose sub-pixel y is closest to the fit. This
    //         prevents densely-featured calipers from over-weighting
    //         the final LSQ and makes the "1 inlier = 1 caliper
    //         contribution" interpretation honest.
    {
        const double half_w_img_l = 0.5 * W;
        auto eval_v = [&](double x){
            double u = (x - half_w_img_l) / half_w_img_l;
            return poly_eval(winner_coeffs, u);
        };
        std::vector<int>    cal_best_idx(N_CAL, -1);
        std::vector<double> cal_best_res(N_CAL, std::numeric_limits<double>::infinity());
        for (int i : winner_inlier_idx) {
            const Hit& h = winner_hits[i];
            int cid = h.caliper_id;
            if (cid < 0 || cid >= N_CAL) continue;
            double r = std::abs((h.y - y_mid) - eval_v(h.x));
            if (r < cal_best_res[cid]) {
                cal_best_res[cid] = r;
                cal_best_idx[cid] = i;
            }
        }
        std::vector<int> dedup;
        dedup.reserve(N_CAL);
        for (int c = 0; c < N_CAL; ++c)
            if (cal_best_idx[c] >= 0) dedup.push_back(cal_best_idx[c]);

        // Only swap in the deduped set if it still has enough points
        // to re-fit (need + 1 with a small cushion) AND it actually
        // removed duplicates.
        if ((int)dedup.size() >= winner_degree + 2 &&
            dedup.size() < winner_inlier_idx.size()) {
            winner_inlier_idx = std::move(dedup);
            std::vector<double> su, sv;
            su.reserve(winner_inlier_idx.size());
            sv.reserve(winner_inlier_idx.size());
            for (int i : winner_inlier_idx) {
                const Hit& h = winner_hits[i];
                su.push_back((h.x - half_w_img_l) / half_w_img_l);
                sv.push_back(h.y - y_mid);
            }
            std::vector<double> refined;
            if (fit_poly(su, sv, winner_degree, refined)) winner_coeffs = refined;
        }
    }

    // ---- 3. Sample. Inside [x_min_inlier, x_max_inlier] evaluate the
    //        polynomial. Outside, use linear extrapolation anchored at
    //        the inlier boundary with the polynomial's derivative at
    //        that boundary — C¹-continuous.
    //
    //        The span uses INLIER_THR (user tolerance), not FIT_THR.
    //        This is where the user's tolerance setting is effective:
    //          * a tighter INLIER_THR shrinks the polynomial segment
    //            (more of the output is the safe linear tail),
    //          * a looser INLIER_THR extends the polynomial across
    //            hits the caller deems acceptable even if they sit
    //            slightly outside the tight fit band.
    //        The fit quality itself stays locked to FIT_THR — only
    //        the evaluation range responds to the user's setting.
    auto poly_v_at_x = [&](double x) {
        double u = (x - half_w_img) / half_w_img;
        return poly_eval(winner_coeffs, u);
    };
    double x_min_in =  1e30, x_max_in = -1e30;
    for (const auto& h : winner_hits) {
        double r = std::abs((h.y - y_mid) - poly_v_at_x(h.x));
        if (r <= INLIER_THR) {
            if (h.x < x_min_in) x_min_in = h.x;
            if (h.x > x_max_in) x_max_in = h.x;
        }
    }
    if (!(x_min_in < x_max_in)) { x_min_in = 0; x_max_in = W - 1; }

    auto eval_at_x = [&](double x) {
        double u = (x - half_w_img) / half_w_img;
        return poly_eval(winner_coeffs, u) + y_mid;
    };
    auto dy_dx_at_x = [&](double x) {
        // dy/dx = (1/half_w_img) * d(poly)/du at u.
        double u = (x - half_w_img) / half_w_img;
        double dv_du = 0.0;
        for (int k = (int)winner_coeffs.size() - 1; k >= 1; --k) {
            dv_du = dv_du * u + k * winner_coeffs[k];
        }
        return dv_du / half_w_img;
    };
    const double y_lo_edge = eval_at_x(x_min_in);
    const double y_hi_edge = eval_at_x(x_max_in);
    const double m_lo      = dy_dx_at_x(x_min_in);
    const double m_hi      = dy_dx_at_x(x_max_in);

    std::vector<cv::Point2d> out;
    out.reserve(W);
    for (int x = 0; x < W; ++x) {
        double y;
        double xd = (double)x;
        if (xd < x_min_in)      y = y_lo_edge + m_lo * (xd - x_min_in);
        else if (xd > x_max_in) y = y_hi_edge + m_hi * (xd - x_max_in);
        else                    y = eval_at_x(xd);
        out.emplace_back(xd, y);
    }
    return out;
}

} // namespace lab
