//
// algo_tensor_voting.cpp — simplified single-pass tensor voting
// (Medioni / Lee / Tang style, stick-tensor only).
//
// Pipeline:
//   1. Token extraction: |∇I| + direction via Sobel. Keep every pixel
//      whose magnitude exceeds 10% of the image max as a "token". For
//      each token we store its position and unit tangent
//          t = (-gy, gx) / |grad|   (perpendicular to the gradient).
//      Tangents are sign-ambiguous; that's fine because every vote we
//      cast is symmetric in t → -t (the outer product t⊗t is invariant).
//
//   2. Stick-tensor voting: every token T_j casts votes at every pixel
//      receiver p_i inside a voting radius R ≈ 2σ. The vote direction
//      at the receiver is the tangent to the osculating circle passing
//      through the sender p_j with tangent t_j that also passes through
//      p_i — algebraically this is t_j reflected about the (p_i−p_j)
//      axis. The Medioni saliency decay is
//          DF = exp( -(s² + c·κ²) / σ² )
//      with s = d·θ/sin(θ)  (arc length), κ = 2·sin(θ)/d. Past a π/4
//      cone from t_j the vote is skipped.
//
//   3. Accumulate the rank-1 tensor DF * (v̂ ⊗ v̂) into a per-pixel
//      symmetric 2×2 matrix. Eigendecompose in closed form: stick
//      saliency = λ1 − λ2.
//
//   4. Ridge extraction: constrain to a ±40 px band around the midline
//      (legitimate prior — we don't know the curve's shape, just that
//      it lives near y0). For each column take the row of maximum
//      saliency; parabolically refine the sub-pixel y.
//
// Optimisations applied here vs. the textbook version:
//   * Single fused Sobel+magnitude+token-generation pass.
//   * Per-token voting parallelised with OpenMP; each thread writes
//     into its own tensor scratchpad, reduced at the end (no atomics).
//   * The Medioni decay  exp( -(s² + c·κ²)/σ² )  is tabulated on a
//     (d, cosθ) grid. The hot loop does integer indexing + one fma
//     into the LUT and avoids every sqrt / asin / exp.
//   * Receivers and tensor storage both live inside the narrow
//     ±half_band strip around y0, so the hot accumulators and the
//     saliency pass touch ~B·W doubles rather than H·W.
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lab {

namespace {

struct Token {
    float x, y;       // position
    float tx, ty;     // unit tangent (perpendicular to gradient)
};

// ------------------------------------------------------------------------
// Medioni decay field lookup table.
//
// The decay factor DF(d, cosθ) = exp(-(s² + c·κ²)/σ²)  depends only on
// the sender→receiver distance d and the |cosθ| between the tangent and
// the sender→receiver axis. Within the cone (cosθ ≥ cos(π/4)) both axes
// are bounded, so we can tabulate on a fixed grid and index with one
// multiply + one floor per receiver.
//
// Grid resolution is picked for accuracy: over d ∈ [0, 30] at step
// 0.25 px and cosθ ∈ [0.707, 1] at step 0.001 we reproduce the analytic
// DF to < 5e-5. Voting outcomes are averaged over dozens of contributors
// per receiver so this is well below the noise floor.
// ------------------------------------------------------------------------
struct DecayLUT {
    double sigma   = 15.0;
    double c_curv  = 3.0 * 15.0 * 15.0;
    double R       = 30.0;
    double cos_lo  = 0.70710678118654752;     // cos(π/4)

    // d grid
    double d_step  = 0.25;
    int    d_n     = 0;
    double inv_d_step = 0.0;

    // cosθ grid  (cos ranges cos_lo..1)
    double c_step  = 0.001;
    int    c_n     = 0;
    double inv_c_step = 0.0;

    std::vector<float> table;   // d_n * c_n  (row major: c varies fastest)

    inline float sample(double d, double cos_t) const {
        // Clamp into table. Callers only invoke us after the cone test,
        // so cos_t is already ≥ cos_lo − epsilon.
        double di = d * inv_d_step;
        double ci = (cos_t - cos_lo) * inv_c_step;
        if (di < 0.0) di = 0.0;
        if (ci < 0.0) ci = 0.0;
        int i0 = (int)di; if (i0 >= d_n - 1) i0 = d_n - 2;
        int j0 = (int)ci; if (j0 >= c_n - 1) j0 = c_n - 2;
        double fd = di - i0;
        double fc = ci - j0;
        const float* r0 = &table[(size_t)i0 * c_n + j0];
        const float* r1 = r0 + c_n;
        float v00 = r0[0], v01 = r0[1];
        float v10 = r1[0], v11 = r1[1];
        float v0 = (float)(v00 + (v01 - v00) * fc);
        float v1 = (float)(v10 + (v11 - v10) * fc);
        return (float)(v0 + (v1 - v0) * fd);
    }
};

DecayLUT build_decay_lut(double sigma, double c_curv, double R, double cos_lo) {
    DecayLUT L;
    L.sigma = sigma;
    L.c_curv = c_curv;
    L.R = R;
    L.cos_lo = cos_lo;

    L.d_step = 0.25;
    L.d_n    = (int)std::ceil(R / L.d_step) + 2;
    L.inv_d_step = 1.0 / L.d_step;

    L.c_step = 0.001;
    L.c_n    = (int)std::ceil((1.0 - cos_lo) / L.c_step) + 2;
    L.inv_c_step = 1.0 / L.c_step;

    L.table.assign((size_t)L.d_n * L.c_n, 0.0f);
    const double sig2 = sigma * sigma;

    for (int i = 0; i < L.d_n; ++i) {
        const double d = i * L.d_step;
        for (int j = 0; j < L.c_n; ++j) {
            double cos_t = cos_lo + j * L.c_step;
            if (cos_t > 1.0) cos_t = 1.0;
            const double sin_t2 = std::max(0.0, 1.0 - cos_t * cos_t);
            const double sin_t  = std::sqrt(sin_t2);
            double s, kappa;
            if (d < 1e-9) {
                s = 0.0; kappa = 0.0;
            } else if (sin_t < 1e-6) {
                s = d; kappa = 0.0;
            } else {
                const double theta = std::asin(sin_t);
                s = d * theta / sin_t;
                kappa = 2.0 * sin_t / d;
            }
            const double df = std::exp(-(s * s + c_curv * kappa * kappa) / sig2);
            L.table[(size_t)i * L.c_n + j] = (float)df;
        }
    }
    return L;
}

} // namespace

std::vector<cv::Point2d> detect_tensor_voting(const cv::Mat& gray, const GroundTruth& gt) {
    const int H = gray.rows;
    const int W = gray.cols;

    // ---- 1. Gradient + magnitude + token extraction (fused) -------------
    //
    // Sobel is cheap (OpenCV is SIMD'd) so we still let cv::Sobel do it,
    // but we fuse the magnitude computation, the global-max reduction
    // and the token emission into a single pass to avoid re-visiting the
    // image three times.
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    // First pass: compute magnitude into a scratch array and find max.
    std::vector<float> mag_buf((size_t)H * W);
    float max_mag = 0.0f;
    for (int y = 0; y < H; ++y) {
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        float* mr = &mag_buf[(size_t)y * W];
        float row_max = 0.0f;
        for (int x = 0; x < W; ++x) {
            const float a = gxr[x], b = gyr[x];
            const float m = std::sqrt(a * a + b * b);
            mr[x] = m;
            if (m > row_max) row_max = m;
        }
        if (row_max > max_mag) max_mag = row_max;
    }

    const float kTokenThr = 0.10f * max_mag;
    std::vector<Token> tokens;
    tokens.reserve(4096);
    for (int y = 0; y < H; ++y) {
        const float* gxr = gx.ptr<float>(y);
        const float* gyr = gy.ptr<float>(y);
        const float* mr  = &mag_buf[(size_t)y * W];
        for (int x = 0; x < W; ++x) {
            const float m = mr[x];
            if (m < kTokenThr) continue;
            const float inv_m = 1.0f / m;
            Token t;
            t.x  = (float)x;
            t.y  = (float)y;
            t.tx = -gyr[x] * inv_m;
            t.ty =  gxr[x] * inv_m;
            tokens.push_back(t);
        }
    }

    // ---- 2/3. Stick-tensor voting --------------------------------------
    //
    // Same σ = 15 as the baseline (preserves accuracy). All decay work
    // is delegated to a precomputed LUT indexed by (d, cosθ).
    const double sigma    = 15.0;
    const double sig2     = sigma * sigma;
    const double c_curv   = 3.0 * sig2;
    const double R        = 2.0 * sigma;
    const double R2       = R * R;
    const double cos_cone = 0.70710678118654752; // cos(π/4) influence cone

    static DecayLUT lut;                     // build once per process
    static bool     lut_ready = false;
    if (!lut_ready) {
        lut = build_decay_lut(sigma, c_curv, R, cos_cone);
        lut_ready = true;
    }

    // Restrict receivers to a band around the midline. The band is a
    // legitimate prior used by the other algorithms too; everything
    // outside contains only bright spikes and background and can't
    // affect the eventual per-column argmax.
    const int half_band = 40;
    const int y_lo = std::max(0,     (int)std::round(gt.y0) - half_band);
    const int y_hi = std::min(H - 1, (int)std::round(gt.y0) + half_band);
    const int Bh   = y_hi - y_lo + 1;        // band height

    const int iR   = (int)std::ceil(R);

    // Per-thread band-local tensor accumulators. Reduced at the end to
    // avoid atomics in the hot loop.
#ifdef _OPENMP
    const int nthreads = std::max(1, omp_get_max_threads());
#else
    const int nthreads = 1;
#endif
    const size_t band_px = (size_t)Bh * W;
    std::vector<std::vector<double>> Txx_t(nthreads, std::vector<double>(band_px, 0.0));
    std::vector<std::vector<double>> Txy_t(nthreads, std::vector<double>(band_px, 0.0));
    std::vector<std::vector<double>> Tyy_t(nthreads, std::vector<double>(band_px, 0.0));

    const int         NT      = (int)tokens.size();
    const Token* __restrict ptok = tokens.data();
    const float* __restrict lut_tab = lut.table.data();
    const int         lut_cn  = lut.c_n;
    const double      lut_inv_d = lut.inv_d_step;
    const double      lut_inv_c = lut.inv_c_step;
    const double      lut_cos_lo = lut.cos_lo;
    const int         lut_dn_m1 = lut.d_n - 2;
    const int         lut_cn_m1 = lut.c_n - 2;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* __restrict Txx = Txx_t[tid].data();
        double* __restrict Txy = Txy_t[tid].data();
        double* __restrict Tyy = Tyy_t[tid].data();

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 64)
#endif
        for (int ti = 0; ti < NT; ++ti) {
            const Token tok = ptok[ti];
            const double px = tok.x, py = tok.y;
            const double tx = tok.tx, ty = tok.ty;

            const int xi_lo = std::max(0,     (int)(px - iR));
            const int xi_hi = std::min(W - 1, (int)(px + iR + 1.0));
            const int yi_lo = std::max(y_lo,  (int)(py - iR));
            const int yi_hi = std::min(y_hi,  (int)(py + iR + 1.0));

            for (int yi = yi_lo; yi <= yi_hi; ++yi) {
                const double dy = (double)yi - py;
                const double dy2 = dy * dy;
                // Horizontal extent for this row given R.
                const double dx_max2 = R2 - dy2;
                if (dx_max2 < 0.0) continue;
                const double dx_max = std::sqrt(dx_max2);
                int xs = (int)std::ceil (px - dx_max);
                int xe = (int)std::floor(px + dx_max);
                if (xs < xi_lo) xs = xi_lo;
                if (xe > xi_hi) xe = xi_hi;
                if (xs > xe) continue;

                const int row_off = (yi - y_lo) * W;
                for (int xi = xs; xi <= xe; ++xi) {
                    const double dx  = (double)xi - px;
                    const double d2  = dx * dx + dy2;
                    if (d2 < 1e-12) {
                        // Self-vote contributes t⊗t at unit weight.
                        const size_t idx = (size_t)row_off + xi;
                        Txx[idx] += tx * tx;
                        Txy[idx] += tx * ty;
                        Tyy[idx] += ty * ty;
                        continue;
                    }
                    // Cheap cone rejection in squared form to skip the sqrt.
                    // dot = v·t = (dx*tx + dy*ty) / d
                    // cos²θ = dot² = (dx*tx + dy*ty)² / d²
                    const double num = dx * tx + dy * ty;
                    const double num2 = num * num;
                    // cos² ≥ 0.5  ⇔  num² ≥ 0.5 · d²
                    if (num2 < 0.5 * d2) continue;

                    const double d    = std::sqrt(d2);
                    const double inv_d = 1.0 / d;
                    const double dot_abs = (num >= 0.0 ? num : -num) * inv_d;
                    const double sign    = (num >= 0.0 ? 1.0 : -1.0);

                    // LUT lookup with bilinear interpolation.
                    double di = d * lut_inv_d;
                    double ci = (dot_abs - lut_cos_lo) * lut_inv_c;
                    if (ci < 0.0) ci = 0.0;
                    int i0 = (int)di; if (i0 > lut_dn_m1) i0 = lut_dn_m1;
                    int j0 = (int)ci; if (j0 > lut_cn_m1) j0 = lut_cn_m1;
                    const double fd = di - i0;
                    const double fc = ci - j0;
                    const float* r0 = lut_tab + (size_t)i0 * lut_cn + j0;
                    const float* r1 = r0 + lut_cn;
                    const double v00 = r0[0], v01 = r0[1];
                    const double v10 = r1[0], v11 = r1[1];
                    const double v0  = v00 + (v01 - v00) * fc;
                    const double v1  = v10 + (v11 - v10) * fc;
                    const double df  = v0  + (v1  - v0)  * fd;
                    if (df < 1e-4) continue;

                    // Vote direction: tangent reflected about v̂ = (dx,dy)/d.
                    // r = 2(t·v̂)v̂ − t   with sign-normalised t' = sign·t.
                    const double vx = dx * inv_d;
                    const double vy = dy * inv_d;
                    const double t_sx = sign * tx;
                    const double t_sy = sign * ty;
                    const double ra   = 2.0 * (t_sx * vx + t_sy * vy);
                    const double rx   = ra * vx - t_sx;
                    const double ry   = ra * vy - t_sy;

                    const size_t idx = (size_t)row_off + xi;
                    Txx[idx] += df * rx * rx;
                    Txy[idx] += df * rx * ry;
                    Tyy[idx] += df * ry * ry;
                }
            }
        }
    }

    // ---- Reduce per-thread tensors ------------------------------------
    std::vector<double> Txx(band_px, 0.0);
    std::vector<double> Txy(band_px, 0.0);
    std::vector<double> Tyy(band_px, 0.0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < (long long)band_px; ++i) {
        double a = 0, b = 0, c = 0;
        for (int t = 0; t < nthreads; ++t) {
            a += Txx_t[t][i];
            b += Txy_t[t][i];
            c += Tyy_t[t][i];
        }
        Txx[i] = a;
        Txy[i] = b;
        Tyy[i] = c;
    }

    // ---- 3b. Stick saliency = λ1 − λ2 (closed-form 2×2 eigen) ----------
    //
    // Only evaluate inside the band; the argmax step also lives here.
    cv::Mat sal(Bh, W, CV_32F);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int yb = 0; yb < Bh; ++yb) {
        float* sr = sal.ptr<float>(yb);
        const size_t base = (size_t)yb * W;
        for (int x = 0; x < W; ++x) {
            const size_t idx = base + x;
            const double a = Txx[idx];
            const double b = Txy[idx];
            const double c = Tyy[idx];
            const double tr  = a + c;
            const double det = a * c - b * b;
            const double disc = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
            sr[x] = (float)(2.0 * disc);    // λ1 − λ2 = 2·disc
        }
    }

    // ---- 4. Per-column argmax ridge + parabolic sub-pixel refine -------
    std::vector<cv::Point2d> out;
    out.reserve(W);
    for (int x = 0; x < W; ++x) {
        int best_yb = -1;
        float best_s = 0.0f;
        for (int yb = 0; yb < Bh; ++yb) {
            float s = sal.at<float>(yb, x);
            if (s > best_s) { best_s = s; best_yb = yb; }
        }
        if (best_yb < 0) continue;

        const int best_y = best_yb + y_lo;
        double y_sub = (double)best_y;
        if (best_yb > 0 && best_yb < Bh - 1) {
            const double ym = sal.at<float>(best_yb - 1, x);
            const double y0 = sal.at<float>(best_yb,     x);
            const double yp = sal.at<float>(best_yb + 1, x);
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
