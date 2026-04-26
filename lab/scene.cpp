//
// scene.cpp — fixed + randomised test scenes for the weak-edge benchmark.
//
// make_weak_curve_with_spikes(gt) produces the original deterministic
// scene. make_random_scene(seed) samples a fresh scene: random curve
// shape / amplitude / contrast / background, random spike count /
// placement, Gaussian + salt&pepper noise. Same seed → same scene.
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstring>
#include <random>

namespace lab {

namespace {
constexpr double kPi = 3.14159265358979;

// Draw a set of bright spike blocks above `y_band_top`. Deterministic
// given the seed + count + placement ranges.
void draw_spikes(cv::Mat& img, std::mt19937& rng,
                 int count, int block_min, int block_max,
                 int y_lo, int y_hi, int intensity) {
    const int W = img.cols, H = img.rows;
    std::uniform_int_distribution<int> dx(0, W - block_max);
    std::uniform_int_distribution<int> dy(y_lo, y_hi - block_max);
    std::uniform_int_distribution<int> db(block_min, block_max);
    for (int i = 0; i < count; ++i) {
        int sz = db(rng);
        int sx = dx(rng);
        int sy = dy(rng);
        for (int yy = sy; yy < sy + sz && yy < H; ++yy) {
            uint8_t* row = img.ptr<uint8_t>(yy);
            for (int xx = sx; xx < sx + sz && xx < W; ++xx) {
                row[xx] = (uint8_t)intensity;
            }
        }
    }
}

// Draw horizontal bright stripes that mimic real edges. Stripe lengths
// span up to `max_len_frac` of image width; stripe heights are 1–3 px
// so they produce a sharp horizontal gradient line (looks like an edge
// to a Sobel-y detector). Placement spans the full caliper search band
// around y0, including *below* the curve — where real edges don't
// normally exist — so honest detectors can't rely on a "curve is above
// the midline" heuristic to reject them.
void draw_stripes(cv::Mat& img, std::mt19937& rng,
                  int count, double max_len_frac,
                  int y_lo, int y_hi, int intensity) {
    const int W = img.cols, H = img.rows;
    const int min_len = (int)std::max(4.0, 0.05 * (double)W);
    const int max_len = (int)std::max((double)min_len + 1.0,
                                      max_len_frac * (double)W);
    std::uniform_int_distribution<int> dlen(min_len, max_len);
    std::uniform_int_distribution<int> dh(1, 3);
    std::uniform_int_distribution<int> dy(y_lo, y_hi);
    for (int i = 0; i < count; ++i) {
        int len = dlen(rng);
        int h   = dh(rng);
        int x0  = std::uniform_int_distribution<int>(0, std::max(0, W - len))(rng);
        int y0  = dy(rng);
        for (int yy = y0; yy < y0 + h && yy < H; ++yy) {
            uint8_t* row = img.ptr<uint8_t>(yy);
            for (int xx = x0; xx < x0 + len && xx < W; ++xx) {
                row[xx] = (uint8_t)intensity;
            }
        }
    }
}

// Periodically smear a narrow vertical strip across the expected
// edge location with a 1-D vertical Gaussian blur. This replaces the
// locally-sharp bg_lo→bg_hi transition with a soft gradient inside
// the smeared strips — effectively making the edge "dashed": sharp
// in between smears, soft/ambiguous inside them. Simulates real-world
// edges that suffer from periodic defocus, scan-line artefacts, or
// anti-aliasing from sub-pixel stepping.
void apply_dashed_edge(cv::Mat& img, std::mt19937& rng,
                       double y_center, int y_band) {
    const int W = img.cols, H = img.rows;
    std::uniform_int_distribution<int> di(0, 1 << 20);
    const int period  = 18 + (di(rng) % 18);   // ∈ [18, 35]
    const int dash_w  = 6  + (di(rng) % 7);    // ∈ [6, 12] px
    const int sigma_y = 3  + (di(rng) % 4);    // ∈ [3, 6] px
    const int ksize_y = 2 * sigma_y + 1;       // odd kernel height
    const int y_lo = std::max(0,    (int)std::round(y_center) - y_band);
    const int y_hi = std::min(H - 1,(int)std::round(y_center) + y_band);
    const int roi_h = y_hi - y_lo + 1;
    if (roi_h < ksize_y || dash_w < 1) return;

    const int phase = di(rng) % period;
    for (int x = phase; x + dash_w <= W; x += period) {
        cv::Rect roi(x, y_lo, dash_w, roi_h);
        cv::Mat sub = img(roi);
        cv::GaussianBlur(sub, sub, cv::Size(1, ksize_y), 0, (double)sigma_y);
    }
}

// Apply photometric variation in-place. Combines four effects sampled
// per scene; meant to mimic real-world lighting/shading the bench
// scene generator otherwise lacks (uniform tint, non-uniform
// illumination, vignette, smooth gradient).
//
//   1. Linear additive gradient — direction θ uniform; magnitude grad_amp.
//      I(x,y) += grad_amp * (cos θ·u + sin θ·v) / norm,  u,v ∈ [-1,1].
//   2. Radial vignette — multiplicative darken at corners.
//      I(x,y) *= 1 - vign * (r / r_max)^2.
//   3. Gaussian illumination blob — single bright/dark spot with random
//      centre and σ, sign sampled.
//      I(x,y) += blob_amp * exp(-((x-bx)^2+(y-by)^2) / (2 bs^2)).
//   4. Low-frequency multiplicative gain (grayscale 'tint' analog).
//      I(x,y) *= 1 + gain * (cos φ·u + sin φ·v).
//
// 'strength' = 1.0 → moderate; 2.0 → photo-mode aggressive.
void apply_photometric(cv::Mat& img, std::mt19937& rng, double strength) {
    if (strength <= 0.0) return;
    const int W = img.cols, H = img.rows;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> ang(0.0, 2.0 * kPi);
    std::uniform_real_distribution<double> sgn(-1.0, 1.0);

    const double grad_amp = uni(rng) * 30.0 * strength;
    const double grad_th  = ang(rng);
    const double vign     = uni(rng) * 0.20 * strength;
    const double blob_amp = sgn(rng) * 40.0 * strength;
    const double bx       = uni(rng) * (double)W;
    const double by       = uni(rng) * (double)H;
    const double bs       = (0.20 + 0.30 * uni(rng)) * (double)W;
    const double bs2      = 2.0 * bs * bs;
    const double gain_amp = uni(rng) * 0.20 * strength;
    const double gain_th  = ang(rng);

    const double cx = W * 0.5, cy = H * 0.5;
    const double r_max2 = cx * cx + cy * cy;
    const double gvx = std::cos(grad_th), gvy = std::sin(grad_th);
    const double tvx = std::cos(gain_th), tvy = std::sin(gain_th);

    for (int y = 0; y < H; ++y) {
        uint8_t* row = img.ptr<uint8_t>(y);
        const double v_y = (y - cy) / cy;     // [-1, 1]
        for (int x = 0; x < W; ++x) {
            const double u_x = (x - cx) / cx; // [-1, 1]
            double v = (double)row[x];
            // 1. Linear gradient (additive).
            v += grad_amp * (gvx * u_x + gvy * v_y);
            // 2. Vignette (multiplicative).
            const double dx = x - cx, dy = y - cy;
            const double r2 = (dx * dx + dy * dy) / r_max2;
            v *= 1.0 - vign * r2;
            // 3. Gaussian blob (additive).
            const double bdx = x - bx, bdy = y - by;
            const double br2 = (bdx * bdx + bdy * bdy) / bs2;
            v += blob_amp * std::exp(-br2);
            // 4. Low-frequency multiplicative gain.
            v *= 1.0 + gain_amp * (tvx * u_x + tvy * v_y);

            if (v < 0)   v = 0;
            if (v > 255) v = 255;
            row[x] = (uint8_t)v;
        }
    }
}

// Apply Gaussian + salt&pepper noise in-place.
void add_noise(cv::Mat& img, std::mt19937& rng,
               double gaussian_sigma, double salt_pepper_fraction) {
    const int W = img.cols, H = img.rows;
    std::normal_distribution<double>       gn(0.0, gaussian_sigma);
    std::uniform_real_distribution<double> sp(0.0, 1.0);
    for (int y = 0; y < H; ++y) {
        uint8_t* row = img.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            if (gaussian_sigma > 0) {
                double v = (double)row[x] + gn(rng);
                if (v < 0)       v = 0;
                if (v > 255)     v = 255;
                row[x] = (uint8_t)v;
            }
            if (salt_pepper_fraction > 0) {
                double r = sp(rng);
                if      (r < salt_pepper_fraction * 0.5) row[x] = 0;
                else if (r < salt_pepper_fraction)       row[x] = 255;
            }
        }
    }
}
} // anon

// ---- Fixed scene (kept for regression) ---------------------------------

cv::Mat make_weak_curve_with_spikes(const GroundTruth& gt_in) {
    // The caller may have left evaluate unset; patch in the fixed S-curve.
    GroundTruth gt = gt_in;
    const double y0 = gt.y0, A = (gt.amplitude > 0 ? gt.amplitude : 15.0);
    if (!gt.evaluate) {
        gt.evaluate = [y0, A](double x) {
            return y0 + A * std::sin(x * 2.0 * kPi / 320.0);
        };
    }
    cv::Mat img(gt.H, gt.W, CV_8UC1);
    for (int y = 0; y < gt.H; ++y) {
        uint8_t* row = img.ptr<uint8_t>(y);
        for (int x = 0; x < gt.W; ++x) {
            row[x] = (uint8_t)((double)y < gt(x) ? 90 : 150);
        }
    }
    // 30 deterministic spike blocks, same as before.
    for (int i = 0; i < 30; ++i) {
        int sx = 30 + ((i * 13 + 7) % 260);
        int sy = 93 + ((i * 17 + 11) % 18);
        for (int dy = 0; dy < 5; ++dy)
        for (int dx = 0; dx < 5; ++dx) {
            int yy = sy + dy, xx = sx + dx;
            if (yy < 0 || yy >= gt.H || xx < 0 || xx >= gt.W) continue;
            img.at<uint8_t>(yy, xx) = 240;
        }
    }
    return img;
}

// ---- Randomised scene --------------------------------------------------

RandomScene make_random_scene(int seed, NoiseLevel level, bool dashed_edge,
                              bool bumpy_edge) {
    const bool harsh = (level == NoiseLevel::Harsh);
    const bool low   = (level == NoiseLevel::Low);
    const bool none  = (level == NoiseLevel::None);
    const bool photo = (level == NoiseLevel::Photo);
    std::mt19937 rng((uint32_t)seed);
    auto uni_d = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };
    auto uni_i = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };

    const int W = 320, H = 240;
    GroundTruth gt;
    gt.W = W; gt.H = H;
    gt.y0 = 120;

    // Pick a curve shape. Each covers a distinct regime of smoothness /
    // inflection so algorithms that overfit to any single shape lose.
    const int shape_id = uni_i(0, 4);
    const double A     = uni_d(4.0, 28.0);
    const double phase = uni_d(0.0, 2.0 * kPi);
    const double slope = uni_d(-0.25, 0.25);   // px per px
    const double c     = uni_d(-1.0, 1.0);     // cubic shape param

    switch (shape_id) {
        case 0: {
            gt.shape_name = "line_tilted";
            double s = slope, y0 = gt.y0;
            gt.evaluate = [y0, s](double x) {
                return y0 + s * (x - 160.0);
            };
            gt.amplitude = std::abs(s * 160.0);
            break;
        }
        case 1: {
            gt.shape_name = "half_sine";
            double y0 = gt.y0;
            gt.evaluate = [y0, A, phase](double x) {
                return y0 + A * std::sin(x * kPi / 320.0 + phase);
            };
            gt.amplitude = A;
            break;
        }
        case 2: {
            gt.shape_name = "full_sine";
            double y0 = gt.y0;
            gt.evaluate = [y0, A, phase](double x) {
                return y0 + A * std::sin(x * 2.0 * kPi / 320.0 + phase);
            };
            gt.amplitude = A;
            break;
        }
        case 3: {
            gt.shape_name = "cubic";
            double y0 = gt.y0;
            // (u^3 - 3u) / 2 maps [-1, 1] to [-1, 1] with one inflection.
            gt.evaluate = [y0, A, c](double x) {
                double u = (x - 160.0) / 160.0;
                return y0 + A * (0.5 * u * u * u - 1.5 * u + c * u * u);
            };
            gt.amplitude = A;
            break;
        }
        default: {
            gt.shape_name = "quarter_arc";
            // A wide, shallow arc. Use y = y0 + A - sqrt(R² - x²) clipped.
            double y0 = gt.y0;
            double R  = 200.0 + A * 4.0;
            gt.evaluate = [y0, A, R](double x) {
                double xr = x - 160.0;
                double val = std::sqrt(std::max(1e-9, R * R - xr * xr));
                return y0 + (R - val) * (A / (R - std::sqrt(R * R - 160.0 * 160.0) + 1e-9));
            };
            gt.amplitude = A;
            break;
        }
    }

    // Optional polynomial-defeating perturbation: 3 moderate Gaussian
    // bumps superimposed at different x positions. Each bump individually
    // looks smooth and could be absorbed by a degree-3 fit, but the sum
    // creates 3+ inflections beyond what a degree-5 polynomial can
    // represent. Target: visually-smooth curve with ~4-5 px max residual
    // against any polynomial fit, while the per-point deviation stays
    // within what a dense DP can track faithfully.
    if (bumpy_edge) {
        struct Bump { double A, mu, sg; };
        std::vector<Bump> bumps;
        bumps.reserve(3);
        for (int i = 0; i < 3; ++i) {
            bumps.push_back({
                uni_d(1.5, 2.5) * (uni_i(0, 1) ? 1.0 : -1.0),
                uni_d(0.15 * W, 0.85 * W),
                uni_d(20.0, 40.0)
            });
        }
        auto prev_eval = gt.evaluate;
        gt.evaluate = [prev_eval, bumps](double x) {
            double y = prev_eval(x);
            for (const auto& b : bumps) {
                double d = x - b.mu;
                y += b.A * std::exp(-(d * d) / (2.0 * b.sg * b.sg));
            }
            return y;
        };
    }

    // Backgrounds and edge contrast. In harsh mode δ can drop to 15,
    // giving nearly-invisible edges against the background. Low mode
    // guarantees strong contrast so stripe effects can be isolated.
    const int    bg_lo     = uni_i(40, 120);
    const int    delta     = harsh ? uni_i(15, 70)
                            : low   ? uni_i(50, 90)
                            : none  ? uni_i(50, 90)
                            :         uni_i(30, 90);
    const int    bg_hi     = std::min(255, bg_lo + delta);
    gt.edge_contrast       = delta;

    // Paint the curve — above curve = bg_lo, below = bg_hi.
    cv::Mat img(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        uint8_t* row = img.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            row[x] = (uint8_t)(y < gt(x) ? bg_lo : bg_hi);
        }
    }

    // Dashed-edge option: periodically blur vertical strips over the
    // curve's expected y-range. Applied *before* spikes and stripes so
    // the distractors remain sharp while only the real curve edge
    // suffers the discontinuity.
    if (dashed_edge) {
        const int y_band = std::max(12, (int)std::round(gt.amplitude + 8.0));
        apply_dashed_edge(img, rng, gt.y0, y_band);
    }

    // Spike field — above the midline so they don't occlude the curve
    // edge. Intensity chosen so spikes are clearly stronger than the
    // curve's δ. Harsh mode doubles the spike count and widens the
    // block-size range so clusters can span more of the search band.
    const int  spike_count     = harsh ? uni_i(20, 100)
                               : low   ? uni_i(3, 15)
                               : none  ? 0
                               :         uni_i(5, 50);
    const int  spike_bmin      = 3;
    const int  spike_bmax      = harsh ? 12 : 8;
    const int  spike_intensity = std::min(255, bg_lo + delta * 2 + uni_i(20, 60));
    draw_spikes(img, rng, spike_count, spike_bmin, spike_bmax,
                5, (int)gt.y0 - 10, spike_intensity);
    gt.spike_count = spike_count;

    // False horizontal stripes — length up to 30 % of W, placed inside
    // the caliper search band (±40 px of y0). These create real-looking
    // horizontal gradient lines that can outrank the curve on a pure
    // edge-strength metric. Harsh mode roughly triples the count.
    // Low mode keeps stripes at the normal count — they are what we
    // are trying to study in isolation of pixel noise.
    const int    stripe_count     = harsh ? uni_i(6, 12) : uni_i(2, 5);
    const double stripe_max_frac  = 0.30;
    const int    stripe_intensity = std::min(255, bg_lo + delta * 2 + uni_i(10, 40));
    const int    stripe_y_lo      = std::max(5, (int)gt.y0 - 40);
    const int    stripe_y_hi      = std::min(H - 4, (int)gt.y0 + 40);
    draw_stripes(img, rng, stripe_count, stripe_max_frac,
                 stripe_y_lo, stripe_y_hi, stripe_intensity);

    // Photometric augmentation (lighting/shading/tint). Photo mode
    // applies it at full strength; harsh adds a milder pass. Other
    // levels skip — keeps the existing benchmark numbers reproducible.
    const double photo_strength = photo ? uni_d(0.6, 1.5)
                                : harsh ? uni_d(0.0, 0.5)
                                :         0.0;
    if (photo_strength > 0.0) apply_photometric(img, rng, photo_strength);

    // Noise — Gaussian σ and salt-pepper fraction sampled independently.
    // Harsh mode raises both ceilings; low mode nearly eliminates them.
    const double gsigma = harsh ? uni_d(0.0, 12.0)
                         : low   ? uni_d(0.0, 1.5)
                         : none  ? 0.0
                         : photo ? uni_d(0.0, 6.0)
                         :         uni_d(0.0, 6.0);
    const double sp     = harsh ? uni_d(0.0, 0.04)
                         : low   ? uni_d(0.0, 0.002)
                         : none  ? 0.0
                         : photo ? uni_d(0.0, 0.012)
                         :         uni_d(0.0, 0.012);
    add_noise(img, rng, gsigma, sp);
    gt.gaussian_sigma = gsigma;

    return { img, gt };
}

} // namespace lab
