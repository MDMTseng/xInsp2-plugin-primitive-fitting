//
// primitive_fitting.cpp — caliper-based primitive fitting plugin.
//
// Phases (each one is self-contained and independently verifiable):
//   1 [THIS PHASE] Line region + Line fit — parity with line_detection
//   2              Arc region + Circle fit
//   3              Polynomial residual filter (second-stage outlier rejection)
//   4              Ellipse arc region + Ellipse fit
//
// Architecture:
//   * A region_mode picks how the N calipers are laid out (line, arc, …).
//     generate_samples() yields N CaliperSample{ position, normal } pairs;
//     every mode reuses the same 1-D edge finder that scans along `normal`.
//   * A fit_model picks which primitive RANSAC fits through the edge hits.
//   * An optional polynomial residual filter refines the fit by fitting
//     a polynomial to residual-vs-arc-length and rejecting outliers.
//
// Exchange commands:
//   set_region {mode:"line"|"arc"|"ellipse_arc", ...params}
//   set_config {polarity, num_calipers, caliper_width, caliper_span,
//               min_edge_strength, fit_model, ransac_*, poly_*, criteria…}
//   get_preview                      — live src JPEG with region overlay
//   get_last_result                  — last detection PNG (base64)
//   anything else                    — status JSON
//

#include <xi/xi_abi.hpp>
#include <xi/xi_json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

// --- OpenCV / xi::Image adapters ----------------------------------------
static cv::Mat xi_to_cv(const xi::Image& im) {
    if (im.empty()) return {};
    return cv::Mat(im.height, im.width, CV_8UC(im.channels), (void*)im.data()).clone();
}
static xi::Image cv_to_xi(const cv::Mat& m) {
    if (m.empty()) return {};
    cv::Mat c = m.isContinuous() ? m : m.clone();
    return xi::Image(c.cols, c.rows, c.channels(), c.data);
}
static cv::Mat to_gray(const cv::Mat& m) {
    if (m.channels() == 1) return m;
    cv::Mat g; cv::cvtColor(m, g,
        m.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    return g;
}
static cv::Mat to_rgb(const cv::Mat& m) {
    if (m.channels() == 3) return m;
    cv::Mat r; cv::cvtColor(m, r,
        m.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
    return r;
}

// --- base64 -------------------------------------------------------------
static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string b64_encode(const uint8_t* data, size_t n) {
    std::string out; out.reserve(((n + 2) / 3) * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < n) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < n) v |= (uint32_t)data[i + 2];
        out += B64[(v >> 18) & 63];
        out += B64[(v >> 12) & 63];
        out += (i + 1 < n) ? B64[(v >> 6) & 63] : '=';
        out += (i + 2 < n) ? B64[v & 63]        : '=';
    }
    return out;
}

static double parabolic_peak(double ym1, double y0, double yp1) {
    double denom = (ym1 - 2.0 * y0 + yp1);
    if (std::abs(denom) < 1e-9) return 0.0;
    return 0.5 * (ym1 - yp1) / denom;
}

} // anon

// --- plugin types -------------------------------------------------------

enum class RegionMode { Line = 0, Arc = 1, EllipseArc = 2 };
enum class FitModel   { Line = 0, Circle = 1, Ellipse = 2, Polynomial = 3 };
enum class Polarity   { Any = 0, DarkToBright = 1, BrightToDark = 2 };

struct Config {
    Polarity polarity = Polarity::Any;
    int num_calipers  = 15;
    int caliper_width = 5;
    int caliper_span  = 60;
    int min_edge_strength = 20;
    // Per-caliper top-N local peaks (1 = single strongest, Halcon/Cognex-
    // style). Peaks are found as local maxima on |gradient|, NMS-separated
    // by edge_min_separation_px, then the top N by strength are kept.
    int    top_n_per_caliper       = 1;
    int    edge_min_separation_px  = 3;
    // Each kept peak's magnitude must be ≥ alpha × strongest-peak magnitude
    // inside the same caliper (0 disables the relative filter).
    double top_n_min_alpha         = 0.0;

    FitModel fit_model = FitModel::Line;
    double ransac_threshold_px = 1.5;
    int    ransac_iterations   = 200;
    // Expected fraction of outliers among caliper hits. Used to derive the
    // *actual* RANSAC iteration budget via the classical formula
    //   N = log(1 - p) / log(1 - (1-ε)^s),  p = 0.99,
    // with `s` the minimal-sample size per model. `ransac_iterations` acts
    // as an upper cap. Lower ε ⇒ far fewer iterations (ε=0.05 ⇒ ~3–8 for
    // line/circle; 20× faster than the 200-iter default) while still
    // achieving 99% confidence of hitting an all-inlier minimal sample.
    double expected_outlier_rate = 0.5;
    // When true, RANSAC scores a candidate model by Σ hit.strength of its
    // inliers rather than by inlier count — stronger edges outweigh more
    // numerous but weaker ones, useful with top-N on multi-edge scenes.
    bool   ransac_weight_by_strength = false;

    // Phase 3 polynomial residual filter.
    bool   poly_enabled = false;
    int    poly_degree  = 0;
    double poly_reject_sigma = 3.0;
    // Upper bound (in degrees) on the local slope of a polynomial fit,
    // measured as angle from the region's line direction. Candidates
    // whose max |dv/dx| exceeds tan(this) are rejected during RANSAC,
    // preventing wild shape from winning just because it threads a few
    // noise points. 0 or ≥90 disables the check.
    double poly_max_slope_deg = 0.0;

    // Criteria — activated iff > 0.
    double min_length_px    = 0.0;
    double min_inlier_ratio = 0.0;
};

// Region types — only Line is populated in Phase 1; the others carry
// default-zero state so the JSON surface is stable across phases.
struct LineRegion {
    double p1x = 0, p1y = 0, p2x = 0, p2y = 0;
    bool empty() const { return std::hypot(p2x - p1x, p2y - p1y) < 1.0; }
};
struct ArcRegion {
    double cx = 0, cy = 0, r = 0;
    double theta_start_deg = 0, theta_end_deg = 0;
};
struct EllipseArcRegion {
    double cx = 0, cy = 0, a = 0, b = 0;
    double rotation_deg = 0;
    double theta_start_deg = 0, theta_end_deg = 0;
};

struct CaliperSample {
    cv::Point2d pos;    // caliper center in image coords
    cv::Point2d normal; // search direction (unit vector)
};

struct CaliperHit {
    cv::Point2d pos;
    double  strength = 0;
    double  t = 0;      // parametric position along the region (arc length etc.)
    bool    inlier = false;
};

struct FitResult {
    // Unified representation; each FitModel fills the relevant subset.
    FitModel model = FitModel::Line;

    // Line: p1 → p2 endpoints clipped to inlier span.
    cv::Point2d p1{0, 0}, p2{0, 0};
    double angle_deg = 0, length_px = 0;

    // Circle fit: centre + radius, plus the angular span actually covered
    // by inliers so drawing can clip the arc to what was observed.
    double cx = 0, cy = 0, r = 0;
    double theta_start_deg = 0, theta_end_deg = 0;

    // Ellipse fit params (phase 4).
    double a = 0, b = 0, rotation_deg = 0;

    // Polynomial fit: coefficients of v = a0 + a1*u + … + ak*u^k, where
    // (u, v) is the region-local frame (u along the line, normalised to
    // [-1, 1]; v perpendicular offset in pixels). Empty unless the fit
    // model is Polynomial.
    std::vector<double> poly_coeffs;
    // Diagnostics — how many RANSAC candidates were pre-rejected by the
    // slope limit vs fitted but lost on consensus. Helps distinguish
    // "no data to fit" from "constraint was too tight".
    int slope_rejected_count = 0;
    int ransac_attempts      = 0;

    std::vector<CaliperHit> hits;
    int    inlier_count = 0;
    int    total_hits   = 0;
    double inlier_ratio = 0;

    // Polynomial residual filter metrics.
    bool   poly_applied = false;
    int    poly_rejected = 0;
    double residual_std = 0;       // std of (actual residual − polynomial prediction)

    // Confidence / stability metrics (populated by every RANSAC fit).
    //   stability          — mean pairwise Jaccard of the top-3 scoring
    //                         inlier sets observed during RANSAC. 1.0 means
    //                         the three best hypotheses selected the same
    //                         points; 0 means they disagreed completely.
    //   residual_median_px — median |residual| of the final inlier set.
    //   confidence         — combined 0–1 score:
    //                         stability × (1 − median_res/thr) × inlier_ratio.
    double stability          = 0.0;
    double residual_median_px = 0.0;
    double confidence         = 0.0;
    // Fraction of unique calipers (not raw hits) that contributed an
    // inlier. Used in `confidence` instead of inlier_ratio so the metric
    // stays well-behaved when top_n_per_caliper > 1 inflates total_hits
    // without changing the underlying coverage.
    double caliper_coverage   = 0.0;

    bool   ok = false;
    bool   pass = false;
    std::string fail_reason;
};

// --- sample generators --------------------------------------------------
namespace samples {

static std::vector<CaliperSample> from_line(const LineRegion& ln, int N) {
    std::vector<CaliperSample> out;
    const double dx = ln.p2x - ln.p1x, dy = ln.p2y - ln.p1y;
    const double L  = std::hypot(dx, dy);
    if (L < 1) return out;
    const cv::Point2d ex(dx / L, dy / L);       // along line
    const cv::Point2d ey(-ex.y, ex.x);          // perpendicular (search dir)
    const cv::Point2d c((ln.p1x + ln.p2x) * 0.5,
                        (ln.p1y + ln.p2y) * 0.5);
    out.reserve(N);
    for (int i = 0; i < N; ++i) {
        double u = -L * 0.5 + (L * (i + 0.5)) / N;
        out.push_back({c + ex * u, ey});
    }
    return out;
}

// Normalise a [start, end] angle pair (in radians) to a CCW sweep of
// [0, 2π]. A zero-length sweep is treated as the full circle so that
// the common `0 → 360` input means what it says.
static double ccw_span(double t0, double t1) {
    double span = t1 - t0;
    while (span < 0)             span += 2 * CV_PI;
    while (span > 2 * CV_PI + 1e-9) span -= 2 * CV_PI;
    if (span < 1e-6) span = 2 * CV_PI;
    return span;
}

static std::vector<CaliperSample> from_arc(const ArcRegion& arc, int N) {
    std::vector<CaliperSample> out;
    if (arc.r <= 1) return out;
    double t0   = arc.theta_start_deg * CV_PI / 180.0;
    double span = ccw_span(t0, arc.theta_end_deg * CV_PI / 180.0);
    out.reserve(N);
    for (int i = 0; i < N; ++i) {
        double t = t0 + span * (i + 0.5) / N;
        cv::Point2d dir(std::cos(t), std::sin(t));   // radially outward
        cv::Point2d pos(arc.cx + arc.r * dir.x, arc.cy + arc.r * dir.y);
        out.push_back({pos, dir});
    }
    return out;
}

static std::vector<CaliperSample> from_ellipse_arc(const EllipseArcRegion& el, int N) {
    std::vector<CaliperSample> out;
    if (el.a <= 1 || el.b <= 1) return out;
    double t0   = el.theta_start_deg * CV_PI / 180.0;
    double span = ccw_span(t0, el.theta_end_deg * CV_PI / 180.0);
    const double rot = el.rotation_deg * CV_PI / 180.0;
    const double cR = std::cos(rot), sR = std::sin(rot);
    out.reserve(N);
    for (int i = 0; i < N; ++i) {
        double t = t0 + span * (i + 0.5) / N;
        double ct = std::cos(t), st = std::sin(t);
        // Ellipse-local: P(t) = (a cos t, b sin t). Outward normal is the
        // gradient of x²/a² + y²/b², i.e. (cos t / a, sin t / b) normalised.
        double xl = el.a * ct, yl = el.b * st;
        double nlx = ct / el.a, nly = st / el.b;
        double nln = std::hypot(nlx, nly);
        if (nln < 1e-9) continue;
        nlx /= nln; nly /= nln;
        cv::Point2d pos(el.cx + xl * cR - yl * sR,
                        el.cy + xl * sR + yl * cR);
        cv::Point2d dir(nlx * cR - nly * sR,
                        nlx * sR + nly * cR);
        out.push_back({pos, dir});
    }
    return out;
}

} // samples

// --- plugin class -------------------------------------------------------

class PrimitiveFitting : public xi::Plugin {
public:
    using xi::Plugin::Plugin;

    xi::Record process(const xi::Record& input) override {
        const xi::Image& src = input.get_image("src");
        if (src.empty()) return xi::Record().set("error", "no 'src' image");

        cv::Mat rgb  = to_rgb(xi_to_cv(src));
        cv::Mat gray = to_gray(xi_to_cv(src));

        Config cfg; RegionMode mode;
        LineRegion ln; ArcRegion arc; EllipseArcRegion el;
        {
            std::lock_guard<std::mutex> lk(mu_);
            cfg = cfg_; mode = region_mode_;
            ln  = line_; arc = arc_; el = ellipse_;
            last_frame_rgb_  = rgb.clone();
            last_frame_gray_ = gray.clone();
        }

        FitResult r = detect(gray, mode, ln, arc, el, cfg);
        cv::Mat overlay = draw(rgb, mode, ln, arc, el, cfg, r);
        {
            std::lock_guard<std::mutex> lk(mu_);
            last_result_  = r;
            last_overlay_ = overlay.clone();
        }

        xi::Record out;
        out.image("result", cv_to_xi(overlay));
        out.set("pass",         r.pass);
        out.set("found",        r.ok);
        out.set("fail_reason",  r.fail_reason);
        out.set("model",        fit_model_str(r.model));
        out.set("angle",        r.angle_deg);
        out.set("length",       r.length_px);
        out.set("inlier_count", r.inlier_count);
        out.set("total_hits",   r.total_hits);
        out.set("inlier_ratio", r.inlier_ratio);
        out.set("ransac_attempts",      r.ransac_attempts);
        out.set("slope_rejected_count", r.slope_rejected_count);
        out.set("poly_applied", r.poly_applied);
        out.set("poly_rejected", r.poly_rejected);
        out.set("residual_std", r.residual_std);
        out.set("confidence",         r.confidence);
        out.set("stability",          r.stability);
        out.set("residual_median_px", r.residual_median_px);
        out.set("caliper_coverage",   r.caliper_coverage);
        if (r.ok && r.model == FitModel::Line) {
            out.set("x1", r.p1.x).set("y1", r.p1.y);
            out.set("x2", r.p2.x).set("y2", r.p2.y);
        } else if (r.ok && r.model == FitModel::Circle) {
            out.set("cx", r.cx).set("cy", r.cy).set("radius", r.r);
            out.set("theta_start_deg", r.theta_start_deg);
            out.set("theta_end_deg",   r.theta_end_deg);
        } else if (r.ok && r.model == FitModel::Ellipse) {
            out.set("cx", r.cx).set("cy", r.cy);
            out.set("semi_major", r.a).set("semi_minor", r.b);
            out.set("rotation_deg", r.rotation_deg);
        } else if (r.ok && r.model == FitModel::Polynomial) {
            out.set("x1", r.p1.x).set("y1", r.p1.y);
            out.set("x2", r.p2.x).set("y2", r.p2.y);
            out.set("poly_degree_fitted", (int)r.poly_coeffs.size() - 1);
            // Coefficients live in the status / last_result JSONs so the
            // full array is available via exchange commands.
        }
        return out;
    }

    std::string exchange(const std::string& cmd) override {
        auto p = xi::Json::parse(cmd);
        std::string c = p["command"].as_string();
        if      (c == "set_config")      handle_set_config(p);
        else if (c == "set_region")      handle_set_region(p);
        else if (c == "get_preview")     return build_preview_json();
        else if (c == "get_last_result") return build_last_result_json();
        return build_status_json();
    }

    std::string get_def() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return build_def_json_locked();
    }
    bool set_def(const std::string& json) override {
        if (json.empty() || json == "{}") return true;
        auto p = xi::Json::parse(json);
        std::lock_guard<std::mutex> lk(mu_);
        apply_config_json_locked(p);
        apply_region_json_locked(p);
        return true;
    }

private:
    // --- command handlers ------------------------------------------------

    void handle_set_config(const xi::Json& p) {
        std::lock_guard<std::mutex> lk(mu_);
        apply_config_json_locked(p);
    }
    void handle_set_region(const xi::Json& p) {
        std::lock_guard<std::mutex> lk(mu_);
        apply_region_json_locked(p);
    }

    void apply_config_json_locked(const xi::Json& p) {
        if (p["polarity"].is_string()) {
            auto s = p["polarity"].as_string();
            cfg_.polarity = (s == "dark_to_bright") ? Polarity::DarkToBright
                          : (s == "bright_to_dark") ? Polarity::BrightToDark
                                                    : Polarity::Any;
        }
        if (p["fit_model"].is_string()) {
            auto s = p["fit_model"].as_string();
            cfg_.fit_model = (s == "circle")     ? FitModel::Circle
                           : (s == "ellipse")    ? FitModel::Ellipse
                           : (s == "polynomial") ? FitModel::Polynomial
                                                 : FitModel::Line;
        }
        auto I = [&](const char* k, int& v) {
            if (p[k].is_number()) {
                double raw = p[k].as_double();
                if (std::isfinite(raw)) v = (int)raw;
            }
        };
        auto D = [&](const char* k, double& v) {
            if (p[k].is_number()) {
                double raw = p[k].as_double();
                if (std::isfinite(raw)) v = raw;
            }
        };
        auto B = [&](const char* k, bool& v){ if (p[k].is_bool()) v = p[k].as_bool(); };
        I("num_calipers",        cfg_.num_calipers);
        I("caliper_width",       cfg_.caliper_width);
        I("caliper_span",        cfg_.caliper_span);
        I("min_edge_strength",   cfg_.min_edge_strength);
        I("top_n_per_caliper",       cfg_.top_n_per_caliper);
        I("edge_min_separation_px",  cfg_.edge_min_separation_px);
        D("top_n_min_alpha",         cfg_.top_n_min_alpha);
        D("ransac_threshold_px", cfg_.ransac_threshold_px);
        I("ransac_iterations",   cfg_.ransac_iterations);
        D("expected_outlier_rate", cfg_.expected_outlier_rate);
        B("ransac_weight_by_strength", cfg_.ransac_weight_by_strength);
        B("poly_enabled",        cfg_.poly_enabled);
        I("poly_degree",         cfg_.poly_degree);
        D("poly_reject_sigma",   cfg_.poly_reject_sigma);
        D("poly_max_slope_deg",  cfg_.poly_max_slope_deg);
        D("min_length_px",       cfg_.min_length_px);
        D("min_inlier_ratio",    cfg_.min_inlier_ratio);
        cfg_.num_calipers  = std::max(3, cfg_.num_calipers);
        cfg_.caliper_width = std::max(1, cfg_.caliper_width);
        cfg_.caliper_span  = std::max(5, cfg_.caliper_span);
        cfg_.poly_degree   = std::clamp(cfg_.poly_degree, 0, 6);
        cfg_.top_n_per_caliper      = std::max(1, cfg_.top_n_per_caliper);
        cfg_.edge_min_separation_px = std::max(1, cfg_.edge_min_separation_px);
        cfg_.top_n_min_alpha        = std::clamp(cfg_.top_n_min_alpha, 0.0, 1.0);
        cfg_.expected_outlier_rate  = std::clamp(cfg_.expected_outlier_rate, 0.0, 0.95);
    }

    void apply_region_json_locked(const xi::Json& p) {
        if (p["mode"].is_string()) {
            auto m = p["mode"].as_string();
            region_mode_ = (m == "arc")         ? RegionMode::Arc
                         : (m == "ellipse_arc") ? RegionMode::EllipseArc
                                                : RegionMode::Line;
        }
        // Line params (flat or nested under "line").
        xi::Json lp = p["line"].is_object() ? p["line"] : p;
        line_.p1x = lp["p1x"].as_double(line_.p1x);
        line_.p1y = lp["p1y"].as_double(line_.p1y);
        line_.p2x = lp["p2x"].as_double(line_.p2x);
        line_.p2y = lp["p2y"].as_double(line_.p2y);

        // Arc params (flat or nested under "arc").
        xi::Json ap = p["arc"].is_object() ? p["arc"] : p;
        arc_.cx              = ap["cx"].as_double(arc_.cx);
        arc_.cy              = ap["cy"].as_double(arc_.cy);
        arc_.r               = ap["r"] .as_double(arc_.r);
        arc_.theta_start_deg = ap["theta_start_deg"].as_double(arc_.theta_start_deg);
        arc_.theta_end_deg   = ap["theta_end_deg"]  .as_double(arc_.theta_end_deg);

        // Ellipse params (flat or nested under "ellipse").
        xi::Json ep = p["ellipse"].is_object() ? p["ellipse"] : p;
        ellipse_.cx              = ep["ecx"]             .as_double(ellipse_.cx);
        ellipse_.cy              = ep["ecy"]             .as_double(ellipse_.cy);
        ellipse_.a               = ep["ea"]              .as_double(ellipse_.a);
        ellipse_.b               = ep["eb"]              .as_double(ellipse_.b);
        ellipse_.rotation_deg    = ep["rotation_deg"]    .as_double(ellipse_.rotation_deg);
        ellipse_.theta_start_deg = ep["etheta_start_deg"].as_double(ellipse_.theta_start_deg);
        ellipse_.theta_end_deg   = ep["etheta_end_deg"]  .as_double(ellipse_.theta_end_deg);
    }

    // --- detection core --------------------------------------------------

    FitResult detect(const cv::Mat& gray, RegionMode mode,
                     const LineRegion& ln, const ArcRegion& arc,
                     const EllipseArcRegion& el, const Config& cfg) const {
        FitResult r;
        r.model = cfg.fit_model;

        std::vector<CaliperSample> scan;
        const int N = std::max(3, cfg.num_calipers);
        if (mode == RegionMode::Line) {
            if (ln.empty()) { r.fail_reason = "line region empty"; return r; }
            scan = samples::from_line(ln, N);
        } else if (mode == RegionMode::Arc) {
            if (arc.r <= 1) { r.fail_reason = "arc region empty"; return r; }
            scan = samples::from_arc(arc, N);
        } else if (mode == RegionMode::EllipseArc) {
            if (el.a <= 1 || el.b <= 1) { r.fail_reason = "ellipse region empty"; return r; }
            scan = samples::from_ellipse_arc(el, N);
        } else {
            r.fail_reason = "region mode not implemented";
            return r;
        }

        find_edges(gray, scan, cfg, r.hits);
        r.total_hits = (int)r.hits.size();
        int min_hits = 3;
        if (cfg.fit_model == FitModel::Ellipse)    min_hits = 5;
        if (cfg.fit_model == FitModel::Polynomial) min_hits = cfg.poly_degree + 1;
        if (r.total_hits < min_hits) { r.fail_reason = "too few caliper edges"; return r; }

        bool fit_ok = false;
        if      (cfg.fit_model == FitModel::Line)       fit_ok = fit_line_ransac(cfg, r);
        else if (cfg.fit_model == FitModel::Circle)     fit_ok = fit_circle_ransac(cfg, r);
        else if (cfg.fit_model == FitModel::Ellipse)    fit_ok = fit_ellipse_ransac(cfg, r);
        else if (cfg.fit_model == FitModel::Polynomial) fit_ok = fit_polynomial_ransac(mode, ln, cfg, r);
        else { r.fail_reason = "fit model not implemented"; return r; }
        if (!fit_ok) return r;

        // Optional second-stage polynomial residual filter. Redundant when
        // the primary fit is already polynomial, so skip in that case.
        if (cfg.poly_enabled && cfg.poly_degree > 0 && r.inlier_count >= 4 &&
            cfg.fit_model != FitModel::Polynomial) {
            apply_poly_filter(cfg, r);
        }

        // Criteria.
        r.pass = true; r.fail_reason.clear();
        if (cfg.min_length_px > 0 && r.length_px < cfg.min_length_px) {
            r.pass = false; r.fail_reason = "length below min";
        }
        if (cfg.min_inlier_ratio > 0 && r.inlier_ratio < cfg.min_inlier_ratio) {
            r.pass = false;
            if (r.fail_reason.empty()) r.fail_reason = "inlier ratio below min";
        }
        return r;
    }

    // 1-D edge finder shared by every region mode.
    static void find_edges(const cv::Mat& gray,
                           const std::vector<CaliperSample>& scan,
                           const Config& cfg,
                           std::vector<CaliperHit>& out_hits) {
        const int CW = std::max(1, cfg.caliper_width);
        const int SH = std::max(5, cfg.caliper_span);
        out_hits.reserve(scan.size());

        for (size_t i = 0; i < scan.size(); ++i) {
            const auto& s = scan[i];
            const cv::Point2d ey(s.normal);                 // search dir
            const cv::Point2d ex(-ey.y, ey.x);              // tangent

            std::vector<double> prof(SH, 0.0);
            for (int t = 0; t < SH; ++t) {
                double v = -SH * 0.5 + t + 0.5;
                double sum = 0; int nsum = 0;
                for (int k = -CW / 2; k <= CW / 2; ++k) {
                    double sx = s.pos.x + k * ex.x + v * ey.x;
                    double sy = s.pos.y + k * ex.y + v * ey.y;
                    int ix = (int)std::round(sx), iy = (int)std::round(sy);
                    if (ix < 0 || iy < 0 || ix >= gray.cols || iy >= gray.rows) continue;
                    sum += gray.at<uint8_t>(iy, ix);
                    ++nsum;
                }
                prof[t] = nsum > 0 ? sum / nsum : 0.0;
            }

            std::vector<double> grad(SH, 0.0);
            for (int t = 1; t < SH - 1; ++t) grad[t] = (prof[t + 1] - prof[t - 1]) * 0.5;

            // Collect every local-maximum peak on |gradient| that matches
            // the polarity filter and exceeds min_edge_strength.
            struct Peak { int t; double mag; };
            std::vector<Peak> peaks;
            for (int t = 2; t < SH - 2; ++t) {
                double g = grad[t];
                if (cfg.polarity == Polarity::DarkToBright && g <= 0) continue;
                if (cfg.polarity == Polarity::BrightToDark && g >= 0) continue;
                double mag = std::abs(g);
                if (mag < cfg.min_edge_strength) continue;
                if (std::abs(grad[t - 1]) > mag) continue;
                if (std::abs(grad[t + 1]) > mag) continue;
                peaks.push_back({t, mag});
            }
            if (peaks.empty()) continue;

            // Sort strongest → weakest, then drop anything weaker than
            // alpha × strongest (when alpha > 0), then non-maximum
            // suppression by edge_min_separation_px, then cap at top-N.
            std::sort(peaks.begin(), peaks.end(),
                      [](const Peak& a, const Peak& b) { return a.mag > b.mag; });
            const double max_mag   = peaks.front().mag;
            const double alpha_thr = cfg.top_n_min_alpha > 0
                                     ? cfg.top_n_min_alpha * max_mag : 0.0;

            std::vector<Peak> kept;
            kept.reserve(std::min<size_t>(peaks.size(), (size_t)cfg.top_n_per_caliper));
            const int sep = cfg.edge_min_separation_px;
            for (const auto& p : peaks) {
                if (p.mag < alpha_thr) break;                 // sorted, so all weaker fail too
                bool too_close = false;
                for (const auto& k : kept) {
                    if (std::abs(p.t - k.t) < sep) { too_close = true; break; }
                }
                if (too_close) continue;
                kept.push_back(p);
                if ((int)kept.size() >= cfg.top_n_per_caliper) break;
            }

            for (const auto& p : kept) {
                double sub = parabolic_peak(std::abs(grad[p.t - 1]),
                                            std::abs(grad[p.t]),
                                            std::abs(grad[p.t + 1]));
                double v_sub = -SH * 0.5 + p.t + 0.5 + sub;
                CaliperHit h;
                h.pos.x = s.pos.x + v_sub * ey.x;
                h.pos.y = s.pos.y + v_sub * ey.y;
                h.strength = p.mag;
                h.t = (double)i / (double)std::max<size_t>(1, scan.size() - 1);
                out_hits.push_back(h);
            }
        }
    }

    // --- Adaptive RANSAC iteration budget --------------------------------
    //
    // Given expected outlier rate ε and minimal-sample size s, pick the
    // iteration count that gives probability p_conf of drawing at least
    // one all-inlier minimal sample:
    //   N = ceil( log(1 - p_conf) / log(1 - (1-ε)^s) ) × safety
    // Result is clamped to [floor, user_cap] so the UI slider still acts
    // as a hard ceiling.
    static int adaptive_iters(double eps, int s, int user_cap, int floor_n,
                              double p_conf = 0.99, double safety = 3.0) {
        eps = std::clamp(eps, 0.0, 0.95);
        const double p_all_in = std::pow(1.0 - eps, (double)s);
        int need;
        if (p_all_in >= 1.0 - 1e-9) {
            need = floor_n;               // ε ≈ 0 ⇒ one sample is enough
        } else {
            const double num = std::log(1.0 - p_conf);
            const double den = std::log(1.0 - p_all_in);
            need = (int)std::ceil(num / den * safety);
        }
        return std::clamp(need, floor_n, std::max(floor_n, user_cap));
    }

    // --- Top-K hypothesis tracker for confidence scoring -----------------
    //
    // Keeps the highest-scoring K inlier sets seen during a RANSAC run;
    // the pairwise Jaccard similarity of those sets is the "stability"
    // signal — if the top few hypotheses all latched onto the same
    // points, the result is replicable; if they diverged, the RANSAC
    // outcome is brittle. K=3 is enough for a smoothed signal without
    // extra bookkeeping cost.
    static constexpr int CONF_TOPK = 3;
    struct TopKRanker {
        std::array<std::pair<double, std::vector<int>>, CONF_TOPK> slot{};
        TopKRanker() { for (auto& s : slot) s.first = -1.0; }
        void offer(double score, const std::vector<int>& inliers) {
            for (int i = 0; i < CONF_TOPK; ++i) {
                if (score > slot[i].first) {
                    // shift down
                    for (int j = CONF_TOPK - 1; j > i; --j) slot[j] = std::move(slot[j-1]);
                    slot[i] = {score, inliers};
                    return;
                }
            }
        }
    };
    static double jaccard(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.empty() && b.empty()) return 0.0;
        std::vector<int> ia(a), ib(b);
        std::sort(ia.begin(), ia.end()); std::sort(ib.begin(), ib.end());
        std::vector<int> inter, uni;
        std::set_intersection(ia.begin(), ia.end(), ib.begin(), ib.end(),
                              std::back_inserter(inter));
        std::set_union(ia.begin(), ia.end(), ib.begin(), ib.end(),
                       std::back_inserter(uni));
        return uni.empty() ? 0.0 : (double)inter.size() / (double)uni.size();
    }
    static double stability_from_topk(const TopKRanker& tk) {
        double sum = 0;
        int pairs = 0;
        for (int i = 0; i < CONF_TOPK; ++i) {
            if (tk.slot[i].first < 0) continue;
            for (int j = i + 1; j < CONF_TOPK; ++j) {
                if (tk.slot[j].first < 0) continue;
                sum += jaccard(tk.slot[i].second, tk.slot[j].second);
                ++pairs;
            }
        }
        return pairs > 0 ? sum / pairs : 0.0;
    }

    // Compute residual median + unique-caliper coverage, then combine
    // into the 0–1 confidence score. The coverage term uses CaliperHit::t
    // (== caliper_index / (N-1)) to count distinct calipers, which is
    // robust to top_n_per_caliper > 1 inflating the raw hit count.
    static void finalize_confidence(FitResult& r, double stability,
                                    const std::vector<int>& inlier_indices,
                                    const std::vector<double>& residuals_abs,
                                    double thr_px) {
        double med = 0.0;
        if (!residuals_abs.empty()) {
            std::vector<double> rs(residuals_abs);
            std::nth_element(rs.begin(), rs.begin() + rs.size()/2, rs.end());
            med = rs[rs.size()/2];
        }
        // Map t to an integer caliper id (t = i/(N-1) with N up to ~10⁴
        // → 1e6 quantisation is well clear of collision).
        auto key = [](double t) { return (int64_t)std::llround(t * 1e6); };
        std::unordered_set<int64_t> all_cal, in_cal;
        for (const auto& h : r.hits) all_cal.insert(key(h.t));
        for (int idx : inlier_indices) in_cal.insert(key(r.hits[idx].t));
        double cov = all_cal.empty() ? 0.0
            : (double)in_cal.size() / (double)all_cal.size();

        r.stability          = stability;
        r.residual_median_px = med;
        r.caliper_coverage   = cov;
        double tightness     = std::clamp(1.0 - med / std::max(thr_px, 1e-6), 0.0, 1.0);
        r.confidence         = std::clamp(stability * tightness * cov, 0.0, 1.0);
    }

    // --- Line fit (RANSAC + LSQ) -----------------------------------------

    static bool fit_line_ransac(const Config& cfg, FitResult& r) {
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> pick(0, r.total_hits - 1);
        std::vector<int> best_in;
        const double thr = std::max(0.1, cfg.ransac_threshold_px);

        auto line_from = [](const cv::Point2d& a, const cv::Point2d& b) -> cv::Vec3d {
            double dx = b.x - a.x, dy = b.y - a.y;
            double L = std::hypot(dx, dy);
            if (L < 1e-9) return {0, 0, 0};
            double nx = -dy / L, ny = dx / L;
            double c  = -(nx * a.x + ny * a.y);
            return {nx, ny, c};
        };

        const int iters = adaptive_iters(cfg.expected_outlier_rate, 2,
                                         cfg.ransac_iterations, 20);
        double best_score = 0;
        TopKRanker topk;
        for (int it = 0; it < iters; ++it) {
            int i1 = pick(rng), i2 = pick(rng);
            if (i1 == i2) continue;
            auto l = line_from(r.hits[i1].pos, r.hits[i2].pos);
            if (l[0] == 0 && l[1] == 0) continue;
            std::vector<int> inl;
            double score = 0;
            for (int k = 0; k < r.total_hits; ++k) {
                double d = std::abs(l[0] * r.hits[k].pos.x + l[1] * r.hits[k].pos.y + l[2]);
                if (d <= thr) {
                    inl.push_back(k);
                    score += cfg.ransac_weight_by_strength ? r.hits[k].strength : 1.0;
                }
            }
            topk.offer(score, inl);
            if (score > best_score) { best_score = score; best_in = std::move(inl); }
        }
        if (best_in.size() < 2) { r.fail_reason = "ransac found no consensus"; return false; }
        const double stab = stability_from_topk(topk);

        std::vector<cv::Point2d> pts; pts.reserve(best_in.size());
        for (int idx : best_in) { r.hits[idx].inlier = true; pts.push_back(r.hits[idx].pos); }
        cv::Vec4d lf;
        cv::fitLine(pts, lf, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Point2d dir(lf[0], lf[1]), pt(lf[2], lf[3]);

        double tmin = 1e18, tmax = -1e18;
        for (const auto& q : pts) {
            double t = (q.x - pt.x) * dir.x + (q.y - pt.y) * dir.y;
            if (t < tmin) tmin = t;
            if (t > tmax) tmax = t;
        }
        r.p1 = pt + tmin * dir;
        r.p2 = pt + tmax * dir;
        r.length_px = std::hypot(r.p2.x - r.p1.x, r.p2.y - r.p1.y);
        double ang = std::atan2(r.p2.y - r.p1.y, r.p2.x - r.p1.x) * 180.0 / CV_PI;
        while (ang >   90) ang -= 180;
        while (ang <= -90) ang += 180;
        r.angle_deg    = ang;
        r.inlier_count = (int)best_in.size();
        r.inlier_ratio = (double)r.inlier_count / r.total_hits;
        // Residuals of final inliers against the refit line.
        std::vector<double> res; res.reserve(best_in.size());
        double nx = -dir.y, ny = dir.x, c_off = -(nx * pt.x + ny * pt.y);
        for (int idx : best_in) {
            res.push_back(std::abs(nx * r.hits[idx].pos.x +
                                   ny * r.hits[idx].pos.y + c_off));
        }
        finalize_confidence(r, stab, best_in, res, thr);
        r.ok = true;
        return true;
    }

    // --- Circle fit (RANSAC + algebraic LSQ) -----------------------------
    //
    // Algebraic form: x² + y² + Dx + Ey + F = 0 → (cx, cy) = (-D/2, -E/2),
    // r = sqrt(D²/4 + E²/4 - F). Linear in (D, E, F) given (x, y, x²+y²).

    static bool circle_from_3pts(const cv::Point2d& a, const cv::Point2d& b,
                                 const cv::Point2d& c,
                                 double& cx, double& cy, double& r) {
        double d = 2 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
        if (std::abs(d) < 1e-6) return false;
        double a_sq = a.x * a.x + a.y * a.y;
        double b_sq = b.x * b.x + b.y * b.y;
        double c_sq = c.x * c.x + c.y * c.y;
        cx = ((b_sq - a_sq) * (c.y - a.y) - (c_sq - a_sq) * (b.y - a.y)) / d;
        cy = ((c_sq - a_sq) * (b.x - a.x) - (b_sq - a_sq) * (c.x - a.x)) / d;
        r = std::hypot(a.x - cx, a.y - cy);
        return r > 0.5;
    }

    static bool circle_lsq(const std::vector<cv::Point2d>& pts,
                           double& cx, double& cy, double& r) {
        int n = (int)pts.size();
        if (n < 3) return false;
        cv::Mat A(n, 3, CV_64F), b(n, 1, CV_64F);
        for (int i = 0; i < n; ++i) {
            A.at<double>(i, 0) = pts[i].x;
            A.at<double>(i, 1) = pts[i].y;
            A.at<double>(i, 2) = 1.0;
            b.at<double>(i, 0) = -(pts[i].x * pts[i].x + pts[i].y * pts[i].y);
        }
        cv::Mat x;
        if (!cv::solve(A, b, x, cv::DECOMP_SVD)) return false;
        double D = x.at<double>(0), E = x.at<double>(1), F = x.at<double>(2);
        cx = -D * 0.5; cy = -E * 0.5;
        double rr = D * D * 0.25 + E * E * 0.25 - F;
        if (rr < 0) return false;
        r = std::sqrt(rr);
        return true;
    }

    static bool fit_circle_ransac(const Config& cfg, FitResult& r) {
        if (r.total_hits < 3) { r.fail_reason = "circle needs ≥3 hits"; return false; }
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> pick(0, r.total_hits - 1);
        const double thr = std::max(0.1, cfg.ransac_threshold_px);
        std::vector<int> best_in;

        const int iters = adaptive_iters(cfg.expected_outlier_rate, 3,
                                         cfg.ransac_iterations, 50);
        double best_score = 0;
        TopKRanker topk;
        for (int it = 0; it < iters; ++it) {
            int i1 = pick(rng), i2 = pick(rng), i3 = pick(rng);
            if (i1 == i2 || i2 == i3 || i1 == i3) continue;
            double cx, cy, rr;
            if (!circle_from_3pts(r.hits[i1].pos, r.hits[i2].pos, r.hits[i3].pos,
                                  cx, cy, rr)) continue;
            std::vector<int> inl;
            double score = 0;
            for (int k = 0; k < r.total_hits; ++k) {
                double d = std::hypot(r.hits[k].pos.x - cx, r.hits[k].pos.y - cy) - rr;
                if (std::abs(d) <= thr) {
                    inl.push_back(k);
                    score += cfg.ransac_weight_by_strength ? r.hits[k].strength : 1.0;
                }
            }
            topk.offer(score, inl);
            if (score > best_score) { best_score = score; best_in = std::move(inl); }
        }
        if (best_in.size() < 3) { r.fail_reason = "ransac found no consensus"; return false; }
        const double stab = stability_from_topk(topk);

        std::vector<cv::Point2d> pts; pts.reserve(best_in.size());
        for (int idx : best_in) { r.hits[idx].inlier = true; pts.push_back(r.hits[idx].pos); }
        double cx, cy, rr;
        if (!circle_lsq(pts, cx, cy, rr)) { r.fail_reason = "lsq circle fit failed"; return false; }
        r.cx = cx; r.cy = cy; r.r = rr;

        // Arc extent from inlier angles (min/max on a sorted unwrapped ring).
        std::vector<double> angs; angs.reserve(pts.size());
        for (auto& p : pts) angs.push_back(std::atan2(p.y - cy, p.x - cx));
        std::sort(angs.begin(), angs.end());
        // Find the largest gap on the ring — arc spans the complementary range.
        double best_gap = 0, best_gap_end = angs[0];
        for (size_t i = 0; i < angs.size(); ++i) {
            double a0 = angs[i];
            double a1 = (i + 1 < angs.size()) ? angs[i + 1] : angs[0] + 2 * CV_PI;
            double gap = a1 - a0;
            if (gap > best_gap) { best_gap = gap; best_gap_end = a1; }
        }
        double arc_start = best_gap_end;
        double arc_end   = best_gap_end + (2 * CV_PI - best_gap);
        auto norm = [](double a) {
            while (a >   CV_PI) a -= 2 * CV_PI;
            while (a <= -CV_PI) a += 2 * CV_PI;
            return a;
        };
        r.theta_start_deg = norm(arc_start) * 180.0 / CV_PI;
        r.theta_end_deg   = norm(arc_end)   * 180.0 / CV_PI;
        r.length_px       = rr * (2 * CV_PI - best_gap);  // arc length

        r.inlier_count = (int)best_in.size();
        r.inlier_ratio = (double)r.inlier_count / r.total_hits;
        std::vector<double> res; res.reserve(best_in.size());
        for (int idx : best_in) {
            res.push_back(std::abs(std::hypot(r.hits[idx].pos.x - cx,
                                              r.hits[idx].pos.y - cy) - rr));
        }
        finalize_confidence(r, stab, best_in, res, thr);
        r.ok = true;
        return true;
    }

    // --- Ellipse fit (RANSAC + cv::fitEllipse LSQ) -----------------------

    static bool ellipse_from_points(const std::vector<cv::Point2d>& pts,
                                    double& cx, double& cy, double& a,
                                    double& b, double& rot_deg) {
        if (pts.size() < 5) return false;
        std::vector<cv::Point2f> ptsf;
        ptsf.reserve(pts.size());
        for (auto& p : pts) ptsf.emplace_back((float)p.x, (float)p.y);
        cv::RotatedRect rr;
        try { rr = cv::fitEllipse(ptsf); }
        catch (const cv::Exception&) { return false; }
        cx = rr.center.x; cy = rr.center.y;
        // Normalise so a is semi-major, b is semi-minor.
        double sa = rr.size.width * 0.5, sb = rr.size.height * 0.5;
        if (sa >= sb) { a = sa; b = sb; rot_deg = rr.angle; }
        else          { a = sb; b = sa; rot_deg = rr.angle + 90.0; }
        while (rot_deg >= 180) rot_deg -= 180;
        while (rot_deg <  0)   rot_deg += 180;
        return a > 0.5 && b > 0.5;
    }

    // Closest-point approximation: project a hit onto the ellipse via the
    // parametric angle heuristic (single Newton-like iteration). Accurate
    // enough for RANSAC inlier counting; final LSQ comes from cv::fitEllipse.
    static double point_ellipse_distance(double px, double py,
                                         double cx, double cy,
                                         double a, double b, double rot_rad) {
        double dx = px - cx, dy = py - cy;
        double cR = std::cos(-rot_rad), sR = std::sin(-rot_rad);
        double xl = dx * cR - dy * sR;
        double yl = dx * sR + dy * cR;
        double t  = std::atan2(yl * a, xl * b);
        double ex = a * std::cos(t);
        double ey = b * std::sin(t);
        return std::hypot(xl - ex, yl - ey);
    }

    static bool fit_ellipse_ransac(const Config& cfg, FitResult& r) {
        if (r.total_hits < 5) { r.fail_reason = "ellipse needs ≥5 hits"; return false; }
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> pick(0, r.total_hits - 1);
        const double thr = std::max(0.1, cfg.ransac_threshold_px);
        std::vector<int> best_in;
        const int iters = adaptive_iters(cfg.expected_outlier_rate, 5,
                                         cfg.ransac_iterations, 100);

        double best_score = 0;
        TopKRanker topk;
        for (int it = 0; it < iters; ++it) {
            std::vector<cv::Point2d> sample;
            std::vector<int> taken;
            taken.reserve(5);
            while ((int)sample.size() < 5) {
                int i = pick(rng);
                if (std::find(taken.begin(), taken.end(), i) != taken.end()) continue;
                taken.push_back(i);
                sample.push_back(r.hits[i].pos);
            }
            double cx, cy, a, b, rot;
            if (!ellipse_from_points(sample, cx, cy, a, b, rot)) continue;
            double rot_rad = rot * CV_PI / 180.0;
            std::vector<int> inl;
            double score = 0;
            for (int k = 0; k < r.total_hits; ++k) {
                double d = point_ellipse_distance(r.hits[k].pos.x, r.hits[k].pos.y,
                                                  cx, cy, a, b, rot_rad);
                if (d <= thr) {
                    inl.push_back(k);
                    score += cfg.ransac_weight_by_strength ? r.hits[k].strength : 1.0;
                }
            }
            topk.offer(score, inl);
            if (score > best_score) { best_score = score; best_in = std::move(inl); }
        }
        if (best_in.size() < 5) { r.fail_reason = "ransac found no consensus"; return false; }
        const double stab = stability_from_topk(topk);

        std::vector<cv::Point2d> pts; pts.reserve(best_in.size());
        for (int idx : best_in) { r.hits[idx].inlier = true; pts.push_back(r.hits[idx].pos); }
        double cx, cy, a, b, rot;
        if (!ellipse_from_points(pts, cx, cy, a, b, rot)) {
            r.fail_reason = "ellipse lsq failed"; return false;
        }
        r.cx = cx; r.cy = cy;
        r.a  = a;  r.b  = b;
        r.rotation_deg = rot;
        r.inlier_count = (int)best_in.size();
        r.inlier_ratio = (double)r.inlier_count / r.total_hits;
        {
            double rot_rad = rot * CV_PI / 180.0;
            std::vector<double> res; res.reserve(best_in.size());
            for (int idx : best_in) {
                res.push_back(point_ellipse_distance(
                    r.hits[idx].pos.x, r.hits[idx].pos.y, cx, cy, a, b, rot_rad));
            }
            finalize_confidence(r, stab, best_in, res, thr);
        }
        r.ok = true;
        return true;
    }

    // --- Polynomial primary fit (RANSAC + LSQ) ---------------------------
    //
    // Currently supports Line-mode regions only. Fits v = p(u) in the
    // region's local frame — u along the line normalised to [-1, 1], v the
    // signed perpendicular offset. RANSAC picks (degree+1) hits, solves
    // the Vandermonde system via SVD, then counts inliers by distance
    // from the fitted curve. Final LSQ refit through all inliers.

    static bool fit_polynomial_ransac(RegionMode mode, const LineRegion& ln,
                                      const Config& cfg, FitResult& r) {
        if (mode != RegionMode::Line) {
            r.fail_reason = "polynomial fit currently supports line-mode region only";
            return false;
        }
        if (ln.empty()) { r.fail_reason = "line region empty"; return false; }
        const int k = std::clamp(cfg.poly_degree > 0 ? cfg.poly_degree : 2, 1, 6);
        if (r.total_hits < k + 1) { r.fail_reason = "too few hits"; return false; }

        // Local frame for this region.
        const double dx = ln.p2x - ln.p1x, dy = ln.p2y - ln.p1y;
        const double L  = std::hypot(dx, dy);
        // Below 10 px the u-normalisation blows the slope scale up so much
        // that poly_max_slope_deg becomes meaningless; refuse the fit.
        if (L < 10) { r.fail_reason = "line region shorter than 10 px"; return false; }
        const cv::Point2d ex(dx / L, dy / L);
        const cv::Point2d ey(-ex.y, ex.x);
        const cv::Point2d c((ln.p1x + ln.p2x) * 0.5, (ln.p1y + ln.p2y) * 0.5);
        const double half = L * 0.5;

        // Pre-compute (u, v) for every hit in the line-local frame.
        std::vector<double> us(r.total_hits), vs(r.total_hits);
        for (int i = 0; i < r.total_hits; ++i) {
            double du = r.hits[i].pos.x - c.x, dv = r.hits[i].pos.y - c.y;
            us[i] = (du * ex.x + dv * ex.y) / half;
            vs[i] = du * ey.x + dv * ey.y;
        }

        auto fit_with = [&](const std::vector<int>& idx,
                            std::vector<double>& coeffs) -> bool {
            int n = (int)idx.size(), d = k + 1;
            if (n < d) return false;
            cv::Mat A(n, d, CV_64F), b(n, 1, CV_64F);
            for (int i = 0; i < n; ++i) {
                double p = 1.0;
                for (int j = 0; j < d; ++j) { A.at<double>(i, j) = p; p *= us[idx[i]]; }
                b.at<double>(i, 0) = vs[idx[i]];
            }
            cv::Mat x;
            if (!cv::solve(A, b, x, cv::DECOMP_SVD)) return false;
            coeffs.assign(d, 0.0);
            for (int j = 0; j < d; ++j) coeffs[j] = x.at<double>(j);
            return true;
        };
        auto eval = [](const std::vector<double>& co, double u) {
            double v = 0, p = 1;
            for (double a : co) { v += a * p; p *= u; }
            return v;
        };
        // Maximum |dv/dx| along the region (converted to pixel-per-pixel
        // slope using the region's half-length). Used to reject wild
        // candidates. Samples 64 points across u ∈ [-1, 1].
        auto max_abs_slope = [&](const std::vector<double>& co) {
            double max_s = 0;
            const int samples = 64;
            for (int i = 0; i <= samples; ++i) {
                double u = -1.0 + 2.0 * (double)i / samples;
                // dv/du = a1 + 2·a2·u + 3·a3·u² + …
                double dv_du = 0, p = 1;
                for (int j = 1; j < (int)co.size(); ++j) {
                    dv_du += j * co[j] * p;
                    p *= u;
                }
                double dv_dx = dv_du / half;          // chain-rule du/dx = 1/half
                max_s = std::max(max_s, std::abs(dv_dx));
            }
            return max_s;
        };
        const double slope_limit = (cfg.poly_max_slope_deg > 0.0 &&
                                    cfg.poly_max_slope_deg < 89.999)
            ? std::tan(cfg.poly_max_slope_deg * CV_PI / 180.0)
            : 0.0;                                    // 0 → disabled

        // RANSAC.
        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> pick(0, r.total_hits - 1);
        const double thr = std::max(0.1, cfg.ransac_threshold_px);
        std::vector<int> best_in;
        double best_score = 0;
        const int iters = adaptive_iters(cfg.expected_outlier_rate, k + 1,
                                         cfg.ransac_iterations, 30);
        int attempts = 0, slope_rejected = 0;
        TopKRanker topk;
        for (int it = 0; it < iters; ++it) {
            std::vector<int> sample, taken;
            taken.reserve(k + 1);
            while ((int)sample.size() < k + 1) {
                int i = pick(rng);
                if (std::find(taken.begin(), taken.end(), i) != taken.end()) continue;
                taken.push_back(i);
                sample.push_back(i);
            }
            std::vector<double> coeffs;
            if (!fit_with(sample, coeffs)) continue;
            ++attempts;
            // Reject wildly steep candidates before they even count inliers.
            if (slope_limit > 0 && max_abs_slope(coeffs) > slope_limit) {
                ++slope_rejected;
                continue;
            }
            std::vector<int> inl;
            double score = 0;
            for (int kk = 0; kk < r.total_hits; ++kk) {
                if (std::abs(vs[kk] - eval(coeffs, us[kk])) <= thr) {
                    inl.push_back(kk);
                    score += cfg.ransac_weight_by_strength ? r.hits[kk].strength : 1.0;
                }
            }
            topk.offer(score, inl);
            if (score > best_score) { best_score = score; best_in = std::move(inl); }
        }
        const double stab = stability_from_topk(topk);
        r.ransac_attempts      = attempts;
        r.slope_rejected_count = slope_rejected;
        if ((int)best_in.size() < k + 1) {
            // Distinguish "nothing in the data" from "constraint too tight".
            if (slope_rejected > 0 && slope_rejected == attempts) {
                r.fail_reason = "all candidates exceeded slope limit";
            } else if (slope_rejected > attempts / 2) {
                r.fail_reason = "ransac no consensus (slope limit pruned majority of candidates)";
            } else {
                r.fail_reason = "ransac found no consensus";
            }
            return false;
        }

        // Final LSQ refit through inliers.
        std::vector<double> final_coeffs;
        if (!fit_with(best_in, final_coeffs)) {
            r.fail_reason = "polynomial LSQ failed";
            return false;
        }
        // Safety: LSQ through more points can in principle drift outside
        // the slope envelope the candidates obeyed. Re-check and bail
        // rather than silently exceeding the user's constraint.
        if (slope_limit > 0 && max_abs_slope(final_coeffs) > slope_limit) {
            r.fail_reason = "LSQ refit exceeded slope limit";
            return false;
        }
        for (int idx : best_in) r.hits[idx].inlier = true;
        r.poly_coeffs = final_coeffs;

        // Expose endpoints of the fitted curve (u = ±1) so downstream
        // callers still get x1/y1/x2/y2 like the line model does.
        double v_start = eval(final_coeffs, -1.0);
        double v_end   = eval(final_coeffs,  1.0);
        r.p1 = c + ex * (-half) + ey * v_start;
        r.p2 = c + ex * ( half) + ey * v_end;
        r.length_px   = std::hypot(r.p2.x - r.p1.x, r.p2.y - r.p1.y);   // chord
        r.inlier_count = (int)best_in.size();
        r.inlier_ratio = (double)r.inlier_count / r.total_hits;
        {
            std::vector<double> res; res.reserve(best_in.size());
            for (int idx : best_in) {
                res.push_back(std::abs(vs[idx] - eval(final_coeffs, us[idx])));
            }
            finalize_confidence(r, stab, best_in, res, thr);
        }
        r.ok = true;
        return true;
    }

    // --- Polynomial residual filter --------------------------------------
    //
    // After the primary RANSAC+LSQ fit, each inlier has:
    //   t_i — parametric position along the primitive (projection for
    //         line; angle for circle). Normalised to [-1, 1].
    //   r_i — signed residual against the primitive (perpendicular offset
    //         for line; radial deviation for circle).
    // We fit r(t) ≈ a0 + a1*t + a2*t² + … + ak*t^k, then reject any hit
    // whose deviation from r(t_i) exceeds σ × std(deviations). The
    // primitive is then refit through the survivors.

    // Residual + parameter computation for each inlier.
    static void collect_residuals(const FitResult& r,
                                  std::vector<double>& ts,
                                  std::vector<double>& res,
                                  std::vector<int>& idx) {
        ts.clear(); res.clear(); idx.clear();
        if (r.model == FitModel::Line) {
            cv::Point2d dir(std::cos(r.angle_deg * CV_PI / 180.0),
                            std::sin(r.angle_deg * CV_PI / 180.0));
            cv::Point2d nrm(-dir.y, dir.x);
            cv::Point2d pt((r.p1.x + r.p2.x) * 0.5, (r.p1.y + r.p2.y) * 0.5);
            double tmin =  1e18, tmax = -1e18;
            for (size_t i = 0; i < r.hits.size(); ++i) if (r.hits[i].inlier) {
                double t = (r.hits[i].pos.x - pt.x) * dir.x +
                           (r.hits[i].pos.y - pt.y) * dir.y;
                if (t < tmin) tmin = t;
                if (t > tmax) tmax = t;
            }
            double half = 0.5 * std::max(1.0, tmax - tmin);
            double mid  = 0.5 * (tmax + tmin);
            for (size_t i = 0; i < r.hits.size(); ++i) if (r.hits[i].inlier) {
                cv::Point2d d(r.hits[i].pos.x - pt.x, r.hits[i].pos.y - pt.y);
                double t = (d.x * dir.x + d.y * dir.y - mid) / half;   // [-1, 1]
                double re = d.x * nrm.x + d.y * nrm.y;                  // signed perp
                ts.push_back(t); res.push_back(re); idx.push_back((int)i);
            }
        } else if (r.model == FitModel::Circle) {
            for (size_t i = 0; i < r.hits.size(); ++i) if (r.hits[i].inlier) {
                double dx = r.hits[i].pos.x - r.cx;
                double dy = r.hits[i].pos.y - r.cy;
                double ang = std::atan2(dy, dx);                        // [-π, π]
                double t   = ang / CV_PI;                               // [-1, 1]
                double re  = std::hypot(dx, dy) - r.r;                  // signed radial
                ts.push_back(t); res.push_back(re); idx.push_back((int)i);
            }
        }
    }

    // Least-squares polynomial fit. Returns coefficients [a0..ak].
    static std::vector<double> poly_fit(const std::vector<double>& ts,
                                        const std::vector<double>& res,
                                        int degree) {
        int n = (int)ts.size(), k = degree + 1;
        if (n < k) return {};
        cv::Mat A(n, k, CV_64F), b(n, 1, CV_64F);
        for (int i = 0; i < n; ++i) {
            double p = 1.0;
            for (int j = 0; j < k; ++j) { A.at<double>(i, j) = p; p *= ts[i]; }
            b.at<double>(i, 0) = res[i];
        }
        cv::Mat x;
        if (!cv::solve(A, b, x, cv::DECOMP_SVD)) return {};
        std::vector<double> coeffs(k);
        for (int j = 0; j < k; ++j) coeffs[j] = x.at<double>(j);
        return coeffs;
    }

    static double poly_eval(const std::vector<double>& coeffs, double t) {
        double v = 0, p = 1;
        for (double c : coeffs) { v += c * p; p *= t; }
        return v;
    }

    void apply_poly_filter(const Config& cfg, FitResult& r) const {
        std::vector<double> ts, res;
        std::vector<int>    idx;
        collect_residuals(r, ts, res, idx);
        if ((int)ts.size() <= cfg.poly_degree + 1) return;

        auto coeffs = poly_fit(ts, res, cfg.poly_degree);
        if (coeffs.empty()) return;

        // Deviation of each hit from the smooth residual model.
        std::vector<double> dev(ts.size());
        for (size_t i = 0; i < ts.size(); ++i)
            dev[i] = res[i] - poly_eval(coeffs, ts[i]);

        // Robust spread estimate via median absolute deviation — outliers
        // won't inflate σ the way plain std would, so the filter keeps its
        // teeth even when a meaningful fraction of the hits are bad.
        auto median_of = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0 : v[v.size() / 2];
        };
        double med = median_of(dev);
        std::vector<double> absdev(dev.size());
        for (size_t i = 0; i < dev.size(); ++i) absdev[i] = std::abs(dev[i] - med);
        double mad = median_of(absdev);
        double std_robust = mad * 1.4826;                   // Gaussian consistency
        r.residual_std = std_robust;
        if (std_robust < 1e-6) { r.poly_applied = true; return; }

        const double thr = std::max(1e-6, cfg.poly_reject_sigma) * std_robust;
        int rejected = 0;
        for (size_t i = 0; i < dev.size(); ++i) {
            if (std::abs(dev[i] - med) > thr) {
                r.hits[idx[i]].inlier = false;
                ++rejected;
            }
        }
        r.poly_rejected = rejected;
        r.poly_applied  = true;
        if (rejected == 0) return;

        // Refit the primitive through the now-cleaner inliers.
        std::vector<cv::Point2d> pts;
        for (const auto& h : r.hits) if (h.inlier) pts.push_back(h.pos);
        if ((int)pts.size() < (r.model == FitModel::Circle ? 3 : 2)) {
            // Too few left — revert to previous fit (keep metrics).
            return;
        }

        if (r.model == FitModel::Line) {
            cv::Vec4d lf;
            cv::fitLine(pts, lf, cv::DIST_L2, 0, 0.01, 0.01);
            cv::Point2d dir(lf[0], lf[1]), pt(lf[2], lf[3]);
            double tmin = 1e18, tmax = -1e18;
            for (const auto& q : pts) {
                double t = (q.x - pt.x) * dir.x + (q.y - pt.y) * dir.y;
                if (t < tmin) tmin = t;
                if (t > tmax) tmax = t;
            }
            r.p1 = pt + tmin * dir;
            r.p2 = pt + tmax * dir;
            r.length_px = std::hypot(r.p2.x - r.p1.x, r.p2.y - r.p1.y);
            double ang = std::atan2(r.p2.y - r.p1.y, r.p2.x - r.p1.x) * 180.0 / CV_PI;
            while (ang >   90) ang -= 180;
            while (ang <= -90) ang += 180;
            r.angle_deg = ang;
        } else if (r.model == FitModel::Circle) {
            double cx, cy, rr;
            if (circle_lsq(pts, cx, cy, rr)) { r.cx = cx; r.cy = cy; r.r = rr; }
        }

        r.inlier_count = (int)pts.size();
        r.inlier_ratio = r.total_hits > 0
            ? (double)r.inlier_count / r.total_hits : 0.0;
    }

    // --- drawing ---------------------------------------------------------

    static void draw_region_line(cv::Mat& rgb, const LineRegion& ln,
                                 const Config& cfg, const cv::Scalar& color) {
        if (ln.empty()) return;
        const double dx = ln.p2x - ln.p1x, dy = ln.p2y - ln.p1y;
        const double L  = std::hypot(dx, dy);
        const cv::Point2d ex(dx / L, dy / L);
        const cv::Point2d ey(-ex.y, ex.x);
        const cv::Point2d c((ln.p1x + ln.p2x) * 0.5, (ln.p1y + ln.p2y) * 0.5);

        const int segs = 24;
        cv::Point2d a(ln.p1x, ln.p1y), b(ln.p2x, ln.p2y);
        for (int i = 0; i < segs; i += 2) {
            cv::Point2d s1 = a + (b - a) * ((double)i       / segs);
            cv::Point2d s2 = a + (b - a) * ((double)(i + 1) / segs);
            cv::line(rgb, s1, s2, color, 1, cv::LINE_AA);
        }
        const int N = std::max(3, cfg.num_calipers);
        const double span = cfg.caliper_span;
        const double wid  = std::max(4.0, (double)cfg.caliper_width);
        for (int i = 0; i < N; ++i) {
            double u  = -L * 0.5 + (L * (i + 0.5)) / N;
            cv::Point2d cc = c + ex * u;
            cv::Point2d e_u = ex * (wid  * 0.5);
            cv::Point2d e_v = ey * (span * 0.5);
            cv::Point2d c0 = cc - e_u - e_v, c1 = cc + e_u - e_v;
            cv::Point2d c2 = cc + e_u + e_v, c3 = cc - e_u + e_v;
            cv::line(rgb, c0, c1, color, 1, cv::LINE_AA);
            cv::line(rgb, c1, c2, color, 1, cv::LINE_AA);
            cv::line(rgb, c2, c3, color, 1, cv::LINE_AA);
            cv::line(rgb, c3, c0, color, 1, cv::LINE_AA);
        }
        cv::circle(rgb, a, 4, color, -1, cv::LINE_AA);
        cv::circle(rgb, b, 4, color, -1, cv::LINE_AA);
        cv::circle(rgb, a, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::circle(rgb, b, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::Point2d tip = c + ey * (span * 0.5 - 4);
        cv::arrowedLine(rgb, c, tip, color, 1, cv::LINE_AA, 0, 0.25);
    }

    static void draw_region_arc(cv::Mat& rgb, const ArcRegion& arc,
                                const Config& cfg, const cv::Scalar& color) {
        if (arc.r <= 1) return;
        double t0   = arc.theta_start_deg * CV_PI / 180.0;
        double span = samples::ccw_span(t0, arc.theta_end_deg * CV_PI / 180.0);

        // Dashed arc along the nominal circle.
        const int segs = 48;
        for (int i = 0; i < segs; i += 2) {
            double ta = t0 + span * (double)i       / segs;
            double tb = t0 + span * (double)(i + 1) / segs;
            cv::Point2d a(arc.cx + arc.r * std::cos(ta), arc.cy + arc.r * std::sin(ta));
            cv::Point2d b(arc.cx + arc.r * std::cos(tb), arc.cy + arc.r * std::sin(tb));
            cv::line(rgb, a, b, color, 1, cv::LINE_AA);
        }

        // Caliper rectangles — each perpendicular to its radial direction.
        const int N = std::max(3, cfg.num_calipers);
        const double sp = cfg.caliper_span;
        const double wd = std::max(4.0, (double)cfg.caliper_width);
        for (int i = 0; i < N; ++i) {
            double t = t0 + span * (i + 0.5) / N;
            cv::Point2d dir(std::cos(t), std::sin(t));    // radial
            cv::Point2d tan(-dir.y, dir.x);               // tangent
            cv::Point2d cc(arc.cx + arc.r * dir.x, arc.cy + arc.r * dir.y);
            cv::Point2d eu = tan * (wd * 0.5);            // along tangent
            cv::Point2d ev = dir * (sp * 0.5);            // along radial (search)
            cv::Point2d c0 = cc - eu - ev, c1 = cc + eu - ev;
            cv::Point2d c2 = cc + eu + ev, c3 = cc - eu + ev;
            cv::line(rgb, c0, c1, color, 1, cv::LINE_AA);
            cv::line(rgb, c1, c2, color, 1, cv::LINE_AA);
            cv::line(rgb, c2, c3, color, 1, cv::LINE_AA);
            cv::line(rgb, c3, c0, color, 1, cv::LINE_AA);
        }

        // Endpoint handles on the arc.
        cv::Point2d a0(arc.cx + arc.r * std::cos(t0), arc.cy + arc.r * std::sin(t0));
        cv::Point2d a1(arc.cx + arc.r * std::cos(t0 + span), arc.cy + arc.r * std::sin(t0 + span));
        cv::Point2d cm(arc.cx, arc.cy);
        for (auto& p : {a0, a1}) {
            cv::circle(rgb, p, 4, color, -1, cv::LINE_AA);
            cv::circle(rgb, p, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        // Center marker.
        cv::drawMarker(rgb, cm, color, cv::MARKER_CROSS, 8, 1, cv::LINE_AA);
    }

    static void draw_region_ellipse_arc(cv::Mat& rgb, const EllipseArcRegion& el,
                                        const Config& cfg, const cv::Scalar& color) {
        if (el.a <= 1 || el.b <= 1) return;
        double t0   = el.theta_start_deg * CV_PI / 180.0;
        double span = samples::ccw_span(t0, el.theta_end_deg * CV_PI / 180.0);
        const double rot = el.rotation_deg * CV_PI / 180.0;
        const double cR = std::cos(rot), sR = std::sin(rot);

        auto on_ellipse = [&](double t) {
            double xl = el.a * std::cos(t);
            double yl = el.b * std::sin(t);
            return cv::Point2d(el.cx + xl * cR - yl * sR,
                               el.cy + xl * sR + yl * cR);
        };

        // Dashed ellipse outline.
        const int segs = 48;
        for (int i = 0; i < segs; i += 2) {
            double ta = t0 + span * (double)i       / segs;
            double tb = t0 + span * (double)(i + 1) / segs;
            cv::line(rgb, on_ellipse(ta), on_ellipse(tb), color, 1, cv::LINE_AA);
        }

        // Caliper rectangles along the ellipse, each with its normal axis.
        const int N = std::max(3, cfg.num_calipers);
        const double sp = cfg.caliper_span;
        const double wd = std::max(4.0, (double)cfg.caliper_width);
        for (int i = 0; i < N; ++i) {
            double t = t0 + span * (i + 0.5) / N;
            double ct = std::cos(t), st = std::sin(t);
            double xl = el.a * ct, yl = el.b * st;
            double nlx = ct / el.a, nly = st / el.b;
            double nln = std::hypot(nlx, nly);
            if (nln < 1e-9) continue;
            nlx /= nln; nly /= nln;
            cv::Point2d cc(el.cx + xl * cR - yl * sR,
                           el.cy + xl * sR + yl * cR);
            cv::Point2d dir(nlx * cR - nly * sR, nlx * sR + nly * cR);
            cv::Point2d tan(-dir.y, dir.x);
            cv::Point2d eu = tan * (wd * 0.5);
            cv::Point2d ev = dir * (sp * 0.5);
            cv::Point2d c0 = cc - eu - ev, c1 = cc + eu - ev;
            cv::Point2d c2 = cc + eu + ev, c3 = cc - eu + ev;
            cv::line(rgb, c0, c1, color, 1, cv::LINE_AA);
            cv::line(rgb, c1, c2, color, 1, cv::LINE_AA);
            cv::line(rgb, c2, c3, color, 1, cv::LINE_AA);
            cv::line(rgb, c3, c0, color, 1, cv::LINE_AA);
        }

        // Endpoint handles + centre marker.
        cv::Point2d a0 = on_ellipse(t0);
        cv::Point2d a1 = on_ellipse(t0 + span);
        for (auto& p : {a0, a1}) {
            cv::circle(rgb, p, 4, color, -1, cv::LINE_AA);
            cv::circle(rgb, p, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        cv::drawMarker(rgb, cv::Point2d(el.cx, el.cy), color,
                       cv::MARKER_CROSS, 8, 1, cv::LINE_AA);
    }

    static void draw_region(cv::Mat& rgb, RegionMode mode,
                            const LineRegion& ln, const ArcRegion& arc,
                            const EllipseArcRegion& el,
                            const Config& cfg, const cv::Scalar& color) {
        if      (mode == RegionMode::Line)       draw_region_line       (rgb, ln,  cfg, color);
        else if (mode == RegionMode::Arc)        draw_region_arc        (rgb, arc, cfg, color);
        else if (mode == RegionMode::EllipseArc) draw_region_ellipse_arc(rgb, el,  cfg, color);
    }

    cv::Mat draw(const cv::Mat& rgb_in, RegionMode mode, const LineRegion& ln,
                 const ArcRegion& arc, const EllipseArcRegion& el,
                 const Config& cfg, const FitResult& r) const {
        cv::Mat rgb = rgb_in.clone();
        draw_region(rgb, mode, ln, arc, el, cfg, cv::Scalar(80, 180, 255));

        for (const auto& h : r.hits) {
            cv::Scalar c = h.inlier ? cv::Scalar(57, 255, 20)
                                    : cv::Scalar(255, 80, 80);
            cv::circle(rgb, h.pos, 2, c, -1, cv::LINE_AA);
        }
        const cv::Scalar lc = r.ok
            ? (r.pass ? cv::Scalar(40, 220, 40) : cv::Scalar(235, 60, 60))
            : cv::Scalar(0, 0, 0);
        if (r.ok && r.model == FitModel::Line) {
            cv::line(rgb, r.p1, r.p2, lc, 2, cv::LINE_AA);
            char label[160];
            std::snprintf(label, sizeof(label),
                "%s  %.2fdeg  L=%.1fpx  in=%d/%d",
                r.pass ? "PASS" : "FAIL",
                r.angle_deg, r.length_px, r.inlier_count, r.total_hits);
            cv::Point2d mid((r.p1.x + r.p2.x) * 0.5, (r.p1.y + r.p2.y) * 0.5);
            cv::putText(rgb, label, mid + cv::Point2d(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, lc, 1, cv::LINE_AA);
        } else if (r.ok && r.model == FitModel::Circle) {
            double t0 = r.theta_start_deg * CV_PI / 180.0;
            double t1 = r.theta_end_deg   * CV_PI / 180.0;
            double span = t1 - t0;
            while (span >   2 * CV_PI) span -= 2 * CV_PI;
            while (span <= 0)          span += 2 * CV_PI;
            const int segs = 64;
            for (int i = 0; i < segs; ++i) {
                double ta = t0 + span * (double)i       / segs;
                double tb = t0 + span * (double)(i + 1) / segs;
                cv::Point2d a(r.cx + r.r * std::cos(ta), r.cy + r.r * std::sin(ta));
                cv::Point2d b(r.cx + r.r * std::cos(tb), r.cy + r.r * std::sin(tb));
                cv::line(rgb, a, b, lc, 2, cv::LINE_AA);
            }
            cv::drawMarker(rgb, cv::Point2d(r.cx, r.cy), lc,
                           cv::MARKER_CROSS, 10, 2, cv::LINE_AA);
            char label[160];
            std::snprintf(label, sizeof(label),
                "%s  c=(%.1f,%.1f)  r=%.2f  in=%d/%d",
                r.pass ? "PASS" : "FAIL",
                r.cx, r.cy, r.r, r.inlier_count, r.total_hits);
            cv::putText(rgb, label, cv::Point2d(r.cx + r.r + 6, r.cy),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, lc, 1, cv::LINE_AA);
        } else if (r.ok && r.model == FitModel::Polynomial &&
                   mode == RegionMode::Line && !ln.empty()) {
            const double dx = ln.p2x - ln.p1x, dy = ln.p2y - ln.p1y;
            const double L  = std::hypot(dx, dy);
            const cv::Point2d ex(dx / L, dy / L);
            const cv::Point2d ey(-ex.y, ex.x);
            const cv::Point2d c((ln.p1x + ln.p2x) * 0.5, (ln.p1y + ln.p2y) * 0.5);
            const double half = L * 0.5;
            auto eval = [](const std::vector<double>& co, double u) {
                double v = 0, p = 1;
                for (double a : co) { v += a * p; p *= u; }
                return v;
            };
            const int samples = 64;
            cv::Point2d prev;
            for (int i = 0; i <= samples; ++i) {
                double u  = -1.0 + 2.0 * (double)i / samples;
                double vv = eval(r.poly_coeffs, u);
                cv::Point2d pt = c + ex * (u * half) + ey * vv;
                if (i > 0) cv::line(rgb, prev, pt, lc, 2, cv::LINE_AA);
                prev = pt;
            }
            char plabel[160];
            std::snprintf(plabel, sizeof(plabel),
                "%s  poly deg=%d  in=%d/%d",
                r.pass ? "PASS" : "FAIL",
                (int)r.poly_coeffs.size() - 1,
                r.inlier_count, r.total_hits);
            cv::putText(rgb, plabel, r.p1 + cv::Point2d(6, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, lc, 1, cv::LINE_AA);
        } else if (r.ok && r.model == FitModel::Ellipse) {
            cv::ellipse(rgb, cv::Point2d(r.cx, r.cy),
                        cv::Size2d(r.a, r.b),
                        r.rotation_deg, 0, 360, lc, 2, cv::LINE_AA);
            cv::drawMarker(rgb, cv::Point2d(r.cx, r.cy), lc,
                           cv::MARKER_CROSS, 10, 2, cv::LINE_AA);
            char label[160];
            std::snprintf(label, sizeof(label),
                "%s  c=(%.1f,%.1f)  a=%.1f b=%.1f  rot=%.1f deg  in=%d/%d",
                r.pass ? "PASS" : "FAIL",
                r.cx, r.cy, r.a, r.b, r.rotation_deg,
                r.inlier_count, r.total_hits);
            cv::putText(rgb, label, cv::Point2d(r.cx + r.a + 6, r.cy),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, lc, 1, cv::LINE_AA);
        }
        return rgb;
    }

    // --- JSON helpers ----------------------------------------------------

    static std::string polarity_str(Polarity p) {
        return p == Polarity::DarkToBright ? "dark_to_bright"
             : p == Polarity::BrightToDark ? "bright_to_dark"
                                           : "any";
    }
    static std::string fit_model_str(FitModel f) {
        return f == FitModel::Circle     ? "circle"
             : f == FitModel::Ellipse    ? "ellipse"
             : f == FitModel::Polynomial ? "polynomial"
                                         : "line";
    }
    static std::string region_mode_str(RegionMode m) {
        return m == RegionMode::Arc         ? "arc"
             : m == RegionMode::EllipseArc  ? "ellipse_arc"
                                            : "line";
    }

    std::string build_status_json() const {
        std::lock_guard<std::mutex> lk(mu_);
        return build_def_json_locked();
    }
    std::string build_def_json_locked() const {
        auto root = xi::Json::object()
            .set("region_mode",         region_mode_str(region_mode_))
            .set("fit_model",           fit_model_str(cfg_.fit_model))
            .set("polarity",            polarity_str(cfg_.polarity))
            .set("num_calipers",        cfg_.num_calipers)
            .set("caliper_width",       cfg_.caliper_width)
            .set("caliper_span",        cfg_.caliper_span)
            .set("min_edge_strength",   cfg_.min_edge_strength)
            .set("top_n_per_caliper",     cfg_.top_n_per_caliper)
            .set("edge_min_separation_px", cfg_.edge_min_separation_px)
            .set("top_n_min_alpha",        cfg_.top_n_min_alpha)
            .set("ransac_threshold_px", cfg_.ransac_threshold_px)
            .set("ransac_iterations",   cfg_.ransac_iterations)
            .set("expected_outlier_rate", cfg_.expected_outlier_rate)
            .set("ransac_weight_by_strength", cfg_.ransac_weight_by_strength)
            .set("poly_enabled",        cfg_.poly_enabled)
            .set("poly_degree",         cfg_.poly_degree)
            .set("poly_reject_sigma",   cfg_.poly_reject_sigma)
            .set("poly_max_slope_deg",  cfg_.poly_max_slope_deg)
            .set("min_length_px",       cfg_.min_length_px)
            .set("min_inlier_ratio",    cfg_.min_inlier_ratio);

        root.set("line", xi::Json::object()
            .set("p1x", line_.p1x).set("p1y", line_.p1y)
            .set("p2x", line_.p2x).set("p2y", line_.p2y));
        root.set("arc", xi::Json::object()
            .set("cx", arc_.cx).set("cy", arc_.cy).set("r", arc_.r)
            .set("theta_start_deg", arc_.theta_start_deg)
            .set("theta_end_deg",   arc_.theta_end_deg));
        root.set("ellipse", xi::Json::object()
            .set("ecx", ellipse_.cx).set("ecy", ellipse_.cy)
            .set("ea",  ellipse_.a ).set("eb",  ellipse_.b)
            .set("rotation_deg",     ellipse_.rotation_deg)
            .set("etheta_start_deg", ellipse_.theta_start_deg)
            .set("etheta_end_deg",   ellipse_.theta_end_deg));

        root.set("found", last_result_.ok);
        root.set("pass",  last_result_.pass);
        root.set("fail_reason", last_result_.fail_reason);
        if (last_result_.ok) {
            root.set("angle",        last_result_.angle_deg);
            root.set("length",       last_result_.length_px);
            root.set("inlier_count", last_result_.inlier_count);
            root.set("total_hits",   last_result_.total_hits);
            root.set("inlier_ratio", last_result_.inlier_ratio);
            root.set("poly_applied", last_result_.poly_applied);
            root.set("poly_rejected", last_result_.poly_rejected);
            root.set("residual_std", last_result_.residual_std);
            root.set("confidence",         last_result_.confidence);
            root.set("stability",          last_result_.stability);
            root.set("residual_median_px", last_result_.residual_median_px);
            root.set("caliper_coverage",   last_result_.caliper_coverage);
            if (last_result_.model == FitModel::Line) {
                root.set("x1", last_result_.p1.x).set("y1", last_result_.p1.y);
                root.set("x2", last_result_.p2.x).set("y2", last_result_.p2.y);
            } else if (last_result_.model == FitModel::Circle) {
                root.set("cx", last_result_.cx).set("cy", last_result_.cy);
                root.set("radius", last_result_.r);
                root.set("theta_start_deg", last_result_.theta_start_deg);
                root.set("theta_end_deg",   last_result_.theta_end_deg);
            } else if (last_result_.model == FitModel::Ellipse) {
                root.set("cx", last_result_.cx).set("cy", last_result_.cy);
                root.set("semi_major", last_result_.a);
                root.set("semi_minor", last_result_.b);
                root.set("rotation_deg", last_result_.rotation_deg);
            } else if (last_result_.model == FitModel::Polynomial) {
                root.set("x1", last_result_.p1.x).set("y1", last_result_.p1.y);
                root.set("x2", last_result_.p2.x).set("y2", last_result_.p2.y);
                auto arr = xi::Json::array();
                for (double c : last_result_.poly_coeffs) arr.push(c);
                root.set("poly_coeffs", arr);
                root.set("poly_degree_fitted", (int)last_result_.poly_coeffs.size() - 1);
            }
        }
        // Emitted whether or not the fit succeeded — useful to diagnose
        // failures too.
        root.set("ransac_attempts",      last_result_.ransac_attempts);
        root.set("slope_rejected_count", last_result_.slope_rejected_count);
        return root.dump();
    }

    std::string build_preview_json() const {
        cv::Mat rgb; Config cfg; RegionMode mode;
        LineRegion ln; ArcRegion arc; EllipseArcRegion el;
        {
            std::lock_guard<std::mutex> lk(mu_);
            rgb = last_frame_rgb_.clone();
            cfg = cfg_; mode = region_mode_;
            ln  = line_; arc = arc_; el = ellipse_;
        }
        if (rgb.empty()) return xi::Json::object().set_null("preview").dump();
        draw_region(rgb, mode, ln, arc, el, cfg, cv::Scalar(80, 180, 255));
        cv::Mat bgr; cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        std::vector<uint8_t> jpg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
        if (!cv::imencode(".jpg", bgr, jpg, params))
            return xi::Json::object().set_null("preview").dump();
        return xi::Json::object()
            .set("preview_w", rgb.cols).set("preview_h", rgb.rows)
            .set("preview",   b64_encode(jpg.data(), jpg.size()))
            .dump();
    }

    std::string build_last_result_json() const {
        cv::Mat rgb; FitResult r;
        {
            std::lock_guard<std::mutex> lk(mu_);
            rgb = last_overlay_.clone();
            r   = last_result_;
        }
        if (rgb.empty()) return xi::Json::object().set_null("result_png").dump();
        cv::Mat bgr; cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        std::vector<uint8_t> png;
        cv::imencode(".png", bgr, png);
        auto root = xi::Json::object()
            .set("result_w", rgb.cols).set("result_h", rgb.rows)
            .set("result_png", b64_encode(png.data(), png.size()))
            .set("found", r.ok).set("pass", r.pass)
            .set("fail_reason", r.fail_reason)
            .set("model", fit_model_str(r.model));
        if (r.ok) {
            root.set("angle",        r.angle_deg);
            root.set("length",       r.length_px);
            root.set("inlier_count", r.inlier_count);
            root.set("total_hits",   r.total_hits);
            root.set("inlier_ratio", r.inlier_ratio);
            if (r.model == FitModel::Line) {
                root.set("x1", r.p1.x).set("y1", r.p1.y);
                root.set("x2", r.p2.x).set("y2", r.p2.y);
            } else if (r.model == FitModel::Circle) {
                root.set("cx", r.cx).set("cy", r.cy).set("radius", r.r);
                root.set("theta_start_deg", r.theta_start_deg);
                root.set("theta_end_deg",   r.theta_end_deg);
            } else if (r.model == FitModel::Ellipse) {
                root.set("cx", r.cx).set("cy", r.cy);
                root.set("semi_major", r.a).set("semi_minor", r.b);
                root.set("rotation_deg", r.rotation_deg);
            } else if (r.model == FitModel::Polynomial) {
                root.set("x1", r.p1.x).set("y1", r.p1.y);
                root.set("x2", r.p2.x).set("y2", r.p2.y);
                auto arr = xi::Json::array();
                for (double c : r.poly_coeffs) arr.push(c);
                root.set("poly_coeffs", arr);
                root.set("poly_degree_fitted", (int)r.poly_coeffs.size() - 1);
            }
            root.set("ransac_attempts",      r.ransac_attempts);
            root.set("slope_rejected_count", r.slope_rejected_count);
        }
        return root.dump();
    }

    // --- state -----------------------------------------------------------
    mutable std::mutex mu_;
    Config           cfg_{};
    RegionMode       region_mode_ = RegionMode::Line;
    LineRegion       line_{};
    ArcRegion        arc_{};
    EllipseArcRegion ellipse_{};
    cv::Mat          last_frame_rgb_;
    cv::Mat     last_frame_gray_;
    cv::Mat     last_overlay_;
    FitResult   last_result_{};
};

XI_PLUGIN_IMPL(PrimitiveFitting)
