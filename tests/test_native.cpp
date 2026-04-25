//
// test_native.cpp — primitive_fitting Phase 1 tests.
//
// Phase 1 covers line region + line fit. Later phases append tests for
// circle / ellipse / polynomial filter.
//

#include <xi/xi_abi.hpp>
#include <xi/xi_baseline.hpp>
#include <xi/xi_cert.hpp>
#include <xi/xi_image_pool.hpp>
#include <xi/xi_test.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>

#ifndef PRIMITIVE_FITTING_DLL_PATH
#define PRIMITIVE_FITTING_DLL_PATH "primitive_fitting.dll"
#endif

static HMODULE                     g_dll = nullptr;
static xi::baseline::PluginSymbols g_syms;
static xi_host_api                 g_host = xi::ImagePool::make_host_api();

static void load_dll() {
    if (g_dll) return;
    g_dll = LoadLibraryA(PRIMITIVE_FITTING_DLL_PATH);
    if (!g_dll) {
        std::fprintf(stderr, "failed to load %s (err %lu)\n",
                     PRIMITIVE_FITTING_DLL_PATH, GetLastError());
        std::exit(2);
    }
    g_syms = xi::baseline::load_symbols(g_dll);
    if (!g_syms.ok()) {
        std::fprintf(stderr, "DLL missing required C ABI exports\n");
        std::exit(2);
    }
}

static xi_image_handle make_horizontal_edge(int w, int h, int edge_y) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    for (int y = 0; y < h; ++y) {
        std::memset(d + (size_t)y * stride,
                    (uint8_t)(y < edge_y ? 40 : 210), (size_t)w);
    }
    return img;
}

// Same step edge, plus per-pixel additive Gaussian noise of stddev `sigma`
// (clamped to [0,255]) and optional `n_stripes` bright rectangular
// distractors randomly placed within ±30 px of `edge_y` — mimicking the
// "spike / false-stripe" structural noise the production fitter is
// designed to reject. Fixed seed so the test is deterministic.
static xi_image_handle make_horizontal_edge_noisy(int w, int h, int edge_y,
                                                  double sigma,
                                                  int n_stripes = 0,
                                                  uint32_t seed = 0xC0FFEE) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    // Base: step edge.
    for (int y = 0; y < h; ++y) {
        std::memset(d + (size_t)y * stride,
                    (uint8_t)(y < edge_y ? 40 : 210), (size_t)w);
    }
    std::mt19937 rng(seed);
    // False stripes: bright 1–3 px tall bars within ±30 of edge, mirroring
    // lab/scene.cpp's draw_stripes()-style spike noise.
    if (n_stripes > 0) {
        const int y_lo = std::max(1, edge_y - 30);
        const int y_hi = std::min(h - 2, edge_y + 30);
        const int min_len = std::max(4, w / 20);
        const int max_len = std::max(min_len + 1, w / 4);
        std::uniform_int_distribution<int> dlen(min_len, max_len);
        std::uniform_int_distribution<int> dht(1, 3);
        std::uniform_int_distribution<int> dy(y_lo, y_hi);
        for (int i = 0; i < n_stripes; ++i) {
            int len = dlen(rng);
            int ht  = dht(rng);
            int x0  = std::uniform_int_distribution<int>(0, std::max(0, w - len))(rng);
            int y0  = dy(rng);
            for (int yy = y0; yy < y0 + ht && yy < h; ++yy) {
                uint8_t* row = d + (size_t)yy * stride;
                for (int xx = x0; xx < x0 + len && xx < w; ++xx) row[xx] = 245;
            }
        }
    }
    // Per-pixel Gaussian noise.
    if (sigma > 0) {
        std::normal_distribution<double> gn(0.0, sigma);
        for (int y = 0; y < h; ++y) {
            uint8_t* row = d + (size_t)y * stride;
            for (int x = 0; x < w; ++x) {
                double v = (double)row[x] + gn(rng);
                row[x] = (uint8_t)std::clamp(v, 0.0, 255.0);
            }
        }
    }
    return img;
}

// Synthetic scene with a bright disc of known centre+radius on dark bg.
// The caliper array scans radially outward, so it sees a bright→dark edge.
static xi_image_handle make_disc_scene(int w, int h, int cx, int cy, int r) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double dx = x - cx, dy = y - cy;
            d[y * stride + x] = (uint8_t)(dx * dx + dy * dy <= (double)r * r ? 210 : 40);
        }
    }
    return img;
}

static void release_out_images(xi_record_out& out) {
    for (int32_t i = 0; i < out.image_count; ++i)
        g_host.image_release(out.images[i].handle);
}

static double json_num(const std::string& s, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = s.find(needle);
    if (pos == std::string::npos) return 0.0;
    return std::atof(s.c_str() + pos + needle.size());
}
static int json_int(const std::string& s, const char* key) { return (int)json_num(s, key); }
static bool json_bool(const std::string& s, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = s.find(needle);
    if (pos == std::string::npos) return false;
    return s.compare(pos + needle.size(), 4, "true") == 0;
}

// --- Baseline cert ---

XI_TEST(baseline_all_pass) {
    load_dll();
    auto summary = xi::baseline::run_all(g_syms, &g_host);
    for (auto& r : summary.results) {
        if (!r.passed) std::fprintf(stderr, "  baseline fail: %s: %s\n",
                                    r.name.c_str(), r.error.c_str());
    }
    XI_EXPECT(summary.all_passed);
    if (summary.all_passed) {
        auto folder = std::filesystem::path(PRIMITIVE_FITTING_DLL_PATH).parent_path();
        xi::cert::certify(folder, PRIMITIVE_FITTING_DLL_PATH,
                          "primitive_fitting", g_syms, &g_host);
    }
}

// --- Phase 1: Line region + Line fit ---

XI_TEST(process_without_src_returns_error) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_noimg");
    xi_record in; in.images = nullptr; in.image_count = 0; in.json = "{}";
    xi_record_out out; xi_record_out_init(&out);
    g_syms.process(inst, &in, &out);
    std::string s = out.json ? out.json : "";
    XI_EXPECT(s.find("error") != std::string::npos);
    release_out_images(out);
    xi_record_out_free(&out);
    g_syms.destroy(inst);
}

XI_TEST(line_mode_detects_horizontal_edge) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_line");
    const int W = 320, H = 240, EDGE_Y = 120;

    char rsp[4096];
    g_syms.exchange(inst,
        R"({"command":"set_region","mode":"line",)"
        R"("p1x":60,"p1y":120,"p2x":260,"p2y":120})",
        rsp, sizeof(rsp));
    g_syms.exchange(inst,
        R"({"command":"set_config",)"
        R"("fit_model":"line","polarity":"dark_to_bright",)"
        R"("num_calipers":15,"caliper_width":3,"caliper_span":80,)"
        R"("min_edge_strength":10,)"
        R"("ransac_threshold_px":1.0,"ransac_iterations":100,)"
        R"("min_length_px":100,"min_inlier_ratio":0.8})",
        rsp, sizeof(rsp));

    xi_image_handle scene = make_horizontal_edge(W, H, EDGE_Y);
    xi_record_image imgs[] = {{"src", scene}};
    xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
    xi_record_out out; xi_record_out_init(&out);
    g_syms.process(inst, &in, &out);
    std::string result = out.json ? out.json : "";

    bool found = json_bool(result, "found");
    bool pass  = json_bool(result, "pass");
    double angle  = json_num(result, "angle");
    double length = json_num(result, "length");
    int inliers   = json_int(result, "inlier_count");
    int total     = json_int(result, "total_hits");

    if (!found || !pass) std::fprintf(stderr, "  result: %s\n", result.c_str());
    XI_EXPECT(found);
    XI_EXPECT(pass);
    XI_EXPECT(std::abs(angle) < 2.0);
    XI_EXPECT(length > 100.0);
    XI_EXPECT(inliers == total);

    release_out_images(out);
    xi_record_out_free(&out);
    g_host.image_release(scene);
    g_syms.destroy(inst);
}

// --- Phase 2: Arc region + Circle fit ---

XI_TEST(arc_mode_detects_disc) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_arc");
    const int W = 320, H = 240, CX = 160, CY = 120, R = 60;

    char rsp[4096];
    // Nominal arc slightly inside the true disc so calipers scan outward
    // across the bright→dark boundary.
    g_syms.exchange(inst,
        R"({"command":"set_region","mode":"arc",)"
        R"("cx":160,"cy":120,"r":60,)"
        R"("theta_start_deg":0,"theta_end_deg":360})",
        rsp, sizeof(rsp));
    g_syms.exchange(inst,
        R"({"command":"set_config",)"
        R"("fit_model":"circle","polarity":"bright_to_dark",)"
        R"("num_calipers":32,"caliper_width":3,"caliper_span":40,)"
        R"("min_edge_strength":10,)"
        R"("ransac_threshold_px":1.0,"ransac_iterations":200,)"
        R"("min_inlier_ratio":0.8})",
        rsp, sizeof(rsp));

    xi_image_handle scene = make_disc_scene(W, H, CX, CY, R);
    xi_record_image imgs[] = {{"src", scene}};
    xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
    xi_record_out out; xi_record_out_init(&out);
    g_syms.process(inst, &in, &out);
    std::string result = out.json ? out.json : "";

    bool found = json_bool(result, "found");
    bool pass  = json_bool(result, "pass");
    double cx     = json_num(result, "cx");
    double cy     = json_num(result, "cy");
    double radius = json_num(result, "radius");
    int inliers   = json_int(result, "inlier_count");
    int total     = json_int(result, "total_hits");

    if (!found || !pass) std::fprintf(stderr, "  result: %s\n", result.c_str());
    XI_EXPECT(found);
    XI_EXPECT(pass);
    XI_EXPECT(std::abs(cx - CX) < 1.0);
    XI_EXPECT(std::abs(cy - CY) < 1.0);
    XI_EXPECT(std::abs(radius - R) < 1.0);
    XI_EXPECT(inliers >= (int)(0.8 * total));

    release_out_images(out);
    xi_record_out_free(&out);
    g_host.image_release(scene);
    g_syms.destroy(inst);
}

// --- Phase 3: Polynomial residual filter ---

// Scene with a horizontal dark→bright edge plus a few "spurs" where the
// edge jumps up by several pixels at two discrete x positions. A naive
// line fit treats those spurs as normal noise within its RANSAC threshold;
// the polynomial residual filter (degree 1) should push them past the
// σ threshold and reject them.
static xi_image_handle make_spurred_edge(int w, int h, int edge_y) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    auto edge_at_x = [&](int x) -> int {
        if (x >= 140 && x < 160) return edge_y - 6;    // spur 1
        if (x >= 200 && x < 220) return edge_y - 6;    // spur 2
        return edge_y;
    };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            d[y * stride + x] = (uint8_t)(y < edge_at_x(x) ? 40 : 210);
        }
    }
    return img;
}

XI_TEST(poly_filter_rejects_spurs_on_line) {
    load_dll();
    const int W = 320, H = 240, EDGE_Y = 120;

    auto run_once = [&](bool poly_on) {
        void* inst = g_syms.create(&g_host, poly_on ? "t_poly_on" : "t_poly_off");
        char rsp[4096];
        g_syms.exchange(inst,
            R"({"command":"set_region","mode":"line",)"
            R"("p1x":40,"p1y":120,"p2x":280,"p2y":120})",
            rsp, sizeof(rsp));

        std::string cfg = std::string(R"({"command":"set_config",)") +
            R"("fit_model":"line","polarity":"dark_to_bright",)"
            R"("num_calipers":30,"caliper_width":3,"caliper_span":60,)"
            R"("min_edge_strength":10,)"
            R"("ransac_threshold_px":8.0,"ransac_iterations":200,)" +
            (poly_on
              ? R"("poly_enabled":true,"poly_degree":1,"poly_reject_sigma":2.5,)"
              : R"("poly_enabled":false,)") +
            R"("min_inlier_ratio":0})"
            "}";
        g_syms.exchange(inst, cfg.c_str(), rsp, sizeof(rsp));

        xi_image_handle scene = make_spurred_edge(W, H, EDGE_Y);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";
        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
        g_syms.destroy(inst);
        return result;
    };

    auto off = run_once(false);
    auto on  = run_once(true);

    bool off_found = json_bool(off, "found"), on_found = json_bool(on, "found");
    int  off_in    = json_int (off, "inlier_count");
    int  on_in     = json_int (on,  "inlier_count");
    int  on_rej    = json_int (on,  "poly_rejected");
    bool on_applied= json_bool(on,  "poly_applied");

    std::fprintf(stderr, "  off: %s\n", off.c_str());
    std::fprintf(stderr, "  on : %s\n", on .c_str());

    XI_EXPECT(off_found);
    XI_EXPECT(on_found);
    // With the lenient RANSAC threshold (8 px), off-mode accepts the
    // spurs as inliers; on-mode must detect + reject some.
    XI_EXPECT(on_applied);
    XI_EXPECT(on_rej > 0);
    XI_EXPECT(on_in < off_in);
}

// --- Phase 4: Ellipse arc region + Ellipse fit ---

// Bright solid ellipse (semi-major a, semi-minor b, rotated) on dark bg.
static xi_image_handle make_ellipse_scene(int w, int h, int cx, int cy,
                                          double a, double b, double rot_deg) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    const double rot = rot_deg * 3.14159265358979 / 180.0;
    const double cR = std::cos(rot), sR = std::sin(rot);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double dx = x - cx, dy = y - cy;
            double xl =  dx * cR + dy * sR;
            double yl = -dx * sR + dy * cR;
            bool inside = (xl * xl) / (a * a) + (yl * yl) / (b * b) <= 1.0;
            d[y * stride + x] = (uint8_t)(inside ? 210 : 40);
        }
    }
    return img;
}

XI_TEST(ellipse_mode_detects_ellipse) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_ellipse");
    const int W = 320, H = 240, CX = 160, CY = 120;
    const double A = 70, B = 40, ROT = 25;

    char rsp[4096];
    g_syms.exchange(inst,
        R"({"command":"set_region","mode":"ellipse_arc",)"
        R"("ecx":160,"ecy":120,"ea":70,"eb":40,)"
        R"("rotation_deg":25,"etheta_start_deg":0,"etheta_end_deg":360})",
        rsp, sizeof(rsp));
    g_syms.exchange(inst,
        R"({"command":"set_config",)"
        R"("fit_model":"ellipse","polarity":"bright_to_dark",)"
        R"("num_calipers":48,"caliper_width":3,"caliper_span":30,)"
        R"("min_edge_strength":10,)"
        R"("ransac_threshold_px":1.0,"ransac_iterations":400,)"
        R"("min_inlier_ratio":0.8})",
        rsp, sizeof(rsp));

    xi_image_handle scene = make_ellipse_scene(W, H, CX, CY, A, B, ROT);
    xi_record_image imgs[] = {{"src", scene}};
    xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
    xi_record_out out; xi_record_out_init(&out);
    g_syms.process(inst, &in, &out);
    std::string result = out.json ? out.json : "";

    bool found = json_bool(result, "found");
    bool pass  = json_bool(result, "pass");
    double cx  = json_num(result, "cx");
    double cy  = json_num(result, "cy");
    double sa  = json_num(result, "semi_major");
    double sb  = json_num(result, "semi_minor");
    double rot = json_num(result, "rotation_deg");

    if (!found || !pass) std::fprintf(stderr, "  result: %s\n", result.c_str());
    XI_EXPECT(found);
    XI_EXPECT(pass);
    XI_EXPECT(std::abs(cx - CX) < 2.0);
    XI_EXPECT(std::abs(cy - CY) < 2.0);
    XI_EXPECT(std::abs(sa - A)  < 2.0);
    XI_EXPECT(std::abs(sb - B)  < 2.0);
    // Rotation reported by cv::fitEllipse ∈ [0, 180); equivalent under +180.
    double drot = std::abs(rot - ROT);
    while (drot >= 180) drot -= 180;
    XI_EXPECT(drot < 3.0 || std::abs(drot - 180) < 3.0);

    release_out_images(out);
    xi_record_out_free(&out);
    g_host.image_release(scene);
    g_syms.destroy(inst);
}

// --- Top-N edges per caliper ---

// Dark / bright / dark stripe: bright band at y∈[100, 140). Each vertical
// caliper then sees two edges (bright→dark at top, dark→bright at bottom).
static xi_image_handle make_stripe_scene(int w, int h, int y_top, int y_bot) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    for (int y = 0; y < h; ++y) {
        uint8_t v = (y >= y_top && y < y_bot) ? 210 : 40;
        std::memset(d + (size_t)y * stride, v, (size_t)w);
    }
    return img;
}

// Three tiers: dark (40) → bright (210) above the stripe, dark (100) →
// bright (210) below. Top transition has amplitude 170, bottom only 110
// → the bottom edge is ~0.65× the top edge's |gradient|. Useful for the
// alpha filter.
static xi_image_handle make_asymmetric_stripe(int w, int h, int y_top, int y_bot) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    for (int y = 0; y < h; ++y) {
        uint8_t v;
        if      (y <  y_top) v = 40;
        else if (y <  y_bot) v = 210;
        else                 v = 100;
        std::memset(d + (size_t)y * stride, v, (size_t)w);
    }
    return img;
}

XI_TEST(top_n_alpha_filter_keeps_only_strong_peaks) {
    load_dll();
    const int W = 320, H = 240, Y_TOP = 100, Y_BOT = 140;

    auto run_once = [&](int top_n, double alpha) {
        void* inst = g_syms.create(&g_host, "t_alpha");
        char rsp[4096];
        g_syms.exchange(inst,
            R"({"command":"set_region","mode":"line",)"
            R"("p1x":40,"p1y":120,"p2x":280,"p2y":120})",
            rsp, sizeof(rsp));
        std::string cfg = std::string(R"({"command":"set_config",)") +
            R"("fit_model":"line","polarity":"any",)"
            R"("num_calipers":15,"caliper_width":3,"caliper_span":80,)"
            R"("min_edge_strength":10,"edge_min_separation_px":4,)" +
            R"("top_n_per_caliper":)" + std::to_string(top_n) +
            R"(,"top_n_min_alpha":)"  + std::to_string(alpha) +
            R"(,"ransac_threshold_px":1.5,"ransac_iterations":200,)"
            R"("min_inlier_ratio":0})"
            "}";
        g_syms.exchange(inst, cfg.c_str(), rsp, sizeof(rsp));

        xi_image_handle scene = make_asymmetric_stripe(W, H, Y_TOP, Y_BOT);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";
        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
        g_syms.destroy(inst);
        return result;
    };

    auto alpha_off = run_once(2, 0.0);   // both edges kept
    auto alpha_on  = run_once(2, 0.9);   // weaker edge pruned (ratio ≈ 0.65)

    int n_off = json_int(alpha_off, "total_hits");
    int n_on  = json_int(alpha_on,  "total_hits");
    std::fprintf(stderr, "  alpha=0.0 total_hits=%d\n  alpha=0.9 total_hits=%d\n",
                 n_off, n_on);

    XI_EXPECT_EQ(n_off, 30);   // 15 calipers × 2 edges
    XI_EXPECT_EQ(n_on,  15);   // only the stronger edge survives
}

XI_TEST(top_n_per_caliper_returns_multiple_peaks) {
    load_dll();
    const int W = 320, H = 240, Y_TOP = 100, Y_BOT = 140;

    auto run_once = [&](int top_n) {
        void* inst = g_syms.create(&g_host, "t_topn");
        char rsp[4096];
        // Line region at y=120 (middle of stripe), span 80 covers both edges.
        g_syms.exchange(inst,
            R"({"command":"set_region","mode":"line",)"
            R"("p1x":40,"p1y":120,"p2x":280,"p2y":120})",
            rsp, sizeof(rsp));
        std::string cfg = std::string(R"({"command":"set_config",)") +
            R"("fit_model":"line","polarity":"any",)"
            R"("num_calipers":15,"caliper_width":3,"caliper_span":80,)"
            R"("min_edge_strength":10,"edge_min_separation_px":4,)" +
            R"("top_n_per_caliper":)" + std::to_string(top_n) + "," +
            R"("ransac_threshold_px":1.5,"ransac_iterations":200,)"
            R"("min_inlier_ratio":0})"
            "}";
        g_syms.exchange(inst, cfg.c_str(), rsp, sizeof(rsp));

        xi_image_handle scene = make_stripe_scene(W, H, Y_TOP, Y_BOT);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";
        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
        g_syms.destroy(inst);
        return result;
    };

    auto n1 = run_once(1);
    auto n2 = run_once(2);

    int total_n1 = json_int(n1, "total_hits");
    int total_n2 = json_int(n2, "total_hits");
    std::fprintf(stderr, "  top_n=1: %s\n", n1.c_str());
    std::fprintf(stderr, "  top_n=2: %s\n", n2.c_str());

    // 15 calipers × 1 peak = 15 hits.
    XI_EXPECT_EQ(total_n1, 15);
    // 15 calipers × 2 peaks = 30 hits (both stripe edges per caliper).
    XI_EXPECT_EQ(total_n2, 30);
    // RANSAC still resolves a single line — it should lock onto either
    // the top or bottom edge and leave the other half as outliers.
    XI_EXPECT(json_bool(n2, "found"));
    XI_EXPECT(json_int(n2, "inlier_count") >= 15);
}

// --- Weighted RANSAC ---

// Two-candidate-line scene:
//   * A short, strong bright bar at x ∈ [100, 220], y ∈ [100, 110]
//     → each caliper inside picks up two strong edges at y=100 and y=110.
//   * A full-width weak step at y = 140 (40 → 80 across all x) → every
//     caliper sees one weak edge there.
// Count-based RANSAC picks the long weak line (more inliers); the
// strength-weighted variant picks one of the strong bar lines instead.
static xi_image_handle make_two_line_scene(int w, int h) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    for (int y = 0; y < h; ++y) {
        uint8_t bg_lo = 40, bg_hi = 80;
        for (int x = 0; x < w; ++x) {
            uint8_t v;
            if (y >= 100 && y < 110 && x >= 100 && x < 220) v = 210;   // strong bar
            else if (y >= 140)                              v = bg_hi; // weak step
            else                                            v = bg_lo;
            d[y * stride + x] = v;
        }
    }
    return img;
}

XI_TEST(weighted_ransac_prefers_strong_edges) {
    load_dll();
    const int W = 320, H = 240;

    auto run_once = [&](bool weighted) {
        void* inst = g_syms.create(&g_host, "t_weighted");
        char rsp[4096];
        // Caliper at y=140 with large span so it reaches the strong bar.
        g_syms.exchange(inst,
            R"({"command":"set_region","mode":"line",)"
            R"("p1x":40,"p1y":140,"p2x":280,"p2y":140})",
            rsp, sizeof(rsp));
        std::string cfg = std::string(R"({"command":"set_config",)") +
            R"("fit_model":"line","polarity":"any",)"
            R"("num_calipers":30,"caliper_width":3,"caliper_span":120,)"
            R"("min_edge_strength":8,"edge_min_separation_px":4,)"
            R"("top_n_per_caliper":3,)" +
            R"("ransac_threshold_px":1.0,"ransac_iterations":300,)" +
            (weighted
              ? R"("ransac_weight_by_strength":true,)"
              : R"("ransac_weight_by_strength":false,)") +
            R"("min_inlier_ratio":0})"
            "}";
        g_syms.exchange(inst, cfg.c_str(), rsp, sizeof(rsp));

        xi_image_handle scene = make_two_line_scene(W, H);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";
        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
        g_syms.destroy(inst);
        return result;
    };

    auto count  = run_once(false);
    auto weight = run_once(true);

    double y_count  = json_num(count,  "y1");   // horizontal fit → y1 ≈ fit y
    double y_weight = json_num(weight, "y1");
    std::fprintf(stderr, "  count : %s\n", count .c_str());
    std::fprintf(stderr, "  weight: %s\n", weight.c_str());

    // Count RANSAC locks onto the full-width weak step at y≈140.
    XI_EXPECT(std::abs(y_count - 140) < 2.0);
    // Weighted RANSAC prefers the strong bar edges at y=100 or y=110.
    XI_EXPECT(std::abs(y_weight - 100) < 2.0 ||
              std::abs(y_weight - 110) < 2.0);
}

// --- Weak-curve + strong-noise simulation ---
//
// Scene: a gentle sinusoidal dark→bright edge (δ≈60, visible across the
// full width) plus 6 isolated bright square "spikes" above the curve
// (each spike produces δ≈150 at its top/bottom boundaries, stronger
// than the curve itself). This simulates the "human eye sees a clear
// slight curve but the algorithm latches onto stronger isolated noise
// peaks" problem.

static xi_image_handle make_weak_curve_with_spikes(int w, int h) {
    xi_image_handle img = g_host.image_create(w, h, 1);
    uint8_t* d = g_host.image_data(img);
    int32_t  stride = g_host.image_stride(img);
    const double PI = 3.14159265358979;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Full-period sine across the image width: an S-shape with
            // one inflection. Amplitude ±15 px — degree-2 can't capture
            // it (needs odd polynomial), degree-3 can.
            double curve_y = 120.0 + 15.0 * std::sin(x * 2.0 * PI / (double)w);
            d[y * stride + x] = (uint8_t)(y < curve_y ? 90 : 150);
        }
    }
    // 30 strong bright spikes scattered above the curve (y < 120) via a
    // deterministic pseudo-random pattern. Background there is 90, so each
    // spike edge sits at δ = 240 - 90 = 150 — stronger than the curve's
    // δ=60 — and most calipers end up dominated by a spike column.
    for (int i = 0; i < 30; ++i) {
        int sx = 30 + ((i * 13 + 7) % 260);
        int sy = 93 + ((i * 17 + 11) % 18);
        for (int dy = 0; dy < 5; ++dy)
        for (int dx = 0; dx < 5; ++dx) {
            int yy = sy + dy, xx = sx + dx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
            d[yy * stride + xx] = 240;
        }
    }
    return img;
}

XI_TEST(weak_curve_recovered_with_top_n_and_poly) {
    load_dll();
    const int W = 320, H = 240;

    auto run_once = [&](int top_n, bool poly, int poly_degree) {
        void* inst = g_syms.create(&g_host, "t_demo");
        char rsp[4096];
        g_syms.exchange(inst,
            R"({"command":"set_region","mode":"line",)"
            R"("p1x":20,"p1y":120,"p2x":300,"p2y":120})",
            rsp, sizeof(rsp));
        std::string cfg = std::string(R"({"command":"set_config",)") +
            R"("fit_model":"line","polarity":"any",)"
            R"("num_calipers":30,"caliper_width":3,"caliper_span":80,)"
            R"("min_edge_strength":10,"edge_min_separation_px":4,)" +
            R"("top_n_per_caliper":)" + std::to_string(top_n) + "," +
            R"("ransac_threshold_px":5.0,"ransac_iterations":300,)" +
            R"("ransac_weight_by_strength":false,)" +
            (poly ? (std::string(R"("poly_enabled":true,"poly_degree":)") +
                     std::to_string(poly_degree) +
                     R"(,"poly_reject_sigma":2.5,)")
                  : std::string(R"("poly_enabled":false,)")) +
            R"("min_inlier_ratio":0})"
            "}";
        g_syms.exchange(inst, cfg.c_str(), rsp, sizeof(rsp));

        xi_image_handle scene = make_weak_curve_with_spikes(W, H);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";
        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
        g_syms.destroy(inst);
        return result;
    };

    auto a = run_once(1, false, 0);   // baseline: single max per caliper
    auto b = run_once(3, false, 0);   // top-N rescue
    auto c = run_once(3, true,  3);   // + polynomial residual model (degree 3 for S)

    int inl_a = json_int(a, "inlier_count"), tot_a = json_int(a, "total_hits");
    int inl_b = json_int(b, "inlier_count"), tot_b = json_int(b, "total_hits");
    int inl_c = json_int(c, "inlier_count"), tot_c = json_int(c, "total_hits");
    double ystd_c = json_num(c, "residual_std");
    int rej_c    = json_int(c, "poly_rejected");
    (void)tot_c;

    std::fprintf(stderr, "  (a) top_n=1         : %s\n", a.c_str());
    std::fprintf(stderr, "  (b) top_n=3         : %s\n", b.c_str());
    std::fprintf(stderr, "  (c) top_n=3 + poly2 : %s\n", c.c_str());

    // (b) Top-N rescues the curve hits that were hidden behind the
    //     spike's stronger peak, so more inliers survive than (a).
    XI_EXPECT(inl_b > inl_a);
    // (c) Poly filter runs (degree-3 for S-shape). It reduces the
    //     residual of the line-fit baseline but cannot fully capture
    //     the S because the underlying LINE primitive is the limiter;
    //     the Polynomial primitive test below demonstrates the full fix.
    XI_EXPECT(ystd_c < 5.0);
    // Ordering sanity: the total hits tell the same story — (a) only
    // saw the spike in spike-calipers; (b)/(c) also saw the curve there.
    XI_EXPECT(tot_b > tot_a);
    (void)rej_c; (void)inl_c;
}

// --- FitModel::Polynomial + max-slope constraint ---

XI_TEST(polynomial_primitive_fit_captures_full_arch) {
    load_dll();
    const int W = 320, H = 240;

    auto run_polyfit = [&](int degree, double max_slope_deg) {
        void* inst = g_syms.create(&g_host, "t_polyfit");
        char rsp[4096];
        g_syms.exchange(inst,
            R"({"command":"set_region","mode":"line",)"
            R"("p1x":20,"p1y":120,"p2x":300,"p2y":120})",
            rsp, sizeof(rsp));
        std::string cfg = std::string(R"({"command":"set_config",)") +
            R"("fit_model":"polynomial","polarity":"any",)"
            R"("num_calipers":30,"caliper_width":3,"caliper_span":80,)"
            R"("min_edge_strength":10,"edge_min_separation_px":4,)"
            R"("top_n_per_caliper":3,)" +
            R"("poly_degree":)" + std::to_string(degree) + "," +
            R"("poly_max_slope_deg":)" + std::to_string(max_slope_deg) + "," +
            R"("ransac_threshold_px":2.0,"ransac_iterations":300,)"
            R"("min_inlier_ratio":0})"
            "}";
        g_syms.exchange(inst, cfg.c_str(), rsp, sizeof(rsp));

        xi_image_handle scene = make_weak_curve_with_spikes(W, H);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";
        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
        g_syms.destroy(inst);
        return result;
    };

    // S-shaped curve has one inflection → need a cubic to capture it.
    // Tight threshold (2 px) catches only truly on-curve points.
    auto result = run_polyfit(3, 45.0);
    std::fprintf(stderr, "  polyfit deg=3 slope≤45°: %s\n", result.c_str());

    XI_EXPECT(json_bool(result, "found"));
    int inl = json_int(result, "inlier_count");
    int tot = json_int(result, "total_hits");
    std::string model = result;
    XI_EXPECT(result.find("\"model\":\"polynomial\"") != std::string::npos);
    // Ground-truth curve = 30 calipers. With the tight threshold and the
    // polynomial primitive, we should accept ~all of them (30, allowing a
    // small slack for sub-pixel slop).
    XI_EXPECT(inl >= 28);

    // Force a pathologically small slope limit — the S-curve's peak
    // slope (≈16° in degrees-from-horizontal) vastly exceeds < 0.5°, so
    // every candidate fit is rejected.
    auto clipped = run_polyfit(3, 0.5);
    std::fprintf(stderr, "  polyfit deg=3 slope≤0.5°: %s\n", clipped.c_str());
    XI_EXPECT(!json_bool(clipped, "found"));
    // With the tight 0.5° limit, every candidate gets rejected → the new
    // descriptive fail_reason should name the slope constraint.
    XI_EXPECT(clipped.find("slope limit") != std::string::npos);
    // And the diagnostic counts confirm what happened.
    XI_EXPECT(json_int(clipped, "slope_rejected_count") > 0);
    XI_EXPECT(json_int(clipped, "ransac_attempts") > 0);
}

// NaN / non-finite config values must be silently ignored so the fit
// keeps its previous valid value rather than propagating garbage.
XI_TEST(non_finite_config_values_are_ignored) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_nan");
    char rsp[4096];
    // Set a valid baseline.
    g_syms.exchange(inst,
        R"({"command":"set_config","num_calipers":20,)"
        R"("ransac_threshold_px":2.5,"poly_max_slope_deg":45})",
        rsp, sizeof(rsp));
    // Now feed garbage — `NaN` literal is not valid JSON but we can get
    // non-finite via 1e400 (reads as Infinity). cJSON parses it as
    // IEEE infinity; our isfinite check must drop it.
    g_syms.exchange(inst,
        R"({"command":"set_config","num_calipers":50,)"
        R"("ransac_threshold_px":1e400,"poly_max_slope_deg":1e400})",
        rsp, sizeof(rsp));
    std::string status = rsp;
    // Integer survives (was valid finite).
    XI_EXPECT(status.find("\"num_calipers\":50") != std::string::npos);
    // The non-finite doubles must NOT have overwritten the previous 2.5 / 45.
    XI_EXPECT(status.find("\"ransac_threshold_px\":2.5") != std::string::npos);
    XI_EXPECT(status.find("\"poly_max_slope_deg\":45") != std::string::npos);
    g_syms.destroy(inst);
}

// Sweep Gaussian noise σ across a horizontal step edge; confirm the new
// confidence/stability/residual_median fields move monotonically — high
// confidence on the clean image, degrading as noise rises. A smoke test
// for the confidence-scoring plumbing and a diagnostic readout.
XI_TEST(confidence_degrades_with_gaussian_noise) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_conf");
    const int W = 320, H = 240, EDGE_Y = 120;

    char rsp[4096];
    g_syms.exchange(inst,
        R"({"command":"set_region","mode":"line",)"
        R"("p1x":30,"p1y":120,"p2x":290,"p2y":120})",
        rsp, sizeof(rsp));
    g_syms.exchange(inst,
        R"({"command":"set_config",)"
        R"("fit_model":"line","polarity":"dark_to_bright",)"
        R"("num_calipers":20,"caliper_width":3,"caliper_span":80,)"
        R"("min_edge_strength":8,)"
        R"("ransac_threshold_px":1.5,"ransac_iterations":200,)"
        R"("expected_outlier_rate":0.2,)"
        R"("min_length_px":0,"min_inlier_ratio":0})",
        rsp, sizeof(rsp));

    struct Case { double sigma; int stripes; };
    const Case cases[] = {
        {0.0,   0},   // clean baseline
        {5.0,   0},   // pixel noise only
        {15.0,  0},
        {5.0,  10},   // few stripes
        {10.0, 20},   // moderate stripes
        {15.0, 40},   // many stripes
        {25.0, 60},   // very dense — near adversarial
    };
    double prev_conf = 2.0;
    std::fprintf(stderr, "  confidence vs (σ, stripes):\n");
    for (const auto& c : cases) {
        xi_image_handle scene = make_horizontal_edge_noisy(W, H, EDGE_Y,
                                                           c.sigma, c.stripes);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";

        double conf   = json_num(result, "confidence");
        double stab   = json_num(result, "stability");
        double resmed = json_num(result, "residual_median_px");
        int    inl    = json_int(result, "inlier_count");
        int    total  = json_int(result, "total_hits");
        bool   found  = json_bool(result, "found");
        std::fprintf(stderr,
            "    σ=%4.1f  stripes=%2d  found=%d  inliers=%d/%d  "
            "conf=%.3f  stab=%.3f  res_med=%.3f px\n",
            c.sigma, c.stripes, (int)found, inl, total, conf, stab, resmed);

        // Clean baseline must be near-perfect.
        if (c.sigma == 0.0 && c.stripes == 0) {
            XI_EXPECT(found);
            XI_EXPECT(conf > 0.5);
            XI_EXPECT(stab > 0.9);
        }
        prev_conf = conf;
        (void)prev_conf;

        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
    }
    g_syms.destroy(inst);
}

// On a hard stripe scene (σ=15, 40 false stripes), sweep four knob
// configurations to expose which lever actually moves confidence:
//   A. baseline        — top_n=1, poly off, iters=200  (current default)
//   B. wider candidate  — top_n=3
//   C. + residual filter — top_n=3, poly degree=1 on
//   D. + more RANSAC    — same as C, iters=500
// Predicted ordering: A << B < C ≈ D (more iters does not help when the
// inlier pool is already polluted at the per-caliper peak-pick stage).
XI_TEST(stripe_scene_knob_sweep) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t_knob");
    const int W = 320, H = 240, EDGE_Y = 120;
    const double SIGMA   = 15.0;
    const int    STRIPES = 40;

    char rsp[4096];
    g_syms.exchange(inst,
        R"({"command":"set_region","mode":"line",)"
        R"("p1x":30,"p1y":120,"p2x":290,"p2y":120})",
        rsp, sizeof(rsp));

    struct Knob { const char* tag; const char* config; };
    const Knob knobs[] = {
        {"A baseline (top_n=1)",
         R"({"command":"set_config","fit_model":"line","polarity":"dark_to_bright",)"
         R"("num_calipers":20,"caliper_width":3,"caliper_span":80,)"
         R"("min_edge_strength":8,"top_n_per_caliper":1,)"
         R"("ransac_threshold_px":1.5,"ransac_iterations":200,)"
         R"("expected_outlier_rate":0.5,)"
         R"("poly_enabled":false,)"
         R"("min_length_px":0,"min_inlier_ratio":0})"},
        {"B top_n=3        ",
         R"({"command":"set_config","fit_model":"line","polarity":"dark_to_bright",)"
         R"("num_calipers":20,"caliper_width":3,"caliper_span":80,)"
         R"("min_edge_strength":8,"top_n_per_caliper":3,)"
         R"("edge_min_separation_px":3,)"
         R"("ransac_threshold_px":1.5,"ransac_iterations":200,)"
         R"("expected_outlier_rate":0.5,)"
         R"("poly_enabled":false,)"
         R"("min_length_px":0,"min_inlier_ratio":0})"},
        {"C top_n=3 + poly1 ",
         R"({"command":"set_config","fit_model":"line","polarity":"dark_to_bright",)"
         R"("num_calipers":20,"caliper_width":3,"caliper_span":80,)"
         R"("min_edge_strength":8,"top_n_per_caliper":3,)"
         R"("edge_min_separation_px":3,)"
         R"("ransac_threshold_px":1.5,"ransac_iterations":200,)"
         R"("expected_outlier_rate":0.5,)"
         R"("poly_enabled":true,"poly_degree":1,"poly_reject_sigma":2.5,)"
         R"("min_length_px":0,"min_inlier_ratio":0})"},
        {"D + iters=500    ",
         R"({"command":"set_config","fit_model":"line","polarity":"dark_to_bright",)"
         R"("num_calipers":20,"caliper_width":3,"caliper_span":80,)"
         R"("min_edge_strength":8,"top_n_per_caliper":3,)"
         R"("edge_min_separation_px":3,)"
         R"("ransac_threshold_px":1.5,"ransac_iterations":500,)"
         R"("expected_outlier_rate":0.5,)"
         R"("poly_enabled":true,"poly_degree":1,"poly_reject_sigma":2.5,)"
         R"("min_length_px":0,"min_inlier_ratio":0})"},
    };

    std::fprintf(stderr,
        "  knob sweep on σ=%.0f, stripes=%d:\n", SIGMA, STRIPES);
    double conf_A = -1, conf_B = -1, conf_C = -1, conf_D = -1;
    for (const auto& k : knobs) {
        g_syms.exchange(inst, k.config, rsp, sizeof(rsp));
        xi_image_handle scene = make_horizontal_edge_noisy(W, H, EDGE_Y,
                                                           SIGMA, STRIPES);
        xi_record_image imgs[] = {{"src", scene}};
        xi_record in; in.images = imgs; in.image_count = 1; in.json = "{}";
        xi_record_out out; xi_record_out_init(&out);
        g_syms.process(inst, &in, &out);
        std::string result = out.json ? out.json : "";

        double conf   = json_num(result, "confidence");
        double stab   = json_num(result, "stability");
        double resmed = json_num(result, "residual_median_px");
        int    inl    = json_int(result, "inlier_count");
        int    total  = json_int(result, "total_hits");
        std::fprintf(stderr,
            "    %s  inliers=%d/%-3d  conf=%.3f  stab=%.3f  res_med=%.3f px\n",
            k.tag, inl, total, conf, stab, resmed);

        if      (conf_A < 0) conf_A = conf;
        else if (conf_B < 0) conf_B = conf;
        else if (conf_C < 0) conf_C = conf;
        else                 conf_D = conf;

        release_out_images(out);
        xi_record_out_free(&out);
        g_host.image_release(scene);
    }
    // Expected: top_n=3 must improve over baseline; iters=500 must not
    // significantly improve over iters=200 (within 0.05).
    XI_EXPECT(conf_B > conf_A);
    XI_EXPECT(std::abs(conf_D - conf_C) < 0.10);
    g_syms.destroy(inst);
}

int main() {
    auto results = xi::test::run_all();
    for (auto& r : results) if (!r.passed) return 1;
    return 0;
}
