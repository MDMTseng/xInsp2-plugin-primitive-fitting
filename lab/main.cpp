//
// main.cpp — batch benchmark harness.
//
// Usage: lab [--seeds N] [--outdir path]
//   Runs every registered algorithm on N randomly-generated scenes
//   (default N = 50) and reports distribution statistics per algorithm:
//   mean / median / p90 RMS, mean coverage, mean runtime, outlier-scene
//   rate (scenes where the algorithm produced ≥ 5 outliers).
//
// A handful of representative overlays are saved to <outdir>/samples/
// for visual sanity checking. RESULTS.md is written alongside the exe.
//

#include "common.hpp"
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>

namespace fs = std::filesystem;

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo  = (size_t)std::floor(idx);
    size_t hi  = (size_t)std::ceil (idx);
    double t   = idx - lo;
    return v[lo] * (1.0 - t) + v[hi] * t;
}
static double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0; for (double x : v) s += x;
    return s / v.size();
}

struct Agg {
    std::string name;
    std::vector<double> rms, coverage, runtime;
    std::vector<int>    outlier_scenes;   // 1 if outlier count ≥5
    std::vector<int>    no_detect_scenes; // 1 if total_points = 0
};

int main(int argc, char** argv) {
    int  seeds  = 50;
    lab::NoiseLevel level = lab::NoiseLevel::Normal;
    bool dashed = false;
    bool bumpy  = false;
    fs::path out_dir = fs::path(argv[0]).parent_path() / "results";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) {
            seeds = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--outdir") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (!std::strcmp(argv[i], "--harsh")) {
            level = lab::NoiseLevel::Harsh;
        } else if (!std::strcmp(argv[i], "--low-noise")) {
            level = lab::NoiseLevel::Low;
        } else if (!std::strcmp(argv[i], "--no-noise")) {
            level = lab::NoiseLevel::None;
        } else if (!std::strcmp(argv[i], "--dashed")) {
            dashed = true;
        } else if (!std::strcmp(argv[i], "--bumpy")) {
            bumpy = true;
        }
    }
    const char* level_name = (level == lab::NoiseLevel::Harsh) ? " (harsh)"
                           : (level == lab::NoiseLevel::Low)   ? " (low-noise, stripes only)"
                           : (level == lab::NoiseLevel::None)  ? " (no pixel noise, stripes only)"
                           :                                     "";
    fs::create_directories(out_dir);
    fs::create_directories(out_dir / "samples");

    struct Case { const char* name; lab::Algo fn; };
    const std::vector<Case> cases = {
        {"naive_threshold",  lab::detect_naive},
        {"dp_scanline",      lab::detect_dp},
        {"dijkstra_path",    lab::detect_dijkstra},
        {"tensor_voting",    lab::detect_tensor_voting},
        {"caliper_ransac",   lab::detect_caliper_ransac},
        {"cr_A_baseline",       lab::detect_caliper_ransac_A},
        {"cr_B_cluster_stop",   lab::detect_caliper_ransac_B},
        {"cr_C_sprt",           lab::detect_caliper_ransac_C},
        {"cr_D_combined+tukey", lab::detect_caliper_ransac_D},
        {"caliper_dp",          lab::detect_caliper_dp},
        {"caliper_dp_poly",     lab::detect_caliper_dp_poly},
        {"subregion_tv_peaks",  lab::detect_subregion_tv_peaks},
        {"subregion_tv_band",   lab::detect_subregion_tv_band},
        {"subregion_tv_strips", lab::detect_subregion_tv_strips},
        {"subregion_dp_strips", lab::detect_subregion_dp_strips},
        {"spline_knot_dp",      lab::detect_spline_knot_dp},
        {"cpoly_grid",          lab::detect_constrained_grid},
        {"cpoly_ransac",        lab::detect_constrained_ransac},
        {"cpoly_knot_dp",       lab::detect_constrained_knot_dp},
    };
    std::vector<Agg> aggs(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) aggs[i].name = cases[i].name;

    // Pre-pick a handful of seeds whose overlays we'll save for sanity.
    std::vector<int> sample_seeds = {0, 1, 2, 3, 5, 7, 13, 21, 42, 66, 77, 99};

    std::printf("Running %d random scenes × %zu algorithms%s%s%s …\n",
                seeds, cases.size(), level_name,
                dashed ? " [dashed edge]" : "",
                bumpy  ? " [bumpy edge]"  : "");
    for (int s = 0; s < seeds; ++s) {
        auto rs = lab::make_random_scene(s, level, dashed, bumpy);
        // Save the raw (unmarked) scene for sample seeds so the input
        // can be eyeballed alongside the overlays.
        if (std::find(sample_seeds.begin(), sample_seeds.end(), s) != sample_seeds.end()) {
            char raw_name[64];
            std::snprintf(raw_name, sizeof(raw_name), "seed%03d_raw.png", s);
            fs::create_directories(out_dir / "samples");
            cv::imwrite((out_dir / "samples" / raw_name).string(), rs.image);
        }
        for (size_t i = 0; i < cases.size(); ++i) {
            std::vector<cv::Point2d> hits;
            double ms = lab::time_ms([&]{ hits = cases[i].fn(rs.image, rs.gt); });
            auto ev = lab::evaluate(hits, rs.gt);
            aggs[i].rms     .push_back(ev.inlier_count > 0 ? ev.rms_px : 0.0);
            aggs[i].coverage.push_back(ev.coverage);
            aggs[i].runtime .push_back(ms);
            aggs[i].outlier_scenes  .push_back(ev.outlier_count >= 5 ? 1 : 0);
            aggs[i].no_detect_scenes.push_back(ev.total_points == 0 ? 1 : 0);

            if (std::find(sample_seeds.begin(), sample_seeds.end(), s) != sample_seeds.end()) {
                char name[256];
                std::snprintf(name, sizeof(name), "seed%03d_%s_overlay.png",
                              s, cases[i].name);
                lab::save_overlay(rs.image, hits, rs.gt,
                    (out_dir / "samples" / name).string(),
                    std::string(cases[i].name) + " | " + rs.gt.shape_name +
                    " A=" + std::to_string((int)rs.gt.amplitude) +
                    " δ=" + std::to_string(rs.gt.edge_contrast) +
                    " σ=" + std::to_string((int)rs.gt.gaussian_sigma));
            }
        }
    }

    // Print + markdown
    std::printf("\n%-18s %6s %6s %6s %6s %6s %6s %6s\n",
        "algorithm", "RMS_p50", "RMS_p90", "covmn", "ms_p50", "ms_p90",
        "outli", "nohit");
    std::printf("%s\n", std::string(70, '-').c_str());

    std::ostringstream md;
    md << "# primitive_fitting lab — batch benchmark\n\n"
       << "Scenes: " << seeds << " randomised 320×240 images per algorithm.\n"
       << "Each scene randomises curve shape (tilted line / half-sine / full-sine / cubic / shallow arc), amplitude ∈ [4, 28] px, edge contrast δ ∈ [30, 90], background intensity, 5–50 bright spike blocks (size 3–8 px, δ≈2× edge), Gaussian σ ∈ [0, 6], salt-pepper ∈ [0, 0.012].\n\n"
       << "Metrics\n"
       << "- `RMS_p50`, `RMS_p90`  — 50th / 90th percentile RMS of inlier points (within ±2 px of GT). Lower is better.\n"
       << "- `covmn`  — mean fraction of image x-range covered by inliers.\n"
       << "- `ms_p50`, `ms_p90`  — runtime distribution.\n"
       << "- `outli`  — fraction of scenes with ≥ 5 outliers.\n"
       << "- `nohit`  — fraction of scenes where the algorithm produced zero points.\n\n"
       << "| Algorithm | RMS p50 (px) | RMS p90 (px) | Mean coverage | Runtime p50 (ms) | Runtime p90 (ms) | Outlier scenes | No-hit scenes |\n"
       << "|---|---:|---:|---:|---:|---:|---:|---:|\n";

    for (const auto& a : aggs) {
        double rms_p50 = percentile(a.rms, 0.50);
        double rms_p90 = percentile(a.rms, 0.90);
        double cov_mn  = mean_of(a.coverage);
        double rt_p50  = percentile(a.runtime, 0.50);
        double rt_p90  = percentile(a.runtime, 0.90);
        double outl    = mean_of(std::vector<double>(a.outlier_scenes.begin(), a.outlier_scenes.end()));
        double nohit   = mean_of(std::vector<double>(a.no_detect_scenes.begin(), a.no_detect_scenes.end()));

        std::printf("%-18s %6.3f %6.3f %6.1f%% %6.3f %6.3f %5.1f%% %5.1f%%\n",
            a.name.c_str(), rms_p50, rms_p90, cov_mn * 100.0,
            rt_p50, rt_p90, outl * 100.0, nohit * 100.0);

        char line[256];
        std::snprintf(line, sizeof(line),
            "| `%s` | %.3f | %.3f | %.1f%% | %.3f | %.3f | %.1f%% | %.1f%% |\n",
            a.name.c_str(), rms_p50, rms_p90, cov_mn * 100.0,
            rt_p50, rt_p90, outl * 100.0, nohit * 100.0);
        md << line;
    }

    md << "\nOverlay samples (seeds " << 0 << ", 1, 7, 42) are under `samples/`.\n";
    std::ofstream((out_dir / "RESULTS.md").string()) << md.str();
    std::printf("\nWrote %s\n", (out_dir / "RESULTS.md").string().c_str());
    return 0;
}
