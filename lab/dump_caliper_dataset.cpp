//
// dump_caliper_dataset.cpp — generate caliper-ROI training data.
//
// Two output formats, selected by --scene-records:
//
// 1. Per-caliper records (default, magic "XICAL"):
//      header  "XICAL\0\0\0", version=1, W, H, N_records
//      record  uint8[H][W] roi, float gt_y, int32 polarity
//    Each surviving caliper (gt_y inside ROI) is one record.
//
// 2. Scene records (--scene-records, magic "XICAS"):
//      header  "XICAS\0\0\0", version=1, K_per_scene, W, H, N_scenes
//      record  K × { uint8[H][W] roi, float gt_y, int32 polarity, int8 valid }
//    All K calipers from each scene are emitted as a single record.
//    `valid` is 0 if gt_y falls outside the ROI height (loss should
//    mask these calipers); 1 otherwise. Required for cross-caliper
//    architectures that need a fixed K-caliper batch per scene.
//
// Usage:
//   dump_caliper_dataset <out_path> [--scenes N] [--harsh|--low-noise|--no-noise|--photo]
//                                   [--calipers-per-scene K] [--scene-records]
//
// Default: 5000 scenes × 16 calipers = 80 000 records (or 5000 scene
// records under --scene-records).
//
// The caliper ROI is taken at evenly-spaced x positions, height = H
// centred on gt.y0. GT y is in ROI-local pixels.

#include "common.hpp"
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Wide-context caliper: 15 columns spanning ±7 px around the caliper
// centre. Lets the CNN see one full half of any false stripe (stripes
// span up to 30 % of W = 96 px, so a 15-col window catches at least
// one stripe edge in roughly half the calipers a stripe traverses)
// instead of just the 3-pixel column triplet that hides stripe-vs-
// edge texture. Increases first-layer params 5×; total still ~12 K.
constexpr int CAL_W = 15;
constexpr int CAL_H = 80;         // vertical extent

struct Record {
    std::vector<uint8_t> roi;     // CAL_H × CAL_W
    float                gt_y;
    int32_t              polarity;
};

// Extract a CAL_H × CAL_W ROI centred at (cx, gt.y0). Out-of-bound rows
// are filled with the nearest in-bound row.
void extract_caliper(const cv::Mat& gray, int cx, double y0,
                     std::vector<uint8_t>& out) {
    const int H = gray.rows, W = gray.cols;
    out.assign((size_t)CAL_H * CAL_W, 0);
    int y_top = (int)std::round(y0) - CAL_H / 2;
    int half_w = CAL_W / 2;
    for (int i = 0; i < CAL_H; ++i) {
        int y = std::clamp(y_top + i, 0, H - 1);
        const uint8_t* row = gray.ptr<uint8_t>(y);
        for (int j = 0; j < CAL_W; ++j) {
            int x = std::clamp(cx - half_w + j, 0, W - 1);
            out[(size_t)i * CAL_W + j] = row[x];
        }
    }
}

double gt_y_in_caliper(double x_center, const lab::GroundTruth& gt) {
    // Caliper ROI top is at  y = round(gt.y0) - CAL_H/2.
    // GT y at this x is      gt(x_center).
    // Local y =              gt(x_center) - y_top
    int y_top = (int)std::round(gt.y0) - CAL_H / 2;
    return gt(x_center) - (double)y_top;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out_path> "
                     "[--scenes N] [--harsh|--low-noise|--no-noise] "
                     "[--calipers-per-scene K]\n", argv[0]);
        return 1;
    }
    fs::path out_path = argv[1];
    int seeds = 5000;
    int K_per_scene = 16;
    bool scene_records = false;
    lab::NoiseLevel level = lab::NoiseLevel::Normal;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--scenes") && i + 1 < argc) {
            seeds = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--calipers-per-scene") && i + 1 < argc) {
            K_per_scene = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--scene-records")) scene_records = true;
        else if  (!std::strcmp(argv[i], "--harsh"))     level = lab::NoiseLevel::Harsh;
        else if  (!std::strcmp(argv[i], "--low-noise")) level = lab::NoiseLevel::Low;
        else if  (!std::strcmp(argv[i], "--no-noise"))  level = lab::NoiseLevel::None;
        else if  (!std::strcmp(argv[i], "--photo"))     level = lab::NoiseLevel::Photo;
    }

    fs::create_directories(out_path.parent_path());
    std::ofstream f(out_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "failed to open %s\n", out_path.string().c_str()); return 2; }

    if (scene_records) {
        // ---- Scene records: "XICAS" format ----
        const char     magic[8] = {'X','I','C','A','S',0,0,0};
        const uint32_t version  = 1;
        const uint32_t K  = (uint32_t)K_per_scene;
        const uint32_t Wc = CAL_W, Hc = CAL_H;
        f.write(magic, 8);
        f.write((const char*)&version, 4);
        f.write((const char*)&K,  4);
        f.write((const char*)&Wc, 4);
        f.write((const char*)&Hc, 4);
        // N_scenes patched at the end.
        const std::streampos n_pos = f.tellp();
        const uint32_t n_pad = 0;
        f.write((const char*)&n_pad, 4);

        uint32_t n_written = 0;
        std::vector<uint8_t> roi;
        for (int s = 0; s < seeds; ++s) {
            auto rs = lab::make_random_scene(s, level, false, false);
            const int W = rs.image.cols;
            // Emit K calipers for this scene back-to-back.
            for (int k = 0; k < K_per_scene; ++k) {
                int cx = (int)((k + 0.5) * W / K_per_scene);
                cx = std::clamp(cx, 1, W - 2);
                extract_caliper(rs.image, cx, rs.gt.y0, roi);
                float gt_y = (float)gt_y_in_caliper((double)cx, rs.gt);
                int32_t polarity = +1;
                int8_t valid = (gt_y >= 4 && gt_y <= CAL_H - 4) ? 1 : 0;
                f.write((const char*)roi.data(), (std::streamsize)roi.size());
                f.write((const char*)&gt_y, sizeof(float));
                f.write((const char*)&polarity, sizeof(int32_t));
                f.write((const char*)&valid, sizeof(int8_t));
            }
            ++n_written;
            if (n_written % 500 == 0) {
                std::fprintf(stderr, "  generated %u / %d scenes\n",
                             n_written, seeds);
            }
        }
        // Patch N_scenes.
        const std::streampos end_pos = f.tellp();
        f.seekp(n_pos);
        f.write((const char*)&n_written, 4);
        f.seekp(end_pos);
        const size_t per_record_bytes = (size_t)K * (CAL_H * CAL_W + 4 + 4 + 1);
        std::fprintf(stderr, "wrote %u scene records (%u calipers each), "
                     "%.1f MB -> %s\n",
                     n_written, K,
                     (24 + (double)n_written * per_record_bytes) / 1.0e6,
                     out_path.string().c_str());
        return 0;
    }

    // ---- Per-caliper records: "XICAL" format (default) ----
    std::vector<Record> records;
    records.reserve((size_t)seeds * K_per_scene);
    for (int s = 0; s < seeds; ++s) {
        auto rs = lab::make_random_scene(s, level, false, false);
        const int W = rs.image.cols;
        for (int k = 0; k < K_per_scene; ++k) {
            int cx = (int)((k + 0.5) * W / K_per_scene);
            cx = std::clamp(cx, 1, W - 2);
            Record r;
            extract_caliper(rs.image, cx, rs.gt.y0, r.roi);
            r.gt_y    = (float)gt_y_in_caliper((double)cx, rs.gt);
            r.polarity = +1;
            if (r.gt_y < 4 || r.gt_y > CAL_H - 4) continue;
            records.push_back(std::move(r));
        }
        if ((s + 1) % 500 == 0) {
            std::fprintf(stderr, "  generated %d / %d scenes (%zu records)\n",
                         s + 1, seeds, records.size());
        }
    }
    const char     magic[8] = {'X','I','C','A','L',0,0,0};
    const uint32_t version  = 1;
    const uint32_t Wc = CAL_W, Hc = CAL_H;
    const uint32_t N  = (uint32_t)records.size();
    f.write(magic, 8);
    f.write((const char*)&version, 4);
    f.write((const char*)&Wc, 4);
    f.write((const char*)&Hc, 4);
    f.write((const char*)&N,  4);
    for (const auto& r : records) {
        f.write((const char*)r.roi.data(), (std::streamsize)r.roi.size());
        f.write((const char*)&r.gt_y, sizeof(float));
        f.write((const char*)&r.polarity, sizeof(int32_t));
    }
    std::fprintf(stderr, "wrote %u records, %.1f MB -> %s\n",
                 N,
                 (8 + 16 + (double)N * (CAL_H * CAL_W + 4 + 4)) / 1.0e6,
                 out_path.string().c_str());
    return 0;
}
