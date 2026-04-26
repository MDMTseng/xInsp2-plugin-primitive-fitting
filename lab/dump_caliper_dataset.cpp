//
// dump_caliper_dataset.cpp — generate caliper-ROI training data.
//
// Produces a binary file
//
//   <out_path> = N records of:
//       uint8[CALIPER_H][CALIPER_W]  — caliper ROI grayscale crop
//       float                         — GT y position relative to ROI top
//                                       (in pixels, sub-pixel)
//       int32                         — polarity (+1 = curve has dark→bright
//                                                   transition top-to-bottom)
//
// And a header at the start:
//   "XICAL\0\0\0"  (8 bytes magic)
//   uint32 version (= 1)
//   uint32 W_caliper, H_caliper
//   uint32 N_records
//
// With this, a Python loader can directly mmap-and-train.
//
// Usage:
//   dump_caliper_dataset <out_path> --scenes 5000 [--harsh] [--low-noise]
//   [--calipers-per-scene K]
//
// Default: 5000 scenes × 16 calipers = 80 000 records.
//
// The caliper ROI is taken at evenly-spaced x positions, half-width 1
// (3 columns horizontally), height = CALIPER_H (default 80) centred on
// gt.y0. The GT y position recorded is gt(x_center) − (gt.y0 −
// CALIPER_H/2) — i.e. the y-coordinate in the ROI's local frame.

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

constexpr int CAL_W = 3;          // 3-column horizontal stripe
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
    lab::NoiseLevel level = lab::NoiseLevel::Normal;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--scenes") && i + 1 < argc) {
            seeds = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--calipers-per-scene") && i + 1 < argc) {
            K_per_scene = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--harsh"))     level = lab::NoiseLevel::Harsh;
        else if  (!std::strcmp(argv[i], "--low-noise")) level = lab::NoiseLevel::Low;
        else if  (!std::strcmp(argv[i], "--no-noise"))  level = lab::NoiseLevel::None;
        else if  (!std::strcmp(argv[i], "--photo"))     level = lab::NoiseLevel::Photo;
    }

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
            // Polarity is fixed by the lab convention (curve = dark→bright
            // when traversed top-to-bottom in image y), but we record it
            // explicitly so production datasets with flipped scenes
            // remain interpretable.
            r.polarity = +1;
            // Reject calipers whose GT y falls outside the ROI height —
            // these would make the model learn "no edge here" cases too
            // often. Keep only well-centred records.
            if (r.gt_y < 4 || r.gt_y > CAL_H - 4) continue;
            records.push_back(std::move(r));
        }
        if ((s + 1) % 500 == 0) {
            std::fprintf(stderr, "  generated %d / %d scenes (%zu records)\n",
                         s + 1, seeds, records.size());
        }
    }

    fs::create_directories(out_path.parent_path());
    std::ofstream f(out_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "failed to open %s\n", out_path.string().c_str()); return 2; }
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
    std::fprintf(stderr, "wrote %u records, %.1f MB → %s\n",
                 N,
                 (8 + 16 + (double)N * (CAL_H * CAL_W + 4 + 4)) / 1.0e6,
                 out_path.string().c_str());
    return 0;
}
