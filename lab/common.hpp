#pragma once
//
// Shared types + scene / eval / timing primitives used by every algorithm
// in the primitive_fitting lab. Each algorithm is a simple free function
//
//     std::vector<cv::Point2d> detect_<name>(const cv::Mat& gray,
//                                            int expected_y_center);
//
// and is evaluated against the same synthetic scene + ground-truth curve.
//

#include <opencv2/core.hpp>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace lab {

// Ground truth is any y = f(x) function plus scene metadata. Algorithms
// may use `y0` as a user-supplied midline prior (it is the caliper band
// centre in the plugin contract), but they may NOT peek at `evaluate` or
// the other metadata — that would be test-set overfitting.
struct GroundTruth {
    int    W = 320, H = 240;
    double y0 = 120;        // midline — legitimate prior for the algorithms
    std::function<double(double)> evaluate;

    // Metadata — for reporting only. Algorithms must not read these.
    std::string shape_name = "fixed_s_curve";
    double amplitude       = 15;
    int    edge_contrast   = 60;
    int    spike_count     = 30;
    double gaussian_sigma  = 0;

    double operator()(double x) const { return evaluate(x); }
};

// Fixed scene kept for backward compat / regression: 320×240 grayscale
// with a half-period S-curve (amp 15) plus 30 deterministic spike blocks.
cv::Mat make_weak_curve_with_spikes(const GroundTruth& gt);

// Randomised scene generator — varies curve shape, amplitude, contrast,
// background level, spike count & size, Gaussian noise, salt&pepper.
// Two distinct seeds must give different scenes; same seed must give
// bit-identical scenes.
struct RandomScene {
    cv::Mat      image;
    GroundTruth  gt;
};
// Photo: same noise distribution as Normal plus a per-scene
// photometric augmentation pass (gradient + vignette + blob + tint).
enum class NoiseLevel { None, Low, Normal, Harsh, Photo };
RandomScene make_random_scene(int seed,
                              NoiseLevel level = NoiseLevel::Normal,
                              bool dashed_edge = false,
                              bool bumpy_edge = false,
                              double blend_alpha = 0.0,
                              int    blend_partner_offset = 1000);

// Evaluate a detected poly-line against ground truth.
//   * rms_px          — root mean square vertical distance, inliers only
//   * coverage        — fraction of GT x-range within ±2 px of *some* hit
//   * outlier_count   — hits further than outlier_thr_px from GT curve
//   * inlier_count    — hits within ±2 px of GT
struct EvalReport {
    int    total_points = 0;
    int    inlier_count = 0;
    int    outlier_count = 0;
    double rms_px = 0;
    double coverage = 0;
};
EvalReport evaluate(const std::vector<cv::Point2d>& detected,
                    const GroundTruth& gt,
                    double inlier_thr_px = 2.0,
                    double outlier_thr_px = 10.0);

// High-resolution wall-clock for a callable returning ms as double.
template <class F>
double time_ms(F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    std::forward<F>(f)();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Paint the detected poly-line + ground truth onto the original grayscale
// scene and write it as PNG. Useful for eyeballing each algorithm's output.
void save_overlay(const cv::Mat& gray,
                  const std::vector<cv::Point2d>& detected,
                  const GroundTruth& gt,
                  const std::string& out_path,
                  const std::string& label);

// --- Algorithm signatures ---
using Algo = std::function<std::vector<cv::Point2d>(const cv::Mat&, const GroundTruth&)>;

std::vector<cv::Point2d> detect_naive          (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_dp             (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_dijkstra       (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_tensor_voting  (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_ransac (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_ransac_A(const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_ransac_B(const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_ransac_C(const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_ransac_D(const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_dp          (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_dp_poly     (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_subregion_tv_peaks  (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_subregion_tv_band   (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_subregion_tv_strips (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_subregion_dp_strips (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_spline_knot_dp      (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_constrained_grid    (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_constrained_ransac  (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_constrained_knot_dp (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_cnn         (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_cnn_cross   (const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_cnn_cross_ort(const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_cnn_cross_ort_prosac(const cv::Mat& gray, const GroundTruth& gt);
std::vector<cv::Point2d> detect_caliper_cnn_cross_ort_spline(const cv::Mat& gray, const GroundTruth& gt);

} // namespace lab
