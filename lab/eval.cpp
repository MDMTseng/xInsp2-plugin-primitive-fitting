//
// eval.cpp — ground-truth comparison + visualisation for the benchmark.
//

#include "common.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace lab {

EvalReport evaluate(const std::vector<cv::Point2d>& detected,
                    const GroundTruth& gt,
                    double inlier_thr_px,
                    double outlier_thr_px) {
    EvalReport r;
    r.total_points = (int)detected.size();
    if (detected.empty()) return r;

    double sum_sq = 0; int inlier = 0;
    for (const auto& p : detected) {
        double err = p.y - gt(p.x);
        if (std::abs(err) <= inlier_thr_px) {
            sum_sq += err * err;
            ++inlier;
        } else if (std::abs(err) > outlier_thr_px) {
            ++r.outlier_count;
        }
    }
    r.inlier_count = inlier;
    r.rms_px = inlier > 0 ? std::sqrt(sum_sq / inlier) : 0.0;

    // Coverage: bucket the ground-truth x-range into 1-px bins, flag any
    // bin with at least one detected point within ±inlier_thr of GT.
    std::vector<bool> covered(gt.W, false);
    for (const auto& p : detected) {
        if (std::abs(p.y - gt(p.x)) > inlier_thr_px) continue;
        int bucket = (int)std::round(p.x);
        if (bucket >= 0 && bucket < gt.W) covered[bucket] = true;
    }
    int c = 0;
    for (bool b : covered) if (b) ++c;
    r.coverage = (double)c / gt.W;
    return r;
}

void save_overlay(const cv::Mat& gray,
                  const std::vector<cv::Point2d>& detected,
                  const GroundTruth& gt,
                  const std::string& out_path,
                  const std::string& label) {
    cv::Mat rgb;
    cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);

    // Dashed ground-truth curve (yellow).
    for (int x = 0; x < gt.W - 2; x += 6) {
        cv::line(rgb,
            cv::Point2d(x,     gt(x)),
            cv::Point2d(x + 3, gt(x + 3)),
            cv::Scalar(0, 220, 220), 1, cv::LINE_AA);
    }

    // Detected hits as coloured dots: green if within ±2 px of GT, red otherwise.
    for (const auto& p : detected) {
        bool inl = std::abs(p.y - gt(p.x)) <= 2.0;
        cv::Scalar c = inl ? cv::Scalar(57, 255, 20) : cv::Scalar(60, 80, 255);
        cv::circle(rgb, p, 2, c, -1, cv::LINE_AA);
    }

    // Label (top-left).
    cv::putText(rgb, label, cv::Point(6, 16), cv::FONT_HERSHEY_SIMPLEX,
                0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::imwrite(out_path, rgb);
}

} // namespace lab
