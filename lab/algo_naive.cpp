//
// algo_naive.cpp — naive baseline: per-column argmax of |∂I/∂y|
// with parabolic sub-pixel refinement. No filtering, no outlier
// rejection, no smoothness prior. By construction, the 30 bright
// spike blocks dominate their columns and pull the curve upward.
//

#include "common.hpp"
#include <opencv2/imgproc.hpp>

#include <cmath>

namespace lab {

std::vector<cv::Point2d> detect_naive(const cv::Mat& gray, const GroundTruth& gt) {
    // 1. Vertical gradient via Sobel in float, then absolute value.
    cv::Mat gy;
    cv::Sobel(gray, gy, CV_32F, /*dx=*/0, /*dy=*/1, /*ksize=*/3);
    cv::Mat absGy = cv::abs(gy);

    const int H = absGy.rows;
    const int W = absGy.cols;

    std::vector<cv::Point2d> out;
    out.reserve(W);

    // 2. For every column, find the row with the largest |gradient|.
    for (int x = 0; x < W; ++x) {
        int best_y = 0;
        float best_v = -1.0f;
        for (int y = 0; y < H; ++y) {
            float v = absGy.at<float>(y, x);
            if (v > best_v) {
                best_v = v;
                best_y = y;
            }
        }

        // 3. Parabolic sub-pixel refinement on (y-1, y, y+1).
        double y_sub = static_cast<double>(best_y);
        if (best_y > 0 && best_y < H - 1) {
            double ym = absGy.at<float>(best_y - 1, x);
            double y0 = absGy.at<float>(best_y,     x);
            double yp = absGy.at<float>(best_y + 1, x);
            double denom = (ym - 2.0 * y0 + yp);
            if (std::abs(denom) > 1e-12) {
                double offset = 0.5 * (ym - yp) / denom;
                // Guard against runaway offsets from near-flat triples.
                if (offset > -1.0 && offset < 1.0) {
                    y_sub += offset;
                }
            }
        }

        // 4. One (x, y_sub) per column.
        out.emplace_back(static_cast<double>(x), y_sub);
    }

    (void)gt; // baseline ignores the ground truth.
    return out;
}

} // namespace lab
