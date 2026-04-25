//
// algo_constrained_grid.cpp — coarse-to-fine grid search over
// constrained polynomial coefficients.
//
// Search space.  p(u) = a0 + a1·u + a2·u² + a3·u³,  u ∈ [-1, 1].
// We grid (a1, a2, a3) inside their box, prune candidates that violate
// the slope/curv/jerk caps, and for each surviving (a1, a2, a3) do a
// 1-D inner search for the best a0 (vertical shift).
//
// This is the "ground-truth" of what a globally-optimal constrained
// polynomial fit looks like — within the discretisation it cannot be
// beaten. The other two algorithms (RANSAC, Bernstein DP) are
// validated against this baseline.
//
// Coarse → fine refinement:
//   1. Coarse grid:  COARSE_BINS per shape axis (default 9 → 729 (a1,a2,a3)
//      triples), inner a0 at COARSE_BINS_A0 = 25.  Saves the top
//      TOP_K best by score.
//   2. Around each top-K winner, fine grid: ±1 coarse-bin window, FINE_BINS
//      = 5 sub-bins per axis. Re-rank, take new winner.
// Total work ≈ 9³·25 + K·5³·25 = 18 K evaluations of `score_poly`,
// each doing W=320 line samples.
//
// Constraint bounds default to PolyConstraints{} from the common header
// which is loose enough to admit all curves the lab generator produces.
//

#include "common.hpp"
#include "constrained_poly_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace lab {

namespace {

constexpr int    BAND_HALF_PX  = 50;

// Coarse grid extents in u-frame coefficients.  Chosen to envelope the
// full lab distribution: a1 carries main slope (max ≈ A·π/2 ≈ 44 for
// A=28 sine), a2 carries curvature (max ≈ A·π² ≈ 276), a3 the cubic
// jerk (max ≈ A·π³/2 ≈ 433). PolyConstraints will prune most of these
// triples before we ever score them.
constexpr double A1_MAX = 60.0;
constexpr double A2_MAX = 60.0;
constexpr double A3_MAX = 60.0;

constexpr int    COARSE_BINS     = 9;
constexpr int    COARSE_BINS_A0  = 25;
constexpr double A0_RANGE_PX     = 40.0;   // ± px around y0
constexpr int    TOP_K           = 8;
constexpr int    FINE_BINS       = 5;
constexpr int    FINE_BINS_A0    = 9;

inline double bin_to_val(int i, int n_bins, double half_range) {
    return -half_range + 2.0 * half_range * (double)i / (double)(n_bins - 1);
}

struct Candidate {
    double score = -1e30;
    double a0=0, a1=0, a2=0, a3=0;
};

double inner_search_a0(const cv::Mat& E,
                       double a1, double a2, double a3,
                       double y0, int band_lo, int band_hi,
                       int bins, double range, double& best_a0) {
    std::vector<double> coeffs = {0.0, a1, a2, a3};
    double best = -1e30; double ba0 = 0.0;
    for (int i = 0; i < bins; ++i) {
        double a0 = bin_to_val(i, bins, range);
        coeffs[0] = a0;
        double s = cpoly::score_poly(E, coeffs, y0, band_lo, band_hi);
        if (s > best) { best = s; ba0 = a0; }
    }
    best_a0 = ba0;
    return best;
}

} // namespace

std::vector<cv::Point2d> detect_constrained_grid(const cv::Mat& gray,
                                                 const GroundTruth& gt) {
    const int W = gray.cols;
    const int H = gray.rows;
    const double y0 = gt.y0;
    const cpoly::PolyConstraints con = cpoly::default_constraints();

    cv::Mat E = cpoly::compute_saturated_evidence(gray, y0, BAND_HALF_PX);
    const int band_lo = std::max(1,     (int)std::round(y0 - BAND_HALF_PX));
    const int band_hi = std::min(H - 2, (int)std::round(y0 + BAND_HALF_PX));

    // ---- 1. Coarse grid over (a1, a2, a3). ------------------------------
    std::vector<Candidate> top;
    top.reserve(TOP_K + 1);
    auto offer = [&](const Candidate& c) {
        if ((int)top.size() < TOP_K) {
            top.push_back(c);
            std::push_heap(top.begin(), top.end(),
                [](const Candidate& a, const Candidate& b){ return a.score > b.score; });
        } else if (c.score > top.front().score) {
            std::pop_heap(top.begin(), top.end(),
                [](const Candidate& a, const Candidate& b){ return a.score > b.score; });
            top.back() = c;
            std::push_heap(top.begin(), top.end(),
                [](const Candidate& a, const Candidate& b){ return a.score > b.score; });
        }
    };

    for (int i1 = 0; i1 < COARSE_BINS; ++i1) {
        double a1 = bin_to_val(i1, COARSE_BINS, A1_MAX);
        for (int i2 = 0; i2 < COARSE_BINS; ++i2) {
            double a2 = bin_to_val(i2, COARSE_BINS, A2_MAX);
            for (int i3 = 0; i3 < COARSE_BINS; ++i3) {
                double a3 = bin_to_val(i3, COARSE_BINS, A3_MAX);
                std::vector<double> coeffs = {0.0, a1, a2, a3};
                if (!cpoly::satisfies(coeffs, con, (double)W)) continue;
                double a0;
                double s = inner_search_a0(E, a1, a2, a3, y0,
                                           band_lo, band_hi,
                                           COARSE_BINS_A0, A0_RANGE_PX, a0);
                offer({s, a0, a1, a2, a3});
            }
        }
    }

    if (top.empty()) {
        // Fall back to flat curve at y0.
        return cpoly::sample_dense({0.0, 0.0, 0.0, 0.0}, y0, W);
    }

    // ---- 2. Fine refinement around each top-K. --------------------------
    // Around each candidate, sweep a 5×5×5 sub-grid spanning one coarse
    // bin in each direction. Inner a0 sweep similarly fine.
    const double da1 = (2.0 * A1_MAX) / (COARSE_BINS - 1);
    const double da2 = (2.0 * A2_MAX) / (COARSE_BINS - 1);
    const double da3 = (2.0 * A3_MAX) / (COARSE_BINS - 1);
    const double da0 = (2.0 * A0_RANGE_PX) / (COARSE_BINS_A0 - 1);

    Candidate best = *std::max_element(top.begin(), top.end(),
        [](const Candidate& a, const Candidate& b){ return a.score < b.score; });

    for (const Candidate& c : top) {
        for (int i1 = 0; i1 < FINE_BINS; ++i1) {
            double a1 = c.a1 + da1 * (-1.0 + 2.0 * (double)i1 / (FINE_BINS - 1));
            for (int i2 = 0; i2 < FINE_BINS; ++i2) {
                double a2 = c.a2 + da2 * (-1.0 + 2.0 * (double)i2 / (FINE_BINS - 1));
                for (int i3 = 0; i3 < FINE_BINS; ++i3) {
                    double a3 = c.a3 + da3 * (-1.0 + 2.0 * (double)i3 / (FINE_BINS - 1));
                    std::vector<double> coeffs = {0.0, a1, a2, a3};
                    if (!cpoly::satisfies(coeffs, con, (double)W)) continue;
                    // Inner a0 sweep (also refined: ±1 coarse bin, FINE_BINS_A0 sub-bins).
                    for (int j = 0; j < FINE_BINS_A0; ++j) {
                        double a0 = c.a0 + da0 * (-1.0 + 2.0 * (double)j / (FINE_BINS_A0 - 1));
                        coeffs[0] = a0;
                        double s = cpoly::score_poly(E, coeffs, y0, band_lo, band_hi);
                        if (s > best.score) best = {s, a0, a1, a2, a3};
                    }
                }
            }
        }
    }

    return cpoly::sample_dense({best.a0, best.a1, best.a2, best.a3}, y0, W);
}

} // namespace lab
