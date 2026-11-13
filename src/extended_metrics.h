// 6.0 extended metrics: the extra sums taken at every status iteration, their
// settings, and the grid-index ranges of lattice sites and pixels. Host only.
#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

// Raw sums over the cropped window [cut, points - cut) unless noted, with
// rho_j = |psi_j|^2 and x_j = (j - points / 2) * step_x. Python turns them into
// the CSV columns (extended_metrics.py); the order is part of the file format.
enum ExtendedSum {
    kExtMass,                // sum rho
    kExtCos,                 // sum rho cos(2 kL x)
    kExtSin,                 // sum rho sin(2 kL x)
    kExtX1,                  // sum rho x
    kExtX2,                  // sum rho x^2
    kExtX3,                  // sum rho x^3
    kExtX4,                  // sum rho x^4
    kExtCentral,             // sum rho over |x| < central half width
    kExtDensityLogDensity,   // sum rho log(rho)
    kExtDensitySquared,      // sum rho^2 (the phase metric's IPR)
    kExtPotential,           // sum rho Re(static potential)
    kExtSiteSquared,         // sum over lattice sites of (site mass)^2
    kExtSiteLog,             // sum over lattice sites of m log m
    kExtPixelSquared,        // sum over pixels of (pixel mass)^2
    kExtReturnReal,          // Re sum conj(psi_0) psi, whole grid
    kExtReturnImag,          // Im sum conj(psi_0) psi, whole grid
    kExtMomentumTotal,       // sum |psi_k|^2, whole grid
    kExtMomentumZero,        // sum |psi_k|^2 for |k| < kL
    kExtMomentumSecond,      // sum |psi_k|^2 for kL <= |k| < 3 kL
    kExtendedSumCount,
};

inline constexpr const char* kExtendedSumNames[kExtendedSumCount] = {
    "mass", "cos_sum", "sin_sum", "x1_sum", "x2_sum", "x3_sum", "x4_sum",
    "central_sum", "rho_log_rho_sum", "rho_squared_sum", "potential_sum",
    "site_squared_sum", "site_log_sum", "pixel_squared_sum",
    "return_real", "return_imag", "momentum_total", "momentum_zero", "momentum_second",
};

// Lengths in solver units.
struct ExtendedSettings {
    double lattice_wavevector = 0.0;  // kL; the lattice cos^2(kL x) has period pi / kL
    double central_halfwidth = 0.0;   // half width of the central window around x = 0
    double site_offset = 0.0;         // site edges sit at site_offset + n pi / kL
    double pixel_width = 0.0;         // coarse-graining pixels, centred on x = 0
};

constexpr double kExtendedPi = 3.141592653589793238462643383279502884;

// Bins are the intervals [offset + s width, offset + (s + 1) width) of x that
// meet the cropped window. Returns the first grid index of every bin followed
// by points - cut, so bin b covers [starts[b], starts[b + 1]).
inline std::vector<int> bin_starts(int points, int cut, double step_x, double width,
                                   double offset) {
    if (cut < 0 || points <= 2 * cut) {
        throw std::runtime_error("The spatial cut removes all grid points.");
    }
    if (!(step_x > 0.0) || !(width > step_x) || !std::isfinite(offset)) {
        throw std::runtime_error("Extended-metric bins must be wider than the grid step.");
    }
    const int end = points - cut;
    const double half = static_cast<double>(points / 2);
    const auto position = [&](int index) {
        return static_cast<double>(index - points / 2) * step_x;
    };
    const long first_bin = static_cast<long>(std::floor((position(cut) - offset) / width));
    const long last_bin = static_cast<long>(std::floor((position(end - 1) - offset) / width));
    std::vector<int> starts{cut};
    for (long bin = first_bin + 1; bin <= last_bin; ++bin) {
        // Separate statements keep compilers from fusing the multiply and add,
        // so the edges round exactly as in extended_metrics.py.
        const double scaled = static_cast<double>(bin) * width;
        const double edge = offset + scaled;
        const double grid = edge / step_x;
        int index = static_cast<int>(std::ceil(grid + half));
        if (index < cut) index = cut;
        if (index > end) index = end;
        if (index > starts.back()) starts.push_back(index);
    }
    starts.push_back(end);
    return starts;
}

// One row per status iteration: the iteration, the drive factor
// cos(omega t + phase), the status norm, kinetic sum and initial-density
// overlap, then the raw sums above.
class ExtendedWriter {
public:
    explicit ExtendedWriter(const std::filesystem::path& path) : stream_(path) {
        if (!stream_) {
            throw std::runtime_error("Could not open extended-metrics file: " + path.string());
        }
        stream_ << "iteration;drive;norm;kinetic_sum;density_overlap";
        for (const char* name : kExtendedSumNames) stream_ << ';' << name;
        stream_ << '\n' << std::setprecision(17);
    }

    void append(int iteration, double drive, double norm, double kinetic_sum,
                double density_overlap, const double* sums) {
        stream_ << iteration << ';' << drive << ';' << norm << ';' << kinetic_sum << ';'
                << density_overlap;
        for (int index = 0; index < kExtendedSumCount; ++index) {
            stream_ << ';' << sums[index];
        }
        stream_ << '\n';
        if (!stream_) throw std::runtime_error("Writing the extended-metrics file failed.");
    }

private:
    std::ofstream stream_;
};
