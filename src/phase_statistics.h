#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

struct SpatialMoments {
    // Unnormalized sum |psi|^2 in the same cropped region as the phase metric.
    double mass = 0.0;
    double mean_x = 0.0;
    double mean_x_squared = 0.0;
    double variance_x = 0.0;
};

// Welford accumulation avoids subtracting two large, nearly equal raw moments.
class RunningMoments {
public:
    void add(double value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite statistics sample.");
        }
        ++count_;
        const double delta = value - mean_;
        mean_ += delta / count_;
        m2_ += delta * (value - mean_);
    }
    int count() const { return count_; }
    double mean() const { return mean_; }
    double variance() const {
        if (count_ == 0 || !std::isfinite(m2_)) {
            throw std::runtime_error("Statistics samples are incomplete or invalid.");
        }
        return std::max(0.0, m2_ / count_);
    }
    double stddev() const { return std::sqrt(variance()); }

private:
    int count_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
};
