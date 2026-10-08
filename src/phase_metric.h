#pragma once

#include "phase_statistics.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

// Same post-step indices and final window as make_phase_diagram_CUDA.py.
// Kept independent of CUDA so the sampling contract can be tested on a CPU.
class PhaseMetricAccumulator {
public:
    PhaseMetricAccumulator(int iterations, int interval, int final_count, int cut)
        : iterations_(iterations), interval_(interval), cut_(cut) {
        if (iterations <= 0 || interval <= 0 || final_count <= 0 || cut < 0) {
            throw std::runtime_error("Invalid phase metric sampling parameters.");
        }
        const int total_samples = 1 + (iterations - 1) / interval;
        expected_samples_ = std::min(total_samples, final_count);
        first_ = (total_samples - expected_samples_) * interval;
        last_ = (total_samples - 1) * interval;
    }

    bool wants_iteration(int iteration) const {
        return iteration >= first_ && iteration <= last_ &&
               iteration % interval_ == 0;
    }

    void add(int iteration, double value, const SpatialMoments& spatial = {}) {
        if (samples_ >= expected_samples_ ||
            iteration != first_ + samples_ * interval_) {
            throw std::runtime_error("Unexpected phase metric sample index.");
        }
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("Phase metric is not finite and nonnegative.");
        }
        if (!std::isfinite(spatial.mass) || spatial.mass < 0.0 ||
            (spatial.mass > 0.0 &&
             (!std::isfinite(spatial.mean_x) || !std::isfinite(spatial.mean_x_squared) ||
              !std::isfinite(spatial.variance_x) || spatial.mean_x_squared < 0.0 ||
              spatial.variance_x < 0.0))) {
            throw std::runtime_error("Invalid spatial statistics sample.");
        }
        metric_moments_.add(value);
        if (spatial.mass > 0.0) {
            centers_.add(spatial.mean_x);
            mean_x_squared_sum_ += spatial.mean_x_squared;
            variance_x_sum_ += spatial.variance_x;
        }
        sum_ += value;
        ++samples_;
    }

    double variance() const { mean(); return metric_moments_.variance(); }
    double stddev() const { return std::sqrt(variance()); }

    double mean() const {
        if (samples_ != expected_samples_ || !std::isfinite(sum_)) {
            throw std::runtime_error("Phase metric samples are incomplete or invalid.");
        }
        return sum_ / samples_;
    }

    void write(const std::filesystem::path& path) const {
        const double value = mean();
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        output << std::setprecision(std::numeric_limits<double>::max_digits10)
               << "{\n"
               << "  \"metric\": \"mean_discrete_sum_abs_psi_fourth_power\",\n"
               << "  \"value\": " << value << ",\n"
               << "  \"statistics_version\": 1,\n"
               << "  \"metric_variance\": " << variance() << ",\n"
               << "  \"metric_std\": " << stddev() << ",\n"
               << "  \"spatial_snapshots_averaged\": " << centers_.count() << ",\n";
        if (centers_.count() > 0) {
            const double spatial_variance = variance_x_sum_ / centers_.count();
            if (!std::isfinite(mean_x_squared_sum_) || !std::isfinite(spatial_variance)) {
                throw std::runtime_error("Spatial statistics overflowed.");
            }
            output << "  \"x_mean\": " << centers_.mean() << ",\n"
                   << "  \"x_squared_mean\": " << mean_x_squared_sum_ / centers_.count() << ",\n"
                   << "  \"sigma_x_squared\": " << spatial_variance << ",\n"
                   << "  \"sigma_x\": " << std::sqrt(spatial_variance) << ",\n"
                   << "  \"x_mean_time_std\": " << centers_.stddev() << ",\n";
        } else {
            output << "  \"x_mean\": null,\n  \"x_squared_mean\": null,\n"
                   << "  \"sigma_x_squared\": null,\n  \"sigma_x\": null,\n"
                   << "  \"x_mean_time_std\": null,\n";
        }
        output
               << "  \"snapshots_averaged\": " << samples_ << ",\n"
               << "  \"cut_points_each_edge\": " << cut_ << ",\n"
               << "  \"first_snapshot_iteration\": " << first_ << ",\n"
               << "  \"last_snapshot_iteration\": " << last_ << ",\n"
               << "  \"save_every_nth_iteration\": " << interval_ << ",\n"
               << "  \"number_of_iterations\": " << iterations_ << "\n}\n";
        output.close();
        if (!output) {
            throw std::runtime_error("Unable to write phase metric: " + path.string());
        }
    }

private:
    int iterations_;
    int interval_;
    int cut_;
    int expected_samples_ = 0;
    int first_ = 0;
    int last_ = 0;
    int samples_ = 0;
    double sum_ = 0.0;
    RunningMoments metric_moments_;
    RunningMoments centers_;
    double mean_x_squared_sum_ = 0.0;
    double variance_x_sum_ = 0.0;
};
