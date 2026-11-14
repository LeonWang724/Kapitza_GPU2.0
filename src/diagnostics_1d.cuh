// Codex CUDA Port: GPU-resident diagnostic calculation interface for the 1D solver.
#pragma once

#include "device_buffer.cuh"
#include "extended_metrics.h"
#include "reductions.cuh"
#include "phase_statistics.h"

#include <cuComplex.h>
#include <cufft.h>

struct EnergyValues {
    double kinetic = 0.0;
    double potential = 0.0;
    double interaction = 0.0;
    double total = 0.0;
    double chemical_potential = 0.0;
};

class Diagnostics1D {
public:
    // extended: also keep psi at t = 0 and the bins for the 6.0 extended sums.
    // smoothing_half_width > 0 (7.0, fused loop only): the enqueued position
    // metrics use |psi|^2 averaged over 2 h + 1 points; see diagnostics_1d.cu.
    Diagnostics1D(int points, double step_x, double beta, bool extended = false,
                  int smoothing_half_width = 0);

    double norm(const cuDoubleComplex* psi);
    EnergyValues energy(const cuDoubleComplex* psi,
                        const cuDoubleComplex* potential,
                        const double* k_squared,
                        cufftHandle fft_plan);
    void capture_initial_density(const cuDoubleComplex* psi);
    double initial_density_overlap(const cuDoubleComplex* psi);
    double cropped_density_squared_sum(const cuDoubleComplex* psi, int cut);
    double phase_statistics(const cuDoubleComplex* psi, int cut, SpatialMoments& spatial);

    // Asynchronous forms for the fused solver: the kernels and reductions of
    // norm(), energy(), initial_density_overlap() and phase_statistics(), with
    // each sum left in device memory at the indices of diagnostic_sums.h.
    void enqueue_status_sums(const cuDoubleComplex* psi,
                             const cuDoubleComplex* potential,
                             const double* k_squared, cufftHandle fft_plan,
                             double* device_sums);
    void enqueue_phase_sums(const cuDoubleComplex* psi, int cut, double* device_sums);
    // Centered second pass over the density of the last enqueued psi.
    void enqueue_variance_sum(int cut, double mean_x, double* device_sums);

    // 6.0 extended metrics (constructed with extended = true).
    void configure_extended(const ExtendedSettings& settings, int cut);
    void capture_initial_state(const cuDoubleComplex* psi);
    // The sums of extended_metrics.h, at the indices of ExtendedSum. Must run
    // right after enqueue_status_sums() on the same psi, whose density and
    // transform it reuses.
    void enqueue_extended_sums(const cuDoubleComplex* psi,
                               const cuDoubleComplex* static_potential,
                               const double* k_squared, double* device_sums);

private:
    int points_;
    double step_x_;
    double beta_;
    bool extended_;
    ExtendedSettings extended_settings_{};
    int extended_cut_ = 0;
    int smoothing_ = 0;                  // 7.0: half width h, 0 = off
    // The unsmoothed |psi|^2 of the last enqueued psi (density_ itself when
    // smoothing is off).
    double* raw_density() { return smoothing_ ? raw_density_.data() : density_.data(); }
    int site_bins_ = 0;
    int pixel_bins_ = 0;
    DeviceBuffer<cuDoubleComplex> initial_psi_;
    DeviceBuffer<int> site_starts_;
    DeviceBuffer<int> pixel_starts_;
    DeviceBuffer<double> pixel_centres_;  // 7.0: nominal pixel centres
    GpuReducer reducer_;
    DeviceBuffer<double> density_;      // smoothed when smoothing_ > 0
    DeviceBuffer<double> raw_density_;  // 7.0: allocated only with smoothing
    DeviceBuffer<double> initial_density_;
    DeviceBuffer<double> terms_a_;
    DeviceBuffer<double> terms_b_;
    DeviceBuffer<cuDoubleComplex> diagnostic_fft_;
};
