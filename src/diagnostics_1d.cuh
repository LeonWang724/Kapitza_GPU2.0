// Codex CUDA Port: GPU-resident diagnostic calculation interface for the 1D solver.
#pragma once

#include "device_buffer.cuh"
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
    Diagnostics1D(int points, double step_x, double beta);

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

private:
    int points_;
    double step_x_;
    double beta_;
    GpuReducer reducer_;
    DeviceBuffer<double> density_;
    DeviceBuffer<double> initial_density_;
    DeviceBuffer<double> terms_a_;
    DeviceBuffer<double> terms_b_;
    DeviceBuffer<cuDoubleComplex> diagnostic_fft_;
};
