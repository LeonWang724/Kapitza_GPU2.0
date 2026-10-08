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
