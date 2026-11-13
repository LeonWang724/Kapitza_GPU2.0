// 6.0 extended metrics: kernels for the extra sums of each status iteration.
#include "extended_kernels_1d.cuh"

#include "cuda_checks.cuh"

#include <cmath>

namespace {

constexpr int kThreads = 256;

int blocks_for(int count) {
    return (count + kThreads - 1) / kThreads;
}

__global__ void density_weight_kernel(const double* density, const cuDoubleComplex* potential,
                                      double* terms, int points, double step_x, int weight,
                                      double parameter) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const double rho = density[index];
    const double x = static_cast<double>(index - points / 2) * step_x;
    double value = 0.0;
    switch (weight) {
        case 0: value = rho * cos(parameter * x); break;
        case 1: value = rho * sin(parameter * x); break;
        case 2: value = rho * x; break;
        case 3: value = rho * x * x; break;
        case 4: value = rho * x * x * x; break;
        case 5: value = rho * (x * x) * (x * x); break;
        case 6: value = (x < parameter && x > -parameter) ? rho : 0.0; break;
        case 7: value = rho > 0.0 ? rho * log(rho) : 0.0; break;
        default: value = rho * potential[index].x; break;
    }
    terms[index] = value;
}

__global__ void bin_mass_kernel(const double* density, const int* starts, int bins,
                                double* squares, double* logs) {
    const int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin >= bins) return;
    double mass = 0.0;
    for (int index = starts[bin]; index < starts[bin + 1]; ++index) mass += density[index];
    squares[bin] = mass * mass;
    logs[bin] = mass > 0.0 ? mass * log(mass) : 0.0;
}

__global__ void return_overlap_kernel(const cuDoubleComplex* initial,
                                      const cuDoubleComplex* psi, double* real, double* imag,
                                      int points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const cuDoubleComplex a = initial[index];
    const cuDoubleComplex b = psi[index];
    real[index] = a.x * b.x + a.y * b.y;
    imag[index] = a.x * b.y - a.y * b.x;
}

__global__ void momentum_zone_kernel(const cuDoubleComplex* fft, const double* k_squared,
                                     double* terms, int points, double k_squared_min,
                                     double k_squared_max) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const double k2 = k_squared[index];
    const double real = fft[index].x;
    const double imag = fft[index].y;
    terms[index] = (k2 >= k_squared_min && k2 < k_squared_max) ? real * real + imag * imag : 0.0;
}

}  // namespace

void launch_density_weight_terms(const double* density, const cuDoubleComplex* potential,
                                 double* terms, int points, double step_x,
                                 DensityWeight weight, double parameter) {
    density_weight_kernel<<<blocks_for(points), kThreads>>>(
        density, potential, terms, points, step_x, static_cast<int>(weight), parameter);
    CUDA_KERNEL_CHECK();
}

void launch_bin_mass_terms(const double* density, const int* starts, int bins,
                           double* squares, double* logs) {
    bin_mass_kernel<<<blocks_for(bins), kThreads>>>(density, starts, bins, squares, logs);
    CUDA_KERNEL_CHECK();
}

void launch_return_overlap_terms(const cuDoubleComplex* initial, const cuDoubleComplex* psi,
                                 double* real, double* imag, int points) {
    return_overlap_kernel<<<blocks_for(points), kThreads>>>(initial, psi, real, imag, points);
    CUDA_KERNEL_CHECK();
}

void launch_momentum_zone_terms(const cuDoubleComplex* fft, const double* k_squared,
                                double* terms, int points, double k_squared_min,
                                double k_squared_max) {
    momentum_zone_kernel<<<blocks_for(points), kThreads>>>(fft, k_squared, terms, points,
                                                           k_squared_min, k_squared_max);
    CUDA_KERNEL_CHECK();
}
