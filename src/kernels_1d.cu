// Codex CUDA Port: native CUDA kernels corresponding to original MKL vector operations.
#include "kernels_1d.cuh"

#include "cuda_checks.cuh"

#include <cmath>

namespace {

constexpr int kThreads = 256;

int blocks_for(int points) {
    return (points + kThreads - 1) / kThreads;
}

__device__ cuDoubleComplex complex_exp(cuDoubleComplex value) {
    const double magnitude = exp(value.x);
    return make_cuDoubleComplex(magnitude * cos(value.y), magnitude * sin(value.y));
}

__global__ void initialize_spectral_kernel(double* k_squared,
                                           cuDoubleComplex* k_propagator,
                                           int points, double delta_k,
                                           double time_step,
                                           bool imaginary_time) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;

    // CUDA PORT: exact original frequency ordering, including the negative
    // Nyquist bin at index N/2 for even N.
    const int signed_index = index >= points / 2 ? index - points : index;
    const double k = static_cast<double>(signed_index) * delta_k;
    const double k2 = k * k;
    k_squared[index] = k2;
    const double exponent = -0.5 * time_step * k2;
    k_propagator[index] = imaginary_time
        ? make_cuDoubleComplex(exp(exponent), 0.0)
        : make_cuDoubleComplex(cos(exponent), sin(exponent));
}

__global__ void copy_complex_kernel(const cuDoubleComplex* input,
                                    cuDoubleComplex* output, int points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < points) output[index] = input[index];
}

__global__ void update_floquet_kernel(const cuDoubleComplex* static_potential,
                                      const cuDoubleComplex* floquet_base,
                                      cuDoubleComplex* floquet_work,
                                      cuDoubleComplex* total_potential,
                                      int points, double factor,
                                      bool legacy) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;

    cuDoubleComplex driven;
    if (legacy) {
        // CUDA PORT: literal source behavior: cblas_zdscal mutates the already
        // scaled array, so the envelope is the cumulative product of cosines.
        floquet_work[index] = make_cuDoubleComplex(
            floquet_work[index].x * factor, floquet_work[index].y * factor);
        driven = floquet_work[index];
    } else {
        // CUDA PORT: physical alternative retaining an unchanged base array.
        driven = make_cuDoubleComplex(floquet_base[index].x * factor,
                                      floquet_base[index].y * factor);
    }
    total_potential[index] = cuCadd(static_potential[index], driven);
}

__global__ void density_kernel(const cuDoubleComplex* psi, double* density,
                               int points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const double real = psi[index].x;
    const double imag = psi[index].y;
    density[index] = real * real + imag * imag;
}

__global__ void position_half_step_kernel(
    const cuDoubleComplex* input, cuDoubleComplex* output,
    const cuDoubleComplex* potential, const double* density, int points,
    double time_step, double beta, bool imaginary_time) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;

    const cuDoubleComplex position_factor = imaginary_time
        ? make_cuDoubleComplex(-0.5 * time_step, 0.0)
        : make_cuDoubleComplex(0.0, -0.5 * time_step);
    const cuDoubleComplex interaction_factor = imaginary_time
        ? make_cuDoubleComplex(-0.5 * beta * time_step, 0.0)
        : make_cuDoubleComplex(0.0, -0.5 * beta * time_step);
    cuDoubleComplex exponent = cuCmul(position_factor, potential[index]);
    exponent = cuCadd(exponent,
                      make_cuDoubleComplex(interaction_factor.x * density[index],
                                           interaction_factor.y * density[index]));
    output[index] = cuCmul(input[index], complex_exp(exponent));
}

__global__ void kinetic_and_scale_kernel(cuDoubleComplex* psi_k,
                                         const cuDoubleComplex* propagator,
                                         int points, double fft_scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const cuDoubleComplex product = cuCmul(psi_k[index], propagator[index]);
    psi_k[index] = make_cuDoubleComplex(product.x * fft_scale,
                                        product.y * fft_scale);
}

__global__ void scale_complex_kernel(cuDoubleComplex* values, int points,
                                     double factor) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < points) {
        values[index].x *= factor;
        values[index].y *= factor;
    }
}

__global__ void kinetic_terms_kernel(const cuDoubleComplex* fft,
                                     const double* k_squared,
                                     double* terms, int points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const double real = fft[index].x;
    const double imag = fft[index].y;
    terms[index] = k_squared[index] * (real * real + imag * imag);
}

__global__ void energy_terms_kernel(const double* density,
                                    const cuDoubleComplex* potential,
                                    double* potential_terms,
                                    double* interaction_terms,
                                    int points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    potential_terms[index] = density[index] * potential[index].x;
    interaction_terms[index] = density[index] * density[index];
}

__global__ void overlap_terms_kernel(const double* initial_density,
                                     const double* current_density,
                                     double* terms, int points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < points) {
        terms[index] = initial_density[index] * current_density[index];
    }
}

}  // namespace

void launch_initialize_spectral(double* k_squared, cuDoubleComplex* k_propagator,
                                int points, double step_x, double time_step,
                                bool imaginary_time) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double delta_k = 2.0 * pi / (static_cast<double>(points) * step_x);
    initialize_spectral_kernel<<<blocks_for(points), kThreads>>>(
        k_squared, k_propagator, points, delta_k, time_step, imaginary_time);
    CUDA_KERNEL_CHECK();
}

void launch_copy_complex(const cuDoubleComplex* input, cuDoubleComplex* output,
                         int points) {
    copy_complex_kernel<<<blocks_for(points), kThreads>>>(input, output, points);
    CUDA_KERNEL_CHECK();
}

void launch_update_floquet(const cuDoubleComplex* static_potential,
                           const cuDoubleComplex* floquet_base,
                           cuDoubleComplex* floquet_work,
                           cuDoubleComplex* total_potential, int points,
                           double factor, FloquetMode mode) {
    update_floquet_kernel<<<blocks_for(points), kThreads>>>(
        static_potential, floquet_base, floquet_work, total_potential,
        points, factor, mode == FloquetMode::Legacy);
    CUDA_KERNEL_CHECK();
}

void launch_density(const cuDoubleComplex* psi, double* density, int points) {
    density_kernel<<<blocks_for(points), kThreads>>>(psi, density, points);
    CUDA_KERNEL_CHECK();
}

void launch_position_half_step(const cuDoubleComplex* input,
                               cuDoubleComplex* output,
                               const cuDoubleComplex* potential,
                               const double* density, int points,
                               double time_step, double beta,
                               bool imaginary_time) {
    position_half_step_kernel<<<blocks_for(points), kThreads>>>(
        input, output, potential, density, points, time_step, beta,
        imaginary_time);
    CUDA_KERNEL_CHECK();
}

void launch_apply_kinetic_and_fft_scale(cuDoubleComplex* psi_k,
                                        const cuDoubleComplex* k_propagator,
                                        int points) {
    kinetic_and_scale_kernel<<<blocks_for(points), kThreads>>>(
        psi_k, k_propagator, points, 1.0 / static_cast<double>(points));
    CUDA_KERNEL_CHECK();
}

void launch_scale_complex(cuDoubleComplex* values, int points, double factor) {
    scale_complex_kernel<<<blocks_for(points), kThreads>>>(values, points, factor);
    CUDA_KERNEL_CHECK();
}

void launch_kinetic_terms(const cuDoubleComplex* unnormalized_fft,
                          const double* k_squared, double* terms,
                          int points) {
    kinetic_terms_kernel<<<blocks_for(points), kThreads>>>(
        unnormalized_fft, k_squared, terms, points);
    CUDA_KERNEL_CHECK();
}

void launch_energy_terms(const double* density,
                         const cuDoubleComplex* potential,
                         double* potential_terms,
                         double* interaction_terms,
                         int points) {
    energy_terms_kernel<<<blocks_for(points), kThreads>>>(
        density, potential, potential_terms, interaction_terms, points);
    CUDA_KERNEL_CHECK();
}

void launch_overlap_terms(const double* initial_density,
                          const double* current_density,
                          double* terms, int points) {
    overlap_terms_kernel<<<blocks_for(points), kThreads>>>(
        initial_density, current_density, terms, points);
    CUDA_KERNEL_CHECK();
}

