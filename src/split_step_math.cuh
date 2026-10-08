// Per-element arithmetic of the real-time split step, shared by the fused CUDA
// kernels and the host equivalence test. Each expression repeats the matching
// operation of the original kernels in kernels_1d.cu, operand for operand, so
// the strict floating-point build (no FMA contraction) gives bitwise-identical
// values. Only redundant work is skipped, never changed.
#pragma once

#include <cuComplex.h>

#include <cmath>

#if defined(__CUDACC__)
#define GPE_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define GPE_HOST_DEVICE inline
#endif

namespace split_step {

struct StepConstants {
    double time_step = 0.0;
    double beta = 0.0;
    // Adds floquet_base * cos(omega t) to the static potential.
    bool floquet = false;
    // beta == 0: both half steps of an iteration use the same factor.
    bool reuse_factor = false;
};

// density_kernel.
GPE_HOST_DEVICE double density(cuDoubleComplex value) {
    const double real = value.x;
    const double imag = value.y;
    return real * real + imag * imag;
}

// Physical branch of update_floquet_kernel.
GPE_HOST_DEVICE cuDoubleComplex physical_potential(cuDoubleComplex static_value,
                                                   cuDoubleComplex floquet_base,
                                                   double factor) {
    const cuDoubleComplex driven = make_cuDoubleComplex(floquet_base.x * factor,
                                                        floquet_base.y * factor);
    return cuCadd(static_value, driven);
}

// Without a drive, the original copied the static potential unchanged.
GPE_HOST_DEVICE cuDoubleComplex potential_at(const StepConstants& constants,
                                             cuDoubleComplex static_value,
                                             cuDoubleComplex floquet_base,
                                             double factor) {
    return constants.floquet
        ? physical_potential(static_value, floquet_base, factor)
        : static_value;
}

// complex_exp of kernels_1d.cu. exp(+-0) is exactly 1 (IEEE 754 and the CUDA
// math API), so it is not evaluated outside the absorbing boundaries, where
// the potential is real; 1 * cos(y) and 1 * sin(y) are exact.
GPE_HOST_DEVICE cuDoubleComplex complex_exp(cuDoubleComplex value) {
    const double magnitude = value.x == 0.0 ? 1.0 : exp(value.x);
    return make_cuDoubleComplex(magnitude * cos(value.y), magnitude * sin(value.y));
}

// Exponential factor of position_half_step_kernel with imaginary_time=false.
GPE_HOST_DEVICE cuDoubleComplex half_step_factor(const StepConstants& constants,
                                                 cuDoubleComplex potential,
                                                 double density_value) {
    const cuDoubleComplex position_factor =
        make_cuDoubleComplex(0.0, -0.5 * constants.time_step);
    const cuDoubleComplex interaction_factor =
        make_cuDoubleComplex(0.0, -0.5 * constants.beta * constants.time_step);
    cuDoubleComplex exponent = cuCmul(position_factor, potential);
    exponent = cuCadd(exponent,
                      make_cuDoubleComplex(interaction_factor.x * density_value,
                                           interaction_factor.y * density_value));
    return complex_exp(exponent);
}

// First real-space half step of an iteration; also returns its factor.
GPE_HOST_DEVICE cuDoubleComplex first_half(const StepConstants& constants,
                                           cuDoubleComplex psi,
                                           cuDoubleComplex potential,
                                           cuDoubleComplex* factor) {
    *factor = half_step_factor(constants, potential, density(psi));
    return cuCmul(psi, *factor);
}

// Second real-space half step. With beta == +-0 the interaction term adds the
// same signed zeros for every finite density, so this iteration's exponent,
// and therefore its factor, equals the first half step's: reuse it.
GPE_HOST_DEVICE cuDoubleComplex second_half(const StepConstants& constants,
                                            cuDoubleComplex psi_half,
                                            cuDoubleComplex potential,
                                            cuDoubleComplex first_half_factor) {
    const cuDoubleComplex factor = constants.reuse_factor
        ? first_half_factor
        : half_step_factor(constants, potential, density(psi_half));
    return cuCmul(psi_half, factor);
}

// Second half step of iteration n-1, then the first half step of iteration n:
// the two pointwise passes the original ran as separate kernels.
GPE_HOST_DEVICE cuDoubleComplex second_then_first_half(
    const StepConstants& constants, cuDoubleComplex psi_half,
    cuDoubleComplex previous_potential, cuDoubleComplex current_potential,
    cuDoubleComplex* factor) {
    const cuDoubleComplex state =
        second_half(constants, psi_half, previous_potential, *factor);
    return first_half(constants, state, current_potential, factor);
}

// kinetic_and_scale_kernel.
GPE_HOST_DEVICE cuDoubleComplex kinetic(cuDoubleComplex psi_k,
                                        cuDoubleComplex propagator,
                                        double fft_scale) {
    const cuDoubleComplex product = cuCmul(psi_k, propagator);
    return make_cuDoubleComplex(product.x * fft_scale, product.y * fft_scale);
}

}  // namespace split_step
