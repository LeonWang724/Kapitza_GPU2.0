// Fused, batched real-time split-step kernels. Arithmetic lives in
// split_step_math.cuh; these kernels only map threads to elements.
#include "fused_kernels_1d.cuh"

#include "cuda_checks.cuh"

#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kThreads = 256;

dim3 grid_for(int points, int systems) {
    return dim3(static_cast<unsigned int>((points + kThreads - 1) / kThreads),
                static_cast<unsigned int>(systems));
}

__device__ __forceinline__ cuDoubleComplex load_floquet_base(
    const split_step::StepConstants& constants,
    const cuDoubleComplex* floquet_base, std::size_t element) {
    return constants.floquet ? floquet_base[element] : make_cuDoubleComplex(0.0, 0.0);
}

__global__ void fused_first_half_kernel(
    const cuDoubleComplex* psi, cuDoubleComplex* psi_half,
    const cuDoubleComplex* static_potential, const cuDoubleComplex* floquet_base,
    cuDoubleComplex* factor_cache, int points,
    split_step::StepConstants constants, SystemFactors current) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const int system = blockIdx.y;
    const std::size_t element = static_cast<std::size_t>(system) * points + index;

    const cuDoubleComplex potential = split_step::potential_at(
        constants, static_potential[element],
        load_floquet_base(constants, floquet_base, element), current.value[system]);
    cuDoubleComplex factor;
    psi_half[element] = split_step::first_half(constants, psi[element], potential, &factor);
    if (constants.reuse_factor) factor_cache[element] = factor;
}

__global__ void fused_second_then_first_half_kernel(
    cuDoubleComplex* psi_half, const cuDoubleComplex* static_potential,
    const cuDoubleComplex* floquet_base, cuDoubleComplex* factor_cache, int points,
    split_step::StepConstants constants, SystemFactors previous,
    SystemFactors current) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const int system = blockIdx.y;
    const std::size_t element = static_cast<std::size_t>(system) * points + index;

    const cuDoubleComplex static_value = static_potential[element];
    const cuDoubleComplex base = load_floquet_base(constants, floquet_base, element);
    const cuDoubleComplex previous_potential = split_step::potential_at(
        constants, static_value, base, previous.value[system]);
    const cuDoubleComplex current_potential = split_step::potential_at(
        constants, static_value, base, current.value[system]);
    cuDoubleComplex factor = constants.reuse_factor
        ? factor_cache[element]
        : make_cuDoubleComplex(0.0, 0.0);
    psi_half[element] = split_step::second_then_first_half(
        constants, psi_half[element], previous_potential, current_potential, &factor);
    if (constants.reuse_factor) factor_cache[element] = factor;
}

__global__ void fused_second_half_kernel(
    const cuDoubleComplex* psi_half, cuDoubleComplex* psi,
    const cuDoubleComplex* static_potential, const cuDoubleComplex* floquet_base,
    const cuDoubleComplex* factor_cache, cuDoubleComplex* potential_out, int points,
    split_step::StepConstants constants, SystemFactors current) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const int system = blockIdx.y;
    const std::size_t element = static_cast<std::size_t>(system) * points + index;

    const cuDoubleComplex potential = split_step::potential_at(
        constants, static_potential[element],
        load_floquet_base(constants, floquet_base, element), current.value[system]);
    const cuDoubleComplex cached = constants.reuse_factor
        ? factor_cache[element]
        : make_cuDoubleComplex(0.0, 0.0);
    psi[element] = split_step::second_half(constants, psi_half[element], potential, cached);
    // Diagnostics and potential snapshots read the current potential.
    potential_out[element] = potential;
}

__global__ void batched_kinetic_kernel(cuDoubleComplex* psi_k,
                                       const cuDoubleComplex* propagator,
                                       int points, double fft_scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= points) return;
    const std::size_t element = static_cast<std::size_t>(blockIdx.y) * points + index;
    psi_k[element] = split_step::kinetic(psi_k[element], propagator[index], fft_scale);
}

void check_systems(int systems) {
    if (systems <= 0 || systems > kMaxBatchSystems) {
        throw std::runtime_error("Batch size is outside the supported range.");
    }
}

}  // namespace

SystemFactors make_system_factors(const std::vector<double>& factors) {
    check_systems(static_cast<int>(factors.size()));
    SystemFactors result{};
    for (std::size_t system = 0; system < factors.size(); ++system) {
        result.value[system] = factors[system];
    }
    return result;
}

void launch_fused_first_half(const cuDoubleComplex* psi, cuDoubleComplex* psi_half,
                             const cuDoubleComplex* static_potential,
                             const cuDoubleComplex* floquet_base,
                             cuDoubleComplex* factor_cache, int points, int systems,
                             const split_step::StepConstants& constants,
                             const SystemFactors& current) {
    check_systems(systems);
    fused_first_half_kernel<<<grid_for(points, systems), kThreads>>>(
        psi, psi_half, static_potential, floquet_base, factor_cache, points,
        constants, current);
    CUDA_KERNEL_CHECK();
}

void launch_fused_second_then_first_half(
    cuDoubleComplex* psi_half, const cuDoubleComplex* static_potential,
    const cuDoubleComplex* floquet_base, cuDoubleComplex* factor_cache,
    int points, int systems, const split_step::StepConstants& constants,
    const SystemFactors& previous, const SystemFactors& current) {
    check_systems(systems);
    fused_second_then_first_half_kernel<<<grid_for(points, systems), kThreads>>>(
        psi_half, static_potential, floquet_base, factor_cache, points, constants,
        previous, current);
    CUDA_KERNEL_CHECK();
}

void launch_fused_second_half(const cuDoubleComplex* psi_half, cuDoubleComplex* psi,
                              const cuDoubleComplex* static_potential,
                              const cuDoubleComplex* floquet_base,
                              const cuDoubleComplex* factor_cache,
                              cuDoubleComplex* potential, int points, int systems,
                              const split_step::StepConstants& constants,
                              const SystemFactors& current) {
    check_systems(systems);
    fused_second_half_kernel<<<grid_for(points, systems), kThreads>>>(
        psi_half, psi, static_potential, floquet_base, factor_cache, potential,
        points, constants, current);
    CUDA_KERNEL_CHECK();
}

void launch_batched_kinetic(cuDoubleComplex* psi_k,
                            const cuDoubleComplex* k_propagator,
                            int points, int systems) {
    check_systems(systems);
    batched_kinetic_kernel<<<grid_for(points, systems), kThreads>>>(
        psi_k, k_propagator, points, 1.0 / static_cast<double>(points));
    CUDA_KERNEL_CHECK();
}
