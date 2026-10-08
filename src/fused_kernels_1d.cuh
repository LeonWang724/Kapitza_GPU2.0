// Launch interface for the fused, batched real-time split-step kernels.
#pragma once

#include "split_step_math.cuh"

#include <cuComplex.h>

#include <vector>

// Grid points simulated together in one GPU batch. Bounded so the per-system
// Floquet factors fit in kernel parameters on every supported architecture.
constexpr int kMaxBatchSystems = 64;

struct SystemFactors {
    double value[kMaxBatchSystems];
};

SystemFactors make_system_factors(const std::vector<double>& factors);

// Each array holds `systems` consecutive fields of `points` elements.
void launch_fused_first_half(const cuDoubleComplex* psi, cuDoubleComplex* psi_half,
                             const cuDoubleComplex* static_potential,
                             const cuDoubleComplex* floquet_base,
                             cuDoubleComplex* factor_cache, int points, int systems,
                             const split_step::StepConstants& constants,
                             const SystemFactors& current);
void launch_fused_second_then_first_half(
    cuDoubleComplex* psi_half, const cuDoubleComplex* static_potential,
    const cuDoubleComplex* floquet_base, cuDoubleComplex* factor_cache,
    int points, int systems, const split_step::StepConstants& constants,
    const SystemFactors& previous, const SystemFactors& current);
void launch_fused_second_half(const cuDoubleComplex* psi_half, cuDoubleComplex* psi,
                              const cuDoubleComplex* static_potential,
                              const cuDoubleComplex* floquet_base,
                              const cuDoubleComplex* factor_cache,
                              cuDoubleComplex* potential, int points, int systems,
                              const split_step::StepConstants& constants,
                              const SystemFactors& current);
// One propagator of `points` elements is shared by every system.
void launch_batched_kinetic(cuDoubleComplex* psi_k,
                            const cuDoubleComplex* k_propagator,
                            int points, int systems);
