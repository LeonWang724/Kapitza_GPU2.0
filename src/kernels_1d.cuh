// Codex CUDA Port: launch interface for traceable 1D split-step CUDA kernels.
#pragma once

#include <cuComplex.h>

enum class FloquetMode {
    Legacy,
    Physical,
};

void launch_initialize_spectral(double* k_squared, cuDoubleComplex* k_propagator,
                                int points, double step_x, double time_step,
                                bool imaginary_time);
void launch_copy_complex(const cuDoubleComplex* input, cuDoubleComplex* output,
                         int points);
void launch_update_floquet(const cuDoubleComplex* static_potential,
                            const cuDoubleComplex* floquet_base,
                            cuDoubleComplex* floquet_work,
                            cuDoubleComplex* total_potential, int points,
                            double factor, FloquetMode mode);
void launch_density(const cuDoubleComplex* psi, double* density, int points);
void launch_position_half_step(const cuDoubleComplex* input,
                               cuDoubleComplex* output,
                               const cuDoubleComplex* potential,
                               const double* density, int points,
                               double time_step, double beta,
                               bool imaginary_time);
void launch_apply_kinetic_and_fft_scale(cuDoubleComplex* psi_k,
                                        const cuDoubleComplex* k_propagator,
                                        int points);
void launch_scale_complex(cuDoubleComplex* values, int points, double factor);
void launch_kinetic_terms(const cuDoubleComplex* unnormalized_fft,
                          const double* k_squared, double* terms,
                          int points);
void launch_energy_terms(const double* density,
                         const cuDoubleComplex* potential,
                         double* potential_terms,
                         double* interaction_terms,
                         int points);
void launch_overlap_terms(const double* initial_density,
                          const double* current_density,
                          double* terms, int points);

