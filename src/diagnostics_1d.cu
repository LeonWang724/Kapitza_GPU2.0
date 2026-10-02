// Codex CUDA Port: norm, energy, and explicitly defined initial-density overlap.
#include "diagnostics_1d.cuh"

#include "cuda_checks.cuh"
#include "kernels_1d.cuh"

#include <stdexcept>

Diagnostics1D::Diagnostics1D(int points, double step_x, double beta)
    : points_(points),
      step_x_(step_x),
      beta_(beta),
      reducer_(points),
      density_(points),
      initial_density_(points),
      terms_a_(points),
      terms_b_(points),
      diagnostic_fft_(points) {}

double Diagnostics1D::norm(const cuDoubleComplex* psi) {
    launch_density(psi, density_.data(), points_);
    return step_x_ * reducer_.sum(density_.data(), points_);
}

EnergyValues Diagnostics1D::energy(const cuDoubleComplex* psi,
                                   const cuDoubleComplex* potential,
                                   const double* k_squared,
                                   cufftHandle fft_plan) {
    launch_density(psi, density_.data(), points_);
    CUFFT_CHECK(cufftExecZ2Z(fft_plan,
                             const_cast<cuDoubleComplex*>(psi),
                             diagnostic_fft_.data(), CUFFT_FORWARD));
    launch_kinetic_terms(diagnostic_fft_.data(), k_squared, terms_a_.data(), points_);
    const double kinetic = 0.5 * step_x_ *
                           reducer_.sum(terms_a_.data(), points_) /
                           static_cast<double>(points_);

    launch_energy_terms(density_.data(), potential, terms_a_.data(),
                        terms_b_.data(), points_);
    const double potential_energy =
        step_x_ * reducer_.sum(terms_a_.data(), points_);
    const double interaction = 0.5 * beta_ * step_x_ *
                               reducer_.sum(terms_b_.data(), points_);

    EnergyValues values;
    values.kinetic = kinetic;
    values.potential = potential_energy;
    values.interaction = interaction;
    values.total = kinetic + potential_energy + interaction;
    values.chemical_potential = values.total + interaction;
    return values;
}

void Diagnostics1D::capture_initial_density(const cuDoubleComplex* psi) {
    launch_density(psi, initial_density_.data(), points_);
}

double Diagnostics1D::initial_density_overlap(const cuDoubleComplex* psi) {
    launch_density(psi, density_.data(), points_);
    launch_overlap_terms(initial_density_.data(), density_.data(),
                         terms_a_.data(), points_);
    // CUDA PORT: discrete sum_j |psi_initial[j]|^2 |psi[j]|^2, deliberately
    // without dx. This matches the scale of the opaque executable's observed
    // InitOverlap field for step_x=1, but equivalence is not yet claimed.
    return reducer_.sum(terms_a_.data(), points_);
}

double Diagnostics1D::cropped_density_squared_sum(const cuDoubleComplex* psi,
                                                  int cut) {
    if (cut < 0 || cut > (points_ - 1) / 2) {
        throw std::runtime_error("Spatial cut removes all metric points.");
    }
    launch_density(psi, density_.data(), points_);
    launch_overlap_terms(density_.data(), density_.data(), terms_a_.data(), points_);
    // Match the existing NumPy metric: no dx and no renormalization.
    return reducer_.sum(terms_a_.data() + cut, points_ - 2 * cut);
}
