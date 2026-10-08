// Codex CUDA Port: norm, energy, and explicitly defined initial-density overlap.
#include "diagnostics_1d.cuh"

#include "cuda_checks.cuh"
#include "diagnostic_sums.h"
#include "kernels_1d.cuh"

#include <stdexcept>
#include <cmath>

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

double Diagnostics1D::phase_statistics(const cuDoubleComplex* psi, int cut,
                                       SpatialMoments& spatial) {
    const double metric = cropped_density_squared_sum(psi, cut);
    const int count = points_ - 2 * cut;
    spatial = {};
    spatial.mass = reducer_.sum(density_.data() + cut, count);
    if (!std::isfinite(spatial.mass) || spatial.mass < 0.0) {
        throw std::runtime_error("Cropped wavefunction mass is invalid.");
    }
    // A completely absorbed state still has a valid zero phase metric, but no
    // normalized spatial distribution. Its spatial measurements stay null.
    if (spatial.mass == 0.0) return metric;
    launch_spatial_moment_terms(density_.data(), terms_a_.data(), terms_b_.data(),
                               points_, step_x_);
    spatial.mean_x = reducer_.sum(terms_a_.data() + cut, count) / spatial.mass;
    spatial.mean_x_squared = reducer_.sum(terms_b_.data() + cut, count) / spatial.mass;
    // A centered second pass is stable even for a narrow cloud far from x=0.
    launch_spatial_variance_terms(density_.data(), terms_a_.data(), points_,
                                 step_x_, spatial.mean_x);
    spatial.variance_x = reducer_.sum(terms_a_.data() + cut, count) / spatial.mass;
    return metric;
}

void Diagnostics1D::enqueue_status_sums(const cuDoubleComplex* psi,
                                        const cuDoubleComplex* potential,
                                        const double* k_squared,
                                        cufftHandle fft_plan,
                                        double* device_sums) {
    // One density serves norm, energy and overlap; each was |psi|^2 of psi.
    launch_density(psi, density_.data(), points_);
    reducer_.sum_into(density_.data(), points_, device_sums + kNormSum);
    CUFFT_CHECK(cufftExecZ2Z(fft_plan,
                             const_cast<cuDoubleComplex*>(psi),
                             diagnostic_fft_.data(), CUFFT_FORWARD));
    launch_kinetic_terms(diagnostic_fft_.data(), k_squared, terms_a_.data(), points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kKineticSum);
    launch_energy_terms(density_.data(), potential, terms_a_.data(),
                        terms_b_.data(), points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kPotentialSum);
    reducer_.sum_into(terms_b_.data(), points_, device_sums + kInteractionSum);
    launch_overlap_terms(initial_density_.data(), density_.data(),
                         terms_a_.data(), points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kOverlapSum);
}

void Diagnostics1D::enqueue_phase_sums(const cuDoubleComplex* psi, int cut,
                                       double* device_sums) {
    if (cut < 0 || cut > (points_ - 1) / 2) {
        throw std::runtime_error("Spatial cut removes all metric points.");
    }
    const int count = points_ - 2 * cut;
    launch_density(psi, density_.data(), points_);
    launch_overlap_terms(density_.data(), density_.data(), terms_a_.data(), points_);
    reducer_.sum_into(terms_a_.data() + cut, count, device_sums + kMetricSum);
    reducer_.sum_into(density_.data() + cut, count, device_sums + kMassSum);
    // Moments of a zero-mass window are computed but never used.
    launch_spatial_moment_terms(density_.data(), terms_a_.data(), terms_b_.data(),
                               points_, step_x_);
    reducer_.sum_into(terms_a_.data() + cut, count, device_sums + kFirstMomentSum);
    reducer_.sum_into(terms_b_.data() + cut, count, device_sums + kSecondMomentSum);
}

void Diagnostics1D::enqueue_variance_sum(int cut, double mean_x,
                                         double* device_sums) {
    const int count = points_ - 2 * cut;
    launch_spatial_variance_terms(density_.data(), terms_a_.data(), points_,
                                 step_x_, mean_x);
    reducer_.sum_into(terms_a_.data() + cut, count, device_sums + kVarianceSum);
}
