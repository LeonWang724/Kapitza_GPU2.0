// Final host arithmetic for diagnostics whose sums were reduced on the GPU
// without intermediate synchronization. The expressions are those of
// Diagnostics1D::norm(), energy(), initial_density_overlap() and
// phase_statistics(), so the reported values do not change.
#pragma once

#include "hdf5_io.h"
#include "phase_statistics.h"

#include <cmath>
#include <stdexcept>

enum StatusSum {
    kNormSum,
    kKineticSum,
    kPotentialSum,
    kInteractionSum,
    kOverlapSum,
    kStatusSumCount,
};

enum PhaseSum {
    kMetricSum,
    kMassSum,
    kFirstMomentSum,
    kSecondMomentSum,
    kVarianceSum,
    kPhaseSumCount,
};

inline StatusData status_from_sums(int iteration, const double* sums,
                                   double step_x, double beta, int points) {
    const double kinetic = 0.5 * step_x * sums[kKineticSum] /
                           static_cast<double>(points);
    const double potential_energy = step_x * sums[kPotentialSum];
    const double interaction = 0.5 * beta * step_x * sums[kInteractionSum];
    const double total = kinetic + potential_energy + interaction;

    StatusData status;
    status.iteration = iteration;
    status.norm = step_x * sums[kNormSum];
    status.kinetic_energy = kinetic;
    status.potential_energy = potential_energy;
    status.interaction_energy = interaction;
    status.total_energy = total;
    status.chemical_potential = total + interaction;
    status.mean_dynamic_potential = 0.0;
    status.initial_density_overlap = sums[kOverlapSum];
    return status;
}

// Returns whether the cropped mass supports normalized spatial moments.
inline bool spatial_mass_is_positive(double mass) {
    if (!std::isfinite(mass) || mass < 0.0) {
        throw std::runtime_error("Cropped wavefunction mass is invalid.");
    }
    return mass != 0.0;
}

inline double spatial_mean_from_sums(const double* sums) {
    return sums[kFirstMomentSum] / sums[kMassSum];
}

// A zero-mass window keeps null spatial values, as in phase_statistics().
inline SpatialMoments spatial_from_sums(const double* sums) {
    SpatialMoments spatial;
    spatial.mass = sums[kMassSum];
    if (!spatial_mass_is_positive(spatial.mass)) return spatial;
    spatial.mean_x = spatial_mean_from_sums(sums);
    spatial.mean_x_squared = sums[kSecondMomentSum] / spatial.mass;
    spatial.variance_x = sums[kVarianceSum] / spatial.mass;
    return spatial;
}
