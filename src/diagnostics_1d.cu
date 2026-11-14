// Codex CUDA Port: norm, energy, and explicitly defined initial-density overlap.
#include "diagnostics_1d.cuh"

#include "cuda_checks.cuh"
#include "diagnostic_sums.h"
#include "extended_kernels_1d.cuh"
#include "kernels_1d.cuh"

#include <stdexcept>
#include <cmath>
#include <vector>

// 7.0 smoothing (smoothing_ > 0) in the enqueued sums of the fused loop:
//   smoothed |psi|^2: phase metric (IPR, centre, width), initial-density
//     overlap, and the extended position sums (moments, central window,
//     entropy, IPR, lattice-site bins);
//   raw |psi|^2: norm and energies, <cos/sin 2kL x> and <V> (they resolve the
//     inside of a lattice site), and the 4 um pixel bins (binning into camera
//     pixels is itself the smoothing).
// The reference loop's functions above enqueue_* never smooth; the solver
// refuses smoothing there.
Diagnostics1D::Diagnostics1D(int points, double step_x, double beta, bool extended,
                             int smoothing_half_width)
    : points_(points),
      step_x_(step_x),
      beta_(beta),
      extended_(extended),
      smoothing_(smoothing_half_width),
      initial_psi_(extended ? static_cast<std::size_t>(points) : 0),
      reducer_(points),
      density_(points),
      raw_density_(smoothing_half_width > 0 ? static_cast<std::size_t>(points) : 0),
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
    if (smoothing_ > 0) {
        launch_density(psi, raw_density_.data(), points_);
        launch_box_smooth(raw_density_.data(), initial_density_.data(), points_, smoothing_);
        return;
    }
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
    double* raw = raw_density();
    launch_density(psi, raw, points_);
    reducer_.sum_into(raw, points_, device_sums + kNormSum);
    CUFFT_CHECK(cufftExecZ2Z(fft_plan,
                             const_cast<cuDoubleComplex*>(psi),
                             diagnostic_fft_.data(), CUFFT_FORWARD));
    launch_kinetic_terms(diagnostic_fft_.data(), k_squared, terms_a_.data(), points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kKineticSum);
    launch_energy_terms(raw, potential, terms_a_.data(), terms_b_.data(), points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kPotentialSum);
    reducer_.sum_into(terms_b_.data(), points_, device_sums + kInteractionSum);
    if (smoothing_ > 0) launch_box_smooth(raw, density_.data(), points_, smoothing_);
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
    launch_density(psi, raw_density(), points_);
    if (smoothing_ > 0) launch_box_smooth(raw_density(), density_.data(), points_, smoothing_);
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

namespace {

void upload_starts(const std::vector<int>& starts, DeviceBuffer<int>& device, int& bins) {
    device.allocate(starts.size());
    CUDA_CHECK(cudaMemcpy(device.data(), starts.data(), starts.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    bins = static_cast<int>(starts.size()) - 1;
}

void upload_centres(const std::vector<double>& centres, DeviceBuffer<double>& device) {
    device.allocate(centres.size());
    CUDA_CHECK(cudaMemcpy(device.data(), centres.data(), centres.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
}

}  // namespace

void Diagnostics1D::configure_extended(const ExtendedSettings& settings, int cut) {
    if (!extended_) throw std::runtime_error("Extended metrics were not enabled.");
    if (cut < 0 || cut > (points_ - 1) / 2) {
        throw std::runtime_error("Spatial cut removes all metric points.");
    }
    if (!(settings.lattice_wavevector > 0.0) || !(settings.central_halfwidth > 0.0)) {
        throw std::runtime_error("Extended metrics need a positive wavevector and window.");
    }
    extended_settings_ = settings;
    extended_cut_ = cut;
    const double period = kExtendedPi / settings.lattice_wavevector;
    upload_starts(bin_starts(points_, cut, step_x_, period, settings.site_offset),
                  site_starts_, site_bins_);
    const BinLayout pixels = bin_layout(points_, cut, step_x_, settings.pixel_width,
                                        -0.5 * settings.pixel_width);
    upload_starts(pixels.starts, pixel_starts_, pixel_bins_);
    upload_centres(pixels.centres, pixel_centres_);
}

void Diagnostics1D::capture_initial_state(const cuDoubleComplex* psi) {
    if (!extended_) throw std::runtime_error("Extended metrics were not enabled.");
    launch_copy_complex(psi, initial_psi_.data(), points_);
}

void Diagnostics1D::enqueue_extended_sums(const cuDoubleComplex* psi,
                                          const cuDoubleComplex* static_potential,
                                          const double* k_squared, double* device_sums) {
    if (!extended_ || site_bins_ <= 0 || pixel_bins_ <= 0) {
        throw std::runtime_error("Extended metrics were not configured.");
    }
    const int cut = extended_cut_;
    const int count = points_ - 2 * cut;
    const auto weighted = [&](const double* density, DensityWeight weight, double parameter,
                              int index) {
        launch_density_weight_terms(density, static_potential, terms_a_.data(),
                                    points_, step_x_, weight, parameter);
        reducer_.sum_into(terms_a_.data() + cut, count, device_sums + index);
    };
    // density_ (smoothed with 7.0 smoothing), the raw density and
    // diagnostic_fft_ hold |psi|^2 and the transform of psi from
    // enqueue_status_sums(); nothing below overwrites them.
    const double* smooth = density_.data();
    const double* raw = raw_density();
    reducer_.sum_into(density_.data() + cut, count, device_sums + kExtMass);
    const double two_k = 2.0 * extended_settings_.lattice_wavevector;
    // Inside a lattice site: always the raw density.
    weighted(raw, DensityWeight::Cos, two_k, kExtCos);
    weighted(raw, DensityWeight::Sin, two_k, kExtSin);
    weighted(smooth, DensityWeight::X1, 0.0, kExtX1);
    weighted(smooth, DensityWeight::X2, 0.0, kExtX2);
    weighted(smooth, DensityWeight::X3, 0.0, kExtX3);
    weighted(smooth, DensityWeight::X4, 0.0, kExtX4);
    weighted(smooth, DensityWeight::Central, extended_settings_.central_halfwidth, kExtCentral);
    weighted(smooth, DensityWeight::DensityLogDensity, 0.0, kExtDensityLogDensity);
    weighted(raw, DensityWeight::Potential, 0.0, kExtPotential);
    launch_overlap_terms(density_.data(), density_.data(), terms_a_.data(), points_);
    reducer_.sum_into(terms_a_.data() + cut, count, device_sums + kExtDensitySquared);

    launch_bin_mass_terms(density_.data(), site_starts_.data(), site_bins_,
                          terms_a_.data(), terms_b_.data());
    reducer_.sum_into(terms_a_.data(), site_bins_, device_sums + kExtSiteSquared);
    reducer_.sum_into(terms_b_.data(), site_bins_, device_sums + kExtSiteLog);
    // Camera pixels bin the raw density: the binning is the smoothing.
    launch_bin_mass_terms(raw, pixel_starts_.data(), pixel_bins_,
                          terms_a_.data(), terms_b_.data());
    reducer_.sum_into(terms_a_.data(), pixel_bins_, device_sums + kExtPixelSquared);
    // 7.0: first and second moments of the 4 um pixel profile (the camera's).
    launch_bin_moment_terms(raw, pixel_starts_.data(), pixel_centres_.data(),
                            pixel_bins_, terms_a_.data(), terms_b_.data());
    reducer_.sum_into(terms_a_.data(), pixel_bins_, device_sums + kExtPixelX1);
    reducer_.sum_into(terms_b_.data(), pixel_bins_, device_sums + kExtPixelX2);

    launch_return_overlap_terms(initial_psi_.data(), psi, terms_a_.data(), terms_b_.data(),
                                points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kExtReturnReal);
    reducer_.sum_into(terms_b_.data(), points_, device_sums + kExtReturnImag);

    launch_density(diagnostic_fft_.data(), terms_a_.data(), points_);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kExtMomentumTotal);
    const double kl2 = extended_settings_.lattice_wavevector *
                       extended_settings_.lattice_wavevector;
    launch_momentum_zone_terms(diagnostic_fft_.data(), k_squared, terms_a_.data(), points_,
                               0.0, kl2);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kExtMomentumZero);
    launch_momentum_zone_terms(diagnostic_fft_.data(), k_squared, terms_a_.data(), points_,
                               kl2, 9.0 * kl2);
    reducer_.sum_into(terms_a_.data(), points_, device_sums + kExtMomentumSecond);
}
