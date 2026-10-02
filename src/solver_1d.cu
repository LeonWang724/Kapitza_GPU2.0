// Codex CUDA Port: persistent-GPU implementation of the original 1D split-step loop.
#include "solver_1d.h"

#include "cuda_checks.cuh"
#include "device_buffer.cuh"
#include "diagnostics_1d.cuh"
#include "hdf5_io.h"
#include "phase_metric.h"

#include <cufft.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FftPlan1D {
public:
    explicit FftPlan1D(int points) {
        CUFFT_CHECK(cufftPlan1d(&handle_, points, CUFFT_Z2Z, 1));
    }
    ~FftPlan1D() {
        if (handle_ != 0) {
            cufftDestroy(handle_);
        }
    }
    FftPlan1D(const FftPlan1D&) = delete;
    FftPlan1D& operator=(const FftPlan1D&) = delete;
    cufftHandle get() const noexcept { return handle_; }

private:
    cufftHandle handle_ = 0;
};

void copy_host_to_device(const std::vector<cuDoubleComplex>& host,
                         DeviceBuffer<cuDoubleComplex>& device) {
    if (host.size() != device.size()) {
        throw std::runtime_error("Host/device array length mismatch.");
    }
    CUDA_CHECK(cudaMemcpy(device.data(), host.data(), device.bytes(),
                          cudaMemcpyHostToDevice));
}

std::vector<cuDoubleComplex> copy_device_to_host(
    const DeviceBuffer<cuDoubleComplex>& device) {
    std::vector<cuDoubleComplex> host(device.size());
    CUDA_CHECK(cudaMemcpy(host.data(), device.data(), device.bytes(),
                          cudaMemcpyDeviceToHost));
    return host;
}

void save_device_field(const std::filesystem::path& path,
                       const DeviceBuffer<cuDoubleComplex>& device) {
    write_complex_hdf5(path, copy_device_to_host(device));
}

}  // namespace

const char* floquet_mode_name(FloquetMode mode) {
    return mode == FloquetMode::Legacy ? "legacy" : "physical";
}

FloquetMode parse_floquet_mode(const char* value) {
    const std::string mode(value);
    if (mode == "legacy") return FloquetMode::Legacy;
    if (mode == "physical") return FloquetMode::Physical;
    throw std::runtime_error("Floquet mode must be 'legacy' or 'physical'.");
}

SolverResult run_solver_1d(const ConfigData& config,
                           const SolverOptions& options) {
    validate_1d_config(config);
    const bool compact = !config.phase_metric_file.empty();
    std::optional<PhaseMetricAccumulator> phase_metric;
    if (compact) {
        phase_metric.emplace(config.number_of_iterations,
                             config.save_every_nth_iteration,
                             config.phase_metric_final_snapshot_count,
                             config.phase_metric_cut_points_each_edge);
    }
    CUDA_CHECK(cudaSetDevice(options.device));

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, options.device));
    std::cout << "CUDA device: " << options.device << " ("
              << device_properties.name << ")\n";
    std::cout << "GPU precision: complex128 (double precision)\n";
    std::cout << "Floquet update mode: "
              << floquet_mode_name(options.floquet_mode) << '\n';

    const int points = config.points_x;
    const auto initial_host = read_complex_hdf5(config.initial_state_file, points);
    const auto static_host = read_complex_hdf5(config.potential_file, points);
    std::vector<cuDoubleComplex> floquet_host(
        static_cast<std::size_t>(points), make_cuDoubleComplex(0.0, 0.0));
    if (config.floquet_potential) {
        floquet_host = read_complex_hdf5(config.floquet_potential_file, points);
    }

    DeviceBuffer<cuDoubleComplex> psi(points);
    DeviceBuffer<cuDoubleComplex> psi_half(points);
    DeviceBuffer<cuDoubleComplex> psi_k(points);
    DeviceBuffer<cuDoubleComplex> static_potential(points);
    DeviceBuffer<cuDoubleComplex> floquet_base(points);
    DeviceBuffer<cuDoubleComplex> floquet_work(points);
    DeviceBuffer<cuDoubleComplex> potential(points);
    DeviceBuffer<cuDoubleComplex> k_propagator(points);
    DeviceBuffer<double> k_squared(points);
    DeviceBuffer<double> density(points);

    copy_host_to_device(initial_host, psi);
    copy_host_to_device(static_host, static_potential);
    copy_host_to_device(floquet_host, floquet_base);
    copy_host_to_device(floquet_host, floquet_work);
    launch_copy_complex(static_potential.data(), potential.data(), points);
    launch_initialize_spectral(k_squared.data(), k_propagator.data(), points,
                               config.step_x, config.time_step,
                               config.imaginary_time);

    FftPlan1D fft_plan(points);
    Diagnostics1D diagnostics(points, config.step_x, config.beta);
    diagnostics.capture_initial_density(psi.data());
    const double initial_norm = diagnostics.norm(psi.data());
    std::cout << "Norm of initial state: " << initial_norm << '\n';

    StatusWriter status_writer(config.status_file);
    double total_time = 0.0;
    double floquet_factor = 0.0;
    double energy_before = 10.0e100;
    int completed_iterations = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();

    for (int count = 0; count < config.number_of_iterations; ++count) {
        if (config.floquet_potential) {
            floquet_factor = std::cos(config.floquet_omega * total_time);
            launch_update_floquet(static_potential.data(), floquet_base.data(),
                                  floquet_work.data(), potential.data(), points,
                                  floquet_factor, options.floquet_mode);
            total_time += config.time_step;
        }

        // CUDA PORT: first real-space half step uses |psi_n|^2.
        launch_density(psi.data(), density.data(), points);
        launch_position_half_step(psi.data(), psi_half.data(), potential.data(),
                                  density.data(), points, config.time_step,
                                  config.beta, config.imaginary_time);

        // CUDA PORT: MKL forward -> explicit 1/N -> kinetic -> MKL backward.
        // cuFFT is also unnormalized in both directions, so this preserves the
        // original normalization and ordering exactly.
        CUFFT_CHECK(cufftExecZ2Z(fft_plan.get(), psi_half.data(), psi_k.data(),
                                 CUFFT_FORWARD));
        launch_apply_kinetic_and_fft_scale(psi_k.data(), k_propagator.data(),
                                           points);
        CUFFT_CHECK(cufftExecZ2Z(fft_plan.get(), psi_k.data(), psi_half.data(),
                                 CUFFT_INVERSE));

        // CUDA PORT: second real-space half step recomputes the nonlinear term.
        launch_density(psi_half.data(), density.data(), points);
        launch_position_half_step(psi_half.data(), psi.data(), potential.data(),
                                  density.data(), points, config.time_step,
                                  config.beta, config.imaginary_time);

        if (config.imaginary_time) {
            const double current_norm = diagnostics.norm(psi.data());
            if (!(current_norm > 0.0) || !std::isfinite(current_norm)) {
                throw std::runtime_error("Imaginary-time normalization became invalid.");
            }
            launch_scale_complex(psi.data(), points,
                                 std::sqrt(initial_norm / current_norm));
        }

        // The historical sampling index zero means AFTER the first time step.
        // Compact mode samples that same schedule without writing field files.
        if (compact && phase_metric->wants_iteration(count)) {
            phase_metric->add(count, diagnostics.cropped_density_squared_sum(
                                        psi.data(), config.phase_metric_cut_points_each_edge));
        }
        // Preserve legacy single-run/validation snapshot behavior, including
        // its historical treatment of save_psi, outside compact mode.
        if (!compact && count % config.save_every_nth_iteration == 0) {
            save_device_field(snapshot_path(config.output_folder, count), psi);
            if (config.save_potential) {
                save_device_field(snapshot_path(config.potential_output_folder, count),
                                  potential);
            }
        }

        if (count % config.show_stats_every_nth_iteration == 0) {
            StatusData status;
            status.iteration = count;
            status.norm = diagnostics.norm(psi.data());
            const EnergyValues energy = diagnostics.energy(
                psi.data(), potential.data(), k_squared.data(), fft_plan.get());
            status.kinetic_energy = energy.kinetic;
            status.potential_energy = energy.potential;
            status.interaction_energy = energy.interaction;
            status.total_energy = energy.total;
            status.chemical_potential = energy.chemical_potential;
            status.mean_dynamic_potential = 0.0;
            status.initial_density_overlap =
                diagnostics.initial_density_overlap(psi.data());

            std::cout << count << " Norm: " << status.norm
                      << " CP: " << status.chemical_potential
                      << " Etot: " << status.total_energy
                      << " Ekin: " << status.kinetic_energy
                      << " Epot: " << status.potential_energy
                      << " Eint: " << status.interaction_energy
                      << " f_flo: " << floquet_factor
                      << " Corr: " << status.initial_density_overlap << '\n';
            status_writer.append(status);

            if (config.imaginary_time) {
                const double delta_energy = energy_before - status.total_energy;
                if (delta_energy < config.stop_delta_energy) {
                    std::cout << "Energy target met.\n";
                    save_device_field(snapshot_path(config.output_folder, count), psi);
                    completed_iterations = count + 1;
                    break;
                }
                std::cout << "DeltaE: " << delta_energy << '\n';
                energy_before = status.total_energy;
            }
        }
        completed_iterations = count + 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto finish = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(finish - start).count();
    if (compact) {
        phase_metric->write(config.phase_metric_file);
    }
    std::cout << "CUDA run complete: " << completed_iterations
              << " iterations in " << elapsed << " s ("
              << (elapsed > 0.0 ? completed_iterations / elapsed : 0.0)
              << " iterations/s).\n";
    return SolverResult{completed_iterations, elapsed};
}
