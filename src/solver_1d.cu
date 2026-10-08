// Codex CUDA Port: persistent-GPU implementation of the original 1D split-step loop.
#include "solver_1d.h"

#include "cuda_checks.cuh"
#include "device_buffer.cuh"
#include "diagnostic_sums.h"
#include "diagnostics_1d.cuh"
#include "fused_kernels_1d.cuh"
#include "fused_loop.h"
#include "hdf5_io.h"
#include "phase_metric.h"

#include <cufft.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FftPlan1D {
public:
    explicit FftPlan1D(int points, int batch = 1) {
        CUFFT_CHECK(cufftPlan1d(&handle_, points, CUFFT_Z2Z, batch));
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

void copy_host_to_device(const std::vector<cuDoubleComplex>& host,
                         cuDoubleComplex* device, std::size_t count) {
    if (host.size() != count) {
        throw std::runtime_error("Host/device array length mismatch.");
    }
    CUDA_CHECK(cudaMemcpy(device, host.data(), count * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
}

void save_device_field(const std::filesystem::path& path,
                       const cuDoubleComplex* device, std::size_t count) {
    std::vector<cuDoubleComplex> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
    write_complex_hdf5(path, host);
}

void apply_wait_mode(WaitMode mode) {
    unsigned int flags = cudaDeviceScheduleAuto;
    switch (mode) {
        case WaitMode::Auto: return;
        case WaitMode::Spin: flags = cudaDeviceScheduleSpin; break;
        case WaitMode::Yield: flags = cudaDeviceScheduleYield; break;
        case WaitMode::Blocking: flags = cudaDeviceScheduleBlockingSync; break;
    }
    // Only scheduling changes; a driver refusal leaves the default in place.
    const cudaError_t status = cudaSetDeviceFlags(flags);
    if (status != cudaSuccess) {
        cudaGetLastError();
        std::cout << "Note: CUDA wait mode unchanged (" << cudaGetErrorName(status)
                  << ").\n";
    }
}

void select_device(const SolverOptions& options) {
    CUDA_CHECK(cudaSetDevice(options.device));
    apply_wait_mode(options.wait_mode);

    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, options.device));
    std::cout << "CUDA device: " << options.device << " ("
              << device_properties.name << ")\n";
    std::cout << "GPU precision: complex128 (double precision)\n";
    std::cout << "Floquet update mode: "
              << floquet_mode_name(options.floquet_mode) << '\n';
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

WaitMode parse_wait_mode(const char* value) {
    const std::string mode(value);
    if (mode == "auto") return WaitMode::Auto;
    if (mode == "spin") return WaitMode::Spin;
    if (mode == "yield") return WaitMode::Yield;
    if (mode == "blocking") return WaitMode::Blocking;
    throw std::runtime_error("Wait mode must be 'auto', 'spin', 'yield' or 'blocking'.");
}

namespace {

// The original loop, one kernel per operation. It remains the reference for
// the fused loop and serves imaginary time and the legacy Floquet update.
SolverResult run_reference_solver_1d(const ConfigData& config,
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
    select_device(options);
    std::cout << "Solver loop: reference\n";

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
            SpatialMoments spatial;
            const double metric = diagnostics.phase_statistics(
                psi.data(), config.phase_metric_cut_points_each_edge, spatial);
            phase_metric->add(count, metric, spatial);
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

constexpr int kSumStride =
    static_cast<int>(kStatusSumCount) + static_cast<int>(kPhaseSumCount);

// GPU state for the fused loop over one or more compatible configs. Each
// device array holds the systems' fields back to back.
class FusedGpuRun {
public:
    explicit FusedGpuRun(const std::vector<ConfigData>& configs)
        : configs_(configs),
          shape_(configs.front()),
          systems_(static_cast<int>(configs.size())),
          points_(configs.front().points_x),
          total_(static_cast<std::size_t>(points_) * configs.size()),
          compact_(!configs.front().phase_metric_file.empty()),
          clock_(omegas(configs), configs.front().time_step),
          psi_(total_),
          psi_half_(total_),
          psi_k_(total_),
          static_potential_(total_),
          floquet_base_(shape_.floquet_potential ? total_ : 0),
          potential_(total_),
          factor_cache_(shape_.beta == 0.0 ? total_ : 0),
          k_propagator_(points_),
          k_squared_(points_),
          sums_(static_cast<std::size_t>(systems_) * kSumStride),
          host_sums_(static_cast<std::size_t>(systems_) * kSumStride, 0.0),
          fft_plan_(points_, systems_) {
        constants_.time_step = shape_.time_step;
        constants_.beta = shape_.beta;
        constants_.floquet = shape_.floquet_potential;
        constants_.reuse_factor = shape_.beta == 0.0;

        for (int system = 0; system < systems_; ++system) {
            const ConfigData& config = configs_[system];
            copy_host_to_device(read_complex_hdf5(config.initial_state_file, points_),
                                psi_at(system), points_);
            copy_host_to_device(read_complex_hdf5(config.potential_file, points_),
                                static_potential_.data() + offset(system), points_);
            if (shape_.floquet_potential) {
                copy_host_to_device(
                    read_complex_hdf5(config.floquet_potential_file, points_),
                    floquet_base_.data() + offset(system), points_);
            }
        }
        CUDA_CHECK(cudaMemcpy(potential_.data(), static_potential_.data(),
                              potential_.bytes(), cudaMemcpyDeviceToDevice));
        launch_initialize_spectral(k_squared_.data(), k_propagator_.data(), points_,
                                   shape_.step_x, shape_.time_step, false);
        // Diagnostics transform one system at a time, as the original did.
        if (systems_ > 1) single_plan_ = std::make_unique<FftPlan1D>(points_, 1);

        for (int system = 0; system < systems_; ++system) {
            const ConfigData& config = configs_[system];
            diagnostics_.push_back(std::make_unique<Diagnostics1D>(
                points_, shape_.step_x, shape_.beta));
            diagnostics_.back()->capture_initial_density(psi_at(system));
            const double initial_norm = diagnostics_.back()->norm(psi_at(system));
            std::cout << prefix(system) << "Norm of initial state: " << initial_norm << '\n';
            if (compact_) {
                metrics_.emplace_back(config.number_of_iterations,
                                      config.save_every_nth_iteration,
                                      config.phase_metric_final_snapshot_count,
                                      config.phase_metric_cut_points_each_edge);
            }
            status_writers_.push_back(std::make_unique<StatusWriter>(config.status_file));
        }
    }

    // Steps interface of run_fused_loop.
    void advance_potential() {
        clock_.advance();
        previous_factors_ = make_system_factors(clock_.previous());
        current_factors_ = make_system_factors(clock_.current());
    }

    void first_half() {
        launch_fused_first_half(psi_.data(), psi_half_.data(), static_potential_.data(),
                                floquet_base_.data(), factor_cache_.data(), points_,
                                systems_, constants_, current_factors_);
    }

    void second_then_first_half() {
        launch_fused_second_then_first_half(
            psi_half_.data(), static_potential_.data(), floquet_base_.data(),
            factor_cache_.data(), points_, systems_, constants_, previous_factors_,
            current_factors_);
    }

    void kinetic() {
        // cuFFT is unnormalized both ways; the kinetic step applies 1/N.
        CUFFT_CHECK(cufftExecZ2Z(fft_plan_.get(), psi_half_.data(), psi_k_.data(),
                                 CUFFT_FORWARD));
        launch_batched_kinetic(psi_k_.data(), k_propagator_.data(), points_, systems_);
        CUFFT_CHECK(cufftExecZ2Z(fft_plan_.get(), psi_k_.data(), psi_half_.data(),
                                 CUFFT_INVERSE));
    }

    void second_half() {
        launch_fused_second_half(psi_half_.data(), psi_.data(), static_potential_.data(),
                                 floquet_base_.data(), factor_cache_.data(),
                                 potential_.data(), points_, systems_, constants_,
                                 current_factors_);
    }

    bool wants_state(int count) const {
        return samples(count) || snapshots(count) || reports_status(count);
    }

    // Same observations, in the same order, as the reference loop. All
    // systems' sums are reduced first and read back with one transfer.
    void observe(int count) {
        const bool sample = samples(count);
        const bool stats = reports_status(count);
        const int cut = shape_.phase_metric_cut_points_each_edge;
        for (int system = 0; sample && system < systems_; ++system) {
            diagnostics_[system]->enqueue_phase_sums(psi_at(system), cut,
                                                     phase_sums_device(system));
        }
        for (int system = 0; stats && system < systems_; ++system) {
            diagnostics_[system]->enqueue_status_sums(
                psi_at(system), potential_.data() + offset(system), k_squared_.data(),
                diagnostic_plan(), status_sums_device(system));
        }
        if (sample || stats) download_sums();

        if (sample) {
            bool centered_pass = false;
            for (int system = 0; system < systems_; ++system) {
                const double* sums = phase_sums_host(system);
                if (spatial_mass_is_positive(sums[kMassSum])) {
                    diagnostics_[system]->enqueue_variance_sum(
                        cut, spatial_mean_from_sums(sums), phase_sums_device(system));
                    centered_pass = true;
                }
            }
            if (centered_pass) download_sums();
            for (int system = 0; system < systems_; ++system) {
                const double* sums = phase_sums_host(system);
                metrics_[system].add(count, sums[kMetricSum], spatial_from_sums(sums));
            }
        }

        if (snapshots(count)) {
            for (int system = 0; system < systems_; ++system) {
                const ConfigData& config = configs_[system];
                save_device_field(snapshot_path(config.output_folder, count),
                                  psi_at(system), points_);
                if (config.save_potential) {
                    save_device_field(snapshot_path(config.potential_output_folder, count),
                                      potential_.data() + offset(system), points_);
                }
            }
        }

        for (int system = 0; stats && system < systems_; ++system) {
            const StatusData status = status_from_sums(
                count, status_sums_host(system), shape_.step_x, shape_.beta, points_);
            const double floquet_factor =
                shape_.floquet_potential ? clock_.current()[system] : 0.0;
            std::cout << prefix(system) << count << " Norm: " << status.norm
                      << " CP: " << status.chemical_potential
                      << " Etot: " << status.total_energy
                      << " Ekin: " << status.kinetic_energy
                      << " Epot: " << status.potential_energy
                      << " Eint: " << status.interaction_energy
                      << " f_flo: " << floquet_factor
                      << " Corr: " << status.initial_density_overlap << '\n';
            status_writers_[system]->append(status);
        }
    }

    void write_phase_metrics() const {
        for (int system = 0; compact_ && system < systems_; ++system) {
            metrics_[system].write(configs_[system].phase_metric_file);
        }
    }

private:
    static std::vector<double> omegas(const std::vector<ConfigData>& configs) {
        std::vector<double> values;
        for (const ConfigData& config : configs) values.push_back(config.floquet_omega);
        return values;
    }

    std::string prefix(int system) const {
        return systems_ > 1 ? "[" + std::to_string(system) + "] " : std::string();
    }

    bool samples(int count) const {
        return compact_ && metrics_.front().wants_iteration(count);
    }
    bool snapshots(int count) const {
        return !compact_ && count % shape_.save_every_nth_iteration == 0;
    }
    bool reports_status(int count) const {
        return count % shape_.show_stats_every_nth_iteration == 0;
    }

    std::size_t offset(int system) const {
        return static_cast<std::size_t>(system) * points_;
    }
    cuDoubleComplex* psi_at(int system) { return psi_.data() + offset(system); }
    cufftHandle diagnostic_plan() const {
        return single_plan_ ? single_plan_->get() : fft_plan_.get();
    }

    double* status_sums_device(int system) {
        return sums_.data() + static_cast<std::size_t>(system) * kSumStride;
    }
    double* phase_sums_device(int system) {
        return status_sums_device(system) + kStatusSumCount;
    }
    const double* status_sums_host(int system) const {
        return host_sums_.data() + static_cast<std::size_t>(system) * kSumStride;
    }
    const double* phase_sums_host(int system) const {
        return status_sums_host(system) + kStatusSumCount;
    }
    void download_sums() {
        CUDA_CHECK(cudaMemcpy(host_sums_.data(), sums_.data(), sums_.bytes(),
                              cudaMemcpyDeviceToHost));
    }

    const std::vector<ConfigData>& configs_;
    const ConfigData& shape_;
    int systems_;
    int points_;
    std::size_t total_;
    bool compact_;
    split_step::StepConstants constants_;
    FloquetClock clock_;
    SystemFactors previous_factors_{};
    SystemFactors current_factors_{};

    DeviceBuffer<cuDoubleComplex> psi_;
    DeviceBuffer<cuDoubleComplex> psi_half_;
    DeviceBuffer<cuDoubleComplex> psi_k_;
    DeviceBuffer<cuDoubleComplex> static_potential_;
    DeviceBuffer<cuDoubleComplex> floquet_base_;
    DeviceBuffer<cuDoubleComplex> potential_;
    DeviceBuffer<cuDoubleComplex> factor_cache_;
    DeviceBuffer<cuDoubleComplex> k_propagator_;
    DeviceBuffer<double> k_squared_;
    DeviceBuffer<double> sums_;
    std::vector<double> host_sums_;

    FftPlan1D fft_plan_;
    std::unique_ptr<FftPlan1D> single_plan_;
    std::vector<std::unique_ptr<Diagnostics1D>> diagnostics_;
    std::vector<PhaseMetricAccumulator> metrics_;
    std::vector<std::unique_ptr<StatusWriter>> status_writers_;
};

}  // namespace

bool fused_loop_supported(const ConfigData& config, const SolverOptions& options) {
    return !options.reference_loop && !config.imaginary_time &&
           options.floquet_mode == FloquetMode::Physical;
}

SolverResult run_solver_batch_1d(const std::vector<ConfigData>& configs,
                                 const SolverOptions& options) {
    if (configs.empty() || configs.size() > static_cast<std::size_t>(kMaxBatchSystems)) {
        throw std::runtime_error("A GPU batch needs between 1 and " +
                                 std::to_string(kMaxBatchSystems) + " configs.");
    }
    for (const ConfigData& config : configs) {
        validate_1d_config(config);
        if (!fused_loop_supported(config, options)) {
            throw std::runtime_error(
                "Batched runs need real-time evolution, the physical Floquet update "
                "and the fused loop.");
        }
    }
    validate_batch_compatible(configs);

    select_device(options);
    std::cout << "Solver loop: fused (bitwise identical to the reference loop)\n";
    std::cout << "Systems in this GPU batch: " << configs.size() << '\n';

    FusedGpuRun run(configs);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    const int completed_iterations =
        run_fused_loop(configs.front().number_of_iterations, run);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto finish = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(finish - start).count();
    run.write_phase_metrics();

    std::cout << "CUDA run complete: " << completed_iterations
              << " iterations in " << elapsed << " s ("
              << (elapsed > 0.0 ? completed_iterations / elapsed : 0.0)
              << " iterations/s).\n";
    if (configs.size() > 1) {
        std::cout << "Batch throughput: " << configs.size() << " systems, "
                  << (elapsed > 0.0 ? completed_iterations * configs.size() / elapsed : 0.0)
                  << " system-iterations/s.\n";
    }
    return SolverResult{completed_iterations, elapsed};
}

SolverResult run_solver_1d(const ConfigData& config,
                           const SolverOptions& options) {
    if (fused_loop_supported(config, options)) {
        return run_solver_batch_1d({config}, options);
    }
    return run_reference_solver_1d(config, options);
}
