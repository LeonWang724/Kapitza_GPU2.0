// Codex CUDA Port: public interface for the native 1D CUDA evolution.
#pragma once

#include "config.h"
#include "kernels_1d.cuh"

#include <vector>

// How the host thread waits for the GPU. Blocking sleeps instead of spinning a
// CPU core, which matters when several worker tabs share one laptop's power
// and thermal budget.
enum class WaitMode {
    Auto,
    Spin,
    Yield,
    Blocking,
};

struct SolverOptions {
    int device = 0;
    // Physical is the verified workflow default. Legacy remains callable only
    // for the source-compatibility validation probes.
    FloquetMode floquet_mode = FloquetMode::Physical;
    // Run the original one-kernel-per-operation loop, kept as the reference
    // for validating the fused loop.
    bool reference_loop = false;
    WaitMode wait_mode = WaitMode::Auto;
};

struct SolverResult {
    int completed_iterations = 0;
    double elapsed_seconds = 0.0;
};

// Real-time runs with the physical Floquet update use the fused loop, which
// reproduces the reference loop bit for bit. Imaginary time, the legacy update
// and --reference-loop run the original loop.
SolverResult run_solver_1d(const ConfigData& config,
                           const SolverOptions& options);
// Evolves several compatible configs (see validate_batch_compatible) together
// on the GPU with the fused loop. Each config keeps its own inputs and outputs.
SolverResult run_solver_batch_1d(const std::vector<ConfigData>& configs,
                                 const SolverOptions& options);
bool fused_loop_supported(const ConfigData& config, const SolverOptions& options);
const char* floquet_mode_name(FloquetMode mode);
FloquetMode parse_floquet_mode(const char* value);
WaitMode parse_wait_mode(const char* value);
