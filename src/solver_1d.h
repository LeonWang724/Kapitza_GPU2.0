// Codex CUDA Port: public interface for the native 1D CUDA evolution.
#pragma once

#include "config.h"
#include "kernels_1d.cuh"

struct SolverOptions {
    int device = 0;
    // Physical is the verified workflow default. Legacy remains callable only
    // for the source-compatibility validation probes.
    FloquetMode floquet_mode = FloquetMode::Physical;
};

struct SolverResult {
    int completed_iterations = 0;
    double elapsed_seconds = 0.0;
};

SolverResult run_solver_1d(const ConfigData& config,
                           const SolverOptions& options);
const char* floquet_mode_name(FloquetMode mode);
FloquetMode parse_floquet_mode(const char* value);
