// Codex CUDA Port: command-compatible entry point for gpe1d_cuda.exe.
#include "build_info.h"
#include "config.h"
#include "cuda_checks.cuh"
#include "fused_kernels_1d.cuh"
#include "solver_1d.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_version_json() {
    std::cout
        << "{\n"
        << "  \"program\": \"gpe1d_cuda\",\n"
        << "  \"compact_phase_metric_version\": 1,\n"
        << "  \"phase_statistics_version\": 1,\n"
        << "  \"fused_solver_version\": 1,\n"
        << "  \"max_batch_systems\": " << kMaxBatchSystems << ",\n"
        << "  \"port_version\": \"" << GPE_PORT_VERSION << "\",\n"
        << "  \"git_commit\": \"" << GPE_GIT_COMMIT << "\",\n"
        << "  \"git_dirty\": " << GPE_GIT_DIRTY << ",\n"
        << "  \"build_utc\": \"" << GPE_BUILD_UTC << "\",\n"
        << "  \"cuda_toolkit\": \"" << GPE_CUDA_TOOLKIT_VERSION << "\",\n"
        << "  \"cuda_architectures\": \"" << GPE_CUDA_ARCHITECTURES << "\",\n"
        << "  \"cxx_compiler\": \"" << GPE_CXX_COMPILER << "\",\n"
        << "  \"cuda_compiler\": \"" << GPE_CUDA_COMPILER << "\"\n"
        << "}\n";
}

void print_device_json(int selected_device) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (selected_device < 0 || selected_device >= count) {
        throw std::runtime_error("Requested CUDA device index is unavailable.");
    }
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, selected_device));
    int driver = 0;
    int runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::cout << "{\n"
              << "  \"device_index\": " << selected_device << ",\n"
              << "  \"device_count\": " << count << ",\n"
              << "  \"name\": \"" << properties.name << "\",\n"
              << "  \"compute_capability\": \"" << properties.major << '.'
              << properties.minor << "\",\n"
              << "  \"global_memory_bytes\": " << properties.totalGlobalMem << ",\n"
              << "  \"driver_version_integer\": " << driver << ",\n"
              << "  \"runtime_version_integer\": " << runtime << "\n"
              << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // Several configs evolve together as one GPU batch.
        std::vector<std::filesystem::path> config_paths;
        SolverOptions options;

        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--version-json") {
                print_version_json();
                return 0;
            }
            if (argument == "--device-info") {
                if (index + 1 < argc && argv[index + 1][0] != '-') {
                    options.device = std::stoi(argv[++index]);
                }
                print_device_json(options.device);
                return 0;
            }
            if (argument == "--device") {
                if (++index >= argc) throw std::runtime_error("--device requires an integer.");
                options.device = std::stoi(argv[index]);
                continue;
            }
            if (argument == "--floquet-mode") {
                if (++index >= argc) throw std::runtime_error("--floquet-mode requires a value.");
                options.floquet_mode = parse_floquet_mode(argv[index]);
                continue;
            }
            if (argument == "--reference-loop") {
                options.reference_loop = true;
                continue;
            }
            if (argument == "--wait") {
                if (++index >= argc) throw std::runtime_error("--wait requires a value.");
                options.wait_mode = parse_wait_mode(argv[index]);
                continue;
            }
            if (!argument.empty() && argument.front() == '-') {
                throw std::runtime_error("Unknown option: " + argument);
            }
            config_paths.emplace_back(argument);
        }

        if (config_paths.empty()) {
            std::cout << "Using standard config file.\n";
            config_paths.emplace_back("gpe1d.config");
        }
        if (config_paths.size() == 1) {
            run_solver_1d(read_config(config_paths.front()), options);
            return 0;
        }
        std::vector<ConfigData> configs;
        for (const auto& path : config_paths) configs.push_back(read_config(path));
        run_solver_batch_1d(configs, options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpe1d_cuda failed: " << error.what() << '\n';
        return 1;
    }
}
