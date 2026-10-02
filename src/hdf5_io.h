// Codex CUDA Port: CPU-side HDF5 and status CSV compatibility layer.
#pragma once

#include <cuComplex.h>

#include <filesystem>
#include <fstream>
#include <vector>

struct StatusData {
    int iteration = 0;
    double norm = 0.0;
    double total_energy = 0.0;
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    double interaction_energy = 0.0;
    double chemical_potential = 0.0;
    double mean_dynamic_potential = 0.0;
    double initial_density_overlap = 0.0;
};

std::vector<cuDoubleComplex> read_complex_hdf5(
    const std::filesystem::path& path, std::size_t expected_size);
void write_complex_hdf5(const std::filesystem::path& path,
                        const std::vector<cuDoubleComplex>& values);
std::filesystem::path snapshot_path(const std::filesystem::path& output_folder,
                                    int iteration);

class StatusWriter {
public:
    explicit StatusWriter(const std::filesystem::path& path);
    void append(const StatusData& status);

private:
    std::ofstream stream_;
};

