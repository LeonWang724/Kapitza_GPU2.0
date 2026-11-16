// Codex CUDA Port: owned representation of the original gpe1d.config format.
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

struct ConfigData {
    int number_of_threads = 0;
    int number_of_iterations = 0;
    int save_every_nth_iteration = 0;
    int show_stats_every_nth_iteration = 0;
    int dimension = 1;

    std::filesystem::path potential_file;
    std::filesystem::path floquet_potential_file;
    std::filesystem::path initial_state_file;
    std::filesystem::path status_file;
    std::filesystem::path output_folder;
    std::filesystem::path potential_output_folder;

    int points_x = 0;
    int points_y = 1;
    int points_z = 1;
    double time_step = 0.0;
    double step_x = 0.0;
    double step_y = 1.0;
    double step_z = 1.0;
    double beta = 0.0;

    bool imaginary_time = false;
    bool dynamic_potential = false;
    bool floquet_potential = false;
    double floquet_omega = 0.0;
    // 5.0: the drive is cos(floquet_omega * t + floquet_phase). The default 0
    // keeps the original cos(floquet_omega * t) bit for bit.
    double floquet_phase = 0.0;
    double floquet_amplitude = 1.0;

    double dynamic_correlation_time = 0.0;
    double dynamic_amplitude = 0.0;
    int dynamic_padding_factor_x = 1;
    int dynamic_padding_factor_y = 1;
    int dynamic_padding_factor_z = 1;
    int dynamic_interpolation_steps = 1;
    bool save_potential = false;
    bool save_psi = true;
    bool save_reduced = false;
    double stop_delta_energy = 1.0e-7;

    // Nonempty path enables scalar-only real-time output. Legacy snapshot
    // behavior is unchanged when this key is absent.
    std::filesystem::path phase_metric_file;
    int phase_metric_final_snapshot_count = 30;
    int phase_metric_cut_points_each_edge = 100;

    // 6.0: a nonempty path writes the extended metrics of extended_metrics.h at
    // every status iteration (fused loop only). Lengths in solver units.
    std::filesystem::path extended_metrics_file;
    double extended_metrics_lattice_wavevector = 0.0;
    double extended_metrics_central_halfwidth = 0.0;
    double extended_metrics_site_offset = 0.0;
    double extended_metrics_pixel_width = 0.0;

    // 7.0: a positive width (solver length units) averages |psi|^2 over a
    // window of about that width before the position metrics (fused loop
    // only; see diagnostics_1d.cu). 0 leaves every result as before.
    double density_smoothing_width = 0.0;

    // 11.0: a nonempty path writes the final |psi|^2 binned into camera pixels
    // of camera_pixel_width (solver length units); see camera_profile.h.
    std::filesystem::path camera_profile_file;
    double camera_pixel_width = 0.0;

    std::unordered_map<std::string, std::string> raw_values;
};

ConfigData read_config(const std::filesystem::path& config_path);
void validate_1d_config(const ConfigData& config);
// Configs evolved together on the GPU may differ only in their input files,
// output paths and drive frequency (floquet_omega).
void validate_batch_compatible(const std::vector<ConfigData>& configs);
