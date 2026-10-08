// Codex CUDA Port: parser preserving the original key=value configuration interface.
#include "config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

bool boolean_value(const std::unordered_map<std::string, std::string>& values,
                   const std::string& key, bool default_value) {
    const auto iterator = values.find(key);
    if (iterator == values.end()) {
        return default_value;
    }
    if (iterator->second == "true") {
        return true;
    }
    if (iterator->second == "false") {
        return false;
    }
    throw std::runtime_error("Invalid boolean for config key '" + key + "': " +
                             iterator->second);
}

double double_value(const std::unordered_map<std::string, std::string>& values,
                    const std::string& key, double default_value) {
    const auto iterator = values.find(key);
    return iterator == values.end() ? default_value : std::stod(iterator->second);
}

int int_value(const std::unordered_map<std::string, std::string>& values,
              const std::string& key, int default_value) {
    const auto iterator = values.find(key);
    if (iterator == values.end()) {
        return default_value;
    }
    const double parsed = std::stod(iterator->second);
    if (parsed < static_cast<double>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Integer config value is out of range for key '" + key + "'.");
    }
    return static_cast<int>(parsed);
}

std::filesystem::path path_value(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key, const std::filesystem::path& default_value = {}) {
    const auto iterator = values.find(key);
    return iterator == values.end() ? default_value : std::filesystem::path(iterator->second);
}

}  // namespace

ConfigData read_config(const std::filesystem::path& config_path) {
    std::ifstream stream(config_path);
    if (!stream) {
        throw std::runtime_error("Unable to open config file: " + config_path.string());
    }

    ConfigData config;
    std::string line;
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#' || line.front() == ';') {
            continue;
        }
        const auto delimiter = line.find('=');
        if (delimiter == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, delimiter));
        const std::string value = trim(line.substr(delimiter + 1));
        config.raw_values[key] = value;
    }

    const auto threads = config.raw_values.find("number_of_threads");
    if (threads != config.raw_values.end() && threads->second != "dynamic") {
        config.number_of_threads = int_value(config.raw_values, "number_of_threads", 0);
    }
    config.number_of_iterations = int_value(config.raw_values, "number_of_iterations", 0);
    config.save_every_nth_iteration =
        int_value(config.raw_values, "save_every_nth_iteration", 0);
    config.show_stats_every_nth_iteration =
        int_value(config.raw_values, "show_stats_every_nth_iteration", 0);
    config.dimension = int_value(config.raw_values, "dimension", 1);

    config.potential_file = path_value(config.raw_values, "potential_file");
    config.floquet_potential_file = path_value(config.raw_values, "floquet_potential_file");
    config.initial_state_file = path_value(config.raw_values, "initial_state_file");
    config.status_file = path_value(config.raw_values, "status_file", "status.csv");
    config.output_folder = path_value(config.raw_values, "output_folder", "out");
    config.potential_output_folder =
        path_value(config.raw_values, "dp_potential_output_folder", "pot_output");

    config.points_x = int_value(config.raw_values, "points_x", 0);
    config.points_y = int_value(config.raw_values, "points_y", 1);
    config.points_z = int_value(config.raw_values, "points_z", 1);
    config.time_step = double_value(config.raw_values, "time_step", 0.0);
    config.step_x = double_value(config.raw_values, "step_x", 0.0);
    config.step_y = double_value(config.raw_values, "step_y", 1.0);
    config.step_z = double_value(config.raw_values, "step_z", 1.0);
    config.beta = double_value(config.raw_values, "beta", 0.0);

    config.imaginary_time = boolean_value(config.raw_values, "imaginary_time", false);
    config.dynamic_potential = boolean_value(config.raw_values, "dynamic_potential", false);
    config.floquet_potential = boolean_value(config.raw_values, "floquet_potential", false);
    config.floquet_omega = double_value(config.raw_values, "floquet_omega", 0.0);
    config.floquet_amplitude = double_value(config.raw_values, "floquet_amplitude", 1.0);

    config.dynamic_correlation_time =
        double_value(config.raw_values, "dp_correlation_time", 0.0);
    config.dynamic_amplitude = double_value(config.raw_values, "dp_amplitude", 0.0);
    config.dynamic_padding_factor_x =
        int_value(config.raw_values, "dp_padding_factor_x", 1);
    config.dynamic_padding_factor_y =
        int_value(config.raw_values, "dp_padding_factor_y", 1);
    config.dynamic_padding_factor_z =
        int_value(config.raw_values, "dp_padding_factor_z", 1);
    config.dynamic_interpolation_steps =
        int_value(config.raw_values, "dp_interpolation_steps", 1);
    config.save_potential = boolean_value(config.raw_values, "dp_save_potential", false);
    config.save_psi = boolean_value(config.raw_values, "save_psi", true);
    config.save_reduced = boolean_value(config.raw_values, "save_reduced", false);
    config.stop_delta_energy =
        double_value(config.raw_values, "stop_itime_on_energy_difference", 1.0e-7);
    config.phase_metric_file = path_value(config.raw_values, "phase_metric_file");
    config.phase_metric_final_snapshot_count =
        int_value(config.raw_values, "phase_metric_final_snapshot_count", 30);
    config.phase_metric_cut_points_each_edge =
        int_value(config.raw_values, "phase_metric_cut_points_each_edge", 100);
    return config;
}

void validate_1d_config(const ConfigData& config) {
    if (!config.phase_metric_file.empty()) {
        if (config.imaginary_time) {
            throw std::runtime_error("Compact phase metrics require real-time evolution.");
        }
        if (config.phase_metric_final_snapshot_count <= 0 ||
            config.phase_metric_cut_points_each_edge < 0 ||
            config.points_x <= 0 ||
            config.phase_metric_cut_points_each_edge > (config.points_x - 1) / 2) {
            throw std::runtime_error("Invalid compact phase metric window or spatial cut.");
        }
    }
    if (config.dimension != 1) {
        throw std::runtime_error(
            "This validated delivery implements dimension=1 only; 2D/3D are deferred.");
    }
    if (config.dynamic_potential) {
        throw std::runtime_error(
            "dynamic_potential=true is not part of the validated phase-diagram path yet. "
            "The stochastic CUDA port is deferred and this run has been stopped explicitly.");
    }
    if (config.points_x <= 0 || config.number_of_iterations <= 0) {
        throw std::runtime_error("points_x and number_of_iterations must be positive.");
    }
    if (config.save_every_nth_iteration <= 0 ||
        config.show_stats_every_nth_iteration <= 0) {
        throw std::runtime_error("Save and status intervals must be positive.");
    }
    if (!(config.time_step > 0.0) || !(config.step_x > 0.0)) {
        throw std::runtime_error("time_step and step_x must be positive.");
    }
    if (config.initial_state_file.empty() || config.potential_file.empty()) {
        throw std::runtime_error("initial_state_file and potential_file are required.");
    }
    if (config.floquet_potential && config.floquet_potential_file.empty()) {
        throw std::runtime_error(
            "floquet_potential_file is required when floquet_potential=true.");
    }
}

void validate_batch_compatible(const std::vector<ConfigData>& configs) {
    if (configs.empty()) {
        throw std::runtime_error("A batch needs at least one config.");
    }
    const ConfigData& first = configs.front();
    const bool compact = !first.phase_metric_file.empty();
    std::set<std::string> status_files;
    std::set<std::string> outputs;
    for (std::size_t index = 0; index < configs.size(); ++index) {
        const ConfigData& config = configs[index];
        const auto require = [&](bool same, const char* key) {
            if (!same) {
                throw std::runtime_error("Batched config " + std::to_string(index + 1) +
                                         " differs from the first in " + key + ".");
            }
        };
        require(config.dimension == first.dimension, "dimension");
        require(config.points_x == first.points_x, "points_x");
        require(config.step_x == first.step_x, "step_x");
        require(config.time_step == first.time_step, "time_step");
        require(config.beta == first.beta, "beta");
        require(config.number_of_iterations == first.number_of_iterations,
                "number_of_iterations");
        require(config.save_every_nth_iteration == first.save_every_nth_iteration,
                "save_every_nth_iteration");
        require(config.show_stats_every_nth_iteration ==
                    first.show_stats_every_nth_iteration,
                "show_stats_every_nth_iteration");
        require(config.imaginary_time == first.imaginary_time, "imaginary_time");
        require(config.dynamic_potential == first.dynamic_potential, "dynamic_potential");
        require(config.floquet_potential == first.floquet_potential, "floquet_potential");
        require(config.save_potential == first.save_potential, "dp_save_potential");
        require(config.phase_metric_file.empty() != compact, "phase_metric_file (output mode)");
        require(config.phase_metric_final_snapshot_count ==
                    first.phase_metric_final_snapshot_count,
                "phase_metric_final_snapshot_count");
        require(config.phase_metric_cut_points_each_edge ==
                    first.phase_metric_cut_points_each_edge,
                "phase_metric_cut_points_each_edge");

        // Every system writes its own results.
        const auto unique = [&](std::set<std::string>& seen,
                                const std::filesystem::path& path, const char* key) {
            if (!seen.insert(path.lexically_normal().string()).second) {
                throw std::runtime_error("Batched configs share the same " +
                                         std::string(key) + ".");
            }
        };
        unique(status_files, config.status_file, "status_file");
        unique(outputs, compact ? config.phase_metric_file : config.output_folder,
               compact ? "phase_metric_file" : "output_folder");
    }
}
