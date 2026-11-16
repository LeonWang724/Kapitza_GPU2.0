// 11.0: the final density binned into camera pixels, written once per run so
// Python can measure the cloud exactly as the lab's camera analysis does
// (lab_measurement.py). Host-only and CUDA-free, so it is tested on a CPU.
#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

struct CameraProfile {
    long long first_pixel = 0;   // index k of the first pixel; pixel k is centred at k * width
    double pixel_width = 0.0;    // solver length units
    std::vector<double> mass;    // sum of |psi|^2 over the grid points in each pixel
};

// Grid point i sits at x_i = (i - points / 2) * step_x, as everywhere in the
// solver. Pixel k covers [(k - 1/2) w, (k + 1/2) w), so a pixel is centred on
// x = 0 and the profile of a mirror-symmetric cloud is symmetric.
inline long long camera_pixel_of(long long index, int points, double step_x, double width) {
    const double x = static_cast<double>(index - points / 2) * step_x;
    return static_cast<long long>(std::floor(x / width + 0.5));
}

inline CameraProfile camera_profile(const double* density, int points, double step_x,
                                    double pixel_width) {
    if (density == nullptr || points <= 0 || !(step_x > 0.0) ||
        !(pixel_width > step_x) || !std::isfinite(pixel_width)) {
        throw std::runtime_error("Invalid camera profile grid or pixel width.");
    }
    CameraProfile profile;
    profile.pixel_width = pixel_width;
    profile.first_pixel = camera_pixel_of(0, points, step_x, pixel_width);
    const long long last = camera_pixel_of(points - 1, points, step_x, pixel_width);
    profile.mass.assign(static_cast<std::size_t>(last - profile.first_pixel + 1), 0.0);
    for (long long i = 0; i < points; ++i) {
        const long long k = camera_pixel_of(i, points, step_x, pixel_width);
        profile.mass[static_cast<std::size_t>(k - profile.first_pixel)] += density[i];
    }
    return profile;
}

inline void write_camera_profile(const std::filesystem::path& path,
                                 const CameraProfile& profile) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\n  \"camera_profile_version\": 1,\n"
           << "  \"pixel_width\": " << profile.pixel_width << ",\n"
           << "  \"first_pixel\": " << profile.first_pixel << ",\n"
           << "  \"count\": " << profile.mass.size() << ",\n"
           << "  \"mass\": [";
    for (std::size_t k = 0; k < profile.mass.size(); ++k) {
        if (!std::isfinite(profile.mass[k])) {
            throw std::runtime_error("The camera profile is not finite.");
        }
        output << (k ? ", " : "") << profile.mass[k];
    }
    output << "]\n}\n";
    output.close();
    if (!output) {
        throw std::runtime_error("Unable to write the camera profile: " + path.string());
    }
}
