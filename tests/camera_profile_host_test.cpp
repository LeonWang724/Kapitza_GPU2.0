// 11.0: camera_profile.h on a CPU. Writes the profile of a fixed test density
// to argv[1]; test_kapitza11.py compares it with lab_measurement.camera_profile.
#include "camera_profile.h"

#include <cassert>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    assert(argc == 2);
    const int points = 20000;
    const double width = 169.34455556399354;    // 3.895 um in 23 nm grid steps
    std::vector<double> density(points);
    double total = 0.0;
    for (int i = 0; i < points; ++i) {
        density[i] = static_cast<double>((i * 37) % 101) / 101.0;
        total += density[i];
    }
    const CameraProfile profile = camera_profile(density.data(), points, 1.0, width);
    double sum = 0.0;
    for (double value : profile.mass) sum += value;
    assert(std::abs(sum - total) <= 1e-9 * total);
    assert(camera_pixel_of(points / 2, points, 1.0, width) == 0);     // pixel 0 at x = 0
    bool threw = false;
    try {
        camera_profile(density.data(), points, 1.0, 0.5);              // narrower than a step
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
    write_camera_profile(argv[1], profile);
    std::puts("camera profile ok");
    return 0;
}
