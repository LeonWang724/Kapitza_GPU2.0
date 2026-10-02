#include "config.h"
#include "phase_metric.h"

#include <cassert>
#include <functional>
#include <limits>
#include <vector>

void must_throw(const std::function<void()>& operation) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    for (int iterations : {1, 9, 30, 31, 347, 70000}) {
        for (int interval : {1, 4, 10, 100}) {
            for (int window : {1, 2, 30, 100}) {
                std::vector<int> saved;
                for (int i = 0; i < iterations; ++i) {
                    if (i % interval == 0) saved.push_back(i);
                }
                if (saved.size() > static_cast<std::size_t>(window)) {
                    saved.erase(saved.begin(), saved.end() - window);
                }
                PhaseMetricAccumulator metric(iterations, interval, window, 0);
                double expected = 0.0;
                std::size_t sampled = 0;
                for (int i = 0; i < iterations; ++i) {
                    if (metric.wants_iteration(i)) {
                        assert(i == saved.at(sampled++));
                        metric.add(i, i + 0.25);
                    }
                }
                for (int i : saved) expected += i + 0.25;
                assert(sampled == saved.size());
                assert(metric.mean() == expected / saved.size());
            }
        }
    }
    must_throw([] { PhaseMetricAccumulator invalid(0, 1, 30, 0); });
    must_throw([] { PhaseMetricAccumulator invalid(10, 0, 30, 0); });
    must_throw([] { PhaseMetricAccumulator invalid(10, 1, 0, 0); });
    must_throw([] { PhaseMetricAccumulator invalid(10, 1, 1, -1); });
    PhaseMetricAccumulator sample(21, 10, 2, 2);
    must_throw([&] { sample.mean(); });
    must_throw([&] { sample.add(0, 1.0); });
    must_throw([&] { sample.add(10, std::numeric_limits<double>::quiet_NaN()); });
    sample.add(10, 1.5);
    must_throw([&] { sample.add(10, 1.0); });
    sample.add(20, 2.5);
    assert(sample.mean() == 2.0);
    sample.write(argv[1]);

    const std::string config_path = std::string(argv[1]) + ".config";
    {
        std::ofstream file(config_path);
        file << "phase_metric_file=metric.json\nphase_metric_final_snapshot_count=2\n"
             << "phase_metric_cut_points_each_edge=0\n";
    }
    const auto parsed = read_config(config_path);
    assert(parsed.phase_metric_file == "metric.json");
    assert(parsed.phase_metric_final_snapshot_count == 2);
    assert(parsed.phase_metric_cut_points_each_edge == 0);

    ConfigData config;
    config.points_x = 12;
    config.number_of_iterations = 21;
    config.save_every_nth_iteration = 10;
    config.show_stats_every_nth_iteration = 1;
    config.time_step = 0.1;
    config.step_x = 0.25;
    config.initial_state_file = "initial.h5";
    config.potential_file = "potential.h5";
    config.phase_metric_file = "metric.json";
    config.phase_metric_cut_points_each_edge = 2;
    validate_1d_config(config);
    config.imaginary_time = true;
    must_throw([&] { validate_1d_config(config); });
    config.imaginary_time = false;
    config.phase_metric_cut_points_each_edge = 6;
    must_throw([&] { validate_1d_config(config); });
    config.phase_metric_file.clear();
    validate_1d_config(config);  // Legacy output ignores metric-only settings.
}
