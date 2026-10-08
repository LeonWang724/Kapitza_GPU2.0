// Host equivalence test for the fused loop. The original kernels are
// transcribed below as CPU loops; run_fused_loop drives the shared per-element
// functions of split_step_math.cuh. With the same FFT for both, every observed
// wavefunction and potential must agree bit for bit. Compile without FMA
// contraction (-ffp-contract=off), as the CUDA build does (--fmad=false).
#include "config.h"
#include "fused_loop.h"
#include "phase_metric.h"
#include "split_step_math.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Field = std::vector<cuDoubleComplex>;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    }
}

bool bitwise_equal(const cuDoubleComplex* a, const cuDoubleComplex* b, int count) {
    return std::memcmp(a, b, sizeof(cuDoubleComplex) * static_cast<std::size_t>(count)) == 0;
}

// Deterministic radix-2 FFT, unnormalized in both directions like cuFFT/MKL.
void fft(cuDoubleComplex* data, int points, int sign) {
    for (int i = 1, j = 0; i < points; ++i) {
        int bit = points >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (int length = 2; length <= points; length <<= 1) {
        for (int start = 0; start < points; start += length) {
            for (int k = 0; k < length / 2; ++k) {
                const double angle = sign * 2.0 * pi * k / length;
                const cuDoubleComplex twiddle = make_cuDoubleComplex(std::cos(angle), std::sin(angle));
                const cuDoubleComplex even = data[start + k];
                const cuDoubleComplex odd = cuCmul(data[start + k + length / 2], twiddle);
                data[start + k] = cuCadd(even, odd);
                data[start + k + length / 2] =
                    make_cuDoubleComplex(even.x - odd.x, even.y - odd.y);
            }
        }
    }
}

void fft_out_of_place(const cuDoubleComplex* input, cuDoubleComplex* output,
                      int points, int sign) {
    std::memcpy(output, input, sizeof(cuDoubleComplex) * static_cast<std::size_t>(points));
    fft(output, points, sign);
}

// --- Original kernels of kernels_1d.cu, transcribed literally -------------

cuDoubleComplex original_complex_exp(cuDoubleComplex value) {
    const double magnitude = exp(value.x);
    return make_cuDoubleComplex(magnitude * cos(value.y), magnitude * sin(value.y));
}

Field original_k_propagator(int points, double step_x, double time_step) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double delta_k = 2.0 * pi / (static_cast<double>(points) * step_x);
    Field propagator(points);
    for (int index = 0; index < points; ++index) {
        const int signed_index = index >= points / 2 ? index - points : index;
        const double k = static_cast<double>(signed_index) * delta_k;
        const double k2 = k * k;
        const double exponent = -0.5 * time_step * k2;
        propagator[index] = make_cuDoubleComplex(cos(exponent), sin(exponent));
    }
    return propagator;
}

void original_update_floquet(const Field& static_potential, const Field& floquet_base,
                             Field& total_potential, double factor) {
    for (std::size_t index = 0; index < total_potential.size(); ++index) {
        const cuDoubleComplex driven = make_cuDoubleComplex(
            floquet_base[index].x * factor, floquet_base[index].y * factor);
        total_potential[index] = cuCadd(static_potential[index], driven);
    }
}

void original_density(const Field& psi, std::vector<double>& density) {
    for (std::size_t index = 0; index < psi.size(); ++index) {
        const double real = psi[index].x;
        const double imag = psi[index].y;
        density[index] = real * real + imag * imag;
    }
}

void original_position_half_step(const Field& input, Field& output, const Field& potential,
                                 const std::vector<double>& density, double time_step,
                                 double beta) {
    const bool imaginary_time = false;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const cuDoubleComplex position_factor = imaginary_time
            ? make_cuDoubleComplex(-0.5 * time_step, 0.0)
            : make_cuDoubleComplex(0.0, -0.5 * time_step);
        const cuDoubleComplex interaction_factor = imaginary_time
            ? make_cuDoubleComplex(-0.5 * beta * time_step, 0.0)
            : make_cuDoubleComplex(0.0, -0.5 * beta * time_step);
        cuDoubleComplex exponent = cuCmul(position_factor, potential[index]);
        exponent = cuCadd(exponent,
                          make_cuDoubleComplex(interaction_factor.x * density[index],
                                               interaction_factor.y * density[index]));
        output[index] = cuCmul(input[index], original_complex_exp(exponent));
    }
}

void original_kinetic(Field& psi_k, const Field& propagator, double fft_scale) {
    for (std::size_t index = 0; index < psi_k.size(); ++index) {
        const cuDoubleComplex product = cuCmul(psi_k[index], propagator[index]);
        psi_k[index] = make_cuDoubleComplex(product.x * fft_scale, product.y * fft_scale);
    }
}

// --- Test problem ------------------------------------------------------------

struct Case {
    std::string name;
    int points = 64;
    int iterations = 1;
    int save_every = 100;
    int show_stats = 100;
    bool compact = true;
    int window = 30;
    double time_step = 0.05;
    double step_x = 0.5;
    double beta = 0.0;
    bool floquet = true;
    std::vector<double> omegas{1.7};
};

struct SystemInputs {
    Field psi;
    Field static_potential;
    Field floquet_base;
};

SystemInputs make_inputs(const Case& test, int system) {
    SystemInputs inputs;
    const int n = test.points;
    for (int index = 0; index < n; ++index) {
        const double x = (index - n / 2) * test.step_x;
        const double envelope = std::exp(-x * x / 40.0);
        inputs.psi.push_back(make_cuDoubleComplex(envelope * std::cos(0.3 * x + system),
                                                  envelope * std::sin(0.3 * x + system)));
        // Absorbing edges (nonzero imaginary part) around a real interior, whose
        // imaginary parts are -0.0 or +0.0 as numpy may produce either.
        const double edge = std::fabs(x) / (n / 2 * test.step_x);
        const double absorber = edge > 0.7 ? -0.2 * std::pow(edge - 0.7, 4) : (index % 2 ? -0.0 : 0.0);
        inputs.static_potential.push_back(make_cuDoubleComplex(
            -(1.0 + system) * std::cos(0.8 * x), absorber));
        inputs.floquet_base.push_back(make_cuDoubleComplex(-0.6 * std::cos(0.8 * x + 0.1), 0.0));
    }
    return inputs;
}

using Observations = std::map<int, std::vector<Field>>;  // iteration -> {psi, potential}

bool wants_state(const Case& test, const PhaseMetricAccumulator& metric, int count) {
    return (test.compact && metric.wants_iteration(count)) ||
           (!test.compact && count % test.save_every == 0) || count % test.show_stats == 0;
}

// The real-time loop of run_reference_solver_1d, recording every iteration.
Observations reference_run(const Case& test, const SystemInputs& inputs, double omega) {
    const int n = test.points;
    Field psi = inputs.psi, psi_half(n), psi_k(n);
    Field potential = inputs.static_potential;
    const Field propagator = original_k_propagator(n, test.step_x, test.time_step);
    std::vector<double> density(n);
    double total_time = 0.0;
    double floquet_factor = 0.0;
    Observations observed;
    for (int count = 0; count < test.iterations; ++count) {
        if (test.floquet) {
            floquet_factor = std::cos(omega * total_time);
            original_update_floquet(inputs.static_potential, inputs.floquet_base, potential,
                                    floquet_factor);
            total_time += test.time_step;
        }
        original_density(psi, density);
        original_position_half_step(psi, psi_half, potential, density, test.time_step, test.beta);
        fft_out_of_place(psi_half.data(), psi_k.data(), n, -1);
        original_kinetic(psi_k, propagator, 1.0 / static_cast<double>(n));
        fft_out_of_place(psi_k.data(), psi_half.data(), n, +1);
        original_density(psi_half, density);
        original_position_half_step(psi_half, psi, potential, density, test.time_step, test.beta);
        observed[count] = {psi, potential};
    }
    return observed;
}

// CPU implementation of the fused steps over a batch of systems.
class CpuFusedSteps {
public:
    CpuFusedSteps(const Case& test, const std::vector<SystemInputs>& inputs)
        : test_(test),
          systems_(static_cast<int>(inputs.size())),
          n_(test.points),
          clock_(test.omegas, test.time_step),
          metric_(test.iterations, test.save_every, test.window, 0),
          propagator_(original_k_propagator(n_, test.step_x, test.time_step)) {
        for (const SystemInputs& system : inputs) {
            psi_.insert(psi_.end(), system.psi.begin(), system.psi.end());
            static_.insert(static_.end(), system.static_potential.begin(),
                           system.static_potential.end());
            base_.insert(base_.end(), system.floquet_base.begin(), system.floquet_base.end());
        }
        psi_half_.resize(psi_.size());
        psi_k_.resize(psi_.size());
        potential_ = static_;
        cache_.assign(psi_.size(), make_cuDoubleComplex(0.0, 0.0));
        constants_.time_step = test.time_step;
        constants_.beta = test.beta;
        constants_.floquet = test.floquet;
        constants_.reuse_factor = test.beta == 0.0;
    }

    void advance_potential() { clock_.advance(); }

    void first_half() {
        for_each([&](int system, std::size_t e) {
            const cuDoubleComplex v = split_step::potential_at(constants_, static_[e], base_[e],
                                                               clock_.current()[system]);
            cuDoubleComplex factor;
            psi_half_[e] = split_step::first_half(constants_, psi_[e], v, &factor);
            if (constants_.reuse_factor) cache_[e] = factor;
        });
    }

    void second_then_first_half() {
        for_each([&](int system, std::size_t e) {
            const cuDoubleComplex previous = split_step::potential_at(
                constants_, static_[e], base_[e], clock_.previous()[system]);
            const cuDoubleComplex current = split_step::potential_at(
                constants_, static_[e], base_[e], clock_.current()[system]);
            cuDoubleComplex factor = constants_.reuse_factor ? cache_[e] : make_cuDoubleComplex(0.0, 0.0);
            psi_half_[e] = split_step::second_then_first_half(constants_, psi_half_[e],
                                                              previous, current, &factor);
            if (constants_.reuse_factor) cache_[e] = factor;
        });
    }

    void kinetic() {
        for (int system = 0; system < systems_; ++system) {
            const std::size_t offset = static_cast<std::size_t>(system) * n_;
            fft_out_of_place(psi_half_.data() + offset, psi_k_.data() + offset, n_, -1);
            for (int index = 0; index < n_; ++index) {
                psi_k_[offset + index] = split_step::kinetic(
                    psi_k_[offset + index], propagator_[index], 1.0 / static_cast<double>(n_));
            }
            fft_out_of_place(psi_k_.data() + offset, psi_half_.data() + offset, n_, +1);
        }
    }

    void second_half() {
        for_each([&](int system, std::size_t e) {
            const cuDoubleComplex v = split_step::potential_at(constants_, static_[e], base_[e],
                                                               clock_.current()[system]);
            psi_[e] = split_step::second_half(constants_, psi_half_[e], v, cache_[e]);
            potential_[e] = v;
        });
    }

    bool wants_state(int count) const { return ::wants_state(test_, metric_, count); }

    void observe(int count) {
        observed_.emplace_back(count);
        if (test_.compact && metric_.wants_iteration(count)) metric_.add(count, 0.0);
    }

    const std::vector<int>& observed() const { return observed_; }
    const cuDoubleComplex* psi(int system) const { return psi_.data() + offset(system); }
    const cuDoubleComplex* potential(int system) const { return potential_.data() + offset(system); }
    // Snapshot of every system after an observation.
    std::vector<Field> snapshot() const { return {psi_, potential_}; }

private:
    template <class Function>
    void for_each(Function function) {
        for (int system = 0; system < systems_; ++system) {
            for (int index = 0; index < n_; ++index) {
                function(system, static_cast<std::size_t>(system) * n_ + index);
            }
        }
    }
    std::size_t offset(int system) const { return static_cast<std::size_t>(system) * n_; }

    const Case& test_;
    int systems_;
    int n_;
    split_step::StepConstants constants_;
    FloquetClock clock_;
    PhaseMetricAccumulator metric_;
    Field propagator_;
    Field psi_, psi_half_, psi_k_, static_, base_, potential_, cache_;
    std::vector<int> observed_;
};

// Records state at each observation by wrapping the CPU steps.
class RecordingSteps : public CpuFusedSteps {
public:
    using CpuFusedSteps::CpuFusedSteps;
    void observe(int count) {
        CpuFusedSteps::observe(count);
        states[count] = snapshot();
    }
    std::map<int, std::vector<Field>> states;
};

void run_case(const Case& test) {
    std::vector<SystemInputs> inputs;
    for (std::size_t system = 0; system < test.omegas.size(); ++system) {
        inputs.push_back(make_inputs(test, static_cast<int>(system)));
    }
    RecordingSteps fused(test, inputs);
    run_fused_loop(test.iterations, fused);

    PhaseMetricAccumulator schedule(test.iterations, test.save_every, test.window, 0);
    std::vector<int> expected_observations;
    for (int count = 0; count < test.iterations; ++count) {
        if (wants_state(test, schedule, count)) expected_observations.push_back(count);
    }
    check(fused.observed() == expected_observations, test.name + ": observation schedule");

    const int n = test.points;
    for (std::size_t system = 0; system < inputs.size(); ++system) {
        const Observations reference = reference_run(test, inputs[system], test.omegas[system]);
        const std::size_t offset = system * static_cast<std::size_t>(n);
        for (const auto& [count, state] : fused.states) {
            const std::vector<Field>& expected = reference.at(count);
            check(bitwise_equal(state[0].data() + offset, expected[0].data(), n),
                  test.name + ": psi differs at iteration " + std::to_string(count) +
                      " for system " + std::to_string(system));
            check(bitwise_equal(state[1].data() + offset, expected[1].data(), n),
                  test.name + ": potential differs at iteration " + std::to_string(count));
        }
        // The loop finalizes psi after the last iteration even when unobserved.
        check(bitwise_equal(fused.psi(static_cast<int>(system)),
                            reference.at(test.iterations - 1)[0].data(), n),
              test.name + ": final psi differs for system " + std::to_string(system));
    }
}

void test_batch_compatibility() {
    const auto config = [](int index) {
        ConfigData c;
        c.points_x = 64;
        c.step_x = 1.0;
        c.time_step = 0.5;
        c.number_of_iterations = 100;
        c.save_every_nth_iteration = 10;
        c.show_stats_every_nth_iteration = 10;
        c.floquet_potential = true;
        c.floquet_omega = 0.1 * index;
        c.status_file = "status_" + std::to_string(index) + ".csv";
        c.phase_metric_file = "metric_" + std::to_string(index) + ".json";
        c.initial_state_file = "psi_" + std::to_string(index) + ".h5";
        return c;
    };
    const auto rejects = [](const std::vector<ConfigData>& configs) {
        try {
            validate_batch_compatible(configs);
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    check(!rejects({config(0), config(1), config(2)}), "compatible batch accepted");
    for (const auto& change : std::vector<std::function<void(ConfigData&)>>{
             [](ConfigData& c) { c.points_x = 128; },
             [](ConfigData& c) { c.step_x = 0.5; },
             [](ConfigData& c) { c.time_step = 0.25; },
             [](ConfigData& c) { c.beta = 1.0; },
             [](ConfigData& c) { c.number_of_iterations = 99; },
             [](ConfigData& c) { c.save_every_nth_iteration = 5; },
             [](ConfigData& c) { c.show_stats_every_nth_iteration = 5; },
             [](ConfigData& c) { c.floquet_potential = false; },
             [](ConfigData& c) { c.phase_metric_file.clear(); },
             [](ConfigData& c) { c.phase_metric_cut_points_each_edge = 3; },
             [](ConfigData& c) { c.phase_metric_final_snapshot_count = 3; },
             [](ConfigData& c) { c.status_file = "status_0.csv"; },
             [](ConfigData& c) { c.phase_metric_file = "metric_0.json"; },
         }) {
        ConfigData changed = config(1);
        change(changed);
        check(rejects({config(0), changed}), "incompatible batch rejected");
    }
    check(rejects({}), "empty batch rejected");
}

}  // namespace

int main() {
    std::vector<Case> cases;
    const std::vector<Case> schedules = {
        {"single_step", 64, 1, 100, 100, true, 30},
        {"two_steps_every_snapshot", 64, 2, 1, 100, false, 30},
        {"short_window", 64, 9, 4, 100, true, 30},
        {"every_step_sampled", 64, 101, 1, 100, true, 30},
        {"phase_diagram_like", 64, 347, 10, 100, true, 30},
        {"odd_stats_full", 64, 347, 10, 7, false, 30},
        {"single_sample", 128, 50, 100, 100, true, 1},
    };
    for (const Case& schedule : schedules) {
        for (double beta : {0.0, -0.0, 0.37}) {
            for (bool floquet : {true, false}) {
                for (const auto& omegas : std::vector<std::vector<double>>{{1.7}, {1.7, 0.31, 2.9}}) {
                    Case test = schedule;
                    test.beta = beta;
                    test.floquet = floquet;
                    test.omegas = omegas;
                    test.name = schedule.name + " beta=" + std::to_string(beta) +
                                (std::signbit(beta) ? "(-0)" : "") +
                                (floquet ? " floquet" : " static") +
                                " systems=" + std::to_string(omegas.size());
                    cases.push_back(test);
                }
            }
        }
    }
    for (const Case& test : cases) run_case(test);
    test_batch_compatibility();

    // Sanity check that the comparison is sensitive: one ulp must be detected.
    cuDoubleComplex a = make_cuDoubleComplex(1.0, 2.0);
    cuDoubleComplex b = make_cuDoubleComplex(std::nextafter(1.0, 2.0), 2.0);
    check(!bitwise_equal(&a, &b, 1), "bitwise comparison detects one ulp");

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", failures);
        return 1;
    }
    std::printf("Fused loop matched the reference loop bit for bit in %zu cases.\n", cases.size());
    return 0;
}
