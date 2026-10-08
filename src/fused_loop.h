// Control flow of the fused real-time loop. It has no CUDA dependency, so the
// host test drives exactly this code with CPU arithmetic.
//
// The original loop ran, for every iteration n: potential update, half step,
// forward FFT, kinetic step, inverse FFT, half step, then optional output.
// Here the trailing half step of n and the leading half step of n+1 run in one
// pointwise pass whenever nothing reads psi in between. Each value is still
// produced by the same operations in the same order.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// Floquet factors in the original sequence: iteration n uses
// cos(omega * t_n), where t_n is time_step accumulated n times.
class FloquetClock {
public:
    FloquetClock(std::vector<double> omegas, double time_step)
        : omegas_(std::move(omegas)),
          time_step_(time_step),
          previous_(omegas_.size(), 0.0),
          current_(omegas_.size(), 0.0) {}

    void advance() {
        previous_ = current_;
        for (std::size_t system = 0; system < omegas_.size(); ++system) {
            current_[system] = std::cos(omegas_[system] * total_time_);
        }
        total_time_ += time_step_;
    }

    const std::vector<double>& previous() const { return previous_; }
    const std::vector<double>& current() const { return current_; }

private:
    std::vector<double> omegas_;
    double time_step_;
    double total_time_ = 0.0;
    std::vector<double> previous_;
    std::vector<double> current_;
};

// Steps must provide:
//   advance_potential()      next iteration's Floquet factors
//   first_half()             psi -> psi_half
//   second_then_first_half() psi_half -> psi_half (previous then current)
//   kinetic()                forward FFT, kinetic factor, inverse FFT
//   second_half()            psi_half -> psi, current potential stored
//   bool wants_state(int)    whether psi is read after this iteration
//   observe(int)             read psi after this iteration
template <class Steps>
int run_fused_loop(int iterations, Steps& steps) {
    if (iterations <= 0) {
        throw std::runtime_error("The fused loop needs a positive iteration count.");
    }
    bool second_half_pending = false;
    for (int count = 0; count < iterations; ++count) {
        steps.advance_potential();
        if (second_half_pending) {
            steps.second_then_first_half();
        } else {
            steps.first_half();
        }
        steps.kinetic();
        const bool observe = steps.wants_state(count);
        const bool last = count + 1 == iterations;
        if (observe || last) {
            steps.second_half();
            if (observe) steps.observe(count);
        }
        second_half_pending = !(observe || last);
    }
    return iterations;
}
