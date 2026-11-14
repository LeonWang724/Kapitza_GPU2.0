// 6.0 extended metrics: kernels for the extra sums of each status iteration.
#pragma once

#include <cuComplex.h>

enum class DensityWeight {
    Cos,                // cos(parameter * x)
    Sin,                // sin(parameter * x)
    X1,                 // x
    X2,                 // x^2
    X3,                 // x^3
    X4,                 // x^4
    Central,            // 1 for |x| < parameter, else 0
    DensityLogDensity,  // log(rho), 0 where rho = 0
    Potential,          // Re(potential)
};

// terms[j] = rho[j] * weight(x_j), with x_j = (j - points / 2) * step_x.
void launch_density_weight_terms(const double* density, const cuDoubleComplex* potential,
                                 double* terms, int points, double step_x,
                                 DensityWeight weight, double parameter);
// For bin b covering grid indices [starts[b], starts[b + 1]) with mass m_b:
// squares[b] = m_b^2 and logs[b] = m_b log(m_b) (0 for an empty bin).
void launch_bin_mass_terms(const double* density, const int* starts, int bins,
                           double* squares, double* logs);
// 7.0: first[b] = m_b * centres[b] and second[b] = m_b * centres[b]^2.
void launch_bin_moment_terms(const double* density, const int* starts, const double* centres,
                             int bins, double* first, double* second);
// 7.0: output[j] = sum of input[i] for |i - j| <= half_width (inside the
// grid), divided by 2 half_width + 1: a moving average, zero beyond the grid.
void launch_box_smooth(const double* input, double* output, int points, int half_width);
// real[j] + i imag[j] = conj(initial[j]) * psi[j].
void launch_return_overlap_terms(const cuDoubleComplex* initial, const cuDoubleComplex* psi,
                                 double* real, double* imag, int points);
// terms[j] = |fft[j]|^2 where k_squared_min <= k_squared[j] < k_squared_max, else 0.
void launch_momentum_zone_terms(const cuDoubleComplex* fft, const double* k_squared,
                                double* terms, int points, double k_squared_min,
                                double k_squared_max);
