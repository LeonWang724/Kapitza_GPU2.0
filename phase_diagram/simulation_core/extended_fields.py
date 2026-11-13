"""6.0 extended metrics: CSV columns and their meaning (no heavy imports).

Three sources fill the columns when ExtendedMetrics = True:
  * the GPU simulation itself (no prefix), from sums the solver records at
    every status iteration (every 100 steps);
  * a q = 0 Floquet model (prefix q0_): the same Hamiltonian for one lattice
    site's Bloch state, sampled once per drive period (stroboscopically);
  * a classical ensemble (prefix cl_): point particles in the same lattice.
"final window" is the window of the existing metric: the last 30 samples.
"""

from __future__ import annotations

import math

# Raw sums written by the solver after iteration;drive;norm;kinetic_sum;
# density_overlap. Must match kExtendedSumNames in src/extended_metrics.h.
EXTENDED_SUM_NAMES = (
    "mass", "cos_sum", "sin_sum", "x1_sum", "x2_sum", "x3_sum", "x4_sum",
    "central_sum", "rho_log_rho_sum", "rho_squared_sum", "potential_sum",
    "site_squared_sum", "site_log_sum", "pixel_squared_sum",
    "return_real", "return_imag", "momentum_total", "momentum_zero", "momentum_second",
)

GPU_FIELDS = {
    "pendulum_cos": "Pendulum angle <cos 2kL x>: +1 all at the bottom of the sites, -1 all on top",
    "pendulum_cos_strobe": "Pendulum angle, only at the drive phase of t = 0 (last half)",
    "pendulum_cos_time_std": "Time fluctuation (std) of the pendulum angle",
    "onsite_spread": "On-site spread 1 - |<exp(2i kL x)>| (0 = all at one spot of every site)",
    "onsite_spread_strobe": "On-site spread, only at the drive phase of t = 0 (last half)",
    "width_growth_rate_per_ms": "Early growth rate of the width: slope of ln(sigma_x^2), first quarter (1/ms)",
    "momentum_growth_rate_per_ms": "Early growth rate of <p^2>: slope of ln<p^2>, first quarter (1/ms)",
    "spreading_speed_um_per_ms": "Spreading speed: slope of sigma_x over the last half (um/ms)",
    "sigma_x_um_initial": "Cloud width at the first status step (um)",
    "width_time_std_um": "Time fluctuation (std) of the cloud width (um)",
    "central_fraction": "Fraction of atoms within 60 um (two initial widths) of the centre",
    "escape_fraction": "Fraction of atoms lost to the absorbers by the end",
    "density_entropy": "Shannon entropy of |psi|^2 on the grid (nats)",
    "kurtosis": "Kurtosis <(x - mean)^4> / sigma^4 of |psi|^2 (3 for a Gaussian)",
    "site_ipr": "IPR of the lattice-site populations",
    "occupied_sites": "Number of occupied lattice sites, 1 / site_ipr",
    "site_entropy": "Shannon entropy of the lattice-site populations (nats)",
    "coarse_ipr": "IPR on 4 um pixels (like the camera)",
    "metric_initial": "IPR (metric) at the first status step",
    "return_probability": "Return probability |<psi_0|psi>|^2",
    "density_overlap": "Density overlap with the start, sum rho_0 rho / sum rho_0^2",
    "p_squared": "Momentum spread <p^2> in (hbar kL)^2",
    "order_0": "Fraction with |p| < hbar kL (zeroth diffraction order)",
    "order_2": "Fraction with hbar kL <= |p| < 3 hbar kL (+-2 hbar kL orders)",
    "energy_absorbed_er": "Energy absorbed in the time-averaged lattice, final - initial (E_R)",
}

Q0_FIELDS = {
    "q0_pendulum_cos": "q=0 model: pendulum angle <cos 2kL x>, stroboscopic",
    "q0_onsite_spread": "q=0 model: on-site spread 1 - |<exp(2i kL x)>|, stroboscopic",
    "q0_p_squared": "q=0 model: <p^2> in (hbar kL)^2",
    "q0_order_0": "q=0 model: population of the p = 0 order",
    "q0_order_2": "q=0 model: population of the +-2 hbar kL orders",
    "q0_energy_absorbed_er": "q=0 model: energy absorbed in the time-averaged lattice (E_R)",
    "q0_onsite_survival": "q=0 model: probability still in the starting on-site state",
    "q0_band0_population": "q=0 model: population of the lowest band of the time-averaged lattice",
    "q0_start_band_population": "q=0 model: population of the band that held most atoms at t = 0",
    "q0_excited_band_population": "q=0 model: population of bands 2 and up (counting from 0)",
    "q0_band_change": "q=0 model: change of the band populations from t = 0 (0 to 1)",
    "q0_floquet_participation": "q=0 model: number of Floquet states the start spreads over",
    "q0_slow_frequency_khz": "q=0 model: main frequency of the pendulum angle (kHz)",
    "q0_slow_frequency_omega0": "q=0 model: that frequency over f0, the lattice's harmonic frequency",
    "q0_echo_fidelity": "q=0 model: overlap at the end with a run at alpha * 1.01 (echo)",
}

CLASSICAL_FIELDS = {
    "cl_pendulum_cos": "Classical ensemble: <cos 2kL x>",
    "cl_near_start": "Classical ensemble: fraction still within a quarter site of the start",
    "cl_site_escape": "Classical ensemble: fraction that left its site at some time",
}

FIELD_TITLES = {**GPU_FIELDS, **Q0_FIELDS, **CLASSICAL_FIELDS}
EXTENDED_FIELDS = tuple(FIELD_TITLES)


def finite_or_none(value):
    """JSON-safe number: None for NaN or infinity."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def extended_columns(record: dict) -> dict:
    """Validated extended columns stored with a point (None where undefined)."""
    values = record.get("extended_metrics")
    if not isinstance(values, dict) or any(key not in values for key in EXTENDED_FIELDS):
        raise ValueError(f"Point {record.get('run_index')} is missing its extended metrics.")
    for key in EXTENDED_FIELDS:
        value = values[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Point {record.get('run_index')} has an invalid {key}: {value!r}.")
    return {key: values[key] for key in EXTENDED_FIELDS}
