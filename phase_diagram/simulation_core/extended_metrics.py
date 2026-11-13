"""6.0 extended metrics: solver settings, the summary of the solver's
extended time series, a q = 0 Floquet model and a classical ensemble.

All three use the 5.0 Hamiltonian
    H(t) = p^2/2m - V0 [1 + alpha sin(omega t + phi)] cos^2(kL x),
with the atoms starting in the ground state of the lattice at t = 0 at the
initial depth, -V_init [1 + alpha sin(phi)] cos^2(kL x). The column meanings
are in extended_fields.py.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from create_initial_state_function import (
    ENVELOPE_SIGMA_M, GRID_SPACING_M, bloch_state_1d_plane_wave, hbar, mli,
)
from cuda_workflow_common import point_drive_phase
from extended_fields import EXTENDED_FIELDS, EXTENDED_SUM_NAMES, finite_or_none
from level_statistics import (
    E_R, HBAR, _A1, _A2, _C1, _C2, _phase_mismatch, lattice_matrices, symmetry_sectors,
)

WAVELENGTH_M = 1064e-9
CENTRAL_HALFWIDTH_M = 2.0 * ENVELOPE_SIGMA_M      # two initial cloud widths, 60 um
PIXEL_M = 4e-6                                     # coarse graining, like the camera
Q0_PLANE_WAVE_CUTOFF = 40
ECHO_RELATIVE_CHANGE = 0.01
CLASSICAL_PARTICLES = 128
CLASSICAL_TIME_STEP = 0.02                         # in units of 1 / omega0
CLASSICAL_SEED = 20261112
STROBE_HALF_WIDTH = math.pi / 8.0
SERIES_COLUMNS = ("iteration", "drive", "norm", "kinetic_sum", "density_overlap") \
    + EXTENDED_SUM_NAMES


def contract_settings() -> dict:
    """The extended-metric settings recorded in the run manifest."""
    return {
        "version": 1,
        "central_halfwidth_um": CENTRAL_HALFWIDTH_M * 1e6,
        "pixel_um": PIXEL_M * 1e6,
        "q0_plane_wave_cutoff": Q0_PLANE_WAVE_CUTOFF,
        "echo_relative_change": ECHO_RELATIVE_CHANGE,
        "classical_particles": CLASSICAL_PARTICLES,
        "classical_time_step_per_omega0": CLASSICAL_TIME_STEP,
        "strobe_half_width_radians": STROBE_HALF_WIDTH,
    }


# --------------------------------------------------------------------------
# Solver settings and the exact NumPy equivalent of its sums
# --------------------------------------------------------------------------
def solver_wavevector() -> float:
    """kL in solver units (1 / grid spacing)."""
    return 2.0 * math.pi / WAVELENGTH_M * GRID_SPACING_M


def loading_coefficient(grid: dict, alpha: float) -> float:
    """c of the loading lattice c cos^2(kL x), in E_R: -V_init [1 + alpha sin(phi)]."""
    phase = point_drive_phase(grid, alpha)
    if phase is None:
        raise ValueError("Extended metrics need a 5.0-style grid with INVERTED.")
    return -float(grid["initial_lattice_depth_v0_er"]) * (1.0 + float(alpha) * math.sin(phase))


def solver_settings(grid: dict, alpha: float) -> dict:
    """Config keys for the solver. Sites are centred where the atoms start."""
    wavevector = solver_wavevector()
    period = math.pi / wavevector
    # c < 0: atoms at cos^2 = 1 (x = n a), edges halfway; c > 0: atoms at x = (n + 1/2) a.
    offset = 0.5 * period if loading_coefficient(grid, alpha) < 0.0 else 0.0
    return {
        "extended_metrics_lattice_wavevector": repr(wavevector),
        "extended_metrics_central_halfwidth": repr(CENTRAL_HALFWIDTH_M / GRID_SPACING_M),
        "extended_metrics_site_offset": repr(offset),
        "extended_metrics_pixel_width": repr(PIXEL_M / GRID_SPACING_M),
    }


def bin_starts(points: int, cut: int, step_x: float, width: float, offset: float) -> np.ndarray:
    """Same as bin_starts() in src/extended_metrics.h."""
    if cut < 0 or points <= 2 * cut:
        raise ValueError("The spatial cut removes all grid points.")
    if not step_x > 0.0 or not width > step_x or not math.isfinite(offset):
        raise ValueError("Extended-metric bins must be wider than the grid step.")
    end = points - cut
    half = float(points // 2)

    def position(index):
        return float(index - points // 2) * step_x

    first_bin = math.floor((position(cut) - offset) / width)
    last_bin = math.floor((position(end - 1) - offset) / width)
    starts = [cut]
    for bin_index in range(first_bin + 1, last_bin + 1):
        edge = offset + float(bin_index) * width
        index = min(max(int(math.ceil(edge / step_x + half)), cut), end)
        if index > starts[-1]:
            starts.append(index)
    starts.append(end)
    return np.array(starts, dtype=np.int64)


def reference_extended_sums(psi, initial_psi, potential, *, cut: int, step_x: float,
                            lattice_wavevector: float, central_halfwidth: float,
                            site_offset: float, pixel_width: float) -> np.ndarray:
    """The solver's raw extended sums for one state, in the order of EXTENDED_SUM_NAMES."""
    psi = np.asarray(psi, dtype=complex)
    points = psi.size
    rho = psi.real ** 2 + psi.imag ** 2
    window = slice(cut, points - cut)
    x = (np.arange(points) - points // 2).astype(float) * step_x
    r, xw = rho[window], x[window]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_log = np.where(r > 0.0, r * np.log(np.where(r > 0.0, r, 1.0)), 0.0)

    def bins(width, offset):
        starts = bin_starts(points, cut, step_x, width, offset)
        mass = np.add.reduceat(rho[:starts[-1]], starts[:-1])
        logs = np.where(mass > 0.0, mass * np.log(np.where(mass > 0.0, mass, 1.0)), 0.0)
        return float(np.sum(mass ** 2)), float(np.sum(logs))

    site_squared, site_log = bins(math.pi / lattice_wavevector, site_offset)
    pixel_squared, _ = bins(pixel_width, -0.5 * pixel_width)
    overlap = np.sum(np.conj(np.asarray(initial_psi, dtype=complex)) * psi)
    fft = np.fft.fft(psi)
    power = fft.real ** 2 + fft.imag ** 2
    k2 = (2.0 * math.pi * np.fft.fftfreq(points, d=step_x)) ** 2
    kl2 = lattice_wavevector ** 2
    two_k = 2.0 * lattice_wavevector
    return np.array([
        np.sum(r), np.sum(r * np.cos(two_k * xw)), np.sum(r * np.sin(two_k * xw)),
        np.sum(r * xw), np.sum(r * xw * xw), np.sum(r * xw * xw * xw),
        np.sum(r * (xw * xw) * (xw * xw)),
        np.sum(np.where((xw < central_halfwidth) & (xw > -central_halfwidth), r, 0.0)),
        np.sum(rho_log), np.sum(r * r), np.sum(r * np.real(np.asarray(potential))[window]),
        site_squared, site_log, pixel_squared, overlap.real, overlap.imag,
        np.sum(power), np.sum(np.where(k2 < kl2, power, 0.0)),
        np.sum(np.where((k2 >= kl2) & (k2 < 9.0 * kl2), power, 0.0)),
    ], dtype=float)


# --------------------------------------------------------------------------
# Summary of the solver's time series
# --------------------------------------------------------------------------
def solver_time_unit_s() -> float:
    """Seconds per solver time unit: m rs^2 / hbar."""
    return mli * GRID_SPACING_M ** 2 / hbar


def run_timing(config: dict, contract: dict) -> tuple[float, float]:
    """Hold time and final-window length in seconds."""
    from phase_metric import expected_sampling

    step = float(config["time_step"]) * solver_time_unit_s()
    sampling = expected_sampling(config, contract)
    hold = int(float(config["number_of_iterations"])) * step
    window = (sampling["last_snapshot_iteration"] - sampling["first_snapshot_iteration"]
              + sampling["save_every_nth_iteration"]) * step
    return hold, window


def read_series(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        if tuple(header) != SERIES_COLUMNS:
            raise ValueError(f"{path} does not have the extended-metric columns.")
        rows = [[float(value) for value in row] for row in reader if row]
    if len(rows) < 2:
        raise ValueError(f"{path} has fewer than two samples.")
    table = np.array(rows, dtype=float)
    return {name: table[:, index] for index, name in enumerate(header)}


def _mean(values, mask=None):
    values = np.asarray(values, dtype=float)
    if mask is not None:
        values = values[mask]
    values = values[np.isfinite(values)]
    return finite_or_none(np.mean(values)) if values.size else None


def _std(values, mask):
    values = np.asarray(values, dtype=float)[mask]
    values = values[np.isfinite(values)]
    return finite_or_none(np.std(values)) if values.size else None


def _slope(time, values, mask, logarithm):
    time = np.asarray(time, dtype=float)[mask]
    values = np.asarray(values, dtype=float)[mask]
    keep = np.isfinite(values) & ((values > 0.0) if logarithm else True)
    if keep.sum() < 3:
        return None
    y = np.log(values[keep]) if logarithm else values[keep]
    return finite_or_none(np.polyfit(time[keep], y, 1)[0])


def series_quantities(series: dict, config: dict) -> dict:
    """Per-sample observables from the raw sums."""
    wavevector = float(config["extended_metrics_lattice_wavevector"])
    step_x = float(config["step_x"])
    with np.errstate(divide="ignore", invalid="ignore"):
        mass = series["mass"]
        mean_x = series["x1_sum"] / mass
        ex2 = series["x2_sum"] / mass
        variance = ex2 - mean_x ** 2
        fourth = (series["x4_sum"] / mass - 4.0 * mean_x * series["x3_sum"] / mass
                  + 6.0 * mean_x ** 2 * ex2 - 3.0 * mean_x ** 4)
        orientation = (series["cos_sum"] + 1j * series["sin_sum"]) / mass
        site_ipr = series["site_squared_sum"] / mass ** 2
        norm0 = series["norm"][0]
        total = series["momentum_total"]
        return {
            "pendulum_cos": orientation.real,
            "onsite_spread": 1.0 - np.abs(orientation),
            "variance": variance,
            "sigma_um": np.sqrt(np.where(variance > 0.0, variance, np.nan))
            * GRID_SPACING_M * 1e6,
            "kurtosis": fourth / variance ** 2,
            "central_fraction": series["central_sum"] / mass,
            "density_entropy": np.log(mass) - series["rho_log_rho_sum"] / mass,
            "site_ipr": site_ipr,
            "occupied_sites": 1.0 / site_ipr,
            "site_entropy": np.log(mass) - series["site_log_sum"] / mass,
            "coarse_ipr": series["pixel_squared_sum"] / mass ** 2,
            "metric": series["rho_squared_sum"],
            "return_probability": (series["return_real"] ** 2 + series["return_imag"] ** 2)
            * step_x ** 2 / norm0 ** 2,
            "density_overlap": series["density_overlap"] / series["density_overlap"][0],
            "p_squared": series["kinetic_sum"] / total / wavevector ** 2,
            "order_0": series["momentum_zero"] / total,
            "order_2": series["momentum_second"] / total,
            # (1/2 <k^2> + <V_static>) in solver energy units, times E_s / E_R = 2 / kL^2.
            "energy_er": (0.5 * series["kinetic_sum"] / total + series["potential_sum"] / mass)
            * 2.0 / wavevector ** 2,
            "norm": series["norm"],
        }


def gpu_summary(path: Path, config: dict, contract: dict) -> dict:
    """GPU columns from one point's extended time series."""
    from phase_metric import expected_sampling

    series = read_series(path)
    values = series_quantities(series, config)
    iteration = series["iteration"]
    step = float(config["time_step"])
    time_solver = (iteration + 1.0) * step
    time_ms = time_solver * solver_time_unit_s() * 1e3
    sampling = expected_sampling(config, contract)
    final = ((iteration >= sampling["first_snapshot_iteration"])
             & (iteration <= sampling["last_snapshot_iteration"]))
    end = time_ms[-1]
    early, late = time_ms <= 0.25 * end, time_ms >= 0.5 * end
    # Drive phase relative to its value at t = 0.
    drive = float(config.get("floquet_omega", 0.0)) * time_solver
    strobe = late & (np.abs((drive + math.pi) % (2.0 * math.pi) - math.pi) < STROBE_HALF_WIDTH)
    return {
        "pendulum_cos": _mean(values["pendulum_cos"], final),
        "pendulum_cos_strobe": _mean(values["pendulum_cos"], strobe),
        "pendulum_cos_time_std": _std(values["pendulum_cos"], final),
        "onsite_spread": _mean(values["onsite_spread"], final),
        "onsite_spread_strobe": _mean(values["onsite_spread"], strobe),
        "width_growth_rate_per_ms": _slope(time_ms, values["variance"], early, True),
        "momentum_growth_rate_per_ms": _slope(time_ms, values["p_squared"], early, True),
        "spreading_speed_um_per_ms": _slope(time_ms, values["sigma_um"], late, False),
        "sigma_x_um_initial": finite_or_none(values["sigma_um"][0]),
        "width_time_std_um": _std(values["sigma_um"], final),
        "central_fraction": _mean(values["central_fraction"], final),
        "escape_fraction": finite_or_none(1.0 - values["norm"][-1] / values["norm"][0]),
        "density_entropy": _mean(values["density_entropy"], final),
        "kurtosis": _mean(values["kurtosis"], final),
        "site_ipr": _mean(values["site_ipr"], final),
        "occupied_sites": _mean(values["occupied_sites"], final),
        "site_entropy": _mean(values["site_entropy"], final),
        "coarse_ipr": _mean(values["coarse_ipr"], final),
        "metric_initial": finite_or_none(values["metric"][0]),
        "return_probability": _mean(values["return_probability"], final),
        "density_overlap": _mean(values["density_overlap"], final),
        "p_squared": _mean(values["p_squared"], final),
        "order_0": _mean(values["order_0"], final),
        "order_2": _mean(values["order_2"], final),
        "energy_absorbed_er": _difference(_mean(values["energy_er"], final),
                                          values["energy_er"][0]),
    }


def _difference(final, initial):
    return None if final is None else finite_or_none(final - initial)


# --------------------------------------------------------------------------
# q = 0 Floquet model
# --------------------------------------------------------------------------
def floquet_operator_phased(static, drive, omega, phase, steps):
    """U(T) for H(t) = static + sin(omega t + phase) drive, starting at t = 0.

    The fourth-order commutator-free scheme of level_statistics.floquet_operator.
    """
    period = 2.0 * math.pi / omega
    h = period / steps
    unitary = np.eye(static.shape[0], dtype=complex)
    for step in range(steps):
        f1 = math.sin(omega * (step + _C1) * h + phase)
        f2 = math.sin(omega * (step + _C2) * h + phase)
        for weight in (_A1 * f1 + _A2 * f2, _A2 * f1 + _A1 * f2):
            values, vectors = eigh(0.5 * static + weight * drive)
            unitary = (vectors * np.exp(-1j * h * values)) @ (vectors.T @ unitary)
    return unitary


def converged_operator(static, drive, omega, phase, *, tolerance=1e-4, initial_steps=32,
                       maximum_steps=2 ** 15):
    """U(T) with time steps doubled until its eigenphases converge."""
    steps = initial_steps
    previous = floquet_operator_phased(static, drive, omega, phase, steps)
    limit = tolerance * 2.0 * math.pi / static.shape[0]
    while steps < maximum_steps:
        steps *= 2
        current = floquet_operator_phased(static, drive, omega, phase, steps)
        if _phase_mismatch(np.linalg.eigvals(current), np.linalg.eigvals(previous)) <= limit:
            return current, steps
        previous = current
    raise RuntimeError(f"The q = 0 Floquet operator did not converge within {maximum_steps} steps.")


def harmonic_frequency_hz(depth_er: float) -> float:
    """f0 = omega0 / 2 pi with omega0 = 2 sqrt(V0 E_R) / hbar."""
    return math.sqrt(depth_er) * E_R / (math.pi * HBAR)


def floquet_q0_metrics(depth_er, initial_depth_er, alpha, frequency_hz, drive_phase,
                       hold_s, window_s, *, n_max=Q0_PLANE_WAVE_CUTOFF,
                       echo=ECHO_RELATIVE_CHANGE) -> dict:
    """q0_ columns: one site's Bloch state (q = 0), sampled once per drive period."""
    omega = 2.0 * math.pi * frequency_hz * HBAR / E_R          # in E_R / hbar
    static, drive = lattice_matrices(-depth_er, alpha, 0.0, n_max)
    even = symmetry_sectors(n_max, 0.0)[0]
    loading = -initial_depth_er * (1.0 + alpha * math.sin(drive_phase))
    *_, coefficients = bloch_state_1d_plane_wave(
        V0=loading, q=0.0, band_index=0, xarr=np.linspace(0.0, math.pi, 8), n_pw=n_max)
    start = even.T @ coefficients
    if abs(np.linalg.norm(start) - 1.0) > 1e-8:
        raise RuntimeError("The starting Bloch state is not even under parity.")
    start = start / np.linalg.norm(start)
    s_even, d_even = even.T @ static @ even, even.T @ drive @ even
    unitary, steps = converged_operator(s_even, d_even, omega, drive_phase)

    period = 2.0 * math.pi / omega
    periods = max(1, int(hold_s * E_R / HBAR / period))
    window = min(periods, max(1, int(window_s * E_R / HBAR / period)))
    states = np.empty((periods + 1, start.size), dtype=complex)
    states[0] = start
    for index in range(periods):
        states[index + 1] = unitary @ states[index]
    full = states @ even.T                                    # plane-wave coefficients
    n = np.arange(-n_max, n_max + 1)
    orientation = np.sum(np.conj(full[:, 1:]) * full[:, :-1], axis=1)   # <exp(2i kL x)>
    weights = np.abs(full) ** 2
    energies, bands = np.linalg.eigh(static)
    populations = np.abs(full @ bands) ** 2
    energy = np.real(np.einsum("ti,ij,tj->t", np.conj(full), static, full))
    survival = np.abs(states @ np.conj(start)) ** 2
    final = slice(periods + 1 - window, periods + 1)
    start_band = int(np.argmax(populations[0]))
    mean_populations = populations[final].mean(axis=0)

    values, vectors = np.linalg.eig(unitary)
    amplitudes = np.abs(np.linalg.solve(vectors, start)) ** 2
    amplitudes /= amplitudes.sum()

    frequency_khz = None
    if periods + 1 >= 16:
        signal = orientation.real - orientation.real.mean()
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
        if spectrum[1:].max() > 1e-12:
            peak = int(np.argmax(spectrum[1:])) + 1
            frequency_khz = np.fft.rfftfreq(signal.size, d=1.0 / frequency_hz)[peak] / 1e3

    _, drive_echo = lattice_matrices(-depth_er, alpha * (1.0 + echo), 0.0, n_max)
    unitary_echo = floquet_operator_phased(s_even, even.T @ drive_echo @ even, omega,
                                           drive_phase, steps)
    echo_state = start.copy()
    for _ in range(periods):
        echo_state = unitary_echo @ echo_state
    f0 = harmonic_frequency_hz(depth_er)
    return {
        "q0_pendulum_cos": finite_or_none(orientation.real[final].mean()),
        "q0_onsite_spread": finite_or_none((1.0 - np.abs(orientation[final])).mean()),
        "q0_p_squared": finite_or_none((weights[final] @ (2.0 * n) ** 2).mean()),
        "q0_order_0": finite_or_none(weights[final, n_max].mean()),
        "q0_order_2": finite_or_none((weights[final, n_max - 1] + weights[final, n_max + 1]).mean()),
        "q0_energy_absorbed_er": finite_or_none(energy[final].mean() - energy[0]),
        "q0_onsite_survival": finite_or_none(survival[final].mean()),
        "q0_band0_population": finite_or_none(mean_populations[0]),
        "q0_start_band_population": finite_or_none(mean_populations[start_band]),
        "q0_excited_band_population": finite_or_none(mean_populations[2:].sum()),
        "q0_band_change": finite_or_none(0.5 * np.abs(mean_populations - populations[0]).sum()),
        "q0_floquet_participation": finite_or_none(1.0 / np.sum(amplitudes ** 2)),
        "q0_slow_frequency_khz": finite_or_none(frequency_khz),
        "q0_slow_frequency_omega0": finite_or_none(
            frequency_khz * 1e3 / f0 if frequency_khz is not None else None),
        "q0_echo_fidelity": finite_or_none(abs(np.vdot(states[-1], echo_state)) ** 2),
    }


# --------------------------------------------------------------------------
# Classical ensemble
# --------------------------------------------------------------------------
def classical_metrics(points, depth_er, initial_depth_er, hold_s, window_s, *,
                      particles=CLASSICAL_PARTICLES, time_step=CLASSICAL_TIME_STEP,
                      seed=CLASSICAL_SEED) -> list[dict]:
    """cl_ columns for points [(alpha, frequency_hz, drive_phase), ...].

    theta = 2 kL x obeys theta'' = -(1 + alpha sin(Omega tau + phi)) sin theta in
    tau = omega0 t (bottom of a site at theta = 0). Particles start from the
    ground state of the loading lattice (harmonic approximation), at the bottom
    or on top, with the same random numbers for every point.
    """
    if not points:
        return []
    omega0 = 2.0 * math.sqrt(depth_er) * E_R / HBAR
    alpha = np.array([float(p[0]) for p in points])[:, None]
    big_omega = np.array([2.0 * math.pi * float(p[1]) / omega0 for p in points])[:, None]
    phase = np.array([float(p[2]) for p in points])[:, None]
    loading = -initial_depth_er * (1.0 + alpha[:, 0] * np.sin(phase[:, 0]))
    start = np.where(loading < 0.0, 0.0, math.pi)[:, None]
    width = math.sqrt(2.0 / math.sqrt(initial_depth_er))
    speed = math.sqrt(initial_depth_er / depth_er) * width
    rng = np.random.default_rng(seed)
    theta = start + width * rng.standard_normal(particles)[None, :]
    velocity = np.broadcast_to(speed * rng.standard_normal(particles)[None, :], theta.shape).copy()

    tau_end = omega0 * hold_s
    steps = max(1, int(math.ceil(tau_end / time_step)))
    dt = tau_end / steps
    window_start = tau_end - omega0 * window_s
    largest = np.zeros_like(theta)
    cos_sum = np.zeros(len(points))
    samples = 0
    scratch = np.empty_like(theta)

    def acceleration(angle, tau, out):
        np.sin(angle, out=out)
        out *= -(1.0 + alpha * np.sin(big_omega * tau + phase))
        return out

    accel = acceleration(theta, 0.0, np.empty_like(theta))
    half = 0.5 * dt
    for step in range(1, steps + 1):
        velocity += half * accel
        theta += dt * velocity
        accel = acceleration(theta, step * dt, accel)
        velocity += half * accel
        if step % 10 == 0 or step == steps:
            np.subtract(theta, start, out=scratch)
            np.abs(scratch, out=scratch)
            np.maximum(largest, scratch, out=largest)
            if step * dt >= window_start:
                cos_sum += np.cos(theta).mean(axis=1)
                samples += 1
    offset = (theta - start + math.pi) % (2.0 * math.pi) - math.pi
    near = (np.abs(offset) < 0.5 * math.pi).mean(axis=1)
    escape = (largest > math.pi).mean(axis=1)
    return [{
        "cl_pendulum_cos": finite_or_none(cos_sum[i] / samples) if samples else None,
        "cl_near_start": finite_or_none(near[i]),
        "cl_site_escape": finite_or_none(escape[i]),
    } for i in range(len(points))]


def empty_columns() -> dict:
    return dict.fromkeys(EXTENDED_FIELDS)
