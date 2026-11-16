"""Codex CUDA Port: manifest-driven runner for the native CUDA executable."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from create_initial_state_function import (
    ABSORBER_FRACTION, GRID_SPACING_M, check_green_walls, create_init_state,
)
from level_statistics import validate_settings as level_statistics_settings
from phase_metric import smoothing_contract
from compact_sweep import run_compact_sweep
from full_sweep import run_full_sweep
from parallel_sweep import print_assignments, run_parallel_sweep, split_point_ranges
from quasimomentum_spread import COVERAGE_WARNING, describe, spread_weights, validate_values
from lab_measurement import validate_camera, write_profiles_csv
from cuda_workflow_common import (
    PORT_ROOT,
    RESULTS_DIRECTORY,
    SCRIPT_DIRECTORY,
    experiment_drive_phase,
    git_provenance,
    locate_executable,
    phase_dataset_label,
    query_json,
    read_config,
    sha256_file,
    update_config,
    utc_now,
    write_json_atomic,
)


# ---------------------------------------------------------------------------
# CUDA PORT: PHYSICAL PARAMETER SECTION. This is the single authoritative grid.
# The plotting script reads these values from the generated manifest.
# ---------------------------------------------------------------------------
# Verified CPU-matching sweep from run_manifest.json (10 x 10 points).
ALPHA_VALUES = np.linspace(10.0, 120.0, 10)
DRIVE_FREQUENCY_HZ_VALUES = np.linspace(1.0e6, 6.0e6, 10)
LATTICE_DEPTH_V0_ER = 20.0
INITIAL_LATTICE_DEPTH_V0_ER = 40.0

# 5.0: loading and drive as in the experiment. The lattice stays fixed,
#   H(t) = p^2/2m - V0 [1 + alpha sin(omega t + phi)] cos^2(kL x),
# with alpha the experiment's alpha*beta, and the atoms start in the ground
# state of the lattice at t = 0 at INITIAL_LATTICE_DEPTH_V0_ER:
#   False: non-inverted, phi = 0; atoms at the bottom (cos^2 = 1).
#   True:  inverted, the lattice at t = 0 is flipped and the atoms start on top.
INVERTED = False

# 11.0: where in its cycle the drive starts.
#   "hardware": as the lab's AWG (KpPredistortion.getPhaseTarget in MMUser-li):
#       phi = arcsin((+-V_init / V0 - 1) / alpha), minus when inverted, so the
#       drive starts exactly at the loading lattice (-20 E_R inverted for 20
#       E_R loading into 10 E_R) with no jump. Needs every alpha >= 3 then.
#   "theory": phi = arcsin(-2/alpha) inverted, 0 non-inverted: the lattice
#       jumps from the loading depth to -+V0 at t = 0 (5.0-10.0, and the lab's
#       theory script kp_PhaseDiagramRealSpace.m).
DRIVE_PHASE_CONVENTION = "hardware"

# 7.0: starting quasimomentum q of every atom, in units of hbar kL (-1 < q < 1).
# One setting for every model: the GPU starting state (the lowest-band Bloch
# state at q under the Gaussian envelope), the Bloch-state Floquet model (q0_
# columns) and the level statistics. The classical ensemble has no
# quasimomentum. 0 gives the 4.0-6.0 BEC at rest. A single q > 0 makes the cloud
# drift; the widths are measured about its moving centre.
# 10.0: used only when QUASIMOMENTUM_VALUES_KL below is None.
INITIAL_QUASIMOMENTUM_KL = 0.1

# 10.0: a spread of quasimomenta. A real cloud has atoms at many q; without
# interactions each group of atoms with the same q evolves on its own and the
# camera sees the sum of their clouds. Every grid point runs once per value in
# QUASIMOMENTUM_VALUES_KL (hbar kL; each run is the group of atoms at that q),
# and MAKE and the plot tools add the runs up, weighted by a Gaussian
# distribution of q with this centre and spread (standard deviation). Negative
# q are the mirror images of positive q, so a spread centred at 0 needs only
# q >= 0. The centre and spread can be changed afterwards without rerunning;
# the values set what can be covered (5 values = 5x the GPU time of 9.0).
# None runs only INITIAL_QUASIMOMENTUM_KL for every atom, exactly as 9.0.
QUASIMOMENTUM_VALUES_KL = [0.0, 0.05, 0.10, 0.15, 0.20]
QUASIMOMENTUM_CENTER_KL = 0.0
QUASIMOMENTUM_SPREAD_KL = 0.1

# 7.0: smoothing of the density |psi|^2 over this width (um) before every
# position metric, like the camera's 4 um pixels: cloud width and centre, IPR,
# central fraction, entropies, kurtosis, lattice-site populations and the
# density overlap. The pendulum angle, on-site spread and <V> keep the raw
# density (they resolve the inside of a 532 nm site, which 4 um averages away);
# momentum and amplitude columns are not position images. 0 turns it off.
DensitySmoothing_um = 4.0

# 8.0: the simulated box, in 23 nm grid points. 262144 is +-3.0 mm with
# absorbing walls from +-2.1 mm, so the fast atoms that make the widest
# experimental clouds (standard deviation ~0.9 mm after 2 ms) stay in the box.
# 65536 (+-0.75 mm, walls from +-0.53 mm) is the box of 4.0-7.0, about 4.5x
# faster. An even number of at least 1024; powers of 2 run fastest.
GRID_POINTS = 262144

# 8.0: the starting cloud: standard deviation of |psi|^2 of the Gaussian
# envelope, in um (the atom cloud along the lattice, not a beam waist). The
# central window (central_fraction) is two of these widths.
# 11.0: 35, measured in the lab's images (Gaussian sigma 38-42 um in the
# undriven lattice trial 10084, 28-45 um in the weakest-drive shots of trials
# 10098 and 10366; measure_lab_images.py). 9.0-10.0 used 340, which matched the
# lab's StdDev2 noise floor rather than the atoms; 4.0-8.0 used 30.
INITIAL_CLOUD_WIDTH_UM = 35.0

# 8.0: hold time with the drive on, in ms (the lab's t_mod), turned into time
# steps of the solver. None keeps number_of_iterations from gpe1d.config
# (70000 steps = 2.045 ms, as 4.0-7.0).
HOLD_TIME_MS = 2.0

# 12.0: lattice hold between the drive and the image, in ms. The drive runs
# for HOLD_TIME_MS, then the lattice stays static for LATTICE_HOLD_MS more, and
# the cloud is imaged (measured) at the end, HOLD_TIME_MS + LATTICE_HOLD_MS
# after the drive started. 0 images right after the drive, as 8.0-11.0.
LATTICE_HOLD_MS = 2.0
# The static lattice during the hold, as D in -D cos^2(kL x), in E_R.
#   None: it stays where the drive ends. With DRIVE_WHOLE_CYCLES that is the
#         lattice the drive started from: the loading lattice, -20 E_R
#         (inverted, INITIAL_LATTICE_DEPTH_V0_ER) for INVERTED = True.
#   A number: 10 = the normal 10 E_R lattice, -20 = the inverted 20 E_R
#         lattice, 0 = no lattice (free expansion).
LATTICE_HOLD_DEPTH_ER = None
# 12.0: drive N = round(HOLD_TIME_MS x f) whole cycles, like the lab's
# hw_KPModNCycle, so the drive ends where it started. The total time stays
# HOLD_TIME_MS + LATTICE_HOLD_MS for every point (Cicero's latticehold step);
# the hold takes up the difference of at most half a drive period. False:
# drive exactly HOLD_TIME_MS. Used only with a lattice hold.
DRIVE_WHOLE_CYCLES = True

# Green walls: two repulsive Gaussian walls around the initial wavefunction,
# added to the static potential (the drive does not modulate them). They are
# centred at -gap/2 and +gap/2 around the cloud at x = 0. False leaves every
# input exactly as before.
GreenWalls = False
GreenWallHeight_ER = 100.0   # peak height of each wall, in E_R like the lattice depth
GreenWallSigma_um = 5.0      # Gaussian sigma of each wall, exp(-x^2/(2 sigma^2)), in um
GreenWallGap_um = 200.0      # centre-to-centre distance between the two walls, in um

# Level statistics (level_statistics.py): <r>, <r^2> and the Wigner-Dyson
# versus Poisson factors eta = |(<r> - <r>_P) / (<r>_GOE - <r>_P)| and
# eta_r_squared (same with <r^2>) of the Floquet quasienergies of the driven
# lattice at each point, added as columns of metrics.csv. The CPU computes them
# while the GPU runs; they do not use the GPU result or the green walls.
# 7.0: at q = INITIAL_QUASIMOMENTUM_KL (parity sectors are split only at q = 0).
CalculateLevelStatistics = False
LevelStatisticsPlaneWaveCutoff = 40    # plane waves |n| <= n_max, i.e. 2 n_max + 1 levels

# 6.0 extended metrics: True adds every metric of extended_fields.py to
# metrics.csv, from three sources: the GPU simulation (pendulum angle <cos 2kL x>,
# on-site spread, growth rates, momentum orders, return probability, site IPR,
# entropy, kurtosis, ...), a q = 0 Floquet model (band populations, Floquet
# participation, slow frequency, echo, ...) and a classical ensemble. It also
# turns on the level statistics above. Needs compact storage, the fused loop
# and a 7.0 build; the CPU adds roughly 0.1-0.3 s per point at q = 0 and about
# three times that at any other q (the Bloch-state model then needs every plane wave).
# 7.0 adds the cloud width from 4 um pixels (sigma_x_4um_um).
ExtendedMetrics = True

# 11.0: measure every point as the lab's camera analysis does (lab_measurement.py):
# the GPU writes the final cloud in camera pixels, and lab_stddev2_um is
# kpAnalysis.m's StdDev2 of it on the lab's camera window with the lab's camera
# noise (columns lab_stddev2_um, lab_stddev_um, lab_stddev2_no_noise_um,
# camera_width_um, lab_window_fraction, camera_fov_fraction, and the file
# camera_profiles.csv). Needs compact storage and an 11.0 build.
LabMeasurement = True
# The lab's camera: Andor iXon 888 (ANDORTOP in MMUser-li's mmConfig.m),
# 13 um pixels behind a magnification of 3.338 (3.895 um at the atoms), 1024
# pixels (3.99 mm); the AdData crop is sensor rows 120-780 along the lattice
# with the cloud at row ~456 (kpAnalysis.m). Noise: measure_lab_images.py on
# trial 10366. Values not given here take lab_measurement.DEFAULT_CAMERA.
LAB_CAMERA = {
    "pixel_um": 13.0 / 3.337676782237963,
    "sensor_pixels": 1024,
    "cloud_sensor_row": 455.5,
    "roi_sensor_rows": [120, 780],
    "lab_um_per_pixel": 4.0,             # kpAnalysis.m's 4e-6 m per pixel
    "edge_fraction": 0.10,               # computeStDevWithBgSub's baseline
    "noise_floor": 0.0,                  # kpAnalysis.m noiseFloor
    "noise_per_pixel_over_signal": 3.78e-4,
    "noise_draws": 200,
    "window": "roi",                     # "roi" = the AdData crop, "sensor" = the whole field
}

# Number of simultaneous terminal tabs sharing this grid. Any positive integer
# works; leftover points are distributed one each to the first tabs.
# 1 runs here. Values greater than 1 open Windows Terminal tabs (or windows).
NUMBER_OF_TABS = 1

# The physical update reproduced the established CPU phase diagram. The
# cumulative legacy update is intentionally unavailable in this user workflow.
FLOQUET_MODE = "physical"
CUDA_DEVICE = 0

# Grid points each tab simulates together as one GPU batch (1 to 64). The
# points evolve side by side with unchanged numerics; larger batches keep a big
# GPU busy. BENCHMARK_CUDA.bat measures the fastest value for your GPU.
POINTS_PER_BATCH = 8


def green_wall_settings() -> tuple[dict | None, float | None]:
    """Validated wall settings and the initial cloud fraction between them."""
    if not isinstance(GreenWalls, bool):
        raise ValueError("GreenWalls must be True or False.")
    if not GreenWalls:
        return None, None
    checked = check_green_walls(GreenWallHeight_ER, GreenWallSigma_um, GreenWallGap_um,
                                grid_points=GRID_POINTS,
                                envelope_sigma_m=float(INITIAL_CLOUD_WIDTH_UM) * 1e-6)
    fraction = checked.pop("initial_fraction_between_walls")
    return checked, fraction


def lab_settings() -> dict | None:
    """11.0: the validated camera settings, or None when LabMeasurement is off."""
    if not isinstance(LabMeasurement, bool):
        raise ValueError("LabMeasurement must be True or False.")
    return validate_camera(LAB_CAMERA) if LabMeasurement else None


def quasimomentum_settings() -> dict | None:
    """10.0: the validated spread of quasimomenta, or None for a single q."""
    if QUASIMOMENTUM_VALUES_KL is None:
        return None
    if isinstance(QUASIMOMENTUM_VALUES_KL, (str, bytes)) \
            or not hasattr(QUASIMOMENTUM_VALUES_KL, "__iter__"):
        raise ValueError("QUASIMOMENTUM_VALUES_KL must be a list of numbers or None.")
    values = validate_values(list(QUASIMOMENTUM_VALUES_KL))
    weights, coverage = spread_weights(values, QUASIMOMENTUM_CENTER_KL, QUASIMOMENTUM_SPREAD_KL)
    return {
        "values_kl": values,
        "center_kl": float(QUASIMOMENTUM_CENTER_KL),
        "spread_kl": float(QUASIMOMENTUM_SPREAD_KL),
        "distribution": "gaussian",
        "coverage": coverage,
        "negative_q": "mirror image x -> -x of +q when -q is not simulated",
        "combination": "quasimomentum_spread.py",
    }


def extended_settings() -> dict | None:
    if not ExtendedMetrics:
        return None
    from extended_metrics import contract_settings
    return contract_settings({"initial_cloud_width_um": float(INITIAL_CLOUD_WIDTH_UM)})


def scan_settings(base: dict) -> dict:
    """8.0: the validated box, starting cloud and hold time of the scan."""
    if isinstance(GRID_POINTS, bool) or not isinstance(GRID_POINTS, int) \
            or GRID_POINTS < 1024 or GRID_POINTS % 2:
        raise ValueError("GRID_POINTS must be an even integer of at least 1024.")
    width = INITIAL_CLOUD_WIDTH_UM
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not width > 0:
        raise ValueError("INITIAL_CLOUD_WIDTH_UM must be a positive number of micrometres.")
    half = GRID_POINTS // 2 * GRID_SPACING_M * 1e6
    walls = half - ABSORBER_FRACTION * (GRID_POINTS - 1) * GRID_SPACING_M * 1e6
    if 3.0 * width >= walls:
        raise ValueError("The starting cloud does not fit inside the absorbing walls; "
                         "raise GRID_POINTS or lower INITIAL_CLOUD_WIDTH_UM.")
    from extended_metrics import solver_time_unit_s
    step_s = (float(base["time_step"]) * solver_time_unit_s()) if "time_step" in base else None
    if HOLD_TIME_MS is None:
        steps = int(float(base["number_of_iterations"]))
    elif step_s is None:
        raise ValueError("HOLD_TIME_MS needs time_step in gpe1d.config.")
    elif isinstance(HOLD_TIME_MS, bool) or not isinstance(HOLD_TIME_MS, (int, float)) \
            or not HOLD_TIME_MS > 0:
        raise ValueError("HOLD_TIME_MS must be None or a positive number of milliseconds.")
    else:
        steps = int(round(float(HOLD_TIME_MS) * 1e-3 / step_s))
    hold = LATTICE_HOLD_MS
    if hold is None or hold == 0:
        hold = None
    elif isinstance(hold, bool) or not isinstance(hold, (int, float)) or not hold > 0:
        raise ValueError("LATTICE_HOLD_MS must be 0, None or a positive number of milliseconds.")
    elif HOLD_TIME_MS is None:
        raise ValueError("LATTICE_HOLD_MS needs HOLD_TIME_MS (the drive time) to be set.")
    else:
        # 12.0: the run lasts the drive plus the hold; each point stops its
        # drive after its own whole cycles (cuda_workflow_common.drive_hold).
        steps = int(round((float(HOLD_TIME_MS) + float(hold)) * 1e-3 / step_s))
    depth = LATTICE_HOLD_DEPTH_ER
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, (int, float))
                              or not math.isfinite(depth)):
        raise ValueError("LATTICE_HOLD_DEPTH_ER must be None or a number of E_R.")
    if not isinstance(DRIVE_WHOLE_CYCLES, bool):
        raise ValueError("DRIVE_WHOLE_CYCLES must be True or False.")
    return {
        "grid_points": GRID_POINTS, "box_half_width_um": half, "absorber_starts_um": walls,
        "initial_cloud_width_um": float(width),
        "hold_time_ms": None if HOLD_TIME_MS is None else float(HOLD_TIME_MS),
        "number_of_iterations": steps,
        "simulated_hold_ms": None if step_s is None else steps * step_s * 1e3,
        "lattice_hold_ms": None if hold is None else float(hold),
        "lattice_hold_depth_er": None if hold is None or depth is None else float(depth),
        "drive_whole_cycles": None if hold is None else DRIVE_WHOLE_CYCLES,
    }


def dataset_label(green_walls: dict | None) -> str:
    return phase_dataset_label(
        LATTICE_DEPTH_V0_ER, INITIAL_LATTICE_DEPTH_V0_ER, 0.0, green_walls, inverted=INVERTED,
        quasimomentum=float(INITIAL_QUASIMOMENTUM_KL), smoothing_um=float(DensitySmoothing_um),
        grid_points=GRID_POINTS, cloud_width_um=float(INITIAL_CLOUD_WIDTH_UM),
        hold_time_ms=HOLD_TIME_MS,
        quasimomentum_values=(None if QUASIMOMENTUM_VALUES_KL is None
                              else list(QUASIMOMENTUM_VALUES_KL)),
        drive_phase_convention=DRIVE_PHASE_CONVENTION,
        lattice_hold_ms=LATTICE_HOLD_MS or None,
        lattice_hold_depth_er=LATTICE_HOLD_DEPTH_ER if LATTICE_HOLD_MS else None,
    )


def unique_results_directory(green_walls: dict | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("cuda_%Y%m%dT%H%M%SZ")
    parameter_label = dataset_label(green_walls)
    path = RESULTS_DIRECTORY / f"{parameter_label}_{timestamp}"
    if path.exists():
        raise FileExistsError(f"Refusing to reuse results directory: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--device", type=int, default=CUDA_DEVICE)
    parser.add_argument("--tabs", type=int, default=NUMBER_OF_TABS,
                        help="override NUMBER_OF_TABS in the settings above")
    parser.add_argument("--batch", type=int, default=POINTS_PER_BATCH,
                        help="override POINTS_PER_BATCH (grid points per GPU batch)")
    parser.add_argument("--reference-loop", action="store_true",
                        help="use the original, slower solver loop, one point at a time")
    parser.add_argument("--plan", action="store_true",
                        help="show point assignments without launching simulations")
    parser.add_argument("--headless", action="store_true",
                        help="run parallel workers in the background instead of opening tabs")
    parser.add_argument(
        "--storage", choices=("compact", "full"), default="compact",
        help="compact (default): one metric per point; full: retain wavefunction snapshots",
    )
    # Retain the successful explicit command for compatibility while rejecting
    # legacy phase-diagram requests.
    parser.add_argument(
        "--floquet-mode", choices=("physical",), default=FLOQUET_MODE,
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if isinstance(arguments.tabs, bool) or not isinstance(arguments.tabs, int) or arguments.tabs <= 0:
        parser.error("NUMBER_OF_TABS / --tabs must be a positive integer.")
    if isinstance(arguments.batch, bool) or not isinstance(arguments.batch, int) or arguments.batch <= 0:
        parser.error("POINTS_PER_BATCH / --batch must be a positive integer.")
    try:
        green_walls, fraction_between = green_wall_settings()
        if not isinstance(CalculateLevelStatistics, bool):
            raise ValueError("CalculateLevelStatistics must be True or False.")
        if not isinstance(ExtendedMetrics, bool):
            raise ValueError("ExtendedMetrics must be True or False.")
        if isinstance(INITIAL_QUASIMOMENTUM_KL, bool) \
                or not isinstance(INITIAL_QUASIMOMENTUM_KL, (int, float)) \
                or not -1.0 < INITIAL_QUASIMOMENTUM_KL < 1.0:
            raise ValueError("INITIAL_QUASIMOMENTUM_KL must be a number with -1 < q < 1.")
        spread = quasimomentum_settings()
        lab = lab_settings()
        base_values = read_config(SCRIPT_DIRECTORY / "gpe1d.config")
        smoothing = smoothing_contract(DensitySmoothing_um, float(base_values["step_x"]),
                                       GRID_SPACING_M)
        scan = scan_settings(base_values)
        # 10.0: each run uses its own q; the manifest keeps the centre.
        level_statistics = (level_statistics_settings(
                                float(spread["center_kl"] if spread else INITIAL_QUASIMOMENTUM_KL),
                                LevelStatisticsPlaneWaveCutoff)
                            if CalculateLevelStatistics or ExtendedMetrics else None)
        # Checks INVERTED and, when inverted, that every alpha >= 2.
        drive_phases = [experiment_drive_phase(alpha, INVERTED, DRIVE_PHASE_CONVENTION,
                                               LATTICE_DEPTH_V0_ER, INITIAL_LATTICE_DEPTH_V0_ER)
                        for alpha in ALPHA_VALUES]
    except ValueError as error:
        parser.error(str(error))
    # Snapshot output and the reference loop keep one point per solver run.
    batch = 1 if arguments.storage == "full" or arguments.reference_loop else arguments.batch
    if spread and arguments.storage != "compact":
        parser.error("QUASIMOMENTUM_VALUES_KL needs compact storage (no --storage full).")
    if lab and (arguments.storage != "compact" or arguments.reference_loop):
        parser.error("LabMeasurement needs compact storage and the fused loop "
                     "(no --storage full, no --reference-loop).")
    quasimomenta = spread["values_kl"] if spread else [float(INITIAL_QUASIMOMENTUM_KL)]
    total = len(ALPHA_VALUES) * len(DRIVE_FREQUENCY_HZ_VALUES) * len(quasimomenta)
    assignments = split_point_ranges(total, arguments.tabs)
    print_assignments(total, arguments.tabs, assignments)
    if batch > 1:
        print(f"Each tab simulates up to {batch} grid points at a time as one GPU batch.")
    start = (f"the drive starts at the loading lattice, {'-' if INVERTED else '+'}"
             f"{INITIAL_LATTICE_DEPTH_V0_ER:g} E_R (hardware)" if DRIVE_PHASE_CONVENTION == "hardware"
             else f"the lattice jumps to {'-' if INVERTED else '+'}{LATTICE_DEPTH_V0_ER:g} E_R "
                  "at t = 0 (theory)")
    print(f"Loading: {'inverted, atoms start on top' if INVERTED else 'non-inverted'}, "
          f"{INITIAL_LATTICE_DEPTH_V0_ER:g} E_R. Drive sin(omega t + phi), phi from "
          f"{max(drive_phases):.4g} to {min(drive_phases):.4g} rad; {start}.")
    if green_walls:
        print(f"Green walls: {green_walls['height_er']:g} E_R high, sigma "
              f"{green_walls['sigma_um']:g} um, centres at +/-{green_walls['gap_um'] / 2:g} um; "
              f"{fraction_between:.1%} of the initial cloud starts between them.")
        if fraction_between < 0.95:
            print("Note: the walls overlap the initial cloud.")
    else:
        print("Green walls: off")
    print(f"Box: {scan['grid_points']:,} grid points, +/-{scan['box_half_width_um'] / 1e3:.2f} mm, "
          f"absorbing walls from +/-{scan['absorber_starts_um'] / 1e3:.2f} mm; "
          f"starting cloud {scan['initial_cloud_width_um']:g} um.")
    hold = (f"{scan['hold_time_ms']:g} ms" if scan["hold_time_ms"] is not None
            else "from gpe1d.config")
    simulated = (f" ({scan['simulated_hold_ms']:.4f} ms)"
                 if scan["simulated_hold_ms"] is not None else "")
    if scan["lattice_hold_ms"] is None:
        print(f"Hold time: {hold} = {scan['number_of_iterations']:,} steps{simulated}; "
              "imaged right after the drive.")
    else:
        cycles = ("N = round(drive time x f) whole cycles, ending where it started"
                  if scan["drive_whole_cycles"] else "exactly")
        lattice = ("where the drive ends" if scan["lattice_hold_depth_er"] is None
                   else f"-({scan['lattice_hold_depth_er']:g} E_R) cos^2(kL x)")
        print(f"Drive: {hold}, {cycles}. Then a {scan['lattice_hold_ms']:g} ms lattice hold, "
              f"static at {lattice}; imaged at the end.")
        print(f"Run length: {scan['number_of_iterations']:,} steps{simulated} for every point.")
    if spread:
        print(f"Quasimomentum spread: every grid point runs at q = "
              f"{', '.join(f'{q:g}' for q in quasimomenta)} hbar kL "
              f"({len(ALPHA_VALUES) * len(DRIVE_FREQUENCY_HZ_VALUES):,} grid points x "
              f"{len(quasimomenta)} = {total:,} GPU runs; negative q by mirror symmetry).")
        print("  Combined as a " + describe(quasimomenta, spread["center_kl"], spread["spread_kl"]))
        if spread["coverage"] < COVERAGE_WARNING:
            print(f"  Warning: only {spread['coverage']:.1%} of this spread lies within the "
                  "simulated values; add larger q to QUASIMOMENTUM_VALUES_KL.")
        print("  The centre and spread can be changed afterwards in MAKE and the plot tools.")
    else:
        print(f"Initial quasimomentum: q = {float(INITIAL_QUASIMOMENTUM_KL):g} hbar kL for every "
              "atom (GPU start, Bloch-state model, level statistics).")
    if lab:
        rows = lab["roi_sensor_rows"]
        print(f"Lab measurement: camera {lab['sensor_pixels']} pixels of {lab['pixel_um']:.3f} um "
              f"({lab['sensor_pixels'] * lab['pixel_um'] / 1e3:.2f} mm), cloud at row "
              f"{lab['cloud_sensor_row']:g}; StdDev2 window "
              + (f"rows {rows[0]}-{rows[1]} ({rows[1] - rows[0] + 1} px)" if lab["window"] == "roi"
                 else "the whole sensor")
              + f", noise {lab['noise_per_pixel_over_signal']:g} of the signal per pixel, "
              f"{lab['noise_draws']} draws.")
    else:
        print("Lab measurement: off")
    if smoothing:
        if arguments.reference_loop:
            parser.error("DensitySmoothing_um needs the fused loop (no --reference-loop).")
        print(f"Density smoothing: {smoothing['width_um']:g} um ({2 * smoothing['half_width_points'] + 1} "
              f"grid points, {smoothing['window_um']:.3f} um) for the position metrics.")
    else:
        print("Density smoothing: off")
    if ExtendedMetrics:
        if arguments.storage != "compact" or arguments.reference_loop:
            parser.error("ExtendedMetrics needs compact storage and the fused loop "
                         "(no --storage full, no --reference-loop).")
        if scan["lattice_hold_ms"] is not None:
            # 12.0: the models drive until the end, so they do not apply to a hold.
            print("Extended metrics: on (GPU columns; the Bloch-state Floquet model and "
                  "classical ensemble columns stay empty with a lattice hold; level "
                  "statistics on).")
        else:
            print("Extended metrics: on (GPU, Bloch-state Floquet model and classical ensemble "
                  "columns; level statistics on).")
    else:
        print("Extended metrics: off")
    if level_statistics:
        where = ("at each run's own q" if spread
                 else f"at q = {level_statistics['quasimomentum']:g} kL")
        print(f"Level statistics: <r>, <r^2>, eta and eta_r_squared of the Floquet quasienergies "
              f"{where} with plane waves |n| <= {level_statistics['plane_wave_cutoff']}.")
    else:
        print("Level statistics: off")
    if arguments.plan:
        return 0

    executable = locate_executable(arguments.executable)
    build_info = query_json(executable, "--version-json")
    if arguments.storage == "compact" and build_info.get("compact_phase_metric_version") != 1:
        raise RuntimeError(
            "This executable does not support compact output. Run BUILD_CUDA.bat "
            "after updating the source, or use --storage full for the old snapshot workflow."
        )
    if arguments.storage == "compact" and build_info.get("phase_statistics_version") != 1:
        raise RuntimeError(
            "This executable does not support the new variance/std columns. "
            "Run BUILD_CUDA.bat, then VALIDATE_COMPACT_CUDA.bat before starting the scan."
        )
    fused = build_info.get("fused_solver_version") == 1
    if (arguments.storage == "compact" or arguments.reference_loop) and not fused:
        raise RuntimeError(
            "This executable predates the faster fused solver. Run BUILD_CUDA.bat, "
            "then VALIDATE_FAST_CUDA.bat, before starting the scan."
        )
    if batch > int(build_info.get("max_batch_systems", 1)):
        raise RuntimeError(
            f"--batch {batch} exceeds this executable's limit of "
            f"{build_info.get('max_batch_systems', 1)} points per GPU batch."
        )
    # An older solver would ignore floquet_phase and silently drive cos(omega t).
    if build_info.get("drive_phase_version") != 1:
        raise RuntimeError(
            "This executable predates the 5.0 drive phase. Run BUILD_CUDA.bat, then "
            "VALIDATE_FAST_CUDA.bat, before starting the scan."
        )
    if ExtendedMetrics and build_info.get("extended_metrics_version") != 2:
        raise RuntimeError(
            "This executable predates the 7.0 extended metrics. Run BUILD_CUDA.bat, then "
            "VALIDATE_EXTENDED_CUDA.bat, before starting the scan."
        )
    # An older solver would ignore drive_iterations and keep driving through the hold.
    if scan["lattice_hold_ms"] is not None and build_info.get("drive_hold_version") != 1:
        raise RuntimeError(
            "This executable predates the 12.0 lattice hold. Run BUILD_CUDA.bat, then "
            "VALIDATE_FAST_CUDA.bat, before starting the scan.")
    if lab and build_info.get("camera_profile_version") != 1:
        raise RuntimeError(
            "This executable predates the 11.0 camera profile. Run BUILD_CUDA.bat, then "
            "VALIDATE_EXTENDED_CUDA.bat, before starting the scan.")
    if smoothing and arguments.storage == "compact" \
            and build_info.get("density_smoothing_version") != 1:
        raise RuntimeError(
            "This executable predates the 7.0 density smoothing. Run BUILD_CUDA.bat, then "
            "VALIDATE_COMPACT_CUDA.bat, before starting the scan."
        )
    # Sleeping instead of spinning while waiting for the GPU leaves CPU power
    # and thermal headroom to the GPU, especially with several tabs.
    extra_arguments = (["--wait", "blocking"] if fused else []) + (
        ["--reference-loop"] if arguments.reference_loop else [])
    results_root = (arguments.results.resolve() if arguments.results
                    else unique_results_directory(green_walls))
    if results_root.exists() and any(results_root.iterdir()):
        raise FileExistsError(f"Results directory must be absent or empty: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    base_config = SCRIPT_DIRECTORY / "gpe1d.config"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    # 8.0: one config for the whole scan, with its box and hold time; every
    # point and tab starts from it.
    scan_config = results_root / "gpe1d_scan.config"
    shutil.copy2(base_config, scan_config)
    update_config(scan_config, {"points_x": scan["grid_points"],
                                "number_of_iterations": scan["number_of_iterations"]})
    base_config = scan_config

    manifest_path = results_root / "run_manifest.json"
    parameter_label = dataset_label(green_walls)
    manifest = {
        "_codex_cuda_port": "Native CUDA phase-diagram run manifest.",
        "manifest_version": 2,
        "storage_mode": arguments.storage,
        "dataset_name": results_root.name,
        "parameter_label": parameter_label,
        "created_utc": utc_now(),
        "status": "running",
        "results_root": str(results_root),
        "solver": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "build": build_info,
            "device": query_json(executable, "--device-info", str(arguments.device)),
            "device_index": arguments.device,
            "floquet_mode": arguments.floquet_mode,
            "compatibility_default_confirmed": True,
            "loop": "reference" if arguments.reference_loop or not fused else "fused",
            "points_per_batch": batch,
            "extra_arguments": extra_arguments,
        },
        "source_tree": git_provenance(PORT_ROOT),
        "parameter_grid": {
            "iteration_order": ("alpha_outer_frequency_middle_quasimomentum_inner" if spread
                                else "alpha_outer_frequency_inner"),
            "alpha_values": [float(value) for value in ALPHA_VALUES],
            "drive_frequency_hz_values": [
                float(value) for value in DRIVE_FREQUENCY_HZ_VALUES
            ],
            "lattice_depth_v0_er": LATTICE_DEPTH_V0_ER,
            "initial_lattice_depth_v0_er": INITIAL_LATTICE_DEPTH_V0_ER,
            # 5.0: fixed lattice, drive sin(omega t + phi) with phi = 0, or
            # arcsin(-2/alpha) per point when inverted (see INVERTED above).
            "inverted": INVERTED,
            "drive": "sin(omega t + phi)",
            # 11.0: "hardware" (drive starts at the loading lattice) or "theory".
            "drive_phase_convention": DRIVE_PHASE_CONVENTION,
            # None, or the walls' height (E_R), sigma (um) and centre gap (um).
            "green_walls": green_walls,
            # 7.0: starting quasimomentum (units of kL) and density smoothing (um).
            "initial_quasimomentum_kl": float(spread["center_kl"] if spread
                                              else INITIAL_QUASIMOMENTUM_KL),
            "density_smoothing_um": float(DensitySmoothing_um),
            # 10.0: the q values every grid point runs at (absent for a single q)
            # and the default spread they are combined with.
            **({"quasimomentum_values_kl": spread["values_kl"],
                "quasimomentum_spread": {key: spread[key] for key in
                                         ("center_kl", "spread_kl", "distribution")}}
               if spread else {}),
            # 8.0: box, starting cloud width and hold time (None = gpe1d.config's steps).
            "grid_points": scan["grid_points"],
            "initial_cloud_width_um": scan["initial_cloud_width_um"],
            "hold_time_ms": scan["hold_time_ms"],
            "number_of_iterations": scan["number_of_iterations"],
            # 12.0: lattice hold after the drive (None = imaged right after it),
            # the hold lattice (None = where the drive ends) and whole cycles.
            **({"lattice_hold_ms": scan["lattice_hold_ms"],
                "lattice_hold_depth_er": scan["lattice_hold_depth_er"],
                "drive_whole_cycles": scan["drive_whole_cycles"]}
               if scan["lattice_hold_ms"] is not None else {}),
        },
        "analysis_contract": {
            "statistics_version": 1,
            "temporal_variance_ddof": 0,
            "spatial_weights": "cropped_abs_psi_squared_normalized_per_snapshot",
            "position_coordinates": "(index - points_x // 2) * step_x",
            "position_units": "solver_length_units",
            "sigma_x_squared": "time_mean_of_per_snapshot_centered_variance",
            "snapshot_sort": "numeric_iteration",
            "final_snapshot_count": 30,
            "cut_points_each_edge": 100,
            "metric": "mean_discrete_sum_abs_psi_fourth_power",
            # None, or the Floquet level-statistics settings and references.
            "level_statistics": level_statistics,
            # None, or the 6.0 extended-metric settings (extended_metrics.py).
            "extended_metrics": extended_settings(),
            # None, or the 7.0 smoothing of |psi|^2 (phase_metric.smoothing_contract).
            "density_smoothing": smoothing,
            # None, or the 10.0 quasimomentum spread (quasimomentum_spread.py).
            "quasimomentum_spread": spread,
            # None, or the 11.0 camera and analysis settings (lab_measurement.py).
            "lab_measurement": lab,
            "normalization": "none",
            "frequency_axis_inverted": True,
            "colormap": "inferno",
        },
        "runs": [],
    }
    write_json_atomic(manifest_path, manifest)

    if len(assignments) > 1:
        run_parallel_sweep(
            results_root=results_root, manifest=manifest, base_config=base_config,
            tabs=arguments.tabs, headless=arguments.headless,
        )
    else:
        sweep = run_compact_sweep if arguments.storage == "compact" else run_full_sweep
        try:
            sweep(
                results_root=results_root, manifest=manifest, base_config=base_config,
                executable=executable, device=arguments.device,
                floquet_mode=arguments.floquet_mode, create_initial_state=create_init_state,
            )
        except BaseException as error:
            manifest["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["error"] = str(error)
            write_json_atomic(manifest_path, manifest)
            raise
        manifest["status"] = "completed"
        manifest["finished_utc"] = utc_now()
        write_json_atomic(manifest_path, manifest)
    if lab and arguments.storage == "compact":
        # 11.0: every run's final camera profile in one file, for the plot tools.
        saved = json.loads((results_root / "run_manifest.json").read_text(encoding="utf-8"))
        journal = results_root / saved.get("point_results_file", "point_results.jsonl")
        if saved.get("status") == "completed" and journal.is_file():
            records = [json.loads(line) for line in
                       journal.read_text(encoding="utf-8").splitlines() if line.strip()]
            count = write_profiles_csv(results_root / "camera_profiles.csv", records, lab)
            print(f"Camera profiles: {results_root / 'camera_profiles.csv'} ({count} runs)")
    print(f"\nCompleted {manifest['completed_points']:,} points.")
    print(f"Dataset: {results_root}")
    if arguments.storage == "compact":
        print(f"Results table: {results_root / 'metrics.csv'}")
    print("Create the phase diagram with MAKE_PHASE_DIAGRAM_CUDA.bat")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CUDA phase-diagram runner failed: {error}", file=sys.stderr)
        raise
