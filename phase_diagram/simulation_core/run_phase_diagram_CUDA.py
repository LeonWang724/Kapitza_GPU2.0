"""Codex CUDA Port: manifest-driven runner for the native CUDA executable."""

from __future__ import annotations

import argparse
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
#   True:  inverted, phi = arcsin(-2/alpha) for each alpha, so the lattice at
#          t = 0 is flipped and the atoms start on top; needs every alpha >= 2.
INVERTED = False

# 7.0: starting quasimomentum q of every atom, in units of hbar kL (-1 < q < 1).
# One setting for every model: the GPU starting state (the lowest-band Bloch
# state at q under the Gaussian envelope), the Bloch-state Floquet model (q0_
# columns) and the level statistics. The classical ensemble has no
# quasimomentum. 0 gives the 4.0-6.0 BEC at rest. A single q > 0 makes the cloud
# drift; the widths are measured about its moving centre.
INITIAL_QUASIMOMENTUM_KL = 0.1

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
# 9.0: 340, the experiment's cloud: trial 11243's StdDev2 is 340-370 um where
# the drive barely acts (alpha = 4). 30 is the cloud of 4.0-8.0. Set it from
# the lab's unmodulated reference trial when you have it.
INITIAL_CLOUD_WIDTH_UM = 340.0

# 8.0: hold time with the drive on, in ms (the lab's t_mod), turned into time
# steps of the solver. None keeps number_of_iterations from gpe1d.config
# (70000 steps = 2.045 ms, as 4.0-7.0).
HOLD_TIME_MS = 2.0

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
    return {
        "grid_points": GRID_POINTS, "box_half_width_um": half, "absorber_starts_um": walls,
        "initial_cloud_width_um": float(width),
        "hold_time_ms": None if HOLD_TIME_MS is None else float(HOLD_TIME_MS),
        "number_of_iterations": steps,
        "simulated_hold_ms": None if step_s is None else steps * step_s * 1e3,
    }


def dataset_label(green_walls: dict | None) -> str:
    return phase_dataset_label(
        LATTICE_DEPTH_V0_ER, INITIAL_LATTICE_DEPTH_V0_ER, 0.0, green_walls, inverted=INVERTED,
        quasimomentum=float(INITIAL_QUASIMOMENTUM_KL), smoothing_um=float(DensitySmoothing_um),
        grid_points=GRID_POINTS, cloud_width_um=float(INITIAL_CLOUD_WIDTH_UM),
        hold_time_ms=HOLD_TIME_MS,
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
        base_values = read_config(SCRIPT_DIRECTORY / "gpe1d.config")
        smoothing = smoothing_contract(DensitySmoothing_um, float(base_values["step_x"]),
                                       GRID_SPACING_M)
        scan = scan_settings(base_values)
        level_statistics = (level_statistics_settings(float(INITIAL_QUASIMOMENTUM_KL),
                                                      LevelStatisticsPlaneWaveCutoff)
                            if CalculateLevelStatistics or ExtendedMetrics else None)
        # Checks INVERTED and, when inverted, that every alpha >= 2.
        drive_phases = [experiment_drive_phase(alpha, INVERTED) for alpha in ALPHA_VALUES]
    except ValueError as error:
        parser.error(str(error))
    # Snapshot output and the reference loop keep one point per solver run.
    batch = 1 if arguments.storage == "full" or arguments.reference_loop else arguments.batch
    total = len(ALPHA_VALUES) * len(DRIVE_FREQUENCY_HZ_VALUES)
    assignments = split_point_ranges(total, arguments.tabs)
    print_assignments(total, arguments.tabs, assignments)
    if batch > 1:
        print(f"Each tab simulates up to {batch} grid points at a time as one GPU batch.")
    if INVERTED:
        print(f"Loading: inverted. Drive sin(omega t + phi), phi = arcsin(-2/alpha) from "
              f"{max(drive_phases):.4g} to {min(drive_phases):.4g} rad; atoms start on top.")
    else:
        print("Loading: non-inverted. Drive sin(omega t); atoms start at the bottom.")
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
    print(f"Hold time: {hold} = {scan['number_of_iterations']:,} steps{simulated}.")
    print(f"Initial quasimomentum: q = {float(INITIAL_QUASIMOMENTUM_KL):g} hbar kL for every atom "
          "(GPU start, Bloch-state model, level statistics).")
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
        print("Extended metrics: on (GPU, Bloch-state Floquet model and classical ensemble "
              "columns; level statistics on).")
    else:
        print("Extended metrics: off")
    if level_statistics:
        print(f"Level statistics: <r>, <r^2>, eta and eta_r_squared of the Floquet quasienergies "
              f"at q = {level_statistics['quasimomentum']:g} kL with plane waves "
              f"|n| <= {level_statistics['plane_wave_cutoff']}.")
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
            "iteration_order": "alpha_outer_frequency_inner",
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
            # None, or the walls' height (E_R), sigma (um) and centre gap (um).
            "green_walls": green_walls,
            # 7.0: starting quasimomentum (units of kL) and density smoothing (um).
            "initial_quasimomentum_kl": float(INITIAL_QUASIMOMENTUM_KL),
            "density_smoothing_um": float(DensitySmoothing_um),
            # 8.0: box, starting cloud width and hold time (None = gpe1d.config's steps).
            "grid_points": scan["grid_points"],
            "initial_cloud_width_um": scan["initial_cloud_width_um"],
            "hold_time_ms": scan["hold_time_ms"],
            "number_of_iterations": scan["number_of_iterations"],
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
