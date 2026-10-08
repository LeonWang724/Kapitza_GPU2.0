"""Codex CUDA Port: manifest-driven runner for the native CUDA executable."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from create_initial_state_function import create_init_state
from compact_sweep import run_compact_sweep
from full_sweep import run_full_sweep
from parallel_sweep import print_assignments, run_parallel_sweep, split_point_ranges
from cuda_workflow_common import (
    PORT_ROOT,
    RESULTS_DIRECTORY,
    SCRIPT_DIRECTORY,
    git_provenance,
    locate_executable,
    phase_dataset_label,
    query_json,
    sha256_file,
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
PHASE_RADIANS = 0.0

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


def unique_results_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("cuda_%Y%m%dT%H%M%SZ")
    parameter_label = phase_dataset_label(
        LATTICE_DEPTH_V0_ER, INITIAL_LATTICE_DEPTH_V0_ER, PHASE_RADIANS
    )
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
    # Snapshot output and the reference loop keep one point per solver run.
    batch = 1 if arguments.storage == "full" or arguments.reference_loop else arguments.batch
    total = len(ALPHA_VALUES) * len(DRIVE_FREQUENCY_HZ_VALUES)
    assignments = split_point_ranges(total, arguments.tabs)
    print_assignments(total, arguments.tabs, assignments)
    if batch > 1:
        print(f"Each tab simulates up to {batch} grid points at a time as one GPU batch.")
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
    # Sleeping instead of spinning while waiting for the GPU leaves CPU power
    # and thermal headroom to the GPU, especially with several tabs.
    extra_arguments = (["--wait", "blocking"] if fused else []) + (
        ["--reference-loop"] if arguments.reference_loop else [])
    results_root = arguments.results.resolve() if arguments.results else unique_results_directory()
    if results_root.exists() and any(results_root.iterdir()):
        raise FileExistsError(f"Results directory must be absent or empty: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    base_config = SCRIPT_DIRECTORY / "gpe1d.config"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)

    manifest_path = results_root / "run_manifest.json"
    parameter_label = phase_dataset_label(
        LATTICE_DEPTH_V0_ER, INITIAL_LATTICE_DEPTH_V0_ER, PHASE_RADIANS
    )
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
            "phase_radians": PHASE_RADIANS,
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
