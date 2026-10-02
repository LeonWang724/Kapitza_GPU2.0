"""Codex CUDA Port: manifest-driven runner for the native CUDA executable."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from create_initial_state_function import create_init_state
from compact_sweep import run_compact_sweep
from cuda_workflow_common import (
    PORT_ROOT,
    RESULTS_DIRECTORY,
    SCRIPT_DIRECTORY,
    git_provenance,
    legacy_input_path,
    locate_executable,
    phase_dataset_label,
    query_json,
    read_config,
    run_logged,
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
PHASE_RADIANS = 0.0

# The physical update reproduced the established CPU phase diagram. The
# cumulative legacy update is intentionally unavailable in this user workflow.
FLOQUET_MODE = "physical"
CUDA_DEVICE = 0


def unique_results_directory() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("cuda_%Y%m%dT%H%M%SZ")
    parameter_label = phase_dataset_label(
        LATTICE_DEPTH_V0_ER, INITIAL_LATTICE_DEPTH_V0_ER, PHASE_RADIANS
    )
    path = RESULTS_DIRECTORY / f"{parameter_label}_{timestamp}"
    if path.exists():
        raise FileExistsError(f"Refusing to reuse results directory: {path}")
    return path


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--device", type=int, default=CUDA_DEVICE)
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

    executable = locate_executable(arguments.executable)
    build_info = query_json(executable, "--version-json")
    if arguments.storage == "compact" and build_info.get("compact_phase_metric_version") != 1:
        raise RuntimeError(
            "This executable does not support compact output. Run BUILD_CUDA.bat "
            "after updating the source, or use --storage full for the old snapshot workflow."
        )
    results_root = arguments.results.resolve() if arguments.results else unique_results_directory()
    if results_root.exists() and any(results_root.iterdir()):
        raise FileExistsError(f"Results directory must be absent or empty: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        # The supplied input generator uses working-directory-relative filenames.
        # Running it here preserves its numerical implementation unchanged.
        import os

        os.chdir(SCRIPT_DIRECTORY)
        base_config = SCRIPT_DIRECTORY / "gpe1d.config"
        if not base_config.is_file():
            raise FileNotFoundError(base_config)
        (SCRIPT_DIRECTORY / "in").mkdir(exist_ok=True)

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
                "floquet_mode": arguments.floquet_mode,
                "compatibility_default_confirmed": True,
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

        if arguments.storage == "compact":
            run_compact_sweep(
                results_root=results_root, manifest=manifest, base_config=base_config,
                executable=executable, device=arguments.device,
                floquet_mode=arguments.floquet_mode, create_initial_state=create_init_state,
            )
            manifest["status"] = "completed"
            manifest["finished_utc"] = utc_now()
            write_json_atomic(manifest_path, manifest)
            print(f"\nCompleted {manifest['completed_points']} points with compact output.")
            print(f"Results table: {results_root / 'metrics.csv'}")
            print("Create the phase diagram with MAKE_PHASE_DIAGRAM_CUDA.bat")
            return 0

        run_index = 0
        for alpha_index, alpha in enumerate(ALPHA_VALUES):
            for frequency_index, frequency_hz in enumerate(DRIVE_FREQUENCY_HZ_VALUES):
                output_directory = results_root / f"out_{run_index:03d}"
                input_directory = results_root / "inputs" / f"run_{run_index:03d}"
                config_path = results_root / "configs" / f"run_{run_index:03d}.config"
                status_path = results_root / "status" / f"run_{run_index:03d}.csv"
                log_path = results_root / "logs" / f"run_{run_index:03d}.log"
                output_directory.mkdir(parents=True, exist_ok=False)
                input_directory.mkdir(parents=True, exist_ok=False)
                config_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"\n--- Starting native CUDA run {run_index:03d} ---")
                print(f"alpha={alpha:.17g}, frequency={frequency_hz:.17g} Hz")
                create_init_state(
                    LATTICE_DEPTH_V0_ER,
                    float(alpha),
                    float(frequency_hz),
                    PHASE_RADIANS,
                    INITIAL_LATTICE_DEPTH_V0_ER,
                )

                copied_inputs: dict[str, Path] = {}
                for filename in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
                    destination = input_directory / filename
                    shutil.copy2(legacy_input_path(filename), destination)
                    copied_inputs[filename] = destination

                shutil.copy2(base_config, config_path)
                generated_values = read_config(base_config)
                update_config(
                    config_path,
                    {
                        "initial_state_file": copied_inputs["lattice_gauss.h5"],
                        "potential_file": copied_inputs["vstatic.h5"],
                        "floquet_potential_file": copied_inputs["vflo.h5"],
                        "output_folder": output_directory,
                        "status_file": status_path,
                    },
                )

                run_record = {
                    "run_index": run_index,
                    "alpha_index": alpha_index,
                    "frequency_index": frequency_index,
                    "alpha": float(alpha),
                    "drive_frequency_hz": float(frequency_hz),
                    "lattice_depth_v0_er": LATTICE_DEPTH_V0_ER,
                    "initial_lattice_depth_v0_er": INITIAL_LATTICE_DEPTH_V0_ER,
                    "phase_radians": PHASE_RADIANS,
                    "floquet_omega_dimensionless": float(generated_values["floquet_omega"]),
                    "status": "running",
                    "started_utc": utc_now(),
                    "output_directory": relative(output_directory, results_root),
                    "config_file": relative(config_path, results_root),
                    "status_file": relative(status_path, results_root),
                    "log_file": relative(log_path, results_root),
                    "config_sha256": sha256_file(config_path),
                    "inputs": {
                        name: {
                            "path": relative(path, results_root),
                            "sha256": sha256_file(path),
                        }
                        for name, path in copied_inputs.items()
                    },
                }
                manifest["runs"].append(run_record)
                write_json_atomic(manifest_path, manifest)

                command = [
                    str(executable),
                    str(config_path),
                    "--device",
                    str(arguments.device),
                    "--floquet-mode",
                    arguments.floquet_mode,
                ]
                return_code = run_logged(command, SCRIPT_DIRECTORY, log_path)
                run_record["return_code"] = return_code
                run_record["finished_utc"] = utc_now()
                if return_code != 0:
                    run_record["status"] = "failed"
                    manifest["status"] = "failed"
                    manifest["failed_run_index"] = run_index
                    write_json_atomic(manifest_path, manifest)
                    raise RuntimeError(
                        f"gpe1d_cuda.exe failed for run {run_index:03d}; see {log_path}"
                    )
                run_record["status"] = "completed"
                write_json_atomic(manifest_path, manifest)
                run_index += 1

        manifest["status"] = "completed"
        manifest["finished_utc"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        print(f"\nCompleted {run_index} runs.")
        print(f"Dataset: {results_root.name}")
        print(f"Manifest: {manifest_path}")
        print("Create the newest phase diagram with MAKE_PHASE_DIAGRAM_CUDA.bat")
        return 0
    finally:
        import os

        os.chdir(original_cwd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CUDA phase-diagram runner failed: {error}", file=sys.stderr)
        raise
