"""Snapshot-retaining sweep with private inputs and global point indices."""

from __future__ import annotations

from pathlib import Path

from cuda_workflow_common import (
    generate_point_inputs, grid_points, read_config, run_logged, sha256_file,
    update_config, utc_now, write_json_atomic,
)
from level_statistics import level_statistics_task


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_full_sweep(
    *, results_root: Path, manifest: dict, base_config: Path,
    executable: Path, device: int, floquet_mode: str, create_initial_state,
) -> None:
    grid = manifest["parameter_grid"]
    manifest_path = results_root / "run_manifest.json"
    manifest["runs"] = []
    manifest["completed_points"] = 0
    manifest["simulation_config"] = read_config(base_config)
    extra_arguments = [str(argument) for argument in
                       manifest.get("solver", {}).get("extra_arguments", [])]
    level_settings = manifest["analysis_contract"].get("level_statistics")
    for index, alpha_index, frequency_index, alpha, frequency in grid_points(manifest):
        output = results_root / f"out_{index:03d}"
        inputs_directory = results_root / "inputs" / f"run_{index:03d}"
        config = results_root / "configs" / f"run_{index:03d}.config"
        status = results_root / "status" / f"run_{index:03d}.csv"
        log = results_root / "logs" / f"run_{index:03d}.log"
        output.mkdir(parents=True, exist_ok=False)
        status.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n--- CUDA point {index:03d}: alpha={alpha:g}, frequency={frequency:g} Hz ---")
        record = {
            "run_index": index, "alpha_index": alpha_index,
            "frequency_index": frequency_index, "alpha": alpha,
            "drive_frequency_hz": frequency,
            "lattice_depth_v0_er": grid["lattice_depth_v0_er"],
            "initial_lattice_depth_v0_er": grid["initial_lattice_depth_v0_er"],
            "phase_radians": grid["phase_radians"],
            "status": "running", "started_utc": utc_now(),
            "output_directory": relative(output, results_root),
            "config_file": relative(config, results_root),
            "status_file": relative(status, results_root),
            "log_file": relative(log, results_root),
        }
        manifest["runs"].append(record)
        write_json_atomic(manifest_path, manifest)
        try:
            inputs = generate_point_inputs(
                base_config, config, inputs_directory, grid, alpha, frequency,
                create_initial_state,
            )
            update_config(config, {
                "initial_state_file": inputs["lattice_gauss.h5"],
                "potential_file": inputs["vstatic.h5"],
                "floquet_potential_file": inputs["vflo.h5"],
                "output_folder": output, "status_file": status,
                # A copied compact single-run config must still retain snapshots.
                "phase_metric_file": "", "save_psi": "true",
            })
            record["floquet_omega_dimensionless"] = float(read_config(config)["floquet_omega"])
            record["config_sha256"] = sha256_file(config)
            if level_settings:
                record["level_statistics"] = level_statistics_task(
                    (grid["lattice_depth_v0_er"], alpha, frequency, level_settings))
            record["inputs"] = {
                name: {"path": relative(path, results_root), "sha256": sha256_file(path)}
                for name, path in inputs.items()
            }
            write_json_atomic(manifest_path, manifest)
            code = run_logged(
                [str(executable), str(config), "--device", str(device),
                 "--floquet-mode", floquet_mode, *extra_arguments], results_root, log,
            )
            record["return_code"] = code
            if code != 0:
                raise RuntimeError(f"CUDA point {index} failed with exit code {code}; see {log}")
            record["status"] = "completed"
            record["finished_utc"] = utc_now()
            manifest["completed_points"] += 1
            write_json_atomic(manifest_path, manifest)
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = str(error)
            record["finished_utc"] = utc_now()
            manifest["status"] = "failed"
            manifest["failed_run_index"] = index
            write_json_atomic(manifest_path, manifest)
            raise
