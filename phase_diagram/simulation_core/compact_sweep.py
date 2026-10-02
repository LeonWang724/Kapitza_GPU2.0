"""Persist one scalar per grid point, using one disposable working directory."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from cuda_workflow_common import (
    SCRIPT_DIRECTORY, legacy_input_path, read_config, run_logged, sha256_file,
    update_config, utc_now, write_json_atomic,
)
from phase_metric import CSV_FIELDS, expected_sampling, validate_summary


def run_compact_sweep(
    *, results_root: Path, manifest: dict, base_config: Path,
    executable: Path, device: int, floquet_mode: str,
    create_initial_state: Callable,
) -> None:
    manifest_path = results_root / "run_manifest.json"
    contract = manifest["analysis_contract"]
    manifest["simulation_config"] = read_config(base_config)
    expected_sampling(manifest["simulation_config"], contract)
    provenance = results_root / "provenance"
    provenance.mkdir()
    shutil.copy2(base_config, provenance / "gpe1d.config")
    # Preserve the mass, wavelength and state-generation choices once per grid.
    generator = SCRIPT_DIRECTORY / "create_initial_state_function.py"
    shutil.copy2(generator, provenance / generator.name)
    manifest["input_generator_sha256"] = sha256_file(generator)
    manifest.pop("runs", None)
    manifest["point_results_file"] = "point_results.jsonl"
    manifest["completed_points"] = 0
    write_json_atomic(manifest_path, manifest)
    grid = manifest["parameter_grid"]

    with (
        (results_root / "metrics.csv").open("w", encoding="utf-8", newline="") as table,
        (results_root / "point_results.jsonl").open("w", encoding="utf-8") as journal,
    ):
        writer = csv.DictWriter(table, fieldnames=CSV_FIELDS)
        writer.writeheader()
        table.flush()
        # Reuse one small scratch directory for the entire sweep. It holds only
        # the current point's inputs/log/status/config and scalar output.
        with tempfile.TemporaryDirectory(prefix=".work_", dir=results_root) as temporary:
            work = Path(temporary)
            for alpha_index, alpha in enumerate(grid["alpha_values"]):
                for frequency_index, frequency in enumerate(grid["drive_frequency_hz_values"]):
                    index = alpha_index * len(grid["drive_frequency_hz_values"]) + frequency_index
                    record = {
                        "run_index": index, "alpha_index": alpha_index,
                        "frequency_index": frequency_index, "alpha": alpha,
                        "drive_frequency_hz": frequency, "status": "running",
                        "started_utc": utc_now(),
                    }
                    print(f"\n--- Compact CUDA point {index:03d}: alpha={alpha:g}, frequency={frequency:g} Hz ---")
                    # Delete only this runner's previous scratch artifacts.
                    for path in work.iterdir():
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                    try:
                        create_initial_state(
                            grid["lattice_depth_v0_er"], float(alpha), float(frequency),
                            grid["phase_radians"], grid["initial_lattice_depth_v0_er"],
                        )
                        inputs = {}
                        for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
                            inputs[name] = work / name
                            shutil.copy2(legacy_input_path(name), inputs[name])
                        config_path = work / "run.config"
                        shutil.copy2(base_config, config_path)
                        summary_path = work / "metric.json"
                        update_config(config_path, {
                            "initial_state_file": inputs["lattice_gauss.h5"],
                            "potential_file": inputs["vstatic.h5"],
                            "floquet_potential_file": inputs["vflo.h5"],
                            "output_folder": work / "unused_snapshots",
                            "status_file": work / "status.csv",
                            "phase_metric_file": summary_path,
                            "phase_metric_final_snapshot_count": contract["final_snapshot_count"],
                            "phase_metric_cut_points_each_edge": contract["cut_points_each_edge"],
                            "save_psi": "false", "dp_save_potential": "false",
                        })
                        generated = read_config(config_path)
                        record["floquet_omega_dimensionless"] = float(generated["floquet_omega"])
                        record["input_sha256"] = {name: sha256_file(path) for name, path in inputs.items()}
                        record["config_sha256"] = sha256_file(config_path)
                        log_path = work / "solver.log"
                        code = run_logged(
                            [str(executable), str(config_path), "--device", str(device),
                             "--floquet-mode", floquet_mode], SCRIPT_DIRECTORY, log_path,
                        )
                        record["return_code"] = code
                        if code != 0:
                            raise RuntimeError(f"CUDA point {index} failed with exit code {code}.")
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        value, samples = validate_summary(summary, generated, contract)
                        record["metric_summary"] = summary
                        record["status"] = "completed"
                        record["finished_utc"] = utc_now()
                        # Append each point once, avoiding quadratic rewrites of
                        # all previous records on large grids.
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        os.fsync(journal.fileno())
                        writer.writerow({
                            "run_index": index, "alpha": alpha, "drive_frequency_hz": frequency,
                            "snapshots_averaged": samples, "metric": value,
                        })
                        table.flush()
                        os.fsync(table.fileno())
                        manifest["completed_points"] = index + 1
                        write_json_atomic(manifest_path, manifest)
                        print(f"Saved scalar metric: {value:.12g}")
                    except BaseException as error:
                        # Keep one failed point's small diagnostics for inspection.
                        failure = results_root / "failed_point"
                        shutil.copytree(work, failure, dirs_exist_ok=True)
                        (failure / "error.txt").write_text(str(error), encoding="utf-8")
                        record["status"] = "failed"
                        record["finished_utc"] = utc_now()
                        write_json_atomic(failure / "point_record.json", record)
                        manifest["status"] = "failed"
                        manifest["failed_run_index"] = index
                        write_json_atomic(manifest_path, manifest)
                        raise
