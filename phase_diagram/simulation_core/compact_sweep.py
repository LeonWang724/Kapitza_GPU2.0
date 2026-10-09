"""Persist one scalar per grid point, using one disposable working directory.

Consecutive points can run together as one GPU batch; each keeps its own
inputs, config, status and scalar summary.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from cuda_workflow_common import (
    SCRIPT_DIRECTORY, generate_point_inputs, grid_points, read_config, run_logged, sha256_file,
    update_config, utc_now, write_json_atomic,
)
from level_statistics import level_statistics_task
from phase_metric import csv_fields, csv_row, expected_sampling, validate_summary


def batched(points, size: int):
    """Group consecutive grid points into GPU batches of at most `size`."""
    if size <= 0:
        raise ValueError("Points per batch must be positive.")
    batch = []
    for point in points:
        batch.append(point)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_compact_sweep(
    *, results_root: Path, manifest: dict, base_config: Path,
    executable: Path, device: int, floquet_mode: str,
    create_initial_state: Callable,
) -> None:
    manifest_path = results_root / "run_manifest.json"
    contract = manifest["analysis_contract"]
    solver = manifest.get("solver", {})
    batch_size = int(solver.get("points_per_batch", 1))
    extra_arguments = [str(argument) for argument in solver.get("extra_arguments", [])]
    level_settings = contract.get("level_statistics")
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
        writer = csv.DictWriter(table, fieldnames=csv_fields(contract))
        writer.writeheader()
        table.flush()
        # Reuse one small scratch directory for the entire sweep. It holds only
        # the current batch's inputs/logs/status/configs and scalar outputs.
        # Level statistics need only each point's parameters, so one background
        # thread computes them on the CPU while the GPU runs the batch.
        with (
            tempfile.TemporaryDirectory(prefix=".work_", dir=results_root) as temporary,
            ThreadPoolExecutor(max_workers=1) if level_settings else nullcontext() as levels,
        ):
            work = Path(temporary)
            for batch in batched(grid_points(manifest), batch_size):
                # Delete only this runner's previous scratch artifacts.
                for path in work.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                records, points = [], []
                failing_index = batch[0][0]
                pending = [levels.submit(level_statistics_task,
                                         (grid["lattice_depth_v0_er"], alpha, frequency, level_settings))
                           for _, _, _, alpha, frequency in batch] if levels else []
                try:
                    for position, (index, alpha_index, frequency_index, alpha, frequency) in enumerate(batch):
                        failing_index = index
                        record = {
                            "run_index": index, "alpha_index": alpha_index,
                            "frequency_index": frequency_index, "alpha": alpha,
                            "drive_frequency_hz": frequency, "status": "running",
                            "started_utc": utc_now(),
                        }
                        if len(batch) > 1:
                            record["solver_batch"] = {"size": len(batch), "position": position,
                                                      "first_run_index": batch[0][0]}
                        records.append(record)
                        print(f"\n--- Compact CUDA point {index:03d}: alpha={alpha:g}, frequency={frequency:g} Hz ---")
                        point_work = work / f"point_{position:02d}"
                        config_path = point_work / "run.config"
                        inputs = generate_point_inputs(
                            base_config, config_path, point_work / "in", grid,
                            alpha, frequency, create_initial_state,
                        )
                        summary_path = point_work / "metric.json"
                        update_config(config_path, {
                            "initial_state_file": inputs["lattice_gauss.h5"],
                            "potential_file": inputs["vstatic.h5"],
                            "floquet_potential_file": inputs["vflo.h5"],
                            "output_folder": point_work / "unused_snapshots",
                            "status_file": point_work / "status.csv",
                            "phase_metric_file": summary_path,
                            "phase_metric_final_snapshot_count": contract["final_snapshot_count"],
                            "phase_metric_cut_points_each_edge": contract["cut_points_each_edge"],
                            "save_psi": "false", "dp_save_potential": "false",
                        })
                        generated = read_config(config_path)
                        record["floquet_omega_dimensionless"] = float(generated["floquet_omega"])
                        record["input_sha256"] = {name: sha256_file(path) for name, path in inputs.items()}
                        record["config_sha256"] = sha256_file(config_path)
                        points.append((config_path, summary_path, generated))
                    failing_index = batch[0][0]
                    if len(batch) > 1:
                        print(f"\nSimulating points {batch[0][0]:03d}-{batch[-1][0]:03d} "
                              f"together as one GPU batch of {len(batch)}.")
                    log_path = work / "solver.log"
                    code = run_logged(
                        [str(executable), *(str(config) for config, _, _ in points),
                         "--device", str(device), "--floquet-mode", floquet_mode,
                         *extra_arguments], work, log_path,
                    )
                    for record in records:
                        record["return_code"] = code
                    if code != 0:
                        label = (f"point {batch[0][0]}" if len(batch) == 1 else
                                 f"batch of points {batch[0][0]}-{batch[-1][0]}")
                        raise RuntimeError(f"CUDA {label} failed with exit code {code}.")
                    # Validate the whole batch before committing any of it.
                    values = []
                    for record, (_, summary_path, generated) in zip(records, points):
                        failing_index = record["run_index"]
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        value, _ = validate_summary(summary, generated, contract)
                        record["metric_summary"] = summary
                        values.append(value)
                    for record, future in zip(records, pending):
                        failing_index = record["run_index"]
                        record["level_statistics"] = future.result()
                    for record, (_, _, generated) in zip(records, points):
                        record["status"] = "completed"
                        record["finished_utc"] = utc_now()
                        # Append each point once, avoiding quadratic rewrites of
                        # all previous records on large grids.
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        writer.writerow(csv_row(record, record["metric_summary"], generated, contract))
                    journal.flush()
                    os.fsync(journal.fileno())
                    table.flush()
                    os.fsync(table.fileno())
                    manifest["completed_points"] += len(batch)
                    write_json_atomic(manifest_path, manifest)
                    for record, value in zip(records, values):
                        print(f"Saved scalar metric for point {record['run_index']:03d}: {value:.12g}")
                except BaseException as error:
                    # Keep one failed batch's small diagnostics for inspection.
                    failure = results_root / "failed_point"
                    shutil.copytree(work, failure, dirs_exist_ok=True)
                    (failure / "error.txt").write_text(str(error), encoding="utf-8")
                    for record in records:
                        record["status"] = "failed"
                        record["finished_utc"] = utc_now()
                    failed = next((record for record in records if record["run_index"] == failing_index),
                                  records[0] if records else {"run_index": failing_index})
                    write_json_atomic(failure / "point_record.json", failed)
                    if len(records) > 1:
                        write_json_atomic(failure / "batch_records.json", records)
                    manifest["status"] = "failed"
                    manifest["failed_run_index"] = failing_index
                    if len(batch) > 1:
                        manifest["failed_batch_run_indices"] = [point[0] for point in batch]
                    write_json_atomic(manifest_path, manifest)
                    for future in pending:
                        future.cancel()
                    raise
