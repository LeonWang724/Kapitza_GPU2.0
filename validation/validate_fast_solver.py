"""Check the fused and batched CUDA loops against the original loop on this GPU.

The fused loop must reproduce the reference loop bit for bit: identical metric
summaries, status CSV files and wavefunction snapshots. Points simulated
together as one GPU batch use a batched cuFFT plan, which may round
differently from a single transform; they must agree with single runs to
within --batch-rtol, and the report records whether they are bitwise equal.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tables as tb

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
from cuda_workflow_common import (
    locate_executable, query_json, read_config, run_logged, update_config, write_json_atomic,
)
from create_initial_state_function import create_init_state


def write_state(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", np.ascontiguousarray(values.real))
        handle.create_array("/", "IMAGINARY", np.ascontiguousarray(values.imag))


def read_state(path: Path) -> np.ndarray:
    """REAL and IMAGINARY rows exactly as stored, for bitwise comparison."""
    with tb.open_file(path, "r") as handle:
        return np.stack([handle.root.REAL[:], handle.root.IMAGINARY[:]])


def synthetic_inputs(directory: Path, system: int) -> dict[str, Path]:
    """Small lattice problem with absorbing edges, as in validate_compact.py."""
    points, step = 1024, 0.25
    x = (np.arange(points) - points // 2) * step
    psi = np.exp(-(x - 3.0 * system) ** 2 / 200.0 + 0.12j * x)
    psi /= np.sqrt(step * np.sum(np.abs(psi) ** 2))
    static = (-(0.2 + 0.05 * system) * np.cos(0.4 * x)
              - 0.005j * (np.abs(x) / np.max(np.abs(x))) ** 8)
    floquet = (-(0.07 + 0.02 * system) * np.cos(0.4 * x)).astype(np.complex128)
    paths = {name: directory / f"{name}.h5" for name in ("initial", "static", "floquet")}
    for name, values in (("initial", psi), ("static", static), ("floquet", floquet)):
        write_state(paths[name], values)
    return paths


def synthetic_config(directory: Path, case: dict, system: int, inputs: dict[str, Path]) -> Path:
    compact = case["storage"] == "compact"
    values = {
        "dimension": 1, "points_x": 1024, "step_x": 0.25, "time_step": 0.01,
        "number_of_iterations": case["iterations"],
        "save_every_nth_iteration": case["interval"],
        "show_stats_every_nth_iteration": case.get("stats", 100),
        "imaginary_time": "false", "dynamic_potential": "false",
        "floquet_potential": "false" if case.get("static") else "true",
        "floquet_omega": 1.7 + 0.45 * system, "beta": case.get("beta", 0.0),
        "initial_state_file": inputs["initial"], "potential_file": inputs["static"],
        "floquet_potential_file": inputs["floquet"], "save_psi": "true",
        "output_folder": directory / "out", "status_file": directory / "status.csv",
        "dp_save_potential": "false" if compact else "true",
        "dp_potential_output_folder": directory / "potentials",
        "phase_metric_file": directory / "metric.json" if compact else "",
        "phase_metric_final_snapshot_count": 30, "phase_metric_cut_points_each_edge": 5,
    }
    path = directory / "run.config"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return path


def realistic_config(directory: Path, system: int, iterations: int) -> Path:
    """The phase-diagram grid and inputs, shortened to `iterations` steps."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.config"
    shutil.copy2(CORE / "gpe1d.config", path)
    alpha, frequency = (30.0, 2.0e6), (90.0, 4.5e6)
    create_init_state(20.0, alpha[system % 2] + 7.0 * system, frequency[system % 2],
                      0.0, 40.0, output_directory=directory / "in", config_path=path)
    update_config(path, {
        "initial_state_file": directory / "in" / "lattice_gauss.h5",
        "potential_file": directory / "in" / "vstatic.h5",
        "floquet_potential_file": directory / "in" / "vflo.h5",
        "number_of_iterations": iterations, "output_folder": directory / "out",
        "status_file": directory / "status.csv", "phase_metric_file": directory / "metric.json",
        "phase_metric_final_snapshot_count": 30, "phase_metric_cut_points_each_edge": 100,
        "save_psi": "false", "dp_save_potential": "false",
    })
    return path


def run(executable: Path, configs: list[Path], device: int, log: Path, *extra: str) -> None:
    code = run_logged([str(executable), *map(str, configs), "--device", str(device),
                       "--floquet-mode", "physical", *extra], ROOT, log)
    if code != 0:
        raise RuntimeError(f"Solver failed with exit code {code}; see {log}")


def outputs(config: Path) -> dict:
    values = read_config(config)
    result = {"status": Path(values["status_file"]).read_text(encoding="utf-8")}
    if values.get("phase_metric_file"):
        result["metric"] = json.loads(Path(values["phase_metric_file"]).read_text(encoding="utf-8"))
    else:
        snapshots = {}
        for folder in ("output_folder", "dp_potential_output_folder"):
            directory = Path(values[folder])
            if directory.is_dir():
                for snapshot in sorted(directory.glob("*.h5")):
                    snapshots[f"{folder}/{snapshot.name}"] = read_state(snapshot)
        if not snapshots:
            raise AssertionError(f"No snapshots were written for {config}")
        result["snapshots"] = snapshots
    return result


def assert_bitwise(name: str, expected: dict, actual: dict) -> None:
    if expected.get("metric") != actual.get("metric"):
        raise AssertionError(f"{name}: metric summaries differ:\n{expected.get('metric')}\n{actual.get('metric')}")
    if expected["status"] != actual["status"]:
        raise AssertionError(f"{name}: status CSV files differ.")
    if expected.get("snapshots", {}).keys() != actual.get("snapshots", {}).keys():
        raise AssertionError(f"{name}: snapshot schedules differ.")
    for key, reference in expected.get("snapshots", {}).items():
        if not np.array_equal(reference.view(np.uint8), actual["snapshots"][key].view(np.uint8)):
            raise AssertionError(f"{name}: snapshot {key} differs.")


def relative_difference(expected: dict, actual: dict) -> float:
    """Largest relative difference over every numeric output."""
    worst = 0.0

    def compare(a: float, b: float) -> None:
        nonlocal worst
        if a is None or b is None:
            if a is not b:
                raise AssertionError("A null statistic became non-null.")
            return
        scale = max(abs(a), abs(b))
        if scale > 0:
            worst = max(worst, abs(a - b) / scale)

    for key, value in expected.get("metric", {}).items():
        if isinstance(value, float):
            compare(value, actual["metric"][key])
        elif value != actual["metric"][key]:
            raise AssertionError(f"Metric field {key} differs: {value} != {actual['metric'][key]}")
    reference_rows = [line.split(";") for line in expected["status"].splitlines()[1:]]
    actual_rows = [line.split(";") for line in actual["status"].splitlines()[1:]]
    if len(reference_rows) != len(actual_rows):
        raise AssertionError("Status CSV files have different lengths.")
    for row_a, row_b in zip(reference_rows, actual_rows):
        for a, b in zip(row_a, row_b):
            compare(float(a), float(b))
    for key, reference in expected.get("snapshots", {}).items():
        scale = float(np.max(np.abs(reference))) or 1.0
        worst = max(worst, float(np.max(np.abs(reference - actual["snapshots"][key]))) / scale)
    return worst


SYNTHETIC_CASES = [
    {"name": "compact_linear", "storage": "compact", "iterations": 347, "interval": 10},
    {"name": "compact_nonlinear_every_step", "storage": "compact", "iterations": 101,
     "interval": 1, "beta": 0.03},
    {"name": "compact_nonlinear_fused", "storage": "compact", "iterations": 260,
     "interval": 7, "stats": 13, "beta": 0.03},
    {"name": "compact_without_drive", "storage": "compact", "iterations": 120,
     "interval": 10, "static": True},
    {"name": "full_snapshots_and_potentials", "storage": "full", "iterations": 95,
     "interval": 10, "stats": 7},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--batch-rtol", type=float, default=1e-9)
    parser.add_argument("--realistic-iterations", type=int, default=3000)
    arguments = parser.parse_args()
    executable = locate_executable(arguments.executable)
    build = query_json(executable, "--version-json")
    if build.get("fused_solver_version") != 1:
        raise RuntimeError("Rebuild with BUILD_CUDA.bat before validating the fused solver.")
    timestamp = datetime.now(timezone.utc).strftime("fast_%Y%m%dT%H%M%S%fZ")
    results = (arguments.results or ROOT / "validation" / "results" / timestamp).resolve()
    results.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "build": build, "batch_rtol": arguments.batch_rtol, "checks": []}

    def record(name: str, **details) -> None:
        report["checks"].append({"name": name, "status": "passed", **details})
        write_json_atomic(results / "report.json", report)
        extra = ", ".join(f"{key}={value}" for key, value in details.items())
        print(f"PASS {name}" + (f" ({extra})" if extra else ""), flush=True)

    with tempfile.TemporaryDirectory(prefix=".check_", dir=results) as temporary:
        work = Path(temporary)
        try:
            for case in SYNTHETIC_CASES:
                systems = []
                for system in range(3):
                    inputs = synthetic_inputs(work / case["name"] / f"inputs_{system}", system)
                    systems.append({
                        mode: synthetic_config(work / case["name"] / f"{mode}_{system}", case, system, inputs)
                        for mode in ("reference", "fused", "batch")
                    })
                for system, configs in enumerate(systems):
                    run(executable, [configs["reference"]], arguments.device,
                        configs["reference"].with_name("solver.log"), "--reference-loop")
                    run(executable, [configs["fused"]], arguments.device,
                        configs["fused"].with_name("solver.log"))
                    assert_bitwise(f"{case['name']}[{system}]", outputs(configs["reference"]),
                                   outputs(configs["fused"]))
                record(f"{case['name']}: fused == reference (bitwise)", systems=len(systems))
                batch_configs = [configs["batch"] for configs in systems]
                run(executable, batch_configs, arguments.device, work / case["name"] / "batch.log")
                worst, bitwise = 0.0, True
                for configs in systems:
                    single, batched = outputs(configs["fused"]), outputs(configs["batch"])
                    difference = relative_difference(single, batched)
                    worst = max(worst, difference)
                    try:
                        assert_bitwise(case["name"], single, batched)
                    except AssertionError:
                        bitwise = False
                if not worst <= arguments.batch_rtol:
                    raise AssertionError(f"{case['name']}: batch differs from single runs by {worst:.3g}.")
                record(f"{case['name']}: batch of 3 == single runs", max_relative_difference=worst,
                       bitwise=bitwise)

            # The real phase-diagram grid size and inputs, shortened in time.
            iterations = arguments.realistic_iterations
            realistic = [realistic_config(work / "realistic" / f"point_{system}", system, iterations)
                         for system in range(4)]
            reference = realistic_config(work / "realistic" / "reference", 0, iterations)
            run(executable, [reference], arguments.device, work / "realistic" / "reference.log",
                "--reference-loop")
            run(executable, [realistic[0]], arguments.device, work / "realistic" / "single.log")
            assert_bitwise("realistic", outputs(reference), outputs(realistic[0]))
            record("realistic 65536-point grid: fused == reference (bitwise)", iterations=iterations)
            for system in range(1, 4):
                run(executable, [realistic[system]], arguments.device,
                    work / "realistic" / f"single_{system}.log")
            batch = [realistic_config(work / "realistic" / f"batch_{system}", system, iterations)
                     for system in range(4)]
            run(executable, batch, arguments.device, work / "realistic" / "batch.log")
            worst = max(relative_difference(outputs(single), outputs(batched))
                        for single, batched in zip(realistic, batch))
            if not worst <= arguments.batch_rtol:
                raise AssertionError(f"Realistic batch differs from single runs by {worst:.3g}.")
            record("realistic 65536-point grid: batch of 4 == single runs",
                   max_relative_difference=worst)
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            shutil.copytree(work, results / "failure_details", dirs_exist_ok=True)
            write_json_atomic(results / "report.json", report)
            raise
    write_json_atomic(results / "report.json", report)
    print(f"All fused-solver checks passed. Report: {results / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
