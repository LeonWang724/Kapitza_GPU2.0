"""Compare the actual GPU scalar with NumPy analysis of identical full runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tables as tb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase_diagram" / "simulation_core"))
from cuda_workflow_common import locate_executable, query_json, run_logged, write_json_atomic
from phase_metric import METRIC_NAME, STATISTICS_FIELDS, statistics_columns, statistics_from_probabilities, validate_summary


def write_state(path: Path, values: np.ndarray) -> None:
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", values.real)
        handle.create_array("/", "IMAGINARY", values.imag)


def compare_case(executable: Path, work: Path, case: dict, device: int) -> dict:
    work.mkdir()
    points = 1024
    step = 0.25
    x = (np.arange(points) - points // 2) * step
    psi = np.exp(-x ** 2 / 200.0 + 0.12j * x)
    if case.get("zero_state"):
        psi[:] = 0
    elif case.get("translated_narrow"):
        psi = np.exp(-(x - 20.0) ** 2 / (2 * 0.6 ** 2) + 0.12j * x)
    norm = np.sqrt(step * np.sum(np.abs(psi) ** 2))
    if norm > 0:
        psi /= norm
    static = -0.2 * np.cos(0.4 * x) - 0.005j * (np.abs(x) / np.max(np.abs(x))) ** 8
    floquet = (-0.07 * np.cos(0.4 * x)).astype(np.complex128)
    for name, values in (("initial", psi), ("static", static), ("floquet", floquet)):
        write_state(work / f"{name}.h5", values)
    config = {
        "dimension": 1, "points_x": points, "step_x": step, "time_step": 0.01,
        "number_of_iterations": case["iterations"],
        "save_every_nth_iteration": case["interval"],
        "show_stats_every_nth_iteration": 100,
        "imaginary_time": "false", "dynamic_potential": "false",
        "floquet_potential": "true", "floquet_omega": 1.7, "beta": case.get("beta", 0.0),
        "initial_state_file": work / "initial.h5", "potential_file": work / "static.h5",
        "floquet_potential_file": work / "floquet.h5", "save_psi": "true",
        "phase_metric_final_snapshot_count": case["window"],
        "phase_metric_cut_points_each_edge": case["cut"],
    }
    for mode in ("full", "compact"):
        directory = work / mode
        directory.mkdir()
        config.update({
            "output_folder": directory / "out", "status_file": directory / "status.csv",
            # Save potential too in full mode; compact must suppress BOTH fields.
            "dp_save_potential": "true", "dp_potential_output_folder": directory / "potentials",
            "phase_metric_file": directory / "metric.json" if mode == "compact" else "",
        })
        path = directory / "run.config"
        path.write_text("".join(f"{key}={value}\n" for key, value in config.items()), encoding="utf-8")
        code = run_logged(
            [str(executable), str(path), "--device", str(device), "--floquet-mode", "physical"],
            ROOT, directory / "solver.log",
        )
        if code:
            raise RuntimeError(f"{case['name']} {mode} failed with exit code {code}.")
    snapshots = sorted((work / "full" / "out").glob("*.h5"), key=lambda p: int(p.stem))
    if len(snapshots) != len(range(0, case["iterations"], case["interval"])):
        raise AssertionError("Full solver emitted an unexpected snapshot schedule.")
    selected = snapshots[-case["window"]:]
    def probabilities():
        for snapshot in selected:
            with tb.open_file(snapshot, "r") as handle:
                yield handle.root.REAL[:] ** 2 + handle.root.IMAGINARY[:] ** 2
    reference_statistics = statistics_from_probabilities(probabilities(), cut=case["cut"], step_x=step)
    reference = reference_statistics["metric"]
    summary = json.loads((work / "compact" / "metric.json").read_text())
    compact, samples = validate_summary(summary, config, {
        "metric": METRIC_NAME, "final_snapshot_count": case["window"],
        "cut_points_each_edge": case["cut"], "statistics_version": 1,
    })
    compact_statistics = statistics_columns(summary, required=True)
    for key in STATISTICS_FIELDS:
        if reference_statistics[key] is None:
            if compact_statistics[key] is not None:
                raise AssertionError(f"{case['name']}: {key} must be null for a zero-mass state.")
        else:
            np.testing.assert_allclose(compact_statistics[key], reference_statistics[key], rtol=1e-10, atol=1e-12,
                                       err_msg=f"{case['name']}: {key}")
    if list((work / "compact").rglob("*.h5")):
        raise AssertionError("Compact solver wrote unwanted HDF5 snapshots.")
    np.testing.assert_allclose(compact, reference, rtol=1e-11, atol=1e-14)
    # Reading the metric must not change evolution or other diagnostics.
    full_status = np.loadtxt(work / "full" / "status.csv", delimiter=";", skiprows=1)
    compact_status = np.loadtxt(work / "compact" / "status.csv", delimiter=";", skiprows=1)
    np.testing.assert_allclose(compact_status, full_status, rtol=1e-11, atol=1e-14)
    return {
        "case": case, "status": "passed", "snapshot_metric": reference,
        "compact_metric": compact, "absolute_error": abs(compact - reference),
        "samples_averaged": samples, "compact_hdf5_files": 0,
        "snapshot_statistics": {key: reference_statistics[key] for key in STATISTICS_FIELDS},
        "compact_statistics": compact_statistics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--results", type=Path)
    arguments = parser.parse_args()
    executable = locate_executable(arguments.executable)
    build = query_json(executable, "--version-json")
    if build.get("compact_phase_metric_version") != 1 or build.get("phase_statistics_version") != 1:
        raise RuntimeError("Rebuild with BUILD_CUDA.bat before validating compact output.")
    timestamp = datetime.now(timezone.utc).strftime("compact_%Y%m%dT%H%M%S%fZ")
    results = (arguments.results or ROOT / "validation" / "results" / timestamp).resolve()
    results.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "build": build, "rtol": 1e-11, "atol": 1e-14,
              "statistics_rtol": 1e-10, "statistics_atol": 1e-12, "cases": []}
    cases = [
        {"name": "final_30", "iterations": 347, "interval": 10, "window": 30, "cut": 5},
        {"name": "short_zero_cut", "iterations": 9, "interval": 4, "window": 30, "cut": 0},
        {"name": "every_step_nonlinear", "iterations": 101, "interval": 1, "window": 30, "cut": 8, "beta": 0.03},
        {"name": "first_post_step", "iterations": 1, "interval": 100, "window": 30, "cut": 4},
        {"name": "zero_mass", "iterations": 9, "interval": 4, "window": 30, "cut": 0, "zero_state": True},
        {"name": "translated_narrow", "iterations": 13, "interval": 4, "window": 30,
         "cut": 5, "translated_narrow": True},
    ]
    with tempfile.TemporaryDirectory(prefix=".check_", dir=results) as temporary:
        try:
            for case in cases:
                outcome = compare_case(executable, Path(temporary) / case["name"], case, arguments.device)
                report["cases"].append(outcome)
                write_json_atomic(results / "report.json", report)
                print(f"PASS {case['name']}: compact={outcome['compact_metric']:.16g}")
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            shutil.copytree(temporary, results / "failure_details")
            write_json_atomic(results / "report.json", report)
            raise
    write_json_atomic(results / "report.json", report)
    print(f"All compact-output checks passed. Report: {results / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
