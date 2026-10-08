"""Measure phase-diagram solver speed on this GPU and recommend POINTS_PER_BATCH.

Runs a few thousand iterations of real grid points from
run_phase_diagram_CUDA.py with the original loop and with the fused loop at
several batch sizes, then projects the time per grid point for the full
configured iteration count, including process startup.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
import run_phase_diagram_CUDA as runner
from create_initial_state_function import create_init_state
from cuda_workflow_common import (
    generate_point_inputs, locate_executable, query_json, read_config, update_config,
    write_json_atomic,
)

COMPLETE = re.compile(r"CUDA run complete: (\d+) iterations in ([0-9.eE+-]+) s")


def prepare_points(work: Path, count: int, iterations: int) -> list[Path]:
    grid = {
        "lattice_depth_v0_er": runner.LATTICE_DEPTH_V0_ER,
        "initial_lattice_depth_v0_er": runner.INITIAL_LATTICE_DEPTH_V0_ER,
        "phase_radians": runner.PHASE_RADIANS,
    }
    points = [(float(alpha), float(frequency)) for alpha in runner.ALPHA_VALUES
              for frequency in runner.DRIVE_FREQUENCY_HZ_VALUES]
    configs = []
    for index in range(count):
        alpha, frequency = points[index % len(points)]
        directory = work / f"point_{index:02d}"
        config = directory / "run.config"
        inputs = generate_point_inputs(CORE / "gpe1d.config", config, directory / "in", grid,
                                       alpha, frequency, create_init_state)
        update_config(config, {
            "initial_state_file": inputs["lattice_gauss.h5"],
            "potential_file": inputs["vstatic.h5"],
            "floquet_potential_file": inputs["vflo.h5"],
            "number_of_iterations": iterations,
            "output_folder": directory / "unused_snapshots",
            "status_file": directory / "status.csv",
            "phase_metric_file": directory / "metric.json",
            "phase_metric_final_snapshot_count": 30,
            "phase_metric_cut_points_each_edge": 100,
            "save_psi": "false", "dp_save_potential": "false",
        })
        configs.append(config)
    return configs


def timed_run(executable: Path, configs: list[Path], device: int, log: Path,
              *extra: str) -> tuple[float, float]:
    """Return (wall seconds including startup, seconds inside the time loop)."""
    command = [str(executable), *map(str, configs), "--device", str(device),
               "--floquet-mode", "physical", "--wait", "blocking", *extra]
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - start
    output = log.read_text(encoding="utf-8", errors="replace")
    match = COMPLETE.search(output)
    if code != 0 or match is None:
        tail = "\n".join(output.splitlines()[-5:])
        raise RuntimeError(f"Solver exited with code {code}:\n{tail}")
    return wall, float(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--device", type=int, default=runner.CUDA_DEVICE)
    parser.add_argument("--iterations", type=int, default=3000,
                        help="iterations per timed run (the projection scales to the config)")
    parser.add_argument("--batches", default="1,2,4,8,16,32",
                        help="comma-separated batch sizes to time")
    parser.add_argument("--results", type=Path)
    arguments = parser.parse_args()

    executable = locate_executable(arguments.executable)
    build = query_json(executable, "--version-json")
    if build.get("fused_solver_version") != 1:
        raise RuntimeError("Rebuild with BUILD_CUDA.bat before benchmarking.")
    limit = int(build.get("max_batch_systems", 1))
    batches = sorted({int(value) for value in arguments.batches.split(",") if value.strip()})
    batches = [batch for batch in batches if 1 <= batch <= limit] or [1]
    full_iterations = int(float(read_config(CORE / "gpe1d.config")["number_of_iterations"]))
    grid_points = len(runner.ALPHA_VALUES) * len(runner.DRIVE_FREQUENCY_HZ_VALUES)
    device = query_json(executable, "--device-info", str(arguments.device))
    print(f"GPU: {device.get('name')}  |  timing {arguments.iterations:,} of "
          f"{full_iterations:,} iterations per point  |  grid: {grid_points:,} points\n")

    timestamp = datetime.now(timezone.utc).strftime("benchmark_%Y%m%dT%H%M%SZ")
    results = (arguments.results or ROOT / "validation" / "results" / timestamp).resolve()
    results.mkdir(parents=True, exist_ok=False)
    report = {"build": build, "device": device, "iterations": arguments.iterations,
              "full_iterations": full_iterations, "runs": []}
    scale = full_iterations / arguments.iterations

    def measure(label: str, configs: list[Path], *extra: str) -> dict | None:
        try:
            wall, loop = timed_run(executable, configs, arguments.device,
                                   results / f"{label}.log", *extra)
        except RuntimeError as error:
            print(f"{label:>18}: stopped ({str(error).splitlines()[-1]})")
            return None
        startup = max(wall - loop, 0.0)
        entry = {"label": label, "batch": len(configs), "wall_seconds": wall,
                 "loop_seconds": loop, "startup_seconds": startup,
                 "projected_seconds_per_point": (loop * scale + startup) / len(configs)}
        report["runs"].append(entry)
        write_json_atomic(results / "report.json", report)
        return entry

    with tempfile.TemporaryDirectory(prefix=".bench_", dir=results) as temporary:
        configs = prepare_points(Path(temporary), max(batches), arguments.iterations)
        reference = measure("reference loop", configs[:1], "--reference-loop")
        fused = []
        for batch in batches:
            entry = measure(f"fused batch {batch}", configs[:batch])
            if entry is None:
                break
            fused.append(entry)
    if reference is None or not fused:
        raise RuntimeError(f"Benchmark could not complete; see logs in {results}")

    base = reference["projected_seconds_per_point"]
    print(f"{'run':>18} | {'s/point (full run)':>18} | {'speedup':>7} | {'startup s':>9}")
    for entry in [reference, *fused]:
        per_point = entry["projected_seconds_per_point"]
        print(f"{entry['label']:>18} | {per_point:18.2f} | {base / per_point:6.2f}x | "
              f"{entry['startup_seconds']:9.2f}")
    best = min(entry["projected_seconds_per_point"] for entry in fused)
    # The smallest batch within 5% of the best keeps progress fine-grained.
    chosen = next(entry for entry in fused if entry["projected_seconds_per_point"] <= 1.05 * best)
    report["recommended_points_per_batch"] = chosen["batch"]
    write_json_atomic(results / "report.json", report)
    hours_before = base * grid_points / 3600
    hours_after = chosen["projected_seconds_per_point"] * grid_points / 3600
    print(f"\nRecommended: POINTS_PER_BATCH = {chosen['batch']} "
          f"(set it in run_phase_diagram_CUDA.py or pass --batch {chosen['batch']}).")
    print(f"Projected for this {grid_points:,}-point grid in one tab: "
          f"{hours_after:.2f} h, versus {hours_before:.2f} h for the original loop.")
    print("A second tab can still hide input generation and process startup; more tabs")
    print("than that mostly share the same GPU time.")
    print(f"Report: {results / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
