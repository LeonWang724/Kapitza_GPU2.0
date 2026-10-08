"""Exact partition coverage, independent workers, failures, and merged plotting."""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))

import parallel_sweep
import run_phase_diagram_CUDA as runner
from create_initial_state_function import create_init_state
from cuda_workflow_common import read_config, sha256_file, write_json_atomic
from make_phase_diagram_CUDA import analyze
from phase_metric import METRIC_NAME


# A stand-in executable exercises real Python subprocesses, configs and files.
# It does no numerical evolution and is not a CUDA performance benchmark.
FAKE_SOLVER = '''#!/usr/bin/env python3
import json, math, os, sys, time
from pathlib import Path
if "--version-json" in sys.argv:
    print(json.dumps({"compact_phase_metric_version": 1, "phase_statistics_version": 1})); raise SystemExit(0)
if "--device-info" in sys.argv:
    print(json.dumps({"device_index": 0, "name": "Test solver"})); raise SystemExit(0)
config = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1); config[key.strip()] = value.strip()
for key in ("initial_state_file", "potential_file", "floquet_potential_file"):
    assert Path(config[key]).is_file(), key
events = Path(config["test_events"])
def event(kind):
    data = json.dumps({"kind": kind, "time": time.monotonic(), "pid": os.getpid(),
                       "config": sys.argv[1], "input": config["initial_state_file"]}) + "\\n"
    fd = os.open(events, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    os.write(fd, data.encode()); os.close(fd)
event("start")
failing = config.get("test_fail") == "true" and "tab_001" in Path(sys.argv[1]).parts
time.sleep(0.5 if config.get("test_fail") != "true" or failing else 5)
if failing:
    print("Intentional solver failure", flush=True); raise SystemExit(7)
iterations = int(float(config["number_of_iterations"]))
stride = int(float(config["save_every_nth_iteration"]))
schedule = list(range(0, iterations, stride))
value = float(config["floquet_omega"])
if config.get("phase_metric_file"):
    count = int(config["phase_metric_final_snapshot_count"])
    cut = int(config["phase_metric_cut_points_each_edge"])
    selected = schedule[-count:]
    summary = {"metric": "mean_discrete_sum_abs_psi_fourth_power", "value": value,
               "snapshots_averaged": len(selected), "cut_points_each_edge": cut,
               "first_snapshot_iteration": selected[0], "last_snapshot_iteration": selected[-1],
               "save_every_nth_iteration": stride, "number_of_iterations": iterations}
    step = float(config["step_x"])
    interior = int(config["points_x"]) - 2 * cut
    variance = step * step * (interior * interior - 1) / 12
    summary.update(statistics_version=1, metric_std=0.0, metric_variance=0.0,
                   x_mean=-0.5 * step, x_squared_mean=variance + 0.25 * step * step,
                   sigma_x_squared=variance, sigma_x=math.sqrt(variance),
                   x_mean_time_std=0.0, spatial_snapshots_averaged=len(selected))
    Path(config["phase_metric_file"]).write_text(json.dumps(summary))
else:
    import numpy as np, tables as tb
    points = int(config["points_x"])
    state = np.full(points, (value / (points - 200)) ** 0.25)
    for iteration in schedule:
        output = Path(config["output_folder"]) / f"{iteration:016d}.h5"
        with tb.open_file(output, "w") as handle:
            handle.create_array("/", "REAL", state)
            handle.create_array("/", "IMAGINARY", np.zeros(points))
Path(config["status_file"]).write_text("status\\n")
event("end")
'''


class PartitionTests(unittest.TestCase):
    def test_requested_ten_thousand_point_examples(self):
        for tabs, expected in ((20, [500] * 20), (11, [910] + [909] * 10)):
            ranges = parallel_sweep.split_point_ranges(10000, tabs)
            self.assertEqual([part["stop"] - part["start"] for part in ranges], expected)
            flattened = [point for part in ranges for point in range(part["start"], part["stop"])]
            self.assertEqual(flattened, list(range(10000)))

    def test_small_grids_odd_counts_and_excess_tabs_have_exact_coverage(self):
        for total in (1, 2, 7, 23, 100):
            for tabs in (1, 2, 3, 11, 20, 101):
                ranges = parallel_sweep.split_point_ranges(total, tabs)
                points = [i for part in ranges for i in range(part["start"], part["stop"])]
                counts = [part["stop"] - part["start"] for part in ranges]
                self.assertEqual(points, list(range(total)))
                self.assertTrue(all(count > 0 for count in counts))
                self.assertLessEqual(max(counts) - min(counts), 1)
                self.assertEqual(len(ranges), min(total, tabs))
        for total, tabs in ((0, 2), (10, 0), (10, -1)):
            with self.assertRaises(ValueError):
                parallel_sweep.split_point_ranges(total, tabs)

    def test_editable_tab_count_and_plan_do_not_require_cuda(self):
        with (patch.object(runner, "NUMBER_OF_TABS", 11),
              patch.object(runner, "ALPHA_VALUES", np.linspace(1, 100, 100)),
              patch.object(runner, "DRIVE_FREQUENCY_HZ_VALUES", np.linspace(1, 100, 100)),
              patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        self.assertIn("requested tabs: 11", output.getvalue())
        self.assertEqual(output.getvalue().count(": 910 points"), 1)
        self.assertEqual(output.getvalue().count(": 909 points"), 10)

    def test_invalid_edited_count_is_rejected_before_cuda(self):
        for count in (0, -1, 2.5, True):
            with (patch.object(runner, "NUMBER_OF_TABS", count),
                  patch.object(runner, "locate_executable") as locate,
                  patch.object(sys, "argv", ["runner"]),
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit):
                    runner.main()
            locate.assert_not_called()

    def test_windows_terminal_launcher_passes_spaced_paths_without_a_shell(self):
        with patch.object(parallel_sweep.subprocess, "Popen") as launch:
            parallel_sweep.launch_worker(Path("C:/Scan Folder/run_manifest.json"),
                                        "Kapitza 1/11", "Kapitza-test",
                                        headless=False, terminal="wt.exe")
        command = launch.call_args.args[0]
        self.assertEqual(command[:4], ["wt.exe", "--window", "Kapitza-test", "new-tab"])
        self.assertEqual(command[-1], "C:/Scan Folder/run_manifest.json")
        self.assertNotIn("shell", launch.call_args.kwargs)


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ParallelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="parallel scan ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        self.events = self.root / "events.jsonl"
        self.base = self.core / "gpe1d.config"
        self.base.write_text(
            "points_x=65536\nnumber_of_iterations=21\nsave_every_nth_iteration=10\n"
            "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
            f"test_events={self.events}\n"
        )
        self.original_hash = sha256_file(self.base)
        self.solver = self.root / "stand in solver.exe"
        self.solver.write_text(FAKE_SOLVER)
        self.solver.chmod(0o755)
        self.results = self.root / "results with spaces"

    def run_grid(self, storage="compact", fail=False, tabs=3, alpha_count=2):
        if fail:
            with self.base.open("a") as stream:
                stream.write("test_fail=true\n")
        with (patch.object(runner, "SCRIPT_DIRECTORY", self.core),
              patch.object(runner, "NUMBER_OF_TABS", tabs),
              patch.object(runner, "ALPHA_VALUES", np.linspace(1, 3, alpha_count)),
              patch.object(runner, "DRIVE_FREQUENCY_HZ_VALUES", np.array([10.0, 20.0])),
              patch.object(sys, "argv", ["runner", "--headless", "--storage", storage,
                                        "--executable", str(self.solver), "--results", str(self.results)]),
              contextlib.redirect_stdout(io.StringIO())):
            return runner.main()

    def test_real_workers_merge_every_point_and_keep_inputs_private(self):
        self.assertEqual(self.run_grid(alpha_count=3), 0)
        parent = parallel_sweep.read_manifest(self.results / "run_manifest.json")
        self.assertEqual(parent["status"], "completed")
        self.assertEqual(parent["completed_points"], 6)
        self.assertEqual(parent["parallel_execution"]["active_workers"], 3)
        records = [json.loads(line) for line in (self.results / "point_results.jsonl").read_text().splitlines()]
        self.assertEqual([record["run_index"] for record in records], list(range(6)))
        ratios = [record["metric_summary"]["value"] / record["drive_frequency_hz"] for record in records]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-14)
        self.assertEqual(sha256_file(self.base), self.original_hash)
        self.assertFalse((self.core / "in").exists())
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        starts = [event for event in events if event["kind"] == "start"]
        self.assertEqual(len(starts), 6)
        self.assertEqual(len({event["config"] for event in starts}), 3)
        self.assertEqual(len({event["input"] for event in starts}), 3)
        active = maximum = 0
        for event in sorted(events, key=lambda event: event["time"]):
            active += 1 if event["kind"] == "start" else -1
            maximum = max(maximum, active)
        self.assertGreaterEqual(maximum, 2)
        self.assertFalse(list(self.results.rglob(".work_*")))
        self.assertFalse(list(self.results.rglob("*.h5")))
        with (self.results / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["sigma_x_squared"] and row["metric_std"] for row in rows))
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = analyze(self.results / "run_manifest.json")
        with np.load(outputs["npz"]) as data:
            self.assertEqual(data["metric_matrix"].shape, (2, 3))
            np.testing.assert_allclose(data["sigma_x_squared_matrix"], records[0]["metric_summary"]["sigma_x_squared"])

    def test_uneven_full_workers_merge_paths_and_plot(self):
        self.assertEqual(self.run_grid(storage="full"), 0)
        parent = parallel_sweep.read_manifest(self.results / "run_manifest.json")
        self.assertEqual([entry["stop"] - entry["start"] for entry in parent["parallel_execution"]["workers"]], [2, 1, 1])
        self.assertEqual([record["run_index"] for record in parent["runs"]], list(range(4)))
        for record in parent["runs"]:
            self.assertTrue((self.results / record["config_file"]).is_file())
            self.assertEqual(len(list((self.results / record["output_directory"]).glob("*.h5"))), 3)
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = analyze(self.results / "run_manifest.json")
        with np.load(outputs["npz"]) as data:
            for record in parent["runs"]:
                self.assertAlmostEqual(data["metric_matrix"][record["frequency_index"], record["alpha_index"]],
                                       record["floquet_omega_dimensionless"], places=12)
        self.assertEqual(sha256_file(self.base), self.original_hash)

    def test_failure_never_publishes_a_completed_or_partial_combined_dataset(self):
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.run_grid(fail=True)
        parent = parallel_sweep.read_manifest(self.results / "run_manifest.json")
        self.assertEqual(parent["status"], "failed")
        self.assertFalse((self.results / "metrics.csv").exists())
        self.assertTrue(list(self.results.rglob("failed_point/solver.log")))
        for entry in parent["parallel_execution"]["workers"]:
            worker = parallel_sweep.read_manifest(self.results / entry["directory"] / "run_manifest.json")
            if worker.get("worker_pid"):
                self.assertFalse(parallel_sweep.process_exists(worker["worker_pid"]))
        for event in map(json.loads, self.events.read_text().splitlines()):
            self.assertFalse(parallel_sweep.process_exists(event["pid"]))

    def test_coordinator_interrupt_stops_workers_and_their_solver_children(self):
        real_sleep = parallel_sweep.time.sleep
        interrupted = False

        def interrupt_after_solver_starts(duration):
            nonlocal interrupted
            if not interrupted and self.events.exists() and self.events.stat().st_size:
                interrupted = True
                raise KeyboardInterrupt("Test coordinator interrupt")
            real_sleep(duration)

        with patch.object(parallel_sweep.time, "sleep", interrupt_after_solver_starts):
            with self.assertRaises(KeyboardInterrupt):
                self.run_grid()
        parent = parallel_sweep.read_manifest(self.results / "run_manifest.json")
        self.assertEqual(parent["status"], "cancelled")
        self.assertFalse((self.results / "metrics.csv").exists())
        for event in map(json.loads, self.events.read_text().splitlines()):
            self.assertFalse(parallel_sweep.process_exists(event["pid"]))


class MergeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"points_x": "12", "number_of_iterations": "21", "save_every_nth_iteration": "10"}
        self.parent = {
            "status": "running", "storage_mode": "compact", "simulation_config": self.config,
            "parameter_grid": {"alpha_values": [1.0], "drive_frequency_hz_values": [10.0, 20.0]},
            "analysis_contract": {"final_snapshot_count": 2, "cut_points_each_edge": 2},
            "parallel_execution": {"workers": [{"worker_index": 0, "start": 0, "stop": 2, "directory": "worker"}]},
        }
        summary = {"metric": METRIC_NAME, "value": 1.0, "snapshots_averaged": 2,
                   "cut_points_each_edge": 2, "first_snapshot_iteration": 10,
                   "last_snapshot_iteration": 20, "save_every_nth_iteration": 10,
                   "number_of_iterations": 21}
        self.records = [{"status": "completed", "run_index": i, "alpha_index": 0,
                         "frequency_index": i, "alpha": 1.0, "drive_frequency_hz": float((i + 1) * 10),
                         "metric_summary": copy.deepcopy(summary)} for i in range(2)]
        self.worker_root = self.root / "worker"
        self.worker_root.mkdir()
        self.worker = copy.deepcopy(self.parent)
        self.worker.update(status="completed", completed_points=2, point_results_file="point_results.jsonl",
                           work_partition={"worker_index": 0, "start": 0, "stop": 2})
        self.worker.pop("parallel_execution")
        write_json_atomic(self.worker_root / "run_manifest.json", self.worker)

    def write_records(self, records):
        (self.worker_root / "point_results.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    def test_missing_duplicate_and_wrong_parameters_are_rejected(self):
        wrong = copy.deepcopy(self.records)
        wrong[1]["drive_frequency_hz"] = 99
        for records in (self.records[:1], [self.records[0], self.records[0]], wrong):
            self.write_records(records)
            with self.assertRaises(ValueError):
                parallel_sweep.merge_workers(self.root, copy.deepcopy(self.parent))
            self.assertFalse((self.root / "metrics.csv").exists())

    def test_out_of_order_records_are_merged_in_global_order(self):
        self.write_records(list(reversed(self.records)))
        parallel_sweep.merge_workers(self.root, self.parent)
        self.assertEqual(self.parent["status"], "completed")
        records = [json.loads(line) for line in (self.root / "point_results.jsonl").read_text().splitlines()]
        self.assertEqual([record["run_index"] for record in records], [0, 1])


if __name__ == "__main__":
    unittest.main()
