"""Compact orchestration, scalar integrity and legacy plotting interoperability."""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tables as tb

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
import compact_sweep
import run_phase_diagram_CUDA as runner
from cuda_workflow_common import read_config, write_json_atomic
from make_phase_diagram_CUDA import analyze
from phase_metric import METRIC_NAME, expected_sampling, validate_summary


def state_file(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", state.real)
        handle.create_array("/", "IMAGINARY", state.imag)


class CompactStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        (self.core / "create_initial_state_function.py").write_text("# specimen generator\n")
        self.config = {
            "points_x": "12", "number_of_iterations": "21", "save_every_nth_iteration": "10",
            "imaginary_time": "false", "floquet_omega": "1.0", "step_x": "0.25",
        }
        self.base = self.core / "gpe1d.config"
        self.base.write_text("".join(f"{k}={v}\n" for k, v in self.config.items()))
        self.contract = {
            "final_snapshot_count": 2, "cut_points_each_edge": 2,
            "metric": METRIC_NAME, "colormap": "inferno", "frequency_axis_inverted": True,
        }
        self.manifest = {
            "status": "running", "storage_mode": "compact", "dataset_name": "compact_test",
            "parameter_grid": {
                "alpha_values": [1.0, 2.0], "drive_frequency_hz_values": [10.0, 20.0],
                "lattice_depth_v0_er": 20.0, "initial_lattice_depth_v0_er": 40.0,
                "phase_radians": 0.0,
            },
            "analysis_contract": self.contract, "runs": [],
        }
        self.results = self.root / "compact"
        self.results.mkdir()
        self.full = self.root / "full"
        self.full.mkdir()
        self.full_manifest = copy.deepcopy(self.manifest)
        self.full_manifest["storage_mode"] = "full"
        self.expected = np.zeros((2, 2))
        self.calls = 0

    def generate(self, depth, alpha, frequency, phase, initial_depth):
        self.current = (alpha, frequency)
        values = np.arange(1, 13) * (alpha + frequency / 100) * (1 + 0.3j)
        self.current_states = [values, values * 2, values * 3]
        for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
            state_file(self.core / "in" / name, values)

    def solve(self, command, cwd, log_path):
        generated = read_config(Path(command[1]))
        self.assertEqual(generated["save_psi"], "false")
        self.assertEqual(generated["dp_save_potential"], "false")
        self.assertTrue(Path(generated["initial_state_file"]).is_file())
        # Independent snapshot-based reference, including imaginary components.
        values = [np.sum((s.real ** 2 + s.imag ** 2)[2:-2] ** 2) for s in self.current_states[-2:]]
        mean = float(np.mean(values))
        summary = {
            "metric": METRIC_NAME, "value": mean, "snapshots_averaged": 2,
            "cut_points_each_edge": 2, "first_snapshot_iteration": 10,
            "last_snapshot_iteration": 20, "save_every_nth_iteration": 10,
            "number_of_iterations": 21,
        }
        Path(generated["phase_metric_file"]).write_text(json.dumps(summary))
        Path(generated["status_file"]).write_text("status\n")
        log_path.write_text("solver diagnostics\n")
        alpha, frequency = self.current
        ai, fi = int(alpha - 1), int(frequency / 10 - 1)
        self.expected[fi, ai] = mean
        output = f"out_{self.calls:03d}"
        for iteration, state in zip((0, 10, 20), self.current_states):
            state_file(self.full / output / f"{iteration:016d}.h5", state)
        self.full_manifest["runs"].append({
            "run_index": self.calls, "alpha_index": ai, "frequency_index": fi,
            "alpha": alpha, "drive_frequency_hz": frequency,
            "status": "completed", "output_directory": output,
        })
        self.calls += 1
        return 0

    def run_sweep(self, solver=None):
        with (
            patch.object(compact_sweep, "SCRIPT_DIRECTORY", self.core),
            patch.object(compact_sweep, "legacy_input_path", lambda name: self.core / "in" / name),
            patch.object(compact_sweep, "run_logged", solver or self.solve),
        ):
            compact_sweep.run_compact_sweep(
                results_root=self.results, manifest=self.manifest, base_config=self.base,
                executable=Path("specimen.exe"), device=0, floquet_mode="physical",
                create_initial_state=self.generate,
            )

    def test_compact_sweep_and_plot_match_snapshot_reference(self):
        self.run_sweep()
        self.assertFalse(list(self.results.rglob("*.h5")))
        self.assertFalse(list(self.results.glob(".work_*")))
        self.assertFalse(list(self.results.glob("out_*")))
        self.assertEqual(self.manifest["completed_points"], 4)
        self.assertNotIn("runs", self.manifest)
        with (self.results / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        for manifest, root in ((self.manifest, self.results), (self.full_manifest, self.full)):
            manifest["status"] = "completed"
            write_json_atomic(root / "run_manifest.json", manifest)
        compact_outputs = analyze(self.results / "run_manifest.json")
        full_outputs = analyze(self.full / "run_manifest.json")
        with np.load(compact_outputs["npz"]) as compact, np.load(full_outputs["npz"]) as full:
            np.testing.assert_array_equal(compact["metric_matrix"], full["metric_matrix"])
            np.testing.assert_array_equal(compact["metric_matrix"], self.expected)
        self.assertTrue(compact_outputs["image"].is_file())

    def test_failed_point_preserves_prior_scalar_and_diagnostics(self):
        def fail_second(command, cwd, log):
            if self.calls == 1:
                log.write_text("intentional solver failure\n")
                return 7
            return self.solve(command, cwd, log)
        with self.assertRaisesRegex(RuntimeError, "exit code 7"):
            self.run_sweep(fail_second)
        self.assertEqual(self.manifest["status"], "failed")
        self.assertEqual(self.manifest["completed_points"], 1)
        self.assertEqual(self.manifest["failed_run_index"], 1)
        self.assertEqual(len((self.results / "point_results.jsonl").read_text().splitlines()), 1)
        self.assertTrue((self.results / "failed_point" / "solver.log").is_file())
        self.assertFalse(list(self.results.glob(".work_*")))

    def test_rejects_invalid_scalar_before_committing(self):
        def corrupt(command, cwd, log):
            self.solve(command, cwd, log)
            summary_path = Path(read_config(Path(command[1]))["phase_metric_file"])
            summary = json.loads(summary_path.read_text())
            summary["value"] = float("nan")
            summary_path.write_text(json.dumps(summary))
            return 0
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_sweep(corrupt)
        self.assertEqual((self.results / "point_results.jsonl").read_text(), "")

    def test_sampling_contract_short_runs_zero_cut_and_wrong_window(self):
        config = dict(self.config, number_of_iterations="9", save_every_nth_iteration="4")
        contract = dict(self.contract, final_snapshot_count=30, cut_points_each_edge=0)
        expected = expected_sampling(config, contract)
        self.assertEqual(expected["snapshots_averaged"], 3)
        self.assertEqual(expected["first_snapshot_iteration"], 0)
        self.assertEqual(expected["last_snapshot_iteration"], 8)
        summary = dict(expected, metric=METRIC_NAME, value=0.125)
        self.assertEqual(validate_summary(summary, config, contract), (0.125, 3))
        summary["first_snapshot_iteration"] = 1
        with self.assertRaisesRegex(ValueError, "first_snapshot_iteration"):
            validate_summary(summary, config, contract)

    def test_old_executable_is_rejected_before_creating_results(self):
        destination = self.root / "must_not_exist"
        with (
            patch.object(sys, "argv", ["runner", "--results", str(destination)]),
            patch.object(runner, "locate_executable", return_value=Path("old.exe")),
            patch.object(runner, "query_json", return_value={"port_version": "old"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "BUILD_CUDA"):
                runner.main()
        self.assertFalse(destination.exists())

    def test_host_cpp_sampling_and_json_contract(self):
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No host C++ compiler available")
        executable = self.root / "phase_metric_test"
        subprocess.run([
            compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic",
            "-I", str(ROOT / "src"), str(ROOT / "tests" / "phase_metric_host_test.cpp"),
            str(ROOT / "src" / "config.cpp"), "-o", str(executable),
        ], check=True, capture_output=True, text=True)
        summary_path = self.root / "summary.json"
        subprocess.run([str(executable), str(summary_path)], check=True)
        self.assertEqual(validate_summary(json.loads(summary_path.read_text()), self.config, self.contract), (2.0, 2))


if __name__ == "__main__":
    unittest.main()
