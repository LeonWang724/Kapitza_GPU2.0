"""Compact sweeps that run several grid points per solver call."""

from __future__ import annotations

import contextlib
import csv
import io
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

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
import compact_sweep
from cuda_workflow_common import read_config
from phase_metric import METRIC_NAME


def write_state(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", values.real)
        handle.create_array("/", "IMAGINARY", values.imag)


class BatchedCompactSweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        (self.core / "create_initial_state_function.py").write_text("# specimen generator\n")
        self.base = self.core / "gpe1d.config"
        self.base.write_text("points_x=12\nnumber_of_iterations=21\nsave_every_nth_iteration=10\n"
                             "imaginary_time=false\nfloquet_omega=1.0\nstep_x=0.25\n")
        self.results = self.root / "compact"
        self.results.mkdir()
        self.manifest = {
            "status": "running", "storage_mode": "compact",
            "parameter_grid": {
                "alpha_values": [1.0, 2.0], "drive_frequency_hz_values": [10.0, 20.0, 30.0],
                "lattice_depth_v0_er": 20.0, "initial_lattice_depth_v0_er": 40.0,
                "phase_radians": 0.0,
            },
            "analysis_contract": {"final_snapshot_count": 2, "cut_points_each_edge": 2,
                                  "metric": METRIC_NAME},
            "solver": {"points_per_batch": 4, "extra_arguments": ["--wait", "blocking"]},
        }
        self.commands = []
        self.batches = []

    @staticmethod
    def generate(depth, alpha, frequency, phase, initial_depth, *, output_directory, config_path):
        values = np.full(12, alpha * 1000 + frequency, dtype=np.complex128)
        for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
            write_state(output_directory / name, values)

    def solve(self, command, cwd, log_path, *, corrupt_position=None):
        """Stand-in executable: one summary per config listed before the options."""
        self.commands.append(command)
        configs = [read_config(Path(path)) for path in command[1:command.index("--device")]]
        self.batches.append(configs)
        for position, config in enumerate(configs):
            with tb.open_file(config["initial_state_file"], "r") as handle:
                value = float(handle.root.REAL[0])
            summary = {
                "metric": METRIC_NAME, "value": float("nan") if position == corrupt_position else value,
                "snapshots_averaged": 2, "cut_points_each_edge": 2,
                "first_snapshot_iteration": 10, "last_snapshot_iteration": 20,
                "save_every_nth_iteration": 10, "number_of_iterations": 21,
            }
            Path(config["phase_metric_file"]).write_text(json.dumps(summary))
            Path(config["status_file"]).write_text("status\n")
        log_path.write_text("solver diagnostics\n")
        return 0

    def run_sweep(self, solver=None):
        with (
            patch.object(compact_sweep, "SCRIPT_DIRECTORY", self.core),
            patch.object(compact_sweep, "run_logged", solver or self.solve),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            compact_sweep.run_compact_sweep(
                results_root=self.results, manifest=self.manifest, base_config=self.base,
                executable=Path("specimen.exe"), device=0, floquet_mode="physical",
                create_initial_state=self.generate,
            )

    def journal(self):
        path = self.results / "point_results.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_points_run_in_batches_and_commit_in_grid_order(self):
        self.run_sweep()
        self.assertEqual([len(command[1:command.index("--device")]) for command in self.commands], [4, 2])
        self.assertTrue(all(command[-2:] == ["--wait", "blocking"] for command in self.commands))
        records = self.journal()
        self.assertEqual([record["run_index"] for record in records], list(range(6)))
        for record in records:
            self.assertEqual(record["metric_summary"]["value"],
                             record["alpha"] * 1000 + record["drive_frequency_hz"])
        self.assertEqual([record["solver_batch"]["position"] for record in records], [0, 1, 2, 3, 0, 1])
        self.assertEqual([record["solver_batch"]["first_run_index"] for record in records], [0] * 4 + [4] * 2)
        with (self.results / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["metric"]) for row in rows],
                         [record["metric_summary"]["value"] for record in records])
        self.assertEqual(self.manifest["completed_points"], 6)
        self.assertFalse(list(self.results.glob(".work_*")))

    def test_each_point_of_a_batch_has_private_inputs_and_outputs(self):
        self.run_sweep()
        first_batch = self.batches[0]
        self.assertEqual(len(first_batch), 4)
        for key in ("initial_state_file", "potential_file", "floquet_potential_file",
                    "status_file", "phase_metric_file"):
            self.assertEqual(len({config[key] for config in first_batch}), 4, key)
        self.assertEqual(len({config["floquet_omega"] for config in first_batch}), 1)

    def test_failed_batch_keeps_earlier_batches_and_commits_none_of_its_points(self):
        def fail_second_batch(command, cwd, log):
            if self.commands:
                self.commands.append(command)
                log.write_text("intentional solver failure\n")
                return 7
            return self.solve(command, cwd, log)
        with self.assertRaisesRegex(RuntimeError, "batch of points 4-5 failed with exit code 7"):
            self.run_sweep(fail_second_batch)
        self.assertEqual([record["run_index"] for record in self.journal()], [0, 1, 2, 3])
        self.assertEqual(self.manifest["completed_points"], 4)
        self.assertEqual(self.manifest["failed_run_index"], 4)
        self.assertEqual(self.manifest["failed_batch_run_indices"], [4, 5])
        failure = self.results / "failed_point"
        self.assertTrue((failure / "solver.log").is_file())
        self.assertEqual(len(json.loads((failure / "batch_records.json").read_text())), 2)
        self.assertTrue((failure / "point_01" / "run.config").is_file())

    def test_invalid_scalar_rejects_the_whole_batch(self):
        def corrupt_third(command, cwd, log):
            return self.solve(command, cwd, log, corrupt_position=2)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_sweep(corrupt_third)
        self.assertEqual(self.journal(), [])
        self.assertEqual(self.manifest["failed_run_index"], 2)
        with (self.results / "metrics.csv").open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])

    def test_batch_size_one_runs_one_config_per_call(self):
        self.manifest["solver"]["points_per_batch"] = 1
        self.run_sweep()
        self.assertEqual([len(command[1:command.index("--device")]) for command in self.commands], [1] * 6)
        self.assertTrue(all("solver_batch" not in record for record in self.journal()))

    def test_batched_rejects_nonpositive_size(self):
        with self.assertRaises(ValueError):
            list(compact_sweep.batched(range(3), 0))
        self.assertEqual(list(compact_sweep.batched(range(5), 2)), [[0, 1], [2, 3], [4]])


class FusedSolverHostTests(unittest.TestCase):
    """Builds the CPU equivalence test and type-checks the CUDA sources."""

    def setUp(self):
        self.compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if self.compiler is None:
            self.skipTest("No host C++ compiler available")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.stubs = ROOT / "tests" / "cuda_host_stubs"

    def test_fused_loop_matches_reference_loop_bit_for_bit(self):
        executable = Path(self.temporary.name) / "fused_solver_host_test"
        # -ffp-contract=off mirrors the CUDA build's --fmad=false.
        subprocess.run([
            self.compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-ffp-contract=off",
            "-I", str(self.stubs), "-I", str(ROOT / "src"),
            str(ROOT / "tests" / "fused_solver_host_test.cpp"), str(ROOT / "src" / "config.cpp"),
            "-o", str(executable),
        ], check=True, capture_output=True, text=True)
        completed = subprocess.run([str(executable)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertIn("bit for bit", completed.stdout)

    def test_cuda_sources_type_check_with_clang(self):
        probe = subprocess.run(
            [self.compiler, "-x", "cuda", "--cuda-host-only", "-nocudainc", "-nocudalib",
             "-fsyntax-only", "-"], input="", capture_output=True, text=True)
        if probe.returncode != 0:
            self.skipTest("This compiler cannot parse CUDA")
        for source in sorted((ROOT / "src").glob("*.cu")):
            completed = subprocess.run([
                self.compiler, "-x", "cuda", "--cuda-host-only", "-nocudainc", "-nocudalib",
                "-fsyntax-only", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                "-Wno-unused-command-line-argument", "-include", "cuda_runtime.h",
                "-I", str(self.stubs), "-I", str(ROOT / "src"), str(source),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, f"{source.name}:\n{completed.stderr[-3000:]}")
        for source in ("main.cpp", "config.cpp"):
            completed = subprocess.run([
                self.compiler, "-fsyntax-only", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                "-I", str(self.stubs), "-I", str(ROOT / "src"), str(ROOT / "src" / source),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, f"{source}:\n{completed.stderr[-3000:]}")


if __name__ == "__main__":
    unittest.main()
