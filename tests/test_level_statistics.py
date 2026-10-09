"""Wigner-Dyson versus Poisson level statistics in the phase-diagram outputs."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tables as tb
from scipy.integrate import quad

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compact_sweep
import level_statistics as levels
import run_phase_diagram_CUDA as runner
from cuda_workflow_common import read_config, write_json_atomic
from make_phase_diagram_CUDA import analyze
from phase_metric import CSV_FIELDS, METRIC_NAME
from test_parallel_sweep import FAKE_SOLVER

SETTINGS = levels.validate_settings(0.0, 4)
COLUMNS = ("mean_r", "mean_r_squared", "eta", "eta_r_squared", "level_ratio_count")


class DefinitionTests(unittest.TestCase):
    def test_built_in_verification_passes(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(levels.verify())
        self.assertNotIn("FAIL", output.getvalue())

    def test_poisson_references_are_exact(self):
        self.assertAlmostEqual(levels.R_POISSON, quad(lambda r: 2 * r / (1 + r) ** 2, 0, 1)[0], places=12)
        self.assertAlmostEqual(levels.R2_POISSON, quad(lambda r: 2 * r * r / (1 + r) ** 2, 0, 1)[0], places=12)
        self.assertAlmostEqual(levels.R_POISSON, 0.386294, places=6)

    def test_goe_references_are_reproduced(self):
        r, r_error, r2, r2_error = levels.goe_reference(size=400, matrices=40, seed=3)
        self.assertLess(abs(r - levels.R_GOE), 5 * r_error + 1e-3)
        self.assertLess(abs(r2 - levels.R2_GOE), 5 * r2_error + 1e-3)

    def test_ratios_and_absolute_eta(self):
        np.testing.assert_allclose(levels.restricted_ratios([0.0, 2.0, 3.0, 7.0]), [0.5, 0.25])
        self.assertTrue(np.all(levels.restricted_ratios(np.random.default_rng(1).uniform(0, 9, 50)) <= 1))
        # Below the Poisson value eta stays positive: it is an absolute value.
        below = levels.statistics_from_ratios([0.2, 0.3])
        self.assertAlmostEqual(below["eta"], abs((0.25 - levels.R_POISSON) / (levels.R_GOE - levels.R_POISSON)))
        self.assertGreater(below["eta"], 0)
        self.assertEqual(levels.statistics_from_ratios([])["level_ratio_count"], 0)
        self.assertIsNone(levels.statistics_from_ratios([])["eta"])

    def test_point_results_are_deterministic_and_complete(self):
        first = levels.point_level_statistics(20.0, 60.0, 3.5e6, plane_wave_cutoff=8)
        second = levels.level_statistics_task((20.0, 60.0, 3.5e6, levels.validate_settings(0.0, 8)))
        self.assertEqual(first, second)
        self.assertEqual(first["level_ratio_count"], 17)       # 9 even + 8 odd levels
        generic = levels.point_level_statistics(20.0, 60.0, 3.5e6, quasimomentum=0.3, plane_wave_cutoff=8)
        self.assertEqual(generic["level_ratio_count"], 17)      # one sector, 2 n_max + 1 levels
        for key in COLUMNS[:-1]:
            self.assertTrue(math.isfinite(first[key]) and first[key] >= 0)

    def test_invalid_settings_are_rejected(self):
        for q, cutoff in ((1.0, 40), (-1.0, 40), (True, 40), (0.0, 1), (0.0, 40.0), (0.0, False)):
            with self.assertRaises(ValueError):
                levels.validate_settings(q, cutoff)


class SweepTests(unittest.TestCase):
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
            "status": "running", "storage_mode": "compact", "dataset_name": "levels",
            "parameter_grid": {
                "alpha_values": [10.0, 60.0], "drive_frequency_hz_values": [1.0e6, 3.5e6, 6.0e6],
                "lattice_depth_v0_er": 20.0, "initial_lattice_depth_v0_er": 40.0, "phase_radians": 0.0,
            },
            "analysis_contract": {"final_snapshot_count": 2, "cut_points_each_edge": 2,
                                  "metric": METRIC_NAME, "colormap": "inferno", "frequency_axis_inverted": True,
                                  "level_statistics": SETTINGS},
            "solver": {"points_per_batch": 4},
        }

    @staticmethod
    def generate(depth, alpha, frequency, phase, initial_depth, *, output_directory, config_path):
        for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
            output_directory.mkdir(parents=True, exist_ok=True)
            with tb.open_file(output_directory / name, "w") as handle:
                handle.create_array("/", "REAL", np.ones(12))
                handle.create_array("/", "IMAGINARY", np.zeros(12))

    @staticmethod
    def solve(command, cwd, log_path):
        for path in command[1:command.index("--device")]:
            config = read_config(Path(path))
            Path(config["phase_metric_file"]).write_text(json.dumps({
                "metric": METRIC_NAME, "value": 1.0, "snapshots_averaged": 2,
                "cut_points_each_edge": 2, "first_snapshot_iteration": 10,
                "last_snapshot_iteration": 20, "save_every_nth_iteration": 10,
                "number_of_iterations": 21}))
        log_path.write_text("log\n")
        return 0

    def run_sweep(self, solver=None):
        with (patch.object(compact_sweep, "SCRIPT_DIRECTORY", self.core),
              patch.object(compact_sweep, "run_logged", solver or self.solve),
              contextlib.redirect_stdout(io.StringIO())):
            compact_sweep.run_compact_sweep(
                results_root=self.results, manifest=self.manifest, base_config=self.base,
                executable=Path("specimen.exe"), device=0, floquet_mode="physical",
                create_initial_state=self.generate)

    def test_metrics_csv_has_level_columns_equal_to_direct_computation(self):
        self.run_sweep()
        with (self.results / "metrics.csv").open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(tuple(reader.fieldnames), CSV_FIELDS + COLUMNS)
            rows = list(reader)
        self.assertEqual(len(rows), 6)
        for row in rows:
            expected = levels.point_level_statistics(20.0, float(row["alpha"]),
                                                     float(row["drive_frequency_hz"]), plane_wave_cutoff=4)
            for key in COLUMNS[:-1]:
                self.assertAlmostEqual(float(row[key]), expected[key], places=12)
            self.assertEqual(int(row["level_ratio_count"]), 9)
        journal = [json.loads(line) for line in (self.results / "point_results.jsonl").read_text().splitlines()]
        self.assertTrue(all("floquet_time_steps" in record["level_statistics"] for record in journal))
        self.manifest["status"] = "completed"
        write_json_atomic(self.results / "run_manifest.json", self.manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = analyze(self.results / "run_manifest.json")
        with np.load(outputs["npz"]) as data:
            self.assertEqual(data["eta_matrix"].shape, (3, 2))
            self.assertAlmostEqual(float(data["eta_matrix"][0, 0]), float(rows[0]["eta"]), places=12)
        with outputs["csv"].open() as stream:
            self.assertEqual(tuple(csv.DictReader(stream).fieldnames), CSV_FIELDS + COLUMNS)

    def test_disabled_keeps_the_previous_columns(self):
        self.manifest["analysis_contract"]["level_statistics"] = None
        self.run_sweep()
        with (self.results / "metrics.csv").open() as stream:
            self.assertEqual(tuple(csv.DictReader(stream).fieldnames), CSV_FIELDS)
        journal = [json.loads(line) for line in (self.results / "point_results.jsonl").read_text().splitlines()]
        self.assertTrue(all("level_statistics" not in record for record in journal))

    def test_failed_batch_commits_nothing(self):
        calls = []
        def fail_second(command, cwd, log):
            calls.append(command)
            return 7 if len(calls) == 2 else self.solve(command, cwd, log)
        with self.assertRaisesRegex(RuntimeError, "exit code 7"):
            self.run_sweep(fail_second)
        journal = (self.results / "point_results.jsonl").read_text().splitlines()
        self.assertEqual(len(journal), 4)


class RunnerTests(unittest.TestCase):
    def plan(self, **settings) -> str:
        with (patch.multiple(runner, **settings),
              patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        return output.getvalue()

    def test_plan_reports_setting(self):
        self.assertIn("Level statistics: off", self.plan(CalculateLevelStatistics=False))
        self.assertIn("at q = 0 kL with plane waves |n| <= 40",
                      self.plan(CalculateLevelStatistics=True))

    def test_invalid_settings_stop_before_cuda(self):
        for settings in ({"CalculateLevelStatistics": "yes"},
                         {"CalculateLevelStatistics": True, "LevelStatisticsPlaneWaveCutoff": 1},
                         {"CalculateLevelStatistics": True, "LevelStatisticsQuasimomentum": 1.5}):
            with (patch.multiple(runner, **settings),
                  patch.object(runner, "locate_executable") as locate,
                  patch.object(sys, "argv", ["runner", "--plan"]),
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit, msg=str(settings)):
                    runner.main()
            locate.assert_not_called()


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """Real workers, merging and analysis with a stand-in solver."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="levels scan ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        (self.core / "gpe1d.config").write_text(
            "points_x=65536\nnumber_of_iterations=21\nsave_every_nth_iteration=10\n"
            "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
            f"test_events={self.root / 'events.jsonl'}\n")
        self.solver = self.root / "stand in solver.exe"
        self.solver.write_text(FAKE_SOLVER)
        self.solver.chmod(0o755)
        self.results = self.root / "results"

    def run_grid(self, storage):
        with (patch.object(runner, "SCRIPT_DIRECTORY", self.core),
              patch.multiple(runner, CalculateLevelStatistics=True, LevelStatisticsPlaneWaveCutoff=6,
                             NUMBER_OF_TABS=2, POINTS_PER_BATCH=2, ALPHA_VALUES=np.array([10.0, 60.0]),
                             DRIVE_FREQUENCY_HZ_VALUES=np.array([1.0e6, 3.0e6])),
              patch.object(sys, "argv", ["runner", "--headless", "--storage", storage,
                                        "--executable", str(self.solver), "--results", str(self.results)]),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(runner.main(), 0)
            return analyze(self.results / "run_manifest.json")

    def expected(self, alpha, frequency):
        return levels.point_level_statistics(20.0, alpha, frequency, plane_wave_cutoff=6)

    def test_two_tab_compact_scan_merges_level_columns(self):
        outputs = self.run_grid("compact")
        with (self.results / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["run_index"]) for row in rows], [0, 1, 2, 3])
        for row in rows:
            expected = self.expected(float(row["alpha"]), float(row["drive_frequency_hz"]))
            self.assertAlmostEqual(float(row["eta"]), expected["eta"], places=12)
            self.assertAlmostEqual(float(row["mean_r_squared"]), expected["mean_r_squared"], places=12)
        with np.load(outputs["npz"]) as data:
            self.assertFalse(np.isnan(data["mean_r_matrix"]).any())

    def test_two_tab_full_scan_reports_level_columns_in_analysis(self):
        outputs = self.run_grid("full")
        with outputs["csv"].open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        for row in rows:
            expected = self.expected(float(row["alpha"]), float(row["drive_frequency_hz"]))
            self.assertAlmostEqual(float(row["eta_r_squared"]), expected["eta_r_squared"], places=12)


if __name__ == "__main__":
    unittest.main()
