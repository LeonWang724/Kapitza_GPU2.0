"""10.0: a spread of starting quasimomenta (q runs per grid point, combined afterwards)."""

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

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_phase_diagram_CUDA as make  # noqa: E402
import quasimomentum_spread as qs  # noqa: E402
import run_phase_diagram_CUDA as runner  # noqa: E402
from cuda_workflow_common import (  # noqa: E402
    grid_points, phase_dataset_label, point_coordinates, point_count, quasimomentum_values,
)
from test_parallel_sweep import FAKE_SOLVER  # noqa: E402

GRID = {"alpha_values": [4.0, 8.0, 12.0], "drive_frequency_hz_values": [1e5, 2e5]}


class EnumerationTests(unittest.TestCase):
    def test_single_q_keeps_the_old_indices(self):
        grid = dict(GRID, initial_quasimomentum_kl=0.1)
        self.assertEqual(quasimomentum_values(grid), [0.1])
        self.assertEqual(point_count(grid), 6)
        for index in range(6):
            self.assertEqual(point_coordinates(grid, index)[:2], divmod(index, 2))
        points = list(grid_points({"parameter_grid": grid}))
        self.assertEqual(points[3], (3, 1, 1, 8.0, 2e5, 0.1))

    def test_q_is_the_innermost_axis(self):
        grid = dict(GRID, quasimomentum_values_kl=[0.0, 0.05, 0.1])
        self.assertEqual(point_count(grid), 18)
        points = list(grid_points({"parameter_grid": grid}))
        self.assertEqual([p[5] for p in points[:4]], [0.0, 0.05, 0.1, 0.0])
        self.assertEqual(points[4][:3], (4, 0, 1))
        self.assertEqual(point_coordinates(grid, 17), (2, 1, 2))
        partition = {"parameter_grid": grid, "work_partition": {"start": 5, "stop": 8}}
        self.assertEqual([p[0] for p in grid_points(partition)], [5, 6, 7])

    def test_label_names_the_q_range(self):
        label = phase_dataset_label(10, 20, 0.0, inverted=True, quasimomentum=0.0,
                                    quasimomentum_values=[0.0, 0.05, 0.1, 0.15, 0.2])
        self.assertTrue(label.endswith("_Inverted_Q0to0p2x5kL"), label)


class WeightTests(unittest.TestCase):
    def test_centred_spread_uses_mirror_images(self):
        weights, coverage = qs.spread_weights([0.0, 0.05, 0.1, 0.15, 0.2], 0.0, 0.1)
        self.assertAlmostEqual(coverage, math.erf(2.25 / math.sqrt(2)), places=12)
        self.assertAlmostEqual(sum(w for *_, w in weights), 1.0, places=12)
        by_q = {round(q, 6): (simulated, sign, w) for q, simulated, sign, w in weights}
        self.assertEqual(sorted(by_q), [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2])
        self.assertEqual(by_q[-0.1][:2], (0.1, -1))
        self.assertAlmostEqual(by_q[-0.1][2], by_q[0.1][2], places=14)
        self.assertGreater(by_q[0.0][2], by_q[0.05][2])

    def test_zero_spread_is_one_simulated_value(self):
        weights, coverage = qs.spread_weights([0.0, 0.05, 0.1], 0.05, 0.0)
        self.assertEqual((weights, coverage), ([(0.05, 0.05, 1, 1.0)], 1.0))
        self.assertEqual(qs.spread_weights([0.0, 0.05, 0.1], -0.1, 0.0)[0], [(-0.1, 0.1, -1, 1.0)])
        with self.assertRaises(ValueError):
            qs.spread_weights([0.0, 0.05, 0.1], 0.07, 0.0)

    def test_bad_values_are_refused(self):
        for values in ([], [0.1, 0.1], [1.0], [True, 0.1], ["a"]):
            with self.assertRaises(ValueError, msg=str(values)):
                qs.validate_values(values)
        with self.assertRaises(ValueError):
            qs.spread_weights([0.0, 0.1], 0.0, -0.1)


def cloud_series(q, x, times):
    """A made-up cloud of the group at q: two lumps drifting at q-dependent speeds."""
    frames = []
    for t in times:
        # Every group starts as the same cloud at x = 0, as in the simulation.
        lumps = (np.exp(-0.5 * ((x - 30000 * q * t) / (900 + 9000 * q * t)) ** 2)
                 + 0.4 * np.exp(-0.5 * ((x + 7000 * q * t) / (600 + 3000 * q * t)) ** 2))
        lumps *= 1.0 + 0.3 * np.cos(0.3 * x)
        frames.append(lumps / lumps.sum())
    return np.array(frames)


PIXEL = 174.0          # grid points per camera pixel (4 um / 23 nm)


def summary_row(frames, x):
    """The metrics.csv columns of one run, defined as the solver's."""
    centres = frames @ x
    seconds = frames @ (x * x)
    variances = seconds - centres ** 2
    pixel = np.floor(x / PIXEL + 0.5)
    pixel_variance = []
    for frame in frames:
        mass = np.bincount((pixel - pixel.min()).astype(int), weights=frame)
        middle = (np.arange(mass.size) + pixel.min()) * PIXEL
        mean = mass @ middle
        pixel_variance.append(mass @ (middle * middle) - mean * mean)
    return {
        "x_mean": centres.mean(), "x_mean_time_std": centres.std(),
        "sigma_x_squared": variances.mean(), "x_squared_mean": seconds.mean(),
        "sigma_x_4um_um": math.sqrt(np.mean(pixel_variance)) * qs.GRID_SPACING_UM,
        "sigma_x_um_initial": math.sqrt(variances[0]) * qs.GRID_SPACING_UM,
        "central_fraction": np.mean(frames[:, np.abs(x) < 3000].sum(axis=1)),
        "pendulum_cos": np.mean(frames @ np.cos(0.3 * x)),
        "escape_fraction": 0.0, "metric": np.mean((frames ** 2).sum(axis=1)),
    }


class CombinationTests(unittest.TestCase):
    """Combining rows must equal measuring the summed density directly."""

    def setUp(self):
        # Symmetric about x = 0, so x -> -x is reversing the array.
        self.x = np.arange(-20000.0, 20001.0)
        self.values = [0.0, 0.05, 0.1, 0.15, 0.2]
        times = np.linspace(0.0, 1.0, 30)
        self.frames = {q: cloud_series(q, self.x, times) for q in self.values}

    def direct(self, weights):
        mixture = sum(w * (self.frames[simulated] if sign > 0 else self.frames[simulated][:, ::-1])
                      for _, simulated, sign, w in weights)
        return summary_row(mixture, self.x)

    def combined(self, weights):
        rows = []
        for _, simulated, sign, _ in weights:
            row = summary_row(self.frames[simulated], self.x)
            rows.append(row)
        return qs.combine_cell(rows, weights)

    def test_centred_spread_is_exact(self):
        weights, _ = qs.spread_weights(self.values, 0.0, 0.1)
        direct, combined = self.direct(weights), self.combined(weights)
        for key in ("x_mean", "x_squared_mean", "central_fraction", "pendulum_cos"):
            self.assertAlmostEqual(combined[key], direct[key], delta=1e-9 * max(1, abs(direct[key])))
        self.assertAlmostEqual(combined["x_mean"], 0.0, delta=1e-6)
        self.assertAlmostEqual(math.sqrt(combined["sigma_x_squared"]),
                               math.sqrt(direct["sigma_x_squared"]),
                               delta=1e-9 * math.sqrt(direct["sigma_x_squared"]))
        self.assertAlmostEqual(combined["sigma_x_4um_um"], direct["sigma_x_4um_um"],
                               delta=2e-3 * direct["sigma_x_4um_um"])
        self.assertAlmostEqual(combined["sigma_x_um_initial"], direct["sigma_x_um_initial"],
                               delta=1e-3 * direct["sigma_x_um_initial"])

    def test_off_centre_spread_with_steadily_moving_centres_is_exact(self):
        # Each group's centre moves almost steadily here (its two lumps change
        # weight a little as they widen), as the estimate assumes.
        weights, _ = qs.spread_weights(self.values, 0.1, 0.05)
        direct, combined = self.direct(weights), self.combined(weights)
        self.assertAlmostEqual(combined["x_mean"], direct["x_mean"], delta=1e-9 * abs(direct["x_mean"]))
        self.assertAlmostEqual(combined["x_squared_mean"], direct["x_squared_mean"],
                               delta=1e-9 * direct["x_squared_mean"])
        self.assertGreater(direct["x_mean_time_std"], 100.0)     # a real wobble
        self.assertAlmostEqual(combined["x_mean_time_std"], direct["x_mean_time_std"],
                               delta=1e-3 * direct["x_mean_time_std"])
        self.assertAlmostEqual(combined["sigma_x_squared"], direct["sigma_x_squared"],
                               delta=1e-5 * direct["sigma_x_squared"])
        self.assertAlmostEqual(combined["sigma_x_4um_um"], direct["sigma_x_4um_um"],
                               delta=2e-3 * direct["sigma_x_4um_um"])

    def test_one_q_reproduces_that_run(self):
        weights, _ = qs.spread_weights(self.values, 0.1, 0.0)
        row = summary_row(self.frames[0.1], self.x)
        combined = qs.combine_cell([row], weights)
        for key in ("x_mean", "central_fraction", "pendulum_cos", "metric", "sigma_x_um_initial"):
            self.assertAlmostEqual(combined[key], row[key], places=12)
        self.assertAlmostEqual(combined["sigma_x_4um_um"], row["sigma_x_4um_um"],
                               delta=1e-3 * row["sigma_x_4um_um"])

    def test_rows_of_a_table_are_grouped_by_point(self):
        rows = []
        for index, (alpha, q) in enumerate((a, q) for a in (4.0, 8.0) for q in self.values):
            row = {"run_index": index, "alpha": alpha, "drive_frequency_hz": 1e5,
                   "initial_quasimomentum_kl": q, **summary_row(self.frames[q], self.x)}
            rows.append({key: str(value) for key, value in row.items()})
        self.assertTrue(qs.is_spread_table(rows))
        combined, coverage = qs.combine_rows(rows, 0.0, 0.1)
        self.assertEqual([row["run_index"] for row in combined], [0, 5])
        self.assertEqual(combined[1]["alpha"], 8.0)
        self.assertEqual(combined[0]["quasimomentum_spread_kl"], 0.1)
        self.assertAlmostEqual(combined[0]["quasimomentum_coverage"], coverage)
        self.assertFalse(qs.is_spread_table(rows[:1]))
        with self.assertRaises(ValueError):
            qs.combine_rows(rows[:-1], 0.0, 0.1)            # a missing q run


class RunnerTests(unittest.TestCase):
    def plan(self, **settings):
        with (patch.multiple(runner, NUMBER_OF_TABS=1, **settings),
              patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        return output.getvalue()

    def test_plan_reports_the_runs_and_the_spread(self):
        text = self.plan()
        self.assertEqual(runner.QUASIMOMENTUM_VALUES_KL, [0.0, 0.05, 0.10, 0.15, 0.20])
        self.assertIn("Scan: 500 points", text)
        self.assertIn("q = 0, 0.05, 0.1, 0.15, 0.2 hbar kL (100 grid points x 5 = 500 GPU runs", text)
        self.assertIn("centred at 0 hbar kL with spread 0.1 hbar kL; 97.6% of it", text)

    def test_none_is_the_single_q_of_9_0(self):
        text = self.plan(QUASIMOMENTUM_VALUES_KL=None)
        self.assertIn("Scan: 100 points", text)
        self.assertIn("Initial quasimomentum: q = 0.1 hbar kL for every atom", text)
        self.assertNotIn("Quasimomentum spread", text)

    def test_narrow_coverage_warns(self):
        text = self.plan(QUASIMOMENTUM_CENTER_KL=0.1, QUASIMOMENTUM_SPREAD_KL=0.1)
        self.assertIn("Warning: only", text)

    def test_bad_settings_are_refused(self):
        for settings in ({"QUASIMOMENTUM_VALUES_KL": [0.1, 0.1]},
                         {"QUASIMOMENTUM_VALUES_KL": "0.1"},
                         {"QUASIMOMENTUM_SPREAD_KL": 0.0, "QUASIMOMENTUM_CENTER_KL": 0.07}):
            with (patch.multiple(runner, **settings),
                  patch.object(sys, "argv", ["runner", "--plan"]),
                  contextlib.redirect_stderr(io.StringIO()),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaises(SystemExit, msg=str(settings)):
                    runner.main()
        with (patch.object(sys, "argv", ["runner", "--plan", "--storage", "full"]),
              contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(SystemExit):
                runner.main()


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """A compact spread scan in two tabs with a stand-in solver, then MAKE."""

    def test_two_tab_spread_scan_and_make(self):
        with tempfile.TemporaryDirectory(prefix="ten scan ") as temporary:
            root = Path(temporary)
            core = root / "core"
            core.mkdir()
            (core / "gpe1d.config").write_text(
                "points_x=65536\nnumber_of_iterations=3000\nsave_every_nth_iteration=100\n"
                "show_stats_every_nth_iteration=100\ntime_step=0.5\n"
                "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
                f"test_events={root / 'events.jsonl'}\n")
            solver = root / "stand in solver.exe"
            solver.write_text(FAKE_SOLVER)
            solver.chmod(0o755)
            results = root / "results"
            with (patch.object(runner, "SCRIPT_DIRECTORY", core),
                  patch.multiple(runner, LabMeasurement=False, DRIVE_PHASE_CONVENTION="theory",
                                 LATTICE_HOLD_MS=0),
                  patch.multiple(runner, INVERTED=True, NUMBER_OF_TABS=2, GRID_POINTS=65536,
                                 INITIAL_CLOUD_WIDTH_UM=30.0, HOLD_TIME_MS=None,
                                 QUASIMOMENTUM_VALUES_KL=[0.0, 0.1, 0.2],
                                 LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                                 ALPHA_VALUES=np.array([4.0, 8.0]),
                                 DRIVE_FREQUENCY_HZ_VALUES=np.array([1.0e6])),
                  patch.dict(os.environ, {"KAPITZA_CORE": str(CORE)}),
                  patch.object(sys, "argv", ["runner", "--executable", str(solver),
                                            "--results", str(results), "--headless"]),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 0)
            manifest = json.loads((results / "run_manifest.json").read_text())
            with open(results / "metrics.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with contextlib.redirect_stdout(io.StringIO()):
                data = make.dataset_analysis(results / "run_manifest.json")
                single = make.dataset_analysis(results / "run_manifest.json",
                                               quasimomentum_center=-0.1, quasimomentum_spread=0.0)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["completed_points"], 6)
        self.assertEqual(manifest["parameter_grid"]["quasimomentum_values_kl"], [0.0, 0.1, 0.2])
        self.assertTrue(manifest["parameter_label"].endswith("_Q0to0p2x3kL_Smooth4um"),
                        manifest["parameter_label"])
        self.assertEqual([float(row["initial_quasimomentum_kl"]) for row in rows],
                         [0.0, 0.1, 0.2, 0.0, 0.1, 0.2])
        self.assertEqual(list(rows[0])[:4], ["run_index", "alpha", "drive_frequency_hz",
                                             "initial_quasimomentum_kl"])
        # MAKE: one combined row per grid point, with the run's spread.
        self.assertEqual(len(data["rows"]), 2)
        first = data["rows"][0]
        self.assertEqual((first["initial_quasimomentum_kl"], first["quasimomentum_spread_kl"]),
                         (0.0, 0.1))
        self.assertIn("quasimomentum_coverage", data["fields"])
        self.assertEqual(len(data["fields"]), len(set(data["fields"])))
        self.assertTrue(np.isfinite(data["metric"]).all())
        # q = -0.1 alone is the mirror image of the q = 0.1 run: same width, centre flipped.
        run = next(row for row in rows if row["alpha"] == "4.0" and row["initial_quasimomentum_kl"] == "0.1")
        mirror = single["rows"][0]
        self.assertAlmostEqual(mirror["sigma_x_4um_um"], float(run["sigma_x_4um_um"]),
                               delta=1e-3 * float(run["sigma_x_4um_um"]))
        self.assertAlmostEqual(mirror["x_mean"], -float(run["x_mean"]), places=9)
        self.assertAlmostEqual(mirror["metric"], float(run["metric"]), places=12)


if __name__ == "__main__":
    unittest.main()
