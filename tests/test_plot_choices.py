"""Choosing plotted columns, colour limits and combined datasets in MAKE."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.axes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))

import make_phase_diagram_CUDA as make
from phase_metric import METRIC_NAME

ALPHAS = [10.0, 20.0, 30.0]
FREQUENCIES = [1.0e6, 2.0e6]


def write_dataset(root: Path, name: str, wall_height: float | None) -> Path:
    """A completed compact dataset whose metric encodes alpha, frequency and wall."""
    directory = root / name
    directory.mkdir(parents=True)
    walls = None if wall_height is None else {"height_er": wall_height, "sigma_um": 5.0, "gap_um": 200.0}
    records = []
    for index, (ai, fi) in enumerate((a, f) for a in range(3) for f in range(2)):
        value = ALPHAS[ai] + FREQUENCIES[fi] / 1e6 + (wall_height or 0.0) / 100
        records.append({
            "run_index": index, "alpha_index": ai, "frequency_index": fi,
            "alpha": ALPHAS[ai], "drive_frequency_hz": FREQUENCIES[fi], "status": "completed",
            "metric_summary": {
                "metric": METRIC_NAME, "value": value, "snapshots_averaged": 2,
                "cut_points_each_edge": 2, "first_snapshot_iteration": 10,
                "last_snapshot_iteration": 20, "save_every_nth_iteration": 10,
                "number_of_iterations": 21, "statistics_version": 1,
                "metric_std": value / 10, "metric_variance": (value / 10) ** 2,
                "x_mean": 0.0, "x_squared_mean": 4.0, "sigma_x_squared": 4.0, "sigma_x": 2.0,
                "x_mean_time_std": 0.0, "spatial_snapshots_averaged": 2,
            },
        })
    (directory / "point_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = {
        "status": "completed", "storage_mode": "compact", "dataset_name": name,
        "point_results_file": "point_results.jsonl",
        "simulation_config": {"points_x": "12", "number_of_iterations": "21",
                              "save_every_nth_iteration": "10", "imaginary_time": "false"},
        "parameter_grid": {"alpha_values": ALPHAS, "drive_frequency_hz_values": FREQUENCIES,
                           "lattice_depth_v0_er": 20.0, "initial_lattice_depth_v0_er": 40.0,
                           "phase_radians": 0.0, "green_walls": walls},
        "analysis_contract": {"final_snapshot_count": 2, "cut_points_each_edge": 2,
                              "metric": METRIC_NAME, "statistics_version": 1,
                              "colormap": "inferno", "frequency_axis_inverted": True},
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    return directory


class PlotChoiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.walls_off = write_dataset(self.root, "walls_off", None)
        self.walls_50 = write_dataset(self.root, "walls_50", 50.0)

    def rows(self, directory):
        with contextlib.redirect_stdout(io.StringIO()):
            return make.dataset_analysis(directory / "run_manifest.json")["rows"]

    def captured_plot(self, call):
        """Run `call` and return the pcolormesh arguments it used."""
        original = matplotlib.axes.Axes.pcolormesh
        calls = []

        def spy(axis, *arguments, **keywords):
            calls.append({"arguments": arguments, "keywords": keywords})
            return original(axis, *arguments, **keywords)

        with patch.object(matplotlib.axes.Axes, "pcolormesh", spy), contextlib.redirect_stdout(io.StringIO()):
            result = call()
        # The first call draws the diagram; later ones come from the colour bar.
        return result, calls[0]

    def test_default_is_ipr_against_alpha_and_frequency(self):
        outputs, captured = self.captured_plot(lambda: make.analyze(self.walls_off / "run_manifest.json"))
        self.assertTrue(outputs["image"].name.endswith("_phase_diagram.png"))
        x_mesh, y_mesh, values = captured["arguments"]
        np.testing.assert_array_equal(x_mesh[0], ALPHAS)
        np.testing.assert_array_equal(y_mesh[:, 0], FREQUENCIES)
        self.assertEqual(values[1, 2], 30.0 + 2.0)
        self.assertIsNone(captured["keywords"]["vmin"])
        with outputs["csv"].open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row["green_wall_height_er"]), 0.0)
        self.assertEqual(row["green_wall_sigma_um"], "")

    def test_std_colour_is_the_cloud_width_with_limits(self):
        outputs, captured = self.captured_plot(lambda: make.analyze(
            self.walls_off / "run_manifest.json", color="sigma_x_um", color_limits=(0.0, 0.8)))
        self.assertTrue(outputs["image"].name.endswith("_sigma_x_um.png"))
        self.assertEqual((captured["keywords"]["vmin"], captured["keywords"]["vmax"]), (0.0, 0.8))
        # sigma_x = 2 grid spacings of 23 nm in the specimen data.
        self.assertAlmostEqual(captured["arguments"][2][0, 0], 2 * 0.023)
        self.assertIn("Standard deviation of $|\\psi|^2$", make.COLUMN_TITLES["sigma_x_um"])
        with outputs["csv"].open() as stream:
            self.assertAlmostEqual(float(next(csv.DictReader(stream))["sigma_x_um"]), 0.046)

    def test_ipr_time_fluctuation_is_not_called_the_standard_deviation_of_psi(self):
        self.assertNotIn("psi", make.COLUMN_TITLES["metric_std"])
        self.assertIn("time", make.COLUMN_TITLES["metric_std"])

    def test_csv_without_std_offers_only_its_columns(self):
        path = self.root / "old.csv"
        path.write_text("run_index,alpha,drive_frequency_hz,metric\n0,1,5,0.5\n1,2,5,0.7\n")
        rows = make.read_csv_rows(path)
        self.assertEqual(make.numeric_columns(rows), ["run_index", "alpha", "drive_frequency_hz", "metric"])
        with self.assertRaisesRegex(ValueError, "sigma_x_um"):
            make.grid_from_rows(rows, "alpha", "drive_frequency_hz", "sigma_x_um")
        # Blank statistics columns of an older run are not offered either.
        path.write_text("alpha,drive_frequency_hz,metric,sigma_x\n1,5,0.5,\n2,5,0.7,\n")
        self.assertNotIn("sigma_x_um", make.numeric_columns(make.read_csv_rows(path)))
        # A run CSV with sigma_x gains the micrometre column.
        path.write_text("alpha,drive_frequency_hz,metric,sigma_x\n1,5,0.5,100\n")
        self.assertAlmostEqual(make.read_csv_rows(path)[0]["sigma_x_um"], 2.3)

    def test_overlapping_points_need_a_fixed_column(self):
        rows = self.rows(self.walls_off) + self.rows(self.walls_50)
        with self.assertRaises(make.OverlappingPointsError) as caught:
            make.grid_from_rows(rows, "green_wall_height_er", "alpha", "metric")
        self.assertIn("drive_frequency_hz", caught.exception.columns)
        xs, ys, values = make.grid_from_rows(make.select_rows(rows, {"drive_frequency_hz": 2e6}),
                                             "green_wall_height_er", "alpha", "metric")
        np.testing.assert_array_equal(xs, [0.0, 50.0])
        self.assertAlmostEqual(values[2, 1], 30.0 + 2.0 + 0.5)
        with self.assertRaisesRegex(ValueError, "available"):
            make.select_rows(rows, {"drive_frequency_hz": 3e6})

    def test_terminal_questions(self):
        rows = self.rows(self.walls_off) + self.rows(self.walls_50)
        answers = iter(["green_wall_height_er", "", "sigma_x_um", "nonsense", "0 0.8",
                        "metric", "alpha", "20"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            choice = make.ask_plot_settings(rows, "alpha", "drive_frequency_hz", "metric", None, {},
                                            ask=lambda prompt: next(answers))
        x, y, color, limits, where = choice
        self.assertEqual((x, y, color, limits), ("green_wall_height_er", "drive_frequency_hz",
                                                 "sigma_x_um", (0.0, 0.8)))
        self.assertEqual(where, {"alpha": 20.0})
        self.assertIn("metric = IPR", output.getvalue())
        self.assertIn("sigma_x_um = standard deviation of |psi|^2", output.getvalue())
        self.assertIn("inputs:  alpha, drive_frequency_hz", output.getvalue())
        # Only inputs, never results, are offered for fixing.
        self.assertIn("'metric' is not one of: alpha", output.getvalue())

    def test_terminal_questions_fix_a_column_for_overlaps(self):
        rows = self.rows(self.walls_off) + self.rows(self.walls_50)
        answers = iter(["green_wall_height_er", "alpha", "", "", "", "3e6", "1e6"])
        with contextlib.redirect_stdout(io.StringIO()):
            x, y, color, limits, where = make.ask_plot_settings(
                rows, "alpha", "drive_frequency_hz", "metric", None, {}, ask=lambda prompt: next(answers))
        self.assertEqual((x, y, color, limits), ("green_wall_height_er", "alpha", "metric", None))
        self.assertEqual(where, {"drive_frequency_hz": 1e6})

    def test_command_line_combines_datasets_along_wall_height(self):
        arguments = ["make", str(self.walls_off), str(self.walls_50), "--x", "green_wall_height_er",
                     "--y", "alpha", "--color", "metric", "--where", "drive_frequency_hz=1e6",
                     "--vmax", "1", "--no-ask"]
        with patch.object(make, "RESULTS_DIRECTORY", self.root), patch.object(sys, "argv", arguments):
            _, captured = self.captured_plot(make.main)
        self.assertEqual((captured["keywords"]["vmin"], captured["keywords"]["vmax"]), (None, 1.0))
        np.testing.assert_array_equal(captured["arguments"][0][0], [0.0, 50.0])
        combined = list((self.root / "combined_plots").glob("Combined_*.csv"))
        self.assertEqual(len(combined), 1)
        with combined[0].open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["dataset"] for row in rows}, {"walls_off", "walls_50"})
        sources = json.loads(next((self.root / "combined_plots").glob("*_sources.json")).read_text())
        self.assertEqual(sources["where"], {"drive_frequency_hz": 1e6})

    def test_command_line_rejects_bad_where(self):
        with patch.object(sys, "argv", ["make", str(self.walls_off), "--where", "alpha", "--no-ask"]):
            with self.assertRaises(SystemExit):
                make.main()


if __name__ == "__main__":
    unittest.main()
