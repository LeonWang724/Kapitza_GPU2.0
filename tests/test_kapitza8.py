"""8.0: the simulated box, the starting cloud width, the hold time and the
initial-state references for normalizing."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import shutil
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_metrics as em  # noqa: E402
import run_phase_diagram_CUDA as runner  # noqa: E402
from create_initial_state_function import (  # noqa: E402
    ABSORBER_FRACTION, GRID_SPACING_M, create_init_state,
)
from cuda_workflow_common import phase_dataset_label, read_config  # noqa: E402
from make_phase_diagram_CUDA import DATASET_COLUMNS, dataset_columns  # noqa: E402
from test_parallel_sweep import FAKE_SOLVER  # noqa: E402


def generate(directory: Path, **keywords):
    directory.mkdir()
    config = directory / "run.config"
    shutil.copy2(CORE / "gpe1d.config", config)
    create_init_state(10.0, 12.0, 1.0e6, 0.0, 20.0, output_directory=directory / "in",
                      config_path=config, drive_phase=0.0, **keywords)
    arrays = {}
    for name in ("lattice_gauss", "vstatic", "vflo"):
        with tb.open_file(directory / "in" / f"{name}.h5") as handle:
            arrays[name] = handle.root.REAL[:] + 1j * handle.root.IMAGINARY[:]
    return arrays, read_config(config)


class GeneratorTests(unittest.TestCase):
    def test_default_is_the_7_0_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default, config = generate(root / "default")
            explicit, _ = generate(root / "explicit", grid_points=65536, envelope_sigma_m=30e-6)
        for name in default:
            np.testing.assert_array_equal(default[name], explicit[name])
        self.assertEqual(int(float(config["points_x"])), 65536)

    def test_box_and_cloud_follow_the_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays, config = generate(Path(temporary) / "big", grid_points=262144,
                                      envelope_sigma_m=40e-6)
        psi, vstatic = arrays["lattice_gauss"], arrays["vstatic"]
        self.assertEqual(psi.size, 262144)
        self.assertEqual(int(float(config["points_x"])), 262144)
        x = (np.arange(psi.size) - psi.size // 2) * GRID_SPACING_M
        density = np.abs(psi) ** 2
        mean = np.sum(x * density) / density.sum()
        width = math.sqrt(np.sum((x - mean) ** 2 * density) / density.sum())
        self.assertAlmostEqual(width * 1e6, 40.0, delta=0.4)     # the envelope's 40 um
        # Absorbing walls only beyond +-2.11 mm: the outer 15% of the 6 mm box.
        absorbing = np.abs(vstatic.imag) > 0
        inner = np.abs(x).min(where=absorbing, initial=np.inf)
        expected = x[-1] - ABSORBER_FRACTION * (x[-1] - x[0])
        self.assertAlmostEqual(inner, expected, delta=2 * GRID_SPACING_M)
        self.assertGreater(inner, 2.1e-3)

    def test_bad_box_or_cloud_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, keywords in enumerate(({"grid_points": 1001}, {"grid_points": 512},
                                              {"envelope_sigma_m": 0.0},
                                              {"envelope_sigma_m": 200e-6})):
                with self.assertRaises(ValueError, msg=str(keywords)):
                    generate(root / f"bad{index}", **keywords)


class SettingsTests(unittest.TestCase):
    def test_hold_time_becomes_steps(self):
        base = {"time_step": "0.5", "number_of_iterations": "70000"}
        with patch.multiple(runner, GRID_POINTS=262144, INITIAL_CLOUD_WIDTH_UM=30.0,
                            HOLD_TIME_MS=2.0):
            scan = runner.scan_settings(base)
        step_ms = 0.5 * em.solver_time_unit_s() * 1e3
        self.assertEqual(scan["number_of_iterations"], round(2.0 / step_ms))
        self.assertAlmostEqual(scan["simulated_hold_ms"], 2.0, delta=step_ms)
        self.assertAlmostEqual(scan["absorber_starts_um"], 2110.6, delta=1.0)
        with patch.multiple(runner, HOLD_TIME_MS=None):
            self.assertEqual(runner.scan_settings(base)["number_of_iterations"], 70000)
        for settings in ({"GRID_POINTS": 1001}, {"GRID_POINTS": True}, {"HOLD_TIME_MS": 0.0},
                         {"INITIAL_CLOUD_WIDTH_UM": -1.0},
                         {"GRID_POINTS": 4096, "INITIAL_CLOUD_WIDTH_UM": 30.0}):
            with patch.multiple(runner, **settings), self.assertRaises(ValueError, msg=str(settings)):
                runner.scan_settings(base)

    def test_plan_reports_box_cloud_and_hold(self):
        with (patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              patch.multiple(runner, INITIAL_CLOUD_WIDTH_UM=30.0),     # the 8.0 default
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        text = output.getvalue()
        self.assertIn("Box: 262,144 grid points, +/-3.01 mm, absorbing walls from +/-2.11 mm; "
                      "starting cloud 30 um.", text)
        self.assertIn("Hold time: 2 ms = ", text)

    def test_central_window_is_two_cloud_widths(self):
        self.assertAlmostEqual(em.central_halfwidth_m(), 60e-6)
        self.assertAlmostEqual(em.central_halfwidth_m({"initial_cloud_width_um": 45.0}), 90e-6)
        grid = {"initial_lattice_depth_v0_er": 20.0, "inverted": False, "initial_cloud_width_um": 45.0}
        settings = em.solver_settings(grid, 10.0)
        self.assertAlmostEqual(float(settings["extended_metrics_central_halfwidth"]),
                               90e-6 / GRID_SPACING_M)
        self.assertAlmostEqual(em.contract_settings(grid)["central_halfwidth_um"], 90.0)

    def test_labels_and_columns_record_the_settings(self):
        label = phase_dataset_label(10, 20, 0.0, inverted=True, quasimomentum=0.1, smoothing_um=4.0,
                                    grid_points=262144, cloud_width_um=40.0, hold_time_ms=2.0)
        self.assertTrue(label.endswith("_Q0p1kL_Smooth4um_Box262144_W40um_T2ms"), label)
        self.assertEqual(phase_dataset_label(10, 20, 0.0, inverted=True, grid_points=65536,
                                             cloud_width_um=30.0),
                         "LatticeDepth_10ER_InitialDepth_20ER_Inverted")
        for column in ("grid_points", "initial_cloud_width_um", "hold_time_ms"):
            self.assertIn(column, DATASET_COLUMNS)
        old = dataset_columns({"lattice_depth_v0_er": 10.0, "inverted": True})
        self.assertEqual((old["grid_points"], old["initial_cloud_width_um"], old["hold_time_ms"]),
                         (65536, 30.0, None))


class NormalizingTests(unittest.TestCase):
    def test_summary_keeps_the_starting_camera_values(self):
        names = em.SERIES_COLUMNS
        rows = []
        for iteration in range(0, 3000, 100):
            grow = 1.0 + iteration / 3000.0
            sums = dict.fromkeys(names, 1.0)
            sums.update(iteration=iteration, mass=2.0, x1_sum=0.0, x2_sum=2.0 * (1300.0 * grow) ** 2,
                        pixel_x1_sum=0.0, pixel_x2_sum=2.0 * (1301.0 * grow) ** 2,
                        pixel_squared_sum=4.0 * 0.02 / grow, x3_sum=0.0,
                        x4_sum=2.0 * 3 * (1300.0 * grow) ** 4, momentum_total=1.0)
            rows.append([sums[name] for name in names])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "extended.csv"
            path.write_text(";".join(names) + "\n"
                            + "".join(";".join(repr(v) for v in row) + "\n" for row in rows))
            config = {"extended_metrics_lattice_wavevector": repr(em.solver_wavevector()),
                      "step_x": "1", "time_step": "0.5", "number_of_iterations": "3000",
                      "save_every_nth_iteration": "100", "points_x": "65536", "floquet_omega": "0.1"}
            summary = em.gpu_summary(path, config, {"final_snapshot_count": 30,
                                                    "cut_points_each_edge": 100})
        self.assertAlmostEqual(summary["sigma_x_4um_um_initial"], 1301.0 * 0.023, places=9)
        self.assertAlmostEqual(summary["coarse_ipr_initial"], 0.02, places=12)
        self.assertGreater(summary["sigma_x_4um_um"], summary["sigma_x_4um_um_initial"])


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """A compact 8.0 scan with the default settings and a stand-in solver."""

    def test_scan_uses_one_scan_config_for_box_and_hold(self):
        with tempfile.TemporaryDirectory(prefix="eight scan ") as temporary:
            root = Path(temporary)
            core = root / "core"
            core.mkdir()
            (core / "gpe1d.config").write_text(
                "points_x=65536\nnumber_of_iterations=70000\nsave_every_nth_iteration=100\n"
                "show_stats_every_nth_iteration=100\ntime_step=0.5\n"
                "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
                f"test_events={root / 'events.jsonl'}\n")
            solver = root / "stand in solver.exe"
            solver.write_text(FAKE_SOLVER)
            solver.chmod(0o755)
            results = root / "results"
            with (patch.object(runner, "SCRIPT_DIRECTORY", core),
                  patch.multiple(runner, INVERTED=True, NUMBER_OF_TABS=1, INITIAL_CLOUD_WIDTH_UM=40.0,
                                 QUASIMOMENTUM_VALUES_KL=None,
                                 LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                                 ALPHA_VALUES=np.array([12.0]),
                                 DRIVE_FREQUENCY_HZ_VALUES=np.array([1.0e6])),
                  patch.dict(os.environ, {"KAPITZA_CORE": str(CORE)}),
                  patch.object(sys, "argv", ["runner", "--executable", str(solver),
                                            "--results", str(results)]),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 0)
            manifest = json.loads((results / "run_manifest.json").read_text())
            scan_config = read_config(results / "gpe1d_scan.config")
            provenance = read_config(results / "provenance" / "gpe1d.config")
            with open(results / "metrics.csv", newline="") as handle:
                row = next(csv.DictReader(handle))
        steps = round(2.0e-3 / (0.5 * em.solver_time_unit_s()))
        self.assertEqual(manifest["status"], "completed")
        grid = manifest["parameter_grid"]
        self.assertEqual((grid["grid_points"], grid["initial_cloud_width_um"], grid["hold_time_ms"],
                          grid["number_of_iterations"]), (262144, 40.0, 2.0, steps))
        for config in (scan_config, provenance, manifest["simulation_config"]):
            self.assertEqual(int(float(config["points_x"])), 262144)
            self.assertEqual(int(float(config["number_of_iterations"])), steps)
        self.assertAlmostEqual(manifest["analysis_contract"]["extended_metrics"]["central_halfwidth_um"],
                               80.0)
        self.assertTrue(manifest["parameter_label"].endswith("_Box262144_W40um_T2ms"))
        # The stand-in returns the starting state's sums: the 40 um cloud.
        self.assertAlmostEqual(float(row["sigma_x_4um_um_initial"]), 40.0, delta=1.0)
        self.assertGreater(float(row["coarse_ipr_initial"]), 0.0)


if __name__ == "__main__":
    unittest.main()
