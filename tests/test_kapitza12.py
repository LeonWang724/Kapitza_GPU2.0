"""12.0: a lattice hold between the drive and the image.

The drive runs N = round(HOLD_TIME_MS x f) whole cycles (the lab's
hw_KPModNCycle); then the lattice stays static (by default where the drive
ended: the loading lattice) until the cloud is imaged at HOLD_TIME_MS +
LATTICE_HOLD_MS. The solver gets drive_iterations and hold_floquet_factor per
point; fused_solver_host_test.cpp checks the solver's stop bit for bit.
"""
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

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_metrics as em  # noqa: E402
import run_phase_diagram_CUDA as runner  # noqa: E402
from create_initial_state_function import create_init_state  # noqa: E402
import cuda_workflow_common as cwc  # noqa: E402
from cuda_workflow_common import (  # noqa: E402
    drive_hold, experiment_drive_phase, phase_dataset_label, read_config,
)
from test_parallel_sweep import FAKE_SOLVER  # noqa: E402

STEP = 0.5                                   # gpe1d.config time_step (solver units)
STEP_S = STEP * em.solver_time_unit_s()


def point_config(alpha: float, frequency: float, total: int, inverted: bool = True) -> dict:
    """The keys drive_hold reads, as create_init_state writes them."""
    phase = experiment_drive_phase(alpha, inverted, "hardware", 10.0, 20.0)
    return {"time_step": repr(STEP), "number_of_iterations": str(total),
            "floquet_omega": repr(2.0 * math.pi * frequency * em.solver_time_unit_s()),
            "floquet_phase": repr(phase - 0.5 * math.pi)}


def grid(**overrides) -> dict:
    values = {"lattice_depth_v0_er": 10.0, "initial_lattice_depth_v0_er": 20.0, "inverted": True,
              "hold_time_ms": 2.0, "lattice_hold_ms": 2.0, "lattice_hold_depth_er": None,
              "drive_whole_cycles": True}
    values.update(overrides)
    return values


class DefaultsTests(unittest.TestCase):
    def test_the_runner_holds_2_ms_where_the_drive_ends(self):
        self.assertEqual(runner.LATTICE_HOLD_MS, 2.0)
        self.assertIsNone(runner.LATTICE_HOLD_DEPTH_ER)
        self.assertIs(runner.DRIVE_WHOLE_CYCLES, True)
        self.assertEqual(runner.HOLD_TIME_MS, 2.0)

    def test_scan_lasts_drive_plus_hold(self):
        base = {"time_step": repr(STEP), "number_of_iterations": "70000"}
        scan = runner.scan_settings(base)
        self.assertEqual(scan["number_of_iterations"], round(4.0e-3 / STEP_S))
        self.assertAlmostEqual(scan["simulated_hold_ms"], 4.0, delta=STEP_S * 1e3)
        self.assertEqual((scan["lattice_hold_ms"], scan["lattice_hold_depth_er"],
                          scan["drive_whole_cycles"]), (2.0, None, True))
        with patch.multiple(runner, LATTICE_HOLD_MS=0):
            off = runner.scan_settings(base)
        self.assertEqual(off["number_of_iterations"], round(2.0e-3 / STEP_S))
        self.assertIsNone(off["lattice_hold_ms"])
        with patch.multiple(runner, LATTICE_HOLD_MS=0.5, LATTICE_HOLD_DEPTH_ER=-20.0):
            self.assertEqual(runner.scan_settings(base)["lattice_hold_depth_er"], -20.0)

    def test_invalid_settings_are_refused(self):
        base = {"time_step": repr(STEP), "number_of_iterations": "70000"}
        for settings in ({"LATTICE_HOLD_MS": -1.0}, {"LATTICE_HOLD_MS": True},
                         {"LATTICE_HOLD_MS": "2"}, {"LATTICE_HOLD_MS": 2.0, "HOLD_TIME_MS": None},
                         {"LATTICE_HOLD_DEPTH_ER": float("nan")}, {"LATTICE_HOLD_DEPTH_ER": "10"},
                         {"DRIVE_WHOLE_CYCLES": 1}):
            with patch.multiple(runner, **settings), self.assertRaises(ValueError, msg=str(settings)):
                runner.scan_settings(base)

    def test_label_names_the_hold(self):
        label = phase_dataset_label(10, 20, 0.0, inverted=True, hold_time_ms=2.0,
                                    drive_phase_convention="hardware", lattice_hold_ms=2.0)
        self.assertTrue(label.endswith("_T2ms_PhaseHw_Hold2ms"), label)
        label = phase_dataset_label(10, 20, 0.0, inverted=True, lattice_hold_ms=1.5,
                                    lattice_hold_depth_er=-20.0)
        self.assertTrue(label.endswith("_Hold1p5ms_HoldDm20ER"), label)
        self.assertNotIn("Hold", phase_dataset_label(10, 20, 0.0, inverted=True))


class DriveHoldTests(unittest.TestCase):
    total = round(4.0e-3 / STEP_S)

    def test_whole_cycles_end_at_the_loading_lattice(self):
        for alpha, frequency in ((24.22222222222222, 866666.6666666667), (4.0, 1377777.7777777778),
                                 (30.0, 2.4e6), (4.0, 1.0e5)):
            hold = drive_hold(grid(), alpha, frequency, point_config(alpha, frequency, self.total))
            cycles = round(2.0e-3 * frequency)
            self.assertEqual(hold["drive_cycles"], cycles)
            self.assertEqual(hold["drive_iterations"], round(cycles / frequency / STEP_S))
            # Within half a step of N / f, and the rest of the 4 ms is the hold.
            self.assertLessEqual(abs(hold["drive_time_s"] - cycles / frequency), 0.5 * STEP_S)
            self.assertAlmostEqual(hold["drive_time_s"] + hold["hold_time_s"], self.total * STEP_S)
            # The hardware drive starts (and so, after whole cycles, ends) at -20 E_R.
            self.assertAlmostEqual(hold["hold_floquet_factor"], -3.0 / alpha, places=9)
            self.assertAlmostEqual(hold["hold_lattice_er"], -20.0, places=7)

    def test_without_whole_cycles_the_drive_lasts_exactly_the_drive_time(self):
        alpha, frequency = 24.22222222222222, 866666.6666666667
        config = point_config(alpha, frequency, self.total)
        hold = drive_hold(grid(drive_whole_cycles=False), alpha, frequency, config)
        self.assertIsNone(hold["drive_cycles"])
        self.assertEqual(hold["drive_iterations"], round(2.0e-3 / STEP_S))
        expected = math.cos(float(config["floquet_omega"]) * 2.0e-3 / em.solver_time_unit_s()
                            + float(config["floquet_phase"]))
        self.assertAlmostEqual(hold["hold_floquet_factor"], expected, places=10)

    def test_a_set_hold_depth(self):
        alpha, frequency = 12.0, 1.0e6
        config = point_config(alpha, frequency, self.total)
        for depth, factor in ((10.0, 0.0), (0.0, -1.0 / 12.0), (-20.0, -3.0 / 12.0), (25.0, 1.5 / 12.0)):
            hold = drive_hold(grid(lattice_hold_depth_er=depth), alpha, frequency, config)
            self.assertAlmostEqual(hold["hold_floquet_factor"], factor, places=12)
            self.assertAlmostEqual(hold["hold_lattice_er"], depth, places=9)
        self.assertEqual(drive_hold(grid(lattice_hold_depth_er=10.0), 0.0, frequency,
                                    config)["hold_floquet_factor"], 0.0)
        with self.assertRaises(ValueError):
            drive_hold(grid(lattice_hold_depth_er=0.0), 0.0, frequency, config)

    def test_no_hold_means_no_stop(self):
        config = point_config(4.0, 1.0e6, self.total)
        self.assertIsNone(drive_hold(grid(lattice_hold_ms=None), 4.0, 1.0e6, config))
        self.assertIsNone(drive_hold(grid(lattice_hold_ms=0), 4.0, 1.0e6, config))
        self.assertIsNone(drive_hold({"lattice_depth_v0_er": 10.0}, 4.0, 1.0e6, config))

    def test_the_held_potential_is_the_set_lattice(self):
        """vstatic + factor * vflo from the real generator is -D cos^2(kL x) + const."""
        alpha, frequency = 12.0, 1.0e6
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "run.config"
            config_path.write_text((CORE / "gpe1d.config").read_text())
            phase = experiment_drive_phase(alpha, True, "hardware", 10.0, 20.0)
            with contextlib.redirect_stdout(io.StringIO()):
                create_init_state(10.0, alpha, frequency, 0.0, 20.0, output_directory=directory,
                                  config_path=config_path, drive_phase=phase,
                                  grid_points=16384, envelope_sigma_m=30e-6)
            config = dict(read_config(config_path), number_of_iterations=str(self.total))

            def load(name):
                with tb.open_file(str(directory / name)) as handle:
                    return handle.root.REAL[:]
            static, drive = load("vstatic.h5"), load("vflo.h5")
            x = (np.arange(16384) - 8192) * 23e-9
            lattice = np.cos(2 * math.pi / 1064e-9 * x) ** 2
            unit = (1.0545718e-34 * 2 * math.pi / 1064e-9) ** 2 / (2 * 7.016004 * 1.66053907e-27)
            from create_initial_state_function import hbar, mli
            energy_scale = hbar ** 2 / (mli * 23e-9 ** 2)
            for depth in (None, 10.0, 0.0, -20.0):
                hold = drive_hold(grid(lattice_hold_depth_er=depth), alpha, frequency, config)
                held = static + hold["hold_floquet_factor"] * drive
                slope = np.polyfit(lattice, held, 1)[0] * energy_scale / unit
                expected = -20.0 if depth is None else depth
                self.assertAlmostEqual(-slope, expected, delta=2e-3 * max(1.0, abs(expected)),
                                       msg=f"depth {depth}")
                self.assertLess(np.ptp(held - np.polyval(np.polyfit(lattice, held, 1), lattice)),
                                1e-9 * max(1.0, np.ptp(held)))


class RunnerTests(unittest.TestCase):
    def test_plan_reports_the_drive_and_the_hold(self):
        with (patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        text = output.getvalue()
        self.assertIn("Drive: 2 ms, N = round(drive time x f) whole cycles, ending where it "
                      "started. Then a 2 ms lattice hold, static at where the drive ends; "
                      "imaged at the end.", text)
        self.assertIn(f"Run length: {round(4.0e-3 / STEP_S):,} steps", text)
        self.assertIn("classical ensemble columns stay empty with a lattice hold", text)

    def test_an_old_build_is_refused(self):
        old_build = {"compact_phase_metric_version": 1, "phase_statistics_version": 1,
                     "fused_solver_version": 1, "drive_phase_version": 1,
                     "extended_metrics_version": 2, "density_smoothing_version": 1,
                     "camera_profile_version": 1, "max_batch_systems": 64}
        with (patch.object(runner, "locate_executable", return_value=Path("solver")),
              patch.object(runner, "query_json", return_value=old_build),
              patch.object(sys, "argv", ["runner"]), contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "12.0 lattice hold"):
                runner.main()


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """A 12.0 scan in two tabs: every point's solver config carries its own stop."""

    def test_scan_gives_every_point_its_drive_and_hold(self):
        with tempfile.TemporaryDirectory(prefix="twelve scan ") as temporary:
            root = Path(temporary)
            core = root / "core"
            core.mkdir()
            events = root / "events.jsonl"
            (core / "gpe1d.config").write_text(
                "points_x=65536\nnumber_of_iterations=3000\nsave_every_nth_iteration=100\n"
                "show_stats_every_nth_iteration=100\ntime_step=0.5\n"
                "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
                f"test_events={events}\n")
            solver = root / "stand in solver.exe"
            solver.write_text(FAKE_SOLVER)
            solver.chmod(0o755)
            results = root / "results"
            alphas, frequencies = np.array([4.0, 24.0]), np.array([1.0e6, 2.0e6])
            with (patch.object(runner, "SCRIPT_DIRECTORY", core),
                  patch.multiple(runner, INVERTED=True, NUMBER_OF_TABS=2, GRID_POINTS=65536,
                                 QUASIMOMENTUM_VALUES_KL=[0.0, 0.1], QUASIMOMENTUM_SPREAD_KL=0.05,
                                 LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                                 ALPHA_VALUES=alphas, DRIVE_FREQUENCY_HZ_VALUES=frequencies),
                  patch.dict(os.environ, {"KAPITZA_CORE": str(CORE)}),
                  patch.object(sys, "argv", ["runner", "--executable", str(solver),
                                            "--results", str(results), "--headless"]),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 0)
            manifest = json.loads((results / "run_manifest.json").read_text())
            records = [json.loads(line) for line in
                       (results / "point_results.jsonl").read_text().splitlines() if line.strip()]
            with open(results / "metrics.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            starts = [json.loads(line) for line in events.read_text().splitlines()
                      if json.loads(line)["kind"] == "start"]
        grid_values = manifest["parameter_grid"]
        total = round(4.0e-3 / STEP_S)
        self.assertEqual(grid_values["number_of_iterations"], total)
        self.assertEqual((grid_values["lattice_hold_ms"], grid_values["lattice_hold_depth_er"],
                          grid_values["drive_whole_cycles"]), (2.0, None, True))
        self.assertTrue(manifest["parameter_label"].endswith("_PhaseHw_Hold2ms"))
        self.assertEqual(len(records), 8)
        self.assertEqual(len(starts), 8)
        for record in records:
            frequency, alpha = record["drive_frequency_hz"], record["alpha"]
            cycles = round(2.0e-3 * frequency)
            self.assertEqual(record["drive_hold"]["drive_cycles"], cycles)
            self.assertEqual(record["drive_hold"]["drive_iterations"],
                             round(cycles / frequency / STEP_S))
            self.assertAlmostEqual(record["drive_hold"]["hold_lattice_er"], -20.0, places=7)
        # What the solver received: the whole run plus each point's own stop.
        for event in starts:
            self.assertEqual(int(event["number_of_iterations"]), total)
            self.assertGreater(int(event["drive_iterations"]), 0)
            self.assertLess(int(event["drive_iterations"]), total)
            self.assertTrue(math.isfinite(float(event["hold_floquet_factor"])))
        self.assertEqual({int(event["drive_iterations"]) for event in starts},
                         {round(round(2.0e-3 * f) / f / STEP_S) for f in frequencies})
        # Model columns stay empty with a hold; the GPU columns are filled.
        self.assertEqual(len(rows), 8)
        for row in rows:
            for column in ("q0_pendulum_cos", "cl_pendulum_cos"):
                self.assertIn(column, row)
                self.assertIn(row[column], ("", "None"), column)
            self.assertNotIn(row["pendulum_cos"], ("", "None"))


if __name__ == "__main__":
    unittest.main()


# The input gpe1d_cuda.exe could not open on the laptop (2026-11-17): the scan's
# scratch folder sat inside the long results folder, and the 12.0 label's
# "_Hold2ms" took the path from 259 characters (11.0, just fitting) to 267.
LAPTOP_INPUT = (r"C:\Users\leonw\Desktop\Kapitza_GPU12.0\phase_diagram\simulation_core\results_cuda"
                r"\LatticeDepth_10ER_InitialDepth_20ER_Inverted_Q0to0p2x5kL_Smooth4um_Box262144_W35um"
                r"_T2ms_PhaseHw_Hold2ms_cuda_20261117T062535Z\workers\tab_001\.work_zry1qx7r"
                r"\point_00\in\lattice_gauss.h5")


def write_config(directory: Path, **values) -> Path:
    path = directory / "run.config"
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return path


class SolverPathTests(unittest.TestCase):
    """The solver's files must stay under Windows' 260-character path limit."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paths ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_the_laptop_input_is_refused_before_the_solver_starts(self):
        self.assertEqual(len(LAPTOP_INPUT), 267)
        config = write_config(self.directory, initial_state_file=LAPTOP_INPUT, save_psi="false")
        with self.assertRaisesRegex(RuntimeError, r"267 characters.*limit 259.*initial_state_file"):
            cwc.check_solver_paths([config], self.directory, limit=259)

    def test_short_paths_pass(self):
        files = {key: str(self.directory / "in" / f"{key}.h5") for key in cwc.SOLVER_PATH_KEYS}
        config = write_config(self.directory, save_psi="false", **files)
        cwc.check_solver_paths([config], self.directory, limit=259)

    def test_each_kind_of_path_counts(self):
        cwd = self.directory / ("c" * 40)
        cases = {
            "relative input, measured from the solver's folder":
                ({"potential_file": "p" * (300 - len(str(cwd)))}, "potential_file"),
            "snapshots inside output_folder":
                ({"output_folder": str(self.directory / ("o" * (255 - len(str(self.directory))))),
                  "save_psi": "true"}, "snapshot"),
            "the config itself":
                ({}, "config"),
        }
        for name, (values, key) in cases.items():
            with self.subTest(name):
                folder = self.directory / ("x" * (150 if key == "config" else 1))
                folder.mkdir(exist_ok=True)
                config = write_config(folder, **values)
                limit = len(str(config)) - 1 if key == "config" else 259
                with self.assertRaisesRegex(RuntimeError, key):
                    cwc.check_solver_paths([config], cwd, limit=limit)
        # Without saved snapshots the output folder is not opened.
        config = write_config(self.directory, save_psi="false",
                              output_folder=str(self.directory / ("o" * 300)))
        cwc.check_solver_paths([config], cwd, limit=259)

    def test_only_windows_is_limited_by_default(self):
        self.assertEqual(cwc.SOLVER_PATH_LIMIT, 259 if os.name == "nt" else None)
        config = write_config(self.directory, initial_state_file=LAPTOP_INPUT)
        with patch.object(cwc, "SOLVER_PATH_LIMIT", None):
            cwc.check_solver_paths([config], self.directory)
        with patch.object(cwc, "SOLVER_PATH_LIMIT", 259), self.assertRaises(RuntimeError):
            cwc.check_solver_paths([config], self.directory)

    def test_the_solver_reason_is_read_from_its_log(self):
        log = self.directory / "solver.log"
        self.assertIsNone(cwc.solver_failure_reason(log))
        log.write_text("CUDA device: 0\n  minor: Unable to open file\n"
                       "gpe1d_cuda failed: Unable to open HDF5 input: in\\lattice_gauss.h5\n")
        self.assertEqual(cwc.solver_failure_reason(log),
                         "gpe1d_cuda failed: Unable to open HDF5 input: in\\lattice_gauss.h5")
        log.write_text("Systems in this GPU batch: 32\n")
        self.assertIsNone(cwc.solver_failure_reason(log))


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class LongResultsFolderTests(unittest.TestCase):
    """A scan in a results folder as long as the laptop's runs, under the Windows limit."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="long scan ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        self.events = self.root / "events.jsonl"
        (self.core / "gpe1d.config").write_text(
            "points_x=65536\nnumber_of_iterations=3000\nsave_every_nth_iteration=100\n"
            "show_stats_every_nth_iteration=100\ntime_step=0.5\n"
            "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
            f"test_events={self.events}\n")
        self.solver = self.root / "stand in solver.exe"
        self.solver.write_text(FAKE_SOLVER)
        self.solver.chmod(0o755)
        # As long as the laptop's results folder plus workers\tab_001 (209).
        parent = self.root / ("r" * max(1, 200 - len(str(self.root)) - len("/results")))
        self.results = parent / "results"
        self.assertGreaterEqual(len(str(self.results)), 200)

    def scan(self, tabs: int) -> int:
        with (patch.object(runner, "SCRIPT_DIRECTORY", self.core),
              patch.multiple(runner, INVERTED=True, NUMBER_OF_TABS=tabs, GRID_POINTS=65536,
                             QUASIMOMENTUM_VALUES_KL=[0.0, 0.1], QUASIMOMENTUM_SPREAD_KL=0.05,
                             LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                             ALPHA_VALUES=np.array([4.0]),
                             DRIVE_FREQUENCY_HZ_VALUES=np.array([1.0e6, 2.0e6])),
              patch.object(cwc, "SOLVER_PATH_LIMIT", 259),
              patch.dict(os.environ, {"KAPITZA_CORE": str(CORE)}),
              patch.object(sys, "argv", ["runner", "--executable", str(self.solver),
                                        "--results", str(self.results), "--headless"]),
              contextlib.redirect_stdout(io.StringIO())):
            return runner.main()

    def test_the_solver_files_stay_short_and_are_removed(self):
        scratch = Path(tempfile.gettempdir())
        before = set(scratch.glob("kapitza_*"))
        self.assertEqual(self.scan(tabs=1), 0)
        starts = [json.loads(line) for line in self.events.read_text().splitlines()
                  if json.loads(line)["kind"] == "start"]
        self.assertEqual(len(starts), 4)
        for event in starts:
            for path in (event["config"], event["input"]):
                self.assertLessEqual(len(path), 259, path)
                self.assertNotIn(str(self.results), path)
                self.assertTrue(any(part.startswith("kapitza_") for part in Path(path).parts), path)
        self.assertEqual(set(scratch.glob("kapitza_*")) - before, set())
        with open(self.results / "metrics.csv", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_a_failed_tab_says_why(self):
        failing = FAKE_SOLVER.replace(
            'print("Intentional solver failure", flush=True)',
            'print("gpe1d_cuda failed: Unable to open HDF5 input: test input", flush=True)')
        self.assertNotEqual(failing, FAKE_SOLVER)
        self.solver.write_text(failing)
        with (self.core / "gpe1d.config").open("a") as stream:
            stream.write("test_fail=true\n")
        with self.assertRaisesRegex(
                RuntimeError, r"Tab 1 failed: CUDA (point|batch of points) \S+ failed with exit code 7\. "
                              r"gpe1d_cuda failed: Unable to open HDF5 input: test input; see "):
            self.scan(tabs=2)
