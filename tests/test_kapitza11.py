"""11.0: the hardware drive phase, the measured 35 um start, and measuring the
cloud as the lab's camera analysis does (camera profile + lab_measurement.py)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_metrics as em  # noqa: E402
import lab_measurement as lm  # noqa: E402
import make_phase_diagram_CUDA as make  # noqa: E402
import quasimomentum_spread as qs  # noqa: E402
import run_phase_diagram_CUDA as runner  # noqa: E402
from create_initial_state_function import create_init_state, loading_lattice_coefficient  # noqa: E402
from cuda_workflow_common import (  # noqa: E402
    experiment_drive_phase, phase_dataset_label, point_drive_phase, read_config,
)
from test_parallel_sweep import FAKE_SOLVER  # noqa: E402


class DrivePhaseTests(unittest.TestCase):
    def test_hardware_phase_starts_at_the_loading_lattice(self):
        for alpha in (3.0, 4.0, 24.2, 30.0):
            phase = experiment_drive_phase(alpha, True, "hardware", 10.0, 20.0)
            self.assertAlmostEqual(phase, math.asin(-3.0 / alpha))
            self.assertAlmostEqual(10.0 * (1 + alpha * math.sin(phase)), -20.0)
            normal = experiment_drive_phase(alpha, False, "hardware", 10.0, 20.0)
            self.assertAlmostEqual(10.0 * (1 + alpha * math.sin(normal)), 20.0)
        self.assertEqual(experiment_drive_phase(4.0, True), math.asin(-0.5))   # theory default
        for bad in ((2.9, True, "hardware", 10.0, 20.0), (4.0, True, "other", 10.0, 20.0),
                    (4.0, True, "hardware", None, 20.0)):
            with self.assertRaises(ValueError, msg=str(bad)):
                experiment_drive_phase(*bad)
        grid = {"inverted": True, "drive_phase_convention": "hardware",
                "lattice_depth_v0_er": 10.0, "initial_lattice_depth_v0_er": 20.0}
        self.assertAlmostEqual(point_drive_phase(grid, 4.0), math.asin(-0.75))
        self.assertAlmostEqual(point_drive_phase(dict(grid, drive_phase_convention="theory"), 4.0),
                               math.asin(-0.5))

    def test_loading_lattice_is_the_loading_depth_in_both_conventions(self):
        for alpha in (4.0, 24.2):
            theory = math.asin(-2.0 / alpha)
            self.assertEqual(loading_lattice_coefficient(20.0, alpha, theory),
                             -20.0 * (1.0 + alpha * math.sin(theory)))       # bit for bit
            self.assertEqual(loading_lattice_coefficient(20.0, alpha, math.asin(-3.0 / alpha)), 20.0)
            self.assertEqual(loading_lattice_coefficient(20.0, alpha, math.asin(1.0 / alpha)), -20.0)
        array = loading_lattice_coefficient(20.0, np.array([4.0, 8.0]),
                                            np.arcsin(-3.0 / np.array([4.0, 8.0])))
        np.testing.assert_array_equal(array, [20.0, 20.0])
        grid = {"inverted": True, "drive_phase_convention": "hardware",
                "lattice_depth_v0_er": 10.0, "initial_lattice_depth_v0_er": 20.0}
        self.assertEqual(em.loading_coefficient(grid, 4.0), 20.0)

    def test_generator_loads_the_same_state_and_writes_the_hardware_phase(self):
        states, phases = {}, {}
        with tempfile.TemporaryDirectory() as temporary:
            for name, phase in (("theory", math.asin(-2.0 / 4.0)), ("hardware", math.asin(-0.75))):
                root = Path(temporary) / name
                root.mkdir()
                config = root / "run.config"
                config.write_text("points_x=65536\n")
                create_init_state(10.0, 4.0, 1.0e6, 0.0, 20.0, output_directory=root / "in",
                                  config_path=config, drive_phase=phase)
                with tb.open_file(str(root / "in" / "lattice_gauss.h5")) as handle:
                    states[name] = handle.root.REAL[:] + 1j * handle.root.IMAGINARY[:]
                phases[name] = float(read_config(config)["floquet_phase"])
        np.testing.assert_allclose(states["hardware"], states["theory"], rtol=0, atol=1e-12)
        self.assertAlmostEqual(phases["hardware"], math.asin(-0.75) - 0.5 * math.pi)


class RunnerTests(unittest.TestCase):
    def plan(self, **settings):
        with (patch.multiple(runner, NUMBER_OF_TABS=1, **settings),
              patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        return output.getvalue()

    def test_defaults(self):
        self.assertEqual(runner.INITIAL_CLOUD_WIDTH_UM, 35.0)
        self.assertEqual(runner.DRIVE_PHASE_CONVENTION, "hardware")
        self.assertTrue(runner.LabMeasurement)
        text = self.plan(INVERTED=True, LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                         ALPHA_VALUES=np.linspace(4.0, 30.0, 10))
        self.assertIn("the drive starts at the loading lattice, -20 E_R (hardware)", text)
        self.assertIn("starting cloud 35 um", text)
        self.assertIn("StdDev2 window rows 120-780 (661 px)", text)
        label = phase_dataset_label(10, 20, 0.0, inverted=True, grid_points=262144,
                                    cloud_width_um=35.0, hold_time_ms=2.0,
                                    drive_phase_convention="hardware")
        self.assertTrue(label.endswith("_Box262144_W35um_T2ms_PhaseHw"), label)

    def test_alpha_below_the_hardware_limit_is_refused(self):
        with (patch.multiple(runner, INVERTED=True, LATTICE_DEPTH_V0_ER=10.0,
                             INITIAL_LATTICE_DEPTH_V0_ER=20.0, ALPHA_VALUES=np.array([2.5, 4.0])),
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_lab_measurement_needs_compact_storage_and_an_11_build(self):
        with (patch.object(sys, "argv", ["runner", "--plan", "--storage", "full"]),
              patch.multiple(runner, QUASIMOMENTUM_VALUES_KL=None, LATTICE_HOLD_MS=0),
              contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(SystemExit):
                runner.main()
        old_build = {"compact_phase_metric_version": 1, "phase_statistics_version": 1,
                     "fused_solver_version": 1, "drive_phase_version": 1,
                     "extended_metrics_version": 2, "density_smoothing_version": 1,
                     "max_batch_systems": 64}
        with (patch.object(runner, "locate_executable", return_value=Path("solver")),
              patch.object(runner, "query_json", return_value=old_build),
              patch.multiple(runner, LATTICE_HOLD_MS=0),
              patch.object(sys, "argv", ["runner"]), contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "11.0 camera profile"):
                runner.main()


def port_compute_st_dev(profile, px):
    """A line-by-line port of kpAnalysis.m's computeStDev for one profile."""
    one = np.asarray(profile, dtype=float)
    one = one / one.sum()
    pos = px * np.arange(1, one.size + 1)
    mean = np.sum(pos * one) / np.sum(one)
    return math.sqrt(abs(np.sum((pos - mean) ** 2 * one) / np.sum(one)))


def port_compute_st_dev_with_bg_sub(profile, px, noise_floor):
    """A line-by-line port of kpAnalysis.m's computeStDevWithBgSub."""
    one = np.asarray(profile, dtype=float).copy()
    n = one.size
    edge = max(1, round(n * 0.10))
    mask = np.zeros(n, dtype=bool)
    mask[list(range(edge)) + list(range(n - edge, n))] = True
    one = one - np.sum(one * mask) / np.sum(mask)
    one[one < noise_floor * one.max()] = 0
    mass = one.sum()
    pos = px * np.arange(1, n + 1)
    mean = np.sum(pos * one) / mass
    return math.sqrt(np.sum((pos - mean) ** 2 * one) / mass)


class LabMeasurementTests(unittest.TestCase):
    def test_camera_profile_matches_the_cpp_binning(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No C++ compiler")
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "camera_profile_host_test"
            subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-I", str(ROOT / "src"),
                            str(ROOT / "tests" / "camera_profile_host_test.cpp"), "-o", str(executable)],
                           check=True, capture_output=True, text=True)
            output = Path(temporary) / "profile.json"
            completed = subprocess.run([str(executable), str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            profile = lm.read_profile(output)
        density = ((np.arange(20000) * 37) % 101) / 101.0
        first, mass = lm.camera_profile(density, 1.0, 169.34455556399354)
        self.assertEqual(profile["first_pixel"], first)
        np.testing.assert_array_equal(profile["mass"], mass)

    def test_formulas_are_kpanalysis(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            profile = np.exp(-0.5 * ((np.arange(661) - 330) / 40.0) ** 2) + rng.normal(0, 0.01, 661)
            self.assertAlmostEqual(lm.lab_stddev(profile, 4.0)[0], port_compute_st_dev(profile, 4.0),
                                   places=9)
            for floor in (0.0, 0.02):
                self.assertAlmostEqual(lm.lab_stddev2(profile, 4.0, 0.10, floor)[0],
                                       port_compute_st_dev_with_bg_sub(profile, 4.0, floor), places=9)

    def test_window_mapping_and_fractions(self):
        camera = lm.validate_camera({"noise_per_pixel_over_signal": 0.0, "noise_draws": 1})
        mass = np.zeros(1549)
        centre = 774                                     # pixel k = 0
        mass[centre - 3:centre + 4] = 1.0                # 7 px around the cloud centre
        mass[centre + 500] = 1.0                         # beyond the crop (row 956), on the sensor
        mass[centre + 700] = 1.0                         # beyond the sensor (row 1156)
        profile = {"first_pixel": -centre, "pixel_width": camera["pixel_width_solver"],
                   "mass": mass.tolist()}
        result = lm.measure(profile, camera)
        self.assertAlmostEqual(result["lab_window_fraction"], 7 / 9)
        self.assertAlmostEqual(result["camera_fov_fraction"], 8 / 9)
        self.assertAlmostEqual(result["camera_width_um"], camera["pixel_um"] * 2.0, places=9)
        self.assertAlmostEqual(result["lab_stddev2_um"], result["lab_stddev2_no_noise_um"])
        window = lm._rows(profile, camera, (120, 780))
        self.assertEqual(int(np.argmax(window[:400])), 456 - 120 - 3)

    def test_noise_inflates_a_narrow_cloud_like_the_lab(self):
        camera = lm.validate_camera(None)
        k = np.arange(-774, 775)
        sigma_px = 35.0 / camera["pixel_um"]
        mass = np.exp(-0.5 * (k / sigma_px) ** 2)
        profile = {"first_pixel": -774, "pixel_width": camera["pixel_width_solver"],
                   "mass": (mass / mass.sum()).tolist()}
        result = lm.measure(profile, camera, seed=1)
        self.assertAlmostEqual(result["camera_width_um"], 35.0, delta=0.5)
        # Rectified noise: ~0.4 x noise per pixel over 661 px, spread over the window.
        fake = 661 * camera["noise_per_pixel_over_signal"] / math.sqrt(2 * math.pi)
        window_um = 661 * camera["lab_um_per_pixel"]
        expected = math.sqrt(((35.0 * 4.0 / camera["pixel_um"]) ** 2
                              + fake * window_um ** 2 / 12) / (1 + fake))
        self.assertAlmostEqual(result["lab_stddev2_um"], expected, delta=0.12 * expected)
        self.assertGreater(result["lab_stddev2_um"], 4 * result["lab_stddev2_no_noise_um"])

    def test_mirror_and_combination(self):
        profile = {"first_pixel": -2, "pixel_width": 10.0, "mass": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        mirror = lm.mirrored(profile)
        self.assertEqual((mirror["first_pixel"], mirror["mass"]), (-3, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
        combined = lm.combine_profiles([profile, profile], [(0.1, 0.1, 1, 0.5), (-0.1, 0.1, -1, 0.5)])
        self.assertEqual(combined["first_pixel"], -3)
        np.testing.assert_allclose(combined["mass"], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

    def test_profiles_file_round_trip(self):
        records = [{"run_index": 0, "alpha": 4.0, "drive_frequency_hz": 1e5,
                    "initial_quasimomentum_kl": 0.05,
                    "camera_profile": {"first_pixel": -1, "pixel_width": 169.0, "mass": [0.25, 0.5, 0.25]}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "camera_profiles.csv"
            self.assertEqual(lm.write_profiles_csv(path, records, lm.validate_camera({"noise_draws": 7})), 1)
            profiles, camera = lm.read_profiles_csv(path)
        self.assertEqual(camera["noise_draws"], 7)
        self.assertEqual(profiles[0]["profile"]["mass"], [0.25, 0.5, 0.25])
        self.assertAlmostEqual(profiles[0]["profile"]["pixel_width"], 169.0)
        self.assertEqual(profiles[0]["initial_quasimomentum_kl"], 0.05)


class ValidationTests(unittest.TestCase):
    def test_validate_extended_checks_the_gpu_camera_profile(self):
        sys.path.insert(0, str(ROOT / "validation"))
        import validate_extended as ve
        rng = np.random.default_rng(5)
        psi = rng.normal(size=4096) + 1j * rng.normal(size=4096)
        config = {"number_of_iterations": "601", "save_every_nth_iteration": "100",
                  "step_x": "1", "camera_pixel_width": "169.34455556399354"}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "out").mkdir()
            with tb.open_file(str(directory / "out" / f"{600:016d}.h5"), "w") as handle:
                handle.create_array("/", "REAL", psi.real)
                handle.create_array("/", "IMAGINARY", psi.imag)
            first, mass = lm.camera_profile(np.abs(psi) ** 2, 1.0, 169.34455556399354)
            path = directory / "camera_profile.json"
            def write(values):
                path.write_text(json.dumps({"camera_profile_version": 1,
                                            "pixel_width": 169.34455556399354, "first_pixel": first,
                                            "count": len(values), "mass": list(values)}))
            write(mass.tolist())
            self.assertEqual(ve.compare_camera_profile(directory, config, 1e-12), 0.0)
            write((mass * (1 + 1e-6)).tolist())
            with self.assertRaises(AssertionError):
                ve.compare_camera_profile(directory, config, 1e-9)
            with self.assertRaises(ValueError):
                ve.compare_camera_profile(directory, dict(config, number_of_iterations="600"), 1e-9)
        self.assertEqual(ve.main.__defaults__, None)
        self.assertIn("601", Path(ve.__file__).read_text())


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """An 11.0 spread scan in two tabs with the lab measurement, then MAKE."""

    def test_scan_writes_lab_columns_and_profiles(self):
        with tempfile.TemporaryDirectory(prefix="eleven scan ") as temporary:
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
                  patch.multiple(runner, INVERTED=True, NUMBER_OF_TABS=2, GRID_POINTS=262144,
                                 HOLD_TIME_MS=None, LATTICE_HOLD_MS=0, QUASIMOMENTUM_VALUES_KL=[0.0, 0.1],
                                 QUASIMOMENTUM_SPREAD_KL=0.05,
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
            profiles, camera = lm.read_profiles_csv(results / "camera_profiles.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                data = make.dataset_analysis(results / "run_manifest.json")
            # The command line finds camera_profiles.csv next to metrics.csv.
            combined_csv = root / "combined.csv"
            completed = subprocess.run(
                [sys.executable, str(CORE / "quasimomentum_spread.py"), str(results / "metrics.csv"),
                 "--center", "0", "--spread", "0.05", "--output", str(combined_csv)],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("lab columns exact from camera_profiles.csv", completed.stdout)
            with open(combined_csv, newline="") as handle:
                command_line = {(float(r["alpha"]), float(r["drive_frequency_hz"])): r
                                for r in csv.DictReader(handle)}
        grid = manifest["parameter_grid"]
        self.assertEqual(grid["drive_phase_convention"], "hardware")
        self.assertEqual(grid["initial_cloud_width_um"], 35.0)
        self.assertTrue(manifest["parameter_label"].endswith("_PhaseHw"))
        self.assertEqual(len(rows), 4)
        for row in rows:
            for key in lm.LAB_FIELDS:
                self.assertTrue(math.isfinite(float(row[key])), key)
            # The stand-in returns the starting state: the 35 um cloud on the camera.
            self.assertAlmostEqual(float(row["camera_width_um"]), 35.0, delta=1.0)
            self.assertAlmostEqual(float(row["lab_window_fraction"]), 1.0, places=6)
            self.assertGreater(float(row["lab_stddev2_um"]), 150.0)
        self.assertEqual(sorted(profiles), [0, 1, 2, 3])
        self.assertEqual(camera["roi_sensor_rows"], [120, 780])
        # MAKE: the combined cloud's lab columns come from the added-up images.
        weights = qs.spread_weights([0.0, 0.1], 0.0, 0.05)[0]
        cells = {}
        for index, entry in profiles.items():
            cells.setdefault((entry["alpha"], entry["drive_frequency_hz"]), {})[
                entry["initial_quasimomentum_kl"]] = (index, entry["profile"])
        exact = lm.exact_spread_columns(cells, weights, camera)
        self.assertEqual(len(data["rows"]), 2)
        for row in data["rows"]:
            expected = exact[(float(row["alpha"]), float(row["drive_frequency_hz"]))]
            for key in lm.LAB_FIELDS:
                self.assertAlmostEqual(row[key], expected[key], places=9)
                self.assertAlmostEqual(float(command_line[(float(row["alpha"]), float(
                    row["drive_frequency_hz"]))][key]), expected[key], places=9)
        self.assertIn("lab_stddev2_um", data["fields"])


if __name__ == "__main__":
    unittest.main()
