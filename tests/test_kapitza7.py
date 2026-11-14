"""7.0: density smoothing, the 4 um pixel width and the starting quasimomentum."""

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
import run_phase_diagram_CUDA as runner  # noqa: E402
from create_initial_state_function import GRID_SPACING_M, create_init_state  # noqa: E402
from cuda_workflow_common import generate_point_inputs, phase_dataset_label  # noqa: E402
from extended_fields import EXTENDED_SUM_NAMES  # noqa: E402
from make_phase_diagram_CUDA import DATASET_COLUMNS, dataset_columns  # noqa: E402
from phase_metric import smooth_density, smoothing_contract, smoothing_half_width  # noqa: E402
from test_parallel_sweep import FAKE_SOLVER  # noqa: E402

# The 7.0 tests keep the 4.0-7.0 box, cloud and step count; 8.0 has its own tests.
SETTINGS_7 = {"GRID_POINTS": 65536, "INITIAL_CLOUD_WIDTH_UM": 30.0, "HOLD_TIME_MS": None}

KL = em.solver_wavevector()
PIXEL = em.PIXEL_M / GRID_SPACING_M
SUMS = {name: index for index, name in enumerate(EXTENDED_SUM_NAMES)}


def compile_and_run(source: str) -> list[str]:
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise unittest.SkipTest("No C++ compiler")
    with tempfile.TemporaryDirectory() as temporary:
        program = Path(temporary) / "probe"
        (Path(temporary) / "probe.cpp").write_text(source)
        subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src"),
                        str(Path(temporary) / "probe.cpp"), "-o", str(program)],
                       check=True, capture_output=True, text=True)
        return subprocess.run([str(program)], check=True, capture_output=True,
                              text=True).stdout.split("\n")


def gaussian_state(points=8192, sigma=300.0, centre=0.0, modulated=True):
    x = (np.arange(points) - points // 2).astype(float)
    psi = np.exp(-(x - centre) ** 2 / (4.0 * sigma ** 2)).astype(complex)
    if modulated:
        psi *= np.cos(KL * x)          # atoms at the bottom of every site
    return psi / math.sqrt(np.sum(np.abs(psi) ** 2))


def reference(psi, half=0, **overrides):
    settings = dict(cut=100, step_x=1.0, lattice_wavevector=KL, central_halfwidth=500.0,
                    site_offset=0.5 * math.pi / KL, pixel_width=PIXEL)
    settings.update(overrides)
    potential = -np.cos(KL * (np.arange(psi.size) - psi.size // 2)) ** 2 + 0j
    return em.reference_extended_sums(psi, psi, potential, smoothing_half_width=half, **settings)


class SmoothingTests(unittest.TestCase):
    def test_half_width_matches_the_solver(self):
        cases = [(173.91304347826087, 1.0), (3.0, 0.25), (7.0, 1.0), (2.0, 1.0), (0.0, 1.0)]
        lines = compile_and_run(
            '#include "extended_metrics.h"\n#include <cstdio>\n'
            "int main() { const double cases[][2] = {"
            + ", ".join("{%r, %r}" % case for case in cases)
            + "};\n  for (const auto& c : cases) std::printf(\"%d\\n\", "
              "smoothing_half_width(c[0], c[1]));\n  return 0; }\n")
        for case, line in zip(cases, lines):
            self.assertEqual(smoothing_half_width(*case), int(line), case)
        self.assertEqual(smoothing_half_width(173.91304347826087, 1.0), 87)
        with self.assertRaises(ValueError):     # rounds to no neighbour at all
            smoothing_half_width(0.8, 1.0)

    def test_four_micrometres_is_175_grid_points(self):
        contract = smoothing_contract(4.0, 1.0, GRID_SPACING_M)
        self.assertEqual(contract["half_width_points"], 87)
        self.assertAlmostEqual(contract["window_um"], 175 * 0.023, places=12)
        self.assertIsNone(smoothing_contract(0.0, 1.0, GRID_SPACING_M))
        for bad in (-1.0, True, float("nan")):
            with self.assertRaises(ValueError):
                smoothing_contract(bad, 1.0, GRID_SPACING_M)

    def test_moving_average_keeps_mass_and_adds_the_window_variance(self):
        rho = np.abs(gaussian_state(sigma=400.0, modulated=False)) ** 2
        half = 87
        smooth = smooth_density(rho, half)
        self.assertEqual(smooth.size, rho.size)
        # Each value is the plain average of its 2h + 1 neighbours.
        index = 4096 + 13
        self.assertAlmostEqual(smooth[index], rho[index - half:index + half + 1].mean(), places=15)
        self.assertAlmostEqual(smooth.sum(), rho.sum(), places=12)
        x = np.arange(rho.size) - rho.size // 2

        def variance(density):
            mean = np.sum(x * density) / density.sum()
            return np.sum((x - mean) ** 2 * density) / density.sum()
        # A moving average over 2h + 1 points adds ((2h+1)^2 - 1) / 12 = h (h + 1) / 3.
        self.assertAlmostEqual(variance(smooth) - variance(rho), half * (half + 1) / 3.0, places=6)
        np.testing.assert_array_equal(smooth_density(rho, 0), rho)

    def test_on_site_sums_and_pixels_keep_the_raw_density(self):
        psi = gaussian_state()
        raw, smooth = reference(psi), reference(psi, half=87)
        for name in ("cos_sum", "sin_sum", "potential_sum", "pixel_squared_sum",
                     "pixel_x1_sum", "pixel_x2_sum"):
            self.assertEqual(raw[SUMS[name]], smooth[SUMS[name]], name)
        # The pendulum angle survives only without smoothing: 4 um is 7.5 sites.
        # (A cos^2 density has <cos 2kL x> = 1/2.)
        self.assertAlmostEqual(raw[SUMS["cos_sum"]] / raw[SUMS["mass"]], 0.5, places=6)
        self.assertLess(abs(np.sum(smooth_density(np.abs(psi) ** 2, 87)
                                   * np.cos(2 * KL * (np.arange(psi.size) - psi.size // 2)))), 0.05)
        # Position metrics change: mass is kept, the width grows by the window.
        self.assertAlmostEqual(smooth[SUMS["mass"]], raw[SUMS["mass"]], places=12)
        grown = smooth[SUMS["x2_sum"]] - raw[SUMS["x2_sum"]]
        self.assertAlmostEqual(grown / raw[SUMS["mass"]], 87 * 88 / 3.0, delta=1.0)
        self.assertLess(smooth[SUMS["rho_squared_sum"]], raw[SUMS["rho_squared_sum"]])


class PixelWidthTests(unittest.TestCase):
    def test_bin_centres_match_the_solver(self):
        cases = [(65536, 100, 1.0, PIXEL, -0.5 * PIXEL), (4096, 7, 0.25, 9.5, 1.25),
                 (2000, 0, 1.0, 3.0, 0.0)]
        lines = compile_and_run(
            '#include "extended_metrics.h"\n#include <cstdio>\n'
            "struct Case { int points, cut; double step, width, offset; };\n"
            "int main() { const Case cases[] = {"
            + ", ".join("{%d, %d, %r, %r, %r}" % case for case in cases)
            + "};\n  for (const Case& c : cases) { auto l = bin_layout(c.points, c.cut, c.step, "
              "c.width, c.offset);\n    std::printf(\"%zu\", l.centres.size());\n"
              "    for (double v : l.centres) std::printf(\" %.17g\", v); std::printf(\"\\n\"); }\n"
              "  return 0; }\n")
        for case, line in zip(cases, lines):
            starts, centres = em.bin_layout(*case)
            values = line.split()
            self.assertEqual(int(values[0]), centres.size, case)
            self.assertEqual(centres.size, starts.size - 1)
            np.testing.assert_array_equal(np.array(values[1:], dtype=float), centres)
            # Every bin's centre lies within half a bin of its grid points.
            positions = (starts[:-1] - case[0] // 2) * case[2]
            self.assertTrue(np.all(np.abs(positions - centres) <= 0.5 * case[3] + case[2]))

    def test_pixel_width_is_the_width_plus_the_pixel(self):
        for centre in (0.0, 431.7):
            psi = gaussian_state(points=65536, sigma=1300.0, centre=centre)
            sums = reference(psi)
            mass = sums[SUMS["mass"]]
            pixel_mean = sums[SUMS["pixel_x1_sum"]] / mass
            pixel = sums[SUMS["pixel_x2_sum"]] / mass - pixel_mean ** 2
            fine_mean = sums[SUMS["x1_sum"]] / mass
            fine = sums[SUMS["x2_sum"]] / mass - fine_mean ** 2
            # Sheppard: binning into pixels adds about pixel^2 / 12 to the variance.
            self.assertAlmostEqual(pixel - fine, PIXEL ** 2 / 12.0, delta=0.05 * PIXEL ** 2)
            self.assertAlmostEqual(pixel_mean, fine_mean, delta=0.05 * PIXEL)

    def test_summary_reports_the_pixel_width_in_micrometres(self):
        names = em.SERIES_COLUMNS
        rows = []
        for iteration in range(0, 3000, 100):
            sums = dict.fromkeys(names, 1.0)
            sums.update(iteration=iteration, mass=2.0, x1_sum=0.0, x2_sum=2.0 * 1300.0 ** 2,
                        pixel_x1_sum=0.0, pixel_x2_sum=2.0 * 1301.0 ** 2,
                        x3_sum=0.0, x4_sum=2.0 * 3 * 1300.0 ** 4, momentum_total=1.0)
            rows.append([sums[name] for name in names])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "extended.csv"
            path.write_text(";".join(names) + "\n"
                            + "".join(";".join(repr(v) for v in row) + "\n" for row in rows))
            config = {"extended_metrics_lattice_wavevector": repr(KL), "step_x": "1",
                      "time_step": "0.5", "number_of_iterations": "3000",
                      "save_every_nth_iteration": "100", "points_x": "65536",
                      "floquet_omega": "0.1"}
            summary = em.gpu_summary(path, config, {"final_snapshot_count": 30,
                                                    "cut_points_each_edge": 100})
        self.assertAlmostEqual(summary["sigma_x_4um_um"], 1301.0 * 0.023, places=9)


class QuasimomentumTests(unittest.TestCase):
    def generate(self, directory: Path, **keywords):
        config = directory / "run.config"
        directory.mkdir()
        shutil.copy2(CORE / "gpe1d.config", config)
        create_init_state(10.0, 4.0, 1.9e6, 0.0, 20.0, output_directory=directory / "in",
                          config_path=config, drive_phase=0.0, **keywords)
        with tb.open_file(directory / "in" / "lattice_gauss.h5") as handle:
            return handle.root.REAL[:] + 1j * handle.root.IMAGINARY[:]

    def test_every_atom_starts_at_q(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = self.generate(root / "default")
            zero = self.generate(root / "zero", quasimomentum=0.0)
            shifted = self.generate(root / "shifted", quasimomentum=0.1)
        np.testing.assert_array_equal(zero, default)    # q = 0 is the 6.0 state
        k = 2 * math.pi * np.fft.fftfreq(default.size) / KL     # in units of kL
        for psi, q in ((default, 0.0), (shifted, 0.1)):
            power = np.abs(np.fft.fft(psi)) ** 2
            # Plane waves only at k = q + 2n (the Gaussian envelope's width ~ 0.003).
            near = np.abs(((k - q + 1.0) % 2.0) - 1.0) < 0.02
            self.assertGreater(power[near].sum() / power.sum(), 0.999, q)
            self.assertAlmostEqual(k[np.argmax(power)], q, delta=0.005)

    def test_invalid_quasimomentum_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in (1.0, -1.0, 1.5, True):
                with self.assertRaises(ValueError, msg=str(value)):
                    self.generate(Path(temporary) / f"bad{value}", quasimomentum=value)

    def test_point_inputs_pass_q_only_when_set(self):
        seen = []

        def generator(*arguments, **keywords):
            seen.append(keywords.get("quasimomentum"))
            for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
                (keywords["output_directory"] / name).write_text("")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.config"
            base.write_text("step_x=1\n")
            for index, q in enumerate((0.0, 0.1, None)):
                grid = {"lattice_depth_v0_er": 10.0, "initial_lattice_depth_v0_er": 20.0,
                        "inverted": False}
                if q is not None:
                    grid["initial_quasimomentum_kl"] = q
                generate_point_inputs(base, root / f"p{index}" / "run.config",
                                      root / f"p{index}" / "in", grid, 4.0, 1e6, generator)
        self.assertEqual(seen, [None, 0.1, None])

    def test_bloch_model_at_zero_is_unchanged_and_follows_q(self):
        arguments = (10.0, 20.0, 12.0, 0.8e6, 0.0, 2.0e-5, 1.0e-5)
        default = em.floquet_q0_metrics(*arguments)
        self.assertEqual(em.floquet_q0_metrics(*arguments, quasimomentum=0.0), default)
        shifted = em.floquet_q0_metrics(*arguments, quasimomentum=0.1)
        self.assertEqual(set(shifted), set(default))
        self.assertTrue(all(value is None or math.isfinite(value) for value in shifted.values()))
        # <p^2> counts the plane waves at 2n + q: at least q^2 = 0.01 more at the start.
        self.assertNotEqual(shifted["q0_p_squared"], default["q0_p_squared"])
        with self.assertRaises(ValueError):
            em.floquet_q0_metrics(*arguments, quasimomentum=1.0)


class RecordingTests(unittest.TestCase):
    def test_labels_and_columns_record_q_and_smoothing(self):
        label = phase_dataset_label(10, 20, 0.0, inverted=True, quasimomentum=0.1, smoothing_um=4.0)
        self.assertEqual(label, "LatticeDepth_10ER_InitialDepth_20ER_Inverted_Q0p1kL_Smooth4um")
        self.assertEqual(phase_dataset_label(10, 20, 0.0, inverted=True),
                         "LatticeDepth_10ER_InitialDepth_20ER_Inverted")
        for column in ("initial_quasimomentum_kl", "density_smoothing_um"):
            self.assertIn(column, DATASET_COLUMNS)
        grid = {"lattice_depth_v0_er": 10.0, "inverted": True}
        self.assertEqual(dataset_columns(grid)["initial_quasimomentum_kl"], 0.0)
        grid.update(initial_quasimomentum_kl=0.1, density_smoothing_um=4.0)
        self.assertEqual(dataset_columns(grid)["density_smoothing_um"], 4.0)

    def test_plan_reports_both_settings(self):
        with (patch.multiple(runner, **SETTINGS_7),
              patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        text = output.getvalue()
        self.assertIn("Initial quasimomentum: q = 0.1 hbar kL", text)
        self.assertIn("Density smoothing: 4 um (175 grid points, 4.025 um)", text)
        self.assertIn("at q = 0.1 kL", text)

    def test_reference_loop_is_refused_with_smoothing(self):
        with (patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan", "--reference-loop"]),
              contextlib.redirect_stderr(io.StringIO())):
            with self.assertRaises(SystemExit):
                runner.main()
        locate.assert_not_called()


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """A compact 7.0 scan (q = 0.1, 4 um smoothing) with a stand-in solver."""

    def test_scan_records_and_applies_both_settings(self):
        with tempfile.TemporaryDirectory(prefix="seven scan ") as temporary:
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
                  patch.multiple(runner, **SETTINGS_7),
                  patch.multiple(runner, INVERTED=False, NUMBER_OF_TABS=1,
                                 LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                                 ALPHA_VALUES=np.array([4.0]),
                                 DRIVE_FREQUENCY_HZ_VALUES=np.array([1.9e6])),
                  patch.dict(os.environ, {"KAPITZA_CORE": str(CORE)}),
                  patch.object(sys, "argv", ["runner", "--executable", str(solver),
                                            "--results", str(results)]),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(runner.main(), 0)
            manifest = json.loads((results / "run_manifest.json").read_text())
            with open(results / "metrics.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            record = json.loads((results / "point_results.jsonl").read_text().splitlines()[0])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["parameter_grid"]["initial_quasimomentum_kl"], 0.1)
        self.assertEqual(manifest["parameter_grid"]["density_smoothing_um"], 4.0)
        self.assertEqual(manifest["analysis_contract"]["density_smoothing"]["half_width_points"], 87)
        self.assertEqual(manifest["analysis_contract"]["level_statistics"]["quasimomentum"], 0.1)
        self.assertTrue(manifest["parameter_label"].endswith("_Q0p1kL_Smooth4um"))
        row = rows[0]
        # The stand-in solver returns the starting state's sums: a 30 um cloud,
        # widened slightly by the 4 um pixels.
        self.assertAlmostEqual(float(row["sigma_x_4um_um"]), 30.0, delta=1.0)
        self.assertGreater(float(row["sigma_x_4um_um"]), 30.0)
        self.assertEqual(record["extended_metrics"]["q0_p_squared"], float(row["q0_p_squared"]))


if __name__ == "__main__":
    unittest.main()
