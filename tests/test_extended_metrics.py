"""6.0 extended metrics: bins, sums, summaries, the q = 0 model, the classical
ensemble, CSV columns, and a whole scan with a stand-in solver."""

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

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_metrics as em
import run_phase_diagram_CUDA as runner
from cuda_workflow_common import experiment_drive_phase
from extended_fields import EXTENDED_FIELDS, EXTENDED_SUM_NAMES, GPU_FIELDS, extended_columns
from phase_metric import csv_fields
from test_parallel_sweep import FAKE_SOLVER

KL = em.solver_wavevector()
PERIOD = math.pi / KL
BIN_CASES = [
    (65536, 100, 1.0, PERIOD, 0.5 * PERIOD),
    (65536, 100, 1.0, PERIOD, 0.0),
    (65536, 100, 1.0, 4e-6 / 23e-9, -0.5 * 4e-6 / 23e-9),
    (4096, 7, 0.37, 5.3, 0.0),
    (1000, 0, 1.0, 10.0, 3.5),
]


class BinTests(unittest.TestCase):
    def test_python_bins_match_the_solver(self):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No C++ compiler")
        cases = ", ".join("{%d, %d, %r, %r, %r}" % case for case in BIN_CASES)
        source = (
            '#include "extended_metrics.h"\n#include <cstdio>\n'
            "struct Case { int points, cut; double step, width, offset; };\n"
            f"int main() {{ const Case cases[] = {{{cases}}};\n"
            "  for (const Case& c : cases) { auto s = bin_starts(c.points, c.cut, c.step, c.width, c.offset);\n"
            '    std::printf("%zu", s.size()); for (int v : s) std::printf(" %d", v); std::printf("\\n"); }\n'
            "  return 0; }\n")
        with tempfile.TemporaryDirectory() as temporary:
            program = Path(temporary) / "bins"
            (Path(temporary) / "bins.cpp").write_text(source)
            subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src"),
                            str(Path(temporary) / "bins.cpp"), "-o", str(program)],
                           check=True, capture_output=True, text=True)
            lines = subprocess.run([str(program)], check=True, capture_output=True,
                                   text=True).stdout.split("\n")
        for case, line in zip(BIN_CASES, lines):
            starts = em.bin_starts(*case)
            self.assertEqual([len(starts), *starts.tolist()], [int(v) for v in line.split()], case)
            self.assertEqual(starts[0], case[1])
            self.assertEqual(starts[-1], case[0] - case[1])
            self.assertTrue(np.all(np.diff(starts) > 0))

    def test_site_bins_are_one_lattice_period(self):
        starts = em.bin_starts(65536, 100, 1.0, PERIOD, 0.5 * PERIOD)
        widths = np.diff(starts)[1:-1]
        self.assertLessEqual(abs(widths.mean() - PERIOD), 0.01)
        self.assertTrue(set(widths.tolist()) <= {23, 24})


def lattice_state(points, sigma, *, top=False, step=1.0):
    """Gaussian envelope times a lattice-site profile, centred at the bottom or top."""
    x = (np.arange(points) - points // 2) * step
    site = np.cos(KL * x + (math.pi / 2 if top else 0.0)) ** 8
    psi = np.sqrt(np.exp(-x * x / (2 * sigma * sigma)) * site).astype(complex)
    return psi / math.sqrt(np.sum(np.abs(psi) ** 2) * step)


def series_file(path, states, initial, iterations, potential, config):
    with open(path, "w") as out:
        out.write(";".join(em.SERIES_COLUMNS) + "\n")
        for iteration, psi in zip(iterations, states):
            sums = em.reference_extended_sums(
                psi, initial, potential, cut=int(config["phase_metric_cut_points_each_edge"]),
                step_x=1.0, lattice_wavevector=KL, central_halfwidth=2608.7,
                site_offset=0.5 * PERIOD, pixel_width=173.9)
            density = np.abs(psi) ** 2
            fft = np.fft.fft(psi)
            k2 = (2 * math.pi * np.fft.fftfreq(psi.size)) ** 2
            fixed = [density.sum(), np.sum(k2 * np.abs(fft) ** 2),
                     np.sum(np.abs(initial) ** 2 * density)]
            out.write(";".join(repr(float(v)) for v in [iteration, 1.0, *fixed, *sums]) + "\n")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = {
            "time_step": "0.5", "number_of_iterations": "3000", "save_every_nth_iteration": "100",
            "points_x": "8192", "imaginary_time": "false", "step_x": "1.0",
            "extended_metrics_lattice_wavevector": repr(KL), "floquet_omega": "0.1",
            "floquet_phase": "0.0", "phase_metric_cut_points_each_edge": "10",
        }
        self.contract = {"final_snapshot_count": 30, "cut_points_each_edge": 10}

    def test_static_state_gives_its_own_values(self):
        psi = lattice_state(8192, 300.0)
        path = Path(self.temporary.name) / "extended.csv"
        series_file(path, [psi] * 30, psi, range(0, 3000, 100), np.zeros(8192), self.config)
        summary = em.gpu_summary(path, self.config, self.contract)
        self.assertEqual(set(summary), set(GPU_FIELDS))
        x = (np.arange(8192) - 4096).astype(float)
        density = np.abs(psi) ** 2
        expected_cos = np.sum(density * np.cos(2 * KL * x)) / density.sum()
        self.assertGreater(expected_cos, 0.7)                       # atoms at the bottom
        self.assertAlmostEqual(summary["pendulum_cos"], expected_cos, places=12)
        self.assertAlmostEqual(summary["return_probability"], 1.0, places=10)
        self.assertAlmostEqual(summary["density_overlap"], 1.0, places=12)
        self.assertAlmostEqual(summary["central_fraction"], 1.0, places=12)
        self.assertAlmostEqual(summary["escape_fraction"], 0.0, places=12)
        self.assertAlmostEqual(summary["energy_absorbed_er"], 0.0, places=9)
        self.assertAlmostEqual(summary["pendulum_cos_time_std"], 0.0, places=12)
        self.assertAlmostEqual(summary["sigma_x_um_initial"],
                               300.0 * 0.023, delta=0.05 * 300.0 * 0.023)
        self.assertGreater(summary["kurtosis"], 2.5)
        self.assertLess(summary["kurtosis"], 3.5)
        # A 300-point Gaussian covers about sqrt(4 pi) 300 / 23.1 = 46 sites.
        self.assertAlmostEqual(summary["occupied_sites"], math.sqrt(4 * math.pi) * 300 / PERIOD,
                               delta=3.0)
        self.assertLessEqual(summary["order_0"] + summary["order_2"], 1.0 + 1e-12)
        self.assertGreater(summary["p_squared"], 0.0)

    def test_growth_rate_of_a_widening_cloud(self):
        rate_per_ms = 20.0
        step_ms = 0.5 * em.solver_time_unit_s() * 1e3
        iterations = list(range(0, 3000, 100))
        states = [lattice_state(8192, 200.0 * math.exp(0.5 * rate_per_ms * (i + 1) * step_ms))
                  for i in iterations]
        path = Path(self.temporary.name) / "extended.csv"
        series_file(path, states, states[0], iterations, np.zeros(8192), self.config)
        summary = em.gpu_summary(path, self.config, self.contract)
        self.assertAlmostEqual(summary["width_growth_rate_per_ms"], rate_per_ms,
                               delta=0.05 * rate_per_ms)
        self.assertGreater(summary["spreading_speed_um_per_ms"], 0.0)


class ModelTests(unittest.TestCase):
    hold, window = 0.4e-3, 40e-6

    def test_q0_model_values_are_physical(self):
        for inverted in (False, True):
            phase = experiment_drive_phase(4.0, inverted)
            values = em.floquet_q0_metrics(10.0, 20.0, 4.0, 1.9e6, phase, self.hold, self.window)
            for key in ("q0_onsite_survival", "q0_band0_population", "q0_start_band_population",
                        "q0_excited_band_population", "q0_band_change", "q0_echo_fidelity",
                        "q0_order_0", "q0_order_2", "q0_onsite_spread"):
                self.assertGreaterEqual(values[key], -1e-12, key)
                self.assertLessEqual(values[key], 1.0 + 1e-9, key)
            self.assertGreaterEqual(values["q0_floquet_participation"], 1.0 - 1e-9)
            # Non-inverted atoms stay at the bottom, in the lowest band.
            if not inverted:
                self.assertGreater(values["q0_pendulum_cos"], 0.3)
                self.assertGreater(values["q0_band0_population"], 0.8)

    def test_classical_ensemble_stays_or_falls(self):
        points = [(4.0, 1.9e6, experiment_drive_phase(4.0, False)),
                  (2.5, 1.9e6, experiment_drive_phase(2.5, True))]
        stays, falls = em.classical_metrics(points, 10.0, 20.0, 2.045e-3, 85e-6)
        self.assertGreater(stays["cl_near_start"], 0.8)
        self.assertLess(stays["cl_site_escape"], 0.2)
        self.assertGreater(falls["cl_site_escape"], 0.5)


class ColumnTests(unittest.TestCase):
    def test_columns_are_added_only_when_on(self):
        self.assertFalse(set(EXTENDED_FIELDS) & set(csv_fields({})))
        fields = csv_fields({"extended_metrics": em.contract_settings()})
        self.assertEqual(fields[-len(EXTENDED_FIELDS):], EXTENDED_FIELDS)
        self.assertEqual(len(EXTENDED_SUM_NAMES), 21)

    def test_columns_are_validated(self):
        record = {"run_index": 3, "extended_metrics": em.empty_columns()}
        self.assertEqual(extended_columns(record), dict.fromkeys(EXTENDED_FIELDS))
        record["extended_metrics"]["kurtosis"] = float("nan")
        with self.assertRaises(ValueError):
            extended_columns(record)
        with self.assertRaises(ValueError):
            extended_columns({"run_index": 3, "extended_metrics": {}})


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """A whole compact scan with ExtendedMetrics = True and a stand-in solver."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="extended scan ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        (self.core / "gpe1d.config").write_text(
            "points_x=65536\nnumber_of_iterations=3000\nsave_every_nth_iteration=100\n"
            "show_stats_every_nth_iteration=100\ntime_step=0.5\n"
            "imaginary_time=false\nfloquet_omega=0\nstep_x=1\n"
            f"test_events={self.root / 'events.jsonl'}\n")
        self.solver = self.root / "stand in solver.exe"
        self.solver.write_text(FAKE_SOLVER)
        self.solver.chmod(0o755)
        self.results = self.root / "results"

    def run_scan(self, inverted, *arguments, quasimomentum=0.0, smoothing_um=0.0):
        with (patch.object(runner, "SCRIPT_DIRECTORY", self.core),
              patch.multiple(runner, INITIAL_QUASIMOMENTUM_KL=quasimomentum,
                             DensitySmoothing_um=smoothing_um, GRID_POINTS=65536,
                             INITIAL_CLOUD_WIDTH_UM=30.0, HOLD_TIME_MS=None,
                             QUASIMOMENTUM_VALUES_KL=None, LabMeasurement=False,
                             DRIVE_PHASE_CONVENTION="theory"),
              patch.multiple(runner, ExtendedMetrics=True, INVERTED=inverted, NUMBER_OF_TABS=1,
                             LATTICE_DEPTH_V0_ER=10.0, INITIAL_LATTICE_DEPTH_V0_ER=20.0,
                             ALPHA_VALUES=np.array([4.0, 8.0]),
                             DRIVE_FREQUENCY_HZ_VALUES=np.array([1.9e6])),
              patch.dict(os.environ, {"KAPITZA_CORE": str(CORE)}),
              patch.object(sys, "argv", ["runner", "--executable", str(self.solver),
                                        "--results", str(self.results), *arguments]),
              contextlib.redirect_stdout(io.StringIO())):
            return runner.main()

    def test_scan_writes_every_extended_column(self):
        self.assertEqual(self.run_scan(True, "--batch", "2"), 0)
        manifest = json.loads((self.results / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["analysis_contract"]["extended_metrics"]["version"], 2)
        self.assertTrue(manifest["analysis_contract"]["level_statistics"])
        with open(self.results / "metrics.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        for column in EXTENDED_FIELDS + ("eta",):
            self.assertIn(column, rows[0])
        for row in rows:
            # The stand-in solver does not evolve: inverted atoms stay on top.
            self.assertLess(float(row["pendulum_cos"]), -0.5)
            self.assertAlmostEqual(float(row["return_probability"]), 1.0, places=6)
            for column in ("q0_onsite_survival", "q0_band_change", "cl_near_start",
                           "site_ipr", "order_0"):
                self.assertTrue(0.0 <= float(row[column]) <= 1.0, column)

    def test_full_storage_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_scan(False, "--storage", "full")


if __name__ == "__main__":
    unittest.main()
