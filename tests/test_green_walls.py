"""Optional Gaussian green walls around the initial wavefunction."""

from __future__ import annotations

import contextlib
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

import create_initial_state_function as generator
import run_phase_diagram_CUDA as runner
from cuda_workflow_common import phase_dataset_label
from make_phase_diagram_CUDA import automatic_plot_title, grid_parameter_label
from test_parallel_sweep import FAKE_SOLVER

WALLS = {"height_er": 100.0, "sigma_um": 5.0, "gap_um": 200.0}
X = np.arange(-generator.GRID_POINTS // 2, generator.GRID_POINTS // 2) * generator.GRID_SPACING_M
K_L = 2.0 * np.pi / 1064e-9
E_R = generator.hbar**2 * K_L**2 / (2 * generator.mli)
E_S = generator.hbar**2 / (generator.mli * generator.GRID_SPACING_M**2)


def read_inputs(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = {}
    for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
        with tb.open_file(directory / name, "r") as handle:
            arrays[name] = (handle.root.REAL[:], handle.root.IMAGINARY[:])
    return arrays


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def generate(self, name: str, **keywords) -> dict:
        directory = self.root / name
        directory.mkdir()
        shutil.copy2(CORE / "gpe1d.config", directory / "run.config")
        with contextlib.redirect_stdout(io.StringIO()):
            generator.create_init_state(20.0, 50.0, 3.0e6, 0.0, 40.0, output_directory=directory / "in",
                                        config_path=directory / "run.config", **keywords)
        return read_inputs(directory / "in")

    def test_walls_off_is_bitwise_identical_to_no_keyword(self):
        plain, off = self.generate("plain"), self.generate("off", green_walls=None)
        for name in plain:
            for part in (0, 1):
                self.assertTrue(np.array_equal(plain[name][part].view(np.uint64),
                                               off[name][part].view(np.uint64)), name)

    def test_walls_add_two_gaussians_to_the_static_potential_only(self):
        plain, walls = self.generate("plain"), self.generate("walls", green_walls=WALLS)
        added = walls["vstatic.h5"][0] - plain["vstatic.h5"][0]
        sigma, centre = 5e-6, 100e-6
        expected = 100.0 * E_R / E_S * (np.exp(-(X + centre) ** 2 / (2 * sigma ** 2))
                                         + np.exp(-(X - centre) ** 2 / (2 * sigma ** 2)))
        np.testing.assert_allclose(added, expected, rtol=1e-9, atol=1e-12 * expected.max())
        peak = np.argmin(np.abs(X - centre))
        self.assertAlmostEqual(added[peak] / (E_R / E_S), 100.0, delta=0.01)
        # Symmetric about the cloud centre x = 0 (grid index n/2).
        middle = generator.GRID_POINTS // 2
        mirrored = np.arange(1, middle)
        np.testing.assert_allclose(added[middle + mirrored], added[middle - mirrored], rtol=1e-12, atol=1e-15)
        self.assertLess(added[middle], 1e-12 * added.max())
        self.assertTrue(np.array_equal(plain["vstatic.h5"][1], walls["vstatic.h5"][1]))
        for name in ("lattice_gauss.h5", "vflo.h5"):
            for part in (0, 1):
                self.assertTrue(np.array_equal(plain[name][part], walls[name][part]), name)

    def test_invalid_settings_are_rejected(self):
        for change in ({"height_er": 0.0}, {"height_er": -5.0}, {"sigma_um": 0.0},
                       {"gap_um": -1.0}, {"gap_um": float("nan")}, {"height_er": float("inf")},
                       {"sigma_um": True}, {"gap_um": "200"}):
            with self.assertRaises(ValueError, msg=str(change)):
                generator.check_green_walls(**{**WALLS, **change})
        # Centres at +-550 um reach into the absorber, which starts near 527 um.
        with self.assertRaisesRegex(ValueError, "absorbing layer"):
            generator.check_green_walls(100.0, 5.0, 1100.0)
        generator.check_green_walls(100.0, 5.0, 1000.0)

    def test_check_reports_fraction_of_initial_cloud_between_walls(self):
        checked = generator.check_green_walls(**WALLS)
        self.assertEqual({key: checked[key] for key in WALLS}, WALLS)
        self.assertAlmostEqual(checked["initial_fraction_between_walls"],
                               math.erf(100.0 / (math.sqrt(2) * 30.0)), places=12)


class RunnerTests(unittest.TestCase):
    def plan(self, **settings) -> str:
        with (patch.multiple(runner, **settings),
              patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        return output.getvalue()

    def test_plan_reports_walls_or_off(self):
        self.assertIn("Green walls: off", self.plan(GreenWalls=False))
        text = self.plan(GreenWalls=True, GreenWallHeight_ER=80.0, GreenWallSigma_um=4.0,
                         GreenWallGap_um=60.0)
        self.assertIn("Green walls: 80 E_R high, sigma 4 um, centres at +/-30 um", text)
        self.assertIn("overlap the initial cloud", text)

    def test_invalid_settings_stop_before_cuda(self):
        for settings in ({"GreenWalls": "yes"}, {"GreenWalls": 1},
                         {"GreenWalls": True, "GreenWallSigma_um": 0.0},
                         {"GreenWalls": True, "GreenWallGap_um": 5000.0}):
            with (patch.multiple(runner, **settings),
                  patch.object(runner, "locate_executable") as locate,
                  patch.object(sys, "argv", ["runner", "--plan"]),
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit, msg=str(settings)):
                    runner.main()
            locate.assert_not_called()

    def test_labels_and_titles_name_the_walls(self):
        self.assertEqual(phase_dataset_label(20.0, 40.0, 0.0),
                         "LatticeDepth_20ER_InitialDepth_40ER_Phase_0rad")
        grid = {"lattice_depth_v0_er": 20.0, "initial_lattice_depth_v0_er": 40.0,
                "phase_radians": 0.0, "green_walls": {"height_er": 100.0, "sigma_um": 2.5,
                                                      "gap_um": 200.0}}
        self.assertEqual(grid_parameter_label(grid),
                         "LatticeDepth_20ER_InitialDepth_40ER_Phase_0rad_Walls_H100ER_S2p5um_G200um")
        self.assertIn("Green walls 100 E_R, sigma 2.5 um, gap 200 um", automatic_plot_title(grid))
        grid["green_walls"] = None
        self.assertNotIn("Green walls", automatic_plot_title(grid))


@unittest.skipIf(os.name == "nt", "The stand-in solver uses a POSIX shebang; run CUDA validation on Windows.")
class ScanTests(unittest.TestCase):
    """Real input generation in worker tabs, with a stand-in solver."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="walls scan ")
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

    def test_full_scan_in_two_tabs_writes_walls_into_every_point(self):
        with (patch.object(runner, "SCRIPT_DIRECTORY", self.core),
              patch.multiple(runner, GreenWalls=True, GreenWallHeight_ER=100.0,
                             GreenWallSigma_um=5.0, GreenWallGap_um=200.0, NUMBER_OF_TABS=2,
                             ALPHA_VALUES=np.array([10.0, 20.0]),
                             DRIVE_FREQUENCY_HZ_VALUES=np.array([1.0e6])),
              patch.object(sys, "argv", ["runner", "--headless", "--storage", "full",
                                        "--executable", str(self.solver), "--results", str(self.results)]),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(runner.main(), 0)
        manifest = json.loads((self.results / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["parameter_grid"]["green_walls"], WALLS)
        self.assertTrue(manifest["parameter_label"].endswith("_Walls_H100ER_S5um_G200um"))
        peak = np.argmin(np.abs(X - 100e-6))
        middle = generator.GRID_POINTS // 2
        for record in manifest["runs"]:
            vstatic = read_inputs((self.results / record["inputs"]["vstatic.h5"]["path"]).parent)["vstatic.h5"][0]
            # The wall stands about 100 E_R above the lattice next to it.
            self.assertGreater((vstatic[peak] - vstatic[middle]) / (E_R / E_S), 90.0)


if __name__ == "__main__":
    unittest.main()
