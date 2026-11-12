"""5.0: the experiment's drive sin(omega t + phi) with a fixed lattice and loading."""

from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tables as tb

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))

import create_initial_state_function as generator
from cuda_workflow_common import (
    experiment_drive_phase, generate_point_inputs, phase_dataset_label, read_config,
)
from make_phase_diagram_CUDA import automatic_plot_title, dataset_columns, grid_parameter_label

X = np.arange(-generator.GRID_POINTS // 2, generator.GRID_POINTS // 2) * generator.GRID_SPACING_M
K_L = 2.0 * np.pi / 1064e-9
ALPHA = 4.0
CENTRE = slice(generator.GRID_POINTS // 2 - 200, generator.GRID_POINTS // 2 + 200)


def read(directory: Path, name: str) -> np.ndarray:
    with tb.open_file(directory / name, "r") as handle:
        return handle.root.REAL[:] + 1j * handle.root.IMAGINARY[:]


class DrivePhaseTests(unittest.TestCase):
    def test_phase_is_zero_or_arcsin(self):
        self.assertEqual(experiment_drive_phase(10.0, False), 0.0)
        self.assertAlmostEqual(experiment_drive_phase(4.0, True), -math.pi / 6)
        self.assertAlmostEqual(experiment_drive_phase(2.0, True), -math.pi / 2)
        for alpha in (4.0, 7.3, 30.0):
            phase = experiment_drive_phase(alpha, True)
            self.assertAlmostEqual(1.0 + alpha * math.sin(phase), -1.0)

    def test_inverted_needs_alpha_at_least_two(self):
        with self.assertRaises(ValueError):
            experiment_drive_phase(1.9, True)
        self.assertEqual(experiment_drive_phase(1.9, False), 0.0)
        with self.assertRaises(ValueError):
            experiment_drive_phase(4.0, 1)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def generate(self, name: str, **keywords) -> tuple[dict, dict]:
        directory = self.root / name
        directory.mkdir()
        config = directory / "run.config"
        shutil.copy2(CORE / "gpe1d.config", config)
        with contextlib.redirect_stdout(io.StringIO()):
            generator.create_init_state(10.0, ALPHA, 1.0e6, 0.0, 20.0,
                                        output_directory=directory / "in",
                                        config_path=config, **keywords)
        arrays = {name: read(directory / "in", name)
                  for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5")}
        return arrays, read_config(config)

    def atoms_at(self, arrays: dict) -> np.ndarray:
        """Positions, near the centre, where the starting density peaks."""
        density = np.abs(arrays["lattice_gauss.h5"][CENTRE]) ** 2
        return X[CENTRE][density > 0.99 * density.max()]

    def test_4p0_model_is_unchanged_and_writes_no_phase(self):
        arrays, config = self.generate("old")
        self.assertNotIn("floquet_phase", config)
        # Atoms at cos^2(kL x) = 0, as in 4.0.
        np.testing.assert_allclose(np.cos(K_L * self.atoms_at(arrays)) ** 2, 0.0, atol=2e-3)

    def test_drive_becomes_sin_with_the_phase(self):
        for inverted in (False, True):
            phase = experiment_drive_phase(ALPHA, inverted)
            _, config = self.generate(f"drive_{inverted}", drive_phase=phase)
            self.assertAlmostEqual(float(config["floquet_phase"]), phase - math.pi / 2, places=15)
            # The solver's cos(omega t + floquet_phase) is sin(omega t + phase).
            for t in (0.0, 0.37, 2.9):
                self.assertAlmostEqual(math.cos(t + float(config["floquet_phase"])),
                                       math.sin(t + phase), places=12)

    def test_lattice_is_fixed_and_equals_the_4p0_phase_zero_lattice(self):
        old, _ = self.generate("old")
        for inverted in (False, True):
            new, _ = self.generate(f"fixed_{inverted}",
                                   drive_phase=experiment_drive_phase(ALPHA, inverted))
            for name in ("vstatic.h5", "vflo.h5"):
                self.assertTrue(np.array_equal(old[name], new[name]), name)

    def test_atoms_start_like_the_experiment(self):
        non_inverted, _ = self.generate("non", drive_phase=experiment_drive_phase(ALPHA, False))
        inverted, config = self.generate("inv", drive_phase=experiment_drive_phase(ALPHA, True))
        # Non-inverted: bottom of -V0 cos^2, where cos^2 = 1. Inverted: cos^2 = 0.
        np.testing.assert_allclose(np.cos(K_L * self.atoms_at(non_inverted)) ** 2, 1.0, atol=2e-3)
        np.testing.assert_allclose(np.cos(K_L * self.atoms_at(inverted)) ** 2, 0.0, atol=2e-3)
        # Time-averaged lattice (the static part): bottom vs top.
        for arrays, top in ((non_inverted, False), (inverted, True)):
            average = arrays["vstatic.h5"].real[CENTRE]
            here = average[np.isin(X[CENTRE], self.atoms_at(arrays))].mean()
            target = average.max() if top else average.min()
            self.assertAlmostEqual(here, target, delta=1e-3 * np.ptp(average))
        # At t = 0 the inverted atoms sit at the bottom of the flipped lattice,
        # as they were loaded.
        start = (inverted["vstatic.h5"].real
                 + inverted["vflo.h5"].real * math.cos(float(config["floquet_phase"])))[CENTRE]
        here = start[np.isin(X[CENTRE], self.atoms_at(inverted))].mean()
        self.assertAlmostEqual(here, start.min(), delta=1e-3 * np.ptp(start))

    def test_position_phase_must_be_zero_with_the_drive_phase(self):
        directory = self.root / "bad"
        directory.mkdir()
        shutil.copy2(CORE / "gpe1d.config", directory / "run.config")
        with self.assertRaises(ValueError):
            generator.create_init_state(10.0, ALPHA, 1.0e6, math.pi, 20.0,
                                        output_directory=directory / "in",
                                        config_path=directory / "run.config", drive_phase=0.0)


class WorkflowTests(unittest.TestCase):
    def test_point_inputs_follow_the_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {"lattice_depth_v0_er": 10.0, "initial_lattice_depth_v0_er": 20.0}
            for name, grid, expected in (
                ("old", {**base, "phase_radians": 0.0}, None),
                ("non", {**base, "inverted": False}, -math.pi / 2),
                ("inv", {**base, "inverted": True}, math.asin(-2.0 / ALPHA) - math.pi / 2),
            ):
                config = root / name / "run.config"
                with contextlib.redirect_stdout(io.StringIO()):
                    generate_point_inputs(CORE / "gpe1d.config", config, root / name / "in",
                                          grid, ALPHA, 1.0e6, generator.create_init_state)
                values = read_config(config)
                if expected is None:
                    self.assertNotIn("floquet_phase", values)
                else:
                    self.assertAlmostEqual(float(values["floquet_phase"]), expected, places=15)

    def test_labels_and_columns(self):
        self.assertEqual(phase_dataset_label(10.0, 20.0, 0.0, inverted=True),
                         "LatticeDepth_10ER_InitialDepth_20ER_Inverted")
        self.assertEqual(phase_dataset_label(10.0, 20.0, 0.0, inverted=False),
                         "LatticeDepth_10ER_InitialDepth_20ER_NonInverted")
        self.assertEqual(phase_dataset_label(10.0, 20.0, 3.0),
                         "LatticeDepth_10ER_InitialDepth_20ER_Phase_3rad")
        grid = {"lattice_depth_v0_er": 10.0, "initial_lattice_depth_v0_er": 20.0, "inverted": True}
        self.assertEqual(grid_parameter_label(grid), "LatticeDepth_10ER_InitialDepth_20ER_Inverted")
        self.assertIn("Inverted", automatic_plot_title(grid))
        columns = dataset_columns(grid)
        self.assertEqual(columns["inverted"], 1.0)
        self.assertIsNone(columns["phase_radians"])
        old = dataset_columns({"lattice_depth_v0_er": 10.0, "phase_radians": math.pi})
        self.assertIsNone(old["inverted"])
        self.assertEqual(old["phase_radians"], math.pi)


if __name__ == "__main__":
    unittest.main()
