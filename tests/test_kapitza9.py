"""9.0: the experiment's starting cloud (340 um) is the default."""

from __future__ import annotations

import contextlib
import io
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

import extended_metrics as em  # noqa: E402
import run_phase_diagram_CUDA as runner  # noqa: E402
from create_initial_state_function import GRID_SPACING_M, create_init_state  # noqa: E402
from cuda_workflow_common import phase_dataset_label  # noqa: E402


class StartingCloudTests(unittest.TestCase):
    def test_default_is_the_experiments_cloud(self):
        self.assertEqual(runner.INITIAL_CLOUD_WIDTH_UM, 340.0)
        self.assertEqual(runner.GRID_POINTS, 262144)
        scan = runner.scan_settings({"time_step": "0.5", "number_of_iterations": "70000"})
        self.assertEqual(scan["initial_cloud_width_um"], 340.0)
        self.assertAlmostEqual(em.contract_settings(
            {"initial_cloud_width_um": 340.0})["central_halfwidth_um"], 680.0)

    def test_plan_reports_the_cloud_and_the_label(self):
        with (patch.object(runner, "locate_executable") as locate,
              patch.object(sys, "argv", ["runner", "--plan"]),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(runner.main(), 0)
        locate.assert_not_called()
        self.assertIn("absorbing walls from +/-2.11 mm; starting cloud 340 um.", output.getvalue())
        label = phase_dataset_label(10, 20, 0.0, inverted=True, quasimomentum=0.1, smoothing_um=4.0,
                                    grid_points=262144, cloud_width_um=340.0, hold_time_ms=2.0)
        self.assertTrue(label.endswith("_Box262144_W340um_T2ms"), label)

    def test_the_old_box_cannot_hold_this_cloud(self):
        with patch.multiple(runner, GRID_POINTS=65536), self.assertRaises(ValueError):
            runner.scan_settings({"time_step": "0.5", "number_of_iterations": "70000"})

    def test_generated_cloud_has_the_requested_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "run.config"
            config.write_text("points_x=65536\n")
            create_init_state(10.0, 20.0, 1.0e6, 0.0, 20.0, output_directory=root / "in",
                              config_path=config, drive_phase=-np.pi / 2, quasimomentum=0.1,
                              grid_points=262144, envelope_sigma_m=340e-6)
            with tb.open_file(str(root / "in" / "lattice_gauss.h5")) as handle:
                psi = handle.root.REAL[:] + 1j * handle.root.IMAGINARY[:]
            density = np.abs(psi) ** 2
            x = (np.arange(density.size) - density.size // 2) * GRID_SPACING_M
            mean = np.sum(x * density) / np.sum(density)
            width = np.sqrt(np.sum((x - mean) ** 2 * density) / np.sum(density))
            self.assertAlmostEqual(width * 1e6, 340.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
