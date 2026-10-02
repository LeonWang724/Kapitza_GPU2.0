"""Codex CUDA Port: unit test for the manifest-driven historical metric."""

from __future__ import annotations

import json
import os
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

from make_phase_diagram_CUDA import (  # noqa: E402
    analyze,
    automatic_plot_title,
    grid_parameter_label,
)


def write_state(path: Path, real: np.ndarray) -> None:
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", real)
        handle.create_array("/", "IMAGINARY", np.zeros_like(real))


class AnalysisMetricTests(unittest.TestCase):
    def test_numeric_sort_cut_and_final_average(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out_000"
            output.mkdir()
            base = np.arange(12, dtype=np.float64) / 10.0
            write_state(output / "0000000000000010.h5", base * 2.0)
            write_state(output / "0000000000000000.h5", base)
            write_state(output / "0000000000000020.h5", base * 3.0)
            manifest = {
                "status": "completed",
                "parameter_grid": {
                    "alpha_values": [0.0],
                    "drive_frequency_hz_values": [0.0],
                    "lattice_depth_v0_er": 20.0,
                    "initial_lattice_depth_v0_er": 40.0,
                    "phase_radians": 0.0,
                },
                "analysis_contract": {
                    "final_snapshot_count": 2,
                    "cut_points_each_edge": 2,
                    "colormap": "inferno",
                    "frequency_axis_inverted": True,
                },
                "runs": [
                    {
                        "run_index": 0,
                        "alpha_index": 0,
                        "frequency_index": 0,
                        "alpha": 0.0,
                        "drive_frequency_hz": 0.0,
                        "status": "completed",
                        "output_directory": "out_000",
                    }
                ],
            }
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            outputs = analyze(manifest_path)
            second_outputs = analyze(manifest_path)
            with np.load(outputs["npz"]) as data:
                measured = float(data["metric_matrix"][0, 0])
            expected = np.mean(
                [
                    np.sum(((base * 2.0) ** 2)[2:-2] ** 2),
                    np.sum(((base * 3.0) ** 2)[2:-2] ** 2),
                ]
            )
            self.assertAlmostEqual(measured, expected)
            self.assertNotEqual(outputs["image"], second_outputs["image"])
            self.assertTrue(outputs["image"].is_file())
            self.assertTrue(outputs["image"].name.startswith("Analysis_"))
            self.assertLess(len(outputs["image"].name), 70)

    def test_parameterized_default_title_and_label(self) -> None:
        grid = {
            "lattice_depth_v0_er": 20.0,
            "initial_lattice_depth_v0_er": 40.0,
            "phase_radians": 0.5,
        }
        self.assertEqual(
            grid_parameter_label(grid),
            "LatticeDepth_20ER_InitialDepth_40ER_Phase_0p5rad",
        )
        self.assertEqual(
            automatic_plot_title(grid),
            "Lattice Depth 20 E_R * Initial Depth 40 E_R * Phase 0.5 rad",
        )


if __name__ == "__main__":
    unittest.main()
