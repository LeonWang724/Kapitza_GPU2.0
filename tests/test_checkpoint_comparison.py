"""Codex CUDA Port: unit test for phase-invariant checkpoint comparison."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tables as tb


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"
sys.path.insert(0, str(VALIDATION))

from compare_checkpoints import compare  # noqa: E402


def write_state(path: Path, values: np.ndarray) -> None:
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", values.real)
        handle.create_array("/", "IMAGINARY", values.imag)


def write_status(path: Path) -> None:
    path.write_text(
        "Iteration;Norm;Etot;Ekin;Epot;Eint;Chpot;Meandynpot;InitOverlap\n"
        "0;1;2;0.5;1.25;0.25;2.25;0;0.1\n",
        encoding="utf-8",
    )


class CheckpointComparisonTests(unittest.TestCase):
    def test_global_phase_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpu = root / "cpu"
            cuda = root / "cuda"
            cpu.mkdir()
            cuda.mkdir()
            values = np.linspace(0.1, 0.8, 8) + 1.0j * np.linspace(-0.2, 0.5, 8)
            write_state(cpu / "0000000000000000.h5", values)
            write_state(cuda / "0000000000000000.h5", values * np.exp(0.7j))
            cpu_status = root / "cpu.csv"
            cuda_status = root / "cuda.csv"
            write_status(cpu_status)
            write_status(cuda_status)
            summary = compare(
                cpu,
                cuda,
                cpu_status,
                cuda_status,
                0.25,
                root / "comparison.csv",
            )
            self.assertLess(summary["maximum_wavefunction_relative_l2"], 1.0e-14)
            self.assertLess(summary["maximum_density_absolute_error"], 1.0e-14)
            self.assertLess(summary["maximum_norm_absolute_error"], 1.0e-14)


if __name__ == "__main__":
    unittest.main()

