"""validation/validate_extended.py: its GPU batches and its tolerance, without a GPU."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tables as tb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))

import validate_extended as ve  # noqa: E402
import extended_metrics as em  # noqa: E402

POINTS, CUT, STEP = 4096, 100, 1.0
SETTINGS = {"lattice_wavevector": em.solver_wavevector(), "central_halfwidth": 500.0,
            "site_offset": 0.5 * math.pi / em.solver_wavevector(), "pixel_width": 174.0}


def write_state(path: Path, values: np.ndarray) -> None:
    with tb.open_file(path, "w") as handle:
        handle.create_array("/", "REAL", np.ascontiguousarray(values.real))
        handle.create_array("/", "IMAGINARY", np.ascontiguousarray(values.imag))


def symmetric_states():
    """A lattice-modulated Gaussian, symmetric about x = 0 like the real cloud,
    and a later state with a symmetric phase pattern."""
    x = (np.arange(POINTS) - POINTS // 2) * STEP
    initial = np.exp(-x ** 2 / (2.0 * 300.0 ** 2)) * np.cos(SETTINGS["lattice_wavevector"] * x)
    initial = initial.astype(complex) / math.sqrt(STEP * np.sum(np.abs(initial) ** 2))
    later = initial * np.exp(-0.7j * np.cos(x / 90.0)) * np.exp(-x ** 2 / (2.0 * 2500.0 ** 2))
    potential = -5.0 * np.cos(SETTINGS["lattice_wavevector"] * x) ** 2 + 0.0j
    return initial, potential, {0: initial, 100: later}


def fixed_sums(psi, initial):
    density = np.abs(psi) ** 2
    k2 = (2.0 * math.pi * np.fft.fftfreq(POINTS, d=STEP)) ** 2
    return [STEP * density.sum(), np.sum(k2 * np.abs(np.fft.fft(psi)) ** 2),
            np.sum(np.abs(initial) ** 2 * density)]


class ValidateExtendedTest(unittest.TestCase):
    def test_each_batch_shares_the_solver_settings(self):
        # The solver refuses a batch whose points differ in any extended
        # setting; inverted and non-inverted loading differ in the site offset.
        offsets = {}
        for loading, points in ve.BATCHES.items():
            grid = {"initial_lattice_depth_v0_er": ve.INITIAL,
                    "inverted": loading == "inverted"}
            settings = [em.solver_settings(grid, point["alpha"]) for point in points]
            self.assertGreaterEqual(len(points), 2, loading)
            for other in settings[1:]:
                self.assertEqual(other, settings[0], loading)
            offsets[loading] = settings[0]["extended_metrics_site_offset"]
        self.assertNotEqual(offsets["inverted"], offsets["non_inverted"])

    def run_compare(self, perturb):
        initial, potential, snapshots = symmetric_states()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "out").mkdir()
            write_state(directory / "initial.h5", initial)
            write_state(directory / "potential.h5", potential)
            rows = []
            for iteration, psi in snapshots.items():
                write_state(directory / "out" / f"{iteration:016d}.h5", psi)
                sums = np.concatenate([fixed_sums(psi, initial), em.reference_extended_sums(
                    psi, initial, potential, cut=CUT, step_x=STEP, **SETTINGS)])
                magnitude = np.concatenate([fixed_sums(psi, initial), em.reference_extended_sums(
                    psi, initial, potential, cut=CUT, step_x=STEP, absolute=True, **SETTINGS)])
                rows.append([iteration, 1.0, *perturb(sums, magnitude)])
            with open(directory / "extended.csv", "w", encoding="utf-8") as handle:
                handle.write(";".join(em.SERIES_COLUMNS) + "\n")
                for row in rows:
                    handle.write(";".join(repr(float(value)) for value in row) + "\n")
            config = {"initial_state_file": str(directory / "initial.h5"),
                      "potential_file": str(directory / "potential.h5"),
                      "step_x": repr(STEP), "phase_metric_cut_points_each_edge": str(CUT),
                      **{f"extended_metrics_{key}": repr(value)
                         for key, value in SETTINGS.items()}}
            return ve.compare(directory, config, 1e-9)

    def test_rounding_in_sums_that_cancel_to_zero_passes(self):
        # A GPU reduction rounds differently from NumPy, by about 1e-16 of the
        # summed magnitudes. sin_sum, x1_sum and x3_sum of a symmetric cloud
        # are themselves only rounding, so they must be judged on that scale.
        names = em.SERIES_COLUMNS[2:]
        odd = [names.index(name) for name in ("sin_sum", "x1_sum", "x3_sum")]
        old_scale_failures = []

        def gpu_rounding(sums, magnitude):
            signs = np.where(np.arange(sums.size) % 2 == 0, 1.0, -1.0)
            gpu = sums + 1e-14 * magnitude * signs
            with np.errstate(divide="ignore"):
                old_scale_failures.append(np.abs(gpu - sums)[odd] / np.abs(sums)[odd])
            return gpu

        self.assertLess(self.run_compare(gpu_rounding), 1e-12)
        # Against the size of the sums themselves, the same rounding is huge.
        self.assertTrue(np.all(np.max(old_scale_failures, axis=0) > 1e-9))

    def test_a_wrong_sum_fails_and_is_named(self):
        column = em.SERIES_COLUMNS[2:].index("site_squared_sum")

        def wrong_site_sum(sums, magnitude):
            wrong = sums.copy()
            wrong[column] *= 1.0 + 1e-6
            return wrong

        with self.assertRaisesRegex(AssertionError, "site_squared_sum"):
            self.run_compare(wrong_site_sum)


if __name__ == "__main__":
    unittest.main()
