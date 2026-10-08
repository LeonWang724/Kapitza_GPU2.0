"""Analytic spatial/time statistics, lost norm, and strict summary validation."""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase_diagram" / "simulation_core"))
import run_phase_diagram_CUDA as runner
from phase_metric import (
    STATISTICS_FIELDS, expected_sampling, statistics_columns,
    statistics_from_probabilities, validate_summary,
)


class PhaseStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.config = {"points_x": "8", "number_of_iterations": "11",
                       "save_every_nth_iteration": "10", "step_x": "0.5"}
        self.contract = {"final_snapshot_count": 2, "cut_points_each_edge": 1,
                         "statistics_version": 1}
        # The first cloud has equal mass at x=-1,0; the second at x=-0.5,+0.5.
        # Edge mass must be cropped out. The second cloud has twice the norm.
        first, second = np.zeros(8), np.zeros(8)
        first[[2, 4]] = 1
        second[[3, 5]] = 2
        first[[0, 7]] = second[[0, 7]] = 1000
        self.probabilities = [first, second]

    def summary(self):
        values = statistics_from_probabilities(self.probabilities, cut=1, step_x=0.5)
        values["value"] = values.pop("metric")
        values.update(metric="mean_discrete_sum_abs_psi_fourth_power", statistics_version=1,
                      **expected_sampling(self.config, self.contract))
        return values

    def test_known_moving_cloud_has_independent_spatial_and_temporal_spreads(self):
        values = self.summary()
        self.assertEqual(values["value"], 5.0)  # instantaneous metrics 2 and 8
        self.assertEqual(values["metric_variance"], 9.0)
        self.assertEqual(values["metric_std"], 3.0)
        self.assertEqual(values["sigma_x_squared"], 0.25)
        self.assertEqual(values["sigma_x"], 0.5)
        self.assertEqual(values["x_mean"], -0.25)
        self.assertEqual(values["x_squared_mean"], 0.375)
        self.assertEqual(values["x_mean_time_std"], 0.25)
        self.assertEqual(values["spatial_snapshots_averaged"], 2)
        self.assertEqual(validate_summary(values, self.config, self.contract), (5.0, 2))

    def test_zero_mass_has_no_spatial_width_and_one_sample_has_zero_time_std(self):
        zero = statistics_from_probabilities([np.zeros(8)], cut=0, step_x=0.5)
        self.assertEqual(zero["metric_std"], 0)
        self.assertEqual(zero["spatial_snapshots_averaged"], 0)
        self.assertIsNone(zero["sigma_x"])
        one = statistics_from_probabilities(self.probabilities[:1], cut=1, step_x=0.5)
        self.assertEqual(one["metric_variance"], 0)
        self.assertEqual(one["sigma_x_squared"], 0.25)
        mixed = statistics_from_probabilities([self.probabilities[0], np.zeros(8)], cut=1, step_x=0.5)
        self.assertEqual(mixed["snapshots_averaged"], 2)
        self.assertEqual(mixed["spatial_snapshots_averaged"], 1)
        self.assertEqual(mixed["sigma_x_squared"], 0.25)

    def test_normalization_and_nonunit_spacing_are_explicit(self):
        base = self.probabilities[:1]
        original = statistics_from_probabilities(base, cut=1, step_x=0.5)
        rescaled = statistics_from_probabilities([base[0] * 0.03], cut=1, step_x=1.0)
        self.assertAlmostEqual(rescaled["sigma_x"], original["sigma_x"] * 2)
        self.assertAlmostEqual(rescaled["sigma_x_squared"], original["sigma_x_squared"] * 4)
        self.assertAlmostEqual(rescaled["metric"], original["metric"] * 0.03 ** 2)

    def test_centered_variance_stays_accurate_far_from_zero(self):
        probability = np.zeros(100000)
        probability[90000:90002] = 1
        values = statistics_from_probabilities([probability], cut=0, step_x=1e7)
        self.assertEqual(values["sigma_x_squared"], 2.5e13)
        self.assertEqual(values["sigma_x"], 5e6)

    def test_old_compact_statistics_are_blank_instead_of_invented(self):
        summary = self.summary()
        summary.pop("statistics_version")
        for key in STATISTICS_FIELDS:
            summary.pop(key)
        self.assertEqual(statistics_columns(summary), dict.fromkeys(STATISTICS_FIELDS))
        with self.assertRaisesRegex(ValueError, "rebuild"):
            validate_summary(summary, self.config, self.contract)

    def test_corrupt_new_statistics_are_rejected(self):
        for key, value in (("metric_std", float("nan")), ("metric_variance", -1),
                           ("sigma_x", 99), ("sigma_x_squared", True),
                           ("spatial_snapshots_averaged", 3), ("x_squared_mean", 99)):
            summary = copy.deepcopy(self.summary())
            summary[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_summary(summary, self.config, self.contract)

    def test_old_binary_is_rejected_before_starting_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "must_not_exist"
            with (patch.object(sys, "argv", ["runner", "--results", str(results)]),
                  patch.object(runner, "locate_executable", return_value=Path("old.exe")),
                  patch.object(runner, "query_json", return_value={"compact_phase_metric_version": 1})):
                with self.assertRaisesRegex(RuntimeError, "BUILD_CUDA"):
                    runner.main()
            self.assertFalse(results.exists())


if __name__ == "__main__":
    unittest.main()
