"""Shared contract for the online GPU scalar and the historical snapshot metric."""

from __future__ import annotations

import math

METRIC_NAME = "mean_discrete_sum_abs_psi_fourth_power"
CSV_FIELDS = (
    "run_index", "alpha", "drive_frequency_hz", "snapshots_averaged", "metric"
)


def expected_sampling(config: dict, contract: dict) -> dict[str, int]:
    iterations = int(float(config["number_of_iterations"]))
    interval = int(float(config["save_every_nth_iteration"]))
    points = int(float(config["points_x"]))
    final_count = int(contract["final_snapshot_count"])
    cut = int(contract["cut_points_each_edge"])
    if str(config.get("imaginary_time", "false")).lower() != "false":
        raise ValueError("Compact phase metrics require real-time evolution.")
    if iterations <= 0 or interval <= 0 or final_count <= 0:
        raise ValueError("Metric iteration count, sample interval, and window must be positive.")
    if cut < 0 or points <= 2 * cut:
        raise ValueError("Spatial cut removes all metric points.")
    total_samples = 1 + (iterations - 1) // interval
    samples = min(final_count, total_samples)
    return {
        "snapshots_averaged": samples,
        "cut_points_each_edge": cut,
        "first_snapshot_iteration": (total_samples - samples) * interval,
        "last_snapshot_iteration": (total_samples - 1) * interval,
        "save_every_nth_iteration": interval,
        "number_of_iterations": iterations,
    }


def validate_summary(summary: dict, config: dict, contract: dict) -> tuple[float, int]:
    if contract.get("metric", METRIC_NAME) != METRIC_NAME:
        raise ValueError("Unsupported phase-diagram metric contract.")
    if summary.get("metric") != METRIC_NAME:
        raise ValueError("Solver returned a different phase-diagram metric.")
    for key, expected in expected_sampling(config, contract).items():
        if type(summary.get(key)) is not int or summary[key] != expected:
            raise ValueError(f"Metric summary has incorrect {key}; expected {expected}.")
    value = summary.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Solver did not return a numeric metric.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Solver returned a nonfinite or negative metric.")
    return value, summary["snapshots_averaged"]
