"""Shared contract for the online GPU scalar and the historical snapshot metric."""

from __future__ import annotations

import math

METRIC_NAME = "mean_discrete_sum_abs_psi_fourth_power"
STATISTICS_FIELDS = (
    "metric_std", "metric_variance", "x_mean", "x_squared_mean",
    "sigma_x", "sigma_x_squared", "x_mean_time_std", "spatial_snapshots_averaged",
)
CSV_FIELDS = (
    "run_index", "alpha", "drive_frequency_hz", "snapshots_averaged", "metric"
) + STATISTICS_FIELDS


def statistics_columns(summary: dict, *, required: bool = False) -> dict:
    """Validate the statistics extension; leave unavailable legacy values blank."""
    version = summary.get("statistics_version")
    if version is None and not required:
        return dict.fromkeys(STATISTICS_FIELDS)
    if type(version) is not int or version != 1:
        raise ValueError("Missing or unsupported phase statistics; rebuild with BUILD_CUDA.bat.")
    values = {key: summary.get(key) for key in STATISTICS_FIELDS}
    count = values["spatial_snapshots_averaged"]
    if type(count) is not int or not 0 <= count <= summary["snapshots_averaged"]:
        raise ValueError("Incorrect spatial_snapshots_averaged.")
    spatial_fields = ("x_mean", "x_squared_mean", "sigma_x", "sigma_x_squared", "x_mean_time_std")
    for key in STATISTICS_FIELDS[:-1]:
        value = values[key]
        if count == 0 and key in spatial_fields:
            if value is not None:
                raise ValueError("Spatial statistics must be null for a zero-mass window.")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Statistics summary has nonfinite or missing {key}.")
        if key != "x_mean" and value < 0:
            raise ValueError(f"Statistics summary has negative {key}.")
    if not math.isclose(values["metric_std"] ** 2, values["metric_variance"], rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("Metric standard deviation and variance disagree.")
    if count:
        if not math.isclose(values["sigma_x"] ** 2, values["sigma_x_squared"], rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError("Spatial sigma and variance disagree.")
        second = values["x_mean"] ** 2 + values["x_mean_time_std"] ** 2 + values["sigma_x_squared"]
        if not math.isclose(second, values["x_squared_mean"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Spatial moments and variance disagree.")
    return values


def csv_row(record: dict, summary: dict, config: dict, contract: dict) -> dict:
    value, samples = validate_summary(summary, config, contract)
    return {
        "run_index": record["run_index"], "alpha": record["alpha"],
        "drive_frequency_hz": record["drive_frequency_hz"],
        "snapshots_averaged": samples, "metric": value,
        **statistics_columns(summary, required=contract.get("statistics_version", 0) == 1),
    }


def statistics_from_probabilities(probabilities, *, cut: int, step_x: float | None) -> dict:
    """Independent snapshot reference using normalized, cropped position weights.

    sigma_x_squared averages each snapshot's centered spatial variance;
    sigma_x is its square root (RMS width over the window). Temporal standard
    deviations use the population convention, ddof=0.
    """
    import numpy as np

    if cut < 0 or (step_x is not None and (not math.isfinite(step_x) or step_x <= 0)):
        raise ValueError("Invalid spatial cut or grid spacing.")
    metrics, centers, seconds, variances = [], [], [], []
    for probability in probabilities:
        probability = np.asarray(probability, dtype=np.float64)
        if (probability.ndim != 1 or probability.size <= 2 * cut
                or not np.all(np.isfinite(probability)) or np.any(probability < 0)):
            raise ValueError("Invalid probability array or spatial cut.")
        cropped = probability[cut:-cut] if cut else probability
        metrics.append(float(np.sum(cropped * cropped)))
        mass = float(np.sum(cropped))
        if step_x is None or mass == 0:
            continue
        positions = (np.arange(cut, probability.size - cut) - probability.size // 2) * step_x
        weights = cropped / mass
        center = float(np.sum(weights * positions))
        centers.append(center)
        seconds.append(float(np.sum(weights * positions * positions)))
        variances.append(float(np.sum(weights * (positions - center) ** 2)))
    if not metrics:
        raise ValueError("Statistics need at least one snapshot.")
    metric_variance = float(np.var(metrics, ddof=0))
    result = dict.fromkeys(STATISTICS_FIELDS)
    result.update(metric=float(np.mean(metrics)), snapshots_averaged=len(metrics),
                  metric_variance=metric_variance, metric_std=math.sqrt(metric_variance))
    if step_x is not None:
        result["spatial_snapshots_averaged"] = len(centers)
    if centers:
        spatial_variance = float(np.mean(variances))
        result.update(x_mean=float(np.mean(centers)), x_squared_mean=float(np.mean(seconds)),
                      sigma_x_squared=spatial_variance, sigma_x=math.sqrt(spatial_variance),
                      x_mean_time_std=float(np.std(centers, ddof=0)))
    for key, value in result.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"Snapshot statistics overflowed in {key}.")
    return result


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
    statistics_columns(summary, required=contract.get("statistics_version", 0) == 1)
    return value, summary["snapshots_averaged"]
