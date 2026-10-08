"""Create a preserved phase-diagram analysis from a named or latest CUDA dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tables as tb

from cuda_workflow_common import RESULTS_DIRECTORY, phase_dataset_label, read_config
from phase_metric import CSV_FIELDS, STATISTICS_FIELDS, csv_row, statistics_from_probabilities


# ---------------------------------------------------------------------------
# PLOT CONFIGURATION. Leave DATASET_NAME as None to analyze the newest
# completed dataset, or paste one results_cuda folder name between the quotes.
# Set PLOT_TITLE to None to build it from lattice depth, initial depth, and phase.
# ---------------------------------------------------------------------------
DATASET_NAME: str | None = None
PLOT_TITLE: str | None = None
X_AXIS_TITLE = r"$\alpha$"
Y_AXIS_TITLE = r"$\nu_{\mathrm{flo}}$ (Hz)"
COLORBAR_TITLE = r"Mean discrete $\sum_j |\psi_j|^4$"


def iteration(path: Path) -> int:
    match = re.fullmatch(r"(\d{16})\.h5", path.name)
    return int(match.group(1)) if match else -1


def read_probability(path: Path) -> np.ndarray:
    with tb.open_file(path, "r") as handle:
        real = handle.root.REAL[:]
        imaginary = handle.root.IMAGINARY[:]
    return real * real + imaginary * imaginary


def grid_parameter_label(grid: dict) -> str:
    lattice_depth = float(grid["lattice_depth_v0_er"])
    initial_depth = float(
        grid.get("initial_lattice_depth_v0_er", 2.0 * lattice_depth)
    )
    phase = float(grid.get("phase_radians", 0.0))
    return phase_dataset_label(lattice_depth, initial_depth, phase)


def automatic_plot_title(grid: dict) -> str:
    lattice_depth = float(grid["lattice_depth_v0_er"])
    initial_depth = float(
        grid.get("initial_lattice_depth_v0_er", 2.0 * lattice_depth)
    )
    phase = float(grid.get("phase_radians", 0.0))
    return (
        f"Lattice Depth {lattice_depth:g} E_R * "
        f"Initial Depth {initial_depth:g} E_R * Phase {phase:g} rad"
    )


def resolve_manifest(dataset: str | Path | None = None) -> Path:
    """Resolve a manifest path, results folder, dataset name, or newest dataset."""
    selected = dataset if dataset is not None else DATASET_NAME
    if selected is not None:
        candidate = Path(selected).expanduser()
        if candidate.is_file():
            manifest_path = candidate
        elif candidate.is_dir():
            manifest_path = candidate / "run_manifest.json"
        else:
            manifest_path = RESULTS_DIRECTORY / str(selected) / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Completed dataset manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise RuntimeError(f"Dataset is not completed: {manifest_path.parent.name}")
        return manifest_path.resolve()

    completed: list[tuple[float, Path]] = []
    for manifest_path in RESULTS_DIRECTORY.glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "completed":
            completed.append((manifest_path.stat().st_mtime, manifest_path))
    if not completed:
        raise FileNotFoundError(
            f"No completed CUDA datasets were found under {RESULTS_DIRECTORY}"
        )
    return max(completed, key=lambda item: item[0])[1].resolve()


def analyze(
    manifest_path: Path,
    *,
    plot_title: str | None = None,
    x_axis_title: str = X_AXIS_TITLE,
    y_axis_title: str = Y_AXIS_TITLE,
    colorbar_title: str = COLORBAR_TITLE,
) -> dict[str, Path]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("The manifest is not marked completed; refusing partial analysis.")
    root = manifest_path.parent
    grid = manifest["parameter_grid"]
    contract = manifest["analysis_contract"]
    alpha_values = np.asarray(grid["alpha_values"], dtype=np.float64)
    frequency_values = np.asarray(grid["drive_frequency_hz_values"], dtype=np.float64)
    metric = np.full((frequency_values.size, alpha_values.size), np.nan)
    statistics_matrices = {key: np.full_like(metric, np.nan) for key in STATISTICS_FIELDS}

    final_count = int(contract["final_snapshot_count"])
    cut = int(contract["cut_points_each_edge"])
    if final_count <= 0 or cut < 0:
        raise ValueError("Invalid metric window or spatial cut.")
    compact = manifest.get("storage_mode") == "compact"
    if compact:
        journal_path = root / manifest["point_results_file"]
        runs = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        runs = manifest["runs"]
    if len(runs) != alpha_values.size * frequency_values.size:
        raise ValueError("Completed dataset does not contain every grid point.")
    rows = []
    seen = set()
    for run in runs:
        if run.get("status") != "completed":
            raise RuntimeError(f"Run {run['run_index']} is incomplete.")
        if compact:
            row = csv_row(run, run["metric_summary"], manifest["simulation_config"], contract)
            mean_metric = row["metric"]
        else:
            output_directory = root / run["output_directory"]
            snapshots = sorted(
                (path for path in output_directory.glob("*.h5") if iteration(path) >= 0),
                key=iteration,
            )
            if not snapshots:
                raise FileNotFoundError(f"No snapshots in {output_directory}")
            selected = snapshots[-final_count:]
            config = manifest.get("simulation_config", {})
            if "step_x" not in config and run.get("config_file"):
                config = read_config(root / run["config_file"])
            step_x = float(config["step_x"]) if "step_x" in config else None
            if step_x is None and contract.get("statistics_version") == 1:
                raise ValueError("Spatial statistics require the saved step_x grid spacing.")
            statistics = statistics_from_probabilities(
                (read_probability(snapshot) for snapshot in selected), cut=cut, step_x=step_x,
            )
            mean_metric = statistics["metric"]
            row = {"run_index": run["run_index"], "alpha": run["alpha"],
                   "drive_frequency_hz": run["drive_frequency_hz"], **statistics}
        alpha_index = int(run["alpha_index"])
        frequency_index = int(run["frequency_index"])
        if not (0 <= alpha_index < alpha_values.size and 0 <= frequency_index < frequency_values.size):
            raise ValueError("Grid point index is out of range.")
        coordinate = (frequency_index, alpha_index)
        if coordinate in seen:
            raise ValueError("Dataset contains a duplicate grid point.")
        if run["alpha"] != alpha_values[alpha_index] or run["drive_frequency_hz"] != frequency_values[frequency_index]:
            raise ValueError("Grid point parameters do not match the manifest.")
        seen.add(coordinate)
        metric[frequency_index, alpha_index] = mean_metric
        for key in STATISTICS_FIELDS:
            if row[key] is not None:
                statistics_matrices[key][frequency_index, alpha_index] = row[key]
        rows.append(row)
        print(
            f"out_{run['run_index']:03d}: alpha={run['alpha']:.10g}, "
            f"frequency={run['drive_frequency_hz']:.10g} Hz, metric={mean_metric:.10g}"
        )

    analysis_directory = root / "analysis"
    analysis_directory.mkdir(exist_ok=True)
    dataset_name = str(manifest.get("dataset_name", root.name))
    descriptive_name = str(manifest.get("parameter_label", grid_parameter_label(grid)))
    analyzed_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # The containing folder/manifest and NPZ retain dataset identity. Repeating
    # it in filenames exceeded Windows' path limit in the original workflow.
    artifact_stem = f"Analysis_{analyzed_utc}"
    npz_path = analysis_directory / f"{artifact_stem}_metric.npz"
    csv_path = analysis_directory / f"{artifact_stem}_metric.csv"
    image_path = analysis_directory / f"{artifact_stem}_phase_diagram.png"
    np.savez(
        npz_path,
        dataset_name=dataset_name,
        parameter_label=descriptive_name,
        alpha_values=alpha_values,
        drive_frequency_hz_values=frequency_values,
        metric_matrix=metric,
        **{f"{key}_matrix": value for key, value in statistics_matrices.items()},
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    alpha_mesh, frequency_mesh = np.meshgrid(alpha_values, frequency_values)
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.pcolormesh(
        alpha_mesh, frequency_mesh, metric, shading="nearest", cmap=contract["colormap"]
    )
    figure.colorbar(image, ax=axis, label=colorbar_title)
    axis.set_xlabel(x_axis_title)
    axis.set_ylabel(y_axis_title)
    if contract["frequency_axis_inverted"]:
        axis.invert_yaxis()
    axis.set_title(plot_title or automatic_plot_title(grid))
    figure.tight_layout()
    figure.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved preserved analysis files to {analysis_directory}")
    print(f"  {image_path.name}")
    print(f"  {csv_path.name}")
    print(f"  {npz_path.name}")
    return {"npz": npz_path, "csv": csv_path, "image": image_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "optional dataset folder name, results directory, or run_manifest.json; "
            "the newest completed dataset is used when omitted"
        ),
    )
    parser.add_argument("--title", default=PLOT_TITLE)
    parser.add_argument("--x-axis-title", default=X_AXIS_TITLE)
    parser.add_argument("--y-axis-title", default=Y_AXIS_TITLE)
    parser.add_argument("--colorbar-title", default=COLORBAR_TITLE)
    arguments = parser.parse_args()
    manifest_path = resolve_manifest(arguments.dataset)
    print(f"Analyzing dataset: {manifest_path.parent.name}")
    analyze(
        manifest_path,
        plot_title=arguments.title,
        x_axis_title=arguments.x_axis_title,
        y_axis_title=arguments.y_axis_title,
        colorbar_title=arguments.colorbar_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
