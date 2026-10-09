"""Create a preserved phase-diagram analysis from CUDA datasets or CSV files.

By default the colour is the IPR (the `metric` column, mean discrete
sum_j |psi_j|^4) against alpha and the drive frequency. Any numeric column can
be chosen for either axis and the colour, with optional colour limits. Several
datasets or CSV files can be combined, for example scans at different green-wall
heights, to plot along a parameter that is fixed within one scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tables as tb

from cuda_workflow_common import RESULTS_DIRECTORY, phase_dataset_label, read_config
from phase_metric import (
    LEVEL_STATISTICS_FIELDS, STATISTICS_FIELDS, csv_fields, csv_row, level_statistics_columns,
    statistics_from_probabilities,
)


# ---------------------------------------------------------------------------
# PLOT CONFIGURATION. Leave DATASET_NAME as None to analyze the newest
# completed dataset, or paste one results_cuda folder name between the quotes.
# Set PLOT_TITLE to None to build it from lattice depth, initial depth, and phase.
# ---------------------------------------------------------------------------
DATASET_NAME: str | None = None
PLOT_TITLE: str | None = None
# List the available columns and ask which to plot; Enter keeps the values below.
ASK_FOR_VARIABLES = True
X_VARIABLE = "alpha"
Y_VARIABLE = "drive_frequency_hz"
COLOR_VARIABLE = "metric"        # the IPR, mean discrete sum_j |psi_j|^4
# None for automatic colour limits, or (minimum, maximum) such as (0.0, 0.8).
COLOR_LIMITS: tuple[float | None, float | None] | None = None
# None picks a title for the plotted column.
X_AXIS_TITLE: str | None = None
Y_AXIS_TITLE: str | None = None
COLORBAR_TITLE: str | None = None

COLUMN_TITLES = {
    "alpha": r"$\alpha$",
    "drive_frequency_hz": r"$\nu_{\mathrm{flo}}$ (Hz)",
    "metric": r"IPR: mean discrete $\sum_j |\psi_j|^4$",
    "metric_std": r"Standard deviation of $\sum_j |\psi_j|^4$",
    "metric_variance": r"Variance of $\sum_j |\psi_j|^4$",
    "sigma_x": r"$\sigma_x$ (solver length units)",
    "sigma_x_squared": r"$\sigma_x^2$ (solver length units$^2$)",
    "x_mean": r"$\langle x \rangle$ (solver length units)",
    "mean_r": r"$\langle r \rangle$",
    "mean_r_squared": r"$\langle r^2 \rangle$",
    "eta": r"$\eta$ from $\langle r \rangle$",
    "eta_r_squared": r"$\eta$ from $\langle r^2 \rangle$",
    "lattice_depth_v0_er": r"Lattice depth $V_0$ ($E_R$)",
    "green_wall_height_er": r"Green wall height ($E_R$)",
    "green_wall_sigma_um": r"Green wall $\sigma$ ($\mu$m)",
    "green_wall_gap_um": r"Green wall gap ($\mu$m)",
}
# Scan-wide parameters added to every row, so that datasets can be combined
# and plotted along them. A scan without walls has wall height 0.
DATASET_COLUMNS = (
    "lattice_depth_v0_er", "initial_lattice_depth_v0_er", "phase_radians",
    "green_wall_height_er", "green_wall_sigma_um", "green_wall_gap_um",
)
# Inputs that can be held fixed when several rows share one plotted point.
PARAMETER_COLUMNS = ("alpha", "drive_frequency_hz") + DATASET_COLUMNS


class OverlappingPointsError(ValueError):
    """Several rows share one (x, y) point; another column must be fixed."""

    def __init__(self, message: str, columns: list[str]):
        super().__init__(message)
        self.columns = columns


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
    return phase_dataset_label(lattice_depth, initial_depth, phase, grid.get("green_walls"))


def automatic_plot_title(grid: dict) -> str:
    lattice_depth = float(grid["lattice_depth_v0_er"])
    initial_depth = float(
        grid.get("initial_lattice_depth_v0_er", 2.0 * lattice_depth)
    )
    phase = float(grid.get("phase_radians", 0.0))
    title = (
        f"Lattice Depth {lattice_depth:g} E_R * "
        f"Initial Depth {initial_depth:g} E_R * Phase {phase:g} rad"
    )
    walls = grid.get("green_walls")
    if walls:
        title += (f"\nGreen walls {walls['height_er']:g} E_R, sigma {walls['sigma_um']:g} um, "
                  f"gap {walls['gap_um']:g} um")
    return title


def dataset_columns(grid: dict) -> dict:
    depth = float(grid["lattice_depth_v0_er"])
    walls = grid.get("green_walls")
    return {
        "lattice_depth_v0_er": depth,
        "initial_lattice_depth_v0_er": float(grid.get("initial_lattice_depth_v0_er", 2.0 * depth)),
        "phase_radians": float(grid.get("phase_radians", 0.0)),
        "green_wall_height_er": float(walls["height_er"]) if walls else 0.0,
        "green_wall_sigma_um": float(walls["sigma_um"]) if walls else None,
        "green_wall_gap_um": float(walls["gap_um"]) if walls else None,
    }


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


def dataset_analysis(manifest_path: Path) -> dict:
    """Validated rows and (frequency x alpha) matrices of one completed dataset."""
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
    levels = bool(contract.get("level_statistics"))
    matrix_fields = STATISTICS_FIELDS + (LEVEL_STATISTICS_FIELDS if levels else ())
    statistics_matrices = {key: np.full_like(metric, np.nan) for key in matrix_fields}

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
    scan_columns = dataset_columns(grid)
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
            if levels:
                row.update(level_statistics_columns(run))
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
        for key in matrix_fields:
            if row[key] is not None:
                statistics_matrices[key][frequency_index, alpha_index] = row[key]
        row.update(scan_columns)
        rows.append(row)
        print(
            f"out_{run['run_index']:03d}: alpha={run['alpha']:.10g}, "
            f"frequency={run['drive_frequency_hz']:.10g} Hz, metric={mean_metric:.10g}"
        )
    return {
        "manifest": manifest, "root": root, "rows": rows,
        "fields": csv_fields(contract) + DATASET_COLUMNS,
        "alpha_values": alpha_values, "frequency_values": frequency_values,
        "metric": metric, "matrices": statistics_matrices,
    }


# ---------------------------------------------------------------------------
# Plotting any two numeric columns
# ---------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict]:
    """Rows of a results CSV; blank cells become None and numbers floats."""
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, text in raw.items():
                text = (text or "").strip()
                try:
                    row[key] = float(text) if text else None
                except ValueError:
                    row[key] = text
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def numeric_columns(rows: list[dict]) -> list[str]:
    """Columns with at least one value and only numbers, in first-seen order."""
    names: list[str] = []
    for row in rows:
        names.extend(key for key in row if key not in names)

    def is_number(value) -> bool:
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)

    return [name for name in names
            if any(row.get(name) is not None for row in rows)
            and all(row.get(name) is None or is_number(row[name]) for row in rows)]


def distinct_values(rows: list[dict], column: str) -> list[float]:
    return sorted({float(row[column]) for row in rows if row.get(column) is not None})


def select_rows(rows: list[dict], where: dict[str, float]) -> list[dict]:
    """Rows whose column equals each fixed value (to a relative 1e-9)."""
    for column, value in where.items():
        available = distinct_values(rows, column)
        if not available:
            raise ValueError(f"No rows have a value for {column}.")
        rows = [row for row in rows if row.get(column) is not None
                and np.isclose(float(row[column]), value, rtol=1e-9, atol=0.0)]
        if not rows:
            raise ValueError(f"No rows have {column} = {value:g}; available: "
                             + ", ".join(f"{item:.10g}" for item in available))
    return rows


def grid_from_rows(rows: list[dict], x: str, y: str, color: str):
    """Sorted unique x and y values and the colour on that grid (NaN where missing)."""
    available = numeric_columns(rows)
    for name in (x, y, color):
        if name not in available:
            raise ValueError(f"'{name}' is not a numeric column here. Available: "
                             + ", ".join(available))
    points: dict[tuple[float, float], float] = {}
    overlapping = False
    for row in rows:
        if row.get(x) is None or row.get(y) is None:
            continue
        key = (float(row[x]), float(row[y]))
        overlapping |= key in points
        points[key] = np.nan if row.get(color) is None else float(row[color])
    if overlapping:
        varying = [name for name in available if name in PARAMETER_COLUMNS
                   and name not in (x, y) and len(distinct_values(rows, name)) > 1]
        raise OverlappingPointsError(
            f"Several rows share the same ({x}, {y}). Fix one of these columns with "
            f"--where COLUMN=VALUE: " + ", ".join(varying), varying)
    if not points:
        raise ValueError(f"No rows have both {x} and {y}.")
    xs = np.array(sorted({key[0] for key in points}))
    ys = np.array(sorted({key[1] for key in points}))
    values = np.full((ys.size, xs.size), np.nan)
    for (x_value, y_value), value in points.items():
        values[np.searchsorted(ys, y_value), np.searchsorted(xs, x_value)] = value
    return xs, ys, values


def plot_rows(rows: list[dict], path: Path, *, x: str, y: str, color: str,
              color_limits=None, title: str = "", x_title: str | None = None,
              y_title: str | None = None, color_title: str | None = None,
              colormap: str = "inferno", invert_frequency: bool = True) -> Path:
    xs, ys, values = grid_from_rows(rows, x, y, color)
    minimum, maximum = color_limits if color_limits is not None else (None, None)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.pcolormesh(x_mesh, y_mesh, values, shading="nearest", cmap=colormap,
                            vmin=minimum, vmax=maximum)
    figure.colorbar(image, ax=axis, label=color_title or COLUMN_TITLES.get(color, color))
    axis.set_xlabel(x_title or COLUMN_TITLES.get(x, x))
    axis.set_ylabel(y_title or COLUMN_TITLES.get(y, y))
    # The historical diagrams show the drive frequency decreasing upwards.
    if invert_frequency and y == "drive_frequency_hz":
        axis.invert_yaxis()
    if invert_frequency and x == "drive_frequency_hz":
        axis.invert_xaxis()
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def ask_plot_settings(rows: list[dict], x: str, y: str, color: str, color_limits,
                      where: dict[str, float], ask=input):
    """Ask in the terminal for the plotted columns, colour limits and fixed columns."""
    columns = numeric_columns(rows)
    inputs = [name for name in columns if name in PARAMETER_COLUMNS]
    results = [name + (" (IPR)" if name == "metric" else "") for name in columns
               if name not in PARAMETER_COLUMNS and name != "run_index"]
    print("\nColumns available for plotting")
    print("  inputs:  " + ", ".join(inputs))
    print("  results: " + ", ".join(results))

    def choose(label: str, default: str, options: list[str]) -> str:
        while True:
            answer = ask(f"{label} [{default}]: ").strip() or default
            if answer in options:
                return answer
            print(f"  '{answer}' is not one of: " + ", ".join(options))

    x = choose("x axis", x, columns)
    y = choose("y axis", y, columns)
    color = choose("colour", color, columns)
    shown = "auto" if color_limits is None else " ".join(
        "auto" if value is None else f"{value:g}" for value in color_limits)
    while True:
        answer = ask(f"colour limits as 'min max', or 'auto' [{shown}]: ").strip().lower()
        if not answer:
            break
        if answer == "auto":
            color_limits = None
            break
        parts = answer.replace(",", " ").split()
        try:
            if len(parts) != 2:
                raise ValueError
            color_limits = tuple(None if part == "auto" else float(part) for part in parts)
            break
        except ValueError:
            print("  Enter two numbers, e.g. 0 0.8, or 'auto'.")
    where = dict(where)
    while True:
        try:
            grid_from_rows(select_rows(rows, where), x, y, color)
            return x, y, color, color_limits, where
        except OverlappingPointsError as error:
            if not error.columns:
                raise
            print(f"\nSeveral rows share each ({x}, {y}).")
            column = choose("column to fix", error.columns[0], error.columns)
            available = distinct_values(select_rows(rows, where), column)
            print("  values: " + ", ".join(f"{value:.10g}" for value in available))
            while True:
                try:
                    value = float(ask(f"{column} = ").strip())
                    select_rows(rows, {**where, column: value})
                    where[column] = value
                    break
                except ValueError:
                    print("  Enter one of the values listed.")


def write_dataset_analysis(manifest_path: Path, data: dict, *, x: str = "alpha",
                           y: str = "drive_frequency_hz", color: str = "metric",
                           color_limits=None, where: dict[str, float] | None = None,
                           plot_title: str | None = None, x_axis_title: str | None = None,
                           y_axis_title: str | None = None,
                           colorbar_title: str | None = None) -> dict[str, Path]:
    """Write the preserved NPZ, CSV and plot of one dataset."""
    manifest, root = data["manifest"], data["root"]
    grid, contract = manifest["parameter_grid"], manifest["analysis_contract"]
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
    image_path = analysis_directory / (f"{artifact_stem}_phase_diagram.png" if color == "metric"
                                       else f"{artifact_stem}_{color}.png")
    np.savez(
        npz_path,
        dataset_name=dataset_name,
        parameter_label=descriptive_name,
        alpha_values=data["alpha_values"],
        drive_frequency_hz_values=data["frequency_values"],
        metric_matrix=data["metric"],
        **{f"{key}_matrix": value for key, value in data["matrices"].items()},
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=data["fields"])
        writer.writeheader()
        writer.writerows(data["rows"])
    title = plot_title or automatic_plot_title(grid)
    if where:
        title += "\n" + ", ".join(f"{column} = {value:g}" for column, value in where.items())
    plot_rows(select_rows(data["rows"], where or {}), image_path, x=x, y=y, color=color,
              color_limits=color_limits, title=title, x_title=x_axis_title,
              y_title=y_axis_title, color_title=colorbar_title, colormap=contract["colormap"],
              invert_frequency=bool(contract.get("frequency_axis_inverted", True)))
    print(f"Saved preserved analysis files to {analysis_directory}")
    print(f"  {image_path.name}")
    print(f"  {csv_path.name}")
    print(f"  {npz_path.name}")
    return {"npz": npz_path, "csv": csv_path, "image": image_path}


def analyze(manifest_path: Path, **plot_settings) -> dict[str, Path]:
    """Analyze one completed dataset; plot settings as for write_dataset_analysis."""
    return write_dataset_analysis(manifest_path, dataset_analysis(manifest_path), **plot_settings)


def parse_where(items: list[str]) -> dict[str, float]:
    where = {}
    for item in items:
        column, separator, value = item.partition("=")
        try:
            if not separator:
                raise ValueError
            where[column.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--where needs COLUMN=NUMBER, not {item!r}") from None
    return where


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "dataset folder names, results directories, run_manifest.json files or "
            "CSV files; several are combined. The newest completed dataset is used "
            "when omitted."
        ),
    )
    parser.add_argument("--x", default=X_VARIABLE, help="column for the x axis")
    parser.add_argument("--y", default=Y_VARIABLE, help="column for the y axis")
    parser.add_argument("--color", default=COLOR_VARIABLE,
                        help="column for the colour (metric is the IPR)")
    parser.add_argument("--vmin", type=float, help="lower colour limit (default automatic)")
    parser.add_argument("--vmax", type=float, help="upper colour limit (default automatic)")
    parser.add_argument("--where", action="append", default=[], metavar="COLUMN=VALUE",
                        help="keep only rows with this value, e.g. --where drive_frequency_hz=3e6")
    parser.add_argument("--ask", action=argparse.BooleanOptionalAction, default=ASK_FOR_VARIABLES,
                        help="ask for the columns and colour limits in the terminal")
    parser.add_argument("--title", default=PLOT_TITLE)
    parser.add_argument("--x-axis-title", default=X_AXIS_TITLE)
    parser.add_argument("--y-axis-title", default=Y_AXIS_TITLE)
    parser.add_argument("--colorbar-title", default=COLORBAR_TITLE)
    arguments = parser.parse_args()

    limits = COLOR_LIMITS
    if arguments.vmin is not None or arguments.vmax is not None:
        limits = (arguments.vmin, arguments.vmax)
    where = parse_where(arguments.where)

    datasets, tables, rows = [], [], []
    for item in arguments.inputs or [None]:
        if item is not None and item.lower().endswith(".csv") and Path(item).is_file():
            path = Path(item).resolve()
            print(f"Reading CSV: {path}")
            table = read_csv_rows(path)
            for row in table:
                row.setdefault("dataset", path.name)
            tables.append(path)
            rows.extend(table)
        else:
            manifest_path = resolve_manifest(item)
            print(f"Analyzing dataset: {manifest_path.parent.name}")
            data = dataset_analysis(manifest_path)
            datasets.append((manifest_path, data))
            rows.extend(dict(row, dataset=manifest_path.parent.name) for row in data["rows"])

    x, y, color = arguments.x, arguments.y, arguments.color
    if arguments.ask and sys.stdin.isatty():
        try:
            x, y, color, limits, where = ask_plot_settings(rows, x, y, color, limits, where)
        except EOFError:
            print("\nNo answer; using the configured settings.")
    titles = {"x_axis_title": arguments.x_axis_title, "y_axis_title": arguments.y_axis_title,
              "colorbar_title": arguments.colorbar_title}

    if len(datasets) == 1 and not tables:
        manifest_path, data = datasets[0]
        write_dataset_analysis(manifest_path, data, x=x, y=y, color=color, color_limits=limits,
                               where=where, plot_title=arguments.title, **titles)
        return 0

    # Several inputs: one combined plot, the rows it was drawn from and their sources.
    output = RESULTS_DIRECTORY / "combined_plots"
    output.mkdir(parents=True, exist_ok=True)
    stem = "Combined_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selected = select_rows(rows, where)
    columns: list[str] = []
    for row in selected:
        columns.extend(key for key in row if key not in columns)
    csv_path = output / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)
    sources = [path.parent.name for path, _ in datasets] + [path.name for path in tables]
    title = arguments.title or f"{len(sources)} datasets combined"
    if where:
        title += "\n" + ", ".join(f"{column} = {value:g}" for column, value in where.items())
    image = plot_rows(selected, output / f"{stem}.png", x=x, y=y, color=color,
                      color_limits=limits, title=title, x_title=titles["x_axis_title"],
                      y_title=titles["y_axis_title"], color_title=titles["colorbar_title"])
    (output / f"{stem}_sources.json").write_text(json.dumps(
        {"sources": sources, "x": x, "y": y, "color": color, "color_limits": limits,
         "where": where}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved combined plot to {output}")
    print(f"  {image.name}")
    print(f"  {csv_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
