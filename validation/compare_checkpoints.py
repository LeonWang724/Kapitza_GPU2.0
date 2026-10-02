"""Codex CUDA Port: phase-aligned checkpoint and status comparison report."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import tables as tb


ENERGY_COLUMNS = ("Etot", "Ekin", "Epot", "Eint", "Chpot")


def iteration(path: Path) -> int:
    match = re.fullmatch(r"(\d{16})\.h5", path.name)
    return int(match.group(1)) if match else -1


def read_wavefunction(path: Path) -> np.ndarray:
    with tb.open_file(path, "r") as handle:
        return handle.root.REAL[:] + 1.0j * handle.root.IMAGINARY[:]


def read_status(path: Path) -> dict[int, dict[str, float]]:
    result = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if not row.get("Iteration"):
                continue
            result[int(float(row["Iteration"]))] = {
                key: float(value)
                for key, value in row.items()
                if key and value not in {None, ""}
            }
    return result


def safe_relative(numerator: float, denominator: float) -> float:
    return abs(numerator) / max(abs(denominator), 1.0e-300)


def compare(
    cpu_directory: Path,
    cuda_directory: Path,
    cpu_status_path: Path,
    cuda_status_path: Path,
    step_x: float,
    report_csv: Path,
) -> dict[str, Any]:
    cpu_files = {iteration(path): path for path in cpu_directory.glob("*.h5") if iteration(path) >= 0}
    cuda_files = {
        iteration(path): path for path in cuda_directory.glob("*.h5") if iteration(path) >= 0
    }
    common = sorted(set(cpu_files).intersection(cuda_files))
    if not common:
        raise FileNotFoundError("No common CPU/CUDA checkpoints were found.")
    cpu_status = read_status(cpu_status_path)
    cuda_status = read_status(cuda_status_path)

    rows = []
    for checkpoint in common:
        cpu = read_wavefunction(cpu_files[checkpoint])
        cuda = read_wavefunction(cuda_files[checkpoint])
        if cpu.shape != cuda.shape:
            raise ValueError(f"Shape mismatch at iteration {checkpoint}")
        inner = np.vdot(cpu, cuda)
        phase = inner / abs(inner) if abs(inner) > 0.0 else 1.0 + 0.0j
        aligned = cuda * np.conjugate(phase)
        difference = aligned - cpu
        cpu_density = np.abs(cpu) ** 2
        cuda_density = np.abs(cuda) ** 2
        row: dict[str, Any] = {
            "iteration": checkpoint,
            "global_phase_radians": float(np.angle(phase)),
            "wavefunction_absolute_l2": float(np.linalg.norm(difference)),
            "wavefunction_relative_l2": float(
                np.linalg.norm(difference) / max(np.linalg.norm(cpu), 1.0e-300)
            ),
            "density_max_absolute": float(np.max(np.abs(cuda_density - cpu_density))),
            "density_relative_l2": float(
                np.linalg.norm(cuda_density - cpu_density)
                / max(np.linalg.norm(cpu_density), 1.0e-300)
            ),
            "norm_cpu": float(step_x * np.sum(cpu_density)),
            "norm_cuda": float(step_x * np.sum(cuda_density)),
        }
        row["norm_absolute_error"] = abs(row["norm_cuda"] - row["norm_cpu"])
        if checkpoint in cpu_status and checkpoint in cuda_status:
            for column in ENERGY_COLUMNS:
                cpu_value = cpu_status[checkpoint][column]
                cuda_value = cuda_status[checkpoint][column]
                row[f"{column}_cpu"] = cpu_value
                row[f"{column}_cuda"] = cuda_value
                row[f"{column}_absolute_error"] = abs(cuda_value - cpu_value)
                row[f"{column}_relative_error"] = safe_relative(
                    cuda_value - cpu_value, cpu_value
                )
        rows.append(row)

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def finite_max(key: str) -> float:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        return max(values) if values else math.nan

    summary = {
        "checkpoint_count": len(rows),
        "first_iteration": common[0],
        "last_iteration": common[-1],
        "maximum_wavefunction_relative_l2": finite_max("wavefunction_relative_l2"),
        "maximum_density_absolute_error": finite_max("density_max_absolute"),
        "maximum_norm_absolute_error": finite_max("norm_absolute_error"),
        "maximum_energy_relative_errors": {
            column: finite_max(f"{column}_relative_error") for column in ENERGY_COLUMNS
        },
        "csv_report": str(report_csv),
    }
    report_csv.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_directory", type=Path)
    parser.add_argument("cuda_directory", type=Path)
    parser.add_argument("cpu_status", type=Path)
    parser.add_argument("cuda_status", type=Path)
    parser.add_argument("--step-x", type=float, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    summary = compare(
        arguments.cpu_directory,
        arguments.cuda_directory,
        arguments.cpu_status,
        arguments.cuda_status,
        arguments.step_x,
        arguments.report,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

