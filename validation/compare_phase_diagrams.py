"""Codex CUDA Port: compare CPU and CUDA small-grid phase-diagram metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compare(cpu_npz: Path, cuda_npz: Path, report: Path) -> dict[str, float | str]:
    with np.load(cpu_npz) as cpu, np.load(cuda_npz) as cuda:
        for coordinate in ("alpha_values", "drive_frequency_hz_values"):
            if not np.array_equal(cpu[coordinate], cuda[coordinate]):
                raise ValueError(f"CPU/CUDA {coordinate} differ.")
        cpu_metric = cpu["metric_matrix"]
        cuda_metric = cuda["metric_matrix"]
    difference = cuda_metric - cpu_metric
    result = {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(cpu_metric), 1.0e-300)
        ),
        "cpu_npz": str(cpu_npz),
        "cuda_npz": str(cuda_npz),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_npz", type=Path)
    parser.add_argument("cuda_npz", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(compare(arguments.cpu_npz, arguments.cuda_npz, arguments.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

