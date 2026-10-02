"""Codex CUDA Port: generate identical HDF5 inputs and configs for validation."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


VALIDATION_DIRECTORY = Path(__file__).resolve().parent
PORT_ROOT = VALIDATION_DIRECTORY.parent
SIMULATION_CORE = PORT_ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(SIMULATION_CORE))

from create_initial_state_function import create_init_state  # noqa: E402
from cuda_workflow_common import (  # noqa: E402
    legacy_input_path,
    read_config,
    sha256_file,
    update_config,
)


def load_specification(path: Path | None = None) -> dict[str, Any]:
    source = path or VALIDATION_DIRECTORY / "validation_cases.json"
    return json.loads(source.read_text(encoding="utf-8"))


def generate_case(
    case: dict[str, Any], defaults: dict[str, Any], destination: Path
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / "inputs"
    inputs.mkdir()
    (SIMULATION_CORE / "in").mkdir(exist_ok=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(SIMULATION_CORE)
        create_init_state(20.0, float(case["alpha"]), float(case["frequency_hz"]), 0.0)
    finally:
        os.chdir(previous_cwd)

    copied = {}
    for filename in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5"):
        target = inputs / filename
        shutil.copy2(legacy_input_path(filename), target)
        copied[filename] = target

    config_path = destination / "base.config"
    shutil.copy2(SIMULATION_CORE / "gpe1d.config", config_path)
    generated = read_config(config_path)
    updates = {
        "number_of_iterations": case.get("iterations", defaults["iterations"]),
        "save_every_nth_iteration": case.get("save_every", defaults["save_every"]),
        "show_stats_every_nth_iteration": case.get("show_every", defaults["show_every"]),
        "initial_state_file": copied["lattice_gauss.h5"],
        "potential_file": copied["vstatic.h5"],
        "floquet_potential_file": copied["vflo.h5"],
        "floquet_potential": str(bool(case["floquet_enabled"])).lower(),
    }
    if "step_x" in case:
        updates["step_x"] = case["step_x"]
    update_config(config_path, updates)
    return {
        "name": case["name"],
        "alpha": float(case["alpha"]),
        "frequency_hz": float(case["frequency_hz"]),
        "floquet_mode": case["floquet_mode"],
        "floquet_enabled": bool(case["floquet_enabled"]),
        "floquet_omega_dimensionless": float(generated["floquet_omega"]),
        "config": config_path,
        "config_sha256": sha256_file(config_path),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in copied.items()
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--specification", type=Path)
    arguments = parser.parse_args()
    specification = load_specification(arguments.specification)
    arguments.destination.mkdir(parents=True, exist_ok=False)
    records = []
    for case in specification["cases"]:
        records.append(
            generate_case(
                case,
                specification["defaults"],
                arguments.destination / case["name"],
            )
        )
    (arguments.destination / "generated_cases.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
