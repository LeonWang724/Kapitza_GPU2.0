"""Codex CUDA Port: shared, non-numerical workflow and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PORT_ROOT = SCRIPT_DIRECTORY.parents[1]
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "results_cuda"


def grid_points(manifest: dict):
    """Yield global indices and parameters for this runner's assigned range."""
    grid = manifest["parameter_grid"]
    frequencies = grid["drive_frequency_hz_values"]
    total = len(grid["alpha_values"]) * len(frequencies)
    assignment = manifest.get("work_partition", {"start": 0, "stop": total})
    start, stop = assignment["start"], assignment["stop"]
    if not (0 <= start < stop <= total):
        raise ValueError("Invalid or empty grid-point assignment.")
    for index in range(start, stop):
        alpha_index, frequency_index = divmod(index, len(frequencies))
        yield (index, alpha_index, frequency_index,
               grid["alpha_values"][alpha_index], frequencies[frequency_index])


def experiment_drive_phase(alpha: float, inverted: bool) -> float:
    """Phase phi of the 5.0 drive sin(omega t + phi), chosen as in the experiment.

    Non-inverted: 0. Inverted: arcsin(-2/alpha), which makes
    1 + alpha sin(phi) = -1, so the lattice at t = 0 is flipped and the atoms
    start on top of the time-averaged lattice. Here alpha is the experiment's
    alpha*beta; the inverted case needs alpha >= 2.
    """
    if not isinstance(inverted, bool):
        raise ValueError("INVERTED must be True or False.")
    if not inverted:
        return 0.0
    alpha = float(alpha)
    if not alpha >= 2.0:
        raise ValueError(f"Inverted loading needs every alpha >= 2; got alpha = {alpha:g}.")
    return math.asin(-2.0 / alpha)


def point_drive_phase(grid: dict, alpha: float) -> float | None:
    """The 5.0 drive phase of a grid point, or None for a 4.0 grid."""
    if "inverted" not in grid:
        return None
    return experiment_drive_phase(alpha, grid["inverted"])


def generate_point_inputs(
    base_config: Path, config_path: Path, input_directory: Path,
    grid: dict, alpha: float, frequency: float, create_initial_state,
) -> dict[str, Path]:
    """Generate inputs and frequency settings without writing shared project files."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    input_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config, config_path)
    # Older manifests have no green walls; generators then see no new keyword.
    walls = grid.get("green_walls")
    extra = {"green_walls": walls} if walls else {}
    # 5.0 grids record "inverted" and use the experiment's drive phase with a
    # fixed lattice; 4.0 grids record the lattice's position phase instead.
    drive_phase = point_drive_phase(grid, alpha)
    if drive_phase is not None:
        extra["drive_phase"] = drive_phase
    create_initial_state(
        grid["lattice_depth_v0_er"], float(alpha), float(frequency),
        grid.get("phase_radians", 0.0), grid["initial_lattice_depth_v0_er"],
        output_directory=input_directory, config_path=config_path, **extra,
    )
    inputs = {name: input_directory / name
              for name in ("lattice_gauss.h5", "vstatic.h5", "vflo.h5")}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Generated input was not found: {path}")
    return inputs


def parameter_token(value: float) -> str:
    """Return a compact, Windows-filename-safe representation of a parameter."""
    return f"{float(value):.10g}".replace("-", "m").replace("+", "").replace(".", "p")


def phase_dataset_label(
    lattice_depth_v0_er: float,
    initial_lattice_depth_v0_er: float,
    phase_radians: float,
    green_walls: dict | None = None,
    *,
    inverted: bool | None = None,
) -> str:
    """Build the shared descriptive label used by datasets and analysis files.

    5.0 scans pass `inverted` and are labelled Inverted or NonInverted; 4.0
    scans keep the position phase in the label.
    """
    if inverted is None:
        start = f"Phase_{parameter_token(phase_radians)}rad"
    else:
        start = "Inverted" if inverted else "NonInverted"
    label = (
        f"LatticeDepth_{parameter_token(lattice_depth_v0_er)}ER_"
        f"InitialDepth_{parameter_token(initial_lattice_depth_v0_er)}ER_"
        f"{start}"
    )
    if green_walls:
        label += (
            f"_Walls_H{parameter_token(green_walls['height_er'])}ER_"
            f"S{parameter_token(green_walls['sigma_um'])}um_"
            f"G{parameter_token(green_walls['gap_um'])}um"
        )
    return label


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_executable(explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    environment_value = os.environ.get("GPE1D_CUDA_EXE")
    if environment_value:
        candidates.append(Path(environment_value))
    candidates.extend(
        [
            PORT_ROOT / "build" / "bin" / "gpe1d_cuda.exe",
            SCRIPT_DIRECTORY / "gpe1d_cuda.exe",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    rendered = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"gpe1d_cuda.exe was not found. Checked:\n{rendered}")


def legacy_input_path(filename: str) -> Path:
    candidates = [SCRIPT_DIRECTORY / "in" / filename, SCRIPT_DIRECTORY / f"in\\{filename}"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Generated input {filename!r} was not found.")


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def update_config(path: Path, updates: dict[str, Any]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    remaining = {key: str(value) for key, value in updates.items()}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")) or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines[index] = f"{key}={remaining.pop(key)}{newline}"
    if remaining:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        lines.extend(f"{key}={value}\n" for key, value in remaining.items())
    path.write_text("".join(lines), encoding="utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def query_json(executable: Path, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), *arguments],
        cwd=SCRIPT_DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def git_provenance(directory: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=directory, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {"commit": commit, "dirty": status not in {"", "unavailable"}}


def run_logged(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            # Stop the child before compact-mode cleanup can remove its inputs.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
