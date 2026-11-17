"""Codex CUDA Port: shared, non-numerical workflow and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PORT_ROOT = SCRIPT_DIRECTORY.parents[1]
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "results_cuda"


def quasimomentum_values(grid: dict) -> list[float]:
    """10.0: the starting quasimomenta every grid point runs at (hbar kL).

    A 10.0 spread scan lists them in quasimomentum_values_kl; every other scan
    runs its single initial_quasimomentum_kl (0 before 7.0)."""
    values = grid.get("quasimomentum_values_kl")
    if values:
        return [float(value) for value in values]
    return [float(grid.get("initial_quasimomentum_kl", 0.0))]


def point_count(grid: dict) -> int:
    """Number of GPU runs of a scan: alpha x frequency (x quasimomentum in 10.0)."""
    return (len(grid["alpha_values"]) * len(grid["drive_frequency_hz_values"])
            * len(quasimomentum_values(grid)))


def point_coordinates(grid: dict, index: int) -> tuple[int, int, int]:
    """(alpha index, frequency index, quasimomentum index) of a run index.

    Alpha is outermost and the quasimomentum innermost, so a single-q scan
    keeps the indices of 4.0-9.0 and the q runs of one point are consecutive."""
    count = len(quasimomentum_values(grid))
    cell, quasimomentum_index = divmod(index, count)
    alpha_index, frequency_index = divmod(cell, len(grid["drive_frequency_hz_values"]))
    return alpha_index, frequency_index, quasimomentum_index


def point_grid(grid: dict, quasimomentum: float) -> dict:
    """The grid as one run sees it: its own starting quasimomentum."""
    return dict(grid, initial_quasimomentum_kl=float(quasimomentum))


def grid_points(manifest: dict):
    """Yield (index, alpha index, frequency index, alpha, frequency, q) for this
    runner's assigned range."""
    grid = manifest["parameter_grid"]
    frequencies = grid["drive_frequency_hz_values"]
    quasimomenta = quasimomentum_values(grid)
    total = point_count(grid)
    assignment = manifest.get("work_partition", {"start": 0, "stop": total})
    start, stop = assignment["start"], assignment["stop"]
    if not (0 <= start < stop <= total):
        raise ValueError("Invalid or empty grid-point assignment.")
    for index in range(start, stop):
        alpha_index, frequency_index, quasimomentum_index = point_coordinates(grid, index)
        yield (index, alpha_index, frequency_index, grid["alpha_values"][alpha_index],
               frequencies[frequency_index], quasimomenta[quasimomentum_index])


DRIVE_PHASE_CONVENTIONS = ("theory", "hardware")


def experiment_drive_phase(alpha: float, inverted: bool, convention: str = "theory",
                           depth_er: float | None = None,
                           initial_depth_er: float | None = None) -> float:
    """Phase phi of the 5.0 drive sin(omega t + phi), chosen as in the experiment.

    "theory" (5.0-10.0, the lab's kp_PhaseDiagramRealSpace.m): non-inverted 0;
    inverted arcsin(-2/alpha), so 1 + alpha sin(phi) = -1 and the lattice at
    t = 0 is the flipped V0 lattice; needs alpha >= 2.
    "hardware" (11.0, KpPredistortion.getPhaseTarget): the drive starts at the
    loading lattice, V0 [1 + alpha sin(phi)] = +-V_init:
    phi = arcsin((+-V_init / V0 - 1) / alpha), minus when inverted; needs
    alpha >= |+-V_init / V0 - 1| (3 for 20 E_R loading into 10 E_R, inverted).
    Here alpha is the experiment's alpha*beta.
    """
    if not isinstance(inverted, bool):
        raise ValueError("INVERTED must be True or False.")
    if convention not in DRIVE_PHASE_CONVENTIONS:
        raise ValueError(f"DRIVE_PHASE_CONVENTION must be one of {DRIVE_PHASE_CONVENTIONS}.")
    alpha = float(alpha)
    if convention == "theory":
        if not inverted:
            return 0.0
        if not alpha >= 2.0:
            raise ValueError(f"Inverted loading needs every alpha >= 2; got alpha = {alpha:g}.")
        return math.asin(-2.0 / alpha)
    if depth_er is None or initial_depth_er is None or not float(depth_er) > 0.0 \
            or not float(initial_depth_er) > 0.0:
        raise ValueError("The hardware drive phase needs positive lattice and loading depths.")
    target = (-1.0 if inverted else 1.0) * float(initial_depth_er) / float(depth_er) - 1.0
    if not alpha >= abs(target):
        raise ValueError(f"The hardware drive phase needs every alpha >= {abs(target):g} "
                         f"(V_init = {float(initial_depth_er):g} E_R, V0 = {float(depth_er):g} E_R"
                         f"{', inverted' if inverted else ''}); got alpha = {alpha:g}.")
    return math.asin(target / alpha)


def point_drive_phase(grid: dict, alpha: float) -> float | None:
    """The 5.0 drive phase of a grid point, or None for a 4.0 grid."""
    if "inverted" not in grid:
        return None
    return experiment_drive_phase(alpha, grid["inverted"],
                                  grid.get("drive_phase_convention", "theory"),
                                  grid.get("lattice_depth_v0_er"),
                                  grid.get("initial_lattice_depth_v0_er"))


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
    # 7.0 grids record the starting quasimomentum; older generators see no keyword.
    quasimomentum = float(grid.get("initial_quasimomentum_kl", 0.0))
    if quasimomentum != 0.0:
        extra["quasimomentum"] = quasimomentum
    # 8.0 grids record the box and the starting cloud width.
    if grid.get("grid_points") is not None:
        extra["grid_points"] = int(grid["grid_points"])
    if grid.get("initial_cloud_width_um") is not None:
        extra["envelope_sigma_m"] = float(grid["initial_cloud_width_um"]) * 1e-6
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
    # 12.0: drive for whole cycles, then hold the lattice until the image.
    hold = drive_hold(grid, alpha, frequency, read_config(config_path))
    if hold is not None:
        update_config(config_path, {"drive_iterations": hold["drive_iterations"],
                                    "hold_floquet_factor": repr(hold["hold_floquet_factor"])})
    return inputs


def drive_hold(grid: dict, alpha: float, frequency: float, config: dict) -> dict | None:
    """12.0: when and how a point's drive stops, or None without a lattice hold.

    The drive lasts grid["hold_time_ms"], rounded to N = round(t f) whole
    cycles when grid["drive_whole_cycles"] (the lab's hw_KPModNCycle), so it
    ends where it started. From then until the end of the run (number_of_iterations,
    the drive plus grid["lattice_hold_ms"]) the Floquet factor stays constant:
    at its value where the drive ends (grid["lattice_hold_depth_er"] None), or
    so that the static lattice is -D cos^2(kL x) with D = lattice_hold_depth_er
    (the generated potential is -V0 [1 + alpha factor] cos^2(kL x)).
    """
    hold_ms = grid.get("lattice_hold_ms")
    if not hold_ms:
        return None
    from extended_metrics import solver_time_unit_s
    step = float(config["time_step"])
    step_s = step * solver_time_unit_s()
    drive_s = float(grid["hold_time_ms"]) * 1e-3
    cycles = None
    if grid.get("drive_whole_cycles", True) and frequency > 0:
        cycles = max(1, int(round(drive_s * float(frequency))))
        drive_s = cycles / float(frequency)
    total = int(float(config["number_of_iterations"]))
    stop = min(max(1, int(round(drive_s / step_s))), total)
    depth = grid.get("lattice_hold_depth_er")
    if depth is None:
        # The drive's value at its exact end (N / f), not at the time step
        # nearest to it: with whole cycles exactly where it started.
        factor = math.cos(float(config["floquet_omega"]) * (drive_s / solver_time_unit_s())
                          + float(config.get("floquet_phase", 0.0)))
    else:
        v0 = float(grid["lattice_depth_v0_er"])
        if alpha == 0:
            if float(depth) != v0:
                raise ValueError("Without a drive (alpha = 0) the hold lattice can only be V0.")
            factor = 0.0
        else:
            factor = (float(depth) / v0 - 1.0) / float(alpha)
    return {"drive_iterations": stop, "drive_cycles": cycles, "drive_time_s": stop * step_s,
            "hold_time_s": (total - stop) * step_s, "hold_floquet_factor": factor,
            "hold_lattice_er": float(grid["lattice_depth_v0_er"]) * (1.0 + float(alpha) * factor)}


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
    quasimomentum: float = 0.0,
    smoothing_um: float = 0.0,
    grid_points: int | None = None,
    cloud_width_um: float | None = None,
    hold_time_ms: float | None = None,
    quasimomentum_values: list[float] | None = None,
    drive_phase_convention: str | None = None,
    lattice_hold_ms: float | None = None,
    lattice_hold_depth_er: float | None = None,
) -> str:
    """Build the shared descriptive label used by datasets and analysis files.

    5.0 scans pass `inverted` and are labelled Inverted or NonInverted; 4.0
    scans keep the position phase in the label. 7.0 adds the starting
    quasimomentum and the density smoothing when they are not 0; 8.0 adds the
    box and the starting cloud width when they differ from 4.0-7.0's (65536
    points, 30 um) and the hold time when it is set. 10.0 spread scans name
    their range of simulated quasimomenta instead of a single q. 11.0 adds
    _PhaseHw when the drive starts at the loading lattice (hardware phase).
    12.0 adds _Hold<ms>ms for a lattice hold between the drive and the image,
    and _HoldD<E_R>ER when the hold lattice is set instead of where the drive ends.
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
    if quasimomentum_values and len(quasimomentum_values) > 1:
        label += (f"_Q{parameter_token(min(quasimomentum_values))}to"
                  f"{parameter_token(max(quasimomentum_values))}x{len(quasimomentum_values)}kL")
    elif quasimomentum:
        label += f"_Q{parameter_token(quasimomentum)}kL"
    if smoothing_um:
        label += f"_Smooth{parameter_token(smoothing_um)}um"
    if grid_points is not None and int(grid_points) != 65536:
        label += f"_Box{int(grid_points)}"
    if cloud_width_um is not None and float(cloud_width_um) != 30.0:
        label += f"_W{parameter_token(cloud_width_um)}um"
    if hold_time_ms is not None:
        label += f"_T{parameter_token(hold_time_ms)}ms"
    if drive_phase_convention == "hardware":
        label += "_PhaseHw"
    if lattice_hold_ms:
        label += f"_Hold{parameter_token(lattice_hold_ms)}ms"
        if lattice_hold_depth_er is not None:
            label += f"_HoldD{parameter_token(lattice_hold_depth_er)}ER"
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


# 12.0: gpe1d_cuda.exe and its HDF5 library open files through the classic
# Windows file API, which refuses a path longer than 259 characters (MAX_PATH),
# even on machines where Python itself handles long paths. None = no limit.
SOLVER_PATH_LIMIT: int | None = 259 if os.name == "nt" else None
SOLVER_PATH_KEYS = ("initial_state_file", "potential_file", "floquet_potential_file",
                    "status_file", "phase_metric_file", "camera_profile_file",
                    "extended_metrics_file")


def check_solver_paths(config_paths, cwd: Path, limit: int | None = None) -> None:
    """Refuse, before the solver starts, any file path it could not open."""
    limit = SOLVER_PATH_LIMIT if limit is None else limit
    if limit is None:
        return
    for config_path in config_paths:
        config_path = Path(config_path)
        config = read_config(config_path)
        named = [("config", str(config_path if config_path.is_absolute() else cwd / config_path))]
        named += [(key, config[key]) for key in SOLVER_PATH_KEYS if config.get(key)]
        if config.get("output_folder") and config.get("save_psi", "").lower() == "true":
            # Snapshots are written as <output_folder>/<16 digits>.h5.
            named.append(("snapshot", str(Path(config["output_folder"]) / ("0" * 16 + ".h5"))))
        for key, value in named:
            absolute = os.path.isabs(value) or PureWindowsPath(value).is_absolute()
            path = value if absolute else os.path.join(str(cwd), value)
            if len(path) > limit:
                raise RuntimeError(
                    f"The CUDA solver cannot open a path of {len(path)} characters on "
                    f"Windows (limit {limit}): {key} = {path}. Move the Kapitza folder "
                    "closer to the top of the drive or shorten the results folder name.")


def solver_failure_reason(log_path: Path) -> str | None:
    """The solver's own last 'gpe1d_cuda failed: ...' line from its log, if any."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    reasons = [line.strip() for line in lines if line.strip().startswith("gpe1d_cuda failed:")]
    return reasons[-1] if reasons else None


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
