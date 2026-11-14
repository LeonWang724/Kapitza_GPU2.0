"""Check the extended-metric sums of this GPU against NumPy.

Two GPU batches of two short points each, with full snapshots: one batch
non-inverted with the 7.0 settings (4 um density smoothing, starting
quasimomentum 0.1 kL) and one inverted with the 6.0 settings (no smoothing,
q = 0), both with the 5.0 drive. Points of different loading cannot share a
batch, because the lattice-site edges sit where the atoms start; a scan never
mixes them either. At every status iteration the
solver's extended sums must match an independent NumPy calculation on the
saved snapshot (extended_metrics.reference_extended_sums), to --rtol of the
summed magnitudes of each column's terms: the scale of the rounding error,
which stays finite where a sum cancels to zero (odd moments of the
symmetric cloud). Then the CSV summary is computed from each file.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tables as tb

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))
from create_initial_state_function import create_init_state  # noqa: E402
from cuda_workflow_common import (  # noqa: E402
    experiment_drive_phase, locate_executable, query_json, read_config, run_logged,
    update_config, write_json_atomic,
)
import extended_metrics  # noqa: E402
from phase_metric import smooth_density, smoothing_half_width  # noqa: E402

# One GPU batch per loading; every point of a batch shares the site offset.
# Settings per batch: (density smoothing in um, starting quasimomentum in kL).
SETTINGS = {"non_inverted": (4.0, 0.1), "inverted": (0.0, 0.0)}
BATCHES = {
    "non_inverted": ({"alpha": 4.0, "frequency_hz": 1.9e6},
                     {"alpha": 12.0, "frequency_hz": 0.8e6}),
    "inverted": ({"alpha": 30.0, "frequency_hz": 1.3e6},
                 {"alpha": 15.0, "frequency_hz": 0.6e6}),
}
DEPTH, INITIAL = 10.0, 20.0


def load(path: Path) -> np.ndarray:
    with tb.open_file(path, "r") as handle:
        return handle.root.REAL[:] + 1j * handle.root.IMAGINARY[:]


def point_label(loading: str, point: dict) -> str:
    return (f"{loading} alpha={point['alpha']:g}, "
            f"f={point['frequency_hz'] / 1e6:g} MHz")


def prepare(directory: Path, point: dict, inverted: bool, iterations: int,
            smoothing_um: float = 0.0, quasimomentum: float = 0.0) -> Path:
    directory.mkdir(parents=True)
    config = directory / "run.config"
    shutil.copy2(CORE / "gpe1d.config", config)
    phase = experiment_drive_phase(point["alpha"], inverted)
    create_init_state(DEPTH, point["alpha"], point["frequency_hz"], 0.0, INITIAL,
                      output_directory=directory / "in", config_path=config, drive_phase=phase,
                      quasimomentum=quasimomentum)
    (directory / "out").mkdir()
    grid = {"initial_lattice_depth_v0_er": INITIAL, "inverted": inverted}
    update_config(config, {
        "initial_state_file": directory / "in" / "lattice_gauss.h5",
        "potential_file": directory / "in" / "vstatic.h5",
        "floquet_potential_file": directory / "in" / "vflo.h5",
        "number_of_iterations": iterations, "save_every_nth_iteration": 100,
        "show_stats_every_nth_iteration": 100, "output_folder": directory / "out",
        "status_file": directory / "status.csv", "phase_metric_file": "",
        "phase_metric_cut_points_each_edge": 100, "save_psi": "true",
        "dp_save_potential": "false",
        **extended_metrics.solver_settings(grid, point["alpha"]),
        "extended_metrics_file": directory / "extended.csv",
    })
    if smoothing_um:
        update_config(config, {"density_smoothing_width":
                               repr(smoothing_um * 1e-6 / extended_metrics.GRID_SPACING_M)})
    return config


def compare(directory: Path, config: dict, rtol: float) -> float:
    """Largest difference between the solver's sums and NumPy, relative to each
    column's summed term magnitudes."""
    series = extended_metrics.read_series(directory / "extended.csv")
    initial = load(Path(config["initial_state_file"]))
    potential = load(Path(config["potential_file"]))
    step = float(config["step_x"])
    settings = dict(
        cut=int(config["phase_metric_cut_points_each_edge"]), step_x=step,
        lattice_wavevector=float(config["extended_metrics_lattice_wavevector"]),
        central_halfwidth=float(config["extended_metrics_central_halfwidth"]),
        site_offset=float(config["extended_metrics_site_offset"]),
        pixel_width=float(config["extended_metrics_pixel_width"]),
        smoothing_half_width=smoothing_half_width(
            float(config.get("density_smoothing_width", 0.0)), step))
    names = extended_metrics.SERIES_COLUMNS[2:]
    gpu, reference, magnitude = [], [], []
    for row, iteration in enumerate(series["iteration"].astype(int)):
        psi = load(directory / "out" / f"{iteration:016d}.h5")
        density = np.abs(psi) ** 2
        fft = np.fft.fft(psi)
        k2 = (2.0 * math.pi * np.fft.fftfreq(psi.size, d=step)) ** 2
        # Norm, kinetic sum and density overlap: every term is non-negative.
        # With 7.0 smoothing the overlap compares the smoothed densities.
        half = settings["smoothing_half_width"]
        fixed = [step * density.sum(), np.sum(k2 * np.abs(fft) ** 2),
                 np.sum(smooth_density(np.abs(initial) ** 2, half) * smooth_density(density, half))]
        reference.append(np.concatenate([fixed, extended_metrics.reference_extended_sums(
            psi, initial, potential, **settings)]))
        magnitude.append(np.concatenate([fixed, extended_metrics.reference_extended_sums(
            psi, initial, potential, **settings, absolute=True)]))
        gpu.append([series[name][row] for name in names])
    gpu, reference = np.array(gpu), np.array(reference)
    scale = np.maximum(np.max(np.array(magnitude), axis=0), 1e-300)
    difference = np.max(np.abs(gpu - reference), axis=0) / scale
    failing = [f"{name} ({value:.3g})" for name, value in zip(names, difference)
               if not value <= rtol]
    if failing:
        raise AssertionError(f"{directory.name}: differs from NumPy by more than {rtol:g} "
                             f"of the summed magnitudes in " + ", ".join(failing) + ".")
    return float(difference.max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--rtol", type=float, default=1e-9)
    arguments = parser.parse_args()
    executable = locate_executable(arguments.executable)
    build = query_json(executable, "--version-json")
    if build.get("extended_metrics_version") != 2 or build.get("density_smoothing_version") != 1:
        raise RuntimeError("Rebuild with BUILD_CUDA.bat before validating the extended metrics.")
    timestamp = datetime.now(timezone.utc).strftime("extended_%Y%m%dT%H%M%S%fZ")
    results = (arguments.results or ROOT / "validation" / "results" / timestamp).resolve()
    results.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "build": build, "rtol": arguments.rtol, "checks": []}
    with tempfile.TemporaryDirectory(prefix=".check_", dir=results) as temporary:
        work = Path(temporary)
        try:
            contract = {"final_snapshot_count": 3, "cut_points_each_edge": 100}
            for loading, points in BATCHES.items():
                inverted = loading == "inverted"
                smoothing_um, quasimomentum = SETTINGS[loading]
                configs = [prepare(work / f"{loading}_{index}", point, inverted,
                                   arguments.iterations, smoothing_um, quasimomentum)
                           for index, point in enumerate(points)]
                log = f"solver_{loading}.log"
                code = run_logged([str(executable), *map(str, configs), "--device",
                                   str(arguments.device), "--floquet-mode", "physical"],
                                  ROOT, work / log)
                if code != 0:
                    raise RuntimeError(f"Solver failed on the {loading} batch with exit code "
                                       f"{code}; see {log}")
                for point, path in zip(points, configs):
                    config = read_config(path)
                    worst = compare(path.parent, config, arguments.rtol)
                    summary = extended_metrics.gpu_summary(path.parent / "extended.csv",
                                                           config, contract)
                    check = {"name": f"{point_label(loading, point)}, smoothing {smoothing_um:g} um, "
                                     f"q = {quasimomentum:g} kL: extended sums == NumPy",
                             "status": "passed", "max_relative_difference": worst,
                             "pendulum_cos": summary["pendulum_cos"],
                             "site_ipr": summary["site_ipr"], "order_0": summary["order_0"]}
                    report["checks"].append(check)
                    write_json_atomic(results / "report.json", report)
                    print(f"PASS {check['name']} (max relative difference {worst:.2g}; "
                          f"pendulum_cos {summary['pendulum_cos']:+.3f})", flush=True)
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            shutil.copytree(work, results / "failure_details", dirs_exist_ok=True)
            write_json_atomic(results / "report.json", report)
            raise
    write_json_atomic(results / "report.json", report)
    print(f"Extended-metric checks passed. Report: {results / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
