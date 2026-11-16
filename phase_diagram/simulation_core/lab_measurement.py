"""11.0: measure a simulated cloud the way the lab's camera analysis does.

The GPU writes each run's final |psi|^2 binned into camera pixels
(src/camera_profile.h). This module places that profile on the lab camera
(Andor iXon 888 behind the 3.338x imaging, 3.895 um per pixel at the atoms, a
1024-pixel sensor with the cloud at row ~456) and measures it exactly as
MuscleMuseum's kpAnalysis.m does on the AdData crop (sensor rows 120-780):

  computeStDev           moments of the window's profile, 4 um per pixel
  computeStDevWithBgSub  minus the mean of the outer 10 % of the window, values
                         below noiseFloor x peak set to 0, then the moments

with camera noise added at the level measured in the lab's own images
(noise per pixel = noise_per_pixel_over_signal x the atoms in the window), and
averaged over many noise draws like the lab's averaged repeats.

Columns (metrics.csv):
  lab_stddev2_um           StdDev2 as the lab reports it (noise, window, 4 um/px)
  lab_stddev_um            StdDev as the lab reports it
  lab_stddev2_no_noise_um  StdDev2 of the same window without camera noise
  camera_width_um          true width of the cloud inside the window (real um)
  lab_window_fraction      share of the atoms (still in the box) inside the window
  camera_fov_fraction      share of the atoms on the camera sensor at all
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GRID_SPACING_UM = 0.023            # the solver's length unit
LAB_FIELDS = ("lab_stddev2_um", "lab_stddev_um", "lab_stddev2_no_noise_um", "camera_width_um",
              "lab_window_fraction", "camera_fov_fraction")

# The lab's camera and analysis (see the README for where each number comes from).
DEFAULT_CAMERA = {
    "pixel_um": 13.0 / 3.337676782237963,   # Andor iXon 888, 13 um pixels, magnification 3.338
    "sensor_pixels": 1024,
    "cloud_sensor_row": 455.5,              # kpAnalysis.m: "bec center here 456"
    "roi_sensor_rows": [120, 780],          # the AdData crop along the lattice (661 rows)
    "lab_um_per_pixel": 4.0,                # kpAnalysis.m passes 4e-6 m per pixel
    "edge_fraction": 0.10,                  # computeStDevWithBgSub's baseline
    "noise_floor": 0.0,                     # kpAnalysis.m noiseFloor
    "noise_per_pixel_over_signal": 3.78e-4, # trial 10366 (measure_lab_images.py)
    "noise_draws": 200,
    "window": "roi",                        # "roi" (the AdData crop) or "sensor" (whole field)
}


def validate_camera(settings: dict | None) -> dict:
    """The camera settings with defaults filled in, checked."""
    camera = dict(DEFAULT_CAMERA, **(settings or {}))
    unknown = set(camera) - set(DEFAULT_CAMERA)
    if unknown:
        raise ValueError(f"Unknown LAB_CAMERA settings: {sorted(unknown)}")
    rows = [int(value) for value in camera["roi_sensor_rows"]]
    sensor = int(camera["sensor_pixels"])
    if not (float(camera["pixel_um"]) > 0 and float(camera["lab_um_per_pixel"]) > 0
            and sensor > 1 and 0 <= rows[0] < rows[1] < sensor
            and 0 <= float(camera["cloud_sensor_row"]) < sensor
            and 0 < float(camera["edge_fraction"]) < 0.5
            and 0 <= float(camera["noise_floor"]) < 1
            and float(camera["noise_per_pixel_over_signal"]) >= 0
            and int(camera["noise_draws"]) >= 1 and camera["window"] in ("roi", "sensor")):
        raise ValueError("Invalid LAB_CAMERA settings.")
    camera.update(roi_sensor_rows=rows, sensor_pixels=sensor,
                  noise_draws=int(camera["noise_draws"]))
    camera["pixel_width_solver"] = float(camera["pixel_um"]) / GRID_SPACING_UM
    return camera


def camera_profile(density, step_x: float, pixel_width: float) -> tuple[int, np.ndarray]:
    """(first pixel index, mass per pixel): the same binning as camera_profile.h."""
    density = np.asarray(density, dtype=float)
    points = density.size
    x = (np.arange(points) - points // 2).astype(float) * step_x
    pixel = np.floor(x / pixel_width + 0.5).astype(np.int64)
    first = int(pixel[0])
    return first, np.bincount(pixel - first, weights=density)


def read_profile(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("camera_profile_version") != 1 or len(data["mass"]) != data["count"]:
        raise ValueError(f"Unexpected camera profile: {path}")
    return {"first_pixel": int(data["first_pixel"]), "pixel_width": float(data["pixel_width"]),
            "mass": [float(value) for value in data["mass"]]}


def _rows(profile: dict, camera: dict, rows: tuple[int, int]) -> np.ndarray:
    """The profile's mass on sensor rows rows[0]..rows[1] (zero where it has no pixels)."""
    centre = int(math.floor(float(camera["cloud_sensor_row"]) + 0.5))
    mass = np.asarray(profile["mass"], dtype=float)
    first = int(profile["first_pixel"])
    out = np.zeros(rows[1] - rows[0] + 1)
    for j, row in enumerate(range(rows[0], rows[1] + 1)):
        index = row - centre - first
        if 0 <= index < mass.size:
            out[j] = mass[index]
    return out


def lab_stddev(window: np.ndarray, um_per_pixel: float) -> np.ndarray:
    """kpAnalysis.m computeStDev of profiles (draws x pixels), in um."""
    window = np.atleast_2d(window)
    pos = um_per_pixel * np.arange(1, window.shape[1] + 1)
    weights = window / window.sum(axis=1, keepdims=True)
    mean = (pos * weights).sum(axis=1) / weights.sum(axis=1)
    variance = ((pos - mean[:, None]) ** 2 * weights).sum(axis=1) / weights.sum(axis=1)
    return np.sqrt(np.abs(variance))


def lab_stddev2(window: np.ndarray, um_per_pixel: float, edge_fraction: float,
                noise_floor: float) -> np.ndarray:
    """kpAnalysis.m computeStDevWithBgSub of profiles (draws x pixels), in um."""
    window = np.atleast_2d(np.asarray(window, dtype=float))
    n = window.shape[1]
    edge = max(1, round(n * edge_fraction))
    mask = np.zeros(n, dtype=bool)
    mask[:edge] = True
    mask[n - edge:] = True
    data = window - (window * mask).sum(axis=1, keepdims=True) / mask.sum()
    peak = data.max(axis=1, keepdims=True)
    data = np.where(data < noise_floor * peak, 0.0, data)
    mass = data.sum(axis=1)
    pos = um_per_pixel * np.arange(1, n + 1)
    mean = (pos * data).sum(axis=1) / mass
    variance = ((pos - mean[:, None]) ** 2 * data).sum(axis=1) / mass
    return np.sqrt(variance)


def measure(profile: dict, camera: dict, seed: int = 0) -> dict:
    """The LAB_FIELDS of one final camera profile."""
    camera = camera if "pixel_width_solver" in camera else validate_camera(camera)
    total = float(np.sum(profile["mass"]))
    sensor = _rows(profile, camera, (0, camera["sensor_pixels"] - 1))
    roi_rows = tuple(camera["roi_sensor_rows"])
    window = sensor if camera["window"] == "sensor" else _rows(profile, camera, roi_rows)
    signal = float(window.sum())
    result = dict.fromkeys(LAB_FIELDS)
    if not total > 0 or not signal > 0:
        return result
    pos_real = float(camera["pixel_um"]) * np.arange(window.size)
    mean = (pos_real * window).sum() / signal
    um = float(camera["lab_um_per_pixel"])
    rng = np.random.default_rng(int(seed))
    sigma = float(camera["noise_per_pixel_over_signal"]) * signal
    noisy = window + rng.normal(0.0, sigma, (camera["noise_draws"], window.size))
    result.update(
        lab_stddev2_um=float(np.mean(lab_stddev2(noisy, um, camera["edge_fraction"],
                                                 camera["noise_floor"]))),
        lab_stddev_um=float(np.mean(lab_stddev(noisy, um))),
        lab_stddev2_no_noise_um=float(lab_stddev2(window, um, camera["edge_fraction"],
                                                  camera["noise_floor"])[0]),
        camera_width_um=float(math.sqrt(max((pos_real ** 2 * window).sum() / signal
                                            - mean ** 2, 0.0))),
        lab_window_fraction=signal / total,
        camera_fov_fraction=float(sensor.sum()) / total,
    )
    return result


def mirrored(profile: dict) -> dict:
    """The profile of the mirror-image run (x -> -x), on the same pixel grid."""
    mass = np.asarray(profile["mass"], dtype=float)[::-1]
    return dict(profile, first_pixel=-(int(profile["first_pixel"]) + mass.size - 1),
                mass=mass.tolist())


def combine_profiles(profiles: list[dict], weights: list[tuple]) -> dict:
    """The camera profiles of a point's q runs added up with 10.0's weights
    [(q, simulated q, sign, weight)], mirrored where sign is -1: exactly the
    image of the combined cloud."""
    shifted = [mirrored(p) if sign < 0 else p for p, (_, _, sign, _) in zip(profiles, weights)]
    first = min(int(p["first_pixel"]) for p in shifted)
    last = max(int(p["first_pixel"]) + len(p["mass"]) - 1 for p in shifted)
    total = np.zeros(last - first + 1)
    for p, (_, _, _, weight) in zip(shifted, weights):
        start = int(p["first_pixel"]) - first
        total[start:start + len(p["mass"])] += weight * np.asarray(p["mass"], dtype=float)
    return {"first_pixel": first, "pixel_width": float(shifted[0]["pixel_width"]),
            "mass": total.tolist()}


PROFILE_HEADER = ("run_index", "alpha", "drive_frequency_hz", "initial_quasimomentum_kl",
                  "first_pixel", "pixel_um")


def write_profiles_csv(path: Path, records: list[dict], camera: dict | None = None) -> int:
    """camera_profiles.csv: a comment line with the camera settings, a header,
    then one row per run with its pixel masses."""
    import csv
    rows = [r for r in records if r.get("camera_profile")]
    if not rows:
        return 0
    count = max(len(r["camera_profile"]["mass"]) for r in rows)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        if camera:
            handle.write("# lab_camera " + json.dumps(
                {k: v for k, v in camera.items() if k in DEFAULT_CAMERA}) + "\n")
        writer = csv.writer(handle)
        writer.writerow(list(PROFILE_HEADER) + [f"m{k}" for k in range(count)])
        for r in rows:
            profile = r["camera_profile"]
            writer.writerow([r["run_index"], repr(float(r["alpha"])),
                             repr(float(r["drive_frequency_hz"])),
                             repr(float(r.get("initial_quasimomentum_kl", 0.0))),
                             profile["first_pixel"],
                             repr(profile["pixel_width"] * GRID_SPACING_UM)]
                            + [repr(float(v)) for v in profile["mass"]])
    return len(rows)


def read_profiles_csv(path: Path) -> tuple[dict[int, dict], dict]:
    """({run_index: {"alpha", "drive_frequency_hz", "initial_quasimomentum_kl",
    "profile"}}, camera settings)."""
    import csv
    camera = dict(DEFAULT_CAMERA)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# lab_camera "):
        camera.update(json.loads(lines[0][len("# lab_camera "):]))
        lines = lines[1:]
    reader = csv.reader(lines)
    header = next(reader)
    n = len(PROFILE_HEADER)
    if tuple(header[:n]) != PROFILE_HEADER:
        raise ValueError(f"{Path(path).name} is not a camera_profiles.csv.")
    out = {}
    for row in reader:
        out[int(row[0])] = {
            "alpha": float(row[1]), "drive_frequency_hz": float(row[2]),
            "initial_quasimomentum_kl": float(row[3]),
            "profile": {"first_pixel": int(row[4]),
                        "pixel_width": float(row[5]) / GRID_SPACING_UM,
                        "mass": [float(v) for v in row[n:] if v != ""]},
        }
    return out, validate_camera(camera)


def exact_spread_columns(cells: dict, weights: list[tuple], camera: dict) -> dict:
    """{(alpha, frequency): LAB_FIELDS} of a 10.0 spread from the runs' profiles.

    cells: {(alpha, frequency): {simulated q: (run_index, profile)}}."""
    out = {}
    for key, by_q in cells.items():
        runs = [by_q[simulated] for (_, simulated, _, _) in weights]
        combined = combine_profiles([profile for _, profile in runs], weights)
        out[key] = measure(combined, camera, seed=min(index for index, _ in runs) + 1)
    return out
