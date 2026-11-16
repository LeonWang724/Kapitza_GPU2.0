"""10.0: combine runs at several starting quasimomenta into one cloud with a spread of q.

A real cloud has atoms at many quasimomenta q. Without interactions each group
of atoms with the same q evolves on its own, and groups at different q do not
interfere in the image, so the camera sees the sum of their clouds. A 10.0 scan
runs every grid point once per simulated q; this module adds those runs up,
weighted by the fraction of atoms a Gaussian distribution of q (centre, spread)
puts at each simulated value.

Weights: each simulated q stands for the q values closer to it than to its
neighbours, and gets the Gaussian's probability in that range. Negative q are
the mirror images x -> -x of positive q (the lattice, drive, starting cloud and
green walls are symmetric), so a run at q > 0 also stands for -q when -q was not
simulated. The coverage is the part of the Gaussian inside the simulated range.

How each metrics.csv column combines:
  * exact, added atom by atom: the widths (sigma_x, sigma_x_4um_um and their
    starting values, from the second moments about x = 0, not by averaging
    widths), the cloud centre x_mean, x_squared_mean, central_fraction,
    escape_fraction, pendulum_cos and pendulum_cos_strobe, p_squared, order_0,
    order_2, energy_absorbed_er, return_probability (the fraction of atoms back
    in their starting state) and the Bloch-state model's populations, momenta
    and energies;
  * the same for every q: the classical ensemble (cl_ columns) and the sampling
    counts;
  * approximate, the average over q weighted like the atoms (APPROXIMATE): every
    other column, e.g. the IPRs, entropies, kurtosis, on-site spread, growth
    rates and level statistics, which depend on the square of the density or on
    the whole time series, which metrics.csv does not keep.
11.0: the lab_* columns (lab_measurement.py) are recomputed exactly from the
added-up camera images when camera_profiles.csv is available (MAKE, the command
line and the plot tools find it next to metrics.csv); without it they are
weighted averages like the other approximate columns.
The widths are averages over the last 30 samples, about the cloud's centre at
each sample. For a spread centred at 0 the combined centre stays at 0 and the
widths are exact. Otherwise the combined centre's own motion during those
samples is estimated from each run's (each run's centre moving steadily away
from x = 0 within the window), which is exact for a single q and close
otherwise.

Command line: python quasimomentum_spread.py metrics.csv --center 0 --spread 0.1
writes metrics_spread_c0_s0p1.csv next to it.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

GRID_SPACING_UM = 0.023        # solver length unit (x_mean, sigma_x are in it)
QUASIMOMENTUM_COLUMN = "initial_quasimomentum_kl"
SPREAD_COLUMNS = ("initial_quasimomentum_kl", "quasimomentum_spread_kl", "quasimomentum_coverage")
COVERAGE_WARNING = 0.95
TOLERANCE = 1e-9

FIRST_COLUMNS = ("run_index",)
CELL_COLUMNS = ("alpha", "drive_frequency_hz")
SAME_COLUMNS = ("snapshots_averaged", "spatial_snapshots_averaged",
                "level_ratio_count", "cl_pendulum_cos", "cl_near_start", "cl_site_escape")
# Per atom still in the box (GPU): weighted by the atoms present.
PRESENT_ATOM_COLUMNS = ("central_fraction", "pendulum_cos", "pendulum_cos_strobe", "p_squared",
                        "order_0", "order_2", "energy_absorbed_er")
# Per starting atom: weighted by the atoms that started.
STARTING_ATOM_COLUMNS = ("escape_fraction", "return_probability",
                         "q0_pendulum_cos", "q0_p_squared", "q0_order_0", "q0_order_2",
                         "q0_energy_absorbed_er", "q0_band0_population",
                         "q0_start_band_population", "q0_excited_band_population")
WIDTH_COLUMNS = ("x_mean", "x_squared_mean", "sigma_x", "sigma_x_squared", "x_mean_time_std",
                 "sigma_x_4um_um", "sigma_x_um_initial", "sigma_x_4um_um_initial")
EXACT_COLUMNS = FIRST_COLUMNS + CELL_COLUMNS + SAME_COLUMNS + PRESENT_ATOM_COLUMNS \
    + STARTING_ATOM_COLUMNS + WIDTH_COLUMNS


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def validate_values(values) -> list[float]:
    """Sorted simulated q values; each -1 < q < 1, no repeats."""
    try:
        numbers = sorted(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError("QUASIMOMENTUM_VALUES_KL must be a list of numbers.") from None
    if not numbers or any(isinstance(value, bool) for value in values) \
            or any(not -1.0 < value < 1.0 for value in numbers):
        raise ValueError("Every quasimomentum must be a number with -1 < q < 1.")
    if any(b - a < TOLERANCE for a, b in zip(numbers, numbers[1:])):
        raise ValueError("QUASIMOMENTUM_VALUES_KL lists a value twice.")
    return numbers


def spread_weights(values, center: float, spread: float) -> tuple[list[tuple], float]:
    """([(q, simulated q, sign, weight)], coverage) for a Gaussian of q.

    sign is -1 where q is the mirror image of a simulated positive value."""
    simulated = validate_values(values)
    if isinstance(center, bool) or not isinstance(center, (int, float)) or not -1.0 < center < 1.0:
        raise ValueError("The quasimomentum centre must be a number with -1 < q < 1.")
    if isinstance(spread, bool) or not isinstance(spread, (int, float)) or not spread >= 0.0:
        raise ValueError("The quasimomentum spread must be a number >= 0.")
    points = {q: (q, 1) for q in simulated}
    for q in simulated:
        if q > TOLERANCE and not any(abs(-q - other) < TOLERANCE for other in points):
            points[-q] = (q, -1)
    qs = sorted(points)
    if spread == 0.0:
        match = [q for q in qs if abs(q - center) < 1e-6]
        if not match:
            raise ValueError(f"A spread of 0 needs q = {center:g} among the simulated values "
                             "(or their mirror images).")
        q = match[0]
        return [(q, *points[q], 1.0)], 1.0
    edges = [qs[0] - (qs[1] - qs[0]) / 2 if len(qs) > 1 else qs[0] - 0.5]
    edges += [(a + b) / 2 for a, b in zip(qs, qs[1:])]
    edges += [qs[-1] + (qs[-1] - qs[-2]) / 2 if len(qs) > 1 else qs[0] + 0.5]
    masses = [normal_cdf((high - center) / spread) - normal_cdf((low - center) / spread)
              for low, high in zip(edges, edges[1:])]
    coverage = sum(masses)
    if not coverage > 1e-6:
        raise ValueError("The simulated quasimomenta miss this spread entirely.")
    weights = [(q, *points[q], mass / coverage) for q, mass in zip(qs, masses) if mass > 0.0]
    return weights, coverage


def number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _weighted(rows, weights, column, *, present: bool):
    total = norm = 0.0
    for row, (_, _, _, weight) in zip(rows, weights):
        value = number(row.get(column))
        if value is None or not math.isfinite(value):
            return None
        share = weight
        if present:
            escape = number(row.get("escape_fraction"))
            share *= 1.0 - (escape if escape is not None and math.isfinite(escape) else 0.0)
        total += share * value
        norm += share
    return total / norm if norm > 0.0 else None


def _present_shares(rows, weights):
    shares = []
    for row, (_, _, _, weight) in zip(rows, weights):
        escape = number(row.get("escape_fraction"))
        shares.append(weight * (1.0 - (escape if escape is not None and math.isfinite(escape)
                                       else 0.0)))
    total = sum(shares)
    return [share / total for share in shares] if total > 0.0 else list(shares)


def _combine_widths(rows, weights, out: dict) -> None:
    signs = [sign for (_, _, sign, _) in weights]
    shares = _present_shares(rows, weights)
    get = lambda row, key: number(row.get(key))           # noqa: E731
    # Grid moments (solver length units), about x = 0.
    if all(get(row, key) is not None for row in rows
           for key in ("x_mean", "x_squared_mean")):
        centre = sum(s * share * get(row, "x_mean") for row, s, share in zip(rows, signs, shares))
        second = sum(share * get(row, "x_squared_mean") for row, share in zip(rows, shares))
        # The combined centre's motion over the window: each run's centre
        # moving steadily away from x = 0 (0 for a spread centred at 0).
        wobble = sum(s * share * math.copysign(get(row, "x_mean_time_std") or 0.0,
                                               get(row, "x_mean"))
                     for row, s, share in zip(rows, signs, shares))
        variance = max(second - centre * centre - wobble * wobble, 0.0)
        out.update(x_mean=centre, x_squared_mean=second, x_mean_time_std=abs(wobble),
                   sigma_x_squared=variance, sigma_x=math.sqrt(variance))
    else:
        out.update(dict.fromkeys(("x_mean", "x_squared_mean", "x_mean_time_std",
                                  "sigma_x_squared", "sigma_x")))
    # Camera-pixel width (um): its second moment about x = 0 from the pixel
    # variance and the run's centre (and the centre's own wobble).
    if out.get("x_mean") is not None and all(get(row, "sigma_x_4um_um") is not None
                                             for row in rows):
        second = 0.0
        for row, share in zip(rows, shares):
            centre_um = get(row, "x_mean") * GRID_SPACING_UM
            wobble_um = (get(row, "x_mean_time_std") or 0.0) * GRID_SPACING_UM
            second += share * (get(row, "sigma_x_4um_um") ** 2 + centre_um ** 2 + wobble_um ** 2)
        centre_um = out["x_mean"] * GRID_SPACING_UM
        wobble_um = (out["x_mean_time_std"] or 0.0) * GRID_SPACING_UM
        out["sigma_x_4um_um"] = math.sqrt(max(second - centre_um ** 2 - wobble_um ** 2, 0.0))
    else:
        out["sigma_x_4um_um"] = None
    # Starting widths: every group starts as the same cloud at x = 0.
    for key in ("sigma_x_um_initial", "sigma_x_4um_um_initial"):
        values = [get(row, key) for row in rows]
        out[key] = (math.sqrt(sum(w * v * v for (_, _, _, w), v in zip(weights, values)))
                    if all(v is not None for v in values) else None)


def combine_cell(rows: list[dict], weights: list[tuple]) -> dict:
    """One combined row from the runs of one grid point, in the order of `weights`."""
    out = {}
    columns = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    for column in columns:
        if column in SPREAD_COLUMNS or column in WIDTH_COLUMNS:
            continue
        if column in FIRST_COLUMNS:
            values = [number(row.get(column)) for row in rows]
            out[column] = min(v for v in values if v is not None) if any(
                v is not None for v in values) else None
            if out[column] is not None and float(out[column]).is_integer():
                out[column] = int(out[column])
        elif column in CELL_COLUMNS:
            out[column] = number(rows[0].get(column))
        elif column in SAME_COLUMNS:
            values = {number(row.get(column)) for row in rows}
            out[column] = (values.pop() if len(values) == 1
                           else _weighted(rows, weights, column, present=False))
        elif column in PRESENT_ATOM_COLUMNS:
            out[column] = _weighted(rows, weights, column, present=True)
        elif column in STARTING_ATOM_COLUMNS:
            out[column] = _weighted(rows, weights, column, present=False)
        elif all(number(row.get(column)) is not None or row.get(column) in (None, "")
                 for row in rows):
            out[column] = _weighted(rows, weights, column, present=True)
        else:
            out[column] = rows[0].get(column)            # text: keep the first
    _combine_widths(rows, weights, out)
    return out


def is_spread_table(rows: list[dict]) -> bool:
    """True when some grid point was run at more than one quasimomentum."""
    if not rows or QUASIMOMENTUM_COLUMN not in rows[0]:
        return False
    seen: dict[tuple, set] = {}
    for row in rows:
        key = (number(row.get("alpha")), number(row.get("drive_frequency_hz")))
        seen.setdefault(key, set()).add(number(row.get(QUASIMOMENTUM_COLUMN)))
    return any(len(values) > 1 for values in seen.values())


def simulated_values(rows: list[dict]) -> list[float]:
    return sorted({number(row[QUASIMOMENTUM_COLUMN]) for row in rows})


def combine_rows(rows: list[dict], center: float, spread: float) -> tuple[list[dict], float]:
    """(one combined row per grid point, coverage) from the rows of a spread scan."""
    values = simulated_values(rows)
    weights, coverage = spread_weights(values, center, spread)
    cells: dict[tuple, dict[float, dict]] = {}
    for row in rows:
        key = (number(row["alpha"]), number(row["drive_frequency_hz"]))
        q = number(row[QUASIMOMENTUM_COLUMN])
        if q in cells.setdefault(key, {}):
            raise ValueError(f"Two rows at alpha = {key[0]:g}, f = {key[1]:g} Hz, q = {q:g}.")
        cells[key][q] = row
    combined = []
    for key, by_q in cells.items():
        missing = [simulated for (_, simulated, _, _) in weights if simulated not in by_q]
        if missing:
            raise ValueError(f"alpha = {key[0]:g}, f = {key[1]:g} Hz lacks q = {missing[0]:g}.")
        row = combine_cell([by_q[simulated] for (_, simulated, _, _) in weights], weights)
        row.update({QUASIMOMENTUM_COLUMN: float(center), "quasimomentum_spread_kl": float(spread),
                    "quasimomentum_coverage": coverage})
        combined.append(row)
    combined.sort(key=lambda row: (row.get("run_index") is None, row.get("run_index") or 0))
    return combined, coverage


APPROXIMATE_NOTE = ("approximate (average over q): every column not in "
                    "quasimomentum_spread.EXACT_COLUMNS")


def find_profiles(csv_path: Path) -> Path | None:
    """camera_profiles.csv belonging to a metrics CSV: <stem>_camera_profiles.csv
    or camera_profiles.csv in the same folder."""
    csv_path = Path(csv_path)
    for candidate in (csv_path.with_name(f"{csv_path.stem}_camera_profiles.csv"),
                      csv_path.with_name("camera_profiles.csv")):
        if candidate.is_file():
            return candidate
    return None


def apply_exact_lab_columns(combined: list[dict], rows: list[dict], profiles_path: Path,
                            center: float, spread: float) -> bool:
    """Replace the combined rows' lab columns by those of the added-up camera
    images (11.0). False (rows unchanged) if the profiles do not match the rows."""
    from lab_measurement import exact_spread_columns, read_profiles_csv
    profiles, camera = read_profiles_csv(profiles_path)
    cells: dict[tuple, dict] = {}
    for row in rows:
        index = int(number(row["run_index"]))
        entry = profiles.get(index)
        q = number(row[QUASIMOMENTUM_COLUMN])
        key = (number(row["alpha"]), number(row["drive_frequency_hz"]))
        if entry is None or abs(entry["alpha"] - key[0]) > 1e-9 \
                or abs(entry["drive_frequency_hz"] - key[1]) > 1e-6 \
                or abs(entry["initial_quasimomentum_kl"] - q) > 1e-12:
            return False
        cells.setdefault(key, {})[q] = (index, entry["profile"])
    weights = spread_weights(simulated_values(rows), center, spread)[0]
    exact = exact_spread_columns(cells, weights, camera)
    for row in combined:
        row.update(exact[(number(row["alpha"]), number(row["drive_frequency_hz"]))])
    return True


def describe(values, center: float, spread: float) -> str:
    weights, coverage = spread_weights(values, center, spread)
    shares = ", ".join(f"{q:+.3g}: {w:.0%}" for q, _, _, w in weights if w >= 0.005)
    return (f"Gaussian of q centred at {center:g} hbar kL with spread {spread:g} hbar kL; "
            f"{coverage:.1%} of it lies within the simulated values ({shares}).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="metrics.csv of a 10.0 spread scan")
    parser.add_argument("--center", type=float, default=0.0, help="centre of q (hbar kL)")
    parser.add_argument("--spread", type=float, default=0.1, help="spread (std) of q (hbar kL)")
    parser.add_argument("--output", type=Path, help="combined CSV (default: next to the input)")
    parser.add_argument("--profiles", type=Path,
                        help="camera_profiles.csv (11.0; default: found next to the input)")
    arguments = parser.parse_args()
    with arguments.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not is_spread_table(rows):
        raise SystemExit(f"{arguments.csv.name} has one quasimomentum per point; nothing to combine.")
    combined, coverage = combine_rows(rows, arguments.center, arguments.spread)
    profiles = arguments.profiles or find_profiles(arguments.csv)
    exact = bool(profiles) and apply_exact_lab_columns(combined, rows, profiles,
                                                       arguments.center, arguments.spread)
    token = lambda value: f"{value:g}".replace("-", "m").replace(".", "p")   # noqa: E731
    output = arguments.output or arguments.csv.with_name(
        f"{arguments.csv.stem}_spread_c{token(arguments.center)}_s{token(arguments.spread)}.csv")
    columns = [column for column in rows[0] if column != QUASIMOMENTUM_COLUMN]
    columns[3:3] = list(SPREAD_COLUMNS)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    print(describe(simulated_values(rows), arguments.center, arguments.spread))
    if coverage < COVERAGE_WARNING:
        print(f"Warning: only {coverage:.1%} of the spread lies within the simulated q values.")
    print(f"{len(combined)} points; {APPROXIMATE_NOTE}"
          + (f"; lab columns exact from {Path(profiles).name}." if exact else "."))
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
