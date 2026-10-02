"""Codex CUDA Port: automated short-time and 3x3 CPU/CUDA validation workflow."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tables as tb


VALIDATION_DIRECTORY = Path(__file__).resolve().parent
PORT_ROOT = VALIDATION_DIRECTORY.parent
SIMULATION_CORE = PORT_ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(SIMULATION_CORE))

from cuda_workflow_common import (  # noqa: E402
    locate_executable,
    read_config,
    run_logged,
    sha256_file,
    update_config,
    utc_now,
    write_json_atomic,
)
from compare_checkpoints import (  # noqa: E402
    compare as compare_checkpoints,
    read_status,
)
from compare_phase_diagrams import compare as compare_phase_diagrams  # noqa: E402
from generate_validation_cases import generate_case, load_specification  # noqa: E402


def discover_cpu_references(explicit: list[str]) -> dict[str, Path]:
    references: dict[str, Path] = {}
    for value in explicit:
        if "=" not in value:
            raise ValueError("--cpu-reference must be LABEL=PATH")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        references[label] = path
    if references:
        return references

    source_reference = PORT_ROOT / "build" / "bin" / "gpe1d_cpu_reference.exe"
    opaque_reference = SIMULATION_CORE / "gpe1d_2.exe"
    if source_reference.is_file():
        references["source_cpu"] = source_reference
    if opaque_reference.is_file():
        references["opaque_workflow_executable"] = opaque_reference
    return references


def prepare_variant_config(base: Path, variant: Path, output: Path, status: Path) -> None:
    variant.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    status.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base, variant)
    update_config(variant, {"output_folder": output, "status_file": status})


def run_cpu_reference(executable: Path, config: Path, log: Path) -> int:
    # The opaque executable's HDF5 and MKL DLLs live in simulation_core.
    return run_logged([str(executable), str(config)], SIMULATION_CORE, log)


def run_cuda(executable: Path, config: Path, log: Path, device: int, mode: str) -> int:
    return run_logged(
        [
            str(executable),
            str(config),
            "--device",
            str(device),
            "--floquet-mode",
            mode,
        ],
        SIMULATION_CORE,
        log,
    )


def passed_tolerances(summary: dict[str, Any], defaults: dict[str, Any]) -> bool:
    maximum_energy = max(summary["maximum_energy_relative_errors"].values())
    return bool(
        summary["maximum_wavefunction_relative_l2"]
        <= defaults["wavefunction_relative_l2_tolerance"]
        and summary["maximum_density_absolute_error"]
        <= defaults["density_max_absolute_tolerance"]
        and summary["maximum_norm_absolute_error"]
        <= defaults["norm_absolute_tolerance"]
        and maximum_energy <= defaults["energy_relative_tolerance"]
    )


def read_probability(path: Path) -> np.ndarray:
    with tb.open_file(path, "r") as handle:
        return handle.root.REAL[:] ** 2 + handle.root.IMAGINARY[:] ** 2


def final_snapshot(output: Path) -> Path:
    snapshots = sorted(output.glob("*.h5"), key=lambda path: int(path.stem))
    if not snapshots:
        raise FileNotFoundError(output)
    return snapshots[-1]


def probe_init_overlap_definition(
    initial_path: Path, output: Path, status_path: Path, step_x: float
) -> dict[str, Any] | None:
    status = read_status(status_path)
    if not status or not any("InitOverlap" in row for row in status.values()):
        return None
    initial_density = read_probability(initial_path)
    discrete_errors = []
    integrated_errors = []
    compared = 0
    for snapshot in sorted(output.glob("*.h5"), key=lambda path: int(path.stem)):
        checkpoint = int(snapshot.stem)
        if checkpoint not in status or "InitOverlap" not in status[checkpoint]:
            continue
        density = read_probability(snapshot)
        discrete = float(np.sum(initial_density * density))
        integrated = step_x * discrete
        observed = status[checkpoint]["InitOverlap"]
        discrete_errors.append(abs(observed - discrete))
        integrated_errors.append(abs(observed - integrated))
        compared += 1
    if not compared:
        return None
    maximum_discrete = max(discrete_errors)
    maximum_integrated = max(integrated_errors)
    if np.isclose(maximum_discrete, maximum_integrated, rtol=1.0e-12, atol=1.0e-15):
        closer_candidate = "indistinguishable_at_this_step_x"
    else:
        closer_candidate = (
            "sum_density_product_without_dx"
            if maximum_discrete < maximum_integrated
            else "step_x_times_sum_density_product"
        )
    return {
        "checkpoints_compared": compared,
        "maximum_error_discrete_without_dx": maximum_discrete,
        "maximum_error_integrated_with_dx": maximum_integrated,
        "closer_candidate": closer_candidate,
    }


def phase_metric(output: Path, final_count: int, cut: int) -> float:
    snapshots = sorted(output.glob("*.h5"), key=lambda path: int(path.stem))[-final_count:]
    if not snapshots:
        raise FileNotFoundError(output)
    values = []
    for snapshot in snapshots:
        probability = read_probability(snapshot)[cut:-cut]
        values.append(float(np.sum(probability * probability)))
    return float(np.mean(values))


def run_small_grid(
    specification: dict[str, Any],
    root: Path,
    cuda_executable: Path,
    cpu_label: str,
    cpu_executable: Path,
    device: int,
) -> dict[str, Any]:
    grid = specification["small_grid"]
    alpha_values = np.asarray(grid["alpha_values"], dtype=float)
    frequency_values = np.asarray(grid["frequency_hz_values"], dtype=float)
    cpu_metric = np.full((frequency_values.size, alpha_values.size), np.nan)
    cuda_metric = np.full_like(cpu_metric, np.nan)
    records = []

    for alpha_index, alpha in enumerate(alpha_values):
        for frequency_index, frequency in enumerate(frequency_values):
            run_index = alpha_index * frequency_values.size + frequency_index
            case = {
                "name": f"grid_{run_index:03d}",
                "alpha": float(alpha),
                "frequency_hz": float(frequency),
                "floquet_enabled": True,
                "floquet_mode": "legacy",
                "iterations": grid["iterations"],
                "save_every": grid["save_every"],
                "show_every": grid["show_every"],
            }
            case_root = root / case["name"]
            generated = generate_case(case, specification["defaults"], case_root)

            cpu_root = case_root / cpu_label
            cpu_output = cpu_root / "out"
            cpu_status = cpu_root / "status.csv"
            cpu_config = cpu_root / "gpe1d.config"
            prepare_variant_config(generated["config"], cpu_config, cpu_output, cpu_status)
            cpu_code = run_cpu_reference(
                cpu_executable, cpu_config, cpu_root / "solver.log"
            )
            if cpu_code != 0:
                raise RuntimeError(f"Small-grid CPU run {run_index} failed.")

            cuda_root = case_root / "cuda"
            cuda_output = cuda_root / "out"
            cuda_status = cuda_root / "status.csv"
            cuda_config = cuda_root / "gpe1d.config"
            prepare_variant_config(generated["config"], cuda_config, cuda_output, cuda_status)
            cuda_code = run_cuda(
                cuda_executable,
                cuda_config,
                cuda_root / "solver.log",
                device,
                "legacy",
            )
            if cuda_code != 0:
                raise RuntimeError(f"Small-grid CUDA run {run_index} failed.")

            cpu_value = phase_metric(
                cpu_output, grid["final_snapshot_count"], grid["cut_points_each_edge"]
            )
            cuda_value = phase_metric(
                cuda_output, grid["final_snapshot_count"], grid["cut_points_each_edge"]
            )
            cpu_metric[frequency_index, alpha_index] = cpu_value
            cuda_metric[frequency_index, alpha_index] = cuda_value
            records.append(
                {
                    "run_index": run_index,
                    "alpha": float(alpha),
                    "frequency_hz": float(frequency),
                    "cpu_metric": cpu_value,
                    "cuda_metric": cuda_value,
                }
            )

    cpu_npz = root / "cpu_phase_metric.npz"
    cuda_npz = root / "cuda_phase_metric.npz"
    common = {
        "alpha_values": alpha_values,
        "drive_frequency_hz_values": frequency_values,
    }
    np.savez(cpu_npz, **common, metric_matrix=cpu_metric)
    np.savez(cuda_npz, **common, metric_matrix=cuda_metric)
    comparison = compare_phase_diagrams(
        cpu_npz, cuda_npz, root / "phase_diagram_comparison.json"
    )
    return {"cpu_reference": cpu_label, "runs": records, "comparison": comparison}


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "<!-- Codex CUDA Port: generated validation summary. -->",
        "# Native CUDA validation report",
        "",
        f"Generated: {report['finished_utc']}",
        "",
        "| Case | CPU reference | Result | Max phase-aligned relative L2 | Max density error |",
        "|---|---|---:|---:|---:|",
    ]
    for case in report["cases"]:
        if not case.get("comparisons"):
            lines.append(f"| {case['name']} | n/a | CUDA mode exercised only | n/a | n/a |")
            continue
        for label, comparison in case["comparisons"].items():
            expected = comparison.get("expected_to_match", True)
            result = (
                "PASS" if comparison.get("within_tolerance") else "FAIL"
            ) if expected else "COMPATIBILITY PROBE"
            lines.append(
                f"| {case['name']} | {label} | {result} "
                f"| {comparison['maximum_wavefunction_relative_l2']:.6e} "
                f"| {comparison['maximum_density_absolute_error']:.6e} |"
            )
    lines.extend(
        [
            "",
            "The phase alignment minimizes the complex wavefunction difference using the CPU/CUDA inner product.",
            "The phase-diagram quantity is the unnormalized cropped discrete sum of |psi|^4 averaged over the final snapshots.",
            "A passing report establishes only the cases and tolerances recorded here; it is not a blanket equivalence claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-executable", type=Path)
    parser.add_argument("--cpu-reference", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--skip-small-grid", action="store_true")
    arguments = parser.parse_args()

    cuda_executable = locate_executable(arguments.cuda_executable)
    cpu_references = discover_cpu_references(arguments.cpu_reference)
    if not cpu_references:
        raise FileNotFoundError("No CPU reference executable was found.")
    timestamp = datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%SZ")
    results = arguments.results.resolve() if arguments.results else VALIDATION_DIRECTORY / "results" / timestamp
    if results.exists():
        raise FileExistsError(results)
    results.mkdir(parents=True)

    specification = load_specification()
    report: dict[str, Any] = {
        "_codex_cuda_port": "Automated native CUDA validation report.",
        "started_utc": utc_now(),
        "status": "running",
        "cuda_executable": {
            "path": str(cuda_executable),
            "sha256": sha256_file(cuda_executable),
        },
        "cpu_references": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in cpu_references.items()
        },
        "tolerances": specification["defaults"],
        "cases": [],
    }
    report_path = results / "validation_report.json"
    write_json_atomic(report_path, report)

    try:
        for case in specification["cases"]:
            case_root = results / "cases" / case["name"]
            generated = generate_case(case, specification["defaults"], case_root)
            cuda_root = case_root / "cuda"
            cuda_output = cuda_root / "out"
            cuda_status = cuda_root / "status.csv"
            cuda_config = cuda_root / "gpe1d.config"
            prepare_variant_config(generated["config"], cuda_config, cuda_output, cuda_status)
            code = run_cuda(
                cuda_executable,
                cuda_config,
                cuda_root / "solver.log",
                arguments.device,
                case["floquet_mode"],
            )
            if code != 0:
                raise RuntimeError(f"CUDA validation case {case['name']} failed.")

            case_report: dict[str, Any] = {
                "name": case["name"],
                "floquet_mode": case["floquet_mode"],
                "generated": {
                    key: value for key, value in generated.items() if key != "config"
                },
                "comparisons": {},
                "overlap_definition_probes": {},
            }
            step_x = float(read_config(cuda_config)["step_x"])
            for label, cpu_executable in cpu_references.items():
                cpu_root = case_root / label
                cpu_output = cpu_root / "out"
                cpu_status = cpu_root / "status.csv"
                cpu_config = cpu_root / "gpe1d.config"
                prepare_variant_config(
                    generated["config"], cpu_config, cpu_output, cpu_status
                )
                code = run_cpu_reference(
                    cpu_executable, cpu_config, cpu_root / "solver.log"
                )
                if code != 0:
                    raise RuntimeError(
                        f"CPU reference {label} failed for case {case['name']}."
                    )
                comparison = compare_checkpoints(
                    cpu_output,
                    cuda_output,
                    cpu_status,
                    cuda_status,
                    step_x,
                    case_root / f"compare_{label}_cuda.csv",
                )
                comparison["expected_to_match"] = case["floquet_mode"] == "legacy"
                comparison["within_tolerance"] = (
                    passed_tolerances(comparison, specification["defaults"])
                    if comparison["expected_to_match"]
                    else None
                )
                case_report["comparisons"][label] = comparison
                overlap_probe = probe_init_overlap_definition(
                    Path(generated["inputs"]["lattice_gauss.h5"]["path"]),
                    cpu_output,
                    cpu_status,
                    step_x,
                )
                if overlap_probe is not None:
                    case_report["overlap_definition_probes"][label] = overlap_probe
            report["cases"].append(case_report)
            write_json_atomic(report_path, report)

        legacy = results / "cases" / "driven_legacy" / "cuda" / "out"
        physical = results / "cases" / "driven_physical" / "cuda" / "out"
        legacy_density = read_probability(final_snapshot(legacy))
        physical_density = read_probability(final_snapshot(physical))
        report["floquet_mode_difference"] = {
            "final_density_max_absolute": float(
                np.max(np.abs(physical_density - legacy_density))
            ),
            "definition": "max abs density difference for identical driven inputs",
        }
        legacy_case = next(item for item in report["cases"] if item["name"] == "driven_legacy")
        physical_case = next(item for item in report["cases"] if item["name"] == "driven_physical")
        compatibility = {}
        for label in set(legacy_case["comparisons"]).intersection(physical_case["comparisons"]):
            legacy_error = legacy_case["comparisons"][label][
                "maximum_wavefunction_relative_l2"
            ]
            physical_error = physical_case["comparisons"][label][
                "maximum_wavefunction_relative_l2"
            ]
            compatibility[label] = {
                "legacy_maximum_wavefunction_relative_l2": legacy_error,
                "physical_maximum_wavefunction_relative_l2": physical_error,
                "closer_mode": "legacy" if legacy_error < physical_error else "physical",
            }
        report["floquet_compatibility_probe"] = compatibility

        if not arguments.skip_small_grid:
            preferred_label = (
                "source_cpu" if "source_cpu" in cpu_references else next(iter(cpu_references))
            )
            report["small_grid"] = run_small_grid(
                specification,
                results / "small_grid",
                cuda_executable,
                preferred_label,
                cpu_references[preferred_label],
                arguments.device,
            )

        all_comparisons = [
            comparison
            for case in report["cases"]
            for comparison in case["comparisons"].values()
            if comparison.get("expected_to_match", True)
        ]
        report["all_short_time_comparisons_within_tolerance"] = bool(
            all_comparisons and all(item["within_tolerance"] for item in all_comparisons)
        )
        report["status"] = "completed"
        report["finished_utc"] = utc_now()
        write_json_atomic(report_path, report)
        write_markdown_report(report, results / "VALIDATION_REPORT.md")
        print(f"Validation report: {report_path}")
        return 0 if report["all_short_time_comparisons_within_tolerance"] else 2
    except Exception:
        report["status"] = "failed"
        report["finished_utc"] = utc_now()
        write_json_atomic(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
