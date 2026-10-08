"""Split a grid among isolated terminal workers and merge complete results."""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from cuda_workflow_common import (
    PORT_ROOT, SCRIPT_DIRECTORY, read_config, sha256_file, utc_now, write_json_atomic,
)


def split_point_ranges(total: int, tabs: int) -> list[dict]:
    """Cover the grid exactly once, with workloads differing by at most one."""
    if total <= 0 or tabs <= 0:
        raise ValueError("The scan and tab count must both be positive.")
    count = min(total, tabs)
    size, remainder = divmod(total, count)
    assignments = []
    start = 0
    for index in range(count):
        stop = start + size + (index < remainder)
        assignments.append({"worker_index": index, "start": start, "stop": stop})
        start = stop
    return assignments


def print_assignments(total: int, tabs: int, assignments: list[dict]) -> None:
    print(f"\nScan: {total:,} points; requested tabs: {tabs}; active workers: {len(assignments)}")
    if tabs > total:
        print("Tab count exceeds point count; opening only workers with assigned points.")
    for assignment in assignments:
        start, stop = assignment["start"], assignment["stop"]
        print(f"  Tab {assignment['worker_index'] + 1:>3}: {stop - start:,} points "
              f"(global indices {start:,} through {stop - 1:,})")


def process_exists(pid: int) -> bool:
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied does not mean dead.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def launch_worker(
    manifest_path: Path, title: str, window_name: str, *, headless: bool,
    terminal: str | None,
) -> subprocess.Popen:
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker-manifest", str(manifest_path)]
    options = {"cwd": PORT_ROOT}
    if not headless and terminal:
        command = [terminal, "--window", window_name, "new-tab", "--title", title,
                   "--suppressApplicationTitle", "--startingDirectory", str(PORT_ROOT),
                   *command]
    elif not headless and os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        options.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Argument lists avoid shell parsing of paths containing spaces.
    return subprocess.Popen(command, **options)


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stop_workers(root: Path, launched: list[tuple[Path, subprocess.Popen]]) -> None:
    # Late-starting tabs also see this marker before beginning a point.
    (root / "cancel_requested").touch()
    for path, process in launched:
        worker = read_manifest(path)
        if worker.get("status") in {"completed", "failed", "cancelled"}:
            continue
        pid = worker.get("worker_pid")
        if os.name == "nt" and pid and process_exists(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def merge_workers(root: Path, manifest: dict) -> None:
    """Validate exact coverage before publishing a completed parent dataset."""
    from phase_metric import CSV_FIELDS, csv_row, validate_summary

    grid = manifest["parameter_grid"]
    frequencies = grid["drive_frequency_hz_values"]
    total = len(grid["alpha_values"]) * len(frequencies)
    records = []
    seen = set()
    for assignment in manifest["parallel_execution"]["workers"]:
        worker_root = root / assignment["directory"]
        worker = read_manifest(worker_root / "run_manifest.json")
        count = assignment["stop"] - assignment["start"]
        if worker.get("status") != "completed" or worker.get("completed_points") != count:
            raise ValueError(f"Worker {assignment['worker_index'] + 1} is incomplete.")
        expected_partition = {key: assignment[key] for key in ("worker_index", "start", "stop")}
        if (worker["parameter_grid"] != grid or worker["analysis_contract"] != manifest["analysis_contract"]
                or worker["storage_mode"] != manifest["storage_mode"]
                or worker["work_partition"] != expected_partition):
            raise ValueError("Worker settings or assignment do not match the parent scan.")
        if manifest["storage_mode"] == "compact":
            if worker["simulation_config"] != manifest["simulation_config"]:
                raise ValueError("Worker sampling configuration differs from the parent scan.")
            runs = [json.loads(line) for line in
                    (worker_root / worker["point_results_file"]).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        else:
            runs = worker["runs"]
        if len(runs) != count:
            raise ValueError("Worker results do not match its assigned point count.")
        for record in runs:
            index = record["run_index"]
            if not (assignment["start"] <= index < assignment["stop"]) or index in seen:
                raise ValueError("Worker results contain duplicate or incorrectly assigned points.")
            ai, fi = divmod(index, len(frequencies))
            if (record.get("status") != "completed" or record["alpha_index"] != ai
                    or record["frequency_index"] != fi or record["alpha"] != grid["alpha_values"][ai]
                    or record["drive_frequency_hz"] != frequencies[fi]):
                raise ValueError("Worker point parameters do not match the parent grid.")
            if manifest["storage_mode"] == "compact":
                validate_summary(record["metric_summary"], manifest["simulation_config"],
                                 manifest["analysis_contract"])
            else:
                # Full-mode snapshots stay in each worker folder; plotting uses
                # parent-relative paths after merging.
                for key in ("output_directory", "config_file", "status_file", "log_file"):
                    record[key] = (Path(assignment["directory"]) / record[key]).as_posix()
                for input_record in record["inputs"].values():
                    input_record["path"] = (Path(assignment["directory"]) / input_record["path"]).as_posix()
            seen.add(index)
            records.append(record)
    if seen != set(range(total)):
        raise ValueError("Parallel scan is missing grid points.")
    records.sort(key=lambda record: record["run_index"])
    if manifest["storage_mode"] == "compact":
        journal = root / "point_results.jsonl.tmp"
        table = root / "metrics.csv.tmp"
        with journal.open("w", encoding="utf-8") as stream, table.open("w", encoding="utf-8", newline="") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                writer.writerow(csv_row(record, record["metric_summary"],
                                        manifest["simulation_config"], manifest["analysis_contract"]))
        journal.replace(root / "point_results.jsonl")
        table.replace(root / "metrics.csv")
        manifest.pop("runs", None)
        manifest["point_results_file"] = "point_results.jsonl"
    else:
        manifest["runs"] = records
    manifest["completed_points"] = total
    manifest["status"] = "completed"
    manifest["finished_utc"] = utc_now()
    write_json_atomic(root / "run_manifest.json", manifest)


def run_parallel_sweep(
    *, results_root: Path, manifest: dict, base_config: Path,
    tabs: int, headless: bool = False,
) -> None:
    grid = manifest["parameter_grid"]
    total = len(grid["alpha_values"]) * len(grid["drive_frequency_hz_values"])
    assignments = split_point_ranges(total, tabs)
    provenance = results_root / "provenance"
    provenance.mkdir()
    snapshot = provenance / "gpe1d.config"
    shutil.copy2(base_config, snapshot)
    source = SCRIPT_DIRECTORY / "create_initial_state_function.py"
    shutil.copy2(source, provenance / source.name)
    manifest["input_generator_sha256"] = sha256_file(source)
    manifest["simulation_config"] = read_config(snapshot)
    manifest["completed_points"] = 0
    terminal = shutil.which("wt.exe") if os.name == "nt" and not headless else None
    mode = "windows_terminal_tabs" if terminal else (
        "console_windows" if os.name == "nt" and not headless else "background_processes")
    manifest["parallel_execution"] = {
        "requested_tabs": tabs, "active_workers": len(assignments),
        "launch_mode": mode, "workers": [],
    }
    print(f"Launching {len(assignments)} workers using {mode.replace('_', ' ')}.")
    print(f"Combined results: {results_root}", flush=True)
    window_name = "Kapitza-" + uuid.uuid4().hex[:12]
    launched = []
    try:
        for assignment in assignments:
            directory = f"workers/tab_{assignment['worker_index'] + 1:03d}"
            entry = dict(assignment, directory=directory)
            manifest["parallel_execution"]["workers"].append(entry)
            worker_root = results_root / directory
            worker_root.mkdir(parents=True)
            worker = copy.deepcopy(manifest)
            worker.pop("parallel_execution", None)
            worker["status"] = "pending"
            worker["results_root"] = str(worker_root)
            worker["work_partition"] = assignment
            worker["worker_base_config"] = str(snapshot)
            worker["parent_results_root"] = str(results_root)
            # A pre-existing Terminal server can have an older DLL search path.
            worker["worker_launch_path"] = os.environ.get("PATH", "")
            path = worker_root / "run_manifest.json"
            write_json_atomic(path, worker)
            write_json_atomic(results_root / "run_manifest.json", manifest)
            process = launch_worker(path, f"Kapitza {assignment['worker_index'] + 1}/{len(assignments)}",
                                    window_name, headless=headless, terminal=terminal)
            entry["launched_monotonic"] = time.monotonic()
            launched.append((path, process))
        last_progress = None
        while True:
            states = []
            for entry, (path, process) in zip(manifest["parallel_execution"]["workers"], launched):
                worker = read_manifest(path)
                exit_code = process.poll()
                if worker["status"] in {"failed", "cancelled"}:
                    raise RuntimeError(f"Tab {entry['worker_index'] + 1} failed; see {path.parent}")
                if worker["status"] != "completed":
                    pid = worker.get("worker_pid")
                    dead = (exit_code is not None and (not terminal or exit_code != 0))
                    dead = dead or (pid is not None and not process_exists(pid))
                    timed_out = pid is None and time.monotonic() - entry["launched_monotonic"] > 120
                    if dead or timed_out:
                        # Re-read to cover a worker finishing between reads.
                        worker = read_manifest(path)
                        if worker["status"] != "completed":
                            raise RuntimeError(f"Tab {entry['worker_index'] + 1} exited or did not start; see {path.parent}")
                states.append(worker)
            completed = sum(worker.get("completed_points", 0) for worker in states)
            finished = sum(worker["status"] == "completed" for worker in states)
            progress = (completed, finished)
            if progress != last_progress:
                manifest["completed_points"] = completed
                write_json_atomic(results_root / "run_manifest.json", manifest)
                print(f"Completed {completed:,}/{total:,} points; {finished}/{len(assignments)} tabs finished.", flush=True)
                last_progress = progress
            if finished == len(assignments):
                break
            time.sleep(0.5)
        for entry in manifest["parallel_execution"]["workers"]:
            entry.pop("launched_monotonic", None)
        merge_workers(results_root, manifest)
    except BaseException as error:
        manifest["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        manifest["finished_utc"] = utc_now()
        write_json_atomic(results_root / "run_manifest.json", manifest)
        stop_workers(results_root, launched)
        raise
    finally:
        # Workers publish completion just before exiting. Reap the actual
        # children (or Terminal launch requests) even on the failure path.
        for _, process in launched:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def worker_main(manifest_path: Path) -> int:
    manifest_path = manifest_path.resolve()
    manifest = read_manifest(manifest_path)
    manifest["worker_pid"] = os.getpid()
    manifest["status"] = "running"
    manifest["started_utc"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    os.environ["PATH"] = manifest.pop("worker_launch_path", os.environ.get("PATH", ""))
    if hasattr(signal, "SIGTERM"):
        def terminate(signum, frame):
            raise KeyboardInterrupt("Worker terminated by coordinator")
        signal.signal(signal.SIGTERM, terminate)
    try:
        from compact_sweep import run_compact_sweep
        from full_sweep import run_full_sweep
        from create_initial_state_function import create_init_state
        if (Path(manifest["parent_results_root"]) / "cancel_requested").exists():
            raise KeyboardInterrupt("Scan was cancelled before this tab started")
        assignment = manifest["work_partition"]
        print(f"Tab {assignment['worker_index'] + 1}: {assignment['stop'] - assignment['start']:,} points", flush=True)
        sweep = run_compact_sweep if manifest["storage_mode"] == "compact" else run_full_sweep
        sweep(results_root=manifest_path.parent, manifest=manifest,
              base_config=Path(manifest["worker_base_config"]), executable=Path(manifest["solver"]["path"]),
              device=manifest["solver"]["device_index"],
              floquet_mode=manifest["solver"]["floquet_mode"], create_initial_state=create_init_state)
        manifest["status"] = "completed"
        manifest["finished_utc"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        print("Assigned points completed.", flush=True)
        return 0
    except BaseException as error:
        manifest["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        manifest["finished_utc"] = utc_now()
        write_json_atomic(manifest_path, manifest)
        (manifest_path.parent / "worker_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-manifest", type=Path, required=True)
    raise SystemExit(worker_main(parser.parse_args().worker_manifest))
