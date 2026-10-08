"""Private native-solver inputs and auditable barrier-scattering sweeps."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import tables as tb

from .model import Settings, band_report, initial_packet, physical_units, potential, tune_second_depth
from .reference import lattice_evolution

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'phase_diagram'/'simulation_core'
sys.path.insert(0, str(CORE))
from cuda_workflow_common import git_provenance, locate_executable, query_json, read_config, run_logged, sha256_file, utc_now, write_json_atomic


def load_settings(path: Path) -> Settings:
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('Settings must be a JSON object.')
    settings = Settings(**values)
    settings.validate()
    return settings


def resolve_model(settings: Settings) -> tuple[float, list[dict]]:
    settings.validate()
    second = settings.second_depth_er
    if second is None:
        second = tune_second_depth(settings.first_depth_er, settings.plane_waves_each_side)
    reports = [band_report(settings, second, phase) for phase in settings.phases_radians]
    for report in reports:
        if report['initial_group_velocity_native'] <= 0:
            raise ValueError('The chosen incoming Bloch state is not right-moving.')
    return float(second), reports


def print_plan(settings: Settings, second: float, reports: list[dict]) -> None:
    count = len(settings.phases_radians)*len(settings.barrier_heights_er)
    times = list(range(0, settings.steps, settings.save_every))
    print(f'Kapitza GPU 3.0: {count} optical-lattice scattering cases')
    print(f'V1={settings.first_depth_er:g} ER; V2={second:.12g} ER; incoming band=2; q={settings.initial_q:g} kL')
    print(f'Grid: {settings.points} points, {settings.cells} cells; dx={settings.dx:.8g}; dt={settings.time_step:g}')
    print(f'{settings.steps} steps; {len(times)} snapshots; last saved time={(times[-1]+1)*settings.time_step:g} hbar/(2 ER)')
    for report in reports:
        print(f"  phase={report['phase_radians']:.8g}: gap={report['gap_er']:.8g} ER; incident relative energy={report['initial_energy_relative_er']:.8g} ER")
    print('Barrier heights (ER):', ', '.join(f'{v:g}' for v in settings.barrier_heights_er))


def write_field(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tb.open_file(path, 'w') as handle:
        handle.create_array('/', 'REAL', np.asarray(values.real, dtype=np.float64))
        handle.create_array('/', 'IMAGINARY', np.asarray(values.imag, dtype=np.float64))


def read_field(path: Path, points: int) -> np.ndarray:
    with tb.open_file(path, 'r') as handle:
        real, imag = handle.root.REAL[:], handle.root.IMAGINARY[:]
    if real.shape != (points,) or imag.shape != (points,) or not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError(f'Invalid wavefunction snapshot: {path}')
    return real+1j*imag


def prepare(settings: Settings, results: Path, *, backend: str, solver_info: dict | None = None,
            device_info: dict | None = None) -> dict:
    second, reports = resolve_model(settings)
    results = results.resolve()
    if results.exists() and any(results.iterdir()):
        raise FileExistsError(f'Refusing to reuse a nonempty results directory: {results}')
    results.mkdir(parents=True, exist_ok=True)
    manifest = {
        'workflow': 'bichromatic_optical_lattice_klein', 'manifest_version': 1,
        'status': 'prepared', 'created_utc': utc_now(), 'backend': backend,
        'settings': asdict(settings), 'resolved_second_depth_er': second,
        'bands': reports, 'units': physical_units(settings),
        'solver_build': solver_info, 'device_info': device_info,
        'git': git_provenance(ROOT), 'model_source_sha256': sha256_file(Path(__file__).with_name('model.py')),
        'snapshot_indices': list(range(0, settings.steps, settings.save_every)),
        'sampling_convention': 'saved iteration n is after (n+1) complete time steps',
        'cases': [], 'completed_cases': 0,
        'physics_reference': 'https://arxiv.org/abs/1108.4447',
        'measurement': 'left/right/central surviving probability divided by initial total norm; absorption accounted separately',
    }
    for phase_index, report in enumerate(reports):
        psi = initial_packet(settings, second, report['phase_radians'])
        initial_norm = float(settings.dx*np.sum(np.abs(psi)**2))
        for height_index, height in enumerate(settings.barrier_heights_er):
            index = len(manifest['cases'])
            directory = results/f'case_{index:04d}'
            directory.mkdir()
            state_path, potential_path = directory/'initial.h5', directory/'potential.h5'
            write_field(state_path, psi)
            write_field(potential_path, potential(settings, second, report['phase_radians'], height))
            config = read_config(CORE/'gpe1d.config')
            config.update({
                'dimension': 1, 'points_x': settings.points, 'step_x': settings.dx,
                'time_step': settings.time_step, 'number_of_iterations': settings.steps,
                'save_every_nth_iteration': settings.save_every,
                'show_stats_every_nth_iteration': settings.save_every,
                'imaginary_time': 'false', 'dynamic_potential': 'false',
                'floquet_potential': 'false', 'floquet_omega': 0, 'beta': 0,
                'initial_state_file': state_path, 'potential_file': potential_path,
                'output_folder': directory/'out', 'status_file': directory/'status.csv',
                'phase_metric_file': '', 'save_psi': 'true', 'dp_save_potential': 'false',
            })
            config_path = directory/'run.config'
            config_path.write_text(''.join(f'{key}={value}\n' for key, value in config.items()), encoding='utf-8')
            manifest['cases'].append({
                'run_index': index, 'phase_index': phase_index, 'height_index': height_index,
                'phase_radians': report['phase_radians'], 'barrier_height_er': float(height),
                'gap_er': report['gap_er'], 'initial_energy_relative_er': report['initial_energy_relative_er'],
                'initial_norm': initial_norm, 'directory': directory.name, 'status': 'prepared',
                'input_sha256': {'initial.h5': sha256_file(state_path), 'potential.h5': sha256_file(potential_path)},
                'config_sha256': sha256_file(config_path),
            })
    write_json_atomic(results/'run_manifest.json', manifest)
    return manifest


def execute(results: Path, manifest: dict, *, executable: Path | None, device: int,
            batch_size: int = 4) -> None:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    settings = Settings(**manifest['settings'])
    settings.validate()
    backend = manifest['backend']
    if backend not in ('cuda', 'numpy'):
        raise ValueError('Unknown evolution backend.')
    info = manifest.get('solver_build') or {}
    max_batch = info.get('max_batch_systems', 1)
    if type(max_batch) is not int or max_batch < 1:
        max_batch = 1
    actual_batch = min(batch_size, max_batch) if backend == 'cuda' else 1
    manifest.update(status='running', started_utc=utc_now(), batch_size=actual_batch)
    write_json_atomic(results/'run_manifest.json', manifest)
    try:
        for start in range(0, len(manifest['cases']), actual_batch):
            cases = manifest['cases'][start:start+actual_batch]
            for case in cases:
                case.update(status='running', started_utc=utc_now())
            write_json_atomic(results/'run_manifest.json', manifest)
            print(f"Running cases {start+1}-{start+len(cases)} of {len(manifest['cases'])} ({backend})", flush=True)
            if backend == 'cuda':
                if executable is None:
                    raise ValueError('A CUDA executable is required.')
                paths = [str(results/case['directory']/'run.config') for case in cases]
                command = [str(executable), *paths, '--device', str(device), '--floquet-mode', 'physical']
                if info.get('fused_solver_version') == 1:
                    command += ['--wait', 'blocking']
                log_path = results/f'batch_{start:04d}.log'
                for case in cases:
                    case['solver_log'] = log_path.name
                code = run_logged(command, ROOT, log_path)
                if code:
                    raise RuntimeError(f'CUDA scattering batch failed with exit code {code}; see {log_path}')
            else:
                case = cases[0]
                directory = results/case['directory']
                psi = read_field(directory/'initial.h5', settings.points)
                field = read_field(directory/'potential.h5', settings.points)
                for iteration, state in lattice_evolution(psi, field, settings):
                    write_field(directory/'out'/f'{iteration:016d}.h5', state)
            for case in cases:
                case.update(status='completed', finished_utc=utc_now())
                manifest['completed_cases'] += 1
            write_json_atomic(results/'run_manifest.json', manifest)
        manifest.update(status='completed', finished_utc=utc_now())
    except BaseException as error:
        manifest.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        for case in manifest['cases']:
            if case['status'] == 'running':
                case.update(status=manifest['status'], error=str(error))
        raise
    finally:
        write_json_atomic(results/'run_manifest.json', manifest)


def default_results() -> Path:
    return ROOT/'klein_tunneling'/'results'/datetime.now(timezone.utc).strftime('klein_%Y%m%dT%H%M%S%fZ')
