"""Check scattering controls, an effective Dirac benchmark and grid convergence.

Default: exercise the CUDA executable, including a short wavefunction comparison
against NumPy. --backend numpy validates the model on a CPU, not CUDA hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from .analyze import analyze, summarize_series
from .model import Settings, probabilities
from .reference import dirac_evolution, lattice_evolution
from .workflow import (ROOT, default_results, execute, locate_executable, prepare,
                       query_json, read_field, resolve_model, write_json_atomic)


def validate(output: Path, backend: str, executable: Path | None, device: int,
             baseline: Path | None = None) -> dict:
    settings = Settings(barrier_heights_er=(0.0, 0.9))
    second, bands = resolve_model(settings)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Validation output must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    info = query_json(executable, '--version-json') if executable else None
    device_info = query_json(executable, '--device-info', str(device)) if executable else None
    report = {'status': 'running', 'backend': backend, 'solver_build': info,
              'device_info': device_info, 'settings': asdict(settings), 'checks': {},
              'cuda_verified': False}

    def check(name: str, passed: bool, **details) -> None:
        report['checks'][name] = {'passed': bool(passed), **details}
        write_json_atomic(output/'validation_report.json', report)
        print(f'{name}: {"PASS" if passed else "FAIL"}', flush=True)
        if not passed:
            raise AssertionError(f'{name} failed; see {output / "validation_report.json"}')

    def sweep(case_settings: Settings, name: str) -> tuple[dict, dict]:
        path = output/name
        manifest = prepare(case_settings, path, backend=backend, solver_info=info, device_info=device_info)
        execute(path, manifest, executable=executable, device=device, batch_size=4)
        analyze(path/'run_manifest.json', make_plots=False)
        summaries = {(c['phase_radians'], c['barrier_height_er']):
                     json.loads((path/c['directory']/'scattering_summary.json').read_text())
                     for c in manifest['cases']}
        return manifest, summaries

    try:
        if backend == 'cuda':
            short = replace(settings, duration=0.05, saved_intervals=5)
            short_manifest, _ = sweep(short, 'cuda_wavefunction_check')
            errors = []
            for case in short_manifest['cases']:
                directory = output/'cuda_wavefunction_check'/case['directory']
                initial = read_field(directory/'initial.h5', short.points)
                potential = read_field(directory/'potential.h5', short.points)
                for iteration, cpu in lattice_evolution(initial, potential, short):
                    gpu = read_field(directory/'out'/f'{iteration:016d}.h5', short.points)
                    errors.append(float(np.sqrt(short.dx*np.sum(np.abs(gpu-cpu)**2))))
            check('cuda_vs_numpy_wavefunction', max(errors) < 1e-10,
                  max_l2_error=max(errors), tolerance=1e-10)
        else:
            report['checks']['cuda_vs_numpy_wavefunction'] = {'status': 'not_run', 'reason': 'CPU backend selected'}

        if baseline:
            path = baseline.resolve()
            manifest = json.loads(path.read_text())
            baseline_settings = Settings(**manifest['settings'])
            # Reuse only a matching, complete, provenance-checked starter sweep.
            differing = {k for k, v in asdict(settings).items()
                         if k != 'barrier_heights_er' and v != asdict(baseline_settings)[k]
                         and not (k == 'phases_radians' and list(v) == asdict(baseline_settings)[k])}
            if differing or manifest['backend'] != backend or not {0.0, 0.9}.issubset(baseline_settings.barrier_heights_er):
                raise ValueError(f'Baseline settings/backend do not match validation: {differing}')
            analyze(path, make_plots=False)
            summaries = {(c['phase_radians'], c['barrier_height_er']):
                         json.loads((path.parent/c['directory']/'scattering_summary.json').read_text())
                         for c in manifest['cases']}
            report['baseline_manifest'] = str(path)
        else:
            _, summaries = sweep(settings, 'scattering_controls')
        selected = [summaries[(phase, height)] for phase in settings.phases_radians
                    for height in settings.barrier_heights_er]
        check('completed_probability_measurements', all(r['measurement_complete'] for r in selected),
              cases=selected)
        check('probability_balance', all(abs(r['transmission']+r['reflection']+
              r['unresolved_probability']+r['absorbed_probability']-1) < 1e-8 for r in selected), tolerance=1e-8)
        check('free_propagation_controls', all(summaries[(phase, 0.0)]['transmission'] > 0.99
              for phase in settings.phases_radians), minimum_transmission=0.99)
        check('gapless_transmission_and_gapped_reflection',
              summaries[(math.pi, 0.9)]['transmission'] > 0.98 and summaries[(0.0, 0.9)]['transmission'] < 0.02,
              gapless_transmission=summaries[(math.pi, 0.9)]['transmission'],
              gapped_transmission=summaries[(0.0, 0.9)]['transmission'])

        # A coarse envelope grid suffices for the two-component Dirac benchmark.
        envelope = replace(settings, points=4096, plane_waves_each_side=2)
        for band in bands:
            phase = band['phase_radians']
            series = [{'iteration': i, 'time_native': (i+1)*envelope.time_step,
                       **probabilities(state, envelope)}
                      for i, state in dirac_evolution(envelope, 0.9, band['dirac_speed_native'], band['dirac_half_gap_native'])]
            speed, delta = band['dirac_speed_native'], band['dirac_half_gap_native']
            group_velocity = speed**2*envelope.initial_q/np.hypot(speed*envelope.initial_q, delta)
            minimum_time = (-envelope.packet_center+3*envelope.packet_sigma+envelope.partition_edge)/group_velocity
            summary = summarize_series(series, envelope, minimum_time_native=minimum_time)
            difference = abs(summary['transmission']-summaries[(phase, 0.9)]['transmission'])
            check(f'effective_dirac_phase_{phase:.5g}', summary['measurement_complete'] and difference < 0.02,
                  dirac_summary=summary, transmission_difference=difference, tolerance=0.02)

        for name, changed in (('half_time_step', replace(settings, time_step=settings.time_step/2,
                              duration=settings.duration+settings.time_step-settings.time_step/2)),
                              ('double_spatial_resolution', replace(settings, points=settings.points*2))):
            _, refined = sweep(replace(changed, barrier_heights_er=(0.9,)), name)
            differences = {str(phase): abs(refined[(phase, 0.9)]['transmission']-
                           summaries[(phase, 0.9)]['transmission']) for phase in settings.phases_radians}
            check(name, all(r['measurement_complete'] for r in refined.values()) and max(differences.values()) < 0.005,
                  transmission_differences=differences, tolerance=0.005)
        report.update(status='passed', cuda_verified=(backend == 'cuda'))
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        write_json_atomic(output/'validation_report.json', report)
    print(f'Validation report: {output / "validation_report.json"}')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cuda', 'numpy'), default='cuda')
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--baseline', type=Path, help='reuse a matching completed starter run_manifest.json')
    args = parser.parse_args()
    if args.device < 0:
        parser.error('device must be nonnegative.')
    executable = locate_executable(args.executable) if args.backend == 'cuda' else None
    output = args.results or ROOT/'validation'/'results'/('klein_'+default_results().name)
    validate(output, args.backend, executable, args.device, args.baseline)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
