"""Transmission, reflection and loss without renormalizing away absorption."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math

import numpy as np

from .model import Settings, bloch_hamiltonian, physical_units, probabilities
from .workflow import ROOT, read_field, sha256_file, write_json_atomic


def case_directory(root: Path, name: str) -> Path:
    path = (root/name).resolve()
    if path.parent != root.resolve():
        raise ValueError('Case paths must be immediate children of the dataset directory.')
    return path


def summarize_series(series: list[dict], settings: Settings, *, minimum_time_native: float) -> dict:
    last = series[-1]
    tail = [row['right_probability'] for row in series[-5:]]
    drift = float(np.ptp(tail))
    complete = (last['time_native'] >= minimum_time_native and
                last['unresolved_probability'] <= settings.max_unresolved_probability and
                last['absorbed_probability'] <= settings.max_absorbed_probability and
                drift <= settings.max_transmission_drift)
    return {
        'transmission': last['right_probability'], 'reflection': last['left_probability'],
        'unresolved_probability': last['unresolved_probability'],
        'absorbed_probability': last['absorbed_probability'],
        'remaining_probability': last['remaining_probability'],
        'transmission_drift': drift, 'transmission_time_std': float(np.std(tail, ddof=0)),
        'measurement_complete': bool(complete), 'final_time_native': last['time_native'],
        'minimum_measurement_time_native': minimum_time_native,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def bin_average(values: np.ndarray) -> np.ndarray:
    """Reduce plots to at most 1024 bins, including grids of arbitrary size."""
    edges = np.linspace(0, len(values), min(1024, len(values))+1, dtype=int)
    return np.add.reduceat(values, edges[:-1])/np.diff(edges)


def analyze(manifest_path: Path, *, make_plots: bool = True) -> dict[str, Path]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('workflow') != 'bichromatic_optical_lattice_klein' or manifest.get('manifest_version') != 1:
        raise ValueError('Unsupported Klein-tunneling dataset.')
    if manifest.get('status') != 'completed':
        raise ValueError('Transmission analysis requires a completed sweep.')
    settings = Settings(**manifest['settings'])
    settings.validate()
    expected = list(range(0, settings.steps, settings.save_every))
    if manifest['snapshot_indices'] != expected:
        raise ValueError('Manifest snapshot schedule does not match the simulation settings.')
    expected_cases = len(settings.phases_radians)*len(settings.barrier_heights_er)
    if len(manifest['cases']) != expected_cases or manifest['completed_cases'] != expected_cases:
        raise ValueError('The scan is missing cases.')
    seen = set()
    rows = []
    saved_series = []
    x = settings.grid()
    units = physical_units(settings)
    if any(not math.isclose(manifest['units'][key], value, rel_tol=1e-12, abs_tol=0)
           for key, value in units.items()):
        raise ValueError('Recorded physical units do not match the settings.')
    if (len(manifest['bands']) != len(settings.phases_radians) or
        any(report['phase_radians'] != phase or report['prepared_band_index'] != 2 or
            not math.isfinite(report['initial_group_velocity_native']) or report['initial_group_velocity_native'] <= 0
            for report, phase in zip(manifest['bands'], settings.phases_radians))):
        raise ValueError('Invalid band metadata.')
    for case in manifest['cases']:
        index = case['run_index']
        if type(index) is not int or index in seen or not 0 <= index < expected_cases:
            raise ValueError('Duplicate or invalid case index.')
        seen.add(index)
        phase_index, height_index = divmod(index, len(settings.barrier_heights_er))
        if (case['status'] != 'completed' or case['phase_index'] != phase_index or
            case['height_index'] != height_index or case['phase_radians'] != settings.phases_radians[phase_index] or
            case['barrier_height_er'] != settings.barrier_heights_er[height_index]):
            raise ValueError('Case parameters do not match the scan.')
        directory = case_directory(root, case['directory'])
        for name, digest in case['input_sha256'].items():
            if name not in ('initial.h5', 'potential.h5') or sha256_file(directory/name) != digest:
                raise ValueError('Saved input provenance does not match the dataset.')
        if set(case['input_sha256']) != {'initial.h5', 'potential.h5'} or sha256_file(directory/'run.config') != case['config_sha256']:
            raise ValueError('Saved configuration provenance does not match the dataset.')
        paths = {p.name: p for p in (directory/'out').glob('*.h5')}
        names = [f'{iteration:016d}.h5' for iteration in expected]
        if set(paths) != set(names):
            raise ValueError(f'Case {index} has missing or unexpected snapshots.')
        initial = read_field(directory/'initial.h5', settings.points)
        initial_norm = float(settings.dx*np.sum(np.abs(initial)**2))
        if not math.isclose(initial_norm, case['initial_norm'], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('The recorded initial norm is incorrect.')
        series, density_frames = [], []
        previous_norm = 1.0
        for iteration, name in zip(expected, names):
            state = read_field(paths[name], settings.points)
            row = {'iteration': iteration, 'time_native': (iteration+1)*settings.time_step,
                   **probabilities(state, settings, initial_norm)}
            if row['remaining_probability'] > previous_norm + 1e-8:
                raise ValueError('Probability increased during absorbing evolution.')
            previous_norm = row['remaining_probability']
            mass = np.abs(state)**2
            norm = float(np.sum(mass))
            center = float(np.sum(mass*x)/norm) if norm else None
            row['x_mean'] = center
            row['sigma_x_squared'] = float(np.sum(mass*(x-center)**2)/norm) if norm else None
            series.append(row)
            density_frames.append(bin_average(mass/initial_norm))
        band = manifest['bands'][phase_index]
        minimum_time = (-settings.packet_center+3*settings.packet_sigma+settings.partition_edge)/band['initial_group_velocity_native']
        summary = summarize_series(series, settings, minimum_time_native=minimum_time)
        row = {
            'run_index': index, 'phase_radians': case['phase_radians'],
            'first_depth_er': settings.first_depth_er,
            'second_depth_er': manifest['resolved_second_depth_er'],
            'barrier_height_er': case['barrier_height_er'],
            'barrier_width': settings.barrier_width, 'barrier_edge': settings.barrier_edge,
            'initial_q': settings.initial_q, 'gap_er': case['gap_er'],
            'initial_energy_relative_er': case['initial_energy_relative_er'],
            **summary, 'final_time_ms': summary['final_time_native']*units['time_unit_s']*1000,
            'x_mean': series[-1]['x_mean'], 'sigma_x_squared': series[-1]['sigma_x_squared'],
        }
        rows.append(row)
        write_csv(directory/'probability_timeseries.csv', series)
        write_json_atomic(directory/'scattering_summary.json', row)
        saved_series.append((case, series, np.asarray(density_frames)))
    rows.sort(key=lambda row: row['run_index'])
    write_csv(root/'transmission.csv', rows)
    shape = (len(settings.phases_radians), len(settings.barrier_heights_er))
    fields = ('transmission', 'reflection', 'unresolved_probability', 'absorbed_probability', 'measurement_complete')
    matrices = {key: np.array([row[key] for row in rows]).reshape(shape) for key in fields}
    np.savez(root/'transmission.npz', phases_radians=settings.phases_radians,
             barrier_heights_er=settings.barrier_heights_er, **matrices)
    incomplete = [row['run_index'] for row in rows if not row['measurement_complete']]
    write_json_atomic(root/'analysis_report.json', {
        'status': 'completed', 'incomplete_measurement_cases': incomplete,
        'definition': 'T + R + unresolved + absorbed = 1 (up to numerical roundoff); no surviving-norm renormalization',
        'note': 'A complete measurement describes finite-time scattering. It does not by itself prove a Dirac approximation or Klein tunneling.',
    })
    outputs = {'csv': root/'transmission.csv', 'npz': root/'transmission.npz'}
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        q = np.linspace(-0.3, 0.3, 161)
        ordered_bands = sorted(manifest['bands'], key=lambda report: report['gap_er'])
        illustrated_bands = [ordered_bands[0]]
        if len(ordered_bands) > 1:
            illustrated_bands.append(ordered_bands[-1])
        for axis, report in zip(axes[:2], illustrated_bands):
            energies = np.array([np.linalg.eigvalsh(bloch_hamiltonian(momentum, settings.first_depth_er,
                                 manifest['resolved_second_depth_er'], report['phase_radians'], settings.plane_waves_each_side))[1:3]
                                 for momentum in q])-report['crossing_energy_er']
            axis.plot(q, energies[:, 0], label='lower band')
            axis.plot(q, energies[:, 1], label='upper band')
            axis.set(title=f"Phase {report['phase_radians']:.3g}; gap {report['gap_er']:.3g} ER",
                     xlabel='Quasimomentum / kL', ylabel='Energy from crossing / ER')
        if len(illustrated_bands) == 1:
            axes[1].set_visible(False)
        for report in manifest['bands']:
            selection = [row for row in rows if row['phase_radians'] == report['phase_radians']]
            axes[2].plot([row['barrier_height_er'] for row in selection], [row['transmission'] for row in selection],
                         'o-', label=f"phase {report['phase_radians']:.3g}")
            failed = [row for row in selection if not row['measurement_complete']]
            if failed:
                axes[2].scatter([row['barrier_height_er'] for row in failed], [row['transmission'] for row in failed],
                                marker='x', s=90, color='red', label='measurement incomplete')
        axes[2].set(xlabel='Barrier height / ER', ylabel='Transmitted probability', ylim=(-0.03, 1.03))
        axes[2].legend()
        figure.savefig(root/'bands_and_transmission.png', dpi=180)
        plt.close(figure)
        representatives = []
        for report in illustrated_bands:
            phase = report['phase_radians']
            selection = [item for item in saved_series if item[0]['phase_radians'] == phase and item[0]['barrier_height_er'] > 0]
            if selection:
                representatives.append(selection[0])
        if representatives:
            figure, axes = plt.subplots(1, len(representatives), figsize=(6*len(representatives), 4.5), squeeze=False, constrained_layout=True)
            positions = bin_average(x)*units['length_unit_m']*1e6
            max_density = max(float(np.max(item[2])) for item in representatives)
            from matplotlib.colors import PowerNorm
            for axis, (case, series, density) in zip(axes[0], representatives):
                times = [row['time_native']*units['time_unit_s']*1000 for row in series]
                plotted = axis.pcolormesh(positions, times, density, shading='nearest',
                                         norm=PowerNorm(0.5, vmin=0, vmax=max_density), cmap='magma')
                for edge in (-settings.barrier_width/2, settings.barrier_width/2):
                    axis.axvline(edge*units['length_unit_m']*1e6, color='cyan', linestyle='--', linewidth=0.8)
                axis.set(title=f"Phase {case['phase_radians']:.3g}; barrier {case['barrier_height_er']:g} ER",
                         xlabel='Position (µm)', ylabel='Time (ms)')
            figure.colorbar(plotted, ax=list(axes[0]), label='Density in native length units')
            figure.savefig(root/'wavepacket_scattering.png', dpi=180)
            plt.close(figure)
            outputs['wavepacket'] = root/'wavepacket_scattering.png'
        outputs['bands'] = root/'bands_and_transmission.png'
    print(f'Transmission CSV: {outputs["csv"]}')
    if incomplete:
        print(f'Measurement incomplete in cases {incomplete}: inspect flight time, unresolved probability, loss and drift before interpreting T/R.')
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', nargs='?', type=Path)
    args = parser.parse_args()
    path = args.manifest
    if path is None:
        candidates = sorted((ROOT/'klein_tunneling'/'results').glob('*/run_manifest.json'), reverse=True)
        path = next((p for p in candidates if json.loads(p.read_text())['status'] == 'completed'), None)
        if path is None:
            raise FileNotFoundError('No completed Klein-tunneling dataset exists.')
    analyze(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
