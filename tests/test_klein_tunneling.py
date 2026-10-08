"""Analytic, band-physics and failure-path checks for the lattice workflow."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from klein_tunneling.analyze import analyze, bin_average, summarize_series
from klein_tunneling.model import (Settings, band_report, bloch_hamiltonian, initial_packet,
                                  physical_units, potential, probabilities, tune_second_depth)
from klein_tunneling.reference import lattice_evolution
from klein_tunneling.workflow import (CORE, execute, prepare, read_config, read_field,
                                     sha256_file, write_field)


class BandPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings()
        cls.second = tune_second_depth(5.0, 10)

    def test_free_bands_are_folded_plane_wave_energies(self):
        q = 0.17
        expected = np.sort((q+2*np.arange(-10, 11))**2)
        np.testing.assert_allclose(np.linalg.eigvalsh(bloch_hamiltonian(q, 0, 0, 0, 10)), expected)

    def test_phase_couplings_are_hermitian_and_match_real_space_potential(self):
        matrix = bloch_hamiltonian(0.12, 5, self.second, 0.7, 10)
        np.testing.assert_array_equal(matrix, matrix.conj().T)
        settings = replace(self.settings, absorber_strength_er=0)
        coefficients = np.fft.fft(potential(settings, self.second, 0.7, 0))/settings.points
        self.assertAlmostEqual(abs(coefficients[settings.cells]-5/8), 0, places=12)
        self.assertAlmostEqual(abs(coefficients[2*settings.cells]-self.second/8*np.exp(0.7j)), 0, places=12)
        # Potential coefficients above are native energies; the Bloch matrix is in ER.
        self.assertAlmostEqual(abs(matrix[2, 0]/2-coefficients[2*settings.cells]), 0, places=12)

    def test_tuned_crossing_and_massive_control_converge_with_basis(self):
        self.assertAlmostEqual(self.second, 1.5625, places=6)
        for phase, minimum, maximum in ((math.pi, 0, 1e-6), (0, 1.3, 1.4)):
            small = np.linalg.eigvalsh(bloch_hamiltonian(0, 5, self.second, phase, 10))[1:3]
            large = np.linalg.eigvalsh(bloch_hamiltonian(0, 5, self.second, phase, 15))[1:3]
            self.assertTrue(minimum <= small[1]-small[0] <= maximum)
            np.testing.assert_allclose(small, large, atol=1e-10, rtol=0)

    def test_packet_is_normalized_localized_and_band_pure(self):
        settings = self.settings
        state = initial_packet(settings, self.second, 0)
        self.assertAlmostEqual(settings.dx*np.sum(abs(state)**2), 1, places=13)
        mean = np.sum(abs(state)**2*settings.grid())*settings.dx
        self.assertLess(abs(mean-settings.packet_center), 1)
        spectrum = np.fft.fft(state)
        qgrid = 2*np.pi*np.fft.fftfreq(settings.points, settings.dx)
        leakage, total = 0.0, 0.0
        n = np.arange(-10, 11)
        for index in np.flatnonzero((qgrid > 0) & (qgrid < 0.3)):
            q = qgrid[index]
            vector = spectrum[(index+n*settings.cells) % settings.points]*np.exp(0.5j*(q+2*n)*settings.length)
            _, eigenvectors = np.linalg.eigh(bloch_hamiltonian(q, 5, self.second, 0, 10))
            projected = eigenvectors.conj().T@vector
            total += np.vdot(vector, vector).real
            leakage += np.sum(abs(np.delete(projected, 2))**2)
        self.assertLess(leakage/total, 1e-20)
        self.assertGreater(band_report(settings, self.second, 0)['initial_group_velocity_native'], 0)

    def test_units_use_twice_recoil_energy(self):
        units = physical_units(self.settings)
        hbar = 1.054571817e-34
        mass = self.settings.atom_mass_amu*1.66053906660e-27
        self.assertAlmostEqual(units['energy_unit_j']/(hbar**2/(mass*units['length_unit_m']**2)), 1, places=13)
        self.assertAlmostEqual(units['energy_unit_j']*units['time_unit_s']/hbar, 1, places=13)

    def test_invalid_grid_and_geometry_are_rejected(self):
        for changed in (replace(self.settings, points=1024), replace(self.settings, cells=20),
                        replace(self.settings, packet_center=0), replace(self.settings, initial_q=0),
                        replace(self.settings, barrier_heights_er=(0.9, 0.9)), replace(self.settings, time_step=float('nan'))):
            with self.subTest(settings=changed), self.assertRaises(ValueError):
                changed.validate()


class EvolutionAndMeasurementTests(unittest.TestCase):
    def test_free_plane_wave_phase_and_snapshot_timing(self):
        settings = replace(Settings(), points=128, time_step=0.1, duration=0.4, saved_intervals=4)
        k = 2*np.pi*3/settings.length
        state = np.exp(1j*k*settings.grid())/np.sqrt(settings.length)
        samples = list(lattice_evolution(state, np.zeros(settings.points), settings))
        self.assertEqual([i for i, _ in samples], [0, 1, 2, 3, 4])
        for i, evolved in samples:
            np.testing.assert_allclose(evolved, state*np.exp(-0.5j*k*k*(i+1)*settings.time_step), atol=1e-15)

    def test_uniform_absorption_has_exact_norm_decay(self):
        settings = replace(Settings(), points=128, duration=0.04, saved_intervals=4)
        state = np.ones(settings.points, complex)/np.sqrt(settings.length)
        for i, evolved in lattice_evolution(state, np.full(settings.points, -0.3j), settings):
            self.assertAlmostEqual(np.sum(abs(evolved)**2)*settings.dx, np.exp(-0.6*(i+1)*settings.time_step), places=13)

    def test_probability_balance_uses_initial_norm_not_surviving_norm(self):
        settings = Settings()
        state = np.zeros(settings.points, complex)
        positions = [np.argmin(abs(settings.grid()-x)) for x in (-200, 0, 200)]
        state[positions] = np.sqrt(np.array([0.2, 0.1, 0.3])/settings.dx)
        result = probabilities(state, settings)
        for key, expected in (('left_probability', 0.2), ('unresolved_probability', 0.1),
                              ('right_probability', 0.3), ('absorbed_probability', 0.4)):
            self.assertAlmostEqual(result[key], expected, places=13)
        with self.assertRaises(ValueError):
            probabilities(state, settings, float('nan'))
        with self.assertRaises(ValueError):
            probabilities(2*state, settings)

    def test_short_run_cannot_mislabel_incoming_packet_as_reflection(self):
        row = {'time_native': 0.1, 'right_probability': 0, 'left_probability': 1,
               'unresolved_probability': 0, 'absorbed_probability': 0, 'remaining_probability': 1}
        self.assertFalse(summarize_series([row], Settings(), minimum_time_native=200)['measurement_complete'])
        row['time_native'] = 360
        self.assertTrue(summarize_series([row]*5, Settings(), minimum_time_native=200)['measurement_complete'])

    def test_loss_unresolved_mass_and_temporal_drift_each_block_measurement(self):
        row = {'time_native': 360, 'right_probability': 0.6, 'left_probability': 0.4,
               'unresolved_probability': 0, 'absorbed_probability': 0, 'remaining_probability': 1}
        for key in ('unresolved_probability', 'absorbed_probability'):
            with self.subTest(key=key):
                self.assertFalse(summarize_series([{**row, key: 0.1}]*5, Settings(), minimum_time_native=200)['measurement_complete'])
        self.assertFalse(summarize_series([{**row, 'right_probability': 0.1}, row], Settings(), minimum_time_native=200)['measurement_complete'])

    def test_plot_downsampling_preserves_constants_with_irregular_grid(self):
        result = bin_average(np.ones(12346)*7)
        self.assertEqual(result.shape, (1024,))
        np.testing.assert_array_equal(result, 7)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = replace(Settings(), cells=256, points=8192, plane_waves_each_side=8,
                                absorber_width=32, duration=0.02, saved_intervals=2,
                                phases_radians=(0.0, math.pi), barrier_heights_er=(0.0, 0.9))

    def tearDown(self):
        self.temp.cleanup()

    def test_case_configs_are_private_static_and_original_is_untouched(self):
        before = sha256_file(CORE/'gpe1d.config')
        manifest = prepare(self.settings, self.root/'dataset', backend='numpy')
        configs = [read_config(self.root/'dataset'/c['directory']/'run.config') for c in manifest['cases']]
        self.assertEqual(len({c['initial_state_file'] for c in configs}), 4)
        self.assertEqual(len({c['output_folder'] for c in configs}), 4)
        for config in configs:
            self.assertEqual(config['beta'], '0')
            self.assertEqual(config['floquet_potential'], 'false')
            self.assertEqual(config['imaginary_time'], 'false')
            self.assertEqual(config['phase_metric_file'], '')
        self.assertEqual(before, sha256_file(CORE/'gpe1d.config'))
        with self.assertRaises(FileExistsError):
            prepare(self.settings, self.root/'dataset', backend='numpy')

    def completed_dataset(self):
        directory = self.root/'dataset'
        manifest = prepare(self.settings, directory, backend='numpy')
        execute(directory, manifest, executable=None, device=0)
        return directory, manifest

    def test_end_to_end_outputs_and_early_measurement_flags(self):
        directory, manifest = self.completed_dataset()
        outputs = analyze(directory/'run_manifest.json', make_plots=False)
        matrices = np.load(outputs['npz'])
        self.assertEqual(matrices['transmission'].shape, (2, 2))
        self.assertFalse(np.any(matrices['measurement_complete']))
        series = (directory/'case_0000'/'probability_timeseries.csv').read_text().splitlines()
        self.assertEqual(len(series), 4)
        self.assertIn('0,0.01,', series[1])
        self.assertEqual(manifest['status'], 'completed')

    def test_missing_snapshot_and_modified_input_are_rejected(self):
        directory, manifest = self.completed_dataset()
        snapshot = directory/'case_0000'/'out'/'0000000000000000.h5'
        snapshot.unlink()
        with self.assertRaisesRegex(ValueError, 'snapshots'):
            analyze(directory/'run_manifest.json', make_plots=False)
        write_field(snapshot, read_field(directory/'case_0000'/'initial.h5', self.settings.points))
        write_field(directory/'case_0000'/'potential.h5', np.zeros(self.settings.points, complex))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            analyze(directory/'run_manifest.json', make_plots=False)

    def test_batch_failure_preserves_failed_manifest_and_private_inputs(self):
        directory = self.root/'failed'
        manifest = prepare(self.settings, directory, backend='cuda', solver_info={'max_batch_systems': 4, 'fused_solver_version': 1})
        with patch('klein_tunneling.workflow.run_logged', return_value=7) as runner:
            with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
                execute(directory, manifest, executable=Path('solver.exe'), device=0, batch_size=4)
        self.assertEqual(len([v for v in runner.call_args.args[0] if v.endswith('run.config')]), 4)
        retained = json.loads((directory/'run_manifest.json').read_text())
        self.assertEqual(retained['status'], 'failed')
        self.assertEqual(retained['completed_cases'], 0)
        self.assertTrue(all(c['status'] == 'failed' for c in retained['cases']))
        self.assertTrue((directory/'case_0003'/'initial.h5').is_file())

    def test_corrupted_units_cannot_produce_plausible_physical_axes(self):
        directory, manifest = self.completed_dataset()
        manifest['units']['time_unit_s'] *= 2
        (directory/'run_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'physical units'):
            analyze(directory/'run_manifest.json', make_plots=False)


if __name__ == '__main__':
    unittest.main()
