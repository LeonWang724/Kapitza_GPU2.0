"""Bloch bands, band-pure moving packets and native-solver unit conversion.

Physical model: Salger et al., PRL 107, 240401 (2011), arXiv:1108.4447.
V/ER = V1/2 cos(2X) + V2/2 cos(4X + theta) + B(X)/ER.
X=kL*x. The existing solver has kinetic -1/2 d_X^2, so its energy
unit is 2*ER and its time unit is hbar/(2*ER).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class Settings:
    first_depth_er: float = 5.0
    second_depth_er: float | None = None  # Tune the band-1/band-2 gap at theta=pi.
    phases_radians: tuple[float, ...] = (0.0, math.pi)
    barrier_heights_er: tuple[float, ...] = (0.0, 0.9, 1.2)
    barrier_width: float = 64.0
    barrier_edge: float = 8.0
    initial_q: float = 0.12  # In units of kL; positive velocity in band index 2.
    packet_sigma: float = 28.0  # Density envelope standard deviation in X.
    packet_center: float = -180.0
    cells: int = 512  # Integer lattice periods make FFT/Bloch momenta compatible.
    points: int = 16384
    plane_waves_each_side: int = 10
    time_step: float = 0.01
    duration: float = 360.0  # Requested duration in hbar/(2*ER).
    saved_intervals: int = 40
    absorber_width: float = 64.0
    absorber_strength_er: float = 2.0
    atom_mass_amu: float = 7.01600455
    wavelength_nm: float = 1064.0
    max_unresolved_probability: float = 0.01
    max_absorbed_probability: float = 0.005
    max_transmission_drift: float = 0.01

    def validate(self) -> None:
        integer_keys = ('cells', 'points', 'plane_waves_each_side', 'saved_intervals')
        for key in integer_keys:
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{key} must be a positive integer.')
        for key, value in asdict(self).items():
            if key in integer_keys or key in ('phases_radians', 'barrier_heights_er'):
                continue
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f'{key} must be finite.')
        positive = ('first_depth_er', 'barrier_width', 'barrier_edge', 'initial_q',
                    'packet_sigma', 'time_step', 'duration', 'absorber_width',
                    'atom_mass_amu', 'wavelength_nm')
        for key in positive:
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive.')
        if self.second_depth_er is not None and self.second_depth_er <= 0:
            raise ValueError('second_depth_er must be positive or null for automatic tuning.')
        if self.absorber_strength_er < 0:
            raise ValueError('absorber_strength_er must be nonnegative.')
        for key in ('max_unresolved_probability', 'max_absorbed_probability', 'max_transmission_drift'):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f'{key} must lie in [0,1).')
        for key in ('phases_radians', 'barrier_heights_er'):
            values = getattr(self, key)
            if not values or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
                raise ValueError(f'{key} must be a nonempty list of finite numbers.')
            if len(set(values)) != len(values):
                raise ValueError(f'{key} must not contain duplicates.')
        if any(v < 0 for v in self.barrier_heights_er):
            raise ValueError('Barrier heights must be nonnegative.')
        if not 0 < self.initial_q < 0.3 or self.initial_q * self.packet_sigma < 3:
            raise ValueError('Use 0 < initial_q < 0.3 and initial_q*packet_sigma >= 3 for a narrow right-moving packet near the crossing.')
        if self.points % 2 or self.points <= 2*self.cells*(self.plane_waves_each_side + 0.5):
            raise ValueError('The even FFT grid must resolve all plane-wave components; increase points or reduce cells.')
        if self.packet_center >= -self.partition_edge - 3*self.packet_sigma:
            raise ValueError('The initial packet must be at least three widths left of the barrier measurement region.')
        if abs(self.packet_center) + 6*self.packet_sigma >= self.length/2 - self.absorber_width:
            raise ValueError('Increase cells: the initial packet is too close to an absorber.')
        if self.partition_edge >= self.length/2 - self.absorber_width:
            raise ValueError('The barrier measurement region overlaps an absorber.')
        if self.steps < 2:
            raise ValueError('The evolution needs at least two time steps.')

    @property
    def length(self) -> float:
        return self.cells * math.pi

    @property
    def dx(self) -> float:
        return self.length / self.points

    @property
    def steps(self) -> int:
        return round(self.duration/self.time_step) + 1

    @property
    def save_every(self) -> int:
        return max(1, (self.steps-1)//self.saved_intervals)

    @property
    def partition_edge(self) -> float:
        return self.barrier_width/2 + 6*self.barrier_edge

    def grid(self) -> np.ndarray:
        return (np.arange(self.points)-self.points//2)*self.dx


def bloch_hamiltonian(q: float, first: float, second: float, phase: float, n_pw: int) -> np.ndarray:
    """Plane-wave matrix in ER, basis exp(i*(q+2*n)*X), n=-n_pw..n_pw."""
    n = np.arange(-n_pw, n_pw+1)
    matrix = np.diag((q+2*n)**2).astype(np.complex128)
    i = np.arange(len(n)-1)
    matrix[i+1, i] = matrix[i, i+1] = first/4
    i = np.arange(len(n)-2)
    matrix[i+2, i] = second/4*np.exp(1j*phase)
    matrix[i, i+2] = second/4*np.exp(-1j*phase)
    return matrix


def tune_second_depth(first: float, n_pw: int) -> float:
    estimate = first**2/16
    def gap(second):
        energies = np.linalg.eigvalsh(bloch_hamiltonian(0, first, second, math.pi, n_pw))
        return float(energies[2]-energies[1])
    result = minimize_scalar(gap, bounds=(0.5*estimate, 1.5*estimate),
                             method='bounded', options={'xatol': 1e-12})
    if not result.success or result.fun > 1e-5:
        raise ValueError('Could not close the excited-band gap; increase the plane-wave basis or choose lattice depths explicitly.')
    return float(result.x)


def band_report(settings: Settings, second: float, phase: float) -> dict:
    def spectrum(q):
        return np.linalg.eigvalsh(bloch_hamiltonian(q, settings.first_depth_er, second,
                                                   phase, settings.plane_waves_each_side))
    e0 = spectrum(0)
    qfit = np.linspace(0.005, 0.06, 12)
    halves = np.array([(spectrum(q)[2]-spectrum(q)[1])/2 for q in qfit])
    slope, _ = np.polyfit(qfit*qfit, halves*halves, 1)
    velocity = (spectrum(settings.initial_q+1e-5)[2]-spectrum(settings.initial_q-1e-5)[2])/4e-5
    center = float(np.mean(e0[1:3]))
    return {
        'phase_radians': float(phase), 'gap_er': float(e0[2]-e0[1]),
        'crossing_energy_er': center, 'dirac_speed_native': float(math.sqrt(max(0, slope))/2),
        'dirac_half_gap_native': float((e0[2]-e0[1])/4),
        'initial_energy_relative_er': float(spectrum(settings.initial_q)[2]-center),
        'initial_group_velocity_native': float(velocity), 'prepared_band_index': 2,
    }


def initial_packet(settings: Settings, second: float, phase: float) -> np.ndarray:
    """Superpose exact upper-band Bloch states, rather than merely adding a kick.

    The Gaussian quasimomentum amplitudes give a Gaussian density envelope.
    Smooth eigenvector phases prevent arbitrary eigensolver signs from creating
    artificial spatial structure. Every populated state is in band index 2.
    """
    settings.validate()
    qgrid = 2*math.pi*np.fft.fftfreq(settings.points, settings.dx)
    selected = np.flatnonzero((qgrid > 0) & (qgrid < 0.3) &
                              (np.abs(qgrid-settings.initial_q)*settings.packet_sigma < 4))
    selected = selected[np.argsort(qgrid[selected])]
    spectrum = np.zeros(settings.points, dtype=np.complex128)
    previous = None
    n = np.arange(-settings.plane_waves_each_side, settings.plane_waves_each_side+1)
    for index in selected:
        q = qgrid[index]
        _, vectors = np.linalg.eigh(bloch_hamiltonian(q, settings.first_depth_er, second,
                                                     phase, settings.plane_waves_each_side))
        vector = vectors[:, 2]
        overlap = vector[np.argmax(np.abs(vector))] if previous is None else np.vdot(previous, vector)
        vector = vector*np.exp(-1j*np.angle(overlap))
        previous = vector
        amplitude = np.exp(-settings.packet_sigma**2*(q-settings.initial_q)**2 - 1j*q*settings.packet_center)
        bins = index+n*settings.cells
        if np.any(np.abs(bins) >= settings.points//2):
            raise ValueError('The plane-wave basis exceeds the FFT Nyquist momentum.')
        momenta = q+2*n
        spectrum[bins % settings.points] += amplitude*vector*np.exp(-0.5j*momenta*settings.length)
    psi = np.fft.ifft(spectrum)
    norm = float(settings.dx*np.sum(np.abs(psi)**2))
    if norm <= 0:
        raise ValueError('No wavepacket momenta fit the grid; increase cells.')
    return psi/math.sqrt(norm)


def barrier_profile(x: np.ndarray, height_er: float, settings: Settings) -> np.ndarray:
    return height_er/2*(np.tanh((x+settings.barrier_width/2)/settings.barrier_edge)
                       - np.tanh((x-settings.barrier_width/2)/settings.barrier_edge))


def potential(settings: Settings, second: float, phase: float, height_er: float) -> np.ndarray:
    x = settings.grid()
    value = (settings.first_depth_er/2*np.cos(2*x)
             + second/2*np.cos(4*x+phase) + barrier_profile(x, height_er, settings))/2
    distance = np.maximum(0, np.abs(x)-(settings.length/2-settings.absorber_width))
    absorber = settings.absorber_strength_er/2*(distance/settings.absorber_width)**4
    return value.astype(np.complex128)-1j*absorber


def probabilities(psi: np.ndarray, settings: Settings, initial_norm: float = 1.0) -> dict:
    if psi.shape != (settings.points,) or not np.all(np.isfinite(psi)) or not math.isfinite(initial_norm) or initial_norm <= 0:
        raise ValueError('Invalid wavefunction or initial norm.')
    density = np.abs(psi)**2*settings.dx/initial_norm
    x = settings.grid()
    left, right = x < -settings.partition_edge, x > settings.partition_edge
    norm = float(np.sum(density))
    if norm > 1+1e-8:
        raise ValueError('Probability grew despite a purely absorbing boundary.')
    return {'left_probability': float(np.sum(density[left])),
            'right_probability': float(np.sum(density[right])),
            'unresolved_probability': float(np.sum(density[~(left | right)])),
            'absorbed_probability': max(0.0, 1-norm), 'remaining_probability': norm}


def physical_units(settings: Settings) -> dict:
    hbar, amu = 1.054571817e-34, 1.66053906660e-27
    wavevector = 2*math.pi/(settings.wavelength_nm*1e-9)
    recoil = hbar*hbar*wavevector*wavevector/(2*settings.atom_mass_amu*amu)
    return {'length_unit_m': 1/wavevector, 'energy_unit_j': 2*recoil,
            'time_unit_s': hbar/(2*recoil), 'recoil_frequency_hz': recoil/(2*math.pi*hbar)}
