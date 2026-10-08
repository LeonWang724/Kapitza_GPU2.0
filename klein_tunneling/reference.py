"""Independent CPU split-step references for numerical and physical validation."""

from __future__ import annotations

import numpy as np

from .model import Settings, barrier_profile


def lattice_evolution(psi, potential, settings: Settings):
    """Native schedule: iteration zero is after the first complete time step."""
    state = np.asarray(psi, dtype=np.complex128).copy()
    k = 2*np.pi*np.fft.fftfreq(settings.points, settings.dx)
    half = np.exp(-0.5j*settings.time_step*potential)
    kinetic = np.exp(-0.5j*settings.time_step*k*k)
    for iteration in range(settings.steps):
        state *= half
        state = np.fft.ifft(np.fft.fft(state)*kinetic)
        state *= half
        if iteration % settings.save_every == 0:
            yield iteration, state.copy()


def dirac_evolution(settings: Settings, height_er: float, speed: float, half_gap: float):
    """H = speed*p*sigma_x + half_gap*sigma_z + barrier*I, complex128.

    This is a coarse-grained benchmark; it is not the production lattice solver.
    The positive-energy incoming packet has the same Gaussian q envelope.
    """
    x = settings.grid()
    k = 2*np.pi*np.fft.fftfreq(settings.points, settings.dx)
    energy = np.sqrt((speed*k)**2+half_gap*half_gap)
    theta = np.arctan2(speed*k, half_gap)
    amplitudes = np.exp(-settings.packet_sigma**2*(k-settings.initial_q)**2
                        -1j*k*settings.packet_center-0.5j*k*settings.length)
    amplitudes[(k <= 0) | (k >= 0.3) | (np.abs(k-settings.initial_q)*settings.packet_sigma >= 4)] = 0
    state = np.fft.ifft(np.stack((amplitudes*np.cos(theta/2), amplitudes*np.sin(theta/2))), axis=1)
    state /= np.sqrt(settings.dx*np.sum(np.abs(state)**2))
    absorber = settings.absorber_strength_er/2*(np.maximum(0, np.abs(x)-(settings.length/2-settings.absorber_width))/settings.absorber_width)**4
    half = np.exp(-0.5j*settings.time_step*(barrier_profile(x, height_er, settings)/2-1j*absorber))
    cosine = np.cos(settings.time_step*energy)
    sine_over_energy = settings.time_step*np.sinc(settings.time_step*energy/np.pi)
    for iteration in range(settings.steps):
        state *= half
        a, b = np.fft.fft(state, axis=1)
        state = np.fft.ifft(np.stack((cosine*a-1j*sine_over_energy*(half_gap*a+speed*k*b),
                                      cosine*b-1j*sine_over_energy*(speed*k*a-half_gap*b))), axis=1)
        state *= half
        if iteration % settings.save_every == 0:
            # Density of the spinor is |psi1|^2+|psi2|^2, not |psi1+psi2|^2.
            yield iteration, np.sqrt(np.sum(np.abs(state)**2, axis=0)).astype(complex)
