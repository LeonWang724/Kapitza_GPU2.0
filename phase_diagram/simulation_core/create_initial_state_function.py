# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:19:23 2026

@author: fermi-laborbenutzer
"""

import numpy as np
import matplotlib.pyplot as plt
import tables as tb
import os
from config_value import update_config_value

debug_flag = False
def bloch_state_1d_plane_wave(
    V0,
    q,
    band_index,
    xarr,
    n_pw=15
):
    """
    Compute a Bloch state psi_{n,q}(x) for a 1D optical lattice
    V(x) = V0 cos^2(k_L x)
    using a plane-wave expansion.

    Units:
        - energies in recoil energies E_R
        - wavevectors in units of k_L
        - positions in units of 1/k_L

    Parameters
    ----------
    V0 : float
        Lattice depth in units of E_R.
    q : float
        Quasimomentum in units of k_L.
        First Brillouin zone is q in [-1, 1).
    band_index : int
        Bloch band index, starting from 0.
    n_pw : int
        Number of plane waves on each side.
        Total basis size = 2*n_pw + 1.

    Returns
    -------
    x : ndarray
        Real-space grid in units of 1/k_L.
    psi_x : ndarray
        Bloch wavefunction psi_{n,q}(x) on the grid.
    energies : ndarray
        Band energies at this q in units of E_R.
    G_indices : ndarray
        Reciprocal-lattice indices m corresponding to G = 2m.
    coeffs : ndarray
        Plane-wave coefficients of the chosen Bloch state.
    """
    # Reciprocal lattice vectors in units of k_L:
    # G_m = 2m because cos^2(k_L x) has period pi/k_L
    G_indices = np.arange(-n_pw, n_pw + 1)
    G = 2.0 * G_indices

    dim = len(G)

    # Build Hamiltonian in plane-wave basis |q + G>
    # In recoil units:
    # kinetic term = (q + G)^2
    # potential: V0 cos^2(x) = V0/2 + (V0/4)(e^{i2x} + e^{-i2x})
    H = np.zeros((dim, dim), dtype=np.float64)

    for i, Gi in enumerate(G):
        H[i, i] = (q + Gi)**2 + V0 / 2.0

    # Off-diagonal couplings from cos(2x): connects G <-> G ± 2
    for i in range(dim):
        for j in range(dim):
            if abs(G[i] - G[j]) == 2.0:
                H[i, j] = V0 / 4.0

    # Diagonalize
    energies, eigvecs = np.linalg.eigh(H)

    # Select requested band
    coeffs = eigvecs[:, band_index]

    # Real-space grid
    lattice_period = np.pi  # in units of 1/k_L
    #x = np.linspace(-n_cells * lattice_period / 2,
    #                n_cells * lattice_period / 2,
    #                n_x)
    
    #x = np.arange(-n_x//2, n_x//2) * rs

    # Construct Bloch state:
    # psi_{n,q}(x) = sum_G c_G exp[i (q+G) x]
    psi_x = np.zeros_like(xarr, dtype=np.complex128)
    for c, Gm in zip(coeffs, G):
        psi_x += c * np.exp(1j * (q + Gm) * xarr)

    # Normalize numerically over the plotted interval
    dx = xarr[1] - xarr[0]
    norm = np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
    psi_x /= norm

    return psi_x, energies, G_indices, coeffs


def plot_bloch_state(x, psi_x, q, band_index, V0):
    """
    Plot real part, imaginary part, and density of the Bloch state.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(x, np.real(psi_x))
    axes[0].set_ylabel(r"Re[$\psi_{n,q}(x)$]")
    axes[0].set_title(
        rf"Bloch state in 1D lattice: $V_0={V0}\,E_R$, band $n={band_index}$, $q={q}\,k_L$"
    )

    axes[1].plot(x, np.imag(psi_x))
    axes[1].set_ylabel(r"Im[$\psi_{n,q}(x)$]")

    axes[2].plot(x, np.abs(psi_x)**2)
    axes[2].set_ylabel(r"$|\psi_{n,q}(x)|^2$")
    #axes[2].set_xlabel(r"$x\,[1/k_L]$")
    axes[2].set_xlabel(r"$x$")

    plt.tight_layout()
    plt.show()

hbar = 1.054571817E-34
amu  = 1.66053906660e-27
kB   = 1.380649e-23
mli = 7.01600455 * amu

    
    
    

def create_init_state(
    V0, alpha, nu_flo, phi, initial_lattice_depth_v0_er=None
):
    # Parameters
    #V0 = 10.0          # lattice depth in E_R
    q = 0.0           # quasimomentum in units of k_L, should lie in [-1, 1)
    band_index = 0    # 0 = lowest band
    n_pw = 40         # plane waves on each side
    n_x = 65536
    #n_cells = 3079
    
    lam = 1064E-09
    kL = 2.0*np.pi / lam
    
    #Parameters gaussian envelope
    sigma = 30E-06
    p0 = 0.0 * kL
    t0 = 0    #Sets momentum spread
    
    #Parameters potential
    #alpha = 27
    #alpha = 1.0
    #phi = np.pi
    
    #nu_flo = 2.0E06
    
    #Scaling parameter
    rs = 23.0E-09
    x = np.arange(-n_x//2, n_x//2) * rs
    #n_cells = int(rs * n_x / 532E-09)
    #print(n_cells)
    
    Er = hbar**2 * kL**2 / (2*mli)

    # Historical runs prepared the initial Bloch state at twice V0. Exposing
    # that depth makes it independently configurable and records its true value
    # in dataset names/manifests without changing the historical default.
    if initial_lattice_depth_v0_er is None:
        initial_lattice_depth_v0_er = 2.0 * V0

    psi_x, energies, G_indices, coeffs = bloch_state_1d_plane_wave(
        V0=initial_lattice_depth_v0_er,
        xarr=x*kL,
        q=q,
        band_index=band_index,
        n_pw=n_pw
    )
    
    #x = x * 1.0/kL
    
    #print(f"x_low: {x[0]}")
    
    
    #imaginary potential
    dx = x[1]-x[0]
    
    #print(dx)
    # absorber settings
    Labs = 0.15*(x[-1]-x[0])   # 15% on each side
    mexp = 4
    W0   = 5*Er  # tune (units of energy)
    
    W = np.zeros_like(x)
    left  = x < (x[0] + Labs)
    right = x > (x[-1] - Labs)
    
    W[left]  = W0 * ((x[0] + Labs - x[left]) / Labs)**mexp
    W[right] = W0 * ((x[right] - (x[-1] - Labs)) / Labs)**mexp
    
    
    
    #gauss =  (2.0 * np.pi * sigma**2)**-0.25 * (1 + 1.0j * hbar * t0 / (2 * mli * sigma) )**0.25  * np.exp(1.0j / hbar * p0) * np.exp(-x**2 / (4 * sigma**2))
    gauss =  (2.0 * np.pi * sigma**2)**-0.25 * (1 + 1.0j * hbar * t0 / (2 * mli * sigma**2) )**-0.5  * np.exp(1.0j / hbar * p0) * np.exp(-x**2 / (4 * sigma**2))



    psi_0 = gauss * psi_x
    if debug_flag:
        plot_bloch_state(x, psi_0, q, band_index, initial_lattice_depth_v0_er)
    
    psi_0_norm = np.sum(np.abs(psi_0)**2)*dx
    #print(psi_0_norm)
    psi_0 = psi_0 / np.sqrt(psi_0_norm)
    #psi_0_norm = np.sum(np.abs(psi_0)**2)*dx
    #print(psi_0_norm)
    
    V0 = V0 * Er
    vstatic = -0.5 * (1.0 + alpha) * V0 - 0.5 * V0 * np.cos(2.0 * kL * x + phi)
    vflo = -0.5 * V0 * alpha * np.cos(2.0 * kL * x + phi)
    
    #vstatic =  -1.0 * V0 * np.cos(2.0 * kL * x + phi)
    #vflo = -1.0 * V0 * alpha * np.cos(2.0 * kL * x + phi)
    
    
    #Add absorbative walls.
    vstatic = vstatic - 1.0j * W
    
    
    
    
    

    Es = hbar**2 / (mli * rs**2)
    ts = hbar / Es
    
    
    
    vstatic = vstatic / Es
    vflo = vflo / Es
    
    omega0 = np.sqrt(2 * V0 * kL**2 / mli)
    
    
    print(f"OMEGA: {nu_flo*2*np.pi / omega0}")
    
    
    if debug_flag:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        #Plot potential
        ax[0].plot(x, np.real(vstatic))
        ax[0].plot(x, vflo)
        
        ax2 = ax[0].twinx()
        ax2.plot(x, np.imag(vstatic), c='red')
        
        ax[1].plot(x, np.abs(psi_0)**2, c='red')
        plt.tight_layout()
        plt.show()
        

    psi_0 = rs**0.5 * psi_0
    f = tb.open_file('in\\lattice_gauss.h5', 'w')
    f.create_array('/', 'REAL', np.real(psi_0))
    f.create_array('/', 'IMAGINARY', np.imag(psi_0))
    f.close()
    
    f = tb.open_file('in\\vstatic.h5', 'w')
    f.create_array('/', 'REAL', np.real(vstatic))
    f.create_array('/', 'IMAGINARY', np.imag(vstatic))
    f.close()
    
    f = tb.open_file('in\\vflo.h5', 'w')
    f.create_array('/', 'REAL', np.real(vflo))
    f.create_array('/', 'IMAGINARY', np.imag(vflo))
    f.close()
    
    if debug_flag:
        print("*********************************************************************")
        print(f"points_x: {n_x}")
        #print(f"step_x: {(n_cells * 1.0/kL / n_x)/rs}")
        #print(f"step_x: {(n_cells * 532E-09 / n_x)/rs}")
        print(f"Es: {Es}")
        print(f"ts: {ts}")
        print(f"floquet_omega: {2.0*np.pi*nu_flo*ts}")
    
    update_config_value("gpe1d.config", "floquet_omega", 2.0*np.pi*nu_flo*ts)
    
    
if __name__ == "__main__":
    
    create_init_state(10, 30, 3.12E06, 0.0)
