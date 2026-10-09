"""Wigner-Dyson versus Poisson level statistics for each phase-diagram point.

Implements "A Scalar Factor for Comparing WD and Poissonian Statistics"
(L. Wang, August 2026) for the Floquet quasienergies of the driven lattice:

    r_n    = min(delta_{n-1}, delta_n) / max(delta_{n-1}, delta_n),  0 <= r_n <= 1   (eq. 1)
    eta    = | (<r>   - <r>_P)   / (<r>_GOE   - <r>_P)   |                        (eq. 12)
    eta_r2 = | (<r^2> - <r^2>_P) / (<r^2>_GOE - <r^2>_P) |                        (eq. 13)

eta ~ 0 is Poisson-like (regular), eta ~ 1 is GOE-like (chaotic).

Levels
------
The levels are the quasienergies of the one-period Floquet operator U(T) of

    H(t) = p^2 / 2m + V0 [1 + alpha cos(omega t)] cos^2(kL x),

the Hamiltonian of lattice_kapitza.floquet_U, at fixed quasimomentum q in the
plane-wave basis exp(i (2n + q) kL x), |n| <= n_max. Energies are in E_R and
times in hbar / E_R. The CUDA phase diagram uses
-V0/2 [1 + alpha cos(omega t)] cos(2 kL x + phase) plus spatially uniform
terms. A translation in x maps one potential onto the other, and uniform terms
only add a global phase to U(T), so both give the same quasienergy spacings.
The quasienergy spectrum therefore depends only on V0, alpha, omega, q and
n_max. Green walls break the lattice translation symmetry and are not part of
this bulk calculation.

Symmetry: at q = 0 the Hamiltonian is parity symmetric (n -> -n). Statistics
are taken separately in the even and odd sectors and the ratios pooled, since
merging independent sectors would look Poisson-like. A cosine drive is time-
reversal symmetric, so each sector is in the circular orthogonal ensemble
(COE), whose local statistics equal the GOE's.

Quasienergies live on a circle (they are defined modulo hbar omega), so
spacings are taken cyclically: the last level's upper spacing wraps around to
the first. Every level of a sector of size N then gives one ratio.

Run this file to check it:  python level_statistics.py --verify
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.linalg import eigh

# Physical constants of create_initial_state_function.py (7Li, 1064 nm lattice).
HBAR = 1.054571817E-34
MASS = 7.01600455 * 1.66053906660e-27
K_L = 2.0 * np.pi / 1064E-09
E_R = HBAR**2 * K_L**2 / (2.0 * MASS)

# Poisson references, exact: <r>_P = 2 ln 2 - 1 (eq. 4), <r^2>_P = 3 - 4 ln 2.
R_POISSON = 2.0 * math.log(2.0) - 1.0
R2_POISSON = 3.0 - 4.0 * math.log(2.0)
# Large-GOE references by the procedure of eqs. 6-10 (goe_reference()):
# 4000 matrices of size 1000 give <r> = 0.53064(15), matching the 0.5307 of
# eq. 11, and <r^2> = 0.34627(16); 400 of size 2500 give 0.53033(33) and
# 0.34603(34). The COE gives the same values for N = 21 to 400.
R_GOE = 0.5307
R2_GOE = 0.3462

COLUMNS = ("mean_r", "mean_r_squared", "eta", "eta_r_squared", "level_ratio_count")

# Commutator-free fourth-order Magnus integrator (Blanes & Moan 2006) with
# Gauss-Legendre nodes c1, c2 and weights a1, a2.
_C1, _C2 = 0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0
_A1, _A2 = 0.25 + math.sqrt(3.0) / 6.0, 0.25 - math.sqrt(3.0) / 6.0


def restricted_ratios(levels, period=None):
    """Restricted spacing ratios r_n of eq. 1.

    levels: real energies. With `period`, they are phases on a circle of that
    circumference (quasienergies), sorted modulo the period with cyclic
    spacings; otherwise they are sorted on a line (N levels give N - 2 ratios).
    Ratios with two zero spacings are undefined and dropped.
    """
    values = np.sort(np.asarray(levels, dtype=np.float64).ravel())
    if period is None:
        spacings = np.diff(values)
        previous, following = spacings[:-1], spacings[1:]
    else:
        values = np.sort(np.mod(values, period))
        spacings = np.diff(np.concatenate([values, values[:1] + period]))
        previous, following = np.roll(spacings, 1), spacings
    if previous.size == 0:
        return np.empty(0)
    largest = np.maximum(previous, following)
    keep = largest > 0
    return np.minimum(previous, following)[keep] / largest[keep]


def likeness(value, poisson, goe):
    """|(value - poisson) / (goe - poisson)|, eqs. 12 and 13."""
    return abs((value - poisson) / (goe - poisson))


def statistics_from_ratios(ratios):
    """<r>, <r^2>, eta and eta_r2 for pooled ratios (blank if there are none)."""
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.size == 0:
        return dict.fromkeys(COLUMNS[:-1]) | {"level_ratio_count": 0}
    mean_r = float(np.mean(ratios))
    mean_r2 = float(np.mean(ratios * ratios))
    return {
        "mean_r": mean_r,
        "mean_r_squared": mean_r2,
        "eta": likeness(mean_r, R_POISSON, R_GOE),
        "eta_r_squared": likeness(mean_r2, R2_POISSON, R2_GOE),
        "level_ratio_count": int(ratios.size),
    }


def lattice_matrices(depth_er, alpha, quasimomentum, n_max):
    """Static part A and driven part B of H(t) = A + cos(omega t) B, in E_R.

    cos^2(kL x) = 1/2 + (e^{2i kL x} + e^{-2i kL x}) / 4 in the plane-wave
    basis exp(i (2n + q) kL x), n = -n_max ... n_max.
    """
    n = np.arange(-n_max, n_max + 1)
    cos2 = 0.5 * np.eye(n.size) + 0.25 * (np.eye(n.size, k=1) + np.eye(n.size, k=-1))
    kinetic = np.diag((2.0 * n + quasimomentum) ** 2)
    return kinetic + depth_er * cos2, depth_er * alpha * cos2


def symmetry_sectors(n_max, quasimomentum):
    """Orthonormal bases of the parity sectors (q = 0) or the whole space."""
    size = 2 * n_max + 1
    if quasimomentum != 0.0:
        return [np.eye(size)]
    centre = n_max
    even = np.zeros((size, n_max + 1))
    odd = np.zeros((size, n_max))
    even[centre, 0] = 1.0
    for n in range(1, n_max + 1):
        even[centre + n, n] = even[centre - n, n] = math.sqrt(0.5)
        odd[centre + n, n - 1] = math.sqrt(0.5)
        odd[centre - n, n - 1] = -math.sqrt(0.5)
    return [even, odd]


def floquet_operator(static, drive, omega, steps):
    """U(T) for H(t) = static + cos(omega t) drive over one period T = 2 pi / omega."""
    period = 2.0 * math.pi / omega
    h = period / steps
    unitary = np.eye(static.shape[0], dtype=complex)
    for step in range(steps):
        f1 = math.cos(omega * (step + _C1) * h)
        f2 = math.cos(omega * (step + _C2) * h)
        # The factor weighting the earlier node acts first.
        for weight in (_A1 * f1 + _A2 * f2, _A2 * f1 + _A1 * f2):
            values, vectors = eigh(0.5 * static + weight * drive)
            unitary = (vectors * np.exp(-1j * h * values)) @ (vectors.T @ unitary)
    return unitary


def _phase_mismatch(first, second):
    """Largest distance on the unit circle from an eigenvalue of `first` to `second`."""
    return float(np.max(np.min(np.abs(np.angle(first[:, None] * np.conj(second[None, :]))), axis=1)))


def quasienergy_phases(static, drive, omega, *, tolerance=1e-4, initial_steps=32,
                       maximum_steps=2**15):
    """Eigenphases of U(T), with time steps doubled until they converge.

    Converged means no eigenphase moves by more than `tolerance` times the mean
    phase spacing 2 pi / N when the step count doubles. Returns the phases and
    the step count used.
    """
    steps = initial_steps
    previous = np.linalg.eigvals(floquet_operator(static, drive, omega, steps))
    limit = tolerance * 2.0 * math.pi / static.shape[0]
    while steps < maximum_steps:
        steps *= 2
        current = np.linalg.eigvals(floquet_operator(static, drive, omega, steps))
        if _phase_mismatch(current, previous) <= limit:
            return np.angle(current), steps
        previous = current
    raise RuntimeError(f"Floquet quasienergies did not converge within {maximum_steps} steps.")


def point_level_statistics(depth_er, alpha, frequency_hz, *, quasimomentum=0.0,
                           plane_wave_cutoff=40, tolerance=1e-4):
    """CSV columns for one phase-diagram point.

    depth_er: lattice depth V0 in E_R; alpha: modulation amplitude; frequency_hz:
    drive frequency nu (omega = 2 pi nu); quasimomentum: q in units of kL, with
    -1 < q < 1; plane_wave_cutoff: n_max.
    """
    if not -1.0 < quasimomentum < 1.0:
        raise ValueError("The quasimomentum must lie strictly between -1 and 1 (units of kL).")
    if plane_wave_cutoff < 2:
        raise ValueError("The plane-wave cutoff must be at least 2.")
    omega = 2.0 * math.pi * frequency_hz * HBAR / E_R
    static, drive = lattice_matrices(depth_er, alpha, quasimomentum, plane_wave_cutoff)
    ratios = []
    largest_steps = 0
    for basis in symmetry_sectors(plane_wave_cutoff, quasimomentum):
        phases, steps = quasienergy_phases(basis.T @ static @ basis, basis.T @ drive @ basis,
                                           omega, tolerance=tolerance)
        largest_steps = max(largest_steps, steps)
        ratios.append(restricted_ratios(phases, period=2.0 * math.pi))
    result = statistics_from_ratios(np.concatenate(ratios))
    result["floquet_time_steps"] = largest_steps
    return result


def level_statistics_task(arguments):
    """Picklable entry point for worker processes: (depth, alpha, frequency, settings)."""
    depth_er, alpha, frequency_hz, settings = arguments
    return point_level_statistics(
        depth_er, alpha, frequency_hz, quasimomentum=settings["quasimomentum"],
        plane_wave_cutoff=settings["plane_wave_cutoff"],
    )


def validate_settings(quasimomentum, plane_wave_cutoff):
    """Settings stored in the manifest; raises ValueError if invalid."""
    if isinstance(quasimomentum, bool) or not isinstance(quasimomentum, (int, float)) \
            or not -1.0 < quasimomentum < 1.0:
        raise ValueError("LevelStatisticsQuasimomentum must be a number with -1 < q < 1.")
    if isinstance(plane_wave_cutoff, bool) or not isinstance(plane_wave_cutoff, int) \
            or plane_wave_cutoff < 2:
        raise ValueError("LevelStatisticsPlaneWaveCutoff must be an integer of at least 2.")
    return {
        "levels": "Floquet quasienergies of the driven bulk lattice (no green walls)",
        "quasimomentum": float(quasimomentum),
        "plane_wave_cutoff": plane_wave_cutoff,
        "parity_sectors_separated": quasimomentum == 0.0,
        "spacings": "cyclic, modulo hbar omega",
        "r_poisson": R_POISSON, "r_squared_poisson": R2_POISSON,
        "r_goe": R_GOE, "r_squared_goe": R2_GOE,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def goe_reference(size=1000, matrices=200, seed=20261009):
    """Recompute <r>_GOE and <r^2>_GOE by eqs. 6-10 with real symmetric GOE matrices.

    Returns (mean <r>, its standard error, mean <r^2>, its standard error).
    """
    rng = np.random.default_rng(seed)
    first, second = [], []
    for _ in range(matrices):
        g = rng.standard_normal((size, size))
        ratios = restricted_ratios(np.linalg.eigvalsh(0.5 * (g + g.T)))
        first.append(ratios.mean())
        second.append((ratios * ratios).mean())
    first, second = np.array(first), np.array(second)
    root = math.sqrt(matrices)
    return first.mean(), first.std(ddof=1) / root, second.mean(), second.std(ddof=1) / root


def sambe_quasienergies(depth_er, alpha, omega, quasimomentum, n_max, harmonics):
    """Quasienergies from Eric's extended-space (Sambe) Floquet Hamiltonian.

    An independent check of floquet_operator; valid for weak drives, where a
    modest number of Fourier harmonics converges.
    """
    static, drive = lattice_matrices(depth_er, alpha, quasimomentum, n_max)
    m = np.arange(-harmonics, harmonics + 1)
    shift = np.eye(m.size, k=1) + np.eye(m.size, k=-1)
    # cos(omega t) couples Fourier harmonics m and m +- 1 with weight 1/2.
    hamiltonian = (np.kron(static, np.eye(m.size)) - np.kron(np.eye(static.shape[0]), np.diag(m * omega))
                   + 0.5 * np.kron(drive, shift))
    return np.linalg.eigvalsh(hamiltonian)


def verify(seed=7):
    """Print checks of every ingredient; returns True when all pass."""
    rng = np.random.default_rng(seed)
    checks = []

    def check(name, passed, detail):
        checks.append(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    check("r_n definition", np.allclose(restricted_ratios([0.0, 1.0, 3.0, 3.5]), [0.5, 0.25]),
          "levels 0, 1, 3, 3.5 give spacings 1, 2, 0.5 and ratios 1/2, 1/4")
    check("cyclic spacings", np.allclose(np.sort(restricted_ratios([0.0, 1.0, 3.0], period=6.0)),
                                         [1 / 3, 0.5, 2 / 3]),
          "phases 0, 1, 3 on a circle of 6 give spacings 1, 2, 3")
    poisson = statistics_from_ratios(np.concatenate(
        [restricted_ratios(rng.uniform(0, 2 * np.pi, 41), period=2 * np.pi) for _ in range(5000)]))
    check("Poisson levels", poisson["eta"] < 0.02 and abs(poisson["mean_r_squared"] - R2_POISSON) < 0.003,
          f"<r> = {poisson['mean_r']:.4f} (exact {R_POISSON:.4f}), eta = {poisson['eta']:.3f}")
    coe = []
    for _ in range(2000):
        z = (rng.standard_normal((41, 41)) + 1j * rng.standard_normal((41, 41))) / math.sqrt(2)
        q, r = np.linalg.qr(z)
        haar = q * (np.diag(r) / np.abs(np.diag(r)))
        coe.append(restricted_ratios(np.angle(np.linalg.eigvals(haar.T @ haar)), period=2 * np.pi))
    coe = statistics_from_ratios(np.concatenate(coe))
    check("COE levels", abs(coe["eta"] - 1) < 0.03 and abs(coe["eta_r_squared"] - 1) < 0.03,
          f"<r> = {coe['mean_r']:.4f}, <r^2> = {coe['mean_r_squared']:.4f}, eta = {coe['eta']:.3f}, "
          f"eta_r2 = {coe['eta_r_squared']:.3f}")

    # Undriven lattice: U(T) = exp(-i A T) exactly, so quasienergies are bands mod omega.
    static, drive = lattice_matrices(20.0, 0.0, 0.0, 12)
    omega = 30.0
    bands = np.linalg.eigvalsh(static)
    phases, _ = quasienergy_phases(static, drive, omega)
    error = _phase_mismatch(np.exp(1j * phases), np.exp(-1j * bands * 2 * np.pi / omega))
    check("undriven lattice", error < 1e-9, f"quasienergies = band energies mod omega, error {error:.1e} rad")

    # Weak drive: compare with the Sambe-space quasienergies.
    depth, alpha, q, omega, n_max = 30.0, 0.3, 0.2, 20.0, 8
    static, drive = lattice_matrices(depth, alpha, q, n_max)
    phases, steps = quasienergy_phases(static, drive, omega, tolerance=1e-6)
    sambe = sambe_quasienergies(depth, alpha, omega, q, n_max, harmonics=24)
    eps = -phases / (2 * np.pi / omega)          # quasienergies in E_R
    distance = np.min(np.abs(np.angle(np.exp(2j * np.pi * (eps[:, None] - sambe[None, :]) / omega))),
                      axis=1) * omega / (2 * np.pi)
    check("Sambe cross-check", np.max(distance) < 1e-6,
          f"V0 = 30, alpha = 0.3, q = 0.2: max |U(T) - Sambe| = {np.max(distance):.1e} E_R ({steps} steps)")

    # Fourth-order convergence of the integrator.
    static, drive = lattice_matrices(20.0, 5.0, 0.0, 10)
    exact = np.linalg.eigvals(floquet_operator(static, drive, 40.0, 4096))
    errors = [_phase_mismatch(np.linalg.eigvals(floquet_operator(static, drive, 40.0, n)), exact)
              for n in (64, 128)]
    order = math.log2(errors[0] / errors[1])
    check("integrator order", 3.5 < order < 4.6, f"error ratio for doubled steps 2^{order:.2f}")

    # Parity: the full q = 0 spectrum is the union of both sectors.
    static, drive = lattice_matrices(20.0, 3.0, 0.0, 10)
    full, _ = quasienergy_phases(static, drive, 25.0, tolerance=1e-7)
    sectors = np.concatenate([quasienergy_phases(b.T @ static @ b, b.T @ drive @ b, 25.0, tolerance=1e-7)[0]
                              for b in symmetry_sectors(10, 0.0)])
    error = max(_phase_mismatch(np.exp(1j * full), np.exp(1j * sectors)),
                _phase_mismatch(np.exp(1j * sectors), np.exp(1j * full)))
    check("parity sectors", error < 1e-6 and sectors.size == full.size,
          f"{sectors.size} sector levels reproduce the full spectrum to {error:.1e} rad")

    print("All checks passed." if all(checks) else "Some checks FAILED.")
    return all(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="run the built-in checks")
    parser.add_argument("--goe", nargs=2, type=int, metavar=("SIZE", "MATRICES"),
                        help="recompute the GOE references, e.g. --goe 1000 200")
    parser.add_argument("--point", nargs=3, type=float, metavar=("V0_ER", "ALPHA", "FREQUENCY_HZ"),
                        help="level statistics for one phase-diagram point")
    parser.add_argument("--quasimomentum", type=float, default=0.0)
    parser.add_argument("--cutoff", type=int, default=40, help="plane-wave cutoff n_max")
    arguments = parser.parse_args()
    if not (arguments.verify or arguments.goe or arguments.point):
        parser.print_help()
        return 0
    passed = True
    if arguments.verify:
        passed = verify()
    if arguments.goe:
        r, r_error, r2, r2_error = goe_reference(*arguments.goe)
        print(f"GOE N={arguments.goe[0]}, M={arguments.goe[1]}: <r> = {r:.5f} +- {r_error:.5f}, "
              f"<r^2> = {r2:.5f} +- {r2_error:.5f}  (stored: {R_GOE}, {R2_GOE})")
    if arguments.point:
        result = point_level_statistics(*arguments.point, quasimomentum=arguments.quasimomentum,
                                        plane_wave_cutoff=arguments.cutoff)
        for key, value in result.items():
            print(f"{key} = {value}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
