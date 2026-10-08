# Optical-lattice Klein tunneling in Kapitza_GPU3.0

This workflow simulates a dilute, noninteracting cold-atom wavepacket in a
one-dimensional engineered optical lattice. The lattice has an excited-band
crossing with approximately Dirac-like dispersion. A smooth barrier can connect
the incoming upper band to the lower band inside the barrier, producing the
cold-atom analogue of Klein tunneling. The construction follows
[Salger et al., PRL 107, 240401 (2011)](https://arxiv.org/abs/1108.4447).

## Run on Windows

Open PowerShell in this project's folder. Run these commands one at a time:

```powershell
.\SETUP_WINDOWS.bat
.\BUILD_CUDA.bat CUDA_ONLY
.\VALIDATE_KLEIN_CUDA.bat
.\RUN_KLEIN_CUDA.bat
```

Setup installs the Python dependencies and build tools. Rebuild in the 3.0
folder even if you previously built 2.0. Validation exercises the actual CUDA
executable and writes a report under `validation/results/`. The run launcher
prints the results folder; it creates a separate directory for every run.

Edit `klein_tunneling/settings.json` to change the scan. Preview the Bloch bands,
grid and time schedule without launching evolution:

```powershell
.\RUN_KLEIN_CUDA.bat --plan
```

Examples with a selected device, GPU batch size or results directory:

```powershell
.\RUN_KLEIN_CUDA.bat --device 0 --batch-size 4
.\RUN_KLEIN_CUDA.bat --results "E:\Kapitza\klein001"
.\MAKE_KLEIN_DIAGRAM.bat "E:\Kapitza\klein001\run_manifest.json"
```

The default batches up to four compatible cases in one GPU process. Increase
`--batch-size` for larger scans, up to the executable's reported limit (64 in
the copied solver). The runner caps the request at that limit and handles the
last smaller batch. Performance depends on the GPU and workload; measure it
on the laptop. The existing `NUMBER_OF_TABS` setting applies to the separate
Kapitza phase-diagram workflow.

## Equation, units and band preparation

With `X = kL*x`, `kL = 2*pi/wavelength`, and recoil energy
`ER = hbar^2*kL^2/(2*m)`, the physical lattice and barrier are

```text
V(X)/ER = V1/2*cos(2X) + V2/2*cos(4X + theta) + B(X)/ER
B(X)/ER = height/2*[tanh((X+width/2)/edge) - tanh((X-width/2)/edge)]
```

The native solver uses energy unit **2 ER** and time unit **hbar/(2 ER)**:

```text
i dpsi/dtau = [-1/2 d²/dX² + V(X)/(2 ER) - i absorber(X)] psi
```

The generator therefore divides the potential in ER by two. `beta=0`,
imaginary-time relaxation and the Floquet drive are disabled. Spatial inputs
(`packet_center`, `packet_sigma`, `barrier_width`, `barrier_edge`, absorber
width) use X units; `time_step` and `duration` use native time units.
`packet_sigma` is the approximate density-envelope standard deviation.
Plots convert X and time into µm and ms using the configured atom mass and
wavelength. Defaults are lithium-7 and a 1064 nm lattice.

Bloch bands are obtained by diagonalizing the plane-wave matrix with momenta
`q + 2*n`. The relevant pair is the first and second excited bands, indices
1 and 2. The incoming packet superposes exact **band-2 eigenstates** with a
Gaussian quasimomentum envelope centered at positive `initial_q`. A smooth
eigenvector phase convention keeps the real-space packet localized. Preparing
the lowest band and giving it a momentum kick would not prepare this state.

Near q=0 the two-band envelope approximately obeys

```text
H_eff = crossing_energy*I + c_eff*p*sigma_x + Delta*sigma_z + barrier*I
```

The full energy gap is `2*Delta`; in native units `Delta = gap_ER/4`.
The Pauli matrices act on band pseudospin. At a closed gap the dispersion is
nearly linear. A barrier exceeding the packet's energy measured from the
crossing can access the lower branch inside the barrier while retaining
forward group velocity. A nonzero gap suppresses this transmission for the
starter barrier settings. The actual evolution retains the full scalar
lattice; the Dirac equation is an independent envelope benchmark.

This is a single-particle cold-atom analogue. It does not model relativistic
pair creation, a two-dimensional incidence angle, or interacting condensates.

## Starter scan and editable settings

| Setting | Default and meaning |
| --- | --- |
| `first_depth_er` | 5 ER |
| `second_depth_er` | `null`: numerically close the band-1/band-2 gap at phase pi; resolves to about 1.5625 ER |
| `phases_radians` | `[0, pi]`: gapped and nearly gapless comparison |
| `barrier_heights_er` | `[0, 0.9, 1.2]`: free control and two scattering barriers |
| `initial_q` | 0.12 kL; upper band, positive velocity |
| `packet_center`, `packet_sigma` | -180, 28 in X units |
| `barrier_width`, `barrier_edge` | 64, 8 in X units |
| `cells`, `points` | 512 lattice periods, 16384 FFT grid points |
| `plane_waves_each_side` | 10, giving 21 plane waves for band preparation |
| `time_step`, `duration` | 0.01, 360 native time units |
| `saved_intervals` | 40; starter run saves 41 snapshots |

`cells` must be an integer so the lattice and Bloch momenta fit the periodic
FFT domain. Increasing it also requires enough `points` to resolve the
plane-wave basis. Settings validation checks this, packet separation, and
absorber placement. Some choices can still yield unresolved scattering:
extend the duration and domain together, then repeat convergence checks.
Very high barriers, momenta far from the crossing, or sharp edges can leave
the two-band regime or couple other bands.

## Outputs and interpretation

Every case retains its initial packet, potential, native configuration and full
wavefunction snapshots. The starter sweep uses roughly 65 MB of raw snapshot
arrays across six cases, plus HDF5 overhead and inputs. Large scans scale with
case count, grid size and number of saved snapshots. These outputs allow
additional analysis of the wavepacket; the original compact Kapitza mode is
still available for Kapitza scans.

The run produces:

- `transmission.csv` and `transmission.npz`: scan axes and final scattering probabilities.
- `bands_and_transmission.png`: representative gapless/gapped bands and transmission versus barrier height.
- `wavepacket_scattering.png`: representative space-time density plots through a nonzero barrier.
- Each case's `probability_timeseries.csv` and `scattering_summary.json`.
- `run_manifest.json`: settings, resolved bands, physical units, build/device provenance, input hashes and case status.
- `analysis_report.json`: any incomplete measurements.

At the final saved time, T is the probability to the right of the barrier
measurement region, R to the left, and unresolved probability within it.
The dividing positions are `±(width/2 + 6*edge)`.
All probabilities use the **initial total norm**, so

```text
T + R + unresolved + absorbed = 1
```

Before scattering ends, left probability includes incoming atoms. The analysis
requires a conservative flight-time estimate, unresolved mass below 1%, loss
below 0.5%, and transmission drift below 1% across the last five snapshots
before flagging `measurement_complete=true`. The time estimate uses the
initial band group velocity and three packet widths beyond the measurement
region. These are configurable numerical diagnostics, not a guarantee that
arbitrary settings demonstrate Klein tunneling. Inspect the density plot,
bands, controls and convergence report when interpreting a scan.

`transmission_time_std` is the population standard deviation over those last
five times, not an uncertainty of the mean or shot-to-shot noise.
`x_mean` and `sigma_x_squared` describe the **final surviving distribution** in
native length units and squared length units. The full time series retains
these at every sampled time. Native iteration zero is saved **after the first
complete time step**, so snapshot n represents `(n+1)*time_step`. The starter's
last snapshot is at 360.01 native units (about 1.14 ms).

## Validation and local CPU reference

`VALIDATE_KLEIN_CUDA.bat` checks:

1. Short actual CUDA wavefunctions against an independent NumPy split-step calculation, L2 tolerance 1e-10.
2. Free propagation, probability balance, low loss and completed scattering.
3. Gapless transmission above 98% and gapped transmission below 2% for the starter 0.9 ER barrier.
4. A two-component effective Dirac benchmark, transmission difference below 2%.
5. Halved time step and doubled spatial resolution, transmission changes below 0.5%.

These tolerances apply to the starter configuration. They are not error bars
for every parameter choice. Band-basis convergence, exact free evolution,
absorption, band purity, failure handling and dataset integrity are additionally
covered by `tests/test_klein_tunneling.py`.

For development on a machine without CUDA, install
`phase_diagram/simulation_core/requirements_CUDA.txt` and run:

```bash
python -m klein_tunneling.run --backend numpy
python -m klein_tunneling.validate --backend numpy
python -m unittest discover -s tests
```

The NumPy backend is a CPU reference. Its report explicitly sets
`cuda_verified=false`. Initial testing on the Mac showed approximately
99.992% transmission through the 0.9 ER barrier at phase pi and less than
0.000002% at phase 0. All six starter cases passed the finite-time measurement
checks. GPU execution, NVCC compilation and Windows launchers must still be
validated on the Windows CUDA machine.
