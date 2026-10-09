# Kapitza_GPU4.0

Version 4.0 adds optional **green walls** to the phase diagram: two repulsive
Gaussian potential walls around the initial wavefunction. Set them in the
physical parameter section of
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py`:

```python
GreenWalls = True            # False: no walls, inputs identical to before
GreenWallHeight_ER = 100.0   # peak height of each wall, in E_R like the lattice depth
GreenWallSigma_um = 5.0      # Gaussian sigma of each wall, exp(-x^2/(2 sigma^2)), in um
GreenWallGap_um = 200.0      # centre-to-centre distance between the two walls, in um
```

The walls are centred at `-gap/2` and `+gap/2`, around the initial cloud at
`x = 0`, and join the static potential only; the drive modulates the lattice,
not the walls. Each wall must end (`gap/2 + 3 sigma`) before the absorbing
layer, which starts about 527.5 um from the centre. The runner checks this before
any GPU work and prints how much of the initial cloud (sigma 30 um) starts
between the walls. Dataset folders, manifests and plot titles record the
walls; `RUN_PHASE_DIAGRAM_CUDA.bat --plan` shows them without running.
With `GreenWalls = False` every generated input is bitwise identical to 3.0.
The example values above are placeholders; set them to your experiment's.

### Level statistics: `<r>`, `<r^2>` and eta

`CalculateLevelStatistics = True` in the same section adds five columns to
`metrics.csv` (and to the MAKE analysis CSV and NPZ), following
*A Scalar Factor for Comparing WD and Poissonian Statistics*:

| Column | Meaning |
| --- | --- |
| `mean_r` | `<r>`, mean restricted spacing ratio `r_n = min(d_{n-1}, d_n) / max(d_{n-1}, d_n)` |
| `mean_r_squared` | `<r^2>` |
| `eta` | `abs((<r> - <r>_P) / (<r>_GOE - <r>_P))`: 0 Poisson-like, 1 GOE-like |
| `eta_r_squared` | the same with `<r^2>` |
| `level_ratio_count` | number of ratios averaged |

The levels are the Floquet quasienergies of the driven lattice
`V0 [1 + alpha cos(omega t)] cos^2(kL x)` at quasimomentum
`LevelStatisticsQuasimomentum` (default 0, the BEC's), from the one-period
evolution operator in a plane-wave basis `|n| <= LevelStatisticsPlaneWaveCutoff`
(default 40). At q = 0 the even and odd parity sectors are treated separately
and their ratios pooled; spacings wrap around the quasienergy circle. The
references are `<r>_P = 2 ln 2 - 1`, `<r^2>_P = 3 - 4 ln 2`, `<r>_GOE = 0.5307`
and `<r^2>_GOE = 0.3462`; the last was computed with 4000 GOE matrices of size
1000. This is a bulk-lattice calculation: it ignores the green walls and does
not use the GPU result, and the CPU computes it while the GPU runs (about 0.03
to 0.2 s per point). The result depends on the cutoff, because high-momentum
plane waves are regular; compare cutoffs before interpreting small differences.

Everything is in `phase_diagram/simulation_core/level_statistics.py`. To check
it, double-click `VERIFY_LEVEL_STATISTICS.bat`. The checks cover the ratio
definition, Poisson and random-matrix levels, the undriven lattice, a
comparison with the lab's Sambe-space Floquet Hamiltonian and the integrator's
order. Single points and the GOE references can be recomputed from a terminal:

```powershell
.\.venv\Scripts\python.exe phase_diagram\simulation_core\level_statistics.py --point 20 60 3.5e6
.\.venv\Scripts\python.exe phase_diagram\simulation_core\level_statistics.py --goe 1000 200
```

### Choosing what MAKE plots

`MAKE_PHASE_DIAGRAM_CUDA.bat` lists the columns it found and asks what to plot;
press Enter to keep each default (IPR against alpha and frequency, automatic
colour limits):

```text
Columns available for plotting
  inputs:  alpha, drive_frequency_hz, lattice_depth_v0_er, ..., green_wall_height_er, ...
  results: metric, metric_std, ..., sigma_x, ..., sigma_x_um, mean_r, eta, ...
    metric = IPR
    sigma_x_um = standard deviation of |psi|^2, the cloud width
    metric_std = time fluctuation of the IPR
x axis [alpha]:
y axis [drive_frequency_hz]:
colour [metric]: sigma_x_um
colour limits as 'min max', or 'auto' [auto]: 0 0.8
```

The standard deviation of `|psi|^2` is the cloud width `sigma_x_um`: the
spread of x weighted by `|psi|^2` in the cropped window, normalized to the
atoms still there, as the root mean square over the final 30 snapshots, in
micrometres (`sigma_x` in 23 nm grid units). `metric_std` is something else:
how much the IPR fluctuates between those snapshots. Only columns present in
that dataset are offered, so older results without these columns do not list
them. Each row also carries the
scan's lattice depth and green-wall height (0 without walls), sigma and gap.
To plot along a parameter that is fixed within one scan, such as the wall
height, pass several datasets or CSV files; when rows share a plotted point,
MAKE asks which other input to fix (for example the frequency):

```powershell
.\MAKE_PHASE_DIAGRAM_CUDA.bat <walls_off_dataset> <walls_50ER_dataset> <walls_100ER_dataset>
```

Combined plots are saved under `results_cuda\combined_plots` with the rows
used and their sources. The same choices work without questions, e.g.
`--x green_wall_height_er --y alpha --color sigma_x_um --vmin 0 --vmax 0.8
--where drive_frequency_hz=3e6 --no-ask`, or as defaults near the top of
`make_phase_diagram_CUDA.py` (`X_VARIABLE`, `Y_VARIABLE`, `COLOR_VARIABLE`,
`COLOR_LIMITS`, `ASK_FOR_VARIABLES`).

Version 3.0 added cold-atom **Klein tunneling in an engineered bichromatic optical
lattice**. It prepares a moving packet in the upper excited Bloch band, evolves
it through a smooth barrier with the existing complex128 CUDA solver, and
exports transmission, reflection, loss, spatial variance and scattering plots.
See [README_KLEIN.md](README_KLEIN.md) for the model, settings and validation.

From the `Kapitza_GPU4.0` folder on Windows:

```powershell
.\SETUP_WINDOWS.bat
.\BUILD_CUDA.bat CUDA_ONLY
.\VALIDATE_KLEIN_CUDA.bat
.\RUN_KLEIN_CUDA.bat
```

The existing Kapitza phase-diagram workflow, tab splitting, statistics and
fused/batched CUDA solver remain available below. This folder was copied from
3.0 (itself copied from 2.0). Its Git remote still names the 2.0 repository,
where 4.0 lives on the `feature/green-walls` branch.

Native CUDA simulation of a one-dimensional driven optical lattice, with compact
phase-diagram output. This version builds on
[Kapitza_Phase_Diagram_GPU](https://github.com/LeonWang724/Kapitza_Phase_Diagram_GPU)
and keeps the complex128 split-step solver and existing phase-diagram metric.

## Existing Kapitza workflow on Windows

```powershell
.\SETUP_WINDOWS.bat
.\BUILD_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
.\VALIDATE_FAST_CUDA.bat
```

Run each command after the previous command finishes successfully. Setup installs
or repairs the Windows prerequisites and may request Administrator access.
Then edit the physical parameter section in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py` and run:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat
.\MAKE_PHASE_DIAGRAM_CUDA.bat
```

## Faster solver: fused steps and GPU batches

The solver evolves the same equation with the same complex128 split-step
method, potentials, time step, grid, FFT normalization and observables. Only
the scheduling of the GPU work changed:

- **Fused steps.** The two half steps that meet between consecutive iterations
  run in one GPU pass, with the Floquet potential computed inline. A step now
  launches 2 small kernels and 2 FFTs instead of 6 kernels and 2 FFTs.
- **No repeated exponentials.** With `beta=0` both half steps of an iteration
  use the same factor `exp(-i dt V / 2)`, so it is computed once. `exp(0)=1`
  is not evaluated where the potential is real (outside the absorbers).
- **Bit-identical by construction.** Each value is produced by the same
  operations in the same order as the original kernels. Results match the
  original loop bit for bit; `VALIDATE_FAST_CUDA.bat` checks this on your GPU.
- **GPU batches.** A 65,536-point system is too small to keep a GPU busy, and
  Windows runs separate processes' kernels one at a time, so extra tabs mostly
  fill gaps. `POINTS_PER_BATCH` (default 8) grid points now run side by side
  in one solver process, using a batched cuFFT plan. A batched FFT may round
  differently from a single one at the 1e-15 level; validation reports the
  actual difference.
- **Less waiting.** Diagnostics every 100 steps need one GPU-to-CPU transfer
  instead of about five, and the sweep tells CUDA to sleep instead of spin
  while waiting, which leaves a laptop's power budget to the GPU.

After pulling this version, rebuild, validate and measure:

```powershell
.\BUILD_CUDA.bat
.\VALIDATE_FAST_CUDA.bat
.\BENCHMARK_CUDA.bat
```

The benchmark times the original loop and the fused loop at batch sizes 1 to
32 on points from your configured grid, and prints the recommended
`POINTS_PER_BATCH` and the projected scan time. Set that value next to
`CUDA_DEVICE` in `run_phase_diagram_CUDA.py`, or pass it per run:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat --batch 16
```

With batching, `NUMBER_OF_TABS = 1` or `2` is usually best: two tabs hide
input generation and process startup, more tabs share the same GPU time.
`--reference-loop` runs the original one-point-at-a-time loop for comparison.
Full storage (`--storage full`) always runs one point per solver call. Each
point in a batch keeps its own inputs, configuration, status file and result
row; a batch is recorded only when all of its points succeed.

## Split a scan across terminal tabs

Set `NUMBER_OF_TABS` beside the physical parameters in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py`, then use the usual
`RUN_PHASE_DIAGRAM_CUDA.bat` command. The default is `1`, which runs in the
current terminal. For example:

```python
NUMBER_OF_TABS = 11
```

The launcher opens that many worker tabs in a dedicated Windows Terminal window.
If Windows Terminal (`wt.exe`) is unavailable, it opens separate console windows.
Each worker receives a distinct group of grid points, private input files and
configuration, and a separate results folder. The originating terminal shows
overall progress and must stay open until the workers finish. Completed results
are automatically combined into one dataset for the usual make command.

A 100 x 100 grid has 10,000 points: `20` tabs get 500 points each; `11` tabs get
910 points in the first tab and 909 in each of the other ten. For any remainder,
the first few tabs receive one extra point each. Workloads differ by at most one
point. A count larger than the grid opens only as many workers as there are
points. Zero, negative, and noninteger counts are rejected.

To preview assignments without starting CUDA, or override the saved count:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat --plan
.\RUN_PHASE_DIAGRAM_CUDA.bat --tabs 20
```

Compact and full storage both support this split. Ctrl+C in the originating
terminal stops its workers. A worker failure stops the other workers and keeps
their existing results and the failed point's diagnostics; the dataset is not
marked complete. Resume is still not implemented. `--headless` runs the same
workers without opening terminal tabs.

The tab launcher alone is a Python workflow update. The statistics below add
GPU diagnostics and require a CUDA rebuild. Local checks cover real worker
subprocesses with a stand-in solver, odd splits, result merging, failures, and
plotting in both storage modes. Actual Windows tab opening, GPU speed, and
thermal behavior require checking on the target laptop. The tab count controls
concurrency; it does not automatically tune performance or enforce temperatures.

## Standard deviations and spatial variance

The scalar and analysis CSV files append these columns to the existing ones:

| Column | Meaning |
| --- | --- |
| `metric_std` | Population standard deviation of the instantaneous phase metric over the selected final snapshot times |
| `metric_variance` | Square of `metric_std` |
| `x_mean` | Time mean of the cloud's normalized mean position |
| `x_squared_mean` | Time mean of its normalized second position moment |
| `sigma_x_squared` | Time mean of the per-snapshot spatial variance, σₓ² |
| `sigma_x` | Square root of `sigma_x_squared`, the RMS cloud width over that window |
| `x_mean_time_std` | Population standard deviation of the cloud's center over time |
| `spatial_snapshots_averaged` | Number of sampled states with nonzero probability in the cropped region |

For each selected snapshot, positions are `(index - points_x // 2) * step_x` and
position weights are `|psi|² / sum(|psi|²)` within the same crop as the original
phase metric. Thus spatial variance is `sum(weight * (x - mean_x)²)`. All spatial
lengths use solver length units, and squared columns use squared length units.
The current supplied generator uses a 23 nm length scale. The spatial RMS width
averages variance before taking its square root; moving centers are measured
separately by `x_mean_time_std`. The moment identity is
`x_squared_mean = x_mean² + x_mean_time_std² + sigma_x_squared`, up to roundoff.

All temporal statistics use the same final window as `metric` (normally the last
30 saved-time indices) and the population convention `ddof=0`. One temporal
sample gives zero temporal standard deviation. A completely absorbed state
does not have a normalized spatial distribution: spatial averages exclude it,
and a window with no surviving mass has blank spatial columns. Temporal spread
describes fluctuations over the window; it is not an uncertainty estimate for
the mean. The original unnormalized phase metric and plot remain unchanged.

Rebuild and validate on the Windows CUDA machine before using the new compact
statistics:

```powershell
.\BUILD_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
.\RUN_PHASE_DIAGRAM_CUDA.bat
.\MAKE_PHASE_DIAGRAM_CUDA.bat
```

Compact mode computes the extra moments on the GPU only at the selected sample
times, keeps scalar results, and saves no extra wavefunction snapshots. Every
worker's statistics survive the combined CSV export. Full mode computes the
same columns from retained snapshots during analysis. NPZ output also includes
one matrix per new statistic. Older compact datasets contain only a mean, so
their unavailable new statistics are left blank. Older full datasets can be
reanalyzed; spatial columns need their saved configuration's `step_x`.

Host C++ and Python tests cover known distributions, changing norm, nonunit grid
spacing, temporal variance, zero-mass states, and parallel export. Actual CUDA
kernel execution remains unverified on this Mac. The compact validation command
now compares every new statistic against NumPy analysis of identical full runs.

## Compact storage

The default saves one metric per grid point in `metrics.csv`, along with a compact
record of the parameters and calculation. It averages the cropped discrete
`sum |psi|^4` at the last 30 historical snapshot times directly on the GPU.
Wavefunction snapshots are not written. Temporary inputs are reused and only a
failed point's diagnostics are retained.

With the original 65,536-point and 70,000-iteration defaults, full output contains
about 700 MiB of raw wavefunction arrays per grid point. Compact output for a
100x100 grid is expected to retain tens of MB including metadata instead of
multiple TB. The exact size depends on the run and generated plots.

To keep wavefunctions for further analysis:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat --storage full
```

To save to an external drive, choose a new or empty folder:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat --results "E:\Kapitza\run001"
.\MAKE_PHASE_DIAGRAM_CUDA.bat "E:\Kapitza\run001"
```

MAKE also reads datasets from the previous snapshot-based version. Compact
results cannot reconstruct wavefunctions or produce different observables later.
Completed point records survive a later point failure, but automatic resume is
not implemented.

## Setup fixes and validation

- Launchers select the project's Python environment explicitly.
- CUDA runtime lookup includes both `bin` and `bin\x64`.
- Launchers honor `CUDA_PATH`, so Pascal GPUs can select CUDA 12.9 even with CUDA
  13.3 installed. See the [compute_61 build fix](README_CUDA.md#pascal-gpu-unsupported-gpu-architecture-compute_61).
- Build temporary files use a path without spaces.
- Builds refresh CMake configuration to avoid stale CUDA compiler paths.
- Short analysis filenames avoid duplicating long dataset names.

Thirty local Python and host C++ tests passed while preparing this release.
Windows setup, NVCC compilation, and GPU execution still require verification on
the target machine. `VALIDATE_COMPACT_CUDA.bat` compares six small full/compact
GPU cases against NumPy snapshot analysis and checks that compact mode produces
no HDF5 output snapshots. Run it before starting a large grid.

See [README_CUDA.md](README_CUDA.md) for configuration, dependencies, numerical
details, and the original CPU/CUDA validation workflow.
