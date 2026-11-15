# Kapitza_GPU10.0

Version 10.0 keeps everything of 9.0 and gives the atoms a spread of starting
quasimomenta instead of one q for every atom. With
`QUASIMOMENTUM_VALUES_KL = None` a run is identical to 9.0.

```python
QUASIMOMENTUM_VALUES_KL = [0.0, 0.05, 0.10, 0.15, 0.20]  # q runs per grid point (hbar kL)
QUASIMOMENTUM_CENTER_KL = 0.0                            # Gaussian of q they are combined with
QUASIMOMENTUM_SPREAD_KL = 0.1
```

- **How it works.** A real cloud has atoms at many q. Without interactions each
  group of atoms with the same q evolves on its own, and groups at different q
  do not interfere in the image, so the camera sees the sum of their clouds.
  Every grid point now runs once per q value (each run is the group of atoms at
  that q), and the runs are added up, weighted by the fraction of atoms a
  Gaussian of q puts at each value (`quasimomentum_spread.py`).
- **Mirror images.** A run at +q is the mirror image x -> -x of one at -q (the
  lattice, drive, starting cloud and green walls are symmetric), so a spread
  centred at 0 needs only q >= 0.
- **Change the spread without rerunning.** `metrics.csv` keeps one row per grid
  point and q (new column `initial_quasimomentum_kl`). MAKE combines them with
  the run's centre and spread, or with `QUASIMOMENTUM_CENTER_KL` /
  `QUASIMOMENTUM_SPREAD_KL` set in MAKE; the plot tools have the same two
  settings, and `python quasimomentum_spread.py metrics.csv --center 0
  --spread 0.1` writes a combined CSV.
- **What combines exactly.** The widths (StdDev from the second moments, not by
  averaging widths), cloud centre, central fraction, atoms lost, pendulum
  angle, momentum spread and orders, energy, return probability and the
  Bloch-state model's populations: they add up atom by atom. The IPRs,
  entropies, kurtosis, on-site spread, growth rates and level statistics
  depend on the square of the density or the whole time series, which
  `metrics.csv` does not keep; they are the average over q, weighted like the
  atoms.
- **Coverage.** The plan prints the share of the Gaussian inside the simulated
  values (97.6% for the defaults) and warns below 95%; widen
  `QUASIMOMENTUM_VALUES_KL` for wider or off-centre spreads.
- **Cost.** 5 values = 5x the GPU runs of 9.0 (500 for a 10 x 10 scan). The
  Bloch-state model and level statistics run at each run's own q. Compact
  storage only. No GPU code change: a 9.0 build runs 10.0 scans.

## Version 9.0

Version 9.0 keeps everything of 8.0 and starts the simulated cloud at the
experiment's width. One setting changed in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py`; with
`INITIAL_CLOUD_WIDTH_UM = 30` a run is identical to 8.0.

```python
INITIAL_CLOUD_WIDTH_UM = 340.0  # starting atom cloud, std of |psi|^2 along the lattice (8.0: 30)
```

- **Why 340 um.** In trial 11243 the cells where the drive barely acts
  (alpha = 4) have StdDev2 of 340-370 um, so the lab's cloud starts about
  340 um wide. 8.0 started at 30 um, which left every weakly driven cell about
  300 um narrower than the experiment. The width is the atom cloud along the
  lattice, not a beam waist. Set it from the lab's unmodulated reference trial
  when you have it.
- **Same box and GPU time as 8.0.** 262144 points (+-3.0 mm, absorbing walls
  from +-2.1 mm); three starting widths fit inside the walls. The 65536 box of
  4.0-7.0 cannot hold this cloud and is refused.
- `central_fraction` counts the atoms within two starting widths, now +-680 um.
- The dataset name ends in `_W340um`; the manifest and MAKE's columns record
  the width, and `sigma_x_4um_um_initial` is about 340 um.
- The plot tools (`~/Desktop/Kapitza-Phase diagrams`, not in this repository)
  have a `START_WIDTH_UM` setting (MATLAB: `startWidth`) that shows an 8.0
  run as if it had started wider, adding the width in quadrature.

## Version 8.0

Version 8.0 keeps everything of 7.0 and makes the simulation comparable with the
lab's phase diagrams (trial 11243: the simulated box cut off exactly the fast
atoms that make the widest experimental clouds). Three settings in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py`; with `GRID_POINTS =
65536`, `INITIAL_CLOUD_WIDTH_UM = 30` and `HOLD_TIME_MS = None` a run is
identical to 7.0.

```python
GRID_POINTS = 262144          # box +-3.0 mm, absorbing walls from +-2.1 mm (7.0: 65536)
INITIAL_CLOUD_WIDTH_UM = 30.0 # starting cloud, std of |psi|^2; set from the reference trial
HOLD_TIME_MS = 2.0            # drive time like the lab's t_mod; None = gpe1d.config's steps
```

- **Box.** 262144 grid points of 23 nm instead of 65536, so atoms flying out at a
  few hbar kL stay in the box for the whole hold instead of reaching the
  absorbing walls at +-0.53 mm and vanishing. About 4.5x the GPU time per point;
  about 52 MB of GPU memory per point (1.7 GB for a batch of 32).
- **Starting cloud.** The width of the Gaussian envelope of the starting
  state. The central window of `central_fraction` is two of these widths.
- **Hold time.** Turned into solver steps (2 ms = 68445 steps of 29.2 ns).
- The runner writes one config for the scan (`gpe1d_scan.config` in the
  results folder) with the box and the steps; every point and tab starts from
  it, and the generator writes the same `points_x` into each point's config.
- **Normalizing like `kpAnalysis.m`'s `isNormalize`.** The lab divides by the
  initial-state metric of its reference trial. The new columns
  `sigma_x_4um_um_initial` and `coarse_ipr_initial` hold the starting values
  of the camera-pixel width and IPR (`sigma_x_um_initial` and `metric_initial`
  already existed), so every point carries its own reference.
- The dataset name, the manifest and MAKE's columns record the box, the cloud
  and the hold time. No change to the GPU code: a 7.0 build runs 8.0 scans.

`VALIDATE_EXTENDED_CUDA.bat` now runs one batch with the 8.0 settings
(262144 points, 40 um cloud, 4 um smoothing, q = 0.1) and one with the 6.0
settings, and checks every sum against NumPy.

## Version 7.0

Version 7.0 keeps everything of 6.0 and adds two settings in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py`. Setting both to 0
gives exactly the 6.0 results.

```python
INITIAL_QUASIMOMENTUM_KL = 0.1   # starting quasimomentum of every atom, hbar kL
DensitySmoothing_um = 4.0        # smoothing of |psi|^2, like the camera's pixels
```

**Starting quasimomentum.** Every atom starts in the lowest-band Bloch state at
q (units of kL, -1 < q < 1) under the 30 um Gaussian envelope, instead of q = 0.
One setting serves every model: the GPU run, the Bloch-state Floquet model
(the `q0_` columns keep their 6.0 names; they now use this q) and the level
statistics. The classical ensemble has no quasimomentum. A single q > 0 makes
the cloud drift; widths are measured about its moving centre. At q != 0 the
Bloch-state model needs every plane wave, about three times the CPU time.

**Density smoothing.** The solver averages |psi|^2 over 175 grid points
(4.025 um) before every position metric: the IPR (`metric`), the cloud width
and centre (`sigma_x`, `x_mean`, ...), the density overlap and the extended
position columns (width growth and spreading speed, central fraction,
entropies, kurtosis, lattice-site IPR and occupied sites). Four kinds of
columns keep the raw density, because smoothing cannot apply to them:

- the pendulum angle and on-site spread (`pendulum_cos`, `onsite_spread` and
  their stroboscopic and time-fluctuation versions): they resolve where atoms
  sit inside a 532 nm site, which a 4 um window averages to about 0. A camera
  cannot see them either;
- the potential-energy part of `energy_absorbed_er`, for the same reason;
- momentum-space columns (`p_squared`, `order_0`, `order_2`) and the
  return probability (an overlap of wavefunctions, not an image);
- the 4 um pixel columns below (binning into pixels is itself the smoothing).

**4 um pixel columns** (MuscleMuseum's definitions on 4 um camera pixels):
`coarse_ipr` (computeIPR, from 6.0) and the new `sigma_x_4um_um`
(computeStDev).

Build and validate on the Windows machine, as for 6.0:

```powershell
.\BUILD_CUDA.bat
.\VALIDATE_FAST_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
.\VALIDATE_EXTENDED_CUDA.bat
```

`VALIDATE_COMPACT_CUDA.bat` adds two smoothed cases (the GPU's smoothed IPR and
width against NumPy on the saved snapshots). `VALIDATE_EXTENDED_CUDA.bat` runs
one batch with the 7.0 settings (4 um smoothing, q = 0.1) and one with the 6.0
settings, and checks every sum against NumPy. Smoothing needs the fused loop;
`--reference-loop` is refused while it is on. Full-storage scans
(`--storage full`) save raw snapshots; MAKE smooths them the same way.

## Version 6.0

Version 6.0 adds **extended metrics**: one switch in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py`,

```python
ExtendedMetrics = True
```

adds 43 columns to `metrics.csv` (and to MAKE's analysis CSV) and turns on the
level statistics. Everything else is the 5.0 model: fixed lattice, drive
sin(omega t + phi), `INVERTED`. With `ExtendedMetrics = False` a run is
identical to 5.0. The meaning of every column is in
`phase_diagram/simulation_core/extended_fields.py`.

| Source | Columns | What |
| --- | --- | --- |
| GPU simulation (no prefix) | 25 | Sums the solver records every status step (every 100 steps): pendulum angle `pendulum_cos` = <cos 2kL x> (+1 bottom, -1 top), on-site spread, stroboscopic versions, early growth rates of the width and of <p^2>, spreading speed, central fraction, escape fraction, density and site entropy, kurtosis, site IPR and occupied sites, 4 um coarse IPR, return probability, density overlap, <p^2>, the p = 0 and +-2 hbar kL orders, absorbed energy, starting width and IPR |
| q = 0 Floquet model (`q0_`) | 15 | The same Hamiltonian for one site's Bloch state, once per drive period: pendulum angle, on-site spread, <p^2>, orders, absorbed energy, on-site survival, band populations, Floquet participation, slow frequency, echo fidelity |
| Classical ensemble (`cl_`) | 3 | Point particles in the same lattice: pendulum angle, fraction near the start, fraction that left its site |

The GPU columns use the same final window as the IPR (the last 30 samples).
Display choices (Mathieu axes, threshold and edge maps, log scales, smoothing,
difference maps) are made when plotting, from these columns.

Extended metrics need compact storage, the fused loop and a 6.0 build; the
runner refuses anything else. Build and validate on the Windows machine:

```powershell
.\BUILD_CUDA.bat
.\VALIDATE_FAST_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
.\VALIDATE_EXTENDED_CUDA.bat
```

`VALIDATE_EXTENDED_CUDA.bat` runs a non-inverted and an inverted point as one
GPU batch with snapshots and checks every extended sum against NumPy. Cost: the
GPU does about 20 extra reductions every 100 steps; the CPU adds about 0.1-0.3 s
per point for the q = 0 model, a few seconds per batch for the classical
ensemble, and the level statistics.

## Version 5.0

Version 5.0 drives and loads the lattice **as in the experiment**. The lattice
stays fixed and the drive is a sine with a phase:

```text
5.0:  H(t) = p^2/2m - V0 [1 + alpha sin(omega t + phi)] cos^2(kL x)
4.0:  H(t) = p^2/2m - V0 [1 + alpha cos(omega t)] cos^2(kL x + PHASE_RADIANS/2)
```

`alpha` is the experiment's alpha*beta. The atoms start in the ground state of
the lattice at t = 0 at `INITIAL_LATTICE_DEPTH_V0_ER`,
`-V_init [1 + alpha sin(phi)] cos^2(kL x)`, as they are loaded in the
experiment, so at t = 0 only the depth changes (V_init to V0). In 4.0 the drive
started at cos(0) = 1, so the lattice jumped to (1 + alpha) V0 at t = 0, and
flipped for the inverted case.

`INVERTED` in `phase_diagram/simulation_core/run_phase_diagram_CUDA.py`
replaces `PHASE_RADIANS`:

| `INVERTED` | phi | Atoms start | Lattice at t = 0 |
| --- | --- | --- | --- |
| `False` | 0 | bottom, cos^2(kL x) = 1 | -V0 cos^2(kL x) |
| `True` | arcsin(-2/alpha), for each alpha; needs alpha >= 2 | top of the time-averaged lattice, cos^2(kL x) = 0 | +V0 cos^2(kL x) (flipped) |

Datasets are named `..._Inverted_...` or `..._NonInverted_...`, MAKE's
analysis CSV has an `inverted` column (1 or 0; blank for 4.0 data, which keeps
`phase_radians`), and `point_results.jsonl` records each point's
`drive_phase_radians`.

The solver change is one config key, `floquet_phase`: the drive is
cos(omega t + floquet_phase), and 5.0 writes floquet_phase = phi - pi/2, which
is sin(omega t + phi). Without the key it is 0, the 4.0 drive bit for bit. Rebuild
and validate on the Windows machine before scanning:

```powershell
.\BUILD_CUDA.bat
.\VALIDATE_FAST_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
```

`VALIDATE_FAST_CUDA.bat` now also checks the 5.0 drive phase: the fused loop
against the original loop bit for bit, and a batch mixing 4.0 and 5.0 points.
The runner refuses an executable built before 5.0, which would ignore the
phase. Everything else is unchanged: the absorber, green walls, the analysis,
and the level statistics (Floquet quasienergies do not depend on where in the
cycle the drive starts). Validation against the CPU reference still uses the
4.0 inputs. Version 4.0 stays on the `ExpansionUpgrade1.0` branch.

## Version 4.0

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

Answer `all` at the colour question, or pass `--color all`, for one picture
per result column that varies across the scan, each with automatic colour
limits. A CSV path plots that file, e.g. a finished scan's `metrics.csv`, with
the pictures saved in an `analysis` folder beside it:

```powershell
.\MAKE_PHASE_DIAGRAM_CUDA.bat path\to\metrics.csv --color all --no-ask
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
