# Kapitza_GPU2.0

Native CUDA simulation of a one-dimensional driven optical lattice, with compact
phase-diagram output. This version builds on
[Kapitza_Phase_Diagram_GPU](https://github.com/LeonWang724/Kapitza_Phase_Diagram_GPU)
and keeps the complex128 split-step solver and existing phase-diagram metric.

## Windows quick start

```powershell
git clone https://github.com/LeonWang724/Kapitza_GPU2.0.git
cd Kapitza_GPU2.0
.\SETUP_WINDOWS.bat
.\BUILD_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
```

Run each command after the previous command finishes successfully. Setup installs
or repairs the Windows prerequisites and may request Administrator access.
Then edit the physical parameter section in
`phase_diagram/simulation_core/run_phase_diagram_CUDA.py` and run:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat
.\MAKE_PHASE_DIAGRAM_CUDA.bat
```

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
- Build temporary files use a path without spaces.
- Builds refresh CMake configuration to avoid stale CUDA compiler paths.
- Short analysis filenames avoid duplicating long dataset names.

Twelve local Python and host C++ tests passed while preparing this release.
Windows setup, NVCC compilation, and GPU execution still require verification on
the target machine. `VALIDATE_COMPACT_CUDA.bat` compares four small full/compact
GPU cases against NumPy snapshot analysis and checks that compact mode produces
no HDF5 output snapshots. Run it before starting a large grid.

See [README_CUDA.md](README_CUDA.md) for configuration, dependencies, numerical
details, and the original CPU/CUDA validation workflow.
