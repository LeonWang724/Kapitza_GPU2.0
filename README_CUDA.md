<!-- Codex CUDA Port: build, run, and validation instructions. -->
# Native CUDA `gpe1d_2` port

This tree contains a native C++/CUDA 1D solver named `gpe1d_cuda.exe`. Python only generates inputs, launches executables, records provenance, and analyzes saved data. The GPE evolution is compiled CUDA code using double-precision complex arithmetic and `cufftExecZ2Z`.

## Validation status

The native target has configured and compiled on the Windows CUDA machine. The
physical Floquet update reproduced the established CPU phase diagram; it is now
the workflow default. The cumulative source-compatibility behavior remains an
internal validation probe and is not offered by the phase-diagram launchers.

The new compact-output path has local Python workflow and host C++ tests. Run
`VALIDATE_COMPACT_CUDA.bat` on the Windows GPU after rebuilding to establish
numerical agreement for your CUDA build; its GPU checks have not been run on the
Mac used to prepare this update.

The validated delivery scope is the 1D phase-diagram path with `dynamic_potential=false`. Stochastic dynamic potential, 2D, and 3D stop with an explicit error or remain deferred; they are not silently run through a different implementation.

## Windows prerequisites

- 64-bit Windows 11 and an NVIDIA driver supporting your GPU and selected toolkit.
- Visual Studio 2022 with **Desktop development with C++**, MSVC x64 tools, and a Windows SDK.
- NVIDIA CUDA Toolkit supporting your GPU, driver, and Visual Studio compiler. Use CUDA 12.9 for Pascal GPUs (including compute capability 6.1); CUDA 13 removed compilation support for architectures older than Turing (7.5). See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures). CUDA 13.3 is suitable for supported newer GPUs, including the RTX 5090.
- CMake 3.27 or newer. The preset uses `CMAKE_CUDA_ARCHITECTURES=native`, documented by CMake 3.24 and later: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
- Ninja 1.11 or newer. The `Ninja Multi-Config` generator uses the MSVC and CUDA compiler environment loaded by `LOAD_CUDA_ENV.bat`.
- A 64-bit HDF5 development installation containing `include`, `lib`, and runtime `bin` directories. Set `HDF5_ROOT` to that installation. The HDF Group's official CMake guide documents `HDF5_ROOT` and the Windows runtime path: https://github.com/HDFGroup/hdf5/blob/develop/docs/INSTALL_CMake.md
- Python 3.12 and the packages in `phase_diagram/simulation_core/requirements_CUDA.txt`.
- Intel oneAPI oneMKL development files for the source-built CPU validation reference. Setup installs these in the project `.venv`.

## One-click Windows setup

The easiest supported path is to double-click `SETUP_WINDOWS.bat`, approve the
Administrator prompt, and let it finish. It installs or repairs Python 3.12,
Git, CMake, Ninja, the Visual Studio 2022 C++ build tools, CUDA Toolkit 13.3.1, HDF5
2.1.1, a project-local `.venv`, oneMKL, and all packages in `requirements_CUDA.txt`.
The full transcript is saved as `SETUP_WINDOWS.log`.

Setup permanently adds the installed Python directory, Python `Scripts`, and
the project `.venv\Scripts` directory to the current user's PATH. It records
the selected interpreter in `GPE_PYTHON` and the project directory in
`GPE_CUDA_PROJECT_ROOT`. CMake, CUDA, and HDF5 are added to the machine PATH;
the project-local oneMKL paths are saved for the current user.
Open a new VS Code terminal after setup to inherit those persistent values.

The CUDA installer can install its bundled display driver, and setup verifies
that `nvidia-smi` works; it does not separately choose between NVIDIA's latest
Game Ready and Studio driver branches.

After setup, double-click these files in order:

```text
CHECK_CUDA.bat
BUILD_CUDA.bat
VALIDATE_CUDA.bat
```

Those three files now pause before closing. All CUDA batch files call
`LOAD_CUDA_ENV.bat`, which automatically loads the project Python environment,
CMake, CUDA, HDF5, oneMKL, and (when compiling) the Visual Studio x64 developer
environment. A special VS command prompt and manual PATH editing are no longer
required.

The loaders explicitly select this project's `.venv` interpreter, include both
CUDA `bin` and `bin\x64` runtime directories, and honor a valid `CUDA_PATH`.
When it is unset or invalid, the loader searches the standard toolkit locations.
Build/check commands use
`%PUBLIC%\KapitzaCudaTemp` for temporary files, avoiding spaces in Windows user
names. Set `GPE_CUDA_TEMP` to another writable path without spaces if needed.
Global Windows TEMP/TMP settings are not changed. Builds refresh CMake's
configuration so an old cached compiler cannot silently remain selected.

The build uses CMake's `Ninja Multi-Config` generator with MSVC and `nvcc`.
This intentionally avoids a dependency on NVIDIA's optional Visual Studio
MSBuild Build Customizations, which are not required for this command-line build.

### Pascal GPU: `Unsupported gpu architecture 'compute_61'`

The `native` architecture preset detected a compute-capability 6.1 GPU, but CUDA
13 cannot compile for it. Install [CUDA Toolkit 12.9 Update 1](https://developer.nvidia.com/cuda-12-9-1-download-archive)
if needed, then select it in PowerShell before building and running:

```powershell
$env:CUDA_PATH = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
if (!(Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { throw "Install CUDA Toolkit 12.9 first." }
.\BUILD_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
```

Keep using this terminal for the run launchers, or save this selection for future
terminals after confirming that CUDA 12.9 is installed:

```powershell
[Environment]::SetEnvironmentVariable('CUDA_PATH', $env:CUDA_PATH, 'User')
```

The general setup installer still defaults to CUDA 13.3.1; it is not necessary to
rerun it to switch between already installed toolkits. Select 12.9 again after
running setup on a Pascal machine. Do not set a newer architecture to bypass the
error: the binary must support the GPU that will execute it.

## Build

If you prefer a terminal instead of the one-click setup, open PowerShell in the
project directory. After `SETUP_WINDOWS.bat` has completed, the build is simply:

```bat
CHECK_CUDA.bat
BUILD_CUDA.bat
```

The normal build compiles both `gpe1d_cuda.exe` and the untouched 1D source as
`gpe1d_cpu_reference.exe`. For a CUDA-only development build:

```bat
BUILD_CUDA.bat CUDA_ONLY
```

The native executable is `build\bin\gpe1d_cuda.exe`. The build embeds its Git commit, dirty state, compilers, CUDA Toolkit, architecture selection, and UTC build time. No fast-math option is used; strict floating-point mode disables FMA by default for traceability.

## Single run

Make sure the config names an empty output directory, then run:

```bat
RUN_SINGLE_CUDA.bat phase_diagram\simulation_core\gpe1d.config
```

Single runs use the verified physical update, which evaluates the unchanged
Floquet base potential times the current cosine.

## Validation before a full grid

```bat
VALIDATE_CUDA.bat
```

This runs the requested static, zero-alpha, driven, source-compatibility,
physical-update, frequent-snapshot, non-unit-`step_x` diagnostic probe, and 3x3
grid checks. The source-built CPU executable is the required reference. If a
preserved opaque workflow executable is present locally, it is used as an
additional reference but is not required. Reports are written under
`validation\results\validation_*`.

The default tolerances in `validation_cases.json` are initial acceptance hypotheses. They become documented evidence only after a completed Windows report. Capture `nvidia-smi` output during a CUDA run as external GPU-use evidence.

For an auditable first Windows pass, run these exact commands from a terminal in
the project directory after setup:

```bat
set GPE_NO_PAUSE=1
CHECK_CUDA.bat > check_cuda.log 2>&1
BUILD_CUDA.bat > build_cuda.log 2>&1
start "RTX 5090 monitor" cmd /k nvidia-smi -l 1
VALIDATE_CUDA.bat > validation_console.log 2>&1
```

Keep `check_cuda.log`, `build_cuda.log`, `validation_console.log`, and the newest `validation\results\validation_*` directory together. A clean Git clone contains the initial-state generator, base configuration, and the untouched 1D source needed for this validation; it does not depend on the old opaque executable.

## Phase diagram with compact storage (default)

Edit only the marked physical parameter section near the top of
`run_phase_diagram_CUDA.py`. `INITIAL_LATTICE_DEPTH_V0_ER` separately records
the depth used to construct the initial Bloch state. Then run:

```bat
RUN_PHASE_DIAGRAM_CUDA.bat
MAKE_PHASE_DIAGRAM_CUDA.bat
```

After updating an existing installation, rebuild once and first run:

```powershell
.\BUILD_CUDA.bat
.\VALIDATE_COMPACT_CUDA.bat
```

Compact mode computes the same unnormalized, cropped discrete sum of `|psi|^4`
at the final 30 historical snapshot times, then averages those scalars. It does
not save the wavefunctions or potentials. The window still uses
`save_every_nth_iteration`; index zero refers to the state AFTER the first step,
just as in the original solver. GPU reduction order can cause tiny floating-point
differences from NumPy; the validation command checks the actual difference.

Each dataset contains a `metrics.csv` with one row per point, an append-only
`point_results.jsonl` with scalar results and per-point provenance, a small
`run_manifest.json`, and one copy of the base config/input-generator source.
Completed points are flushed to disk before temporary inputs are reused. A
single scratch directory holds only the current point's inputs/status/log; it
is removed after the sweep. A failure preserves that point's diagnostics in
`failed_point` and leaves completed scalar records intact. Automatic resume is
not implemented; do not restart into a nonempty results directory.

The 65,536-point defaults previously wrote about 700 MiB per grid point in field
arrays alone (700 snapshots, two float64 arrays each). Compact storage scales as
one table row plus roughly a kilobyte of metadata per point, plus a few MiB of
reusable working inputs. A 100x100 sweep should therefore use tens of MB of
retained results instead of multiple TB. Exact sizes depend on metadata and
plot generation. Compact results cannot reconstruct a wavefunction or compute
new observables afterward; choose full mode if you need that information.

To retain the original snapshot workflow:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat --storage full
```

Both modes accept an external drive and a new/empty results folder:

```powershell
.\RUN_PHASE_DIAGRAM_CUDA.bat --results "E:\Kapitza\run001"
.\MAKE_PHASE_DIAGRAM_CUDA.bat "E:\Kapitza\run001"
```

The make command automatically selects the newest completed dataset. You may
instead pass a dataset folder name or manifest path. Each grid folder is named
like `LatticeDepth_20ER_InitialDepth_40ER_Phase_0rad_cuda_TIMESTAMP`. Only full
mode retains isolated inputs, config, status, log, and `out_###` directories.
Existing snapshot datasets remain readable. Existing data are never deleted or
converted automatically by this update.

Every analysis execution writes a new, timestamped PNG, CSV, and NPZ under that
dataset's `analysis` directory, so earlier analyses are never overwritten. The
artifact filenames use a short timestamp to avoid repeating a long dataset name
and exceeding Windows' path limit. Dataset identity and physical parameters
remain in the manifest/NPZ and plot title. The colorbar now correctly names the
historical `sum |psi|^4` metric; its values have not changed to a different observable. To pin
a particular dataset or customize the plot title and axis labels, edit the
clearly marked configuration values near the top of
`make_phase_diagram_CUDA.py`. Command-line overrides are also available.

## Numerical correspondence

Each real-time iteration performs:

1. Floquet potential update at the current time.
2. Density and nonlinear/potential half step.
3. Unnormalized forward cuFFT.
4. Explicit `1/N` scaling and kinetic propagator.
5. Unnormalized inverse cuFFT.
6. Recomputed density and the second nonlinear/potential half step.
7. Optional imaginary-time normalization.
8. Post-step snapshot and diagnostic output at the original iteration indices.

HDF5 remains CPU-side. Device-to-host transfers occur for input/output fields in
full mode and diagnostic scalars. Compact mode reduces the phase metric on the
GPU and does not transfer or write output wavefunction arrays.
