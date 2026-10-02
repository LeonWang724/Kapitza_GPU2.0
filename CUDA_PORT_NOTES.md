<!-- Codex CUDA Port: audit trail for every port addition and scientific decision. -->
# CUDA port notes

## Reference integrity

`timedependent_GPU_native` began as a copy of the original `timedependent` tree. `reference/gpe1d_2-master` is a byte-for-byte copy of the supplied source tree. Directory comparisons were clean before port files were added. The original Desktop trees were not edited.

The executable bundled with the source and the executable used by the phase-diagram workflow have different SHA-256 hashes. The workflow executable contains `Corr:` and a CSV header ending in `InitOverlap`; the available source does not. See `reference/executable_hashes.json` and `REFERENCE_SHA256SUMS.txt`.

## Numerical decisions

- Complex values are `cuDoubleComplex`; transforms are `CUFFT_Z2Z`.
- The original MKL forward and backward transforms are both unnormalized. The source divides the forward result by `N`; the CUDA port does the same before the kinetic multiplication.
- The original wavenumber ordering maps index `N/2` to the negative Nyquist wavenumber. The CUDA initialization reproduces that ordering.
- Snapshots are post-step values saved at iteration 0 and every configured interval, with sixteen-digit filenames.
- Potential energy uses the real part of `density * potential`, matching the source loop.
- The workflow-compatible `InitOverlap` field is currently defined explicitly as the discrete sum `sum_j |psi_initial[j]|^2 |psi[j]|^2`, without `dx`. Its scale is consistent with observed workflow CSV values when `step_x=1`, but equivalence to the opaque executable is not claimed. A controlled non-unit-`step_x` executable run is needed to recover whether the legacy field includes an integration step.
- The physical Floquet update, which multiplies the unchanged base array by the current cosine, reproduced the established CPU phase diagram and is the user-workflow default. The cumulative update from the supplied source is retained only for internal source-compatibility validation.
- Strict CUDA floating point disables FMA and does not enable fast math. CPU/CUDA FFT and reduction order can still produce roundoff differences.
- The initial wavefunction/density tolerances use binary HDF5 values. Energy comparisons use the supplied CPU status CSV and therefore cannot demand more precision than its approximately six-significant-digit serialization; the initial energy relative threshold is `5e-6` and remains provisional until Windows data is available.

## Phase-diagram quantity

The latest historical scripts select the final 30 HDF5 snapshots, cut 100 points from both spatial edges, and ultimately overwrite a standard-deviation calculation with `sum(probability**2)`. The actual plotted value is therefore the mean unnormalized discrete `sum |psi|^4`, not standard deviation and not a continuum-normalized IPR. The new plotter preserves the alpha-outer/frequency-inner ordering, inverted frequency axis, inferno colormap, cut, and averaging. At the workflow owner's request, the displayed colorbar retains the historical `Standard Deviation of |psi|^2` wording; the manifest and this audit note retain the exact numerical definition. Files are numerically sorted instead of relying on unspecified `glob` order.

## Added files

- Build and provenance: `.gitignore`, `CMakeLists.txt`, `CMakePresets.json`, `cmake/CpuReference.cmake`, `REFERENCE_SHA256SUMS.txt`, `reference/executable_hashes.json`.
- Windows entry points: `BUILD_CUDA.bat`, `CHECK_CUDA.bat`, `RUN_SINGLE_CUDA.bat`, `RUN_PHASE_DIAGRAM_CUDA.bat`, `MAKE_PHASE_DIAGRAM_CUDA.bat`, `VALIDATE_CUDA.bat`.
- Native source: every file under `src/`.
- Workflow: `cuda_workflow_common.py`, `run_phase_diagram_CUDA.py`, `run_single_CUDA.py`, `make_phase_diagram_CUDA.py`, `requirements_CUDA.txt`.
- Validation: every file under `validation/`.
- Local tests: every file under `tests/`.
- Documentation: `README_CUDA.md` and this file.

The clean-clone validation path also tracks the unchanged 1D reference `.cpp`
and `.h` files, `create_initial_state_function.py`, `config_value.py`, and the
base `gpe1d.config`. Large historical outputs and the opaque executable remain
excluded. `src/cpu_reference_main.cpp` is a new 1D-only wrapper around those
untouched CPU source files.

No copied original source or workflow file was modified to implement the port.

## Deferred work

- Stochastic `dynamic_potential=true`: the original uses time-seeded MKL VSL random phases. A CUDA version needs an explicitly chosen cuRAND generator and a statistical, rather than trajectory-identical, validation contract.
- 2D/3D solvers and the 3D reduction path.
- Windows execution of the included non-unit-`step_x` `Corr` probe.
- Formal tolerance evaluation, retained `nvidia-smi` evidence, and performance measurements.
