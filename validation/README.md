<!-- Codex CUDA Port: validation workflow guide. -->
# Validation workflow

`run_validation.py` generates identical per-case HDF5 inputs and separate CPU/CUDA configuration files. It never writes CPU and CUDA snapshots into the same directory.

For legacy Floquet mode it compares every common checkpoint after removing one global complex phase. Reports include wavefunction and density errors, normalization, and all energy columns. The opaque/source CPU trajectory is also compared against the CUDA physical mode as a compatibility probe, not as an expected match.

A short non-unit-`step_x` case compares the executable's `InitOverlap` column with both `sum(|psi_initial|^2 |psi|^2)` and `step_x * sum(|psi_initial|^2 |psi|^2)`. This is designed to recover whether the opaque diagnostic contains an integration step.

The default CPU references are:

1. `build/bin/gpe1d_cpu_reference.exe`, when the untouched MKL source target was built.
2. The opaque `phase_diagram/simulation_core/gpe1d_2.exe`.

The optional 3x3 grid uses the established cropped, unnormalized `sum(|psi|^4)` metric over 30 final snapshots. Default tolerances are hypotheses to be tested on the Windows machine, not an advance claim of equivalence. The energy tolerance is intentionally limited to `5e-6` because the supplied CPU executables serialize status values with roughly six significant digits; tighter energy evidence requires recomputing diagnostics from binary checkpoints or changing the reference output precision.

Run from the project root with `VALIDATE_CUDA.bat`.
