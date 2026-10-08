Minimal declarations of the CUDA, cuFFT and CUB APIs used in `src/`. They let
a machine without the CUDA Toolkit type-check the solver with clang
(`tests/test_fused_solver.py`) and compile the host equivalence test. They are
never used by the real build. `cuComplex.h` reproduces the toolkit's
arithmetic exactly, because the host test relies on it.
