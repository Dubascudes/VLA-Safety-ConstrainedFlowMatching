# Tests

Run `python -m unittest discover -s tests -v` from the repository root. For full coverage, use the OpenPI environment, materialize the runtime, and provide `VLA_OPENPI_SOURCE` and `VLA_SAFELIBERO_SOURCE` Git mirrors. `VLA_RUNTIME` selects a runtime path.

Materialization tests are dependency-light. Policy regressions require NumPy, Torch, JAX, Flax, SciPy, and remaining OpenPI dependencies. Missing runtime/dependencies result in explicit skips. No test downloads weights or loads a VLA checkpoint.
