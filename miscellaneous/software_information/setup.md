# Environment setup notes

These notes identify the captured dependencies, not a tested fresh-install recipe.

## Policy server

The reconstructed `runtime/openpi/pyproject.toml` and `uv.lock` are pinned through the OpenPI commit. Python 3.11+ is required. The manifest includes Torch 2.7.1, Transformers 4.53.2, and JAX 0.5.3 with CUDA 12 support. Follow the pinned upstream README, including any required Transformers replacement instructions. Check SciPy availability for the custom SLSQP solver.

The server selects config `pi05_libero` and checkpoint `gs://openpi-assets/checkpoints/pi05_libero_pytorch`. Weights are external and not distributed here. A checkpoint URI is not a captured weight checksum; exact weight provenance remains to be established.

## Simulator and viewer

The captured `examples/safelibero/requirements.in` and compiled `requirements.txt` specify the older simulator stack (including Torch 1.11/CUDA 11.3 and robosuite 1.4.1). The historical setup used a Python 3.8 simulator environment separate from the server. Do not merge it blindly into the OpenPI environment.

Install the reconstructed `packages/openpi-client` and `third_party/safelibero` into that simulator environment. The latter is linked by the materializer to the patched VLSA-AEGIS SafeLIBERO tree. Ensure that `import libero` resolves to this tree, not another installed LIBERO checkout. Run from `runtime/openpi` and set `PYTHONPATH` to include its `third_party/safelibero` if needed.

Live display requires a graphical session and OpenCV GUI support. Headless rendering and live display are separate concerns. On HiPerGator, model loading, installation/build work, and rollouts belong in scheduled compute allocations, not login nodes.

Trace output is disabled by default; use the server option `--trace-dir` to enable current-call trace saving. The CPU regression suite runs in an existing OpenPI environment, but a fresh dependency installation, GPU inference, and live viewer operation remain unverified.
