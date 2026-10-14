# Safety-Constrained Flow Matching for VLAs

Reference code for **Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching**, William English, Hao Zheng, and Rickard Ewetz.

- [Paper](https://arxiv.org/abs/2607.01378)
- [Project page](https://willenglish.tech/SafetyGuidedFlowMatching/)
- Base policy: OpenPI `pi05_libero` (PyTorch checkpoint).
- Environment: SafeLIBERO from VLSA-AEGIS, with captured local modifications.

## Status

**Initial source extraction, not yet a validated end-to-end release.** Source reconstruction and syntax checks are verified. Fresh dependency installation, GPU inference, live GUI operation, and reproduction of paper results are not yet verified. See [known issues](docs/known-issues.md).

The live viewer is a first-class release component. This extraction preserves its current behavior; repairs will be tested separately.

## Contents

- `patches/openpi.patch`: model, policy, server, and client modifications.
- `patches/safelibero.patch`: environment and obstacle-geometry modifications.
- `overlay/examples/safelibero/`: live viewer, geometry helper, dependency snapshots.
- `provenance/source-manifest.json`: upstream revisions and captured SHA-256 hashes.
- `tools/materialize.py`: reconstruct upstream sources plus patches and overlay.
- `licenses/`: preserved third-party notices.
- `docs/`: setup, provenance, limitations, and implementation plan.
- `tests/`: materialization tests.

Weights, videos, datasets, virtual environments, and unrelated experiments are not included.

## Reconstruct source

Requires Git and Python 3.10+ for the materializer. OpenPI requires Python 3.11+; the captured simulator requirements use a separate older environment.

```bash
python tools/materialize.py
```

This clones pinned upstream revisions into `runtime/`, applies patches, copies the viewer, links SafeLIBERO, and checks captured source hashes. Existing destinations are refused. It downloads upstream simulator assets, but does not install dependencies or download model weights.

Offline reconstruction using existing upstream repositories:

```bash
python tools/materialize.py --dest runtime \
  --openpi-source /path/to/openpi \
  --safelibero-source /path/to/vlsa-aegis
```

Only committed content is taken from mirrors; the release patches supply the research changes.

## Live viewer

Read [setup notes](docs/setup.md) first. After installing the separate policy and simulator environments, run from `runtime/openpi`.

Policy server, in the policy environment:

```bash
python scripts/serve_policy.py --env LIBERO --enable-safety --safety-margin 0.01
```

Viewer, in the simulator environment:

```bash
python examples/safelibero/eval_safelibero_visualized.py \
  --host 127.0.0.1 --port 8000 \
  --task-suite-name safelibero_spatial --safety-level I \
  --task-index 0 --only-episode 0 \
  --safety-filter flow --safety-margin 0.01 \
  --live-display --save-videos \
  --video-out-path data/demo/spatial_I_flow
```

These are source-derived entry points, not recovered paper-run commands or fresh-install validated instructions. Set the server margin explicitly: its default is zero, overriding the model's 0.01 m default.

Modes: `none` sends no safety metadata; `cbf` applies a single-action filter during execution; `flow` requests guidance during generation; `flow+cbf` adds a trajectory filter after flow guidance. Do not label the `cbf` mode a verified reproduction of published AEGIS results without checking baseline provenance.

The viewer provides task/suite dropdowns, safety sliders, pause/restart/skip controls, videos, and trajectory overlays. **Flow-mode slider values currently are not applied by the server.** Omit `--live-display` for noninteractive evaluation; a configuration-logged benchmark runner remains future work.

## Tests

```bash
python -m unittest discover -s tests -v
```

Enable the offline reconstruction integration test with:

```bash
VLA_OPENPI_SOURCE=/path/to/openpi \
VLA_SAFELIBERO_SOURCE=/path/to/vlsa-aegis \
python -m unittest discover -s tests -v
```

Without mirrors, integration testing is explicitly skipped. These checks verify packaging, not robot behavior or safety.

## Provenance and scope

The experiments ran on HiPerGator; the author selected the current local OpenPI implementation for this release after comparison. This is not yet an exact experiment-time snapshot. See [provenance](docs/provenance.md).

The method uses simulator-provided obstacle state and simplified end-effector geometry/dynamics. It does not establish perception-grounded, full-body, or real-world safety guarantees.

## Licensing

Upstream notices are retained in `licenses/` and reconstructed dependencies. A license for original contributions must be selected before publication. The manuscript license does not automatically license the code.

## Citation

```bibtex
@article{english2026safety,
  title={Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching},
  author={English, William and Zheng, Hao and Ewetz, Rickard},
  journal={arXiv preprint arXiv:2607.01378},
  year={2026}
}
```
