# Safety-Constrained Flow Matching for VLAs

Code for [Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching](https://arxiv.org/abs/2607.01378), by William English, Hao Zheng, and Rickard Ewetz.

We add trajectory-level safety guidance to OpenPI’s π0.5 policy and evaluate it in SafeLIBERO. The live viewer lets you run manipulation tasks, compare safety configurations, and save videos of the resulting behavior.

[Paper](https://arxiv.org/abs/2607.01378) · [Project page](https://willenglish.tech/SafetyGuidedFlowMatching/)

## What’s included

- **Safety-guided OpenPI:** corrections to predicted action trajectories during flow-matching inference.
- **SafeLIBERO integration:** environment changes and obstacle geometry used by the safety code.
- **Live experiment viewer:** task selection, safety controls, trajectory visualization, and video recording.
- **Tests and experiment notes:** regression tests, dependency information, and a template for recording runs.

The viewer source lives in `experiments/processing/safelibero/`. Reusable code and patches live in `software_module/`. You can find environment details and experiment procedures in `miscellaneous/`.

## Setup

You’ll need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), an NVIDIA GPU for the policy, and a graphical session for the live viewer. Use separate Python environments for OpenPI and SafeLIBERO because they require different dependency versions.

### 1. Prepare OpenPI and SafeLIBERO

From this repository’s root directory, run:

```bash
python experiments/processing/materialize.py
```

This downloads the pinned upstream code, applies our changes, and places the working code in `runtime/`. Model weights are downloaded separately when you first load the policy. If `runtime/` already exists, move it aside or choose another location with `--dest`.

### 2. Set up the policy environment

```bash
cd runtime/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --link-mode copy
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python --link-mode copy -e . scipy
```

Apply OpenPI’s PyTorch compatibility changes to Transformers:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

The copy link mode above keeps these edits separate from uv’s shared package cache.

### 3. Set up the simulator environment

Still in `runtime/openpi`, create a separate environment for SafeLIBERO:

```bash
uv venv --python 3.8 examples/safelibero/.venv
uv pip install --python examples/safelibero/.venv/bin/python \
  -r examples/safelibero/requirements.txt \
  -r third_party/safelibero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match
uv pip install --python examples/safelibero/.venv/bin/python \
  -e packages/openpi-client -e third_party/safelibero
```

These commands follow the captured dependency files and upstream setup instructions; a fresh-machine installation has not yet been validated. See [environment notes](miscellaneous/software_information/setup.md) for details.

## Run the viewer

Start the policy server from `runtime/openpi`:

```bash
.venv/bin/python scripts/serve_policy.py \
  --env LIBERO --enable-safety \
  --safety-margin 0.01 --guidance-scale 1.0
```

The server uses the `pi05_libero` configuration and the PyTorch π0.5-LIBERO checkpoint. Allow time for the first checkpoint download and model load.

Open a second terminal, change to `runtime/openpi`, and start the viewer:

```bash
export PYTHONPATH="$PWD/third_party/safelibero${PYTHONPATH:+:$PYTHONPATH}"
examples/safelibero/.venv/bin/python \
  examples/safelibero/eval_safelibero_visualized.py \
  --host 127.0.0.1 --port 8000 \
  --task-suite-name safelibero_spatial --safety-level I \
  --task-index 0 --only-episode 0 \
  --safety-filter flow --safety-margin 0.01 --guidance-scale 1.0 \
  --live-display --save-videos \
  --video-out-path data/demo/spatial_I_flow
```

Use the viewer’s dropdowns to choose tasks and its sliders to adjust the safety margin and correction strength. You can pause, restart, or skip an episode. The command above starts with one episode; remove `--only-episode 0` to run multiple trials.

Choose a safety mode with `--safety-filter`:

| Mode | Behavior |
| --- | --- |
| `none` | Run the policy without safety filtering. |
| `cbf` | Filter individual actions before execution. |
| `flow` | Apply trajectory corrections during generation. |
| `flow+cbf` | Combine flow guidance with an additional trajectory filter. |

A guidance strength of **1.0** uses the solver’s correction unchanged. Other values scale that correction and may no longer satisfy its safety constraints.

Videos go to `--video-out-path`. To save denoising traces as well, add `--trace-dir /path/to/traces` to the **server** command. For evaluation without a live window, omit `--live-display` from the viewer command.

## Citation

```bibtex
@article{english2026safety,
  title={Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching},
  author={English, William and Zheng, Hao and Ewetz, Rickard},
  journal={arXiv preprint arXiv:2607.01378},
  year={2026}
}
```
