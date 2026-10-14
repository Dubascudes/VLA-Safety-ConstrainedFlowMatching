# Release status and remaining gates

## Fixed in the release patch

- Viewer margin and correction strength propagate into request-local safety configuration.
- Strength defaults to 1.0 and scales flow and posthoc corrections. Nonunit strength is experimental, not a constraint-preserving operation.
- Server defaults use the model's 0.01 m margin and 0.0 minimum guidance time; explicit server options are applied. Nonfinite and negative margins/strengths are rejected in both server arguments and model configuration.
- Trace collection is off by default. `--trace-dir` selects a portable output location, and each current call is saved immediately with a unique filename. The traced posthoc branch returns corrected posthoc actions, matching the untraced branch.

Original patches and source hashes remain archived; these repairs are not retroactively described as the paper's experiment configuration.

## Remaining

1. Fresh policy/simulator installation and real checkpoint-backed GUI/rollout verification.
2. Mixed numeric task-index conventions in the dropdown/evaluator paths; verify task names before benchmark sweeps.
3. Complete automated per-episode configuration and intervention logging. Manual restart/skip controls still require careful records.
4. Reconcile historical paper tables and video-derived ETS estimates with archived episode artifacts.
5. Validate geometry/action normalization, fallback feasibility, and solver performance. CPU packaging/regression tests are not safety guarantees.
6. Choose an original-contribution license before public publication. Exact checkpoint hashes remain unrecorded.
7. The policy uses request-local config swapping for the existing serial WebSocket server. Concurrent use of one policy instance from multiple threads is not supported.
