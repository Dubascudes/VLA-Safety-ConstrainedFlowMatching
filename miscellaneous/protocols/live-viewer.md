# Live experiment workflow

1. Reconstruct the pinned runtime and install separate policy/simulator environments.
2. Choose a task suite, safety level, task identity, initial episode, seed, and safety mode; create a run record from `templates/experiment_README.md`.
3. Start the server with explicit safety margin and correction multiplier. Enable `--trace-dir` only if traces are needed; tracing performs additional neural passes and is not representative of normal inference latency.
4. Start the viewer with explicit task and episode arguments. Use strength 1.0 for unchanged solver corrections. Log every slider change, restart, skipped episode, and forced failure; the current GUI does not automatically record a complete intervention ledger.
5. Keep outputs in a unique run directory. Record task names as well as numeric IDs because dropdown indexing still needs audit.
6. Aggregate only after checking episode coverage. No paper-reproduction claim follows from a successful demo rollout.

Trace paths are configured on the server, not the viewer. Parameters sent by the viewer are validated and scoped to that inference request; omitted values use server defaults. The policy instance is intended for the existing serial WebSocket server, not concurrent calls from multiple threads.
