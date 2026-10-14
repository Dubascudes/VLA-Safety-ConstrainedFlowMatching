# Known issues and release gates

This extraction intentionally preserves the chosen source rather than silently repairing research behavior.

1. **Dynamic flow parameters:** the viewer sends `_safety_margin` and `_safety_guidance_scale`, but `Policy.infer` has the update call commented out. The flow solver has no current guidance-scale configuration field. Decide meaningful control semantics and add regression tests before advertising functional sliders.
2. **Server margin:** server default 0.0 overrides model default 0.01. Set `--safety-margin 0.01` explicitly for manuscript-aligned geometry parameters; this alone does not establish the paper configuration.
3. **Trace paths:** the policy saves traces to a hard-coded research desktop path. Make this opt-in and portable before running the published package on another machine.
4. **Task indices:** the evaluator arguments and dropdown-update paths contain mixed zero-based and one-based conventions. Verify task identity against benchmark names before automated sweeps.
5. **Manual interventions:** restart and skip controls exist. A reproducible benchmark runner must save effective configuration, task identity, episode initialization, seed, termination status, and interventions. Do not silently omit failures or replace episodes.
6. **Metrics:** historical analysis estimated ETS from video duration. Do not present estimates as direct timestep logs. No paper results have been recomputed in this extraction.
7. **Dependency compatibility:** policy and simulator use separate environments. Their dependency snapshots are retained, not freshly validated. SciPy is required by the SLSQP solver and must be checked explicitly.
8. **Safety claims:** SLSQP failure triggers a greedy fallback. Packaging checks do not establish constraint satisfaction, solver latency, or actual collision avoidance. Test fallback, infeasibility, action normalization, and coordinate scaling separately.
9. **Licensing and publication:** select the original-contribution license, review third-party attribution, and scan final content before pushing. No remote publication has occurred.

## Next acceptance checks

- Install fresh policy and simulator environments.
- Repair and test parameter propagation and trace paths without changing the archived snapshot.
- Launch the actual model-backed viewer and verify a matching GUI window plus a bounded rollout.
- Add a headless, configuration-logged benchmark entry point.
- Reconcile the reported tables against archived episode records, separately from new runs.
