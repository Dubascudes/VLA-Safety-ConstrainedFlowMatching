# Verification scope

Packaging tests reconstruct from pinned local Git mirrors, apply both research and release patches, verify source hashes, compile changed Python files, and refuse existing destinations.

CPU regression tests import the actual reconstructed Policy, SafetyConfig, and SafetyGuidance. A tiny test model replaces neural inference to observe parameters and produce deterministic trace fixtures. The correction-strength test executes the actual numerical guidance implementation, including its solver/fallback path. Server tests replace model loading and socket serving, not parameter configuration. A sampler regression executes the actual Euler loops and numerical safety code with neural forward operations replaced, comparing traced/untraced outputs for flat and posthoc schedules at strengths 0, 0.5, and 1. Both defects found in independent review were reproduced as failing tests and repaired.

These are regression tests, not learned-policy inference or simulation benchmarks. Use an OpenPI-capable environment and set `VLA_RUNTIME` if testing a non-default reconstruction. The tests explicitly skip when dependencies/runtime are absent; a skipped test is not a passing runtime verification.
