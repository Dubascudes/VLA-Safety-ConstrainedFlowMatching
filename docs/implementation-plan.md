# Initial reference release

## Agreed scope
Use the current local OpenPI implementation. Include the SafeLIBERO live experiment viewer as a supported release component. Preserve both research workspaces. Do not publish remotely yet.

## Architecture
Keep pinned upstream OpenPI and SafeLIBERO outside the versioned release, with auditable patches and an overlay containing the viewer and its geometry helper. Materialize them into an ignored runtime directory. Preserve algorithm behavior during extraction; do not conflate packaging tests with reproducing paper results.

## Implementation sequence
1. Capture upstream revisions, source hashes, OpenPI modifications, and SafeLIBERO differences. Retain upstream license notices and exclude outputs, environments, model weights, and unrelated experiments.
2. Add a materializer that refuses existing destinations, supports local upstream repositories for offline verification, checks revisions, applies patches, and copies the overlay.
3. Verify materialization against the captured source hashes and compile the extracted Python sources without importing GPU dependencies.
4. Document the server/viewer entry points, separate Python environments, safety modes, current limitations, and paper provenance. Explicitly distinguish the archived implementation from tested end-to-end inference.

## Acceptance for this first increment
- Offline materialization succeeds against the pinned local upstream repositories.
- All captured research source hashes match the materialized files.
- Extracted Python source compiles.
- Existing destinations are refused without modification.
- No weights, datasets, videos, virtual environments, or credentials enter the repository.

## Follow-up release gates
Verify a fresh dependency installation and model-backed viewer run; repair and test dynamic safety-parameter handling; make trace paths portable; record effective per-episode configurations and manual interventions; reconcile paper tables with saved artifacts. Select a license for original contributions before publication. These are not claimed complete by the initial extraction.
