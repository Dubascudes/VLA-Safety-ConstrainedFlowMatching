# Source provenance

The paper experiments ran on HiPerGator. After comparing copies, the author selected the current local OpenPI implementation for release. It contains the manuscript's trajectory-level SLSQP/CBF method, whereas the current remote model file contains a different reshaping variant. The current files are not established as an exact experiment-time snapshot.

## Preserved baseline

Commit `2bb053e` preserves the initial extraction before layout changes and behavior repairs. `provenance/source-manifest.json` retains its 16 captured source hashes. OpenPI is pinned at `981483dca0fd9acba698fea00aa6e52d56a66c58`; VLSA-AEGIS is pinned at `9a85f173becac69b21d050ea1b4edf66c93100b9`.

The research OpenPI patch captures five tracked files. The original overlay comprised the viewer, geometry helper, and two dependency files. SafeLIBERO comparison covered 981 non-cache files including assets; seven differences were captured, including two obstacle XML files and four environment Python files plus README.

## Repaired release

`software_module/patches/release-fixes.patch` is applied after the research OpenPI patch. The viewer is now maintained under `experiments/processing/safelibero/`. `provenance/release-manifest.json` records the resulting 16 file hashes used by materialization. Changed defaults and parameter/trace semantics are documented in the root README and protocols.

The directory organization follows the local copy of Griffin Chure's reproducible_research template. Its CC0 notice is retained for the template only, not imposed on this project's code. Neither sample datasets nor unrelated biological procedures were imported.

No model weights, claimed paper-result values, or experiment recordings have been invented or bundled. Original local and remote research workspaces remain unchanged.
