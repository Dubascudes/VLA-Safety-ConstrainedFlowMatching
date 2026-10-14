# Source provenance

The author confirmed that paper experiments ran in the HiPerGator VLA workspace, primarily in LIBERO and OpenPI. After comparing the current remote and local copies, the author selected the current local OpenPI implementation for the reference release and requested the live viewer as a release component.

The current local implementation contains the trajectory-level SLSQP/CBF method described by the manuscript. The current remote model file contains a different trajectory-reshaping variant. We preserve this distinction; current files alone do not identify the exact experiment-time snapshot.

## Pinned dependencies

See `provenance/source-manifest.json` for full revisions and SHA-256 hashes. OpenPI is based on commit `981483dca0fd9acba698fea00aa6e52d56a66c58`; SafeLIBERO comes from VLSA-AEGIS commit `9a85f173becac69b21d050ea1b4edf66c93100b9`.

The OpenPI patch captures five modified tracked files. The overlay captures the viewer, geometry helper, and two dependency files. SafeLIBERO comparison covered 981 non-cache files, including assets; seven differed and are represented by the environment patch. This includes two obstacle XML files, benchmark initialization, obstacle objects, region sampling, environment utilities, and the README.

The source-hash manifest covers all 16 captured changed/overlay files. Unmodified upstream content is pinned by Git revision. Materialization verifies these hashes and compiles captured Python files; it does not establish experiment reproducibility.

No model weights, paper result tables, estimated metrics, or experiment recordings are invented or bundled. Older OpenVLA/LIBERO experiments and optional ablation/figure scripts remain outside this first extraction pending relevance review.
