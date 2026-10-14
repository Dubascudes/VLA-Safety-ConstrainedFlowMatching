# Reusable software

`patches/openpi.patch` and `patches/safelibero.patch` preserve the captured research modifications against pinned upstream revisions. `patches/release-fixes.patch` is applied afterward and contains reviewed release repairs; never silently fold these into the original patches.

`openpi_overlay/` contains the reusable geometry helper and captured simulator dependency files. The executable viewer lives in `../experiments/processing/safelibero/`.

The generated `runtime/` tree is disposable, not authoritative source. Changes made there must be exported into the release patch and its manifest before reconstruction. Baseline hashes are preserved independently from release hashes.
