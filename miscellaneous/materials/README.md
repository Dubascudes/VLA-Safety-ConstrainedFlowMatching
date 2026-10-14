# Model and benchmark materials

- Policy configuration: OpenPI `pi05_libero`.
- Checkpoint URI: `gs://openpi-assets/checkpoints/pi05_libero_pytorch`; exact weight checksum remains unrecorded.
- Benchmark: SafeLIBERO from pinned VLSA-AEGIS, with seven captured file modifications including obstacle XML.
- Simulator obstacle geometry supplies the safety state; no perception-grounded obstacle estimator is claimed.

Full source revisions and hashes are in `../software_information/provenance/`. Weights and simulator assets are not republished in this repository.
