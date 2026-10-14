#!/usr/bin/env python3
"""Reconstruct the captured research implementation from pinned upstream commits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=ROOT/'runtime')
    parser.add_argument('--openpi-source', help='Optional local upstream Git repository or mirror URL')
    parser.add_argument('--safelibero-source', help='Optional local VLSA Git repository or mirror URL')
    args = parser.parse_args()
    # lexists also rejects broken symlinks; never replace an existing workspace.
    if args.dest.exists() or args.dest.is_symlink():
        parser.error(f'destination already exists: {args.dest}')
    manifest = json.loads((ROOT/'miscellaneous/software_information/provenance/release-manifest.json').read_text())
    args.dest.mkdir(parents=True)
    try:
        for name, source in [('openpi', args.openpi_source), ('safelibero', args.safelibero_source)]:
            record = manifest[name]
            target = args.dest/name
            run('git', 'clone', '--no-hardlinks', '--no-checkout', '--', source or record['url'], target)
            run('git', '-C', target, 'checkout', '--detach', record['revision'])
            actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != record['revision']:
                raise RuntimeError(f'{name}: unexpected revision {actual}')
            patch = ROOT/'software_module/patches'/f'{name}.patch'
            run('git', '-C', target, 'apply', '--check', patch)
            run('git', '-C', target, 'apply', patch)
        release_patch = ROOT/'software_module/patches/release-fixes.patch'
        run('git', '-C', args.dest/'openpi', 'apply', '--check', release_patch)
        run('git', '-C', args.dest/'openpi', 'apply', release_patch)
        shutil.copytree(ROOT/'software_module/openpi_overlay', args.dest/'openpi', dirs_exist_ok=True)
        shutil.copy2(ROOT/'experiments/processing/safelibero/live_viewer.py',
                     args.dest/'openpi/examples/safelibero/eval_safelibero_visualized.py')
        link = args.dest/'openpi/third_party/safelibero'
        if link.exists() or link.is_symlink():
            raise RuntimeError(f'upstream unexpectedly contains {link}')
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to('../../safelibero/safelibero', target_is_directory=True)
        for relative, expected in manifest['source_sha256'].items():
            path = args.dest/relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'source hash mismatch: {relative}')
            if path.suffix == '.py':
                compile(path.read_bytes(), str(path), 'exec')
        count = len(manifest['source_sha256'])
        print(f'Verified {count} source hashes; release Python sources compile.')
        print(f'Reconstructed source: {args.dest.resolve()}')
        print('Dependencies, model weights, and GPU/viewer execution are NOT verified by this command.')
    except (OSError, subprocess.CalledProcessError, RuntimeError, SyntaxError) as exc:
        print(f'ERROR: {exc}\nPartial destination retained for inspection: {args.dest}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
