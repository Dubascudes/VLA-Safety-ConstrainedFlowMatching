"""Offline integration tests; optional upstream mirrors are supplied by env vars."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MaterializeTests(unittest.TestCase):
    def test_help(self):
        result = subprocess.run([sys.executable, str(ROOT/'tools/materialize.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--openpi-source', result.stdout)

    def test_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'existing'
            dest.mkdir()
            (dest/'sentinel').write_text('keep')
            result = subprocess.run([sys.executable, str(ROOT/'tools/materialize.py'), '--dest', str(dest)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('already exists', result.stderr)
            self.assertEqual((dest/'sentinel').read_text(), 'keep')

    @unittest.skipUnless(os.environ.get('VLA_OPENPI_SOURCE') and os.environ.get('VLA_SAFELIBERO_SOURCE'), 'set upstream mirrors for offline integration')
    def test_offline_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'runtime'
            result = subprocess.run([sys.executable, str(ROOT/'tools/materialize.py'), '--dest', str(dest), '--openpi-source', os.environ['VLA_OPENPI_SOURCE'], '--safelibero-source', os.environ['VLA_SAFELIBERO_SOURCE']], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('Verified 16 source hashes', result.stdout)
            self.assertTrue((dest/'openpi/third_party/safelibero/setup.py').exists())

if __name__ == '__main__':
    unittest.main()
