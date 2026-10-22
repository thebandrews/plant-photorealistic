"""Render backend wrapper and plant-library write-back; no Blender, network or client data."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import blender_backend
import blender_run
import library_writeback

# Shared byte-for-byte with pacifica-planting-renders; update both repositories together.
SHARED_SHA256 = '78bfd724f79c07c62d1757b33357f42d52808e47295c21da157484731dd0fb35'

RECORDER = ('import json,os,sys;from pathlib import Path;'
            'Path(os.environ["RECORD_TO"]).write_text(json.dumps({"argv":sys.argv[1:],"cwd":os.getcwd()}))')


class RenderBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.recorder = self.root / 'fake_render_remote.py'
        self.recorder.write_text(RECORDER, encoding='utf-8')
        self.record = self.root / 'record.json'
        self.env = {'RENDER_REMOTE': str(self.recorder), 'RENDER_REMOTE_PYTHON': sys.executable,
                    'RECORD_TO': str(self.record), 'PLANT_RENDER_BACKEND': 'runpod'}

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_backend_matches_plant_render(self):
        data = Path(blender_backend.__file__).read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(data).hexdigest(), SHARED_SHA256)

    def test_runpod_candidate_render_goes_through_render_remote(self):
        with patch.dict(os.environ, self.env):
            code = blender_run.main(['--cwd', str(self.root), '--output', 'candidate', '--',
                                     '-b', 'candidate.blend', '--python', 'render.py', '--', 'FRONT'])
        self.assertEqual(code, 0)
        seen = json.loads(self.record.read_text())
        argv = seen['argv']
        self.assertEqual(argv[:2], ['exec', '--cwd'])
        self.assertEqual(argv[argv.index('--output') + 1], str((self.root / 'candidate').resolve()))
        self.assertEqual(argv[argv.index('--', 2) + 1:], ['-b', 'candidate.blend', '--python', 'render.py', '--', 'FRONT'])

    def test_existing_output_is_refused(self):
        (self.root / 'candidate').mkdir()
        with patch.dict(os.environ, self.env), self.assertRaises(SystemExit):
            blender_run.main(['--cwd', str(self.root), '--output', 'candidate', '--', '-b', 'x.blend'])
        self.assertFalse(self.record.exists())

    def test_local_backend_requires_blender(self):
        env = {**self.env, 'PLANT_RENDER_BACKEND': 'local', 'BLENDER': str(self.root / 'missing.exe'), 'PATH': ''}
        with patch.dict(os.environ, env), self.assertRaises(SystemExit):
            blender_run.main(['--', '-b', 'x.blend'])


class LibraryWritebackTests(unittest.TestCase):
    def test_without_library_nothing_is_saved(self):
        self.assertEqual(library_writeback.main(['spec.json'], environ={}), 0)

    def test_delegates_to_plant_render_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill = Path(tmp, 'pacifica-planting-renders')
            (skill / 'scripts').mkdir(parents=True)
            record = Path(tmp, 'called.json')
            (skill / 'scripts' / 'plant_library.py').write_text(
                'import json,sys;from pathlib import Path;Path(%r).write_text(json.dumps(sys.argv[1:]))' % str(record),
                encoding='utf-8')
            env = {'PACIFICA_ASSETS_DIR': tmp, 'PLANT_RENDER_SKILL': str(skill)}
            self.assertEqual(library_writeback.main([str(Path(tmp, 'spec.json'))], environ=env), 0)
            self.assertEqual(json.loads(record.read_text()), ['cache', str(Path(tmp, 'spec.json').resolve())])

    def test_missing_plant_render_skill_fails(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(library_writeback, 'plant_library_script', return_value=None):
            self.assertEqual(library_writeback.main(['spec.json'], environ={'PACIFICA_ASSETS_DIR': tmp}), 1)


if __name__ == '__main__':
    unittest.main()
