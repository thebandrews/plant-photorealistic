"""Save an accepted refined botanical asset as a new version in the shared plant library.

  python scripts/library_writeback.py PACKAGE_SPEC.json

The shared library belongs to the pacifica-planting-renders skill (configured
with PACIFICA_ASSETS_DIR). This delegates to that skill's own
`plant_library.py cache`, which hashes files, never overwrites a version and
updates the index. Without a configured library it reports that nothing was
saved and exits 0, so a run elsewhere is not blocked.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def plant_library_script(environ=None):
    env = os.environ if environ is None else environ
    candidates = []
    if env.get('PLANT_RENDER_SKILL'):
        candidates.append(Path(env['PLANT_RENDER_SKILL']) / 'scripts' / 'plant_library.py')
    # Installed side by side in the same skills directory.
    candidates.append(Path(__file__).resolve().parents[2] / 'pacifica-planting-renders' / 'scripts' / 'plant_library.py')
    for path in candidates:
        if path.is_file():
            return path
    return None


def main(argv=None, environ=None):
    env = os.environ if environ is None else environ
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spec', help='plant-render package specification for the refined asset version')
    a = p.parse_args(argv)
    if not env.get('PACIFICA_ASSETS_DIR'):
        print('No shared plant library configured (PACIFICA_ASSETS_DIR); refined asset kept in the run output only.')
        return 0
    script = plant_library_script(env)
    if script is None:
        print('pacifica-planting-renders is not installed next to this skill; set PLANT_RENDER_SKILL.', file=sys.stderr)
        return 1
    return subprocess.run([sys.executable, str(script), 'cache', str(Path(a.spec).resolve())]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
