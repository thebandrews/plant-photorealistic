"""Run one Blender command through the selected render backend (local or RunPod GPUs).

  python scripts/blender_run.py [--render-backend local|runpod] [--blender PATH] [--cwd DIR]
         [--output PATH ...] [--input PATH ...] [--log FILE] -- -b SCENE.blend --python SCRIPT.py -- ARGS

Use it for candidate re-renders (beauty, object IDs, depth, transparent landscape
layer). Snapshot/export scripts are cheap and may call Blender directly. The
backend comes from --render-backend, then PLANT_RENDER_BACKEND, then local.
Outputs must be fresh paths; remote results never overwrite local files.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import blender_backend


def find_blender(value=None, required=True):
    candidate = value or os.environ.get('BLENDER') or shutil.which('blender')
    if candidate and Path(candidate).name.lower() == 'blender-launcher.exe':
        candidate = str(Path(candidate).with_name('blender.exe'))
    if candidate and Path(candidate).is_file():
        return str(Path(candidate).resolve())
    if required:
        raise SystemExit('Blender not found: pass --blender or set BLENDER')
    return 'blender'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    blender_backend.add_argument(p)
    p.add_argument('--blender', help='Blender executable (default: $BLENDER or PATH)')
    p.add_argument('--cwd', default='.', help='Working directory for the command')
    p.add_argument('--output', action='append', default=[], help='Output file/folder (fresh path)')
    p.add_argument('--input', action='append', default=[], help='Extra input file/folder Blender reads')
    p.add_argument('--log', help='Write combined stdout/stderr here instead of the console')
    p.add_argument('blender_args', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    args = a.blender_args[1:] if a.blender_args[:1] == ['--'] else a.blender_args
    if not args:
        p.error('pass Blender arguments after --')
    backend = blender_backend.resolve_backend(a.render_backend)
    cwd = Path(a.cwd).resolve()
    for out in a.output:
        if (cwd / out).exists():
            p.error('output already exists; use a fresh path: ' + str(cwd / out))
    exe = find_blender(a.blender, required=backend == 'local')
    stream = open(a.log, 'ab') if a.log else None
    try:
        result = blender_backend.run([exe] + args, cwd=cwd, outputs=a.output, inputs=a.input, backend=backend,
                                     name='photoreal', stdout=stream, stderr=subprocess.STDOUT if stream else None)
    finally:
        if stream:
            stream.close()
    print('render backend: %s, exit %d' % (backend, result.returncode), file=sys.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
