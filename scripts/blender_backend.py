"""Choose where Blender renders run: this machine (`local`) or RunPod GPUs (`runpod`).

Shared, byte-identical, by the plant-render and plant-photorealistic skills.

Selection order: an explicit value (CLI `--render-backend`), then the
PLANT_RENDER_BACKEND environment variable, then `local`. The `runpod` backend
calls the workstation's `render-remote exec` command, which mirrors the needed
files to a GPU worker, runs the same Blender version there and copies new
outputs back to the same paths. Inputs and outputs are byte-identical, so
recorded hashes stay valid. It never falls back to a local render silently.
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

BACKENDS = ('local', 'runpod')
ENVIRONMENT_KEY = 'PLANT_RENDER_BACKEND'
ALIASES = {'local': 'local', 'cpu': 'local', 'runpod': 'runpod', 'remote': 'runpod', 'gpu': 'runpod'}
BLENDER_NAMES = {'blender', 'blender.exe', 'blender-launcher.exe'}


class BackendError(ValueError):
    pass


def resolve_backend(value=None, environ=None):
    env = os.environ if environ is None else environ
    raw = value or env.get(ENVIRONMENT_KEY) or 'local'
    backend = ALIASES.get(str(raw).strip().lower())
    if backend is None:
        raise BackendError('Unknown render backend %r; choose local or runpod' % raw)
    return backend


def add_argument(parser):
    """Add the shared --render-backend option to an argparse parser."""
    parser.add_argument('--render-backend', choices=BACKENDS,
                        help='Where Blender renders run (default: $%s or local)' % ENVIRONMENT_KEY)


def is_blender(executable):
    return Path(str(executable)).name.lower() in BLENDER_NAMES


def render_remote_command(environ=None):
    """The command prefix that runs `render-remote` on this machine.

    RENDER_REMOTE may name render_remote.py (run with RENDER_REMOTE_PYTHON),
    an executable, or a full command line; otherwise `render-remote` on PATH.
    """
    env = os.environ if environ is None else environ
    configured = env.get('RENDER_REMOTE')
    if configured:
        parts = [p.strip('"') for p in shlex.split(configured, posix=False)]
    else:
        found = shutil.which('render-remote', path=env.get('PATH'))
        if not found:
            raise BackendError('The runpod render backend needs render-remote on PATH (or RENDER_REMOTE); '
                               'use --render-backend local on machines without it')
        parts = [found]
    if parts[0].lower().endswith('.py'):
        parts = [env.get('RENDER_REMOTE_PYTHON') or sys.executable] + parts
    elif parts[0].lower().endswith(('.cmd', '.bat')):
        parts = [env.get('COMSPEC') or 'cmd.exe', '/d', '/c'] + parts
    return parts


def remote_command(argv, cwd, outputs=(), inputs=(), name=None, environ=None):
    """Translate a local Blender argv into the equivalent `render-remote exec` call."""
    argv = [str(x) for x in argv]
    if not argv or not is_blender(argv[0]):
        raise BackendError('Only commands that start with the Blender executable can run on RunPod: %s'
                           % (argv[0] if argv else '<empty>'))
    command = render_remote_command(environ) + ['exec', '--cwd', str(Path(cwd).resolve())]
    for path in outputs:
        command += ['--output', str(path)]
    for path in inputs:
        command += ['--input', str(path)]
    if name:
        command += ['--name', str(name)]
    return command + ['--'] + argv[1:]


def run(argv, cwd=None, outputs=(), inputs=(), backend=None, name=None, environ=None, **kwargs):
    """Run a Blender command with the selected backend; returns CompletedProcess.

    `outputs`/`inputs` are paths relative to `cwd` (or absolute). Outputs must
    not already exist: remote results never overwrite local files.
    """
    cwd = Path(cwd or os.getcwd()).resolve()
    selected = resolve_backend(backend, environ)
    if selected == 'local':
        return subprocess.run([str(x) for x in argv], cwd=str(cwd), **kwargs)
    resolved = lambda paths: [str((cwd / p).resolve()) for p in paths]
    command = remote_command(argv, cwd, resolved(outputs), resolved(inputs), name, environ)
    return subprocess.run(command, cwd=str(cwd), **kwargs)
