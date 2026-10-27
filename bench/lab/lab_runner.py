"""Photoreal lab runner: drains the experiment queue, running agents in parallel (PhotorealLab-Runner task).

The task runs as the VM user (register-lab-runner.ps1), so Codex and Claude Code use that user's sign-ins.
Each queued experiment gets its own git worktree (branch exp/<id>), experiment folder and GPU budget:

  C:\\Workstation\\lab\\queue\\<id>.json   {"id", "layer", "agent": "codex"|"claude", "hypothesis", "method",
                                         "brief", "budget_gpu_seconds", "timeout_minutes", "bench" (optional)}
  -> running\\<id>.json -> done\\<id>.json (agent result, bench violations, score)

After the agent finishes, its code changes are committed to exp/<id>, and the locked bench copy (repo\\bench,
never the worktree's) gates, judges and scores it, then rebuilds the board. An experiment that changed bench/
is not scored.

  python lab_runner.py run          # drain the queue (the task action)
  python lab_runner.py status       # queue, running, done, GPU seconds used
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

LAB = Path(os.environ.get('PHOTOREAL_LAB', r'C:\Workstation\lab'))
REPO = LAB / 'repo'
BENCH = os.environ.get('PHOTOREAL_BENCH', r'C:\Workstation\artifacts\photoreal-bench\reddy-20261025')
EXPERIMENTS = LAB / 'experiments'
VENV = LAB / '.venv'
PIPELINE_BIN = Path(r'C:\Workstation\pipeline\bin')
CLAUDE = Path(os.path.expanduser(r'~\.local\bin\claude.exe'))
PARALLEL = int(os.environ.get('PHOTOREAL_LAB_PARALLEL', 3))
LAB_GPU_SECONDS = float(os.environ.get('PHOTOREAL_LAB_GPU_SECONDS', 30000))
CLAUDE_TOOLS = 'Read,Write,Edit,Glob,Grep,Bash(python:*),Bash(git:*),Bash(render-remote:*)'
score_lock = threading.Lock()


def now():
    return dt.datetime.now().isoformat(timespec='seconds')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def git(*args, cwd=REPO, check=True):
    r = subprocess.run(['git', *args], cwd=str(cwd), capture_output=True, text=True, encoding='utf-8')
    if check and r.returncode:
        raise RuntimeError(f'git {" ".join(args)}: {r.stderr.strip()[:500]}')
    return r.stdout.strip()


def lab_gpu_used():
    total = 0.0
    for f in LAB.glob('experiments*/*/gpu-ledger.json'):
        try:
            total += sum(e.get('gpu_seconds', 0) for e in json.loads(f.read_text())['entries'])
        except (OSError, ValueError, KeyError):
            pass
    return round(total, 1)


def bench_for(spec):
    """The spec's bench ("bench": a folder next to the default bench), else the default bench."""
    return str(Path(BENCH).parent / spec['bench']) if spec.get('bench') else BENCH


def experiments_for(spec):
    """The default bench's experiments live in experiments, another bench's in experiments-<bench>."""
    return LAB / f'experiments-{spec["bench"]}' if spec.get('bench') else EXPERIMENTS


def agent_env(spec, exp_dir):
    env = dict(os.environ)
    env['Path'] = os.pathsep.join([str(VENV / 'Scripts'), env.get('Path', ''),
                                   os.path.expanduser(r'~\.local\bin'), os.path.expandvars(r'%APPDATA%\npm')])
    env.update(PHOTOREAL_BENCH=bench_for(spec), PHOTOREAL_EXPERIMENTS=str(experiments_for(spec)),
               PHOTOREAL_EXPERIMENT_ID=spec['id'],
               PLANT_GPU_BUDGET_SECONDS=str(spec.get('budget_gpu_seconds', 1800)),
               PLANT_GPU_BUDGET_FILE=str(exp_dir / 'gpu-ledger.json'),
               PHOTOREAL_LAB_GPU_SECONDS=str(LAB_GPU_SECONDS), PYTHONIOENCODING='utf-8')
    env.pop('PLANT_RUN_GPU_LEDGER', None)
    return env


def prompt_for(spec, wt, exp_dir):
    return f"""You are running photoreal lab experiment {spec['id']} (layer {spec['layer']}).

First read {wt}\\bench\\EXPERIMENT.md and {wt}\\bench\\README.md and follow them exactly. Hard rules, again:
no plant may move, disappear, appear, or change species or size; pixels outside the edit region must equal the
cleaned photo; bench\\ is locked (never edit it); stay inside the GPU budget; never print keys or SAS URLs.

- Your code: this worktree {wt} (branch exp/{spec['id']}). Put experiment code under photoreal\\ or scripts\\.
  Don't run git commit or change git config: the lab runner commits your changes when you finish.
- Experiment folder: {exp_dir}. Write out\\p49.png, out\\p51.png (3840x2560 RGB), run.json and NOTES.md there.
- Bench (read only): {bench_for(spec)}
- Python: `python` (lab venv: numpy, pillow, opencv, scipy, torch CPU, transformers).
- Diffusion on RunPod: `from lab import comfy` after adding {wt}\\bench to sys.path. comfy.run(workflow, images)
  returns PIL images; comfy.qwen_edit(...) builds a Qwen-Image-Edit workflow; comfy.MODELS lists the models.
  Requests over 10 MB are refused, so send tiles of about 1-2 MP. Every job counts against your budget.
- Blender on RunPod: render-remote (see C:\\Workstation\\CLAUDE.md). Budget: {spec.get('budget_gpu_seconds', 1800)}
  GPU seconds for this experiment, shared by render-remote and diffusion jobs.

Hypothesis: {spec['hypothesis']}
Method: {spec['method']}

{spec.get('brief', '').strip()}

When done, run `python {wt}\\bench\\bench.py gate {spec['id']}` and end your final message with
LAB_STATUS: complete (or failed) and a one-line summary."""


def run_codex(spec, wt, exp_dir, logs, env, timeout):
    sys.path.insert(0, str(PIPELINE_BIN))
    import codex_driver
    drv = codex_driver.ExecDriver(wt, logs, env=env, turn_timeout=timeout)
    r = drv.start(prompt_for(spec, wt, exp_dir))
    return {'final_message': r.final_message[-4000:], 'error': r.error, 'timed_out': r.timed_out,
            'usage_limited': r.usage_limited, 'commands': r.commands, 'session_id': r.session_id}


def run_claude(spec, wt, exp_dir, logs, env, timeout):
    cmd = [str(CLAUDE), '-p', prompt_for(spec, wt, exp_dir), '--output-format', 'json',
           '--permission-mode', 'acceptEdits', '--allowedTools', CLAUDE_TOOLS,
           '--add-dir', str(exp_dir), '--add-dir', bench_for(spec)]
    if spec.get('model'):
        cmd += ['--model', spec['model']]
    out_path = logs / 'claude-output.json'
    with open(out_path, 'w', encoding='utf-8') as out, open(logs / 'claude-stderr.txt', 'w', encoding='utf-8') as err:
        p = subprocess.Popen(cmd, cwd=str(wt), env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        try:
            p.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
            timed_out = True
    try:
        res = json.loads(out_path.read_text(encoding='utf-8'))
    except ValueError:
        res = {}
    return {'final_message': str(res.get('result', ''))[-4000:], 'error': None if p.returncode == 0 else
            f'claude exited {p.returncode}', 'timed_out': timed_out, 'cost_usd_equiv': res.get('total_cost_usd'),
            'turns': res.get('num_turns'), 'session_id': res.get('session_id')}


def bench_py(*args, env):
    py = VENV / 'Scripts' / 'python.exe'
    return subprocess.run([str(py), str(REPO / 'bench' / 'bench.py'), *args], cwd=str(REPO / 'bench'), env=env,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')


def run_one(queue_file):
    spec = json.loads(queue_file.read_text(encoding='utf-8'))
    eid = spec['id']
    running = LAB / 'running' / f'{eid}.json'
    running.parent.mkdir(parents=True, exist_ok=True)
    os.replace(queue_file, running)
    result = {'id': eid, 'spec': spec, 'started': now()}
    logs = LAB / 'logs' / eid
    logs.mkdir(parents=True, exist_ok=True)
    if spec.get('rescore'):
        return rescore(spec, running, logs)
    try:
        if lab_gpu_used() >= LAB_GPU_SECONDS:
            raise RuntimeError(f'lab GPU budget spent ({lab_gpu_used():.0f}s of {LAB_GPU_SECONDS:.0f}s)')
        base = git('rev-parse', 'HEAD')
        wt = LAB / 'worktrees' / eid
        if not wt.exists():
            git('worktree', 'add', '-b', f'exp/{eid}', str(wt), base)
        exp_dir = experiments_for(spec) / eid
        env = agent_env(spec, exp_dir)
        if not (exp_dir / 'experiment.json').exists():
            r = bench_py('new', eid, '--layer', spec['layer'], '--hypothesis', spec['hypothesis'],
                         '--method', spec['method'], '--branch', f'exp/{eid}',
                         '--budget-gpu-s', str(spec.get('budget_gpu_seconds', 1800)), env=env)
            if r.returncode:
                raise RuntimeError(f'bench new: {r.stderr[-500:]}')
        timeout = int(spec.get('timeout_minutes', 180)) * 60
        t0 = time.time()
        runner = run_claude if spec.get('agent') == 'claude' else run_codex
        result['agent'] = runner(spec, wt, exp_dir, logs, env, timeout)
        result['agent_minutes'] = round((time.time() - t0) / 60, 1)

        git('add', '-A', cwd=wt)
        if git('status', '--porcelain', cwd=wt):
            git('-c', 'user.name=Photoreal Lab', '-c', 'user.email=lab@localhost', 'commit', '-q', '-m',
                f'Experiment {eid}: {spec["method"][:60]}', cwd=wt)
        result['commit'] = git('rev-parse', 'HEAD', cwd=wt)
        touched = git('diff', '--name-only', base, 'HEAD', '--', 'bench', cwd=wt)
        result['bench_modified'] = touched.splitlines() if touched else []
        outs = [exp_dir / 'out' / f'{v}.png' for v in ('p49', 'p51')]
        result['outputs'] = all(p.exists() for p in outs)
        if result['bench_modified']:
            result['score'] = {'skipped': 'bench/ was modified'}
        elif not result['outputs']:
            result['score'] = {'skipped': 'outputs missing'}
        else:
            with score_lock:
                r = bench_py('score', eid, env=env)
                (logs / 'score.log').write_text(r.stdout + r.stderr, encoding='utf-8')
                bench_py('board', env=env)
            sp = exp_dir / 'scores.json'
            result['score'] = json.loads(sp.read_text()) if sp.exists() else {'error': r.stderr[-1000:]}
    except Exception as e:      # noqa: BLE001 - one experiment's failure must not stop the queue
        result['error'] = f'{type(e).__name__}: {e}'
    result['finished'] = now()
    result['lab_gpu_seconds'] = lab_gpu_used()
    write_json(LAB / 'done' / f'{eid}.json', result)
    running.unlink(missing_ok=True)
    return result


def rescore(spec, running, logs):
    """{"id": ..., "rescore": true}: score an existing experiment again with the current bench, no agent."""
    eid = spec['id']
    done = LAB / 'done' / f'{eid}.json'
    result = json.loads(done.read_text(encoding='utf-8')) if done.exists() else {'id': eid}
    exp_dir = experiments_for(spec) / eid
    env = agent_env(spec, exp_dir)
    with score_lock:
        r = bench_py('score', eid, env=env)
        (logs / 'rescore.log').write_text(r.stdout + r.stderr, encoding='utf-8')
        bench_py('board', env=env)
    sp = exp_dir / 'scores.json'
    result['score'] = json.loads(sp.read_text()) if sp.exists() else {'error': r.stderr[-1000:]}
    result['rescored'] = now()
    write_json(done, result)
    running.unlink(missing_ok=True)
    return result


def cmd_run(args):
    for d in ('queue', 'running', 'done', 'logs', 'worktrees'):
        (LAB / d).mkdir(parents=True, exist_ok=True)
    for stale in (LAB / 'running').glob('*.json'):       # a previous runner died: requeue
        os.replace(stale, LAB / 'queue' / stale.name)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        active = {}
        while True:
            for f in sorted((LAB / 'queue').glob('*.json')):
                if f.name not in active and len(active) < args.parallel:
                    active[f.name] = pool.submit(run_one, f)
            for name, fut in list(active.items()):
                if fut.done():
                    r = fut.result()
                    score = r.get('score') or {}
                    print(f"{now()} {r['id']}: {r.get('error') or score.get('qualified', score)}", flush=True)
                    del active[name]
            if not active and not any((LAB / 'queue').glob('*.json')):
                break
            time.sleep(10)


def cmd_status(args):
    for d in ('queue', 'running', 'done'):
        print(d + ':', ', '.join(sorted(p.stem for p in (LAB / d).glob('*.json'))) or '-')
    print(f'GPU seconds used: {lab_gpu_used():.0f} of {LAB_GPU_SECONDS:.0f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('--parallel', type=int, default=PARALLEL)
    sub.add_parser('status')
    args = ap.parse_args(argv)
    {'run': cmd_run, 'status': cmd_status}[args.cmd](args)


if __name__ == '__main__':
    main()
