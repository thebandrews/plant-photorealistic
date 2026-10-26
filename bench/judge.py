"""Vision judge: realism of each crop, and species consistency between the render and the output.

The judge is a Claude Code session (`claude -p`) that may only Read the images it is given. Crops are renamed
at random and mixed with anchors whose truth is known (real-photo crops; crops of the current baseline, which
is known synthetic), so every batch also measures the judge. Each set is judged twice in different orders.
Set PHOTOREAL_JUDGE_CLI to a different command (for example a Codex wrapper) for a second model family.
"""
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from PIL import Image

GIVEAWAYS = ['plant_repetition', 'plant_anatomy', 'plastic_material', 'ground_texture', 'shadows_lighting',
             'edges_halo', 'scale', 'camera_signature', 'color_grade', 'compositing', 'other']

REALISM_PROMPT = """You are a forensic photo examiner. Every image file in this folder is either an unedited
camera photograph or contains computer-generated content (3D rendering, AI image generation or compositing).
The share of real photographs in this set is unknown; judge each image on its own evidence.

Open every file with the Read tool and look closely at:
- plants: identical repeated flowers or leaves, flat or faceted leaves, too-regular spacing, missing fine
  structure (veins, spines, dead leaves, damage, stems), wrong habit for the plant;
- materials: plastic or waxy sheen, uniform colour, missing texture, texture at the wrong scale;
- soil, mulch and paving: flat or smeared texture, repetition, no clods, debris or wear;
- light: shadow hardness and direction that disagree with the scene, missing contact shadows, no bounce light;
- edges: halos, cutout fringes, objects that float or do not sit on the ground;
- camera: sensor noise, depth of field, lens softness and fringing, compression, consistent with one camera.

Files: {files}

Reply with JSON only, no prose, as a list with one entry per file:
[{{"file": "<name>", "verdict": "photo" or "synthetic", "p_photo": <0-100>,
   "giveaways": [{{"type": one of {types}, "where": "<region>", "note": "<short>"}}]}}]
Give an empty giveaways list when you find none."""

SPECIES_PROMPT = """Each pair below shows the same location in two images of a planted garden: A is a 3D render
used as the design, B is a finished image. The design may have been made more realistic in B, which can change
leaf detail, texture and lighting. Decide for each pair whether B still shows the same plant at that spot:
the same species or cultivar (leaf shape and arrangement, growth habit, flower colour and form) and roughly the
same size and position. A different plant, a missing plant, an extra plant or a changed species means "same": false.
Open both files of every pair with the Read tool.

Pairs:
{pairs}

Reply with JSON only: [{{"pair": <n>, "same": true or false, "confidence": <0-100>, "note": "<short>"}}]"""


def _cli():
    raw = os.environ.get('PHOTOREAL_JUDGE_CLI')
    if raw:
        return raw.split()
    cmd = ['claude', '-p', '--output-format', 'json', '--allowedTools', 'Read']
    model = os.environ.get('PHOTOREAL_JUDGE_MODEL')
    if model:
        cmd += ['--model', model]
    return cmd


def _parse_list(text):
    m = re.search(r'\[.*\]', text, re.S)
    if not m:
        raise ValueError(f'judge reply had no JSON list: {text[:300]!r}')
    return json.loads(m.group(0))


def ask(prompt, folder, timeout=1800):
    """Run one judge turn in `folder`; returns the reply text."""
    cmd = _cli() + ['--add-dir', str(folder)]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding='utf-8',
                          cwd=folder, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f'judge exited {proc.returncode}: {proc.stderr[-500:]}')
    out = proc.stdout.strip()
    try:
        env = json.loads(out)
        return env.get('result', out) if isinstance(env, dict) else out
    except ValueError:
        return out


def realism(items, batch_size=10, passes=2, seed=0):
    """items: [{'key', 'image' (ndarray), 'truth': None|'photo'|'synthetic'}] -> {key: {p_photo, verdict, giveaways}}"""
    rng = random.Random(seed)
    votes = {}
    for pas in range(passes):
        order = items[:]
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            folder = Path(tempfile.mkdtemp(prefix='judge-'))
            try:
                names = {}
                for it in batch:
                    name = f'img-{uuid.uuid4().hex[:8]}.png'
                    Image.fromarray(it['image']).save(folder / name)
                    names[name] = it['key']
                prompt = REALISM_PROMPT.format(files=', '.join(names), types=GIVEAWAYS)
                for attempt in range(2):
                    try:
                        reply = _parse_list(ask(prompt, folder))
                        break
                    except (ValueError, RuntimeError):
                        if attempt:
                            raise
                for r in reply:
                    key = names.get(r.get('file'))
                    if key is not None:
                        votes.setdefault(key, []).append(r)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
    out = {}
    for it in items:
        rs = votes.get(it['key'], [])
        if not rs:
            out[it['key']] = {'p_photo': None, 'verdict': None, 'giveaways': [], 'votes': 0}
            continue
        p = sum(float(r.get('p_photo') or 0) for r in rs) / len(rs)
        photo_votes = sum(r.get('verdict') == 'photo' for r in rs)
        out[it['key']] = {'p_photo': round(p, 1), 'verdict': 'photo' if photo_votes * 2 > len(rs) else 'synthetic',
                          'votes': len(rs), 'agree': photo_votes in (0, len(rs)),
                          'giveaways': [g for r in rs for g in (r.get('giveaways') or [])]}
    return out


def species(pairs, batch_size=8):
    """pairs: [{'key', 'a' (render crop), 'b' (output crop)}] -> {key: {same, confidence, note}}"""
    out = {}
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        folder = Path(tempfile.mkdtemp(prefix='species-'))
        try:
            lines = []
            for n, pr in enumerate(batch, 1):
                Image.fromarray(pr['a']).save(folder / f'pair{n}-A.png')
                Image.fromarray(pr['b']).save(folder / f'pair{n}-B.png')
                lines.append(f'{n}. pair{n}-A.png and pair{n}-B.png')
            reply = _parse_list(ask(SPECIES_PROMPT.format(pairs='\n'.join(lines)), folder))
            for r in reply:
                n = int(r.get('pair', 0))
                if 1 <= n <= len(batch):
                    out[batch[n - 1]['key']] = {'same': bool(r.get('same')), 'confidence': r.get('confidence'),
                                                'note': r.get('note', '')}
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return out
