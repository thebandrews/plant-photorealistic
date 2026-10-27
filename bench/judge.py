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

SPECIES_PROMPT = """Each pair below shows the same spot in two images of a planted garden: A is a 3D render used as
the design, B is the finished image. For each pair you are told what is at the centre: a plant's species and, if it
flowers, the flowers it should show in bloom; or open ground. B may make plants look more realistic and more like
the named species (leaf detail, anatomy, texture, light) and may show the listed flowers on that plant. Neither is a
change. Open both files of every pair with the Read tool and answer two questions:

- same: does B still show the centre plant at the same spot, as the named species, at roughly the same size? false
  if it is missing, moved, or has become a different plant (another growth form, leaf type, or flower colour/form
  than listed). For open ground, same is true unless the ground became something else (paving, a plant).
- extra: does B contain, anywhere in the crop, a plant that A does not have, or flowers that are not carried by a
  plant that is in A? Flowers that grow from a plant A has (among its leaves, or on stems reaching out from it) are
  not extra, even where A shows that plant with no flowers or with flowers of another size or shape.

Pairs:
{pairs}

Reply with JSON only: [{{"pair": <n>, "same": true or false, "extra": true or false, "confidence": <0-100>,
  "note": "<short>"}}]"""


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


def species(pairs, batch_size=8, recheck=2, seed=0):
    """pairs: [{'key', 'a' (render crop), 'b' (output crop), 'species' (None for open ground), 'flowers'}]
    -> {key: {same, extra, confidence, note, votes}}. A pair judged changed or extra is judged `recheck` more
    times in shuffled batches, and same/extra are the majority of its votes."""
    out = _species_pass(pairs, batch_size)
    flagged = [p for p in pairs if p['key'] in out and (not out[p['key']]['same'] or out[p['key']]['extra'])]
    if not flagged or not recheck:
        return {k: dict(r, votes=1) for k, r in out.items()}
    rng = random.Random(seed)
    ballots = {p['key']: [out[p['key']]] for p in flagged}
    for _ in range(recheck):
        order = list(flagged)
        rng.shuffle(order)
        for k, r in _species_pass(order, batch_size).items():
            ballots[k].append(r)
    for k, rs in ballots.items():
        conf = [r['confidence'] for r in rs if isinstance(r.get('confidence'), (int, float))]
        out[k] = {'same': 2 * sum(r['same'] for r in rs) >= len(rs), 'extra': 2 * sum(r['extra'] for r in rs) > len(rs),
                  'confidence': round(sum(conf) / len(conf)) if conf else None,
                  'note': next((r['note'] for r in rs if r.get('note')), ''), 'votes': len(rs),
                  'extra_votes': sum(r['extra'] for r in rs), 'same_votes': sum(r['same'] for r in rs)}
    return {k: dict(r, votes=r.get('votes', 1)) for k, r in out.items()}


def _species_pass(pairs, batch_size):
    out = {}
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        folder = Path(tempfile.mkdtemp(prefix='species-'))
        try:
            lines = []
            for n, pr in enumerate(batch, 1):
                Image.fromarray(pr['a']).save(folder / f'pair{n}-A.png')
                Image.fromarray(pr['b']).save(folder / f'pair{n}-B.png')
                centre = (f"{pr['species']}" + (f"; in bloom: {pr['flowers']}" if pr.get('flowers') else '; no flowers expected')
                          if pr.get('species') else 'open ground, no plant')
                lines.append(f'{n}. pair{n}-A.png and pair{n}-B.png. Centre: {centre}')
            reply = _parse_list(ask(SPECIES_PROMPT.format(pairs='\n'.join(lines)), folder))
            for r in reply:
                n = int(r.get('pair', 0))
                if 1 <= n <= len(batch):
                    out[batch[n - 1]['key']] = {'same': bool(r.get('same')), 'extra': bool(r.get('extra')),
                                                'confidence': r.get('confidence'), 'note': r.get('note', '')}
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return out


PAIRWISE_PROMPT = """You are a forensic photo examiner. Each numbered pair below shows the same scene region in two
images. At most one of them is an unedited camera photograph; often both contain computer-generated content
(3D rendering, AI generation or compositing). For each pair, decide which image looks more like an unedited
camera photograph: plants (natural anatomy, variation, fine structure), materials, soil and paving texture,
light and shadows, edges, and camera signature. Answer "tie" only when the two look equally real or identical.
Open both files of every pair with the Read tool.

Pairs:
{pairs}

Reply with JSON only: [{{"pair": <n>, "more_real": 1 or 2 or "tie", "confidence": <0-100>, "note": "<short>"}}]"""


def pairwise(pairs, batch_size=6, passes=2, seed=0):
    """pairs: [{'key', 'x' (candidate crop), 'y' (reference crop)}] -> {key: {'win': 0..1, 'votes', 'notes'}}.

    win is the share of votes that found x more real (a tie counts half). The first pass puts x first or second
    at random; each later pass swaps every pair, so position bias cancels."""
    rng = random.Random(seed)
    votes, x_first = {}, {}
    for pas in range(passes):
        order = pairs[:]
        rng.shuffle(order)
        for pr in order:
            x_first[pr['key']] = rng.random() < 0.5 if pas == 0 else not x_first[pr['key']]
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            folder = Path(tempfile.mkdtemp(prefix='pairwise-'))
            try:
                lines = []
                for n, pr in enumerate(batch, 1):
                    a, b = (pr['x'], pr['y']) if x_first[pr['key']] else (pr['y'], pr['x'])
                    tag = uuid.uuid4().hex[:6]
                    Image.fromarray(a).save(folder / f'pair{n}-{tag}-1.png')
                    Image.fromarray(b).save(folder / f'pair{n}-{tag}-2.png')
                    lines.append(f'{n}. pair{n}-{tag}-1.png (image 1) and pair{n}-{tag}-2.png (image 2)')
                for attempt in range(2):
                    try:
                        reply = _parse_list(ask(PAIRWISE_PROMPT.format(pairs='\n'.join(lines)), folder))
                        break
                    except (ValueError, RuntimeError):
                        if attempt:
                            raise
                for r in reply:
                    n = int(r.get('pair', 0))
                    if not 1 <= n <= len(batch):
                        continue
                    key = batch[n - 1]['key']
                    pick = str(r.get('more_real')).strip().lower()
                    if pick == 'tie':
                        w = 0.5
                    elif pick in ('1', '2'):
                        w = 1.0 if (pick == '1') == x_first[key] else 0.0
                    else:
                        continue
                    votes.setdefault(key, []).append({'win': w, 'note': r.get('note', '')})
            finally:
                shutil.rmtree(folder, ignore_errors=True)
    out = {}
    for pr in pairs:
        vs = votes.get(pr['key'], [])
        out[pr['key']] = {'win': round(sum(v['win'] for v in vs) / len(vs), 3) if vs else None, 'votes': len(vs),
                          'notes': [v['note'] for v in vs if v.get('note')][:2]}
    return out


BLOOM_PROMPT = """Each image below is a close crop of one plant in a garden picture, with the flowers that plant should
show in full bloom. The plant being judged is shown at full brightness; everything around it is dimmed and is only
context (ignore flowers on the dimmed neighbours). Open every file with the Read tool and answer for the bright plant:

- in_bloom: open flowers clearly visible on it (not buds only, not only foliage).
- matches: its flowers have the described colour family and flower type (for example trumpets, daisies, spikes,
  globes, starry clusters). Allow for the scene's light and for small flowers seen from a distance. Do not fail it
  for details beyond that (exact shade, ageing tints, the plant's leaves or growth habit); put those in the note.

{items}

Reply with JSON only: [{{"file": "<name>", "in_bloom": true or false, "matches": true or false,
  "confidence": <0-100>, "note": "<short>"}}]"""


def bloom(items, batch_size=8, votes=1, seed=0):
    """items: [{'key', 'image', 'flowers' (expected description)}] -> {key: {in_bloom, matches, confidence, note,
    votes}}. With votes > 1 each plant is judged that many times in shuffled batches; in_bloom and matches are the
    majority of the votes received."""
    rng = random.Random(seed)
    ballots = {}
    for _ in range(votes):
        order = list(items)
        rng.shuffle(order)
        for k, r in _bloom_pass(order, batch_size).items():
            ballots.setdefault(k, []).append(r)
    out = {}
    for k, rs in ballots.items():
        conf = [r['confidence'] for r in rs if isinstance(r.get('confidence'), (int, float))]
        out[k] = {'in_bloom': 2 * sum(r['in_bloom'] for r in rs) > len(rs),
                  'matches': 2 * sum(r['matches'] for r in rs) > len(rs),
                  'confidence': round(sum(conf) / len(conf)) if conf else None,
                  'note': next((r['note'] for r in rs if r.get('note')), ''),
                  'votes': len(rs), 'in_bloom_votes': sum(r['in_bloom'] for r in rs)}
    return out


def _bloom_pass(items, batch_size):
    out = {}
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        folder = Path(tempfile.mkdtemp(prefix='bloom-'))
        try:
            names, lines = {}, []
            for it in batch:
                name = f'plant-{uuid.uuid4().hex[:8]}.png'
                Image.fromarray(it['image']).save(folder / name)
                names[name] = it
                lines.append(f'- {name}: should show {it["flowers"]}')
            for attempt in range(2):
                try:
                    reply = _parse_list(ask(BLOOM_PROMPT.format(items='\n'.join(lines)), folder))
                    break
                except (ValueError, RuntimeError):
                    if attempt:
                        raise
            for r in reply:
                it = names.get(r.get('file'))
                if it is not None:
                    out[it['key']] = {'in_bloom': bool(r.get('in_bloom')), 'matches': bool(r.get('matches')),
                                      'confidence': r.get('confidence'), 'note': r.get('note', '')}
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    return out
