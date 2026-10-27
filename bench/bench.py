"""Photoreal lab bench: score experiments against a frozen pipeline run.

  python bench/bench.py --bench DIR init-crops            choose and freeze the judge crops (once per bench)
  python bench/bench.py --bench DIR baselines             register X0 experiments (current final, raw 3D composite)
  python bench/bench.py --bench DIR new ID --layer A --hypothesis "..." --method "..."
  python bench/bench.py --bench DIR gate ID               hard gates (background, hardscape, plants, new plants)
  python bench/bench.py --bench DIR judge ID              vision judge: realism per crop, species consistency
  python bench/bench.py --bench DIR score ID              gate + judge + summary (scores.json)
  python bench/bench.py --bench DIR board                 leaderboard page (experiments/board/index.html)

An experiment is a folder experiments/<ID>/ with experiment.json, out/<view>.png at full size, and run.json
(gpu_seconds, codex_turns, notes). Experiments never write to the bench. See bench/README.md.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402
import crops as cr  # noqa: E402
import gates as gt  # noqa: E402

SYNTH_ANCHORS_PER_VIEW = 2
REAL_ANCHORS = 6
SPECIES_PAIRS_PER_VIEW = 12
BLOOM_PLANTS_PER_VIEW = 12      # largest plant of each flowering species, biggest first
BLOOM_MIN_PX = 2500
BLOOM_PASS_SHARE = 0.9          # share of checked flowering plants judged in bloom and true to description
BLOOM_VOTES = 3                 # bloom verdicts are a majority of this many judge votes
BLOOM_SAME_TOL = 2              # a plant whose own pixels differ from the baseline's by at most this keeps its verdict
SPECIES_FAIL_CONFIDENCE = 70
PAIRWISE_REF = 'X0-final'      # side-by-side reference: the shipped output
PAIRWISE_SANITY = 4             # pairs of a real-photo crop vs a shipped crop, to measure the judge


def exp_root(bench):
    return Path(os.environ.get('PHOTOREAL_EXPERIMENTS') or bench.root.parent / 'experiments')


def exp_dir(bench, exp_id):
    d = exp_root(bench) / exp_id
    if not (d / 'experiment.json').exists():
        raise SystemExit(f'No experiment {exp_id} in {exp_root(bench)}')
    return d


def load_out(bench, d, view):
    p = d / 'out' / f'{view.id}.png'
    if not p.exists():
        raise SystemExit(f'Missing output {p}')
    return bl.load_rgb(p)


# -- commands -------------------------------------------------------------------
def cmd_init_crops(bench, args):
    p = bench.root / 'crops.json'
    if p.exists() and not args.force:
        raise SystemExit('crops.json exists; the crops are frozen (use --force only before any experiment is scored)')
    data = {'size': cr.SIZE, 'views': {v.id: cr.choose(v) for v in bench.views}}
    bl.write_json(p, data)
    for vid, cs in data['views'].items():
        print(vid, ', '.join(f"{c['kind']}" for c in cs))


def new_experiment(bench, exp_id, layer, hypothesis, method, branch=None, budget_gpu_s=None):
    d = exp_root(bench) / exp_id
    if (d / 'experiment.json').exists():
        raise SystemExit(f'Experiment {exp_id} already exists')
    (d / 'out').mkdir(parents=True, exist_ok=True)
    bl.write_json(d / 'experiment.json', {
        'id': exp_id, 'layer': layer, 'hypothesis': hypothesis, 'method': method, 'branch': branch,
        'bench': bench.name, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'budget_gpu_seconds': budget_gpu_s})
    return d


def cmd_new(bench, args):
    d = new_experiment(bench, args.id, args.layer, args.hypothesis, args.method, args.branch, args.budget_gpu_s)
    print(d)


def cmd_baselines(bench, args):
    specs = [('X0-final', 'Current pipeline output (Reddy run 20261025, photoreal stage as shipped)',
              lambda v: v.baseline_final),
             ('X0-raw3d', 'Raw 3D composite: baseline render straight-alpha over the cleaned photo, no finishing',
              lambda v: v.naive_composite())]
    for exp_id, method, make in specs:
        d = exp_root(bench) / exp_id
        if not (d / 'experiment.json').exists():
            new_experiment(bench, exp_id, 'X', 'Control', method)
        for v in bench.views:
            bl.save_rgb(d / 'out' / f'{v.id}.png', make(v))
        bl.write_json(d / 'run.json', {'gpu_seconds': 0, 'codex_turns': 0, 'notes': 'control'})
        print('registered', exp_id)


_SEG = None


def segmenter():
    global _SEG
    if _SEG is None:
        try:
            from segment import SamSegmenter
            _SEG = SamSegmenter()
        except ImportError as e:
            print(f'warning: plant gate needs torch + transformers ({e})', file=sys.stderr)
            _SEG = False
    return _SEG or None


def _native_mask_cache(bench, view, seg):
    p = exp_root(bench) / '.cache' / f'native-masks-{view.id}.pkl'
    if p.exists():
        return pickle.loads(p.read_bytes())
    plants = [x for x in view.plants if x['pixels'] >= gt.T['plant_small_px']]
    masks = seg.masks(view.naive_composite(), plants)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(masks))
    return masks


def cmd_gate(bench, args, exp_id=None):
    d = exp_dir(bench, exp_id or args.id)
    seg = None if getattr(args, 'no_plants', False) else segmenter()
    result = {'views': {}, 'thresholds': gt.T}
    for v in bench.views:
        out = load_out(bench, d, v)
        cache = {}
        if seg is not None:
            cache[('native_masks', v.id)] = _native_mask_cache(bench, v, seg)
        t0 = time.time()
        r = gt.run_gates(v, out, seg, cache)
        r['seconds'] = round(time.time() - t0, 1)
        result['views'][v.id] = r
        print(v.id, 'PASS' if r['pass'] else 'FAIL',
              '; '.join(f"{g['gate']}={'ok' if g['pass'] else 'FAIL'}" for g in r['gates']))
    result['pass'] = all(r['pass'] for r in result['views'].values())
    bl.write_json(d / 'gates.json', result)
    return result


def _plant_crop(img, p):
    """Square crop around one plant (its box plus 30%), at most 1024 px."""
    from PIL import Image
    x0, y0, x1, y1 = p['box']
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0, 256) * 1.3
    b = [int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(img.shape[1], cx + s / 2)),
         int(min(img.shape[0], cy + s / 2))]
    part = img[b[1]:b[3], b[0]:b[2]]
    long = max(part.shape[:2])
    if long > 1024:
        k = 1024 / long
        part = np.asarray(Image.fromarray(part).resize((round(part.shape[1] * k), round(part.shape[0] * k)),
                                                       Image.LANCZOS))
    return part


def _species_pairs(bench, view, out):
    """The largest plants (named species and expected flowers) plus the fixed ground crops (invented plants)."""
    ref = view.naive_composite()
    flowers = (bench.bloom or {}).get('bloom', {})
    pairs = [{'key': f"{view.id}:{p['id']}", 'species': p['species'], 'flowers': flowers.get(p['species']),
              'a': _plant_crop(ref, p), 'b': _plant_crop(out, p)} for p in view.plants[:SPECIES_PAIRS_PER_VIEW]]
    for c in (bench.crops or {}).get('views', {}).get(view.id, []):
        if c['kind'] == 'ground':
            pairs.append({'key': f"{view.id}:{c['id']}", 'species': None, 'a': cr.cut(ref, c), 'b': cr.cut(out, c)})
    return pairs


def _bloom_items(bench, view, out):
    """The largest visible plant of each flowering species (bloom.json), biggest first."""
    expected = (bench.bloom or {}).get('bloom', {})
    seen, items = set(), []
    for p in sorted(view.plants, key=lambda p: -p['pixels']):
        if p['species'] in expected and p['species'] not in seen and p['pixels'] >= BLOOM_MIN_PX:
            seen.add(p['species'])
            items.append({'key': f"{view.id}:{p['id']}", 'species': p['species'], 'flowers': expected[p['species']],
                          'image': _plant_crop(out, p)})
    return items[:BLOOM_PLANTS_PER_VIEW]


def _plant_unchanged(view, key, out, ref):
    """True when every pixel of the plant (its own object ID) is within BLOOM_SAME_TOL of the reference."""
    p = next(p for p in view.plants if f"{view.id}:{p['id']}" == key)
    m = view.ids == p['pass_index']
    return bool(m.any()) and int(np.abs(out[m].astype(int) - ref[m].astype(int)).max()) <= BLOOM_SAME_TOL


def judge_bloom(bench, d, jd):
    """Bloom verdicts: a plant left exactly as in the baseline keeps the baseline's (majority-vote) verdict; the
    others get a majority of BLOOM_VOTES judge votes. The bloom judge is noisy on the same pixels, so an
    experiment that didn't touch a plant must not gain or lose its bloom by chance."""
    ref_dir = exp_root(bench) / PAIRWISE_REF
    ref_bloom = {}
    if d.name != PAIRWISE_REF and (ref_dir / 'judge.json').exists():
        ref_bloom = {k: x for k, x in (bl.read_json(ref_dir / 'judge.json').get('bloom') or {}).items()
                     if x.get('votes', 1) >= BLOOM_VOTES}
    out, ask = {}, []
    for v in bench.views:
        img = load_out(bench, d, v)
        ref = load_out(bench, ref_dir, v) if ref_bloom else None
        for it in _bloom_items(bench, v, img):
            if it['key'] in ref_bloom and _plant_unchanged(v, it['key'], img, ref):
                out[it['key']] = dict(ref_bloom[it['key']], inherited_from=PAIRWISE_REF)
            else:
                ask.append(it)
    judged = jd.bloom(ask, votes=BLOOM_VOTES) if ask else {}
    out.update({k: dict(r, species=next(i['species'] for i in ask if i['key'] == k)) for k, r in judged.items()})
    return out


def cmd_judge(bench, args, exp_id=None):
    import judge as jd
    d = exp_dir(bench, exp_id or args.id)
    if not bench.crops:
        raise SystemExit('Run init-crops first')
    items, meta = [], {}
    for v in bench.views:
        out = load_out(bench, d, v)
        for c in bench.crops['views'][v.id]:
            if c['kind'] == 'anchor_real':
                img, truth, key = cr.cut(v.cleaned, c), 'photo', f"anchor:{c['id']}"
            else:
                img, truth, key = cr.cut(out, c), None, c['id']
            items.append({'key': key, 'image': img, 'truth': truth})
            meta[key] = c
        if d.name != 'X0-final':
            plants = [c for c in bench.crops['views'][v.id] if c['kind'] == 'plant'][:SYNTH_ANCHORS_PER_VIEW]
            for c in plants:
                key = f"anchor-synth:{c['id']}"
                items.append({'key': key, 'image': cr.cut(v.baseline_final, c), 'truth': 'synthetic'})
                meta[key] = dict(c, kind='anchor_synthetic')
    # Real-photo anchors: crops of the project's other site photos (anchors/real/*.png in the bench)
    for p in sorted((bench.root / 'anchors' / 'real').glob('*.png'))[::2][:REAL_ANCHORS]:
        key = f'anchor:{p.stem}'
        items.append({'key': key, 'image': bl.load_rgb(p), 'truth': 'photo'})
        meta[key] = {'id': p.stem, 'kind': 'anchor_real', 'weight': 0}
    real = jd.realism(items, passes=args.passes, seed=hash(d.name) & 0xffff)
    pairs = []
    if not args.no_species:
        for v in bench.views:
            pairs += _species_pairs(bench, v, load_out(bench, d, v))
    spec = jd.species(pairs) if pairs else {}
    bloom = judge_bloom(bench, d, jd) if bench.bloom else {}
    judged = {'realism': {k: dict(r, kind=meta[k]['kind'], weight=meta[k].get('weight', 0)) for k, r in real.items()},
              'species': {k: dict(r, species=next(p['species'] for p in pairs if p['key'] == k)) for k, r in spec.items()},
              'bloom': bloom,
              'judge_cli': ' '.join(jd._cli()), 'passes': args.passes}
    bl.write_json(d / 'judge.json', judged)
    return judged


def cmd_pairwise(bench, args, exp_id=None):
    """Side by side against the shipped output: for every scored crop, which looks more like a photograph?
    Sanity pairs (a real-photo crop against a shipped crop) measure the judge in the same batches."""
    import judge as jd
    d = exp_dir(bench, exp_id or args.id)
    ref = exp_dir(bench, PAIRWISE_REF)
    if d.name == PAIRWISE_REF:
        return None
    pairs, meta = [], {}
    for v in bench.views:
        out, base = load_out(bench, d, v), load_out(bench, ref, v)
        for c in bench.crops['views'][v.id]:
            if c['kind'] == 'anchor_real':
                continue
            pairs.append({'key': c['id'], 'x': cr.cut(out, c), 'y': cr.cut(base, c)})
            meta[c['id']] = c
    reals = sorted((bench.root / 'anchors' / 'real').glob('*.png'))[1::2][:PAIRWISE_SANITY]
    shipped = [(v, c) for v in bench.views for c in bench.crops['views'][v.id] if c['kind'] == 'plant']
    for i, (p, (v, c)) in enumerate(zip(reals, shipped[::3])):
        pairs.append({'key': f'sanity:{i}', 'x': bl.load_rgb(p), 'y': cr.cut(v.baseline_final, c)})
    res = jd.pairwise(pairs, passes=args.passes, seed=hash(d.name) & 0xffff)
    sanity = [r['win'] for k, r in res.items() if k.startswith('sanity:') and r['win'] is not None]
    result = {'vs': PAIRWISE_REF, 'judge_cli': ' '.join(jd._cli()), 'passes': args.passes,
              'sanity_accuracy': round(sum(sanity) / len(sanity), 3) if sanity else None,
              'crops': {k: dict(r, kind=meta[k]['kind'], weight=meta[k].get('weight', 1)) for k, r in res.items()
                        if not k.startswith('sanity:')}}
    bl.write_json(d / 'pairwise.json', result)
    return result


def summarize(d):
    exp = bl.read_json(d / 'experiment.json')
    gates = bl.read_json(d / 'gates.json') if (d / 'gates.json').exists() else None
    judged = bl.read_json(d / 'judge.json') if (d / 'judge.json').exists() else None
    run = bl.read_json(d / 'run.json') if (d / 'run.json').exists() else {}
    s = {'id': exp['id'], 'layer': exp.get('layer'), 'hypothesis': exp.get('hypothesis'), 'method': exp.get('method'),
         'gpu_seconds': run.get('gpu_seconds'), 'codex_turns': run.get('codex_turns'),
         'gates_pass': gates['pass'] if gates else None,
         'gates': {vid: {g['gate']: g['pass'] for g in r['gates']} for vid, r in (gates or {}).get('views', {}).items()}}
    if judged:
        r = judged['realism']
        scored = [(x, x['weight']) for x in r.values() if x['kind'] not in ('anchor_real', 'anchor_synthetic')
                  and x['p_photo'] is not None]
        wsum = sum(w for _, w in scored) or 1
        s['realism'] = round(sum(x['p_photo'] * w for x, w in scored) / wsum, 1)
        s['photo_share'] = round(sum(w for x, w in scored if x['verdict'] == 'photo') / wsum, 3)
        s['overview'] = {k: x['p_photo'] for k, x in r.items() if x['kind'] == 'overview'}
        anchors = [x for x in r.values() if x['kind'] in ('anchor_real', 'anchor_synthetic') and x['verdict']]
        right = sum((x['kind'] == 'anchor_real') == (x['verdict'] == 'photo') for x in anchors)
        s['judge_anchor_accuracy'] = round(right / len(anchors), 3) if anchors else None
        counts = {}
        for x in r.values():
            if x['kind'] in ('anchor_real', 'anchor_synthetic'):
                continue
            for g in x['giveaways']:
                counts[g.get('type', 'other')] = counts.get(g.get('type', 'other'), 0) + 1
        s['giveaways'] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
        sp = judged.get('species', {})
        sure = {k: x for k, x in sp.items() if (x.get('confidence') or 0) >= SPECIES_FAIL_CONFIDENCE}
        bad = [k for k, x in sure.items() if not x['same']]
        extra = [k for k, x in sure.items() if x.get('extra')]
        s['species'] = {'pairs': len(sp), 'different': bad, 'extra_plants': extra, 'pass': not bad and not extra}
        bl_ = judged.get('bloom') or {}
        if bl_:
            ok = [k for k, x in bl_.items() if x['in_bloom'] and x['matches']]
            share = len(ok) / len(bl_)
            s['bloom'] = {'checked': len(bl_), 'in_bloom': len(ok), 'share': round(share, 3),
                          'pass': share >= BLOOM_PASS_SHARE, 'missing': sorted(x['species'] for k, x in bl_.items()
                                                                               if k not in ok)}
        if s['gates_pass'] is not None:
            s['qualified'] = bool(s['gates_pass'] and s['species']['pass'] and s.get('bloom', {}).get('pass', True))
    if (d / 'pairwise.json').exists():
        pw = bl.read_json(d / 'pairwise.json')
        scored = [(x['win'], x['weight']) for x in pw['crops'].values() if x['win'] is not None]
        wsum = sum(w for _, w in scored) or 1
        s['win_vs_shipped'] = round(sum(x * w for x, w in scored) / wsum, 3) if scored else None
        s['pairwise_sanity'] = pw.get('sanity_accuracy')
    elif d.name == PAIRWISE_REF:
        s['win_vs_shipped'] = 0.5
    bl.write_json(d / 'scores.json', s)
    return s


def cmd_score(bench, args):
    cmd_gate(bench, args, args.id)
    cmd_judge(bench, args, args.id)
    if not args.no_pairwise:
        cmd_pairwise(bench, args, args.id)
    s = summarize(exp_dir(bench, args.id))
    print(json.dumps({k: s.get(k) for k in ('id', 'qualified', 'gates_pass', 'realism', 'photo_share', 'win_vs_shipped',
                                            'pairwise_sanity', 'judge_anchor_accuracy', 'giveaways')}, indent=2))


def cmd_board(bench, args):
    import board
    root = exp_root(bench)
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / 'experiment.json').exists()):
        rows.append((d, summarize(d)))
    path = board.build(bench, root, rows)
    print(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', default=os.environ.get('PHOTOREAL_BENCH'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init-crops'); p.add_argument('--force', action='store_true')
    sub.add_parser('baselines')
    p = sub.add_parser('new'); p.add_argument('id'); p.add_argument('--layer', required=True)
    p.add_argument('--hypothesis', required=True); p.add_argument('--method', required=True)
    p.add_argument('--branch'); p.add_argument('--budget-gpu-s', type=int)
    p = sub.add_parser('gate'); p.add_argument('id'); p.add_argument('--no-plants', action='store_true')
    p = sub.add_parser('judge'); p.add_argument('id'); p.add_argument('--passes', type=int, default=2)
    p.add_argument('--no-species', action='store_true')
    p = sub.add_parser('score'); p.add_argument('id'); p.add_argument('--passes', type=int, default=2)
    p.add_argument('--no-species', action='store_true'); p.add_argument('--no-plants', action='store_true')
    p.add_argument('--no-pairwise', action='store_true')
    p = sub.add_parser('pairwise'); p.add_argument('id'); p.add_argument('--passes', type=int, default=2)
    sub.add_parser('board')
    args = ap.parse_args(argv)
    bench = bl.Bench(args.bench)
    {'init-crops': cmd_init_crops, 'baselines': cmd_baselines, 'new': cmd_new, 'gate': cmd_gate,
     'judge': cmd_judge, 'pairwise': cmd_pairwise, 'score': cmd_score, 'board': cmd_board}[args.cmd](bench, args)


if __name__ == '__main__':
    main()
