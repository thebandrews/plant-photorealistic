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
SPECIES_FAIL_CONFIDENCE = 70


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


def _species_pairs(view, out):
    from PIL import Image
    ref = view.naive_composite()
    pairs = []
    for p in view.plants[:SPECIES_PAIRS_PER_VIEW]:
        x0, y0, x1, y1 = p['box']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(x1 - x0, y1 - y0, 256) * 1.3
        b = [int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(ref.shape[1], cx + s / 2)),
             int(min(ref.shape[0], cy + s / 2))]

        def cut(img):
            part = img[b[1]:b[3], b[0]:b[2]]
            long = max(part.shape[:2])
            if long > 1024:
                k = 1024 / long
                part = np.asarray(Image.fromarray(part).resize((round(part.shape[1] * k), round(part.shape[0] * k)),
                                                               Image.LANCZOS))
            return part
        pairs.append({'key': f"{view.id}:{p['id']}", 'species': p['species'], 'a': cut(ref), 'b': cut(out)})
    return pairs


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
            pairs += _species_pairs(v, load_out(bench, d, v))
    spec = jd.species(pairs) if pairs else {}
    judged = {'realism': {k: dict(r, kind=meta[k]['kind'], weight=meta[k].get('weight', 0)) for k, r in real.items()},
              'species': {k: dict(r, species=next(p['species'] for p in pairs if p['key'] == k)) for k, r in spec.items()},
              'judge_cli': ' '.join(jd._cli()), 'passes': args.passes}
    bl.write_json(d / 'judge.json', judged)
    return judged


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
        bad = [k for k, x in sp.items() if not x['same'] and (x.get('confidence') or 0) >= SPECIES_FAIL_CONFIDENCE]
        s['species'] = {'pairs': len(sp), 'different': bad, 'pass': not bad}
        if s['gates_pass'] is not None:
            s['qualified'] = bool(s['gates_pass'] and s['species']['pass'])
    bl.write_json(d / 'scores.json', s)
    return s


def cmd_score(bench, args):
    cmd_gate(bench, args, args.id)
    cmd_judge(bench, args, args.id)
    s = summarize(exp_dir(bench, args.id))
    print(json.dumps({k: s.get(k) for k in ('id', 'qualified', 'gates_pass', 'realism', 'photo_share',
                                            'judge_anchor_accuracy', 'giveaways')}, indent=2))


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
    sub.add_parser('board')
    args = ap.parse_args(argv)
    bench = bl.Bench(args.bench)
    {'init-crops': cmd_init_crops, 'baselines': cmd_baselines, 'new': cmd_new, 'gate': cmd_gate,
     'judge': cmd_judge, 'score': cmd_score, 'board': cmd_board}[args.cmd](bench, args)


if __name__ == '__main__':
    main()
