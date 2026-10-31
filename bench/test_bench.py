"""Bench gates and scoring on a small synthetic bench (no SAM, no judge)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as bb  # noqa: E402
import benchlib as bl  # noqa: E402
import gates as gt  # noqa: E402

H, W = 192, 256


@pytest.fixture()
def tiny(tmp_path):
    root = tmp_path / 'bench'
    v = root / 'views' / 'v1'
    (v / 'baseline-run' / 'cleanup-masks').mkdir(parents=True)
    (v / 'baseline').mkdir()
    (root / 'handoff').mkdir()
    rng = np.random.default_rng(0)
    photo = (rng.random((H, W, 3)) * 40 + 120).astype(np.uint8)          # noisy grey "photo"
    ids = np.zeros((H, W), np.int32)
    ids[100:190, 20:236] = 10                                              # bed (open ground)
    ids[60:100, 40:216] = 20                                               # hardscape slab
    ids[110:150, 40:80] = 1                                                # plant 1
    ids[120:170, 150:200] = 2                                              # plant 2
    alpha = np.zeros((H, W), np.uint8)
    alpha[55:192, 10:246] = 255
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[ids == 10] = (110, 80, 55)
    rgb[ids == 20] = (200, 200, 195)
    rgb[ids == 1] = (50, 130, 60)
    rgb[ids == 2] = (70, 150, 70)
    final = photo.copy()
    final[alpha > 0] = rgb[alpha > 0]
    Image.fromarray(photo).save(v / 'cleaned-photo.png')
    Image.fromarray(rgb).save(v / 'native.png')
    Image.fromarray(rgb).save(v / 'baseline-run' / 'straight_rgb.png')
    Image.fromarray(alpha).save(v / 'baseline-run' / 'alpha.png')
    Image.fromarray(final).save(v / 'baseline' / 'final.png')
    np.save(v / 'object-ids.npy', ids)
    np.save(v / 'depth.npy', np.ones((H, W), np.float32))
    (v / 'projected-plants.json').write_text(json.dumps({'plants': [
        {'id': 'P-1', 'species': 'Agave montana', 'pass_index': 1},
        {'id': 'P-2', 'species': 'Salvia nemorosa', 'pass_index': 2}]}))
    (root / 'handoff' / 'candidate-catalog-v3.json').write_text(json.dumps({'objects': [
        {'pass_index': 1, 'kind': 'plant'}, {'pass_index': 2, 'kind': 'plant'},
        {'pass_index': 10, 'kind': 'bed'}, {'pass_index': 20, 'kind': 'hardscape'}]}))
    (root / 'bench-manifest.json').write_text(json.dumps({'bench': 'tiny', 'views': [
        {'id': 'v1', 'camera_id': 'cam-1', 'dimensions': [W, H]}], 'files': []}))
    return bl.Bench(root)


def test_plants_and_kinds(tiny):
    v = tiny.view('v1')
    assert [p['id'] for p in v.plants] == ['P-2', 'P-1']
    assert v.plants[0]['box'] == [150, 120, 200, 170]
    assert len(v.plants[0]['points']) == 4
    assert v.plant_mask.sum() == 40 * 40 + 50 * 50


def test_background_gate(tiny):
    v = tiny.view('v1')
    assert gt.gate_background(v, v.baseline_final)['pass']
    bad = v.baseline_final.copy()
    bad[:20] = 255                                                         # sky outside the landscape
    r = gt.gate_background(v, bad)
    assert not r['pass'] and r['changed_share'] > 0.1


def test_new_plants_gate(tiny):
    v = tiny.view('v1')
    assert gt.gate_new_plants(v, v.baseline_final)['pass']
    bad = v.baseline_final.copy()
    bad[160:185, 95:135] = (40, 160, 40)                                   # a green shrub on open soil
    r = gt.gate_new_plants(v, bad)
    assert not r['pass'] and r['components'] == 1


def test_hardscape_gate_catches_a_moved_slab(tiny, monkeypatch):
    monkeypatch.setitem(gt.T, 'hardscape_min_ref_px', 20)
    v = tiny.view('v1')
    ok = gt.gate_hardscape(v, v.baseline_final)
    assert ok['pass'] and ok['mean_px'] < 1
    bad = v.baseline_final.copy()
    bad[50:108, 10:246] = np.roll(bad[50:108, 10:246], (12, 14), axis=(0, 1))  # slab moved 12 px down, 14 right
    assert not gt.gate_hardscape(v, bad)['pass']


def test_dimension_mismatch_fails(tiny):
    v = tiny.view('v1')
    r = gt.run_gates(v, v.baseline_final[:100])
    assert not r['pass'] and r['gates'][0]['gate'] == 'dimensions'


def test_summary_ranks_on_gates_and_species(tiny, monkeypatch):
    monkeypatch.setenv('PHOTOREAL_EXPERIMENTS', str(tiny.root.parent / 'exps'))
    d = bb.new_experiment(tiny, 'E1', 'A', 'h', 'm')
    bl.write_json(d / 'gates.json', {'pass': True, 'views': {'v1': {'pass': True, 'gates': [{'gate': 'background', 'pass': True}]}}})
    bl.write_json(d / 'judge.json', {'realism': {
        'v1-overview': {'p_photo': 80, 'verdict': 'photo', 'giveaways': [], 'kind': 'overview', 'weight': 2.0},
        'v1-plant-01': {'p_photo': 20, 'verdict': 'synthetic', 'giveaways': [{'type': 'plant_anatomy'}], 'kind': 'plant', 'weight': 1.5},
        'anchor:r': {'p_photo': 90, 'verdict': 'photo', 'giveaways': [], 'kind': 'anchor_real', 'weight': 0},
        'anchor-synth:s': {'p_photo': 70, 'verdict': 'photo', 'giveaways': [], 'kind': 'anchor_synthetic', 'weight': 0}},
        'species': {'v1:P-1': {'same': True, 'confidence': 95}, 'v1:P-2': {'same': False, 'confidence': 85}}})
    s = bb.summarize(d)
    assert s['realism'] == round((80 * 2 + 20 * 1.5) / 3.5, 1)
    assert s['judge_anchor_accuracy'] == 0.5
    assert s['giveaways'] == {'plant_anatomy': 1}
    assert s['species']['different'] == ['v1:P-2'] and s['qualified'] is False


def test_pairwise_cancels_position_bias(tmp_path, monkeypatch):
    import judge as jd
    fake = tmp_path / 'fake_judge.py'
    fake.write_text('import json, re, sys\n'
                    'n = len(re.findall(r"^\d+\. ", sys.stdin.read(), re.M))\n'
                    'print(json.dumps([{"pair": i, "more_real": 1} for i in range(1, n + 1)]))\n')
    monkeypatch.setenv('PHOTOREAL_JUDGE_CLI', f'{sys.executable} {fake}')
    img = np.zeros((8, 8, 3), np.uint8)
    res = jd.pairwise([{'key': f'k{i}', 'x': img, 'y': img} for i in range(5)], batch_size=2, passes=2)
    assert all(r['win'] == 0.5 and r['votes'] == 2 for r in res.values())


def test_summary_reports_win_vs_shipped(tiny, monkeypatch):
    monkeypatch.setenv('PHOTOREAL_EXPERIMENTS', str(tiny.root.parent / 'exps'))
    d = bb.new_experiment(tiny, 'E2', 'D', 'h', 'm')
    bl.write_json(d / 'pairwise.json', {'vs': 'X0-final', 'sanity_accuracy': 1.0, 'crops': {
        'v1-overview': {'win': 1.0, 'weight': 2.0, 'kind': 'overview'},
        'v1-plant-01': {'win': 0.25, 'weight': 1.0, 'kind': 'plant'}}})
    s = bb.summarize(d)
    assert s['win_vs_shipped'] == round((2.0 + 0.25) / 3, 3) and s['pairwise_sanity'] == 1.0


def test_bloom_items_and_qualification(tiny, monkeypatch):
    bl.write_json(tiny.root / 'bloom.json', {'bloom': {'Salvia nemorosa': 'rose-magenta spikes'},
                                             'not_required': {'Agave montana': 'tall stalk'}})
    v = tiny.view('v1')
    items = bb._bloom_items(tiny, v, v.baseline_final)
    assert [i['key'] for i in items] == ['v1:P-2'] and items[0]['flowers'] == 'rose-magenta spikes'
    monkeypatch.setenv('PHOTOREAL_EXPERIMENTS', str(tiny.root.parent / 'exps'))
    d = bb.new_experiment(tiny, 'E3', 'D', 'h', 'm')
    bl.write_json(d / 'gates.json', {'pass': True, 'views': {}})
    judged = {'realism': {}, 'species': {}, 'bloom': {
        'v1:P-2': {'in_bloom': False, 'matches': False, 'confidence': 90, 'species': 'Salvia nemorosa'}}}
    bl.write_json(d / 'judge.json', judged)
    s = bb.summarize(d)
    assert s['bloom'] == {'checked': 1, 'in_bloom': 0, 'share': 0.0, 'pass': False, 'missing': ['Salvia nemorosa']}
    assert s['qualified'] is False
    judged['bloom']['v1:P-2'].update(in_bloom=True, matches=True)
    bl.write_json(d / 'judge.json', judged)
    assert bb.summarize(d)['qualified'] is True


def test_invented_plants_disqualify(tiny, monkeypatch):
    monkeypatch.setenv('PHOTOREAL_EXPERIMENTS', str(tiny.root.parent / 'exps'))
    d = bb.new_experiment(tiny, 'E4', 'D', 'h', 'm')
    bl.write_json(d / 'gates.json', {'pass': True, 'views': {}})
    bl.write_json(d / 'judge.json', {'realism': {}, 'species': {
        'v1:P-1': {'same': True, 'extra': True, 'confidence': 85},
        'v1:P-2': {'same': True, 'extra': True, 'confidence': 40}}})
    s = bb.summarize(d)
    assert s['species']['extra_plants'] == ['v1:P-1'] and s['species']['different'] == [] and s['qualified'] is False


def test_species_pairs_name_species_flowers_and_ground(tiny):
    bl.write_json(tiny.root / 'bloom.json', {'bloom': {'Salvia nemorosa': 'rose-magenta spikes'}})
    bl.write_json(tiny.root / 'crops.json', {'views': {'v1': [{'id': 'v1-ground-09', 'kind': 'ground', 'box': [10, 100, 74, 164]}]}})
    v = tiny.view('v1')
    pairs = bb._species_pairs(tiny, v, v.baseline_final)
    by = {p['key']: p for p in pairs}
    assert by['v1:P-2']['flowers'] == 'rose-magenta spikes' and by['v1:P-1']['flowers'] is None
    assert by['v1:v1-ground-09']['species'] is None


def test_bloom_verdict_is_a_majority_of_votes(monkeypatch):
    import judge as jd
    seq = iter([True, False, True, False, False, True])
    monkeypatch.setattr(jd, '_bloom_pass', lambda items, batch_size: {
        it['key']: {'in_bloom': next(seq), 'matches': True, 'confidence': 80, 'note': ''} for it in items})
    items = [{'key': 'a', 'image': None, 'flowers': 'x'}, {'key': 'b', 'image': None, 'flowers': 'y'}]
    out = jd.bloom(items, votes=3)
    assert sorted(out) == ['a', 'b'] and all(x['votes'] == 3 and x['matches'] for x in out.values())
    assert sum(x['in_bloom_votes'] for x in out.values()) == 3
    assert all(x['in_bloom'] == (x['in_bloom_votes'] >= 2) for x in out.values())


def test_unchanged_plants_keep_the_baseline_bloom_verdict(tiny, monkeypatch):
    bl.write_json(tiny.root / 'bloom.json', {'bloom': {'Salvia nemorosa': 'rose-magenta spikes'}})
    monkeypatch.setenv('PHOTOREAL_EXPERIMENTS', str(tiny.root.parent / 'exps'))
    v = tiny.view('v1')
    base = bb.new_experiment(tiny, 'X0-final', 'X', 'h', 'm')
    (base / 'out').mkdir(exist_ok=True)
    Image.fromarray(v.baseline_final).save(base / 'out' / 'v1.png')
    bl.write_json(base / 'judge.json', {'bloom': {'v1:P-2': {'in_bloom': True, 'matches': True, 'votes': 3,
                                                             'species': 'Salvia nemorosa'}}})
    d = bb.new_experiment(tiny, 'E9', 'H', 'h', 'm')
    (d / 'out').mkdir(exist_ok=True)
    img = v.baseline_final.copy()
    img[v.ids == 10] = (90, 60, 40)                     # ground changed, plant 2 untouched
    Image.fromarray(img).save(d / 'out' / 'v1.png')

    class NoJudge:
        @staticmethod
        def bloom(items, votes=1):
            return {it['key']: {'in_bloom': False, 'matches': False, 'votes': votes} for it in items}
    out = bb.judge_bloom(tiny, d, NoJudge)
    assert out['v1:P-2']['in_bloom'] is True and out['v1:P-2']['inherited_from'] == 'X0-final'
    img[v.ids == 2] = (200, 60, 150)                    # plant 2 repainted: judged afresh
    Image.fromarray(img).save(d / 'out' / 'v1.png')
    out = bb.judge_bloom(tiny, d, NoJudge)
    assert out['v1:P-2']['in_bloom'] is False and 'inherited_from' not in out['v1:P-2']
    assert out['v1:P-2']['species'] == 'Salvia nemorosa'


def test_bloom_crop_centres_zooms_and_dims_around_the_plant(tiny):
    v = tiny.view('v1')
    p = next(p for p in v.plants if p['id'] == 'P-1')                 # 40x40 px plant at x 40-80, y 110-150
    img = np.full((H, W, 3), 200, np.uint8)
    crop = bb._bloom_crop(img, v, p)
    assert max(crop.shape[:2]) == bb.BLOOM_CROP_PX                      # small plant zoomed in
    h, w = crop.shape[:2]
    assert crop[h // 2, w // 2].min() >= 195                            # the plant stays bright at the centre
    assert crop[2, 2].max() <= round(200 * bb.BLOOM_DIM) + 2            # the rest is dimmed


def test_flagged_species_pairs_are_rechecked_by_majority(monkeypatch):
    import judge as jd
    calls = []

    def fake(pairs, batch_size):
        calls.append([p['key'] for p in pairs])
        n = len(calls)
        return {p['key']: {'same': True, 'extra': p['key'] == 'a' and n == 1, 'confidence': 80, 'note': ''}
                for p in pairs}
    monkeypatch.setattr(jd, '_species_pass', fake)
    out = jd.species([{'key': 'a'}, {'key': 'b'}], recheck=2)
    assert calls[1:] == [['a'], ['a']]                                  # only the flagged pair is asked again
    assert out['a']['extra'] is False and out['a']['votes'] == 3 and out['a']['extra_votes'] == 1
    assert out['b']['votes'] == 1 and out['b']['extra'] is False


def test_goal_needs_realistic_share_and_no_toys(tiny, monkeypatch):
    monkeypatch.setenv('PHOTOREAL_EXPERIMENTS', str(tiny.root.parent / 'exps'))
    d = bb.new_experiment(tiny, 'E1', 'A', 'h', 'm')
    bl.write_json(d / 'gates.json', {'pass': True, 'views': {}})
    def im(score, kind='plant', weight=1.0):
        return {'score': score, 'realistic': score >= 2.75, 'giveaways': [], 'kind': kind, 'weight': weight}
    real = {'v1-plant-01': {'p_photo': 5, 'verdict': 'synthetic', 'giveaways': [], 'kind': 'plant', 'weight': 1.0}}
    judged = {'realism': real, 'species': {}, 'impression': {
        'v1-overview': im(3.5, 'overview', 2.0), 'v1-plant-01': im(3.0), 'v1-plant-02': im(4.0), 'v1-ground-01': im(2.5),
        'anchor:r': im(5.0, 'anchor_real', 0), 'anchor-synth:s': im(1.5, 'anchor_synthetic', 0)}}
    bl.write_json(d / 'judge.json', judged)
    s = bb.summarize(d)
    assert s['impression_share'] == 0.8 and s['impression_toys'] == [] and s['impression_anchor_ok'] is True
    assert s['goal_met'] is True
    judged['impression']['v1-ground-01'] = im(1.0)
    bl.write_json(d / 'judge.json', judged)
    s = bb.summarize(d)
    assert s['impression_toys'] == ['v1-ground-01'] and s['goal_met'] is False
