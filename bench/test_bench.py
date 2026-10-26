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
