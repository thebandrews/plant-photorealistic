"""Hard gates: an experiment's composite must keep the design. A run that fails any gate is disqualified.

  background   pixels outside the edit region equal the cleaned photo
  hardscape    walls, steps and paving edges stay where the render put them
  plants       every visible plant is still there, unmoved, with the same size and outline envelope
  new_plants   no vegetation appears on open ground where the design has none

Species identity is checked by the vision judge (judge.py species), not here.
Thresholds are the ones agreed for the lab (photoreal lab plan, section 2) and calibrated on the baselines.
"""
import numpy as np

import cv2

T = {
    'background_tol': 3,             # 8-bit difference that still counts as unchanged
    'background_max_changed': 0.001, # share of outside pixels allowed to differ (encoding noise)
    'hardscape_mean_px': 4.0,        # mean distance from render edges to output edges
    'hardscape_min_ref_px': 500,     # fewer reference edge pixels than this: nothing to test
    'plant_big_px': 2500,            # plants at least this visible get the shape test
    'plant_small_px': 400,           # plants at least this visible get the presence test
    'plant_iou': 0.70,
    'plant_shift': 0.03,             # centroid shift / box diagonal
    'plant_height': 0.10,            # relative change of mask height
    'plant_small_iou': 0.40,
    'plant_pass_share': 0.97,        # segmentation is imperfect: share of tested plants that must pass
    'plant_hard_fail_iou': 0.30,     # any big plant below this fails the gate outright
    'plant_struct_corr': 0.12,       # structure fallback: gradient correlation that still counts as the same plant
    'newveg_exg': 0.08,
    'newveg_min_component': 400,
    'newveg_max_share': 0.002,
}


def gate_background(view, out):
    outside = ~view.edit_region
    diff = np.abs(out.astype(np.int16) - view.cleaned.astype(np.int16)).max(axis=2)
    changed = float((diff[outside] > T['background_tol']).mean()) if outside.any() else 0.0
    return {'gate': 'background', 'pass': changed <= T['background_max_changed'],
            'changed_share': round(changed, 6), 'outside_share': round(float(outside.mean()), 4)}


def _edges(rgb):
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    return cv2.Canny(g, 40, 110) > 0


def hardscape_reference(view):
    """Edges of walls, steps and paving in the render, away from plants and inside the landscape matte."""
    from benchlib import HARDSCAPE_KINDS
    hard = view.kind_mask(HARDSCAPE_KINDS) & (view.alpha > 0.5)
    near_plants = cv2.dilate(view.plant_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    region = cv2.dilate(hard.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # include the slab's own outline
    ref = _edges(view.naive_composite()) & region & ~near_plants
    return ref


def gate_hardscape(view, out, ref=None):
    ref = hardscape_reference(view) if ref is None else ref
    if ref.sum() < T['hardscape_min_ref_px']:
        return {'gate': 'hardscape', 'pass': True, 'note': 'too few hardscape edges in view', 'ref_px': int(ref.sum())}
    dist = cv2.distanceTransform((~_edges(out)).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.minimum(dist[ref], 20.0)
    mean = float(d.mean())
    return {'gate': 'hardscape', 'pass': mean <= T['hardscape_mean_px'], 'mean_px': round(mean, 2),
            'p90_px': round(float(np.percentile(d, 90)), 2), 'ref_px': int(ref.sum())}


def _exg(rgb):
    f = rgb.astype(np.float32)
    s = f.sum(axis=2) + 1.0
    return (2 * f[..., 1] - f[..., 0] - f[..., 2]) / s


def gate_new_plants(view, out):
    from benchlib import GROUND_KINDS
    ground = view.kind_mask(GROUND_KINDS) & (view.alpha > 0.5)
    ground &= ~(cv2.dilate(view.plant_mask.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0)
    new = ground & ((_exg(out) - _exg(view.naive_composite())) > T['newveg_exg']) & (_exg(out) > 0.05)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), 8)
    big = [int(a) for a in stats[1:, cv2.CC_STAT_AREA] if a >= T['newveg_min_component']]
    share = sum(big) / new.size
    return {'gate': 'new_plants', 'pass': share <= T['newveg_max_share'], 'new_vegetation_share': round(share, 6),
            'components': len(big)}


def _structure_check(view, ref_img, out, p):
    """Fallback for plants the segmenter can't isolate (sparse fronds, grasses): align the gradient structure
    of the plant's region in the render and the output. Returns (shift / box diagonal, correlation)."""
    x0, y0, x1, y1 = p['box']
    pad = int(0.15 * max(x1 - x0, y1 - y0)) + 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(out.shape[1], x1 + pad), min(out.shape[0], y1 + pad)

    def grad(img):
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return (m - m.mean()) / (m.std() + 1e-6)
    a, b = grad(ref_img), grad(out)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    diag = float(np.hypot(*(np.array(p['box'][2:]) - np.array(p['box'][:2]))))
    corr = float((a * b).mean())
    return float(np.hypot(dx, dy) / diag), corr


def gate_plants(view, out, segmenter, native_masks=None):
    """Segment each visible plant in the output and compare with the same segmentation of the render.
    Plants whose render segmentation doesn't match their own render pixels (IoU < 0.5) can't be judged by
    segmentation; they get the structure-alignment fallback instead."""
    plants = [p for p in view.plants if p['pixels'] >= T['plant_small_px']]
    ref_img = view.naive_composite()
    native_masks = native_masks or segmenter.masks(ref_img, plants)
    out_masks = segmenter.masks(out, plants)
    rows, fails, hard = [], 0, []
    for p in plants:
        a, b = native_masks.get(p['id']), out_masks.get(p['id'])
        big = p['pixels'] >= T['plant_big_px']
        row = {'id': p['id'], 'species': p['species'], 'pixels': p['pixels'], 'big': big}
        if a is None or b is None or not a[0].any():
            row.update(ok=a is None, note='no reference mask' if a is None else 'empty')
            rows.append(row)
            continue
        r = a[1]
        own = view.ids[r[1]:r[3], r[0]:r[2]] == p['pass_index']
        seg_quality = float((a[0] & own).sum() / max((a[0] | own).sum(), 1))
        row['seg_quality'] = round(seg_quality, 3)
        if seg_quality < 0.5:
            shift, corr = _structure_check(view, ref_img, out, p)
            ok = shift <= T['plant_shift'] and corr >= T['plant_struct_corr']
            row.update(method='structure', shift=round(shift, 4), corr=round(corr, 3), ok=bool(ok), iou=None)
            fails += not ok
            rows.append(row)
            continue
        ma, mb = a[0], b[0]
        inter, union = (ma & mb).sum(), (ma | mb).sum()
        iou = float(inter / union) if union else 0.0
        row['iou'] = round(iou, 3)
        if big:
            ya, xa = np.nonzero(ma)
            yb, xb = np.nonzero(mb)
            diag = float(np.hypot(*(np.array(p['box'][2:]) - np.array(p['box'][:2]))))
            shift = float(np.hypot(np.median(xa) - np.median(xb), np.median(ya) - np.median(yb)) / diag) if len(xb) else 1.0
            # robust extents: thin spines and single stray pixels shouldn't count as height
            ha = np.percentile(ya, 98) - np.percentile(ya, 2) + 1
            hb = (np.percentile(yb, 98) - np.percentile(yb, 2) + 1) if len(yb) else 0
            dh = float(abs(hb - ha) / ha)
            row.update(shift=round(shift, 4), height_change=round(float(dh), 3))
            ok = iou >= T['plant_iou'] and shift <= T['plant_shift'] and dh <= T['plant_height']
            if iou < T['plant_hard_fail_iou']:
                hard.append(p['id'])
        else:
            ok = iou >= T['plant_small_iou']
        row['ok'] = bool(ok)
        fails += not ok
        rows.append(row)
    tested = [r for r in rows if 'ok' in r and 'note' not in r]
    share = 1 - fails / len(tested) if tested else 1.0
    return {'gate': 'plants', 'pass': share >= T['plant_pass_share'] and not hard,
            'pass_share': round(share, 4), 'tested': len(tested), 'failed': fails, 'hard_fails': hard,
            'failed_ids': [r['id'] for r in rows if r.get('ok') is False][:40], 'rows': rows}


def run_gates(view, out, segmenter=None, cache=None):
    """All deterministic gates for one view. `cache` (dict) keeps per-view reference data between experiments."""
    cache = cache if cache is not None else {}
    if out.shape != view.cleaned.shape:
        return {'pass': False, 'gates': [{'gate': 'dimensions', 'pass': False,
                                          'note': f'output {out.shape} != view {view.cleaned.shape}'}]}
    ref = cache.setdefault(('hardscape', view.id), None)
    if ref is None:
        ref = cache[('hardscape', view.id)] = hardscape_reference(view)
    gates = [gate_background(view, out), gate_hardscape(view, out, ref), gate_new_plants(view, out)]
    if segmenter is not None:
        key = ('native_masks', view.id)
        if cache.get(key) is None:
            plants = [p for p in view.plants if p['pixels'] >= T['plant_small_px']]
            cache[key] = segmenter.masks(view.naive_composite(), plants)
        gates.append(gate_plants(view, out, segmenter, cache[key]))
    else:
        gates.append({'gate': 'plants', 'pass': False, 'note': 'segmenter not available; plant gate not run'})
    return {'pass': all(g['pass'] for g in gates), 'gates': gates}
