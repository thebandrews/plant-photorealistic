"""Fixed judge crops: the same boxes for every experiment, chosen once from the render and frozen in crops.json.

Per view: the whole view (judged downsized), one full-resolution crop per prominent species (its largest
visible plant), crops of open ground and of hardscape joins, and real-photo anchors taken from the cleaned
photo outside the landscape (real by construction, used to calibrate the judge).
"""
import numpy as np

from benchlib import GROUND_KINDS, HARDSCAPE_KINDS

SIZE = 768


def _box_around(cx, cy, shape, size=SIZE):
    h, w = shape[:2]
    x0 = int(min(max(cx - size / 2, 0), w - size))
    y0 = int(min(max(cy - size / 2, 0), h - size))
    return [x0, y0, x0 + size, y0 + size]


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]))


def _best_boxes(score_map, shape, n, taken, min_score, step=128):
    """Greedy: boxes with the highest mean of score_map, overlapping earlier boxes by < 40%."""
    import cv2
    h, w = shape[:2]
    integral = cv2.integral(score_map.astype(np.float32))
    cands = []
    for y in range(0, h - SIZE + 1, step):
        for x in range(0, w - SIZE + 1, step):
            s = integral[y + SIZE, x + SIZE] - integral[y, x + SIZE] - integral[y + SIZE, x] + integral[y, x]
            cands.append((s / SIZE / SIZE, [x, y, x + SIZE, y + SIZE]))
    cands.sort(key=lambda c: -c[0])
    out = []
    for s, b in cands:
        if s < min_score or len(out) >= n:
            break
        if all(_overlap(b, t) < 0.4 for t in taken + out):
            out.append(b)
    return out


def choose(view, species_crops=8, ground_crops=2, hardscape_crops=2, anchor_crops=3):
    shape = view.ids.shape
    crops = [{'id': f'{view.id}-overview', 'kind': 'overview', 'box': [0, 0, shape[1], shape[0]], 'weight': 2.0}]
    taken = []
    seen = set()
    for p in view.plants:
        if len([c for c in crops if c['kind'] == 'plant']) >= species_crops:
            break
        if p['species'] in seen or p['pixels'] < 20000:
            continue
        b = _box_around(*p['centroid'], shape)
        if any(_overlap(b, t) >= 0.5 for t in taken):
            continue
        seen.add(p['species'])
        taken.append(b)
        crops.append({'id': f'{view.id}-plant-{len(taken):02d}', 'kind': 'plant', 'box': b, 'weight': 1.5,
                      'focus': {'plant_id': p['id'], 'species': p['species']}})
    ground = (view.kind_mask(GROUND_KINDS) & ~view.plant_mask & (view.alpha > 0.5)).astype(np.float32)
    for b in _best_boxes(ground, shape, ground_crops, taken, 0.25):
        taken.append(b)
        crops.append({'id': f'{view.id}-ground-{len(taken):02d}', 'kind': 'ground', 'box': b, 'weight': 1.0})
    hard = (view.kind_mask(HARDSCAPE_KINDS) & (view.alpha > 0.5)).astype(np.float32)
    mix = hard * 0.6 + (view.plant_mask.astype(np.float32) * 0.4)
    for b in _best_boxes(mix, shape, hardscape_crops, taken, 0.25):
        taken.append(b)
        crops.append({'id': f'{view.id}-hardscape-{len(taken):02d}', 'kind': 'hardscape', 'box': b, 'weight': 1.0})
    real = (~view.edit_region).astype(np.float32)
    for i, b in enumerate(_best_boxes(real, shape, anchor_crops, [], 0.98)):
        crops.append({'id': f'{view.id}-anchor-real-{i + 1}', 'kind': 'anchor_real', 'box': b, 'weight': 0.0})
    return crops


def cut(img, crop, overview_long=1600):
    from PIL import Image
    x0, y0, x1, y1 = crop['box']
    part = img[y0:y1, x0:x1]
    if crop['kind'] == 'overview':
        h, w = part.shape[:2]
        s = overview_long / max(h, w)
        part = np.asarray(Image.fromarray(part).resize((round(w * s), round(h * s)), Image.LANCZOS))
    return part
