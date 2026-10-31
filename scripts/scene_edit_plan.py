"""Freeze a scene photographic edit plan for one finished photo view, BEFORE generation.

Input is a checked schema 4 photo_finish review (the composite that would otherwise be delivered). The plan
derives the landscape edit region from that composite's renderer matte and transition permissions, lays out
one full-view job plus overlapping native-resolution tiles over the region, and writes one prompt per job whose
inventory (species, IDs, positions as % of the crop, projected pixel sizes, edge fragments included) is computed
from the actual renderer object IDs. Plant descriptions and bloom come from the agent's plant records file.
Deterministic: validate() re-derives every field. No image generation or recognition happens here.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from handoff import checked, read, record, require
from raster import load, save

KIND = 'scene_photographic_edit_plan'
MODEL = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high'}
IMAGE_TOOLS = ('image_gen.imagegen', 'image_gen__imagegen')
MAX_ATTEMPTS = 2
DEFAULTS = {'tile': [1152, 1024], 'overlap': .3, 'margin_px': 24, 'feather_px': 24}

PREAMBLE = (
    "Use case: sketch-to-render. Edit target: the supplied image{part}, an existing designed garden in a real site photograph. "
    "Make this exact view an ordinary real photograph, with botanically authentic living plants. Keep the same camera, crop, "
    "perspective, lighting, every plant identity, root position, presentation size, extent and overlap. No plant may move, appear, "
    "disappear, change species or size. Only refine surface detail and natural leaf anatomy within the existing extent. "
    "Keep {hardscape} exactly in place with identical geometry; {materials}. {ground} Plants rooted in beds, never in concrete, "
    "paving or steps. No new plants, pots, steps, paving, structures or text. Keep photographic areas outside landscape unchanged. "
    "The following inventory specifies every visible plant, including edge fragments: x and y are percentages from the top left, "
    "width and height are projected visible pixel sizes in the supplied image, not physical centimetres. Do not complete clipped "
    "plants or expose occluded plants. No invented hybrid foliage. Every flowering plant is in full bloom, true to its cultivar; "
    "bloom requirements stay within current plant dimensions.{reference}")
TILE_PART = ' (one native-resolution tile of a larger view)'
TILE_REFERENCE = (" The second supplied image is the matching crop of the already-edited whole view: match its light, colour, "
                  "materials and plant rendering, while following this tile's own layout and finer detail.")
MULCH = 'Ground inside beds is shredded bark mulch.'
NO_MULCH = 'Ground inside beds keeps its existing designed material.'
HARDSCAPE = 'concrete, steps, walls, planters, edging and the house'
MATERIALS = 'realistic concrete, stone and metal textures at true scale'


def half_up(value):
    return int(math.floor(value + .5))


def edit_region(alpha, transition, margin):
    """Landscape pixels of the finished composite, grown by a bounded margin for natural foliage edges."""
    landscape = (alpha > 0) | (transition > 0)
    require(np.any(landscape) and np.any(~landscape), 'Scene edit needs landscape and photographic background')
    distance = cv2.distanceTransform((~landscape).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance <= margin


def region_weight(region, feather):
    """Raised-cosine inward feather; exactly zero outside the region."""
    inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    t = np.clip(inside / feather, 0, 1)
    weight = (.5 - .5 * np.cos(np.pi * t)).astype(np.float32)
    weight[~region] = 0
    return weight


def starts(lo, hi, size, limit, ramp):
    """Window starts covering [lo, hi) inside [0, limit); adjacent windows overlap by at least `ramp` px."""
    size = min(size, limit)
    a, b = max(0, lo), min(limit, hi)
    if b - a < size:
        a = max(0, min(a, limit - size)); b = a + size
    span = b - a
    if span == size:
        return [a], size
    n = math.ceil((span - size) / (size - ramp)) + 1
    return [a + (i * (span - size)) // (n - 1) for i in range(n)], size


def tiles(region, tile, overlap):
    h, w = region.shape
    ys, xs = np.nonzero(region)
    tw, th = min(tile[0], w), min(tile[1], h)
    rx, ry = math.ceil(tw * overlap), math.ceil(th * overlap)
    require(rx < tw and ry < th, 'Tile overlap must leave a tile interior')
    sx, tw = starts(int(xs.min()) - rx, int(xs.max()) + 1 + rx, tw, w, rx)
    sy, th = starts(int(ys.min()) - ry, int(ys.max()) + 1 + ry, th, h, ry)
    boxes = [[x, y, x + tw, y + th] for y in sy for x in sx if np.any(region[y:y + th, x:x + tw])]
    return boxes, [rx, ry]


def inventory(ids, visible, plants, box):
    """Every plant with at least one visible pixel inside the box, in crop coordinates."""
    x0, y0, x1, y1 = box
    crop = np.where(visible[y0:y1, x0:x1], ids[y0:y1, x0:x1], 0)
    w, h = x1 - x0, y1 - y0
    rows = []
    for p in sorted(plants, key=lambda q: q['id']):
        yy, xx = np.nonzero(crop == p['pass_index'])
        if not len(xx):
            continue
        bx0, by0, bx1, by1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
        rows.append({'id': p['id'], 'species': p['species'], 'visible_pixels': int(len(xx)),
                     'position_percent': [half_up(100 * (bx0 + bx1) / 2 / w), half_up(100 * (by0 + by1) / 2 / h)],
                     'size_px': [bx1 - bx0, by1 - by0],
                     'touches_crop_edge': bool(bx0 == 0 and x0 > 0 or by0 == 0 and y0 > 0 or bx1 == w and x1 < ids.shape[1] or by1 == h and y1 < ids.shape[0])})
    return rows


def check_records(records, species):
    require(isinstance(records, dict) and isinstance(records.get('species'), dict), 'Plant records need a species map')
    for name in species:
        r = records['species'].get(name)
        require(isinstance(r, dict) and r.get('common_name') and r.get('description'),
                'Plant records need a common name and plain habit/leaf description for ' + name)
        require(bool(r.get('flowers')) != bool(r.get('bloom_exempt')),
                'Give the in-bloom flowers, or the recorded bloom exemption reason, for ' + name)


def species_line(name, record_, rows):
    flowers = record_['flowers'] if record_.get('flowers') else 'none required; preserve foliage'
    instances = '; '.join(f"{r['id']} at [{r['position_percent'][0]}, {r['position_percent'][1]}]%, "
                          f"size [{r['size_px'][0]}, {r['size_px'][1]}]px" for r in rows)
    return f"{name} ({record_['common_name']}; {record_['description']}); flowers: {flowers}. Instances: {instances}"


def prompt(job, rows, records, mulch):
    tile = job['kind'] == 'tile'
    text = PREAMBLE.format(part=TILE_PART if tile else '', hardscape=records.get('hardscape') or HARDSCAPE,
                           materials=records.get('materials') or MATERIALS, ground=MULCH if mulch else NO_MULCH,
                           reference=TILE_REFERENCE if tile else '')
    order = []
    for r in rows:
        if r['species'] not in order:
            order.append(r['species'])
    lines = [species_line(s, records['species'][s], [r for r in rows if r['species'] == s]) for s in order]
    if not lines:
        lines = ['No designed plant is visible in this crop; refine only bed, ground and hardscape surfaces.']
    return text + '\n' + '\n'.join(lines) + '\n'


def derive(review_path, records_path, params):
    """Recompute region, jobs, inventories and prompts from verified evidence."""
    from photo_finish import review_context
    review_path, records_path = Path(review_path).resolve(), Path(records_path).resolve()
    review = read(review_path)
    require(review.get('schema_version') == 4 and review.get('method') == 'photo_finish',
            'Scene edit starts from a schema 4 photo_finish review')
    result, finish_plan, ctx, proof = review_context(review, review_path.parent)
    require(result['status'] in ('accepted', 'completed_with_limitations'), 'Previous photo composite has not passed its guard')
    hp, view, _, _, alpha, _, transition, _, layers = ctx
    fp = checked(review['finish_proof'], review_path.parent)
    composite = checked(proof['candidate'], fp.parent)
    require(composite == checked(review['candidate'], review_path.parent), 'Review candidate differs from its finish proof')
    records = read(records_path)
    projected = read(checked(view['projected_plants'], hp.parent))['plants']
    ids = layers['ids']
    visible = alpha > 0
    w, h = view['dimensions']
    require(ids.shape == (h, w), 'Object IDs differ from the view canvas')
    tile, overlap = [int(v) for v in params['tile']], float(params['overlap'])
    margin, feather = int(params['margin_px']), int(params['feather_px'])
    require(len(tile) == 2 and all(v >= 8 for v in tile) and .3 <= overlap < .5, 'Tiles need >= 8 px and 30-50% overlap')
    require(0 <= margin <= 96 and 1 <= feather <= 96, 'Bounded edit-region margin/feather required')
    region = edit_region(alpha, transition, margin)
    boxes, ramps = tiles(region, tile, overlap)
    jobs = [{'id': 'full', 'kind': 'full_view', 'box': [0, 0, w, h]}]
    jobs += [{'id': f'tile-{i + 1:02d}', 'kind': 'tile', 'box': b} for i, b in enumerate(boxes)]
    for job in jobs:
        job['inventory'] = inventory(ids, visible, projected, job['box'])
    check_records(records, sorted({r['species'] for r in jobs[0]['inventory']}))
    for job in jobs:
        job['prompt_text'] = prompt(job, job['inventory'], records, layers['bed_on'])
    return {'review': record(review_path), 'records': record(records_path), 'composite': record(composite),
            'cleaned_photo': record(checked(proof['cleaned_photo'], fp.parent)), 'camera_id': view['camera_id'],
            'handoff': record(hp), 'dimensions': [w, h], 'baseline_status': result['status'],
            'params': {'tile': tile, 'overlap': overlap, 'margin_px': margin, 'feather_px': feather},
            'ramps_px': ramps, 'jobs': jobs, 'baseline_result': result}, region


def prepare(review, records, out, **overrides):
    params = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    derived, region = derive(review, records, params)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new scene edit plan directory'); out.mkdir(parents=True)
    rgb, _ = load(derived['composite']['path'])
    rgb8 = rgb if rgb.dtype == np.uint8 else np.rint(rgb / 257).astype(np.uint8)
    Image.fromarray(region.astype(np.uint8) * 255).save(out / 'region.png')
    Image.fromarray(np.rint(region_weight(region, params['feather_px']) * 255).astype(np.uint8)).save(out / 'region-weight.png')
    jobs = []
    for job in derived['jobs']:
        d = out / 'jobs' / job['id']; d.mkdir(parents=True)
        x0, y0, x1, y1 = job['box']
        save(d / 'input.png', rgb8[y0:y1, x0:x1])
        (d / 'prompt.txt').write_text(job['prompt_text'], encoding='utf-8')
        jobs.append({k: v for k, v in job.items() if k != 'prompt_text'} |
                    {'input': record(d / 'input.png'), 'prompt': record(d / 'prompt.txt')})
    plan = {'schema_version': 1, 'kind': KIND, **{k: v for k, v in derived.items() if k not in ('jobs', 'baseline_result')},
            'required_model': MODEL, 'image_tools': list(IMAGE_TOOLS), 'max_attempts': MAX_ATTEMPTS,
            'region': record(out / 'region.png'), 'region_weight_preview': record(out / 'region-weight.png'), 'jobs': jobs}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def validate(plan, base):
    """Re-derive the frozen plan; returns (plan, region, composite, cleaned photo, previous composite guard result)."""
    require(plan.get('kind') == KIND and plan.get('schema_version') == 1, 'Frozen scene edit plan required')
    require(plan.get('required_model') == MODEL and plan.get('max_attempts') == MAX_ATTEMPTS, 'Scene edit model contract changed')
    derived, region = derive(checked(plan['review'], base), checked(plan['records'], base), plan['params'])
    for key in ('review', 'records', 'composite', 'cleaned_photo', 'camera_id', 'handoff', 'dimensions', 'params', 'ramps_px'):
        require(derived[key] == plan[key], 'Scene edit plan differs from its evidence: ' + key)
    require(len(derived['jobs']) == len(plan['jobs']), 'Scene edit job layout differs')
    with Image.open(checked(plan['region'], base)) as im:
        require(np.array_equal(np.array(im) > 0, region), 'Scene edit region differs')
    composite, _ = load(checked(plan['composite'], base))
    cleaned, _ = load(checked(plan['cleaned_photo'], base))
    rgb8 = composite if composite.dtype == np.uint8 else np.rint(composite / 257).astype(np.uint8)
    for want, job in zip(derived['jobs'], plan['jobs']):
        for key in ('id', 'kind', 'box', 'inventory'):
            require(want[key] == job[key], f"Scene edit job {job.get('id')} differs: {key}")
        prompt_path = checked(job['prompt'], base)
        require(prompt_path.read_text(encoding='utf-8') == want['prompt_text'], f"Planned prompt changed: {job['id']}")
        x0, y0, x1, y1 = job['box']
        target, _ = load(checked(job['input'], base))
        require(np.array_equal(target, rgb8[y0:y1, x0:x1]), f"Edit target differs from the composite crop: {job['id']}")
    return plan, region, composite, cleaned, derived['baseline_result']


def resized(raw_path, box):
    """Return an 8-bit RGB generation resized back to its box; aspect must already match."""
    rgb, alpha = load(raw_path)
    require(alpha is None or np.all(alpha == np.iinfo(alpha.dtype).max), 'Generated edits must be opaque')
    if rgb.dtype != np.uint8:
        rgb = np.rint(rgb / 257).astype(np.uint8)
    w, h = box[2] - box[0], box[3] - box[1]
    rh, rw = rgb.shape[:2]
    require(abs((rw / rh) / (w / h) - 1) <= .03, 'Generated edit aspect ratio differs from its box; never warp to fit')
    if (rw, rh) == (w, h):
        return rgb
    return np.array(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))


def reference(plan_path, full_raw, job_id, out):
    """Crop of the resized full-view result at a tile box: the tile's style/consistency reference."""
    pp = Path(plan_path).resolve(); plan = read(pp)
    full = next(j for j in plan['jobs'] if j['kind'] == 'full_view')
    job = next((j for j in plan['jobs'] if j['id'] == job_id and j['kind'] == 'tile'), None)
    require(job is not None, 'Unknown tile job')
    image = reference_pixels(full_raw, full['box'], job['box'])
    out = Path(out).resolve(); require(not out.exists(), 'Use a new reference path'); out.parent.mkdir(parents=True, exist_ok=True)
    save(out, image)
    return {'job': job_id, 'reference': record(out), 'from_full_raw': record(full_raw)}


def reference_pixels(full_raw, full_box, box):
    rgb = resized(full_raw, full_box)
    x0, y0, x1, y1 = box
    return rgb[y0:y1, x0:x1]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan', help='Freeze jobs, region and prompts for one view')
    a.add_argument('--finish-review', required=True, help='Checked schema 4 photo_finish review of the composite')
    a.add_argument('--records', required=True, help='Plant records JSON: species -> common_name, description, flowers|bloom_exempt')
    a.add_argument('--out', required=True)
    a.add_argument('--tile', type=int, nargs=2, metavar=('W', 'H')); a.add_argument('--overlap', type=float)
    a.add_argument('--margin', type=int); a.add_argument('--feather', type=int)
    a = sub.add_parser('reference', help='Write the full-view result crop for one tile')
    a.add_argument('plan'); a.add_argument('--full-raw', required=True); a.add_argument('--job', required=True); a.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'plan':
        result = prepare(a.finish_review, a.records, a.out, tile=a.tile, overlap=a.overlap, margin_px=a.margin, feather_px=a.feather)
        result = {'plan': str(Path(a.out).resolve() / 'plan.json'), 'jobs': [{'id': j['id'], 'box': j['box'], 'input': j['input']['path'],
                  'prompt': j['prompt']['path'], 'plants': len(j['inventory'])} for j in result['jobs']]}
    else:
        result = reference(a.plan, a.full_raw, a.job, a.out)
    print(json.dumps(result, indent=2))
