"""Compose built-in image-tool scene edits back into the finished photo, and check their review.

Current route (plan schema 3, operation scene_edit_composite_tiles_seam_v3, anchor: none): each tile edited its
native-resolution crop of the previous photo_finish composite. The base of the scene is that composite. Each
selected tile output is checked against the composite crop with a structure check (composite_structure_v1): plant
and bed areas must keep the composite's coarse layout, open bed ground must not gain new structure, and hardscape
away from plants must keep its lines and edges, more strictly. A tile that is not selected or fails the check is not
used; its area keeps the composite pixels. Every pixel belongs to exactly one tile. Between neighbouring tiles the
ownership boundary is a minimum-cost seam through their overlap, placed where the two sources agree (lowest
luminance difference), with only a narrow raised-cosine feather across it, so two tiles never overlap-blend into
doubled leaves. A seam metric compares the neighbouring sources in a band along each boundary and flags likely
ghosting for the reviewer. The scene then replaces the previous composite only inside the landscape edit region,
through a raised-cosine inward feather, in linear light. Outside the region every sample stays exactly the
previous composite, which there is exactly the cleaned original photo; both are asserted.

With anchor: full_view (optional), the base is the selected full-view edit upscaled to the photo, tiles edited
its crops, the alignment check is the anchor edge check (luma_edge_ncc_blocks_v1), and dropped tiles take the
anchor pixels. Schema 2 plans (operation scene_edit_anchor_ownership_v2: anchored, nearest-centre ownership) and
schema 1 plans (scene_edit_raised_cosine_tiles_full_view_linear_v1: wide raised-cosine overlaps) and their reviews
stay verifiable. Visual findings remain the agent's judgments; the metrics inform the review, they do not
replace it.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from handoff import checked, read, record, require
from raster import load, save, scaled_like
from scene_edit_plan import (IMAGE_TOOLS, LENIENT, MAX_ATTEMPTS, MODEL, OPEN_GROUND, STRICT, anchored, reference_pixels,
                             region_weight, resized, structure, validate)

OPERATION_V1 = 'scene_edit_raised_cosine_tiles_full_view_linear_v1'
OPERATION_V2 = 'scene_edit_anchor_ownership_v2'
OPERATION = 'scene_edit_composite_tiles_seam_v3'
OPERATIONS = {1: OPERATION_V1, 2: OPERATION_V2, 3: OPERATION}
# Schema 1 review check set (legacy reviews only).
SCENE_CHECKS_V1 = ('species_fidelity', 'no_invented_or_hybrid_plants', 'plants_rooted_in_beds',
                   'hardscape_geometry_preserved', 'flowering_in_bloom', 'no_visible_seams', 'house_background_retained')
# Current review: only real failures block delivery; everything else is a recorded, non-blocking defect.
CHECK_SET = 'real_failures_v1'
BLOCKING_CHECKS = ('plants_rooted_in_beds', 'hardscape_unchanged', 'no_seams_or_double_images', 'species_form_recognizable')
DEFECT_TYPES = ('movement', 'extent_change', 'extra_foliage', 'invented_plant', 'cultivar_detail', 'bloom_missing',
                'colour_shift', 'other')
# Defects that contradict design intent (an extra plant; a flowering plant out of bloom) downgrade an accepted view
# to completed_with_limitations. The others are recorded only.
DOWNGRADING_DEFECTS = ('invented_plant', 'bloom_missing')
# Tile alignment against the anchor: luminance edge maps (Gaussian sigma 1, Sobel magnitude) compared at the
# anchor's own resolution (never below 64 px on the short side, never above native), by normalized cross-correlation
# over the tile's edit-region pixels, globally and in ~48 px blocks. Calibrated on synthetic leaf scenes at the
# production scale (1152x1024 tiles, anchor 0.256x): photographic re-detailing with 2-6 px silhouette jitter scored
# global 0.71-0.93 with at most 6% low blocks; silhouettes moved 12-30 px scored global 0.20-0.46; one moved ninth
# of a tile (25 px) scored global 0.79 but 11% low blocks.
ALIGNMENT = {'metric': 'luma_edge_ncc_blocks_v1', 'min_side_px': 64, 'blur_sigma': 1.0, 'block_px': 48,
             'global_min': .55, 'block_low': .4, 'max_low_block_fraction': .1, 'flat_std': 1e-3}
# Tile structure check against the previous composite (schema 3, no anchor). A real photographic re-render of the CG
# composite changes leaf and surface texture completely, and plants may move a little and fill out, so the check
# compares coarse luminance edge maps (Gaussian sigma 1.5 after INTER_AREA reduction, Sobel magnitude) per class
# of the plan's structure.png, in blocks of about 128 native px:
# - lenient (plants, and bed within 64 px of a plant) at 1/8 scale (~12 px blur): global NCC >= lenient_global_min.
# - open bed ground (> 64 px from any plant) at 1/8 scale: per block, the edge energy the tile adds where the
#   composite has none (max(tile - dilate(composite, 5 px)), relative to the tile's typical composite block edge
#   energy) <= open_ground_added_max.
# - strict (hardscape and photo pixels > 48 px from any plant) at 1/4 scale (~6 px blur): global NCC >=
#   strict_global_min; every block where the composite has structure keeps NCC >= strict_block_min; added edge
#   energy per block <= strict_added_max.
# Calibration: synthetic 1152x1024 gardens (16 leaf-rosette plants on mulch, a jointed slab and a wall; 8 layouts)
# re-rendered "photographically" (all leaf/mulch/concrete texture replaced, colour shift, every plant moved by up to
# sigma 12 px with 40% more leaves): lenient global 0.76-0.97, open-ground added <= 0.16, strict global >= 0.97,
# strict worst block >= 0.51, strict added <= 0.39, so 0 of 40 rejected. With 2-3 px drift and 0.5% zoom on top, 1 of 8
# is rejected (strict added 0.74); with 4 px drift and 1% zoom (about 10 px misregistration at the tile edge, which
# kinks joints at a seam) 7 of 8 are. Failures: whole-content shift 24-40 px lenient global <= 0.50 and strict global <= 0.35; a different layout
# lenient global <= 0.50; a plant added in open ground added 0.34-0.80; a slab joint moved 24 px strict block
# <= 0.07; the slab edge moved 30 px strict block <= 0.17; a slab joint added strict added 0.49-0.61. A missing,
# swapped or moved single plant inside dense planting is NOT reliably measurable this way: the visual review's
# species check and invented_plant defect cover it.
COMPOSITE_ALIGNMENT = {'metric': 'composite_structure_v1', 'min_side_px': 64, 'blur_sigma': 1.5, 'block_native_px': 128,
                       'lenient_scale': .125, 'strict_scale': .25, 'structured_block': .35, 'residual_dilate_px': 5,
                       'flat_std': 1e-4, 'ref_floor': 1e-3, 'lenient_global_min': .65, 'open_ground_added_max': .3,
                       'strict_global_min': .6, 'strict_block_min': .3, 'strict_added_max': .45}
# Seam/double-image metric: the same edge NCC between the two neighbouring sources in a band of
# band_px_per_feather x seam feather on each side of an ownership boundary. Below flag_below = likely ghosting.
SEAMS = {'metric': 'luma_edge_ncc_band_v1', 'band_px_per_feather': 2, 'flag_below': .4}
# Schema 3: a band whose two sources agree in luminance (mean absolute difference below agree_luma_diff, 8-bit levels)
# cannot hold a double image, so it is never flagged however flat (and so uncorrelated) its edges are.
SEAMS_V3 = {**SEAMS, 'metric': 'luma_edge_ncc_band_v2', 'agree_luma_diff': 2.0}
# Ownership boundaries (schema 3): between two neighbouring tiles (same row or same column), a minimum-cost path
# across their overlap, at least seam_feather_px inside both tiles. Cost is the Gaussian-blurred (cost_sigma_px)
# absolute luminance difference of the two sources inside the edit region (0 outside it), plus a small penalty
# (centre_penalty at the edge of the allowed band) that keeps the seam near the overlap midline where they agree.
OWNERSHIP = {'method': 'min_cost_seam_v1', 'cost_sigma_px': 2.0, 'centre_penalty': .01}
LUMA = np.array([.2126, .7152, .0722], np.float32)
BLOCK = 256

_SRGB8 = np.arange(256, dtype=np.float64) / 255
_LIN8 = np.where(_SRGB8 <= .04045, _SRGB8 / 12.92, ((_SRGB8 + .055) / 1.055) ** 2.4).astype(np.float32)
_SRGB16 = np.arange(65536, dtype=np.float64) / 65535
_LIN16 = np.where(_SRGB16 <= .04045, _SRGB16 / 12.92, ((_SRGB16 + .055) / 1.055) ** 2.4).astype(np.float32)


def linear(rgb):
    return (_LIN8 if rgb.dtype == np.uint8 else _LIN16)[rgb]


def encode16(lin):
    c = np.clip(lin.astype(np.float64), 0, 1)
    return np.rint(np.where(c <= .0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - .055) * 65535).astype(np.uint16)


def raised_cosine(t):
    return (.5 - .5 * np.cos(np.pi * np.clip(t, 0, 1))).astype(np.float32)


def window(start, size, limit, ramp):
    """1D tile weight (schema 1): ramps to zero at tile edges inside the canvas; flat at the canvas border."""
    k = np.arange(size, dtype=np.float32) + .5
    w = np.ones(size, np.float32)
    if start > 0:
        w *= raised_cosine(k / ramp)
    if start + size < limit:
        w *= raised_cosine((size - k) / ramp)
    return w


def text(ref, base):
    return checked(ref, base).read_text(encoding='utf-8')


def normalize(manifest_path):
    """Hash the agent's attempt manifest (plain relative paths) into proof records."""
    mp = Path(manifest_path).resolve(); m = read(mp)
    jobs = []
    for job_id, job in m.get('jobs', {}).items():
        attempts = []
        for n, a in enumerate(job.get('attempts', []), 1):
            row = {'attempt': n, 'raw': record(mp.parent / a['raw']), 'prompt': record(mp.parent / a['prompt']),
                   'inspection': a.get('inspection', '')}
            if a.get('reference'):
                row['reference'] = record(mp.parent / a['reference']); row['reference_from'] = a.get('reference_from')
            if a.get('input'):
                row['input'] = record(mp.parent / a['input']); row['anchor_from'] = a.get('anchor_from')
            if a.get('blocking') is not None:
                row['blocking'] = a['blocking']
            attempts.append(row)
        jobs.append({'id': job_id, 'attempts': attempts, 'selected': job.get('selected')})
    return {'agent_model': m.get('agent_model'), 'reasoning_effort': m.get('reasoning_effort'),
            'image_tool': m.get('image_tool'), 'manifest': record(mp), 'jobs': jobs}


def passes(value):
    return value == 'pass' or isinstance(value, dict) and value.get('result') == 'pass'


def check_attempts(plan, attempts, base):
    """Model, attempt count, exact prompts, tile inputs/references and returned aspect for every planned job."""
    require(attempts.get('agent_model') == MODEL['agent_model'] and attempts.get('reasoning_effort') == MODEL['reasoning_effort'],
            'Scene edits must come from gpt-6-astra with high reasoning')
    require(attempts.get('image_tool') in IMAGE_TOOLS, 'Scene edits must use the built-in image tool')
    schema = plan['schema_version']
    planned = {j['id']: j for j in plan['jobs']}
    rows = {j['id']: j for j in attempts['jobs']}
    require(len(rows) == len(attempts['jobs']) and set(rows) == set(planned), 'Record every planned scene edit job exactly once')
    full = next((j for j in plan['jobs'] if j['kind'] == 'full_view'), None)
    with_anchor = schema >= 2 and anchored(schema, plan['params'])
    anchor = rows['full']['selected'] if full is not None else None
    for job_id, row in rows.items():
        job = planned[job_id]; tries = row['attempts']
        require(len(tries) <= MAX_ATTEMPTS, f'At most {MAX_ATTEMPTS} attempts per scene edit job')
        require([a['attempt'] for a in tries] == list(range(1, len(tries) + 1)), 'Attempts are numbered in order')
        require(row['selected'] is None or row['selected'] in range(1, len(tries) + 1), 'Selected attempt does not exist')
        planned_prompt = text(job['prompt'], base)
        for a in tries:
            used = text(a['prompt'], base)
            if a['attempt'] == 1:
                require(used == planned_prompt, f'Attempt 1 of {job_id} must use the exact planned prompt')
            else:
                require(tries[0].get('inspection'), 'Attempt 2 follows a recorded inspection of attempt 1')
                require(used.startswith(planned_prompt) and len(used[len(planned_prompt):].strip()) >= 20 and
                        used != text(tries[0]['prompt'], base), 'Attempt 2 keeps the planned prompt and adds a correction')
            resized(checked(a['raw'], base), job['box'])
            if schema == 1:
                check_attempt_v1(job, a, rows, full, base)
            elif job['kind'] == 'tile' and not with_anchor:
                require(not a.get('anchor_from'), f'Tile {job_id} edits its composite crop; this plan has no anchor')
                for key in ('input', 'reference'):
                    if a.get(key):
                        got, _ = load(checked(a[key], base)); want, _ = load(checked(job['input'], base))
                        require(np.array_equal(got, want), f'Tile {key} is not the planned composite crop: {job_id}')
            elif job['kind'] == 'tile':
                require(anchor, 'Tiles start only from an accepted full-view anchor; with no anchor, fall back without tiles')
                require(a.get('input') and a.get('anchor_from') == anchor,
                        f'Tile {job_id} must edit the crop of the selected full-view anchor (anchor_from {anchor})')
                ref, _ = load(checked(a['input'], base))
                want = reference_pixels(checked(rows['full']['attempts'][anchor - 1]['raw'], base), full['box'], job['box'])
                require(np.array_equal(ref, want), f'Tile input is not the anchor crop: {job_id}')
            else:
                require(not a.get('reference') and not a.get('input'), 'The full view is edited from its own input only')
                blocking = a.get('blocking')
                require(blocking is None or isinstance(blocking, dict) and set(blocking) == set(BLOCKING_CHECKS) and
                        all(v in ('pass', 'fail') for v in blocking.values()),
                        'Full-view blocking findings need pass or fail for each blocking check')
    if with_anchor and anchor:
        blocking = rows['full']['attempts'][anchor - 1].get('blocking')
        require(blocking is not None, 'Review the full-view anchor against the blocking checks before tiles')
        require(all(v == 'pass' for v in blocking.values()),
                'The full-view anchor failed a blocking check; stop and fall back to the previous composite (no tiles)')
    return rows


def check_attempt_v1(job, a, rows, full, base):
    if job['kind'] == 'tile':
        require(a.get('reference') and a.get('reference_from') in range(1, MAX_ATTEMPTS + 1), 'Tiles take the full-view result crop as reference')
        source = rows['full']['attempts']
        require(a['reference_from'] <= len(source), 'Tile reference names a missing full-view attempt')
        ref, _ = load(checked(a['reference'], base))
        want = reference_pixels(checked(source[a['reference_from'] - 1]['raw'], base), full['box'], job['box'])
        require(np.array_equal(ref, want), f"Tile reference is not the full-view crop: {job['id']}")
    else:
        require(not a.get('reference'), 'The full view is edited from its own input only')


def selected_raw(rows, job_id, base):
    row = rows[job_id]
    return checked(row['attempts'][row['selected'] - 1]['raw'], base) if row['selected'] else None


def finish_block(final, composite, mix, wr, b0, b1):
    block = encode16(mix)
    if composite.dtype != np.uint16:
        block = scaled_like(block, composite)
    keep = wr == 0
    block[keep] = composite[b0:b1][keep]
    final[b0:b1] = block


# ---------------------------------------------------------------- schema 1 (legacy) reconstruction

def reconstruct_v1(plan, region, composite, rows, base):
    h, w = composite.shape[:2]
    weight = region_weight(region, plan['params']['feather_px'])
    full_job = next(j for j in plan['jobs'] if j['kind'] == 'full_view')
    full = None
    if rows['full']['selected']:
        full = resized(selected_raw(rows, 'full', base), full_job['box'])
    rx, ry = plan['ramps_px']
    tiles = []
    for job in plan['jobs']:
        row = rows[job['id']]
        if job['kind'] != 'tile' or not row['selected']:
            continue
        x0, y0, x1, y1 = job['box']
        tiles.append((job['box'], linear(resized(selected_raw(rows, job['id'], base), job['box'])),
                      window(x0, x1 - x0, w, rx), window(y0, y1 - y0, h, ry)))
    final = composite.copy(); coverage = np.zeros((h, w), np.float32)
    for b0 in range(0, h, BLOCK):
        b1 = min(h, b0 + BLOCK)
        wr = weight[b0:b1]
        if not np.any(wr > 0):
            continue
        num = np.zeros((b1 - b0, w, 3), np.float32); den = np.zeros((b1 - b0, w), np.float32)
        for (x0, y0, x1, y1), lin, wx, wy in tiles:
            r0, r1 = max(b0, y0), min(b1, y1)
            if r0 >= r1:
                continue
            tw = wy[r0 - y0:r1 - y0, None] * wx[None, :]
            num[r0 - b0:r1 - b0, x0:x1] += lin[r0 - y0:r1 - y0] * tw[..., None]
            den[r0 - b0:r1 - b0, x0:x1] += tw
        base_lin = linear(composite[b0:b1])
        fill = linear(full[b0:b1]) if full is not None else base_lin
        c = np.minimum(den, 1)[..., None]
        tiled = np.divide(num, den[..., None], out=np.zeros_like(num), where=den[..., None] > 0)
        scene = c * tiled + (1 - c) * fill
        mix = wr[..., None] * scene + (1 - wr[..., None]) * base_lin
        finish_block(final, composite, mix, wr, b0, b1)
        coverage[b0:b1] = np.minimum(den, 1)
    require(np.array_equal(final[~region], composite[~region]), 'Scene edit changed pixels outside the landscape edit region')
    return final, weight, coverage


# ---------------------------------------------------------------- schema 2: anchor, ownership, metrics

def tile_jobs(plan):
    return [j for j in plan['jobs'] if j['kind'] == 'tile']


def ownership(plan, h, w, source=None, region=None, feather=None):
    """Label map: index of the owning planned tile; -1 uncovered.

    Starts from the tile whose centre is nearest among the tiles covering a pixel (schema 2 stops there). With
    `source` (k, box) -> RGB8, each pair of neighbouring tiles then re-splits its overlap along a minimum-cost seam
    (seam_path). Each pixel still has exactly one owner, a tile that covers it."""
    tiles = tile_jobs(plan)
    require(len(tiles) < 255, 'Too many scene edit tiles')
    labels = np.full((h, w), -1, np.int16); best = np.full((h, w), np.inf, np.float32)
    for k, job in enumerate(tiles):
        x0, y0, x1, y1 = job['box']
        cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = (xx - cx) ** 2 + (yy - cy) ** 2
        nearer = d < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][nearer] = d[nearer]
        labels[y0:y1, x0:x1][nearer] = k
    if source is None:
        return labels
    for a, b, axis in neighbours(tiles):
        ax0, ay0, ax1, ay1 = tiles[a]['box']; bx0, by0, bx1, by1 = tiles[b]['box']
        box = (max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1))
        x0, y0, x1, y1 = box
        cost = np.abs(luma(source(a, box)) - luma(source(b, box)))
        cost = cv2.GaussianBlur(cost, (0, 0), OWNERSHIP['cost_sigma_px']) * region[y0:y1, x0:x1]
        if axis == 0:  # top/bottom neighbours: a horizontal seam, solved on the transpose
            cost = cost.T
        cut = seam_path(cost, feather)
        if cut is None:
            continue
        across = np.arange(cost.shape[1])[None, :] < cut[:, None]  # True on a's side (left, or above)
        if axis == 0:
            across = across.T
        window = labels[y0:y1, x0:x1]
        mine = (window == a) | (window == b)
        window[mine] = np.where(across, a, b)[mine]
    return labels


def neighbours(tiles):
    """Pairs (a, b, axis) of consecutive overlapping tiles in one row (axis 1, a left of b) or column (axis 0, a above
    b). Row pairs come first, so column seams have the last word where four tiles meet."""
    rows, cols = {}, {}
    for k, t in enumerate(tiles):
        x0, y0, x1, y1 = t['box']
        rows.setdefault((y0, y1), []).append((x0, k)); cols.setdefault((x0, x1), []).append((y0, k))
    out = []
    for axis, groups in ((1, rows), (0, cols)):
        for members in groups.values():
            members.sort()
            for (_, a), (_, b) in zip(members, members[1:]):
                ab, bb = tiles[a]['box'], tiles[b]['box']
                if (bb[0] < ab[2]) if axis == 1 else (bb[1] < ab[3]):
                    out.append((a, b, axis))
    return out


def seam_path(cost, margin):
    """Minimum-cost top-to-bottom path through cost (rows x columns), staying `margin` columns inside both sides.
    Returns the cut column per row (columns < cut belong to the left source), or None when the overlap is too narrow."""
    h, w = cost.shape
    lo, hi = margin, w - margin
    if hi - lo < 1:
        return None
    mid = w / 2; half = max(1., (hi - lo) / 2)  # a cut at column c is the boundary left of pixel c
    c = cost[:, lo:hi].astype(np.float64) + OWNERSHIP['centre_penalty'] * np.abs(np.arange(lo, hi) - mid)[None, :] / half
    acc = c.copy(); step = np.zeros(c.shape, np.int8)
    for y in range(1, h):
        prev = acc[y - 1]
        left = np.concatenate([[np.inf], prev[:-1]]); right = np.concatenate([prev[1:], [np.inf]])
        options = np.stack([left, prev, right])
        pick = np.argmin(options, axis=0)
        step[y] = pick - 1
        acc[y] += options[pick, np.arange(len(prev))]
    path = np.empty(h, np.int64); path[-1] = int(np.argmin(acc[-1]))
    for y in range(h - 1, 0, -1):
        path[y - 1] = path[y] + step[y, path[y]]
    return path + lo


def label_weights(labels, plan, feather):
    """Per label: (window, weight) with a raised-cosine ramp `feather` px wide centred on its ownership boundary.

    Two labels meeting along a boundary get complementary weights (they sum to 1), so each pixel's content comes
    from one tile except inside that narrow band."""
    h, w = labels.shape
    tiles = tile_jobs(plan); out = []
    for k in sorted(int(v) for v in np.unique(labels)):
        mask = labels == k
        ys, xs = np.nonzero(mask)
        y0, y1 = max(0, int(ys.min()) - feather), min(h, int(ys.max()) + 1 + feather)
        x0, x1 = max(0, int(xs.min()) - feather), min(w, int(xs.max()) + 1 + feather)
        m = mask[y0:y1, x0:x1].astype(np.uint8)
        inside = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        signed = np.where(m > 0, inside - .5, .5 - outside)
        weight = raised_cosine(.5 + signed / feather)
        if k >= 0:  # a tile has pixels only inside its own box
            bx0, by0, bx1, by1 = tiles[k]['box']
            cx0, cy0, cx1, cy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
            weight = weight[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            x0, y0, x1, y1 = cx0, cy0, cx1, cy1
        out.append((k, (x0, y0, x1, y1), weight))
    return out


def luma(rgb8):
    return (rgb8.astype(np.float32) / 255) @ LUMA


def edges(y, sigma):
    y = cv2.GaussianBlur(y, (0, 0), sigma)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def ncc(a, b, mask, flat):
    """Zero-mean normalized cross-correlation over mask; both flat = 1 (nothing to disagree on), one flat = 0."""
    if int(mask.sum()) < 4:
        return None
    a = a[mask].astype(np.float64); b = b[mask].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    sa, sb = a.std(), b.std()
    if sa < flat and sb < flat:
        return 1.0
    if sa < flat or sb < flat:
        return 0.0
    return float(np.mean(a * b) / (sa * sb))


def analysis_scale(shape, anchor_scale):
    h, w = shape[:2]
    return min(1.0, max(anchor_scale, ALIGNMENT['min_side_px'] / max(1, min(h, w))))


def edge_maps(images, mask, scale, sigma=None):
    h, w = mask.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    sigma = ALIGNMENT['blur_sigma'] if sigma is None else sigma
    maps = [edges(cv2.resize(luma(im), size, interpolation=cv2.INTER_AREA), sigma) for im in images]
    m = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= .5
    return maps, m


def r4(v):
    return None if v is None else round(float(v), 4)


def alignment(tile8, anchor8, mask, anchor_scale):
    """Does the tile keep the anchor's silhouettes? Edge NCC at anchor resolution, global and per block."""
    a = analysis_scale(mask.shape, anchor_scale)
    (t, an), m = edge_maps([tile8, anchor8], mask, a)
    flat = ALIGNMENT['flat_std']
    g = r4(ncc(an, t, m, flat))
    h, w = m.shape; bp = ALIGNMENT['block_px']
    ny, nx = max(1, round(h / bp)), max(1, round(w / bp))
    blocks = []
    for i in range(ny):
        for j in range(nx):
            ys, xs = slice(i * h // ny, (i + 1) * h // ny), slice(j * w // nx, (j + 1) * w // nx)
            bm = m[ys, xs]
            if bm.sum() < max(4, .25 * bm.size) or an[ys, xs][bm].std() < flat:
                continue  # mostly outside the region, or no anchor structure to keep
            blocks.append(ncc(an[ys, xs], t[ys, xs], bm, flat))
    low = r4(np.mean([v < ALIGNMENT['block_low'] for v in blocks])) if blocks else 0.0
    aligned = g is None or (g >= ALIGNMENT['global_min'] and low <= ALIGNMENT['max_low_block_fraction'])
    return {'analysis_scale': r4(a), 'global_ncc': g, 'blocks': len(blocks), 'low_block_fraction': low,
            'worst_block_ncc': r4(min(blocks)) if blocks else None, 'aligned': bool(aligned)}


def class_scale(shape, scale):
    h, w = shape[:2]
    return min(1.0, max(scale, COMPOSITE_ALIGNMENT['min_side_px'] / max(1, min(h, w))))


def blocks_of(shape, bp):
    h, w = shape; ny, nx = max(1, round(h / bp)), max(1, round(w / bp))
    for i in range(ny):
        for j in range(nx):
            yield slice(i * h // ny, (i + 1) * h // ny), slice(j * w // nx, (j + 1) * w // nx)


def structure_rows(tile8, comp8, region, classes, scale):
    """Per-class global edge NCC and per-block (NCC where the composite has structure, added edge energy) at one scale."""
    cfg = COMPOSITE_ALIGNMENT; flat = cfg['flat_std']
    s = class_scale(region.shape, scale)
    (et, ec), m = edge_maps([tile8, comp8], region, s, cfg['blur_sigma'])
    size = m.shape[::-1]; bp = max(2, round(cfg['block_native_px'] * s))
    grid = list(blocks_of(m.shape, bp))
    means = [float(ec[ys, xs][m[ys, xs]].mean()) for ys, xs in grid if m[ys, xs].mean() >= .25]
    ref = max(float(np.percentile(means, 75)) if means else 0., cfg['ref_floor'])
    k = cfg['residual_dilate_px']
    added_map = np.maximum(et - cv2.dilate(ec, np.ones((k, k), np.uint8)), 0)
    out = {}
    for name, cls in classes.items():
        c = (cv2.resize(cls.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= .99) & m
        rows = []
        for ys, xs in grid:
            bm = c[ys, xs]
            if bm.sum() < max(4, .25 * bm.size):
                continue
            structured = float(ec[ys, xs][bm].mean()) / ref >= cfg['structured_block']
            rows.append({'ncc': ncc(ec[ys, xs], et[ys, xs], bm, flat) if structured else None,
                         'added': float(added_map[ys, xs][bm].mean()) / ref})
        out[name] = {'analysis_scale': r4(s), 'global_ncc': r4(ncc(ec, et, c, flat)), 'rows': rows}
    return out


def composite_alignment(tile8, comp8, classes):
    """Does the tile keep the composite's structure? classes: the plan's structure.png crop (0 outside the region)."""
    cfg = COMPOSITE_ALIGNMENT
    region = classes > 0
    lenient = structure_rows(tile8, comp8, region, {'lenient': (classes == LENIENT) | (classes == OPEN_GROUND),
                                                    'open_ground': classes == OPEN_GROUND}, cfg['lenient_scale'])
    strict = structure_rows(tile8, comp8, region, {'strict': classes == STRICT}, cfg['strict_scale'])['strict']
    worst = [r['ncc'] for r in strict['rows'] if r['ncc'] is not None]
    peak = lambda rows: r4(max([r['added'] for r in rows] or [0.]))
    result = {'lenient': {'analysis_scale': lenient['lenient']['analysis_scale'], 'global_ncc': lenient['lenient']['global_ncc']},
              'open_ground': {'blocks': len(lenient['open_ground']['rows']), 'max_added': peak(lenient['open_ground']['rows'])},
              'strict': {'analysis_scale': strict['analysis_scale'], 'global_ncc': strict['global_ncc'],
                         'structured_blocks': len(worst), 'worst_block_ncc': r4(min(worst)) if worst else None,
                         'max_added': peak(strict['rows'])}}
    failed = []
    g = result['lenient']['global_ncc']
    if g is not None and g < cfg['lenient_global_min']:
        failed.append('lenient_global_ncc')
    if result['open_ground']['max_added'] > cfg['open_ground_added_max']:
        failed.append('open_ground_added_structure')
    g = result['strict']['global_ncc']
    if g is not None and g < cfg['strict_global_min']:
        failed.append('strict_global_ncc')
    if worst and min(worst) < cfg['strict_block_min']:
        failed.append('strict_block_ncc')
    if result['strict']['max_added'] > cfg['strict_added_max']:
        failed.append('strict_added_structure')
    return {**result, 'failed': failed, 'aligned': not failed}


def seam_rows(labels, plan, region, source, used, anchor_scale, feather, base_name='anchor', config=SEAMS):
    """Compare the two neighbouring sources in a band along every tile-tile ownership boundary."""
    tiles = tile_jobs(plan); band = SEAMS['band_px_per_feather'] * feather; rows = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            ax0, ay0, ax1, ay1 = tiles[i]['box']; bx0, by0, bx1, by1 = tiles[j]['box']
            x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
            if x0 >= x1 or y0 >= y1 or not (used[i] or used[j]):
                continue
            lw = labels[y0:y1, x0:x1]; mi, mj = lw == i, lw == j
            if not (mi.any() and mj.any()):
                continue
            di = cv2.distanceTransform((~mi).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            dj = cv2.distanceTransform((~mj).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            b = ((mi & (dj <= band)) | (mj & (di <= band))) & region[y0:y1, x0:x1]
            if not b.any():
                continue
            ys, xs = np.nonzero(b)
            cy0, cy1, cx0, cx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            box = (x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1)
            crops = [source(k, box) for k in (i, j)]
            bm = b[cy0:cy1, cx0:cx1]
            (ei, ej), m = edge_maps(crops, bm, analysis_scale(bm.shape, anchor_scale))
            score = r4(ncc(ei, ej, m, ALIGNMENT['flat_std']))
            diff = np.abs(luma(crops[0]) - luma(crops[1]))[bm]
            mad = r4(255 * float(diff.mean()))
            rows.append({'pair': [tiles[i]['id'], tiles[j]['id']],
                         'sources': ['tile' if used[i] else base_name, 'tile' if used[j] else base_name],
                         'bbox': list(box), 'band_pixels': int(b.sum()),
                         'edge_ncc': score, 'mean_abs_luma_diff': mad,
                         'likely_ghosting': bool(score is not None and score < config['flag_below'] and
                                                 mad >= config.get('agree_luma_diff', 0))})
    return rows


def raw_scale(raw_path, box):
    with Image.open(raw_path) as im:
        return im.width / (box[2] - box[0])


def reconstruct_tiles(plan, region, composite, rows, base):
    """Schema 2 and 3: base image (the composite, or the upscaled anchor), checked tiles, one owner per pixel.
    ref8 is the 8-bit base the tiles are checked and seamed against."""
    h, w = composite.shape[:2]
    schema = plan['schema_version']
    weight = region_weight(region, plan['params']['feather_px'])
    feather = plan['params']['seam_feather_px']
    with_anchor = anchored(schema, plan['params'])
    if with_anchor:
        full_job = next(j for j in plan['jobs'] if j['kind'] == 'full_view')
        full_raw = selected_raw(rows, 'full', base)
        require(full_raw is not None, 'The scene edit needs a selected full-view anchor; otherwise fall back without a proof')
        ref8 = resized(full_raw, full_job['box'])
        scale = raw_scale(full_raw, full_job['box'])
        check = lambda out, box: alignment(out, ref8[box[1]:box[3], box[0]:box[2]], region[box[1]:box[3], box[0]:box[2]], scale)
        base_info, base_name, config = ({'job': 'full', 'attempt': rows['full']['selected'], 'scale': r4(scale)}, 'anchor', ALIGNMENT)
        fill = ref8
    else:
        ref8 = to8(composite)
        classes = structure(plan, base)
        scale = COMPOSITE_ALIGNMENT['strict_scale']  # seam band analysis scale
        check = lambda out, box: composite_alignment(out, ref8[box[1]:box[3], box[0]:box[2]], classes[box[1]:box[3], box[0]:box[2]])
        base_info, base_name, config = ({'source': 'previous_composite'}, 'composite', COMPOSITE_ALIGNMENT)
        fill = composite  # dropped tiles keep the composite at full precision; ref8 is only compared
    tiles = tile_jobs(plan)
    outputs, used, aligned = {}, [], []
    for k, job in enumerate(tiles):
        raw = selected_raw(rows, job['id'], base)
        row = {'id': job['id'], 'selected': rows[job['id']]['selected']}
        if raw is not None:
            out = resized(raw, job['box'])
            row.update(check(out, job['box']))
            if row['aligned']:
                outputs[k] = out
        row['used'] = k in outputs
        used.append(row['used']); aligned.append(row)

    def source(k, box):
        x0, y0, x1, y1 = box
        if k in outputs:
            tx, ty = tiles[k]['box'][:2]
            return outputs[k][y0 - ty:y1 - ty, x0 - tx:x1 - tx]
        return ref8[y0:y1, x0:x1]

    labels = ownership(plan, h, w) if schema == 2 else ownership(plan, h, w, source, region, feather)
    seam_config = SEAMS if schema == 2 else SEAMS_V3
    seams = seam_rows(labels, plan, region, source, used, scale, feather, base_name, seam_config)
    weights = label_weights(labels, plan, feather)
    final = composite.copy()
    for b0 in range(0, h, BLOCK):
        b1 = min(h, b0 + BLOCK)
        wr = weight[b0:b1]
        if not np.any(wr > 0):
            continue
        fill_lin = linear(fill[b0:b1])
        num = np.zeros_like(fill_lin); den = np.zeros((b1 - b0, w), np.float32)
        for k, (x0, y0, x1, y1), lw in weights:
            r0, r1 = max(b0, y0), min(b1, y1)
            if r0 >= r1:
                continue
            wk = lw[r0 - y0:r1 - y0]
            if k in outputs:
                tx, ty = tiles[k]['box'][:2]
                src = linear(outputs[k][r0 - ty:r1 - ty, x0 - tx:x1 - tx])
            else:
                src = fill_lin[r0 - b0:r1 - b0, x0:x1]
            num[r0 - b0:r1 - b0, x0:x1] += src * wk[..., None]
            den[r0 - b0:r1 - b0, x0:x1] += wk
        scene = np.divide(num, den[..., None], out=fill_lin.copy(), where=den[..., None] > 0)
        base_lin = linear(composite[b0:b1])
        mix = wr[..., None] * scene + (1 - wr[..., None]) * base_lin
        finish_block(final, composite, mix, wr, b0, b1)
    require(np.array_equal(final[~region], composite[~region]), 'Scene edit changed pixels outside the landscape edit region')
    metrics = {'alignment': {**config, 'tiles': aligned},
               'seams': {**seam_config, 'seam_feather_px': feather, 'pairs': seams,
                         'flagged': sum(r['likely_ghosting'] for r in seams)},
               'used_tiles': [t['id'] for k, t in enumerate(tiles) if used[k]]}
    if schema == 2:
        metrics['anchor'] = base_info
    else:
        metrics |= {'base': base_info if not with_anchor else {'source': 'anchor', **base_info},
                    'ownership_boundary': {**OWNERSHIP, 'margin_px': feather}}
    return final, weight, (labels + 1).astype(np.uint8), metrics, ref8


def outside_is_cleaned_photo(region, composite, cleaned):
    require(np.array_equal(composite[~region], scaled_like(cleaned, composite)[~region]),
            'Previous composite differs from the cleaned photo outside the edit region')


def to8(a):
    return a if a.dtype == np.uint8 else np.rint(a / 257).astype(np.uint8)


def proof_panel(job, rows, composite, final, base, anchor=None):
    """input | resized output (grey when unselected) | final crop. Tile inputs are the composite crop, or the anchor
    crop in the anchored route."""
    x0, y0, x1, y1 = job['box']
    row = rows[job['id']]
    gen = (resized(checked(row['attempts'][row['selected'] - 1]['raw'], base), job['box']) if row['selected']
           else np.full((y1 - y0, x1 - x0, 3), 128, np.uint8))
    left = anchor[y0:y1, x0:x1] if anchor is not None and job['kind'] == 'tile' else to8(composite[y0:y1, x0:x1])
    panel = np.concatenate([left, gen, to8(final[y0:y1, x0:x1])], axis=1)
    image = Image.fromarray(panel)
    if image.height > 1024:
        image = image.resize((round(image.width * 1024 / image.height), 1024), Image.LANCZOS)
    return image


def compose(plan_path, manifest, out):
    pp = Path(plan_path).resolve(); plan = read(pp)
    plan, region, composite, cleaned, _ = validate(plan, pp.parent)
    outside_is_cleaned_photo(region, composite, cleaned)
    attempts = normalize(manifest)
    rows = check_attempts(plan, attempts, pp.parent)
    require(any(r['selected'] for r in rows.values()), 'Select at least one scene edit, or record a fallback review instead')
    v2 = plan['schema_version'] >= 2
    anchor = None
    if v2:
        final, weight, labels, metrics, base_image = reconstruct_tiles(plan, region, composite, rows, pp.parent)
        anchor = base_image if anchored(plan['schema_version'], plan['params']) else None
    else:
        final, weight, coverage = reconstruct_v1(plan, region, composite, rows, pp.parent)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new scene edit output directory'); (out / 'proofs').mkdir(parents=True)
    save(out / 'candidate.png', final)
    Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(out / 'region-weight.png')
    proofs = {}
    for job in plan['jobs']:
        proof_panel(job, rows, composite, final, pp.parent, anchor).save(out / 'proofs' / f"{job['id']}.png")
        proofs[job['id']] = record(out / 'proofs' / f"{job['id']}.png")
    proof = {'kind': 'scene_edit_proof', 'operation': OPERATIONS[plan['schema_version']], 'plan': record(pp), 'attempts': attempts,
             'candidate': record(out / 'candidate.png'), 'region_weight': record(out / 'region-weight.png')}
    if v2:
        Image.fromarray(labels).save(out / 'ownership.png')
        proof |= {'ownership': record(out / 'ownership.png'), **metrics}
    else:
        Image.fromarray(np.rint(coverage * 255).astype(np.uint8)).save(out / 'tile-coverage.png')
        proof['tile_coverage'] = record(out / 'tile-coverage.png')
    proof |= {'job_proofs': proofs, 'region_pixels': int(region.sum()), 'outside_region_pixels_preserved': int((~region).sum())}
    (out / 'scene-edit-proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def close(a, b, tol=2e-3):
    """Recomputed metrics equal the recorded ones (floats within tol, everything else exactly)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol
    return a == b


def verify(proof, base):
    require(proof.get('kind') == 'scene_edit_proof' and proof.get('operation') in OPERATIONS.values(), 'Scene edit proof required')
    pp = checked(proof['plan'], base)
    plan, region, composite, cleaned, baseline = validate(read(pp), pp.parent)
    require(proof['operation'] == OPERATIONS[plan['schema_version']], 'Scene edit proof operation does not match its plan schema')
    v2 = plan['schema_version'] >= 2
    outside_is_cleaned_photo(region, composite, cleaned)
    attempts = proof['attempts']
    checked(attempts['manifest'], base)
    rows = check_attempts(plan, attempts, pp.parent)
    if v2:
        final, weight, labels, metrics, _ = reconstruct_tiles(plan, region, composite, rows, pp.parent)
        maps = (('region_weight', np.rint(weight * 255)), ('ownership', labels))
        require(close(metrics, {k: proof.get(k) for k in metrics}), 'Scene edit alignment/seam metrics differ')
    else:
        final, weight, coverage = reconstruct_v1(plan, region, composite, rows, pp.parent)
        maps = (('region_weight', np.rint(weight * 255)), ('tile_coverage', np.rint(coverage * 255)))
    actual, alpha = load(checked(proof['candidate'], base))
    require(alpha is None and np.array_equal(actual, final), 'Scene edit reconstruction differs')
    for key, want in maps:
        with Image.open(checked(proof[key], base)) as im:
            require(np.array_equal(np.array(im), want.astype(np.uint8)), f'Scene edit {key} differs')
    require(set(proof['job_proofs']) == {j['id'] for j in plan['jobs']}, 'Every scene edit job needs a proof image')
    for ref in proof['job_proofs'].values():
        checked(ref, base)
    require(proof.get('outside_region_pixels_preserved') == int((~region).sum()), 'Recompute preserved pixel count')
    return plan, rows, baseline


def job_rows(vision, rows, base):
    out = vision.get('jobs', [])
    require(len(out) == len(rows) and {r.get('id') for r in out} == set(rows), 'Review every scene edit job exactly once')
    from botanical_refinement import images
    for r in out:
        want = rows[r['id']]
        require(r.get('attempts') == len(want['attempts']) and r.get('selected_attempt') == want['selected'] and r.get('observations'),
                'Job review must match its recorded attempts and selection')
        if want['selected']:
            images([r.get('evidence')], base)
    return out


def scene_findings_v1(vision, rows, fallback, base):
    checks = vision.get('checks', {})
    require(all(checks.get(k) in ('pass', 'fail', 'estimated') for k in SCENE_CHECKS_V1), 'Every scene edit check needs an actual disposition')
    nonpassing = [k for k in SCENE_CHECKS_V1 if checks[k] != 'pass']
    for r in job_rows(vision, rows, base):
        if rows[r['id']]['selected']:
            require(r.get('species_fidelity') in ('pass', 'fail') and r.get('invented_or_hybrid_plants') in ('none', 'present'),
                    'Selected job needs species fidelity and invented/hybrid plant findings')
            if r['species_fidelity'] != 'pass' or r['invented_or_hybrid_plants'] != 'none':
                require(fallback, f"Job {r['id']} fails species fidelity; deselect it and recompose, or fall back")
                if 'species_fidelity' not in nonpassing:
                    nonpassing.append('species_fidelity')
    return nonpassing


def scene_findings(vision, rows, proof, fallback, base):
    """Blocking checks (real failures) on the whole delivered view, plus the required non-blocking defect list."""
    require(vision.get('check_set') == CHECK_SET, f'Scene edit reviews use check_set {CHECK_SET}')
    checks = vision.get('checks', {})
    require(isinstance(checks, dict) and set(checks) == set(BLOCKING_CHECKS),
            'Record exactly the blocking checks: ' + ', '.join(BLOCKING_CHECKS))
    failing = []
    for k in BLOCKING_CHECKS:
        v = checks[k]
        if passes(v):
            continue
        require(isinstance(v, dict) and v.get('result') == 'fail' and v.get('where') and v.get('observation'),
                f'Blocking check {k} is "pass", or {{"result": "fail", "where", "observation"}} naming where it fails')
        failing.append(k)
    defects = vision.get('defects')
    require(isinstance(defects, list), 'Record the non-blocking defects list (empty when there are none)')
    for d in defects:
        require(isinstance(d, dict) and d.get('type') in DEFECT_TYPES and d.get('where') and d.get('observation'),
                'Each defect needs a type (' + ', '.join(DEFECT_TYPES) + '), where and observation')
    require(vision.get('defect_count') == len(defects), 'defect_count must equal the number of recorded defects')
    if proof['seams']['flagged']:
        require(vision.get('metrics_inspected') is True,
                'Inspect the flagged seam bands (scene-edit-proof seams.pairs) at native size and set metrics_inspected')
    for r in job_rows(vision, rows, base):
        if not rows[r['id']]['selected']:
            continue
        require(r.get('blocking') in ('pass', 'fail'), f"Selected job {r['id']} needs its blocking finding (pass or fail)")
        if fallback:
            continue  # the view is not delivered; its job findings are kept as evidence
        if r['id'] == 'full':
            require(r['blocking'] == 'pass', 'The full-view anchor fails a blocking check: fall back to the previous composite')
        else:
            require(r['blocking'] == 'pass', f"Tile {r['id']} fails a blocking check: deselect it (its area keeps the "
                                             'previous composite, or the anchor in the anchored route) and recompose')
    return failing, defects


def defect_rows(defects):
    return [{'check': f"defect: {d['type']}", 'observations': f"{d['where']}: {d['observation']}", 'blocking': False}
            for d in defects or []]


def check_review(data, base):
    """Schema 4 scene_photographic_edit review: delivered edit, or explicit fallback to the previous composite."""
    from photo_finish import check_review as check_finish
    brp = checked(data['baseline_review'], base); previous = read(brp)
    require(previous.get('schema_version') == 4 and previous.get('method') == 'photo_finish', 'Previous composite needs its schema 4 photo_finish review')
    require(checked(data['handoff'], base) == checked(previous['handoff'], brp.parent) and data['camera_id'] == previous['camera_id'],
            'Scene edit review camera/handoff differs from the previous composite')
    vision = data.get('scene_review', {})
    fallback = data.get('release_policy') == 'fallback_previous_composite'
    require(data.get('release_policy') in (None, 'fallback_previous_composite'), 'Unknown scene edit release policy')
    limitations = data.get('limitations', [])
    require(all(r.get('check') and r.get('observations') for r in limitations), 'Each limitation needs its check and observations')
    defects, proof = None, None
    if not data.get('scene_edit_proof'):
        require(fallback and limitations, 'Without a scene edit proof only a disclosed fallback is possible')
        baseline = check_finish(previous, brp.parent); rows = {}; nonpassing = [r['check'] for r in limitations]
    else:
        sp = checked(data['scene_edit_proof'], base); proof = read(sp)
        plan, rows, baseline = verify(proof, sp.parent)
        require(checked(plan['review'], sp.parent) == brp, 'Scene edit plan was made from a different previous composite')
        model = data.get('model', {})
        require(model.get('agent_model') == proof['attempts']['agent_model'] == MODEL['agent_model'] and
                model.get('reasoning_effort') == proof['attempts']['reasoning_effort'] == MODEL['reasoning_effort'] and
                model.get('image_tool') == proof['attempts']['image_tool'], 'Record the gpt-6-astra/high model and image tool used')
        require(vision.get('reviewer') == 'vision-capable-agent' and vision.get('source_and_final_loaded') is True and
                vision.get('native_size_crops_loaded') is True and vision.get('observations'), 'Actual scene edit review of source, final and native-size crops required')
        from botanical_refinement import images
        images(vision.get('paired_evidence'), base)
        if proof['operation'] in (OPERATION, OPERATION_V2):
            nonpassing, defects = scene_findings(vision, rows, proof, fallback, base)
        else:
            require(vision.get('check_set') is None, 'Schema 1 scene edit proofs keep their original review checks')
            nonpassing = scene_findings_v1(vision, rows, fallback, base)
        if fallback:
            require(nonpassing, 'A fallback needs the failed scene edit findings')
            require({r['check'] for r in limitations} == set(nonpassing) and len(limitations) == len(nonpassing),
                    'Describe each nonpassing scene edit check')
        else:
            require(not nonpassing, 'Scene edit review failed: fall back to the previous composite with disclosed limitations')
            require(checked(data['candidate'], base) == checked(proof['candidate'], sp.parent), 'Scene edit review candidate differs from its proof')
    if fallback:
        require(checked(data['candidate'], base) == checked(previous['candidate'], brp.parent),
                'A fallback delivers the previous photo_finish composite unchanged')
    observed = defect_rows(defects)
    if fallback:
        status, carried = 'completed_with_limitations', limitations
    else:
        downgrade = any(d['type'] in DOWNGRADING_DEFECTS for d in defects or [])
        status = 'completed_with_limitations' if downgrade else baseline['status']
        carried = baseline.get('limitations', []) + (observed if status == 'completed_with_limitations' else [])
    result = {'status': status, 'method': 'scene_photographic_edit', 'camera_id': data['camera_id'],
              'delivered': 'previous_photo_finish_composite' if fallback else 'scene_edit_composite',
              'scene_edit_status': 'rejected' if fallback else 'accepted', 'nonpassing_checks': nonpassing if fallback else [],
              'limitations': carried,
              'previous_composite_status': baseline['status'], 'previous_composite_limitations': baseline.get('limitations', []),
              'model': data.get('model'), 'attempts': {k: len(v['attempts']) for k, v in rows.items()},
              'selected': {k: v['selected'] for k, v in rows.items()},
              'photo_background_composited': True, 'bed_ground': baseline.get('bed_ground'),
              'landscape_evidence_counts': baseline.get('landscape_evidence_counts'),
              'upstream_handoff_kind': baseline.get('upstream_handoff_kind'), 'upstream_certification': baseline.get('upstream_certification'),
              'scope': 'Built-in image-tool photographic edit of the landscape region of a checked photo composite',
              'limitation': 'Generated plant pixels are agent-reviewed derivatives; the native scene stays the placement and species authority'}
    if defects is not None:
        result |= {'check_set': CHECK_SET,
                   'blocking_checks': {k: 'fail' if k in nonpassing else 'pass' for k in BLOCKING_CHECKS},
                   'defects': defects, 'defect_count': len(defects), 'non_blocking_observations': observed,
                   'used_tiles': proof['used_tiles'],
                   'tile_alignment': {t['id']: {k: t.get(k) for k in ('aligned', 'used', 'failed', 'global_ncc', 'low_block_fraction')
                                                if k in t} for t in proof['alignment']['tiles']},
                   'seams_flagged': proof['seams']['flagged']}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('compose'); a.add_argument('plan'); a.add_argument('--attempts', required=True, help='Attempt manifest JSON')
    a.add_argument('--out', required=True)
    a = sub.add_parser('verify'); a.add_argument('proof'); a = p.parse_args()
    if a.command == 'compose':
        r = compose(a.plan, a.attempts, a.out)
        result = {'candidate': r['candidate'], 'proof': str(Path(a.out).resolve() / 'scene-edit-proof.json'), 'region_pixels': r['region_pixels']}
        if r['operation'] != OPERATION_V1:
            result |= {'used_tiles': r['used_tiles'],
                       'rejected_by_alignment': [t['id'] for t in r['alignment']['tiles'] if t['selected'] and not t['used']],
                       'seams_flagged': r['seams']['flagged']}
    else:
        path = Path(a.proof).resolve(); plan, rows, _ = verify(read(path), path.parent)
        result = {'status': 'verified', 'jobs': len(plan['jobs']), 'selected': {k: v['selected'] for k, v in rows.items()}}
    print(json.dumps(result, indent=2))
