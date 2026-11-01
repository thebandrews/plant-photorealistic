"""Freeze a scene photographic edit plan for one finished photo view, BEFORE generation.

Input is a checked schema 4 photo_finish review (the composite that would otherwise be delivered). The plan
derives the landscape edit region from that composite's renderer matte and transition permissions and freezes ONE
full-view job: the whole composite, downscaled to fit the image tool's largest size (`input_size`, 1536x1024 for a
landscape view; never upscaled), with one prompt. The prompt lists every designed species with its plain-words
records, its in-bloom flowers, how many plants of it are visible and where in the view (a 3x3 grid: left, centre or
right; background, middle ground or foreground), computed from the actual renderer object IDs, and the numbers of its
plants on a label map: the reduced composite, dimmed and desaturated, with every plant of at least `label_min_px`
visible native pixels thinly outlined and numbered at its visible centroid. One labelled sheet of real reference
photographs covers the view's species (largest first, at most 16, long edge at most 2048 px). Images go to the tool in
this order: the composite (the edit target), the label map, the reference sheet.
Deterministic: validate() re-derives every field. No image generation or recognition happens here.

`retry_prompt` builds attempt 2's prompt from the review notes of attempt 1: the planned prompt plus a frozen
correction block, one `Fix:` line per real-failure concern, missing bloom, invented plant or species-form drift.

Plan schema 5 (operation scene_edit_full_view_v1) is the only plan this version writes or verifies. The tiled routes
of schemas 1-4 (lab P1-P4) were removed: they delivered patchwork views that mixed photographic and rendered tiles.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from handoff import checked, read, record, require
from raster import load, save

KIND = 'scene_photographic_edit_plan'
MODEL = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high'}
IMAGE_TOOLS = ('image_gen.imagegen', 'image_gen__imagegen')
MAX_ATTEMPTS = 2
PLAN_SCHEMA = 5
JOB_ID = 'full'
# input_size: the image tool's largest size for the view's orientation (landscape W x H; swapped for portrait views).
DEFAULTS = {'margin_px': 24, 'feather_px': 24, 'reference_min_px': 48, 'label_min_px': 400, 'input_size': [1536, 1024]}
# Reference photos: a species gets one when an instance has a visible bounding box at least reference_min_px on both
# sides in the photo. The view gets one labelled contact sheet.
REFERENCE_SHEET_LONG_EDGE = 2048
REFERENCE_SHEET_MAX_SPECIES = 16
REFERENCE_SHEET_NAME = 'references.jpg'
CULTIVAR_MATCHES = ('exact', 'species')
REFERENCE_FIELDS = ('path', 'source_page', 'image_url', 'accessed', 'identification', 'cultivar_match', 'usage_note')
REFERENCE_LINE = (
    "The {ordinal} supplied image is a labelled sheet of real photographs of the species in this view, for identification "
    "only: match each listed plant's leaves, habit and flowers to its photo; do not copy the photo's composition, "
    "background or extra plants into the scene.")
# Label map: every plant with at least label_min_px visible native pixels, outlined and numbered on the dimmed view.
LABEL_MAP_NAME = 'labels.png'
LABEL_LINE = (
    "The {ordinal} supplied image is a numbered map of every plant in the view, for identification only: each number "
    "marks one plant of the species listed for it; keep each numbered plant where it is as that species; do not draw "
    "numbers, outlines or labels.")
ORDINALS = ('first', 'second', 'third')
LABEL_OUTLINE = (255, 214, 0)
# Review notes (the scene review's real-failure observations and defects), also recorded per attempt.
CHECK_SET = 'real_failures_observed_v1'
REAL_FAILURE_CHECKS = ('plants_rooted_in_beds', 'hardscape_unchanged', 'no_seams_or_double_images', 'species_form_recognizable')
DEFECT_TYPES = ('movement', 'extent_change', 'extra_foliage', 'invented_plant', 'cultivar_detail', 'bloom_missing',
                'colour_shift', 'other')
# Attempt 2 is made only for these: every real-failure concern, missing bloom, invented plants, and species-form drift
# (a cultivar_detail or other defect that names a plant of the view).
RETRY_DEFECTS = ('bloom_missing', 'invented_plant')
SPECIES_DRIFT_DEFECTS = ('cultivar_detail', 'other')
RETRY_HEADER = ("CORRECTIONS FOR THIS ATTEMPT. Edit the original view again with everything above, and fix these "
                "problems the review recorded in the previous attempt:")
RETRY_LINE = "Fix: {where} — {observation}; keep everything else as in your previous result's realism"
COLUMNS = ('left', 'centre', 'right')
ROWS = ('background', 'middle ground', 'foreground')
FULL_PREAMBLE = (
    "Use case: photo edit. Edit target: the supplied image, the whole view of an existing designed garden in a real "
    "site photograph. Make the garden an ordinary real photograph, with botanically authentic living plants, in one "
    "pass over the entire view. Keep the same camera, framing, perspective and lighting. Keep {hardscape} exactly in "
    "place with identical geometry: no added, removed, moved or doubled slab joints, edges, steps, walls, gravel strips "
    "or structures; {materials}. {ground} Every plant stays rooted in its bed, never growing out of concrete, paving, "
    "steps or walls; foliage may naturally overhang an edge. No new pots, steps, paving, structures or text, and no new "
    "plants in empty bed ground. Keep the house, sky, street and neighbourhood unchanged. Every plant stays where it is "
    "as its listed species and form: natural leaf movement, fuller foliage and a natural outline are fine, but never "
    "remove, replace or merge a plant. No invented hybrid foliage. Every flowering plant is in full bloom, true to its "
    "cultivar. The planting, by species, with how many plants are visible and where they are in the view (left, centre "
    "or right; background, middle ground or foreground):")
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


def input_dims(w, h, size):
    """The full-view input: the view fitted inside the tool size (oriented like the view), never upscaled."""
    tw, th = (max(size), min(size)) if w >= h else (min(size), max(size))
    s = min(1., tw / w, th / h)
    return [max(1, half_up(w * s)), max(1, half_up(h * s))]


def location(position_percent):
    x, y = position_percent
    return f"{COLUMNS[min(2, int(x * 3 // 100))]} {ROWS[min(2, int(y * 3 // 100))]}"


def inventory(ids, visible, plants, box):
    """Every plant with at least one visible pixel inside the box, in box coordinates."""
    x0, y0, x1, y1 = box
    crop = np.where(visible[y0:y1, x0:x1], ids[y0:y1, x0:x1], 0)
    w, h = x1 - x0, y1 - y0
    rows = []
    for p in sorted(plants, key=lambda q: q['id']):
        yy, xx = np.nonzero(crop == p['pass_index'])
        if not len(xx):
            continue
        bx0, by0, bx1, by1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
        position = [half_up(100 * (bx0 + bx1) / 2 / w), half_up(100 * (by0 + by1) / 2 / h)]
        rows.append({'id': p['id'], 'species': p['species'], 'pass_index': int(p['pass_index']),
                     'visible_pixels': int(len(xx)), 'position_percent': position, 'size_px': [bx1 - bx0, by1 - by0],
                     'centroid_px': [half_up(float(xx.mean())), half_up(float(yy.mean()))], 'location': location(position)})
    return rows


def species_order(rows):
    """Species of an inventory, largest visible pixel area first (the prompt's order)."""
    area = {}
    for r in rows:
        area[r['species']] = area.get(r['species'], 0) + r['visible_pixels']
    return sorted(area, key=lambda name: (-area[name], name))


def location_order(rows):
    """Locations of one species' rows, most plants first (the prompt's order)."""
    where = {}
    for r in rows:
        where[r['location']] = where.get(r['location'], 0) + 1
    return sorted(where, key=lambda loc: (-where[loc], loc))


def assign_labels(rows, min_px):
    """Number every plant with at least min_px visible pixels, in prompt order: species largest first, then its
    locations (most plants first), then left to right and top to bottom. Smaller fragments get label None."""
    n = 0
    for name in species_order(rows):
        mine = [r for r in rows if r['species'] == name]
        for loc in location_order(mine):
            for r in sorted((r for r in mine if r['location'] == loc), key=lambda r: (r['centroid_px'], r['id'])):
                if r['visible_pixels'] >= min_px:
                    n += 1; r['label'] = n
                else:
                    r['label'] = None
    return rows


def check_records(records, species):
    require(isinstance(records, dict) and isinstance(records.get('species'), dict), 'Plant records need a species map')
    for name in species:
        r = records['species'].get(name)
        require(isinstance(r, dict) and r.get('common_name') and r.get('description'),
                'Plant records need a common name and plain habit/leaf description for ' + name)
        require(bool(r.get('flowers')) != bool(r.get('bloom_exempt')),
                'Give the in-bloom flowers, or the recorded bloom exemption reason, for ' + name)


def meaningful_species(rows, min_px):
    """Species in an inventory, largest visible pixel area first, split into those with an instance whose visible
    bbox is at least min_px on both sides (they get a reference) and the rest (small fragments)."""
    area = {}
    for r in rows:
        area[r['species']] = area.get(r['species'], 0) + r['visible_pixels']
    order = sorted(area, key=lambda name: (-area[name], name))
    big = {r['species'] for r in rows if min(r['size_px']) >= min_px}
    return [n for n in order if n in big], [n for n in order if n not in big], area


def is_url(value):
    return isinstance(value, str) and value.startswith(('https://', 'http://')) and len(value) > 10


def check_reference_photo(photo, name, records_dir):
    """One reference photo entry: a real identified photograph kept under the records folder (the scene work folder)."""
    where = f'reference photo of {name}'
    require(isinstance(photo, dict) and all(photo.get(k) for k in REFERENCE_FIELDS),
            f'Each {where} needs ' + ', '.join(REFERENCE_FIELDS))
    rel = Path(photo['path'])
    require(not rel.is_absolute(), f'Record the {where} path relative to RECORDS.json')
    path = (records_dir / rel).resolve()
    require(path.is_relative_to(records_dir) and path.is_file(),
            f'Keep the {where} as a file under the scene work folder (next to RECORDS.json): {photo["path"]}')
    library = os.environ.get('PACIFICA_ASSETS_DIR')
    require(not library or not path.is_relative_to(Path(library).resolve()),
            f'Reference photos are model input only; never keep them in the shared plant library: {photo["path"]}')
    require(is_url(photo['source_page']) and is_url(photo['image_url']), f'Give the source page and image URL of the {where}')
    try:
        datetime.date.fromisoformat(photo['accessed'])
    except (TypeError, ValueError):
        raise ValueError(f'Give the access date (YYYY-MM-DD) of the {where}')
    require(photo['cultivar_match'] in CULTIVAR_MATCHES, f'cultivar_match of the {where} is exact or species')
    require(isinstance(photo['identification'], str) and isinstance(photo['usage_note'], str),
            f'Give who identified the {where} and a usage note')
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        raise ValueError(f'The {where} is not a readable image: {photo["path"]}')
    return path


def check_reference_records(records, records_dir, required, min_px):
    """reference_photos or reference_missing for every species that is large in the view; any given entry is checked."""
    for name in sorted(required):
        r = records['species'][name]
        require('reference_photos' in r or 'reference_missing' in r,
                f'Give reference_photos (or reference_missing with a reason) for {name}; it is at least {min_px} px in the view')
    for name, r in sorted(records['species'].items()):
        has_photos, has_missing = 'reference_photos' in r, 'reference_missing' in r
        if not has_photos and not has_missing:
            continue
        require(has_photos != has_missing, f'Give reference_photos or reference_missing for {name}, not both')
        if has_missing:
            require(isinstance(r['reference_missing'], str) and r['reference_missing'].strip(),
                    f'reference_missing for {name} needs a reason')
        else:
            require(isinstance(r['reference_photos'], list) and r['reference_photos'],
                    f'reference_photos for {name} is a nonempty list (or use reference_missing)')
            for photo in r['reference_photos']:
                check_reference_photo(photo, name, records_dir)


def view_references(rows, records, records_dir, min_px):
    """The species references of the view: which species appear on its sheet (largest area first, one photo each, the
    first exact-cultivar photo if any), and which are left off and why."""
    big, small, area = meaningful_species(rows, min_px)
    shown, omitted = [], [{'species': n, 'reason': 'below_threshold', 'pixel_area': area[n]} for n in small]
    for name in big:
        r = records['species'][name]
        if 'reference_missing' in r:
            omitted.append({'species': name, 'reason': 'reference_missing', 'pixel_area': area[name], 'note': r['reference_missing']})
            continue
        if len(shown) == REFERENCE_SHEET_MAX_SPECIES:
            omitted.append({'species': name, 'reason': 'sheet_cap', 'pixel_area': area[name]})
            continue
        photos = r['reference_photos']
        photo = next((q for q in photos if q['cultivar_match'] == 'exact'), photos[0])
        shown.append({'species': name, 'common_name': r['common_name'], 'pixel_area': area[name],
                      'photo': record(records_dir / photo['path']),
                      **{k: photo[k] for k in ('identification', 'cultivar_match', 'source_page', 'image_url', 'accessed')}})
    return {'min_px': min_px, 'species': shown, 'omitted': omitted}


def sheet_font(size):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, AttributeError):
        return ImageFont.load_default()


def wrap(draw, text_, font, width, lines=3):
    out = ['']
    for word in text_.split():
        trial = (out[-1] + ' ' + word).strip()
        if draw.textlength(trial, font=font) <= width or not out[-1]:
            out[-1] = trial
        elif len(out) < lines:
            out.append(word)
        else:
            out[-1] = out[-1] + ' ...'
            break
    return out


def reference_sheet(entries, base=None):
    """Labelled contact sheet (RGB uint8) of the view's reference photos: one cell per species, the photo fitted into
    its cell and captioned with the botanical and common name (plus 'species photo' when the photo is not the exact
    cultivar). Deterministic for the same photos; long edge at most REFERENCE_SHEET_LONG_EDGE."""
    n = len(entries)
    require(0 < n <= REFERENCE_SHEET_MAX_SPECIES, f'A reference sheet holds 1-{REFERENCE_SHEET_MAX_SPECIES} species')
    cols = math.ceil(math.sqrt(n)); nrows = math.ceil(n / cols)
    cell = min(768, REFERENCE_SHEET_LONG_EDGE // cols)
    photo_h, pad = cell * 3 // 4, max(4, cell // 64)
    size = max(14, cell // 24)
    font = sheet_font(size)
    caption_h = int(size * 3.4) + pad
    sheet = Image.new('RGB', (cols * cell, nrows * (photo_h + caption_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, e in enumerate(entries):
        x0, y0 = (i % cols) * cell, (i // cols) * (photo_h + caption_h)
        draw.rectangle([x0 + pad, y0 + pad, x0 + cell - pad - 1, y0 + photo_h - 1], fill=(232, 232, 232))
        with Image.open(checked(e['photo'], base or Path.cwd())) as im:
            photo = ImageOps.exif_transpose(im).convert('RGB')
        aw, ah = cell - 2 * pad, photo_h - pad
        scale = min(aw / photo.width, ah / photo.height)
        pw, ph = max(1, round(photo.width * scale)), max(1, round(photo.height * scale))
        photo = photo.resize((pw, ph), Image.LANCZOS)
        sheet.paste(photo, (x0 + pad + (aw - pw) // 2, y0 + pad + (ah - ph) // 2))
        label = f"{e['species']} ({e['common_name']})" + ('; species photo' if e['cultivar_match'] == 'species' else '')
        for k, line in enumerate(wrap(draw, label, font, cell - 2 * pad)):
            draw.text((x0 + pad, y0 + photo_h + pad // 2 + k * int(size * 1.1)), line, fill=(0, 0, 0), font=font)
    require(max(sheet.size) <= REFERENCE_SHEET_LONG_EDGE, 'Reference sheet exceeds its size cap')
    return np.array(sheet)


def species_line(name, record_, rows):
    """One species: records text, flowers, and how many plants are where, with their label-map numbers, e.g.
    `3 plants: left foreground 2 (#1, #2), centre middle ground 1 (#3)`. Unnumbered fragments are only counted."""
    flowers = record_['flowers'] if record_.get('flowers') else 'none required; preserve foliage'
    places = []
    for loc in location_order(rows):
        here = [r for r in rows if r['location'] == loc]
        nums = sorted(r['label'] for r in here if r.get('label'))
        places.append(f'{loc} {len(here)}' + (f" ({', '.join(f'#{k}' for k in nums)})" if nums else ''))
    count = len(rows)
    return (f"{name} ({record_['common_name']}; {record_['description']}); flowers: {flowers}. "
            f"{count} plant{'s' if count != 1 else ''}: {', '.join(places)}")


def image_lines(labels, sheet):
    """The sentences naming the supplied images after the composite: the label map, then the reference sheet."""
    lines, n = [], 1
    for present, line in ((labels, LABEL_LINE), (sheet, REFERENCE_LINE)):
        if present:
            lines.append(line.format(ordinal=ORDINALS[n])); n += 1
    return lines


def prompt(rows, records, mulch, sheet=False, labels=False):
    fields = dict(hardscape=records.get('hardscape') or HARDSCAPE, materials=records.get('materials') or MATERIALS,
                  ground=MULCH if mulch else NO_MULCH)
    text = FULL_PREAMBLE.format(**fields)
    if labels:
        text = text.replace('background, middle ground or foreground):',
                            'background, middle ground or foreground), with their numbers on the map:')
    extra = image_lines(labels, sheet)
    if extra:  # the composite is the first image; the image sentences go before the species list
        head, tail = text.split(' The planting, by species', 1)
        text = (head.replace('Edit target: the supplied image', 'Edit target: the first supplied image') + ' '
                + ' '.join(extra) + ' The planting, by species' + tail)
    lines = [species_line(s, records['species'][s], [r for r in rows if r['species'] == s]) for s in species_order(rows)]
    if not lines:
        lines = ['No designed plant is visible in this view; refine only bed, ground and hardscape surfaces.']
    return text + '\n' + '\n'.join(lines) + '\n'


def check_params(params):
    kept = {'margin_px': int(params['margin_px']), 'feather_px': int(params['feather_px']),
            'reference_min_px': int(params['reference_min_px']), 'label_min_px': int(params['label_min_px']),
            'input_size': [int(v) for v in params['input_size']]}
    require(0 <= kept['margin_px'] <= 96 and 1 <= kept['feather_px'] <= 96, 'Bounded edit-region margin/feather required')
    require(1 <= kept['reference_min_px'] <= 512, 'Reference threshold must be 1-512 px')
    require(1 <= kept['label_min_px'] <= 100000, 'Label threshold must be 1-100000 visible px')
    require(len(kept['input_size']) == 2 and all(256 <= v <= 4096 for v in kept['input_size']),
            'The image tool input size is two sides of 256-4096 px')
    return kept


def derive(review_path, records_path, params):
    """Recompute region, the full-view job, its inventory, references and prompt from verified evidence."""
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
    w, h = view['dimensions']
    require(ids.shape == (h, w), 'Object IDs differ from the view canvas')
    kept = check_params(params)
    region = edit_region(alpha, transition, kept['margin_px'])
    box = [0, 0, w, h]  # the whole view: the image tool returns the view's aspect, and the region weight limits the change
    rows = assign_labels(inventory(ids, alpha > 0, projected, box), kept['label_min_px'])
    check_records(records, sorted({r['species'] for r in rows}))
    min_px = kept['reference_min_px']
    check_reference_records(records, records_path.parent, set(meaningful_species(rows, min_px)[0]), min_px)
    refs = view_references(rows, records, records_path.parent, min_px)
    labelled = any(r['label'] for r in rows)
    job = {'id': JOB_ID, 'kind': 'full_view', 'box': box, 'input_size': input_dims(w, h, kept['input_size']),
           'inventory': rows, 'references': refs,
           'prompt_text': prompt(rows, records, layers['bed_on'], bool(refs['species']), labelled)}
    return {'review': record(review_path), 'records': record(records_path), 'composite': record(composite),
            'cleaned_photo': record(checked(proof['cleaned_photo'], fp.parent)), 'camera_id': view['camera_id'],
            'handoff': record(hp), 'dimensions': [w, h], 'baseline_status': result['status'],
            'params': kept, 'job': job, 'baseline_result': result,
            'plant_ids': np.where(alpha > 0, ids, 0)}, region


def to8(a):
    return a if a.dtype == np.uint8 else np.rint(a / 257).astype(np.uint8)


def input_pixels(composite, job):
    """The full-view input: the 8-bit composite at the job box, Lanczos-reduced to input_size."""
    x0, y0, x1, y1 = job['box']
    crop = to8(composite)[y0:y1, x0:x1]
    size = tuple(job['input_size'])
    if size == (x1 - x0, y1 - y0):
        return crop
    return np.array(Image.fromarray(crop).resize(size, Image.LANCZOS))


def label_map(small, plant_ids, rows, box):
    """The numbered plant map (RGB uint8, the input's size): the reduced composite dimmed and desaturated, each labelled
    plant's visible pixels thinly outlined, and its number in a black box at its visible centroid, nudged to the
    nearest free place so numbers never overlap. Deterministic."""
    h, w = small.shape[:2]
    x0, y0, x1, y1 = box
    gray = small.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    out = np.clip(np.rint((.6 * small + .4 * gray[..., None]) * .6), 0, 255).astype(np.uint8)
    crop = plant_ids[y0:y1, x0:x1]  # nearest-neighbour reduction of the visible renderer IDs to the input size
    ys = np.minimum((np.arange(h) + .5) * crop.shape[0] / h, crop.shape[0] - 1).astype(int)
    xs = np.minimum((np.arange(w) + .5) * crop.shape[1] / w, crop.shape[1] - 1).astype(int)
    ids = crop[ys][:, xs]
    labelled = sorted((r for r in rows if r.get('label')), key=lambda r: r['label'])
    kernel = np.ones((3, 3), np.uint8)
    for r in labelled:
        mask = (ids == r['pass_index']).astype(np.uint8)
        out[(mask - cv2.erode(mask, kernel)) > 0] = LABEL_OUTLINE
    image = Image.fromarray(out); draw = ImageDraw.Draw(image)
    font = sheet_font(max(12, half_up(min(w, h) / 48)))
    sx, sy = w / (x1 - x0), h / (y1 - y0)
    spots = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)]
    placed = []
    for r in labelled:
        txt = str(r['label'])
        l, t, rr, b = draw.textbbox((0, 0), txt, font=font)
        bw, bh = rr - l + 4, b - t + 4
        cx, cy = r['centroid_px'][0] * sx, r['centroid_px'][1] * sy
        rects = []
        for dx, dy in spots:
            bx = int(min(max(0, round(cx - bw / 2 + dx * (bw + 2))), max(0, w - bw)))  # nudges keep a 2 px gap
            by = int(min(max(0, round(cy - bh / 2 + dy * (bh + 2))), max(0, h - bh)))
            rects.append((bx, by, bx + bw, by + bh))
        free = [q for q in rects if not any(q[0] < o[2] and o[0] < q[2] and q[1] < o[3] and o[1] < q[3] for o in placed)]
        rect = (free or rects)[0]
        placed.append(rect)
        draw.rectangle([rect[0], rect[1], rect[2] - 1, rect[3] - 1], fill=(0, 0, 0))
        draw.text((rect[0] + 2 - l, rect[1] + 2 - t), txt, fill=(255, 255, 255), font=font)
    return np.array(image)


def check_notes(notes):
    """Review notes of one attempt (or the scene review): the four real-failure observations ('pass', or a concern
    naming where and what) and the defect list. Returns (concerns, defects)."""
    require(isinstance(notes, dict) and notes.get('check_set') == CHECK_SET, f'Scene edit reviews use check_set {CHECK_SET}')
    checks = notes.get('checks', {})
    require(isinstance(checks, dict) and set(checks) == set(REAL_FAILURE_CHECKS),
            'Record exactly the real-failure observations: ' + ', '.join(REAL_FAILURE_CHECKS))
    concerns = []
    for k in REAL_FAILURE_CHECKS:
        v = checks[k]
        if v == 'pass':
            continue
        require(isinstance(v, dict) and v.get('result') == 'concern' and v.get('where') and v.get('observation'),
                f'Real-failure observation {k} is "pass", or {{"result": "concern", "where", "observation"}}')
        concerns.append({'check': k, 'where': v['where'], 'observation': v['observation']})
    defects = notes.get('defects')
    require(isinstance(defects, list), 'Record the defects list (empty when there are none)')
    for d in defects:
        require(isinstance(d, dict) and d.get('type') in DEFECT_TYPES and d.get('where') and d.get('observation'),
                'Each defect needs a type (' + ', '.join(DEFECT_TYPES) + '), where and observation')
    require(notes.get('defect_count') == len(defects), 'defect_count must equal the number of recorded defects')
    return concerns, defects


def plant_names(rows, records):
    """Lower-case words that name a plant of the view: botanical names, genera, common names and their last word."""
    names = set()
    for name in {r['species'] for r in rows}:
        common = records['species'][name]['common_name']
        names |= {name, name.split()[0], common, common.split()[-1]}
    return sorted(n.lower() for n in names if len(n) >= 4)


def retry_items(notes, rows, records):
    """What attempt 2 must fix: every real-failure concern, missing bloom, invented plants, and species-form drift
    (a cultivar_detail or other defect that names a plant of the view)."""
    concerns, defects = check_notes(notes)
    names = plant_names(rows, records)
    names_plant = lambda d: any(n in f"{d['where']} {d['observation']}".lower() for n in names)
    return concerns + [d for d in defects if d['type'] in RETRY_DEFECTS or
                       (d['type'] in SPECIES_DRIFT_DEFECTS and names_plant(d))]


def retry_prompt_text(planned, items):
    """Attempt 2's prompt: the planned prompt, then the frozen correction block with one Fix line per item."""
    require(items, 'Attempt 1 recorded no real-failure concern, missing bloom, invented plant or species-form drift; '
                   'deliver attempt 1 (attempt 2 is only for those)')
    fixes = [RETRY_LINE.format(where=i['where'], observation=i['observation']) for i in items]
    return planned.rstrip('\n') + '\n\n' + RETRY_HEADER + '\n' + '\n'.join(fixes) + '\n'


def score(notes):
    concerns, defects = check_notes(notes)
    return {'concerns': len(concerns), 'defects': len(defects)}


def best_attempt(scores):
    """Fewer real-failure concerns, then fewer defects; a tie keeps the earlier attempt."""
    return min(scores, key=lambda s: (s['concerns'], s['defects'], s['attempt']))['attempt']


def retry_prompt(plan_path, notes_path, out):
    pp = Path(plan_path).resolve()
    plan = validate(read(pp), pp.parent)[0]
    job = plan['jobs'][0]
    records = read(checked(plan['records'], pp.parent))
    text = retry_prompt_text(checked(job['prompt'], pp.parent).read_text(encoding='utf-8'),
                             retry_items(read(Path(notes_path).resolve()), job['inventory'], records))
    out = Path(out).resolve(); require(not out.exists(), 'Use a new prompt path for attempt 2')
    out.write_text(text, encoding='utf-8')
    return text


def prepare(review, records, out, **overrides):
    params = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    derived, region = derive(review, records, params)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new scene edit plan directory'); out.mkdir(parents=True)
    rgb, _ = load(derived['composite']['path'])
    Image.fromarray(region.astype(np.uint8) * 255).save(out / 'region.png')
    Image.fromarray(np.rint(region_weight(region, params['feather_px']) * 255).astype(np.uint8)).save(out / 'region-weight.png')
    job = derived['job']
    d = out / 'jobs' / job['id']; d.mkdir(parents=True)
    small = input_pixels(rgb, job)
    save(d / 'input.png', small)
    (d / 'prompt.txt').write_text(job['prompt_text'], encoding='utf-8')
    if any(r['label'] for r in job['inventory']):
        Image.fromarray(label_map(small, derived['plant_ids'], job['inventory'], job['box'])).save(d / LABEL_MAP_NAME)
        job['labels'] = record(d / LABEL_MAP_NAME)
    if job['references']['species']:
        Image.fromarray(reference_sheet(job['references']['species'])).save(d / REFERENCE_SHEET_NAME, quality=90)
        job['references']['sheet'] = record(d / REFERENCE_SHEET_NAME)
    frozen = {k: v for k, v in job.items() if k != 'prompt_text'} | {'input': record(d / 'input.png'), 'prompt': record(d / 'prompt.txt')}
    plan = {'schema_version': PLAN_SCHEMA, 'kind': KIND,
            **{k: v for k, v in derived.items() if k not in ('job', 'baseline_result', 'plant_ids')},
            'required_model': MODEL, 'image_tools': list(IMAGE_TOOLS), 'max_attempts': MAX_ATTEMPTS,
            'region': record(out / 'region.png'), 'region_weight_preview': record(out / 'region-weight.png'), 'jobs': [frozen]}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def validate(plan, base):
    """Re-derive the frozen plan; returns (plan, region, composite, cleaned photo, previous composite guard result)."""
    require(plan.get('kind') == KIND, 'Frozen scene edit plan required')
    require(plan.get('schema_version') == PLAN_SCHEMA,
            'Only schema 5 full-view scene edit plans are supported; the tiled schemas 1-4 were removed')
    require(plan.get('required_model') == MODEL and plan.get('max_attempts') == MAX_ATTEMPTS, 'Scene edit model contract changed')
    derived, region = derive(checked(plan['review'], base), checked(plan['records'], base), plan['params'])
    for key in ('review', 'records', 'composite', 'cleaned_photo', 'camera_id', 'handoff', 'dimensions', 'params'):
        require(derived[key] == plan[key], 'Scene edit plan differs from its evidence: ' + key)
    require(len(plan['jobs']) == 1, 'A scene edit plan has exactly one full-view job')
    want, job = derived['job'], plan['jobs'][0]
    for key in ('id', 'kind', 'box', 'input_size', 'inventory'):
        require(want[key] == job[key], f"Scene edit job differs: {key}")
    require(checked(job['prompt'], base).read_text(encoding='utf-8') == want['prompt_text'], 'Planned prompt changed')
    check_sheet(want, job, base)
    with Image.open(checked(plan['region'], base)) as im:
        require(np.array_equal(np.array(im) > 0, region), 'Scene edit region differs')
    composite, _ = load(checked(plan['composite'], base))
    cleaned, _ = load(checked(plan['cleaned_photo'], base))
    target, _ = load(checked(job['input'], base))
    small = input_pixels(composite, job)
    require(np.array_equal(target, small), 'Planned input differs from the reduced composite')
    if any(r['label'] for r in want['inventory']):
        require(isinstance(job.get('labels'), dict), 'The view has numbered plants; the plan needs its label map')
        with Image.open(checked(job['labels'], base)) as im:
            frozen = np.array(im.convert('RGB'))
        require(np.array_equal(frozen, label_map(small, derived['plant_ids'], want['inventory'], want['box'])),
                'Label map differs from the composite and plant IDs')
    else:
        require('labels' not in job, 'The view has no numbered plant but records a label map')
    return plan, region, composite, cleaned, derived['baseline_result']


def check_sheet(want, job, base):
    """The reference selection re-derives from the records, and the frozen sheet is the sheet those photos make
    (pixels within JPEG tolerance; the plan hash binds the exact file)."""
    got = job.get('references')
    require(isinstance(got, dict) and {k: v for k, v in got.items() if k != 'sheet'} == want['references'],
            'Scene edit job differs: references')
    if not got['species']:
        require('sheet' not in got, 'The view has no reference species but records a sheet')
        return
    with Image.open(checked(got['sheet'], base)) as im:
        frozen = np.array(im.convert('RGB'))
    rebuilt = reference_sheet(got['species'], base)
    require(frozen.shape == rebuilt.shape and float(np.abs(frozen.astype(np.int16) - rebuilt).mean()) <= 4,
            'Reference sheet differs from its photos')


def resized(raw_path, box):
    """Return an 8-bit RGB generation resized to its box (Lanczos); the aspect must already match within 3%."""
    rgb, alpha = load(raw_path)
    require(alpha is None or np.all(alpha == np.iinfo(alpha.dtype).max), 'Generated edits must be opaque')
    rgb = to8(rgb)
    w, h = box[2] - box[0], box[3] - box[1]
    rh, rw = rgb.shape[:2]
    require(abs((rw / rh) / (w / h) - 1) <= .03, 'Generated edit aspect ratio differs from the view; never warp to fit')
    if (rw, rh) == (w, h):
        return rgb
    return np.array(Image.fromarray(rgb).resize((w, h), Image.LANCZOS))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan', help='Freeze the full-view job, region and prompt for one view')
    a.add_argument('--finish-review', required=True, help='Checked schema 4 photo_finish review of the composite')
    a.add_argument('--records', required=True, help='Plant records JSON: species -> common_name, description, flowers|bloom_exempt')
    a.add_argument('--out', required=True)
    a.add_argument('--margin', type=int); a.add_argument('--feather', type=int)
    a.add_argument('--input-size', type=int, nargs=2, metavar=('W', 'H'),
                   help="The image tool's largest landscape size (default 1536 1024; swapped for portrait views)")
    a.add_argument('--reference-min-px', type=int,
                   help='A species gets a reference photo when an instance is at least this many px on both sides (default 48)')
    a.add_argument('--label-min-px', type=int,
                   help='Plants with at least this many visible px are numbered on the label map (default 400)')
    a = sub.add_parser('retry_prompt', help="Attempt 2's prompt: the planned prompt plus fixes from attempt 1's review notes")
    a.add_argument('plan'); a.add_argument('--review-notes', required=True, help="Attempt 1's review notes JSON")
    a.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'retry_prompt':
        text = retry_prompt(a.plan, a.review_notes, a.out)
        print(json.dumps({'prompt': str(Path(a.out).resolve()), 'fixes': text.count('\nFix: ')}, indent=2))
        raise SystemExit(0)
    plan = prepare(a.finish_review, a.records, a.out, margin_px=a.margin, feather_px=a.feather,
                   input_size=a.input_size, reference_min_px=a.reference_min_px, label_min_px=a.label_min_px)
    job = plan['jobs'][0]
    print(json.dumps({'plan': str(Path(a.out).resolve() / 'plan.json'), 'input': job['input']['path'],
                      'input_size': job['input_size'], 'prompt': job['prompt']['path'], 'plants': len(job['inventory']),
                      'label_map': job.get('labels', {}).get('path'),
                      'numbered_plants': sum(1 for r in job['inventory'] if r['label']),
                      'reference_sheet': job['references'].get('sheet', {}).get('path'),
                      'reference_species': [e['species'] for e in job['references']['species']]}, indent=2))
