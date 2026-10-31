"""Freeze a scene photographic edit plan for one finished photo view, BEFORE generation.

Input is a checked schema 4 photo_finish review (the composite that would otherwise be delivered). The plan
derives the landscape edit region from that composite's renderer matte and transition permissions, lays out
one full-view job plus overlapping native-resolution tiles over the region, and writes one prompt per job whose
inventory (species, IDs, positions as % of the crop, projected pixel sizes, edge fragments included) is computed
from the actual renderer object IDs. Plant descriptions and bloom come from the agent's plant records file.
Deterministic: validate() re-derives every field. No image generation or recognition happens here.

Plan schema 4 (current) is schema 3 plus reference photos: each species recorded in RECORDS.json carries
`reference_photos` (real, identified photographs with provenance, kept in the scene work folder next to the records)
or `reference_missing` with a reason. Every species with an instance at least `reference_min_px` (48) on both sides
in some tile must have one or the other. Each tile with such species gets `jobs/tile-NN/references.jpg`, a labelled
contact sheet (one photo per species, largest pixel area first, at most 8, long edge 1536 px), hashed in the plan;
its prompt says the second supplied image is that sheet, for identification only. Otherwise schema 4 is schema 3.

Plan schema 3: `anchor: none` (default). Each tile edits its native-resolution crop of the photo_finish
composite (`jobs/tile-NN/input.png`) directly, so plants stay large enough for the image tool to keep their species.
There is no full-view job. The plan also freezes `structure.png`, the per-pixel classes the compose step uses to
check each tile against the composite: 1 plants and bed near plants (lenient), 2 open bed ground away from plants
(no added structure), 3 hardscape and photo pixels away from plants (strict). `anchor: full_view` keeps the
optional anchored route: one full-view job, whose accepted result is upscaled and cropped for each tile
(`tile-input`), with the composite crop (`geometry.png`) as a geometry reference only.

Schema 2 plans (always anchored, lab P2) and schema 1 plans (tiles edited from the composite crop, with the
full-view crop as style reference) stay verifiable.
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
PLAN_SCHEMA = 4
SCHEMAS = (1, 2, 3, 4)
ANCHORS = ('none', 'full_view')
DEFAULTS = {'tile': [1152, 1024], 'overlap': .3, 'margin_px': 24, 'feather_px': 24}
DEFAULTS_V2 = {**DEFAULTS, 'seam_feather_px': 20}
DEFAULTS_V3 = {**DEFAULTS_V2, 'anchor': 'none'}
DEFAULTS_V4 = {**DEFAULTS_V3, 'reference_min_px': 48}
# Reference photos (schema 4, anchor: none): a species gets a reference in a tile when one of its instances there has a
# visible bounding box at least reference_min_px on both sides. Tiles get one labelled contact sheet.
REFERENCE_SHEET_LONG_EDGE = 1536
REFERENCE_SHEET_MAX_SPECIES = 8
REFERENCE_SHEET_NAME = 'references.jpg'
CULTIVAR_MATCHES = ('exact', 'species')
REFERENCE_FIELDS = ('path', 'source_page', 'image_url', 'accessed', 'identification', 'cultivar_match', 'usage_note')
REFERENCE_LINE = (
    "The second supplied image is a labelled sheet of real photographs of the species in this tile, for identification "
    "only: match each listed plant's leaves, habit and flowers to its photo; do not copy the photo's composition, "
    "background or extra plants into the scene.")
# structure.png classes (schema 3): how strictly a tile must keep the composite's structure at each pixel.
OUTSIDE, LENIENT, OPEN_GROUND, STRICT = 0, 1, 2, 3
HARDSCAPE_CLEARANCE_PX = 48  # hardscape/photo pixels farther than this from any plant are strict (foliage may overhang)
OPEN_GROUND_CLEARANCE_PX = 64  # bed ground farther than this from any plant must not gain new structure

# Schema 1 prompts (kept so schema 1 plans re-derive exactly).
PREAMBLE_V1 = (
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

# Schema 2 prompts. They hold only what makes a real failure (same camera, plants rooted in beds, hardscape fixed,
# no new structures, each plant recognizably its designed species and form) and allow natural movement and fuller
# foliage. Tiles are detail passes over the anchored full-view result and must not move anything.
FULL_PREAMBLE = (
    "Use case: photo edit. Edit target: the supplied image, an existing designed garden in a real site photograph. "
    "Make this exact view an ordinary real photograph, with botanically authentic living plants. Keep the same camera, crop, "
    "perspective and lighting. Keep {hardscape} exactly in place with identical geometry: no added, removed, moved or doubled "
    "slab joints, edges, steps, walls, gravel strips or structures; {materials}. {ground} Every plant stays rooted in its bed, "
    "never growing out of concrete, paving, steps or walls; foliage may naturally overhang an edge. No new pots, steps, paving, "
    "structures or text, and no new plants in empty bed ground. Keep photographic areas outside the landscape unchanged. "
    "Each designed plant stays clearly its listed species and form, at about its listed position and size; natural movement, "
    "fuller foliage and a natural outline are fine. The following inventory lists every visible plant, including edge "
    "fragments: x and y are percentages from the top left, width and height are projected visible pixel sizes in the supplied "
    "image, not physical centimetres. Do not complete clipped plants or expose occluded plants. No invented hybrid foliage. "
    "Every flowering plant is in full bloom, true to its cultivar.")
TILE_PREAMBLE = (
    "Use case: photo detail. Edit target: the first supplied image, one native-resolution tile of an already-edited "
    "photograph of a designed garden, shown upscaled and soft. Add native-resolution photographic detail: crisp leaf and "
    "flower anatomy, bark and mulch grain, concrete and stone texture, natural light and grain. Do not change any shape, "
    "outline, position, size or species: every plant, leaf mass, flower, shadow and hardscape edge stays exactly where it is "
    "in the first image, so this tile lines up pixel for pixel with its neighbours. Do not add, remove or move plants, "
    "flowers, joints, edges, steps or structures, and keep the colours and light of the first image. The second supplied "
    "image is the planned 3D composite of the same crop: use it only to confirm the {hardscape} geometry and which plant is "
    "which, never to move anything toward it or copy its rendered look. {ground} Plants stay rooted in beds, never in "
    "concrete, paving or steps. For identification only, the plants in this tile (positions are percentages of the tile "
    "from the top left; sizes are approximate pixels):")
# Schema 3 tiles without an anchor: each tile is a full photographic edit of its own native-resolution composite crop.
TILE_PREAMBLE_V3 = (
    "Use case: photo edit. Edit target: the supplied image, one native-resolution tile of an existing designed garden in a "
    "real site photograph. Neighbouring tiles are edited separately and must line up with this one, so keep this exact "
    "crop. Make it an ordinary real photograph, with botanically authentic living plants. Keep the same camera, crop, "
    "perspective and lighting. Keep {hardscape} exactly in place with identical geometry: no added, removed, moved or "
    "doubled slab joints, edges, steps, walls, gravel strips or structures; {materials}. {ground} Every plant stays rooted "
    "in its bed, never growing out of concrete, paving, steps or walls; foliage may naturally overhang an edge. No new "
    "pots, steps, paving, structures or text, and no new plants in empty bed ground. Keep photographic areas outside the "
    "landscape unchanged. Every plant stays in place as its listed species and form, at its listed position and size: "
    "natural leaf movement, fuller foliage and a natural outline are fine, but never move, remove, replace or merge a "
    "plant. Plants cut by the tile edge stay cut there. The following inventory lists every visible plant, including edge "
    "fragments: x and y are percentages from the top left, width and height are projected visible pixel sizes in the "
    "supplied image, not physical centimetres. Do not complete clipped plants or expose occluded plants. No invented "
    "hybrid foliage. Every flowering plant is in full bloom, true to its cultivar.")
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


def references_used(schema, params):
    """Do tiles of this plan get reference sheets? Schema 4 tiles without an anchor."""
    return schema >= 4 and not anchored(schema, params)


def meaningful_species(rows, min_px):
    """Species in a crop's inventory, largest visible pixel area first, split into those with an instance whose visible
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
    """reference_photos or reference_missing for every species that is large in some tile; any given entry is checked."""
    for name in sorted(required):
        r = records['species'][name]
        require('reference_photos' in r or 'reference_missing' in r,
                f'Give reference_photos (or reference_missing with a reason) for {name}; it is at least {min_px} px in a tile')
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


def tile_references(rows, records, records_dir, min_px):
    """The species references of one tile: which species appear on its sheet (largest area first, one photo each, the
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
    """Labelled contact sheet (RGB uint8) of one tile's reference photos: one cell per species, the photo fitted into
    its cell and captioned with the botanical and common name (plus 'species photo' when the photo is not the exact
    cultivar). Deterministic for the same photos; long edge at most REFERENCE_SHEET_LONG_EDGE."""
    n = len(entries)
    require(0 < n <= REFERENCE_SHEET_MAX_SPECIES, 'A reference sheet holds 1-8 species')
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
    flowers = record_['flowers'] if record_.get('flowers') else 'none required; preserve foliage'
    instances = '; '.join(f"{r['id']} at [{r['position_percent'][0]}, {r['position_percent'][1]}]%, "
                          f"size [{r['size_px'][0]}, {r['size_px'][1]}]px" for r in rows)
    return f"{name} ({record_['common_name']}; {record_['description']}); flowers: {flowers}. Instances: {instances}"


def anchored(schema, params):
    """Does this plan edit tiles from a full-view anchor? Schema 2 always; schema 3 with anchor: full_view."""
    return schema == 2 or schema >= 3 and params.get('anchor') == 'full_view'


def structure_classes(region, plants, bed):
    """Per-pixel alignment classes inside the edit region (see module docstring)."""
    near = cv2.distanceTransform((~plants).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    classes = np.full(region.shape, LENIENT, np.uint8)
    classes[bed & (near > OPEN_GROUND_CLEARANCE_PX)] = OPEN_GROUND
    classes[~bed & (near > HARDSCAPE_CLEARANCE_PX)] = STRICT
    classes[~region] = OUTSIDE
    return classes


def prompt(job, rows, records, mulch, schema=PLAN_SCHEMA, anchor=True, sheet=False):
    tile = job['kind'] == 'tile'
    fields = dict(hardscape=records.get('hardscape') or HARDSCAPE, materials=records.get('materials') or MATERIALS,
                  ground=MULCH if mulch else NO_MULCH)
    if schema == 1:
        text = PREAMBLE_V1.format(part=TILE_PART if tile else '', reference=TILE_REFERENCE if tile else '', **fields)
    elif tile and not anchor:
        text = TILE_PREAMBLE_V3.format(**fields)
        if sheet:  # schema 4: the composite crop is the first image, the labelled reference sheet the second
            text = text.replace('the supplied image', 'the first supplied image') + ' ' + REFERENCE_LINE
    else:
        text = (TILE_PREAMBLE if tile else FULL_PREAMBLE).format(**fields)
    order = []
    for r in rows:
        if r['species'] not in order:
            order.append(r['species'])
    lines = [species_line(s, records['species'][s], [r for r in rows if r['species'] == s]) for s in order]
    if not lines:
        lines = ['No designed plant is visible in this crop; refine only bed, ground and hardscape surfaces.']
    return text + '\n' + '\n'.join(lines) + '\n'


def derive(review_path, records_path, params, schema=PLAN_SCHEMA):
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
    kept = {'tile': tile, 'overlap': overlap, 'margin_px': margin, 'feather_px': feather}
    if schema >= 2:
        kept['seam_feather_px'] = int(params['seam_feather_px'])
        require(4 <= kept['seam_feather_px'] <= 48, 'Seam feather must be 4-48 px')
    if schema >= 3:
        kept['anchor'] = params['anchor']
        require(kept['anchor'] in ANCHORS, 'Scene edit anchor is none or full_view')
    if schema >= 4:
        kept['reference_min_px'] = int(params['reference_min_px'])
        require(1 <= kept['reference_min_px'] <= 512, 'Reference threshold must be 1-512 px')
    anchor = anchored(schema, kept)
    region = edit_region(alpha, transition, margin)
    boxes, ramps = tiles(region, tile, overlap)
    jobs = [{'id': 'full', 'kind': 'full_view', 'box': [0, 0, w, h]}] if schema < 3 or anchor else []
    jobs += [{'id': f'tile-{i + 1:02d}', 'kind': 'tile', 'box': b} for i, b in enumerate(boxes)]
    for job in jobs:
        job['inventory'] = inventory(ids, visible, projected, job['box'])
    check_records(records, sorted({r['species'] for r in inventory(ids, visible, projected, [0, 0, w, h])}))
    if references_used(schema, kept):
        min_px = kept['reference_min_px']
        required = {s for job in jobs for s in meaningful_species(job['inventory'], min_px)[0]}
        check_reference_records(records, records_path.parent, required, min_px)
        for job in jobs:
            job['references'] = tile_references(job['inventory'], records, records_path.parent, min_px)
    for job in jobs:
        sheet = bool(job.get('references', {}).get('species'))
        job['prompt_text'] = prompt(job, job['inventory'], records, layers['bed_on'], schema, anchor, sheet)
    classes = structure_classes(region, layers['plants'], layers['bed']) if schema >= 3 else None
    return {'review': record(review_path), 'records': record(records_path), 'composite': record(composite),
            'cleaned_photo': record(checked(proof['cleaned_photo'], fp.parent)), 'camera_id': view['camera_id'],
            'handoff': record(hp), 'dimensions': [w, h], 'baseline_status': result['status'],
            'params': kept,
            'ramps_px': ramps, 'jobs': jobs, 'baseline_result': result, 'classes': classes}, region


def target_name(schema, params, job):
    """Planned crop file: the edit target, except anchored tiles, whose target is the anchor crop (geometry only here)."""
    return 'geometry' if anchored(schema, params) and job['kind'] == 'tile' else 'input'


def prepare(review, records, out, schema=PLAN_SCHEMA, **overrides):
    require(schema in SCHEMAS, 'Unknown scene edit plan schema')
    params = {**{1: DEFAULTS, 2: DEFAULTS_V2, 3: DEFAULTS_V3, 4: DEFAULTS_V4}[schema], **{k: v for k, v in overrides.items() if v is not None}}
    derived, region = derive(review, records, params, schema)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new scene edit plan directory'); out.mkdir(parents=True)
    rgb, _ = load(derived['composite']['path'])
    rgb8 = rgb if rgb.dtype == np.uint8 else np.rint(rgb / 257).astype(np.uint8)
    Image.fromarray(region.astype(np.uint8) * 255).save(out / 'region.png')
    Image.fromarray(np.rint(region_weight(region, params['feather_px']) * 255).astype(np.uint8)).save(out / 'region-weight.png')
    extra = {}
    if derived['classes'] is not None:
        Image.fromarray(derived['classes']).save(out / 'structure.png')
        extra['structure'] = record(out / 'structure.png')
    jobs = []
    for job in derived['jobs']:
        d = out / 'jobs' / job['id']; d.mkdir(parents=True)
        x0, y0, x1, y1 = job['box']
        name = target_name(schema, derived['params'], job)
        save(d / f'{name}.png', rgb8[y0:y1, x0:x1])
        (d / 'prompt.txt').write_text(job['prompt_text'], encoding='utf-8')
        if job.get('references', {}).get('species'):
            Image.fromarray(reference_sheet(job['references']['species'])).save(d / REFERENCE_SHEET_NAME, quality=90)
            job['references']['sheet'] = record(d / REFERENCE_SHEET_NAME)
        jobs.append({k: v for k, v in job.items() if k != 'prompt_text'} |
                    {name: record(d / f'{name}.png'), 'prompt': record(d / 'prompt.txt')})
    plan = {'schema_version': schema, 'kind': KIND, **{k: v for k, v in derived.items() if k not in ('jobs', 'baseline_result', 'classes')},
            'required_model': MODEL, 'image_tools': list(IMAGE_TOOLS), 'max_attempts': MAX_ATTEMPTS,
            'region': record(out / 'region.png'), 'region_weight_preview': record(out / 'region-weight.png'), **extra, 'jobs': jobs}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def validate(plan, base):
    """Re-derive the frozen plan; returns (plan, region, composite, cleaned photo, previous composite guard result).
    The structure classes (schema 3) are re-derived and checked too; read them with structure()."""
    require(plan.get('kind') == KIND and plan.get('schema_version') in SCHEMAS, 'Frozen scene edit plan required')
    require(plan.get('required_model') == MODEL and plan.get('max_attempts') == MAX_ATTEMPTS, 'Scene edit model contract changed')
    schema = plan['schema_version']
    derived, region = derive(checked(plan['review'], base), checked(plan['records'], base), plan['params'], schema)
    for key in ('review', 'records', 'composite', 'cleaned_photo', 'camera_id', 'handoff', 'dimensions', 'params', 'ramps_px'):
        require(derived[key] == plan[key], 'Scene edit plan differs from its evidence: ' + key)
    require(len(derived['jobs']) == len(plan['jobs']), 'Scene edit job layout differs')
    with Image.open(checked(plan['region'], base)) as im:
        require(np.array_equal(np.array(im) > 0, region), 'Scene edit region differs')
    if schema >= 3:
        with Image.open(checked(plan['structure'], base)) as im:
            require(np.array_equal(np.array(im), derived['classes']), 'Scene edit structure classes differ')
    composite, _ = load(checked(plan['composite'], base))
    cleaned, _ = load(checked(plan['cleaned_photo'], base))
    rgb8 = composite if composite.dtype == np.uint8 else np.rint(composite / 257).astype(np.uint8)
    for want, job in zip(derived['jobs'], plan['jobs']):
        for key in ('id', 'kind', 'box', 'inventory'):
            require(want[key] == job[key], f"Scene edit job {job.get('id')} differs: {key}")
        prompt_path = checked(job['prompt'], base)
        require(prompt_path.read_text(encoding='utf-8') == want['prompt_text'], f"Planned prompt changed: {job['id']}")
        if 'references' in want or 'references' in job:
            check_sheet(want, job, base)
        x0, y0, x1, y1 = job['box']
        target, _ = load(checked(job[target_name(schema, plan['params'], job)], base))
        require(np.array_equal(target, rgb8[y0:y1, x0:x1]), f"Planned crop differs from the composite crop: {job['id']}")
    return plan, region, composite, cleaned, derived['baseline_result']


def check_sheet(want, job, base):
    """Schema 4 tile references: the selection re-derives from the records, and the frozen sheet is the sheet those
    photos make (pixels within JPEG tolerance; the plan hash binds the exact file)."""
    got = job.get('references')
    require(isinstance(got, dict) and {k: v for k, v in got.items() if k != 'sheet'} == want.get('references'),
            f"Scene edit job {job['id']} differs: references")
    if not got['species']:
        require('sheet' not in got, f"Tile {job['id']} has no reference species but records a sheet")
        return
    with Image.open(checked(got['sheet'], base)) as im:
        frozen = np.array(im.convert('RGB'))
    rebuilt = reference_sheet(got['species'], base)
    require(frozen.shape == rebuilt.shape and float(np.abs(frozen.astype(np.int16) - rebuilt).mean()) <= 4,
            f"Reference sheet differs from its photos: {job['id']}")


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


def structure(plan, base):
    """The frozen structure classes of a validated schema 3 plan."""
    with Image.open(checked(plan['structure'], base)) as im:
        return np.array(im)


def reference(plan_path, full_raw, job_id, out):
    """Crop of the upscaled full-view result at a tile box: the tile's edit target in the anchored route (schema 2,
    or schema 3 with anchor: full_view) or, for schema 1 plans, its style/consistency reference."""
    pp = Path(plan_path).resolve(); plan = read(pp)
    full = next((j for j in plan['jobs'] if j['kind'] == 'full_view'), None)
    require(full is not None, 'This plan has no full-view anchor; edit jobs/tile-NN/input.png (the composite crop) directly')
    job = next((j for j in plan['jobs'] if j['id'] == job_id and j['kind'] == 'tile'), None)
    require(job is not None, 'Unknown tile job')
    image = reference_pixels(full_raw, full['box'], job['box'])
    out = Path(out).resolve(); require(not out.exists(), 'Use a new reference path'); out.parent.mkdir(parents=True, exist_ok=True)
    save(out, image)
    key = 'input' if plan.get('schema_version', 1) >= 2 else 'reference'
    return {'job': job_id, key: record(out), 'from_full_raw': record(full_raw)}


def reference_pixels(full_raw, full_box, box):
    """Crop of the full-view result, upscaled (Lanczos) to the photo, at one tile box."""
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
    a.add_argument('--seam-feather', type=int, help='Feather at tile ownership boundaries (default 20 px)')
    a.add_argument('--anchor', choices=ANCHORS, help='none (default): tiles edit the composite crop; full_view: anchored tiles')
    a.add_argument('--reference-min-px', type=int,
                   help='A species gets a reference photo in a tile when an instance there is at least this many px on both sides (default 48)')
    for name, text in (('tile-input', "Anchored route only: write a tile's edit target, its crop of the upscaled full-view result"),
                       ('reference', 'Alias of tile-input (for schema 1 plans, the tile style reference)')):
        a = sub.add_parser(name, help=text)
        a.add_argument('plan'); a.add_argument('--full-raw', required=True); a.add_argument('--job', required=True); a.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'plan':
        result = prepare(a.finish_review, a.records, a.out, tile=a.tile, overlap=a.overlap, margin_px=a.margin, feather_px=a.feather,
                         seam_feather_px=a.seam_feather, anchor=a.anchor, reference_min_px=a.reference_min_px)
        result = {'plan': str(Path(a.out).resolve() / 'plan.json'), 'jobs': [{'id': j['id'], 'box': j['box'],
                  'planned_crop': (j.get('input') or j['geometry'])['path'],
                  'prompt': j['prompt']['path'], 'plants': len(j['inventory']),
                  'reference_sheet': j.get('references', {}).get('sheet', {}).get('path'),
                  'reference_species': [e['species'] for e in j.get('references', {}).get('species', [])]}
                  for j in result['jobs']]}
    else:
        result = reference(a.plan, a.full_raw, a.job, a.out)
    print(json.dumps(result, indent=2))
