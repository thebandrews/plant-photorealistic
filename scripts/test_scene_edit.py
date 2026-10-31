"""Synthetic scene photographic edit tests: layout, prompts, composite-input tiles, structure checks, ownership seams,
metrics and review gates (plus the optional anchored route and legacy schemas).
No recognition claims: every visual finding here is synthetic fixture truth."""
import copy
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from appearance_guard import check
from handoff import record
from raster import load, save as save_png
import scene_edit_compose as sec
from scene_edit_compose import (ALIGNMENT, BLOCKING_CHECKS, CHECK_SET, COMPOSITE_ALIGNMENT, OPERATION, OPERATION_V1,
                                OPERATION_V2, SCENE_CHECKS_V1, alignment, composite_alignment, compose, label_weights,
                                ownership, seam_path, seam_rows, verify)
from scene_edit_plan import (LENIENT, OPEN_GROUND, STRICT, prepare, reference, structure_classes, tiles, validate,
                             edit_region, region_weight, resized)
from test_contract import save
import test_finishing

PARAMS = dict(tile=[16, 12], overlap=.3, margin_px=2, feather_px=2, seam_feather_px=4)
PARAMS_V2 = PARAMS
PARAMS_V1 = dict(tile=[16, 12], overlap=.3, margin_px=2, feather_px=2)
RECORDS = {'species': {
    'Agave victoriae-reginae': {'common_name': 'Queen Victoria agave',
                                'description': 'compact dense rosette of narrow stiff dark green leaves with white markings',
                                'bloom_exempt': 'monocarpic agave; flowering ends the rosette'},
    'Synthetic tiny cultivar': {'common_name': 'tiny daisy', 'description': 'low mound of small leaves',
                                'flowers': 'many small white daisies with yellow centres'}}}
ALL_PASS = dict.fromkeys(BLOCKING_CHECKS, 'pass')


def text_of(path):
    return Path(path).read_text(encoding='utf-8')


def gray_of(path):
    with Image.open(path) as im:
        return np.array(im)


def solid(path, size, color):
    Image.fromarray(np.full((size[1], size[0], 3), color, np.uint8)).save(path)


def blocks_image(size, seed, cell=4):
    """Deterministic structured pattern: random-colour cells, so shifts change edges."""
    rng = np.random.default_rng(seed)
    w, h = size
    small = rng.integers(20, 236, ((h + cell - 1) // cell, (w + cell - 1) // cell, 3)).astype(np.uint8)
    return np.repeat(np.repeat(small, cell, 0), cell, 1)[:h, :w]


def garden(seed, photo=False, move=6., fill=.25, shift=(0, 0), *, add=None, add_joint=False, move_joint=0,
           slab_dy=0, layout=None, w=1152, h=1024):
    """Production-scale synthetic tile: 16 leaf-rosette plants on mulch, a jointed slab and a wall. photo=True is a
    'photographic re-render': every texture replaced, colours shifted, each plant moved (sigma `move` px) and
    fuller (`fill`). Returns (image, plant ids, slab mask); ids only for the CG composite."""
    r = np.random.default_rng(seed if layout is None else layout); j = np.random.default_rng(seed + 991 + 7 * photo)
    img = np.zeros((h, w, 3), np.float32); img[:] = (110, 80, 55)
    if photo:
        grain = np.repeat(np.repeat(j.normal(0, 22, (h // 2 + 1, w // 2 + 1, 1)), 2, 0), 2, 1)[:h, :w]
        img += grain + cv2.GaussianBlur(j.normal(0, 30, (h, w)).astype(np.float32), (0, 0), 25)[..., None] * 4
    else:
        img += np.random.default_rng(seed + 5).normal(0, 8, (h, w, 1))
    sy = 720 + slab_dy
    slab = np.zeros((h, w), bool); slab[sy:] = True; slab[:sy, :110] = True
    img[slab] = (182, 178, 170)
    joints = [250, 480 + move_joint, 730, 960] + ([605] if add_joint else [])
    for x in joints:
        x += int(j.integers(-1, 2)) if photo else 0
        img[sy:, x - 2:x + 2] = (95, 92, 88)
    img[870:874, 110:] = (95, 92, 88); img[:sy, 106:110] = (120, 118, 112)
    if photo:
        stain = cv2.GaussianBlur(j.normal(0, 40, (h, w)).astype(np.float32), (0, 0), 18) * 3
        img[slab] = img[slab] * 1.05 + (stain + j.normal(0, 14, (h, w)).astype(np.float32))[slab][:, None]
    ids = np.zeros((h, w), np.uint8)
    for k in range(16):
        cx, cy, rad = int(r.integers(170, w - 40)), int(r.integers(40, 750)), int(r.integers(45, 150))
        col = r.integers(30, 200, 3).astype(np.float32)
        leaves = [(r.uniform(0, 2 * np.pi), rad * r.uniform(.4, 1), r.uniform(.12, .3), np.clip(col + r.normal(0, 18, 3), 0, 255))
                  for _ in range(int(r.integers(25, 70)))]
        dx = dy = 0
        if photo:
            dx, dy = j.normal(0, move, 2)
            leaves += [(j.uniform(0, 2 * np.pi), rad * j.uniform(.6, 1.15), .12 * rad, leaves[0][3]) for _ in range(int(len(leaves) * fill))]
        for a, L, wd, c in leaves:
            wd = max(3, wd * L) if wd < 1 else max(3, wd)
            if photo:
                a += j.normal(0, .08); L *= 1 + j.uniform(-.1, fill * .6)
                c = np.clip(c * np.array((1.06, .98, .9)) + j.normal(0, 10, 3), 0, 255)
            centre = (int(cx + dx + .5 * L * np.cos(a)), int(cy + dy + .5 * L * np.sin(a)))
            axes = (max(1, int(L / 2)), max(1, int(wd / 2)))
            cv2.ellipse(img, centre, axes, float(np.degrees(a)), 0, 360, tuple(float(v) for v in c), -1)
            if not photo:
                cv2.ellipse(ids, centre, axes, float(np.degrees(a)), 0, 360, k + 1, -1)
    if add is not None:  # an invented plant in open bed ground
        cx, cy, rad = add
        for i in range(50):
            a = j.uniform(0, 2 * np.pi); L = rad * j.uniform(.4, 1)
            cv2.ellipse(img, (int(cx + .5 * L * np.cos(a)), int(cy + .5 * L * np.sin(a))), (int(L / 2), max(2, int(L * .1))),
                        float(np.degrees(a)), 0, 360, (60 + i % 3 * 30, 140, 70), -1)
    if photo:
        img = (img + j.normal(0, 12, img.shape)) * np.array((1.04, 1., .95))
        img = np.roll(img, shift, axis=(0, 1))
    return np.clip(img, 0, 255).astype(np.uint8), ids, slab


def leaf_scene(seed, jitter=0., size_jitter=0., where=None, texture=False, w=1400, h=1300, n=260):
    """Synthetic 'garden': overlapping coloured ellipses. jitter moves silhouettes (only where `where` holds)."""
    r = np.random.default_rng(seed); j = np.random.default_rng(seed + 1000)
    img = np.full((h, w, 3), (90, 80, 60), np.uint8)
    for _ in range(n):
        c = tuple(int(v) for v in r.integers(20, 230, 3)); cx, cy = int(r.integers(0, w)), int(r.integers(0, h))
        ax, ay = int(r.integers(8, 60)), int(r.integers(4, 30)); ang = float(r.uniform(0, 180))
        move = jitter if where is None or where(cx, cy) else 0
        dx, dy = j.normal(0, move, 2) if move else (0, 0); s = 1 + j.uniform(-size_jitter, size_jitter)
        cv2.ellipse(img, (int(cx + dx), int(cy + dy)), (max(1, int(ax * s)), max(1, int(ay * s))), ang, 0, 360, c, -1)
    if texture:  # native-resolution 'photographic' detail and a slight colour change
        noise = np.random.default_rng(seed + 7).integers(-35, 35, img.shape)
        img = np.clip((img.astype(np.int16) + noise) * np.array([1.08, .97, .92]), 0, 255).astype(np.uint8)
    return img


class PlanFixture(unittest.TestCase):
    schema = 3
    anchor = None

    def setUp(self):
        self.fin = test_finishing.PhotoFinishing(); self.fin.setUp(); self.addCleanup(self.fin.doCleanups)
        self.p = self.fin.p
        save(self.p / 'finish-review.json', self.fin.review)
        save(self.p / 'records.json', RECORDS)
        params = {1: PARAMS_V1, 2: PARAMS_V2, 3: dict(PARAMS, anchor=self.anchor)}[self.schema]
        self.plan = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene', schema=self.schema, **params)
        self.tiles = [j for j in self.plan['jobs'] if j['kind'] == 'tile']


class SceneEditPlan(PlanFixture):
    def test_layout_covers_region_with_thirty_percent_overlap(self):
        plan = self.plan
        self.assertEqual(plan['schema_version'], 3); self.assertEqual(plan['params']['anchor'], 'none')
        self.assertTrue(all(j['kind'] == 'tile' for j in plan['jobs']))  # no full-view job without an anchor
        region = gray_of(plan['region']['path']) > 0
        covered = np.zeros_like(region)
        boxes = [j['box'] for j in self.tiles]
        self.assertGreater(len(boxes), 1)
        for x0, y0, x1, y1 in boxes:
            self.assertEqual((x1 - x0, y1 - y0), (16, 12)); covered[y0:y1, x0:x1] = True
        self.assertTrue(np.all(covered[region]))
        xs = sorted({b[0] for b in boxes}); ys = sorted({b[1] for b in boxes})
        for a, b in zip(xs, xs[1:]): self.assertGreaterEqual(16 - (b - a), .3 * 16)
        for a, b in zip(ys, ys[1:]): self.assertGreaterEqual(12 - (b - a), .3 * 12)

    def test_full_size_layout_and_region_feather(self):
        region = np.zeros((2560, 3840), bool); region[900:2560, 200:3500] = True
        boxes, ramps = tiles(region, [1152, 1024], .3)
        self.assertEqual(ramps, [346, 308])
        covered = np.zeros_like(region)
        for x0, y0, x1, y1 in boxes:
            self.assertEqual((x1 - x0, y1 - y0), (1152, 1024)); covered[y0:y1, x0:x1] = True
        self.assertTrue(np.all(covered[region]))
        alpha = np.zeros((40, 40), np.uint8); alpha[10:30, 10:30] = 255
        grown = edit_region(alpha, np.zeros_like(alpha), 4)
        self.assertTrue(grown[10, 6] and not grown[10, 5])
        weight = region_weight(grown, 4)
        self.assertEqual(weight[0, 0], 0); self.assertEqual(weight[20, 20], 1); self.assertTrue(0 < weight[20, 6] < 1)

    def test_tile_prompts_are_photographic_edits_of_the_composite_crop(self):
        tile = next(j for j in self.tiles if any(r['id'] == 'tiny' for r in j['inventory']))
        self.assertEqual(tile['box'][:2], [0, 0])
        tiny = next(r for r in tile['inventory'] if r['id'] == 'tiny')
        self.assertEqual(tiny['size_px'], [1, 1]); self.assertEqual(tiny['position_percent'], [22, 29])
        self.assertNotIn('hidden', [r['id'] for r in tile['inventory']])  # no visible pixels
        text = text_of(tile['prompt']['path'])
        self.assertIn('one native-resolution tile of an existing designed garden in a real site photograph', text)
        self.assertIn('Make it an ordinary real photograph', text)
        self.assertIn('must line up with this one', text)
        self.assertIn('never growing out of concrete, paving, steps or walls', text)
        self.assertIn('never move, remove, replace or merge a plant', text)
        self.assertIn('natural leaf movement, fuller foliage and a natural outline are fine', text)
        self.assertIn('shredded bark mulch', text)
        self.assertNotIn('already-edited', text); self.assertNotIn('second supplied image', text)
        self.assertIn('flowers: many small white daisies with yellow centres. Instances: tiny at [22, 29]%, size [1, 1]px', text)
        agave = next(j for j in self.tiles if j['box'][:2] == [8, 6])
        self.assertIn("Agave victoriae-reginae (Queen Victoria agave; compact dense rosette", text_of(agave['prompt']['path']))

    def test_tile_input_is_the_native_composite_crop_and_structure_is_frozen(self):
        composite, _ = load(self.plan['composite']['path'])
        for job in self.tiles:
            self.assertNotIn('geometry', job)
            x0, y0, x1, y1 = job['box']
            got, _ = load(job['input']['path'])
            self.assertTrue(np.array_equal(got, np.rint(composite[y0:y1, x0:x1] / 257).astype(np.uint8)))
        classes = gray_of(self.plan['structure']['path'])
        region = gray_of(self.plan['region']['path']) > 0
        self.assertTrue(np.array_equal(classes > 0, region))
        self.assertEqual(set(np.unique(classes[region])), {LENIENT})  # the 32 px fixture is all near its plant
        with self.assertRaisesRegex(ValueError, 'no full-view anchor'):
            reference(self.p / 'scene/plan.json', self.p / 'photo.png', self.tiles[0]['id'], self.p / 'ref.png')

    def test_structure_classes(self):
        plants = np.zeros((300, 300), bool); plants[100:140, 100:140] = True
        bed = np.zeros_like(plants); bed[:, :200] = True
        region = np.ones_like(plants); region[:, 290:] = False
        classes = structure_classes(region, plants, bed)
        self.assertEqual(classes[120, 120], LENIENT); self.assertEqual(classes[120, 190], LENIENT)  # 50 px from the plant
        self.assertEqual(classes[20, 20], OPEN_GROUND)  # bed ground > 64 px from any plant
        self.assertEqual(classes[120, 210], STRICT)  # hardscape > 48 px from any plant
        self.assertEqual(classes[150, 180], LENIENT)  # hardscape-free bed near the plant
        self.assertEqual(classes[0, 295], 0)

    def test_plan_is_deterministic_and_frozen(self):
        again = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-2', **PARAMS)
        strip = lambda jobs: [{k: v for k, v in j.items() if k not in ('input', 'geometry', 'prompt')} for j in jobs]
        self.assertEqual(strip(again['jobs']), strip(self.plan['jobs']))
        validate(self.plan, self.p)
        tampered = copy.deepcopy(self.plan); tampered['jobs'][1]['inventory'][0]['size_px'] = [99, 99]
        with self.assertRaisesRegex(ValueError, 'differs'): validate(tampered, self.p)
        with self.assertRaisesRegex(ValueError, 'Seam feather'):
            prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-3', **dict(PARAMS, seam_feather_px=2))
        with self.assertRaisesRegex(ValueError, 'anchor is none or full_view'):
            prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-4', **dict(PARAMS, anchor='tiles'))
        Image.fromarray(np.zeros((32, 32), np.uint8)).save(self.plan['structure']['path'])
        tampered = copy.deepcopy(self.plan); tampered['structure'] = record(self.plan['structure']['path'])
        with self.assertRaisesRegex(ValueError, 'structure classes differ'): validate(tampered, self.p)
        path = Path(self.plan['jobs'][0]['prompt']['path']); path.write_text(path.read_text(encoding='utf-8') + 'extra', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Missing or changed'): validate(self.plan, self.p)

    def test_plant_records_and_bloom_disposition_required(self):
        records = copy.deepcopy(RECORDS); del records['species']['Synthetic tiny cultivar']
        save(self.p / 'missing.json', records)
        with self.assertRaisesRegex(ValueError, 'common name'):
            prepare(self.p / 'finish-review.json', self.p / 'missing.json', self.p / 'bad', **PARAMS)
        records = copy.deepcopy(RECORDS); records['species']['Synthetic tiny cultivar']['bloom_exempt'] = 'no reason'
        save(self.p / 'both.json', records)
        with self.assertRaisesRegex(ValueError, 'bloom exemption'):
            prepare(self.p / 'finish-review.json', self.p / 'both.json', self.p / 'bad2', **PARAMS)

    def test_failed_previous_composite_cannot_be_planned(self):
        review = copy.deepcopy(self.fin.review); review['photo_review']['checks']['natural_plant_anatomy'] = 'fail'
        save(self.p / 'failed-review.json', review)
        with self.assertRaisesRegex(ValueError, 'anatomy or photo integration'):
            prepare(self.p / 'failed-review.json', self.p / 'records.json', self.p / 'bad', **PARAMS)


class AlignmentMetric(unittest.TestCase):
    """Calibration at the production scale: 1152x1024 tile, anchor at 0.256x of the photo."""
    F = .256

    @classmethod
    def setUpClass(cls):
        big = leaf_scene(3)
        small = cv2.resize(big, (round(big.shape[1] * cls.F), round(big.shape[0] * cls.F)), interpolation=cv2.INTER_AREA)
        cls.anchor = np.array(Image.fromarray(small).resize((big.shape[1], big.shape[0]), Image.LANCZOS))
        cls.mask = np.ones((1024, 1152), bool)

    def score(self, image):
        crop = lambda a: a[100:1124, 100:1252]
        return alignment(crop(image), crop(self.anchor), self.mask, self.F)

    def test_detail_pass_with_small_jitter_is_aligned(self):
        for jitter, size in ((2, .08), (4, .15)):
            result = self.score(leaf_scene(3, jitter, size, texture=True))
            self.assertTrue(result['aligned'], result)
            self.assertGreater(result['global_ncc'], .75); self.assertLessEqual(result['low_block_fraction'], .05)

    def test_moved_silhouettes_are_rejected(self):
        for jitter in (12, 20):
            result = self.score(leaf_scene(3, jitter, .2, texture=True))
            self.assertFalse(result['aligned'], result); self.assertLess(result['global_ncc'], ALIGNMENT['global_min'])
        shifted = np.roll(leaf_scene(3, texture=True), 24, axis=1)
        self.assertFalse(self.score(shifted)['aligned'])

    def test_local_shift_is_caught_by_blocks(self):
        result = self.score(leaf_scene(3, 25, .2, texture=True, where=lambda x, y: x < 700 and y < 650))
        self.assertGreater(result['global_ncc'], ALIGNMENT['global_min'])  # the global score alone would pass it
        self.assertGreater(result['low_block_fraction'], ALIGNMENT['max_low_block_fraction'])
        self.assertFalse(result['aligned'])

    def test_flat_content_is_aligned(self):
        flat = np.full((60, 70, 3), 90, np.uint8)
        self.assertTrue(alignment(flat, flat + 20, np.ones((60, 70), bool), .5)['aligned'])


class CompositeStructureMetric(unittest.TestCase):
    """composite_structure_v1 at the production scale (1152x1024 tile vs its CG composite crop)."""

    @classmethod
    def setUpClass(cls):
        cls.scenes = {}
        for seed in (1, 2):
            comp, ids, slab = garden(seed)
            cls.scenes[seed] = (comp, structure_classes(np.ones(ids.shape, bool), ids > 0, ~slab))

    def check(self, seed, **kw):
        comp, classes = self.scenes[seed]
        return composite_alignment(garden(seed, photo=True, **kw)[0], comp, classes)

    def test_photographic_rerender_with_movement_and_fuller_foliage_is_aligned(self):
        for seed in self.scenes:
            for move, fill in ((0, 0), (8, .3), (12, .4)):
                result = self.check(seed, move=move, fill=fill, shift=(2, 3) if move == 8 else (0, 0))
                self.assertTrue(result['aligned'], result)
                self.assertGreater(result['lenient']['global_ncc'], .75); self.assertGreater(result['strict']['worst_block_ncc'], .45)

    def test_gross_shift_and_relayout_are_rejected(self):
        for seed in self.scenes:
            result = self.check(seed, shift=(0, 40))
            self.assertIn('lenient_global_ncc', result['failed']); self.assertIn('strict_global_ncc', result['failed'])
            self.assertLess(result['lenient']['global_ncc'], COMPOSITE_ALIGNMENT['lenient_global_min'])
            self.assertIn('lenient_global_ncc', self.check(seed, layout=99)['failed'])

    def test_hardscape_lines_moved_or_added_are_rejected(self):
        for seed in self.scenes:
            moved = self.check(seed, move_joint=24)
            self.assertEqual(moved['failed'], ['strict_block_ncc'], moved)
            self.assertLess(moved['strict']['worst_block_ncc'], .1)
            added = self.check(seed, add_joint=True)
            self.assertEqual(added['failed'], ['strict_added_structure'], added)
            self.assertIn('strict_block_ncc', self.check(seed, slab_dy=-30)['failed'])

    def test_invented_plant_in_open_ground_is_rejected(self):
        comp, classes = self.scenes[1]
        ys, xs = np.nonzero(classes == OPEN_GROUND)
        spot = (int(np.median(xs)), int(np.median(ys)), 90)
        self.assertIn('open_ground_added_structure', self.check(1, add=spot)['failed'])

    def test_flat_and_identical_content(self):
        flat = np.full((60, 70, 3), 90, np.uint8)
        classes = np.full((60, 70), STRICT, np.uint8); classes[:, :30] = LENIENT
        self.assertTrue(composite_alignment(flat, flat + 20, classes)['aligned'])
        comp, classes = self.scenes[2]
        same = composite_alignment(comp, comp, classes)
        self.assertTrue(same['aligned']); self.assertEqual(same['lenient']['global_ncc'], 1.0)


class OwnershipAndSeams(unittest.TestCase):
    PLAN = {'jobs': [{'id': 'full', 'kind': 'full_view', 'box': [0, 0, 120, 60]},
                     {'id': 'tile-01', 'kind': 'tile', 'box': [0, 0, 70, 60]},
                     {'id': 'tile-02', 'kind': 'tile', 'box': [50, 0, 120, 60]}]}

    def test_each_pixel_has_one_owner_and_a_narrow_feather(self):
        labels = ownership(self.PLAN, 60, 120)
        self.assertTrue(np.all(labels[:, :60] == 0) and np.all(labels[:, 60:] == 1))  # boundary at the overlap midpoint
        weights = label_weights(labels, self.PLAN, 8)
        total = np.zeros((60, 120), np.float32); active = np.zeros((60, 120), int)
        for _, (x0, y0, x1, y1), w in weights:
            total[y0:y1, x0:x1] += w; active[y0:y1, x0:x1] += w > 0
        self.assertTrue(np.allclose(total, 1, atol=1e-5))
        self.assertTrue(np.all(active[:, :56] == 1) and np.all(active[:, 64:] == 1))  # one source outside 8 px band
        self.assertTrue(np.all(active[:, 57:63] == 2))

    def test_min_cost_seam_runs_where_neighbours_agree(self):
        scene = blocks_image((120, 60), 5)
        other = scene.copy(); other[:, 50:62] = blocks_image((12, 60), 6)  # tile-02 disagrees in the left of the overlap
        images = {0: scene, 1: other}
        source = lambda k, box: images[k][box[1]:box[3], box[0]:box[2]]
        labels = ownership(self.PLAN, 60, 120, source, np.ones((60, 120), bool), 4)
        cut = (labels == 0).sum(axis=1)  # columns owned by tile-01 per row
        self.assertTrue(np.all((cut >= 62) & (cut <= 66)), cut)  # nearest centre would cut at 60
        self.assertTrue(np.all(labels[:, :50] == 0) and np.all(labels[:, 70:] == 1))
        same = ownership(self.PLAN, 60, 120, lambda k, box: source(0, box), np.ones((60, 120), bool), 4)
        self.assertTrue(np.array_equal(same, ownership(self.PLAN, 60, 120)))  # agreement everywhere: midline
        plan = {'jobs': [{'id': 't1', 'kind': 'tile', 'box': [0, 0, 40, 70]}, {'id': 't2', 'kind': 'tile', 'box': [0, 50, 40, 120]}]}
        tall = np.transpose(scene[:40], (1, 0, 2)).copy(); low = tall.copy(); low[50:62] = 0
        images2 = {0: tall, 1: low}
        rows = ownership(plan, 120, 40, lambda k, box: images2[k][box[1]:box[3], box[0]:box[2]], np.ones((120, 40), bool), 4)
        self.assertTrue(np.all(((rows == 0).sum(axis=0) >= 62) & ((rows == 0).sum(axis=0) <= 66)))
        self.assertIsNone(seam_path(np.zeros((5, 6)), 3))

    def test_disagreeing_neighbours_are_flagged(self):
        labels = ownership(self.PLAN, 60, 120); region = np.ones((60, 120), bool)
        scene = blocks_image((120, 60), 5)
        moved = np.roll(scene, 4, axis=1)

        def sources(images):
            return lambda k, box: images[k][box[1]:box[3], box[0]:box[2]]
        same = seam_rows(labels, self.PLAN, region, sources({0: scene, 1: scene}), [True, True], 1., 4)
        self.assertEqual(len(same), 1); self.assertFalse(same[0]['likely_ghosting']); self.assertEqual(same[0]['edge_ncc'], 1.0)
        ghost = seam_rows(labels, self.PLAN, region, sources({0: scene, 1: moved}), [True, True], 1., 4)
        self.assertTrue(ghost[0]['likely_ghosting']); self.assertGreater(ghost[0]['mean_abs_luma_diff'], 10)
        self.assertEqual(ghost[0]['bbox'], [52, 0, 68, 60])


class SceneEditCompose(PlanFixture):
    """Default route: tiles edit their composite crop; the composite is the base and fills dropped tiles."""
    OFFSET = 30

    def setUp(self):
        super().setUp()
        self.raw = self.p / 'raw'; self.raw.mkdir()
        self.composite8 = np.rint(load(self.plan['composite']['path'])[0] / 257).astype(np.uint8)
        self.manifest = self.make_manifest()
        save(self.raw / 'attempts.json', self.manifest)
        self.proof = compose(self.p / 'scene/plan.json', self.raw / 'attempts.json', self.p / 'scene-out')
        self.review = self.make_review(self.proof, 'scene-out')

    def tile_raw(self, job, bad=False):
        """A photographic edit of the composite crop: same structure, brighter (so its pixels are recognizable).
        bad=True returns unrelated structured content (a re-laid-out tile)."""
        x0, y0, x1, y1 = job['box']
        crop = (blocks_image((x1 - x0, y1 - y0), 9, cell=3) if bad else self.composite8[y0:y1, x0:x1]).astype(np.int16) + self.OFFSET
        return np.array(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((19, 14), Image.LANCZOS))  # 1.357 vs 1.333

    def make_manifest(self, bad=(), name=''):
        manifest = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen', 'jobs': {}}
        for job in self.tiles:
            tid = job['id']
            Image.fromarray(self.tile_raw(job, tid in bad)).save(self.raw / f'{tid}-1{name}.png')
            (self.raw / f'{tid}-1.txt').write_text(text_of(job['prompt']['path']), encoding='utf-8')
            attempt = {'raw': f'{tid}-1{name}.png', 'prompt': f'{tid}-1.txt', 'inspection': 'Synthetic.'}
            if tid == self.tiles[0]['id']:  # the input may be recorded; it must be the planned composite crop
                attempt['input'] = '../' + Path(job['input']['path']).relative_to(self.p).as_posix()
            manifest['jobs'][tid] = {'attempts': [attempt], 'selected': 1}
        return manifest

    def make_review(self, proof, out):
        return {'schema_version': 4, 'method': 'scene_photographic_edit', 'handoff': record(self.p / 'handoff.json'),
                'camera_id': 'front', 'baseline_review': record(self.p / 'finish-review.json'),
                'candidate': proof['candidate'], 'scene_edit_proof': record(self.p / out / 'scene-edit-proof.json'),
                'model': {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen'},
                'scene_review': {'reviewer': 'vision-capable-agent', 'source_and_final_loaded': True, 'native_size_crops_loaded': True,
                                 'observations': 'Synthetic fixture truth, not actual photo recognition.',
                                 'paired_evidence': [proof['candidate']], 'check_set': CHECK_SET, 'checks': dict(ALL_PASS),
                                 'defects': [], 'defect_count': 0, 'metrics_inspected': True,
                                 'jobs': [{'id': j['id'], 'attempts': 1, 'selected_attempt': 1, 'blocking': 'pass',
                                           'observations': 'Synthetic.', 'evidence': proof['job_proofs'][j['id']]}
                                          for j in self.plan['jobs']]}}

    def rewrite(self, manifest, name):
        save(self.raw / name, manifest); return self.raw / name

    def compose_with(self, manifest, name):
        return compose(self.p / 'scene/plan.json', self.rewrite(manifest, name + '.json'), self.p / name)

    def interior(self, labels, k, margin):
        inside = cv2.distanceTransform((labels == k + 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return inside > margin

    def test_exact_outside_region_and_owner_tiles_inside(self):
        self.assertEqual(self.proof['base'], {'source': 'previous_composite'})
        self.assertEqual(self.proof['ownership_boundary']['method'], 'min_cost_seam_v1')
        self.assertEqual(self.proof['alignment']['metric'], 'composite_structure_v1')
        final, _ = load(self.proof['candidate']['path'])
        previous, _ = load(self.plan['composite']['path'])
        cleaned, _ = load(self.plan['cleaned_photo']['path'])
        region = gray_of(self.plan['region']['path']) > 0
        self.assertTrue(np.any(~region))
        self.assertTrue(np.array_equal(final[~region], previous[~region]))
        self.assertTrue(np.array_equal(final[~region], cleaned[~region]))
        self.assertFalse(np.array_equal(final[region], previous[region]))
        self.assertEqual(self.proof['operation'], OPERATION)
        self.assertEqual(self.proof['used_tiles'], [t['id'] for t in self.tiles])
        self.assertTrue(all(t['aligned'] and t['used'] for t in self.proof['alignment']['tiles']))
        self.assertEqual(self.proof['seams']['flagged'], 0)
        weight = gray_of(self.proof['region_weight']['path'])
        self.assertTrue(np.all(weight[~region] == 0)); self.assertEqual(weight[16, 16], 255)
        verify(self.proof, self.p)

    def test_ownership_partition_has_no_double_content(self):
        final, _ = load(self.proof['candidate']['path'])
        labels = gray_of(self.proof['ownership']['path'])
        weight = gray_of(self.proof['region_weight']['path'])
        checked_pixels = 0
        for k, job in enumerate(self.tiles):
            x0, y0, x1, y1 = job['box']
            own = resized(self.raw / f"{job['id']}-1.png", job['box'])
            core = self.interior(labels, k, PARAMS['seam_feather_px'] / 2 + .5) & (weight == 255)
            ys, xs = np.nonzero(core)
            self.assertTrue(np.all((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)))
            got = final[ys, xs].astype(np.int32); want = own[ys - y0, xs - x0].astype(np.int32) * 257
            self.assertLessEqual(int(np.abs(got - want).max(initial=0)), 257)  # exactly the owner's pixels, no blend
            checked_pixels += len(ys)
        self.assertGreater(checked_pixels, 100)

    def test_dropped_tile_keeps_the_composite_pixels(self):
        bad = self.tiles[4]['id']
        proof = self.compose_with(self.make_manifest({bad}, '-bad'), 'dropped')
        row = next(t for t in proof['alignment']['tiles'] if t['id'] == bad)
        self.assertFalse(row['aligned']); self.assertFalse(row['used']); self.assertIn('lenient_global_ncc', row['failed'])
        self.assertTrue(all(t['aligned'] for t in proof['alignment']['tiles'] if t['id'] != bad))
        self.assertNotIn(bad, proof['used_tiles']); self.assertEqual(len(proof['used_tiles']), len(self.tiles) - 1)
        final, _ = load(proof['candidate']['path'])
        previous, _ = load(self.plan['composite']['path'])
        core = self.interior(gray_of(proof['ownership']['path']), 4, PARAMS['seam_feather_px'] / 2 + .5)
        self.assertGreater(int(core.sum()), 4)
        self.assertLessEqual(int(np.abs(final[core].astype(np.int32) - previous[core].astype(np.int32)).max()), 1)
        result = check(self.make_review(proof, 'dropped'), self.p)
        self.assertEqual(result['tile_alignment'][bad], {'aligned': False, 'used': False, 'failed': row['failed']})
        verify(proof, self.p)

    def test_tile_attempts_edit_the_composite_crop(self):
        tile = self.tiles[0]['id']
        solid(self.raw / 'fake-input.png', (16, 12), [1, 2, 3])
        for key in ('input', 'reference'):
            manifest = copy.deepcopy(self.manifest); manifest['jobs'][tile]['attempts'][0][key] = 'fake-input.png'
            with self.assertRaisesRegex(ValueError, f'{key} is not the planned composite crop'): self.compose_with(manifest, 'fake-' + key)
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][tile]['attempts'][0]['anchor_from'] = 1
        with self.assertRaisesRegex(ValueError, 'has no anchor'): self.compose_with(manifest, 'anchor-from')
        manifest = copy.deepcopy(self.manifest); manifest['jobs']['full'] = {'attempts': [], 'selected': None}
        with self.assertRaisesRegex(ValueError, 'every planned scene edit job'): self.compose_with(manifest, 'extra-full')

    def test_review_accepts_with_recorded_defects(self):
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(result['method'], 'scene_photographic_edit')
        self.assertEqual(result['delivered'], 'scene_edit_composite'); self.assertTrue(result['photo_background_composited'])
        self.assertEqual(result['model']['agent_model'], 'gpt-6-astra'); self.assertEqual(result['attempts'][self.tiles[0]['id']], 1)
        self.assertNotIn('full', result['attempts'])
        self.assertEqual(result['check_set'], CHECK_SET); self.assertEqual(result['defect_count'], 0)
        self.assertEqual(check(self.fin.review, self.p)['method'], 'photo_finish')  # previous route unchanged
        review = copy.deepcopy(self.review)
        review['scene_review']['defects'] = [
            {'type': 'extent_change', 'where': 'left agave', 'observation': 'crown about 10% wider'},
            {'type': 'cultivar_detail', 'where': 'fern, lower right', 'observation': 'stripe toward the leaf edge'}]
        review['scene_review']['defect_count'] = 2
        result = check(review, self.p)
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(result['limitations'], [])
        self.assertEqual(result['defect_count'], 2)
        self.assertEqual(result['non_blocking_observations'][0],
                         {'check': 'defect: extent_change', 'observations': 'left agave: crown about 10% wider', 'blocking': False})

    def test_invented_plant_or_missing_bloom_downgrades(self):
        for kind in ('invented_plant', 'bloom_missing'):
            review = copy.deepcopy(self.review)
            review['scene_review']['defects'] = [{'type': kind, 'where': 'bed edge, centre', 'observation': 'synthetic'}]
            review['scene_review']['defect_count'] = 1
            result = check(review, self.p)
            self.assertEqual(result['status'], 'completed_with_limitations'); self.assertEqual(result['delivered'], 'scene_edit_composite')
            self.assertEqual(result['limitations'][-1]['check'], f'defect: {kind}'); self.assertEqual(result['nonpassing_checks'], [])

    def test_defect_schema_is_required(self):
        for change, message in ((lambda v: v.pop('defects'), 'defects list'),
                                (lambda v: v.update(defect_count=3), 'defect_count'),
                                (lambda v: v.update(defects=[{'type': 'wrong_species', 'where': 'x', 'observation': 'y'}], defect_count=1), 'type'),
                                (lambda v: v.update(defects=[{'type': 'movement', 'observation': 'y'}], defect_count=1), 'where'),
                                (lambda v: v.update(check_set=None), 'check_set'),
                                (lambda v: v.update(checks=dict.fromkeys(SCENE_CHECKS_V1, 'pass')), 'exactly the blocking checks')):
            review = copy.deepcopy(self.review); change(review['scene_review'])
            with self.assertRaisesRegex(ValueError, message): check(review, self.p)

    def test_blocking_failure_falls_back_to_previous_composite(self):
        bad = copy.deepcopy(self.review); bad['scene_review']['checks']['plants_rooted_in_beds'] = 'fail'
        with self.assertRaisesRegex(ValueError, 'naming where'): check(bad, self.p)
        bad['scene_review']['checks']['plants_rooted_in_beds'] = {'result': 'fail', 'where': 'front step, left',
                                                                  'observation': 'Synthetic flower growing from the tread.'}
        with self.assertRaisesRegex(ValueError, 'fall back'): check(bad, self.p)
        bad['release_policy'] = 'fallback_previous_composite'
        bad['limitations'] = [{'check': 'plants_rooted_in_beds', 'observations': 'Synthetic flower growing from the step.'}]
        with self.assertRaisesRegex(ValueError, 'previous photo_finish composite'): check(bad, self.p)
        bad['candidate'] = self.fin.review['candidate']
        result = check(bad, self.p)
        self.assertEqual(result['status'], 'completed_with_limitations')
        self.assertEqual(result['delivered'], 'previous_photo_finish_composite')
        self.assertEqual(result['nonpassing_checks'], ['plants_rooted_in_beds'])
        bad['scene_review']['jobs'][0]['blocking'] = 'fail'  # job findings stay evidence in a fallback
        self.assertEqual(check(bad, self.p)['status'], 'completed_with_limitations')
        self.assertEqual(result['blocking_checks']['plants_rooted_in_beds'], 'fail')
        bad['limitations'] = []
        with self.assertRaisesRegex(ValueError, 'Describe each'): check(bad, self.p)
        bad['limitations'] = [{'check': 'plants_rooted_in_beds', 'observations': 'x'}, {'check': 'extent_change', 'observations': 'y'}]
        with self.assertRaisesRegex(ValueError, 'Describe each'): check(bad, self.p)

    def test_tile_blocking_failure_means_deselect_not_view_failure(self):
        bad = copy.deepcopy(self.review); bad['scene_review']['jobs'][1]['blocking'] = 'fail'
        with self.assertRaisesRegex(ValueError, 'deselect it'): check(bad, self.p)
        tile = self.tiles[0]['id']
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][tile]['selected'] = None
        proof = self.compose_with(manifest, 'deselected')
        final, _ = load(proof['candidate']['path']); previous, _ = load(self.plan['composite']['path'])
        core = self.interior(gray_of(proof['ownership']['path']), 0, PARAMS['seam_feather_px'] / 2 + .5)
        self.assertGreater(int(core.sum()), 4)
        self.assertLessEqual(int(np.abs(final[core].astype(np.int32) - previous[core].astype(np.int32)).max()), 1)
        review = self.make_review(proof, 'deselected')
        row = next(r for r in review['scene_review']['jobs'] if r['id'] == tile)
        row.update(selected_attempt=None, blocking='fail', observations='Synthetic: tile grew a plant on the step; deselected.')
        self.assertEqual(check(review, self.p)['status'], 'accepted')
        bad = copy.deepcopy(self.review); bad['scene_review']['jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'every scene edit job'): check(bad, self.p)

    def test_flagged_seams_need_inspection(self):
        proof = read_json(self.p / 'scene-out/scene-edit-proof.json')
        review = copy.deepcopy(self.review); del review['scene_review']['metrics_inspected']
        check(review, self.p)  # nothing flagged: no inspection flag needed
        orig = dict(sec.SEAMS_V3)
        sec.SEAMS_V3.update(flag_below=1.5, agree_luma_diff=0)  # every band now 'flagged'
        try:
            p2 = self.compose_with(self.manifest, 'all-flagged')
            self.assertEqual(p2['seams']['flagged'], len(p2['seams']['pairs']))
            review = self.make_review(p2, 'all-flagged'); del review['scene_review']['metrics_inspected']
            with self.assertRaisesRegex(ValueError, 'metrics_inspected'): check(review, self.p)
        finally:
            sec.SEAMS_V3.update(orig)
        self.assertGreater(len(proof['seams']['pairs']), 0)

    def test_fallback_without_any_scene_edit_is_disclosed(self):
        review = {k: v for k, v in self.review.items() if k not in ('scene_edit_proof', 'scene_review', 'model')}
        review.update(candidate=self.fin.review['candidate'], release_policy='fallback_previous_composite')
        with self.assertRaisesRegex(ValueError, 'disclosed fallback'): check(review, self.p)
        review['limitations'] = [{'check': 'scene_edit_not_run', 'observations': 'Synthetic: the image tool was unavailable.'}]
        self.assertEqual(check(review, self.p)['status'], 'completed_with_limitations')

    def test_model_and_tool_are_required(self):
        for field, value in [('agent_model', 'gpt-5'), ('reasoning_effort', 'medium'), ('image_tool', 'api')]:
            manifest = copy.deepcopy(self.manifest); manifest[field] = value
            with self.assertRaisesRegex(ValueError, 'gpt-6-astra|built-in image tool'):
                compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'model.json'), self.p / f'bad-{field}')
        bad = copy.deepcopy(self.review); bad['model']['reasoning_effort'] = 'low'
        with self.assertRaisesRegex(ValueError, 'gpt-6-astra/high'): check(bad, self.p)

    def test_attempt_rules(self):
        job = self.tiles[0]; tid = job['id']
        planned = text_of(job['prompt']['path'])
        (self.raw / 'changed.txt').write_text(planned + ' improvised', encoding='utf-8')
        (self.raw / 'second.txt').write_text(planned + '\nSECOND ATTEMPT: edit the ORIGINAL input; keep every agave rosette.', encoding='utf-8')
        ok = {'raw': f'{tid}-1.png', 'prompt': f'{tid}-1.txt', 'inspection': 'x'}
        cases = [
            ([dict(ok, prompt='changed.txt')], 'exact planned prompt'),
            ([{'raw': f'{tid}-1.png', 'prompt': f'{tid}-1.txt'}, {'raw': f'{tid}-1.png', 'prompt': 'second.txt'}], 'recorded inspection'),
            ([ok, {'raw': f'{tid}-1.png', 'prompt': f'{tid}-1.txt'}], 'adds a correction'),
            ([ok] * 3, 'At most 2')]
        for attempts, message in cases:
            manifest = copy.deepcopy(self.manifest); manifest['jobs'][tid]['attempts'] = attempts
            with self.assertRaisesRegex(ValueError, message):
                compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'attempts-bad.json'), self.p / 'bad')
        manifest = copy.deepcopy(self.manifest)
        manifest['jobs'][tid] = {'attempts': [dict(ok, inspection='Agave turned into pads.'), {'raw': f'{tid}-1.png', 'prompt': 'second.txt'}],
                                 'selected': 2}
        proof = compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'two.json'), self.p / 'two')
        self.assertEqual(len(proof['attempts']['jobs'][0]['attempts']), 2)
        manifest = copy.deepcopy(self.manifest)
        for t in self.tiles:
            manifest['jobs'][t['id']]['selected'] = None
        with self.assertRaisesRegex(ValueError, 'Select at least one'): self.compose_with(manifest, 'none-selected')

    def test_aspect_mismatch_rejected(self):
        solid(self.raw / 'wide.png', (32, 12), [0, 0, 0])
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][self.tiles[1]['id']]['attempts'][0]['raw'] = 'wide.png'
        with self.assertRaisesRegex(ValueError, 'aspect ratio'): self.compose_with(manifest, 'wide')

    def test_changed_candidate_metrics_or_raw_rejected(self):
        proof = copy.deepcopy(self.proof); proof['alignment']['tiles'][0]['lenient']['global_ncc'] = .1
        with self.assertRaisesRegex(ValueError, 'metrics differ'): verify(proof, self.p)
        proof = copy.deepcopy(self.proof); proof['alignment']['strict_block_min'] = .2
        with self.assertRaisesRegex(ValueError, 'metrics differ'): verify(proof, self.p)
        proof = copy.deepcopy(self.proof); proof['ownership_boundary']['centre_penalty'] = 1
        with self.assertRaisesRegex(ValueError, 'metrics differ'): verify(proof, self.p)
        final, _ = load(self.proof['candidate']['path']); final[0, 31] = 0
        save_png(self.proof['candidate']['path'], final)
        proof = copy.deepcopy(self.proof); proof['candidate'] = record(self.proof['candidate']['path'])
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'): verify(proof, self.p)
        tid = self.tiles[1]['id']
        solid(self.raw / f'{tid}-1.png', (19, 14), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Missing or changed'): verify(self.proof, self.p)


class AnchoredRoute(PlanFixture):
    """Optional anchor: full_view (schema 3) and legacy schema 2: tiles edit crops of the upscaled full-view result."""
    anchor = 'full_view'
    OFFSET = 30

    def setUp(self):
        super().setUp()
        self.raw = self.p / 'raw'; self.raw.mkdir()
        Image.fromarray(blocks_image((16, 16), 1, cell=2)).save(self.raw / 'full-1.png')  # half-size return, same aspect
        self.anchor_img = resized(self.raw / 'full-1.png', [0, 0, 32, 32])
        self.manifest = self.make_manifest()
        self.proof = compose(self.p / 'scene/plan.json', self.rewrite(self.manifest, 'attempts.json'), self.p / 'scene-out')

    def rewrite(self, manifest, name):
        save(self.raw / name, manifest); return self.raw / name

    def compose_with(self, manifest, name):
        return compose(self.p / 'scene/plan.json', self.rewrite(manifest, name + '.json'), self.p / name)

    def make_manifest(self, shifts=None, name=''):
        shifts = shifts or {}
        full = next(j for j in self.plan['jobs'] if j['kind'] == 'full_view')
        (self.raw / 'full-1.txt').write_text(text_of(full['prompt']['path']), encoding='utf-8')
        manifest = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen', 'jobs': {
            'full': {'attempts': [{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Synthetic.', 'blocking': ALL_PASS}],
                     'selected': 1}}}
        for job in self.tiles:
            tid = job['id']; x0, y0, x1, y1 = job['box']
            crop = np.roll(self.anchor_img, shifts.get(tid, 0), axis=1)[y0:y1, x0:x1].astype(np.int16) + self.OFFSET
            Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((19, 14), Image.LANCZOS).save(self.raw / f'{tid}-1{name}.png')
            (self.raw / f'{tid}-1.txt').write_text(text_of(job['prompt']['path']), encoding='utf-8')
            if not (self.raw / f'{tid}-input.png').exists():
                reference(self.p / 'scene/plan.json', self.raw / 'full-1.png', tid, self.raw / f'{tid}-input.png')
            manifest['jobs'][tid] = {'attempts': [{'raw': f'{tid}-1{name}.png', 'prompt': f'{tid}-1.txt',
                                                   'input': f'{tid}-input.png', 'anchor_from': 1, 'inspection': 'Synthetic.'}],
                                     'selected': 1}
        return manifest

    def test_anchored_plan_prompts_and_proof(self):
        self.assertEqual(self.plan['jobs'][0]['id'], 'full'); self.assertEqual(self.plan['jobs'][0]['box'], [0, 0, 32, 32])
        full_text = text_of(self.plan['jobs'][0]['prompt']['path'])
        self.assertIn('Make this exact view an ordinary real photograph', full_text)
        self.assertIn('flowers: none required; preserve foliage. Instances: agave-1 at [50, 50]%, size [20, 20]px', full_text)
        tile = self.tiles[0]
        self.assertIn('one native-resolution tile of an already-edited photograph', text_of(tile['prompt']['path']))
        self.assertIn('geometry', tile); self.assertNotIn('input', tile)
        self.assertEqual(self.proof['operation'], OPERATION)
        self.assertEqual(self.proof['base']['source'], 'anchor'); self.assertEqual(self.proof['alignment']['metric'], 'luma_edge_ncc_blocks_v1')
        self.assertEqual(self.proof['used_tiles'], [t['id'] for t in self.tiles])
        verify(self.proof, self.p)

    def test_shifted_tile_is_rejected_and_anchor_fills_its_area(self):
        bad = self.tiles[4]['id']
        proof = self.compose_with(self.make_manifest({bad: 6}, '-shift'), 'shifted')
        row = next(t for t in proof['alignment']['tiles'] if t['id'] == bad)
        self.assertFalse(row['aligned']); self.assertFalse(row['used']); self.assertLess(row['global_ncc'], ALIGNMENT['global_min'])
        self.assertNotIn(bad, proof['used_tiles'])
        manifest = copy.deepcopy(self.manifest)
        for job in self.tiles:
            manifest['jobs'][job['id']]['selected'] = None
        anchor_only, _ = load(self.compose_with(manifest, 'anchor-only')['candidate']['path'])
        final, _ = load(proof['candidate']['path'])
        inside = cv2.distanceTransform((gray_of(proof['ownership']['path']) == 5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        core = inside > PARAMS['seam_feather_px'] / 2 + .5
        self.assertGreater(int(core.sum()), 4)
        self.assertTrue(np.array_equal(final[core], anchor_only[core]))
        verify(proof, self.p)

    def test_anchor_gates_the_tiles(self):
        manifest = copy.deepcopy(self.manifest); manifest['jobs']['full']['attempts'][0]['blocking'] = dict(ALL_PASS, hardscape_unchanged='fail')
        with self.assertRaisesRegex(ValueError, 'anchor failed a blocking check'): self.compose_with(manifest, 'anchor-fail')
        manifest = copy.deepcopy(self.manifest); del manifest['jobs']['full']['attempts'][0]['blocking']
        with self.assertRaisesRegex(ValueError, 'Review the full-view anchor'): self.compose_with(manifest, 'anchor-unreviewed')
        manifest = copy.deepcopy(self.manifest); manifest['jobs']['full']['selected'] = None
        with self.assertRaisesRegex(ValueError, 'accepted full-view anchor'): self.compose_with(manifest, 'no-anchor')
        tile = self.tiles[0]['id']
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][tile]['attempts'][0]['anchor_from'] = 2
        with self.assertRaisesRegex(ValueError, 'selected full-view anchor'): self.compose_with(manifest, 'anchor-from')
        solid(self.raw / 'fake-input.png', (16, 12), [1, 2, 3])
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][tile]['attempts'][0]['input'] = 'fake-input.png'
        with self.assertRaisesRegex(ValueError, 'not the anchor crop'): self.compose_with(manifest, 'fake-input')


class LegacySchema2(AnchoredRoute):
    """Schema 2 plans (lab P2: always anchored, nearest-centre ownership) and their proofs stay verifiable."""
    schema = 2

    def test_anchored_plan_prompts_and_proof(self):
        self.assertEqual(self.plan['schema_version'], 2); self.assertNotIn('structure', self.plan)
        self.assertEqual(self.proof['operation'], OPERATION_V2); self.assertIn('anchor', self.proof); self.assertNotIn('base', self.proof)
        labels = gray_of(self.proof['ownership']['path']).astype(np.int16) - 1
        self.assertTrue(np.array_equal(labels, ownership(self.plan, 32, 32)))  # nearest centre, no seam search
        verify(self.proof, self.p)


def read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))


class LegacySchema1(PlanFixture):
    """Schema 1 plans, proofs and reviews (independent tiles, seven checks) stay verifiable."""
    schema = 1

    def setUp(self):
        super().setUp()
        self.raw = self.p / 'raw'; self.raw.mkdir()
        manifest = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen', 'jobs': {}}
        solid(self.raw / 'full-1.png', (16, 16), [200, 40, 40])
        (self.raw / 'full-1.txt').write_text(text_of(self.plan['jobs'][0]['prompt']['path']), encoding='utf-8')
        manifest['jobs']['full'] = {'attempts': [{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Synthetic.'}], 'selected': 1}
        for i, job in enumerate(self.tiles):
            solid(self.raw / f"{job['id']}-1.png", (19, 14), [40, 40 + 20 * i, 200])
            (self.raw / f"{job['id']}-1.txt").write_text(text_of(job['prompt']['path']), encoding='utf-8')
            reference(self.p / 'scene/plan.json', self.raw / 'full-1.png', job['id'], self.raw / f"{job['id']}-ref.png")
            manifest['jobs'][job['id']] = {'attempts': [{'raw': f"{job['id']}-1.png", 'prompt': f"{job['id']}-1.txt",
                                           'reference': f"{job['id']}-ref.png", 'reference_from': 1, 'inspection': 'Synthetic.'}], 'selected': 1}
        save(self.raw / 'attempts.json', manifest)
        self.proof = compose(self.p / 'scene/plan.json', self.raw / 'attempts.json', self.p / 'scene-out')
        self.review = {'schema_version': 4, 'method': 'scene_photographic_edit', 'handoff': record(self.p / 'handoff.json'),
                       'camera_id': 'front', 'baseline_review': record(self.p / 'finish-review.json'),
                       'candidate': self.proof['candidate'], 'scene_edit_proof': record(self.p / 'scene-out/scene-edit-proof.json'),
                       'model': {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen'},
                       'scene_review': {'reviewer': 'vision-capable-agent', 'source_and_final_loaded': True, 'native_size_crops_loaded': True,
                           'observations': 'Synthetic fixture truth, not actual photo recognition.',
                           'paired_evidence': [self.proof['candidate']], 'checks': dict.fromkeys(SCENE_CHECKS_V1, 'pass'),
                           'jobs': [{'id': j['id'], 'attempts': 1, 'selected_attempt': 1, 'species_fidelity': 'pass',
                                     'invented_or_hybrid_plants': 'none', 'observations': 'Synthetic.',
                                     'evidence': self.proof['job_proofs'][j['id']]} for j in self.plan['jobs']]}}

    def test_legacy_plan_proof_and_review_verify(self):
        self.assertEqual(self.plan['schema_version'], 1); self.assertIn('input', self.tiles[0])
        self.assertIn('matching crop of the already-edited whole view', text_of(self.tiles[0]['prompt']['path']))
        self.assertEqual(self.proof['operation'], OPERATION_V1)
        final, _ = load(self.proof['candidate']['path'])
        self.assertEqual(int(final[16, 16, 0]) // 257, 40)  # raised-cosine tiles own the interior
        verify(self.proof, self.p)
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'accepted'); self.assertNotIn('check_set', result)

    def test_legacy_gates_unchanged(self):
        bad = copy.deepcopy(self.review); bad['scene_review']['jobs'][1]['invented_or_hybrid_plants'] = 'present'
        with self.assertRaisesRegex(ValueError, 'species fidelity'): check(bad, self.p)
        bad = copy.deepcopy(self.review); bad['scene_review']['checks']['flowering_in_bloom'] = 'fail'
        bad.update(release_policy='fallback_previous_composite', candidate=self.fin.review['candidate'],
                   limitations=[{'check': 'flowering_in_bloom', 'observations': 'Synthetic.'}])
        self.assertEqual(check(bad, self.p)['nonpassing_checks'], ['flowering_in_bloom'])
        bad = copy.deepcopy(self.review); bad['scene_review']['check_set'] = CHECK_SET
        with self.assertRaisesRegex(ValueError, 'original review checks'): check(bad, self.p)


if __name__ == '__main__':
    unittest.main()
