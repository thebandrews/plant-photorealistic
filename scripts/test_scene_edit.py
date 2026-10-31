"""Synthetic scene photographic edit tests: layout, prompts, anchoring, ownership, metrics and review gates.
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
from scene_edit_compose import (ALIGNMENT, BLOCKING_CHECKS, CHECK_SET, OPERATION, OPERATION_V1, SCENE_CHECKS_V1,
                                alignment, compose, label_weights, ownership, seam_rows, verify)
from scene_edit_plan import prepare, reference, tiles, validate, edit_region, region_weight, resized
from test_contract import save
import test_finishing

PARAMS = dict(tile=[16, 12], overlap=.3, margin_px=2, feather_px=2, seam_feather_px=4)
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
    schema = 2

    def setUp(self):
        self.fin = test_finishing.PhotoFinishing(); self.fin.setUp(); self.addCleanup(self.fin.doCleanups)
        self.p = self.fin.p
        save(self.p / 'finish-review.json', self.fin.review)
        save(self.p / 'records.json', RECORDS)
        self.plan = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene', schema=self.schema,
                            **(PARAMS if self.schema == 2 else PARAMS_V1))
        self.tiles = [j for j in self.plan['jobs'] if j['kind'] == 'tile']


class SceneEditPlan(PlanFixture):
    def test_layout_covers_region_with_thirty_percent_overlap(self):
        plan = self.plan
        self.assertEqual(plan['schema_version'], 2)
        self.assertEqual(plan['jobs'][0]['id'], 'full'); self.assertEqual(plan['jobs'][0]['box'], [0, 0, 32, 32])
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

    def test_prompts_carry_species_inventory_bloom_and_fragments(self):
        full = self.plan['jobs'][0]
        ids = [r['id'] for r in full['inventory']]
        self.assertEqual(ids, ['agave-1', 'tiny'])  # hidden plant has no visible pixels
        tiny = full['inventory'][1]
        self.assertEqual(tiny['size_px'], [1, 1]); self.assertEqual(tiny['position_percent'], [11, 11])
        text = text_of(full['prompt']['path'])
        self.assertIn('Make this exact view an ordinary real photograph', text)
        self.assertIn('shredded bark mulch', text)
        self.assertIn('never growing out of concrete, paving, steps or walls', text)
        self.assertIn('natural movement, fuller foliage and a natural outline are fine', text)
        self.assertNotIn('No plant may move', text)
        self.assertIn("Agave victoriae-reginae (Queen Victoria agave; compact dense rosette", text)
        self.assertIn('flowers: none required; preserve foliage. Instances: agave-1 at [50, 50]%, size [20, 20]px', text)
        self.assertIn('flowers: many small white daisies with yellow centres. Instances: tiny at [11, 11]%, size [1, 1]px', text)

    def test_tiles_are_anchor_detail_passes(self):
        tile = self.tiles[0]
        text = text_of(tile['prompt']['path'])
        self.assertIn('one native-resolution tile of an already-edited photograph', text)
        self.assertIn('Do not change any shape, outline, position, size or species', text)
        self.assertIn('planned 3D composite of the same crop: use it only to confirm', text)
        self.assertIn('geometry', tile); self.assertNotIn('input', tile)  # the edit target is the anchor crop
        self.assertTrue(Path(tile['geometry']['path']).name == 'geometry.png')
        self.assertIn('input', self.plan['jobs'][0])

    def test_plan_is_deterministic_and_frozen(self):
        again = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-2', **PARAMS)
        strip = lambda jobs: [{k: v for k, v in j.items() if k not in ('input', 'geometry', 'prompt')} for j in jobs]
        self.assertEqual(strip(again['jobs']), strip(self.plan['jobs']))
        validate(self.plan, self.p)
        tampered = copy.deepcopy(self.plan); tampered['jobs'][1]['inventory'][0]['size_px'] = [99, 99]
        with self.assertRaisesRegex(ValueError, 'differs'): validate(tampered, self.p)
        with self.assertRaisesRegex(ValueError, 'Seam feather'):
            prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-3', **dict(PARAMS, seam_feather_px=2))
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
    OFFSET = 30

    def setUp(self):
        super().setUp()
        self.raw = self.p / 'raw'; self.raw.mkdir()
        self.anchor_small = blocks_image((16, 16), 1, cell=2)  # half-size return, same aspect
        Image.fromarray(self.anchor_small).save(self.raw / 'full-1.png')
        self.anchor = resized(self.raw / 'full-1.png', [0, 0, 32, 32])
        self.manifest = self.make_manifest()
        save(self.raw / 'attempts.json', self.manifest)
        self.proof = compose(self.p / 'scene/plan.json', self.raw / 'attempts.json', self.p / 'scene-out')
        self.review = self.make_review(self.proof, 'scene-out')

    def tile_raw(self, job, shift=0):
        """A detail pass over the anchor crop: same edges, brighter (so its pixels are recognizable)."""
        x0, y0, x1, y1 = job['box']
        crop = np.roll(self.anchor, shift, axis=1)[y0:y1, x0:x1].astype(np.int16) + self.OFFSET
        return np.array(Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((19, 14), Image.LANCZOS))  # 1.357 vs 1.333

    def make_manifest(self, shifts=None, name=''):
        shifts = shifts or {}
        (self.raw / 'full-1.txt').write_text(text_of(self.plan['jobs'][0]['prompt']['path']), encoding='utf-8')
        manifest = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen', 'jobs': {
            'full': {'attempts': [{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Synthetic.', 'blocking': ALL_PASS}],
                     'selected': 1}}}
        for job in self.tiles:
            tid = job['id']
            Image.fromarray(self.tile_raw(job, shifts.get(tid, 0))).save(self.raw / f'{tid}-1{name}.png')
            (self.raw / f'{tid}-1.txt').write_text(text_of(job['prompt']['path']), encoding='utf-8')
            if not (self.raw / f'{tid}-input.png').exists():
                reference(self.p / 'scene/plan.json', self.raw / 'full-1.png', tid, self.raw / f'{tid}-input.png')
            manifest['jobs'][tid] = {'attempts': [{'raw': f'{tid}-1{name}.png', 'prompt': f'{tid}-1.txt',
                                                   'input': f'{tid}-input.png', 'anchor_from': 1, 'inspection': 'Synthetic.'}],
                                     'selected': 1}
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

    def test_shifted_tile_is_rejected_and_anchor_fills_its_area(self):
        bad = self.tiles[4]['id']
        proof = self.compose_with(self.make_manifest({bad: 6}, '-shift'), 'shifted')
        row = next(t for t in proof['alignment']['tiles'] if t['id'] == bad)
        self.assertFalse(row['aligned']); self.assertFalse(row['used']); self.assertLess(row['global_ncc'], ALIGNMENT['global_min'])
        self.assertNotIn(bad, proof['used_tiles']); self.assertEqual(len(proof['used_tiles']), len(self.tiles) - 1)
        manifest = copy.deepcopy(self.manifest)
        for job in self.tiles:
            manifest['jobs'][job['id']]['selected'] = None
        anchor_only, _ = load(self.compose_with(manifest, 'anchor-only')['candidate']['path'])
        final, _ = load(proof['candidate']['path'])
        core = self.interior(gray_of(proof['ownership']['path']), 4, PARAMS['seam_feather_px'] / 2 + .5)
        self.assertGreater(int(core.sum()), 4)
        self.assertTrue(np.array_equal(final[core], anchor_only[core]))
        result = check(self.make_review(proof, 'shifted'), self.p)
        self.assertEqual(result['tile_alignment'][bad], {'aligned': False, 'used': False, 'global_ncc': row['global_ncc'],
                                                         'low_block_fraction': row['low_block_fraction']})
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

    def test_review_accepts_with_recorded_defects(self):
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(result['method'], 'scene_photographic_edit')
        self.assertEqual(result['delivered'], 'scene_edit_composite'); self.assertTrue(result['photo_background_composited'])
        self.assertEqual(result['model']['agent_model'], 'gpt-6-astra'); self.assertEqual(result['attempts']['full'], 1)
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
        orig = sec.SEAMS['flag_below']
        sec.SEAMS['flag_below'] = 1.5  # every band now 'flagged'
        try:
            p2 = self.compose_with(self.manifest, 'all-flagged')
            self.assertEqual(p2['seams']['flagged'], len(p2['seams']['pairs']))
            review = self.make_review(p2, 'all-flagged'); del review['scene_review']['metrics_inspected']
            with self.assertRaisesRegex(ValueError, 'metrics_inspected'): check(review, self.p)
        finally:
            sec.SEAMS['flag_below'] = orig
        self.assertGreater(len(proof['seams']['pairs']), 0)

    def test_fallback_without_any_scene_edit_is_disclosed(self):
        review = {k: v for k, v in self.review.items() if k not in ('scene_edit_proof', 'scene_review', 'model')}
        review.update(candidate=self.fin.review['candidate'], release_policy='fallback_previous_composite')
        with self.assertRaisesRegex(ValueError, 'disclosed fallback'): check(review, self.p)
        review['limitations'] = [{'check': 'hardscape_unchanged', 'observations': 'Synthetic: the full-view anchor moved the step edge twice.'}]
        self.assertEqual(check(review, self.p)['status'], 'completed_with_limitations')

    def test_model_and_tool_are_required(self):
        for field, value in [('agent_model', 'gpt-5'), ('reasoning_effort', 'medium'), ('image_tool', 'api')]:
            manifest = copy.deepcopy(self.manifest); manifest[field] = value
            with self.assertRaisesRegex(ValueError, 'gpt-6-astra|built-in image tool'):
                compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'model.json'), self.p / f'bad-{field}')
        bad = copy.deepcopy(self.review); bad['model']['reasoning_effort'] = 'low'
        with self.assertRaisesRegex(ValueError, 'gpt-6-astra/high'): check(bad, self.p)

    def test_attempt_rules(self):
        planned = text_of(self.plan['jobs'][0]['prompt']['path'])
        (self.raw / 'changed.txt').write_text(planned + ' improvised', encoding='utf-8')
        (self.raw / 'second.txt').write_text(planned + '\nSECOND ATTEMPT: edit the ORIGINAL input; keep every agave rosette.', encoding='utf-8')
        ok = {'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'x', 'blocking': ALL_PASS}
        cases = [
            ([dict(ok, prompt='changed.txt')], 'exact planned prompt'),
            ([{'raw': 'full-1.png', 'prompt': 'full-1.txt'}, {'raw': 'full-1.png', 'prompt': 'second.txt'}], 'recorded inspection'),
            ([ok, {'raw': 'full-1.png', 'prompt': 'full-1.txt'}], 'adds a correction'),
            ([ok] * 3, 'At most 2'),
            ([dict(ok, blocking={'plants_rooted_in_beds': 'pass'})], 'each blocking check')]
        for attempts, message in cases:
            manifest = copy.deepcopy(self.manifest); manifest['jobs']['full']['attempts'] = attempts
            with self.assertRaisesRegex(ValueError, message):
                compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'attempts-bad.json'), self.p / 'bad')
        manifest = copy.deepcopy(self.manifest)
        manifest['jobs']['full'] = {'attempts': [{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Step edge doubled.',
                                                  'blocking': dict(ALL_PASS, hardscape_unchanged='fail')},
                                                 {'raw': 'full-1.png', 'prompt': 'second.txt', 'blocking': ALL_PASS}], 'selected': 2}
        for job in self.tiles:
            manifest['jobs'][job['id']] = {'attempts': [], 'selected': None}
        proof = compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'two.json'), self.p / 'two')
        self.assertEqual(len(proof['attempts']['jobs'][0]['attempts']), 2); self.assertEqual(proof['anchor']['attempt'], 2)

    def test_aspect_mismatch_rejected(self):
        solid(self.raw / 'wide.png', (32, 16), [0, 0, 0])
        manifest = copy.deepcopy(self.manifest); manifest['jobs']['full']['attempts'][0]['raw'] = 'wide.png'
        with self.assertRaisesRegex(ValueError, 'aspect ratio'): self.compose_with(manifest, 'wide')

    def test_changed_candidate_metrics_or_raw_rejected(self):
        proof = copy.deepcopy(self.proof); proof['alignment']['tiles'][0]['global_ncc'] = .1
        with self.assertRaisesRegex(ValueError, 'metrics differ'): verify(proof, self.p)
        proof = copy.deepcopy(self.proof); proof['alignment']['global_min'] = .2
        with self.assertRaisesRegex(ValueError, 'metrics differ'): verify(proof, self.p)
        final, _ = load(self.proof['candidate']['path']); final[0, 31] = 0
        save_png(self.proof['candidate']['path'], final)
        proof = copy.deepcopy(self.proof); proof['candidate'] = record(self.proof['candidate']['path'])
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'): verify(proof, self.p)
        solid(self.raw / 'full-1.png', (16, 16), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Missing or changed'): verify(self.proof, self.p)


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
