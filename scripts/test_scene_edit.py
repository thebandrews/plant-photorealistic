"""Synthetic scene photographic edit tests: the one full-view job (input reduction, locations, prompt), reference photo
records and the view's sheet, compose into the feathered region, the side-by-side compare, and the review that always
delivers and only records real-failure concerns and defects. No recognition claims: every visual finding here is
synthetic fixture truth."""
import copy
import json
import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from appearance_guard import check
from handoff import record
from raster import load, save as save_png
from scene_edit_compose import (CHECK_SET, COMPARE_WIDTH, OPERATION, REAL_FAILURE_CHECKS, compose, side_by_side, verify)
from scene_edit_plan import (JOB_ID, REFERENCE_LINE, REFERENCE_SHEET_LONG_EDGE, input_dims, location, meaningful_species,
                             prepare, reference_sheet, region_weight, edit_region, resized, validate, view_references)
from test_contract import save
import test_finishing

PARAMS = dict(margin_px=2, feather_px=2, reference_min_px=4)
RAW_SIZE = (24, 24)  # the tool returns a smaller image than the 32x32 fixture view: delivered upscaled 1.3333x
RECORDS = {'species': {
    'Agave victoriae-reginae': {'common_name': 'Queen Victoria agave',
                                'description': 'compact dense rosette of narrow stiff dark green leaves with white markings',
                                'bloom_exempt': 'monocarpic agave; flowering ends the rosette',
                                'reference_photos': [
                                    {'path': 'refs/agave-species.png', 'source_page': 'https://example.org/garden/agave',
                                     'image_url': 'https://example.org/garden/agave.jpg', 'accessed': '2026-10-31',
                                     'identification': 'Synthetic botanical garden, Agave victoriae-reginae',
                                     'cultivar_match': 'species', 'usage_note': 'Synthetic fixture photo.'}]},
    'Synthetic tiny cultivar': {'common_name': 'tiny daisy', 'description': 'low mound of small leaves',
                                'flowers': 'many small white daisies with yellow centres'}}}
ALL_PASS = dict.fromkeys(REAL_FAILURE_CHECKS, 'pass')
MODEL = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen'}


def write_reference_photos(base):
    """Synthetic 'reference photos' next to the records (the scene work folder)."""
    (base / 'refs').mkdir(exist_ok=True)
    Image.fromarray(blocks_image((60, 40), 4, cell=5)).save(base / 'refs' / 'agave-species.png')
    Image.fromarray(blocks_image((40, 60), 5, cell=5)).save(base / 'refs' / 'agave-exact.png')


def text_of(path):
    return Path(path).read_text(encoding='utf-8')


def gray_of(path):
    with Image.open(path) as im:
        return np.array(im)


def solid(path, size, color):
    Image.fromarray(np.full((size[1], size[0], 3), color, np.uint8)).save(path)


def blocks_image(size, seed, cell=4):
    """Deterministic structured pattern: random-colour cells."""
    rng = np.random.default_rng(seed)
    w, h = size
    small = rng.integers(20, 236, ((h + cell - 1) // cell, (w + cell - 1) // cell, 3)).astype(np.uint8)
    return np.repeat(np.repeat(small, cell, 0), cell, 1)[:h, :w]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class PlanFixture(unittest.TestCase):
    def setUp(self):
        self.fin = test_finishing.PhotoFinishing(); self.fin.setUp(); self.addCleanup(self.fin.doCleanups)
        self.p = self.fin.p
        save(self.p / 'finish-review.json', self.fin.review)
        write_reference_photos(self.p)
        save(self.p / 'records.json', RECORDS)
        self.plan = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene', **PARAMS)
        self.job = self.plan['jobs'][0]

    def records_with(self, change, name):
        records = copy.deepcopy(RECORDS); change(records['species']['Agave victoriae-reginae'])
        save(self.p / name, records); return self.p / name

    def plan_with(self, change, name, **kw):
        return prepare(self.p / 'finish-review.json', self.records_with(change, name + '.json'), self.p / name, **dict(PARAMS, **kw))


class SceneEditPlan(PlanFixture):
    def test_one_full_view_job(self):
        plan = self.plan
        self.assertEqual(plan['schema_version'], 5)
        self.assertEqual(len(plan['jobs']), 1)
        job = self.job
        self.assertEqual((job['id'], job['kind'], job['box']), (JOB_ID, 'full_view', [0, 0, 32, 32]))
        self.assertEqual(job['input_size'], [32, 32])  # never upscaled
        self.assertNotIn('structure', plan); self.assertNotIn('ramps_px', plan)
        composite, _ = load(plan['composite']['path'])
        got, _ = load(job['input']['path'])
        self.assertTrue(np.array_equal(got, np.rint(composite / 257).astype(np.uint8)))
        region = gray_of(plan['region']['path']) > 0
        self.assertTrue(np.any(region) and np.any(~region))
        weight = gray_of(plan['region_weight_preview']['path'])
        self.assertTrue(np.all(weight[~region] == 0))

    def test_input_size_and_locations(self):
        self.assertEqual(input_dims(3840, 2560, [1536, 1024]), [1536, 1024])
        self.assertEqual(input_dims(2560, 3840, [1536, 1024]), [1024, 1536])
        self.assertEqual(input_dims(4000, 2250, [1536, 1024]), [1536, 864])
        self.assertEqual(input_dims(800, 600, [1536, 1024]), [800, 600])
        self.assertEqual(location([10, 90]), 'left foreground')
        self.assertEqual(location([50, 50]), 'centre middle ground')
        self.assertEqual(location([100, 0]), 'right background')
        alpha = np.zeros((40, 40), np.uint8); alpha[10:30, 10:30] = 255
        grown = edit_region(alpha, np.zeros_like(alpha), 4)
        self.assertTrue(grown[10, 6] and not grown[10, 5])
        weight = region_weight(grown, 4)
        self.assertEqual(weight[0, 0], 0); self.assertEqual(weight[20, 20], 1); self.assertTrue(0 < weight[20, 6] < 1)

    def test_prompt_is_one_whole_view_pass_with_species_locations(self):
        text = text_of(self.job['prompt']['path'])
        self.assertIn('Edit target: the first supplied image, the whole view of an existing designed garden', text)
        self.assertIn('in one pass over the entire view', text)
        self.assertIn('never growing out of concrete, paving, steps or walls', text)
        self.assertIn('never remove, replace or merge a plant', text)
        self.assertIn('natural leaf movement, fuller foliage and a natural outline are fine', text)
        self.assertIn('Keep the house, sky, street and neighbourhood unchanged', text)
        self.assertIn('shredded bark mulch', text)
        self.assertIn(REFERENCE_LINE, text)
        self.assertLess(text.index(REFERENCE_LINE), text.index('The planting, by species'))  # sheet before the list
        self.assertTrue(text.splitlines()[0].endswith('foreground):'))
        self.assertNotIn('tile', text)
        tiny = next(r for r in self.job['inventory'] if r['id'] == 'tiny')
        self.assertEqual(tiny['size_px'], [1, 1]); self.assertEqual(tiny['location'], location(tiny['position_percent']))
        self.assertNotIn('hidden', [r['id'] for r in self.job['inventory']])  # no visible pixels
        self.assertIn(f"flowers: many small white daisies with yellow centres. 1 plant: {tiny['location']} 1", text)
        self.assertIn('Agave victoriae-reginae (Queen Victoria agave; compact dense rosette', text)
        lines = text.strip().split('\n')[1:]
        self.assertTrue(lines[0].startswith('Agave victoriae-reginae'))  # largest species first

    def test_plan_is_deterministic_and_frozen(self):
        again = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-2', **PARAMS)
        strip = lambda job: {k: v for k, v in job.items() if k not in ('input', 'prompt', 'references')}
        self.assertEqual(strip(again['jobs'][0]), strip(self.job))
        validate(self.plan, self.p)
        tampered = copy.deepcopy(self.plan); tampered['jobs'][0]['inventory'][0]['size_px'] = [99, 99]
        with self.assertRaisesRegex(ValueError, 'differs'): validate(tampered, self.p)
        tampered = copy.deepcopy(self.plan); tampered['schema_version'] = 4
        with self.assertRaisesRegex(ValueError, 'Only schema 5'): validate(tampered, self.p)
        with self.assertRaisesRegex(ValueError, 'input size'):
            prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-3', **dict(PARAMS, input_size=[100, 100]))
        path = Path(self.job['prompt']['path']); path.write_text(path.read_text(encoding='utf-8') + 'extra', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Missing or changed'): validate(self.plan, self.p)

    def test_input_is_bound_to_the_composite(self):
        solid(self.job['input']['path'], (32, 32), [1, 2, 3])
        tampered = copy.deepcopy(self.plan); tampered['jobs'][0]['input'] = record(self.job['input']['path'])
        with self.assertRaisesRegex(ValueError, 'Planned input differs'): validate(tampered, self.p)

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


class SceneEditReferences(PlanFixture):
    """Reference photos: records, the view's threshold selection, the labelled sheet, prompt and attempt binding."""

    def test_plan_records_the_view_references(self):
        refs = self.job['references']
        self.assertEqual(refs['min_px'], 4)
        self.assertEqual([e['species'] for e in refs['species']], ['Agave victoriae-reginae'])
        entry = refs['species'][0]
        self.assertEqual(entry['photo'], record(self.p / 'refs' / 'agave-species.png'))
        self.assertEqual(entry['cultivar_match'], 'species')
        self.assertEqual(entry['identification'], 'Synthetic botanical garden, Agave victoriae-reginae')
        self.assertEqual([(o['species'], o['reason']) for o in refs['omitted']], [('Synthetic tiny cultivar', 'below_threshold')])
        sheet = refs['sheet']
        self.assertEqual(Path(sheet['path']), self.p / 'scene' / 'jobs' / JOB_ID / 'references.jpg')
        self.assertEqual(record(sheet['path']), sheet)
        validate(self.plan, self.p)

    def test_threshold_selection_and_sheet_cap(self):
        rows = [{'species': 'A', 'visible_pixels': 100, 'size_px': [60, 40]},  # both sides must reach the threshold
                {'species': 'A', 'visible_pixels': 3000, 'size_px': [50, 48]},
                {'species': 'B', 'visible_pixels': 9000, 'size_px': [400, 30]},  # long thin edge sliver
                {'species': 'C', 'visible_pixels': 5000, 'size_px': [90, 90]}]
        big, small, area = meaningful_species(rows, 48)
        self.assertEqual(big, ['C', 'A']); self.assertEqual(small, ['B']); self.assertEqual(area['A'], 3100)
        records = {'species': {}}
        rows = []
        for k in range(20):
            name = f'Species {k:02d}'
            rows.append({'species': name, 'visible_pixels': 1000 * (k + 1), 'size_px': [64, 64]})
            records['species'][name] = {'common_name': f'plant {k}', 'reference_photos': [
                {'path': 'refs/agave-species.png', 'cultivar_match': 'species', 'identification': 'x', 'source_page': 'https://e.org/p',
                 'image_url': 'https://e.org/i.jpg', 'accessed': '2026-10-31'},
                {'path': 'refs/agave-exact.png', 'cultivar_match': 'exact', 'identification': 'y', 'source_page': 'https://e.org/q',
                 'image_url': 'https://e.org/j.jpg', 'accessed': '2026-10-31'}]}
        records['species']['Species 05'] = {'common_name': 'plant 5', 'reference_missing': 'no identified photo found'}
        refs = view_references(rows, records, self.p, 48)
        shown = [e['species'] for e in refs['species']]
        self.assertEqual(len(shown), 16); self.assertEqual(shown[0], 'Species 19'); self.assertNotIn('Species 05', shown)
        self.assertEqual([(o['species'], o['reason']) for o in refs['omitted']],
                         [('Species 05', 'reference_missing'), ('Species 02', 'sheet_cap'), ('Species 01', 'sheet_cap'),
                          ('Species 00', 'sheet_cap')])
        self.assertTrue(all(e['cultivar_match'] == 'exact' and e['photo'] == record(self.p / 'refs' / 'agave-exact.png')
                            for e in refs['species']))  # the exact-cultivar photo is preferred
        sheet = reference_sheet(refs['species'])
        self.assertLessEqual(max(sheet.shape[:2]), REFERENCE_SHEET_LONG_EDGE)
        self.assertEqual(sheet.shape[1], 4 * 512)
        one = reference_sheet(refs['species'][:1])
        self.assertEqual(one.shape[1], 768); self.assertLess(int(one.min()), 30)  # caption text drawn
        # A higher threshold: the agave is small, so it needs no reference and the view gets no sheet.
        plan = self.plan_with(lambda r: r.pop('reference_photos'), 'high-threshold', reference_min_px=30)
        job = plan['jobs'][0]
        self.assertTrue(not job['references']['species'] and 'sheet' not in job['references'])
        self.assertNotIn(REFERENCE_LINE, text_of(job['prompt']['path']))
        self.assertIn('Edit target: the supplied image', text_of(job['prompt']['path']))
        validate(plan, self.p)

    def test_reference_records_are_validated(self):
        with self.assertRaisesRegex(ValueError, 'Give reference_photos'):
            self.plan_with(lambda r: r.pop('reference_photos'), 'no-ref')
        cases = {'cultivar': (lambda q: q.update(cultivar_match='close'), 'exact or species'),
                 'date': (lambda q: q.update(accessed='last week'), 'access date'),
                 'url': (lambda q: q.update(image_url='agave.jpg'), 'image URL'),
                 'ident': (lambda q: q.pop('identification'), 'needs path'),
                 'absolute': (lambda q: q.update(path=str(self.p / 'refs' / 'agave-species.png')), 'relative to RECORDS'),
                 'outside': (lambda q: q.update(path='../outside.png'), 'under the scene work folder'),
                 'notimage': (lambda q: q.update(path='records.json'), 'not a readable image')}
        for name, (change, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                self.plan_with(lambda r: change(r['reference_photos'][0]), 'bad-' + name)
        with self.assertRaisesRegex(ValueError, 'not both'):
            self.plan_with(lambda r: r.update(reference_missing='none found'), 'both')
        with self.assertRaisesRegex(ValueError, 'needs a reason'):
            self.plan_with(lambda r: (r.pop('reference_photos'), r.update(reference_missing=' ')), 'blank-missing')
        old = os.environ.get('PACIFICA_ASSETS_DIR'); os.environ['PACIFICA_ASSETS_DIR'] = str(self.p / 'refs')
        try:
            with self.assertRaisesRegex(ValueError, 'never keep them in the shared plant library'):
                self.plan_with(lambda r: None, 'library')
        finally:
            if old is None: os.environ.pop('PACIFICA_ASSETS_DIR')
            else: os.environ['PACIFICA_ASSETS_DIR'] = old

    def test_reference_missing_is_allowed(self):
        plan = self.plan_with(lambda r: (r.pop('reference_photos'), r.update(reference_missing='no identified photo found')), 'missing')
        job = plan['jobs'][0]
        self.assertEqual(job['references']['species'], []); self.assertNotIn('sheet', job['references'])
        self.assertIn('reference_missing', [o['reason'] for o in job['references']['omitted']])
        self.assertNotIn(REFERENCE_LINE, text_of(job['prompt']['path']))
        validate(plan, self.p)
        raw = self.p / 'missing-raw'; raw.mkdir()
        solid(raw / 'full.png', RAW_SIZE, [90, 90, 90])
        (raw / 'full.txt').write_text(text_of(job['prompt']['path']), encoding='utf-8')
        manifest = {**MODEL, 'jobs': {JOB_ID: {'attempts': [{'raw': 'full.png', 'prompt': 'full.txt'}], 'selected': 1}}}
        save(raw / 'attempts.json', manifest)
        compose(self.p / 'missing/plan.json', raw / 'attempts.json', self.p / 'missing-out')
        manifest['jobs'][JOB_ID]['attempts'][0]['reference'] = '../scene/jobs/full/references.jpg'
        save(raw / 'attempts-2.json', manifest)
        with self.assertRaisesRegex(ValueError, 'no planned reference sheet'):
            compose(self.p / 'missing/plan.json', raw / 'attempts-2.json', self.p / 'missing-out-2')

    def test_sheet_is_bound_to_its_photos(self):
        size = (768, 576 + int(32 * 3.4) + 12)
        with Image.open(self.job['references']['sheet']['path']) as im:
            self.assertEqual(im.format, 'JPEG'); self.assertEqual(im.size, size)
        sheet = Path(self.job['references']['sheet']['path'])
        Image.fromarray(blocks_image(size, 7, cell=16)).save(sheet, format='JPEG')
        tampered = copy.deepcopy(self.plan); tampered['jobs'][0]['references']['sheet'] = record(sheet)
        with self.assertRaisesRegex(ValueError, 'Reference sheet differs'): validate(tampered, self.p)
        tampered = copy.deepcopy(self.plan); tampered['jobs'][0]['references']['species'][0]['cultivar_match'] = 'exact'
        with self.assertRaisesRegex(ValueError, 'differs: references'): validate(tampered, self.p)
        Image.fromarray(blocks_image((60, 40), 8, cell=5)).save(self.p / 'refs' / 'agave-species.png')
        with self.assertRaisesRegex(ValueError, 'Missing or changed|differs'): validate(self.plan, self.p)


class SceneEditCompose(PlanFixture):
    """The selected full-view edit replaces the composite inside the feathered region; the pass always delivers."""
    OFFSET = 30

    def setUp(self):
        super().setUp()
        self.raw = self.p / 'raw'; self.raw.mkdir()
        self.composite8 = np.rint(load(self.plan['composite']['path'])[0] / 257).astype(np.uint8)
        Image.fromarray(self.edit_raw()).save(self.raw / 'full-1.png')
        (self.raw / 'full-1.txt').write_text(text_of(self.job['prompt']['path']), encoding='utf-8')
        self.manifest = {**MODEL, 'jobs': {JOB_ID: {'attempts': [
            {'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Synthetic.',
             'reference': '../scene/jobs/full/references.jpg', 'input': '../scene/jobs/full/input.png'}], 'selected': 1}}}
        save(self.raw / 'attempts.json', self.manifest)
        self.proof = compose(self.p / 'scene/plan.json', self.raw / 'attempts.json', self.p / 'scene-out')
        self.review = self.make_review(self.proof, 'scene-out')

    def edit_raw(self):
        """A photographic edit of the whole view: same structure, brighter, returned at the tool's (smaller) size."""
        bright = np.clip(self.composite8.astype(np.int16) + self.OFFSET, 0, 255).astype(np.uint8)
        return np.array(Image.fromarray(bright).resize(RAW_SIZE, Image.LANCZOS))

    def make_review(self, proof, out):
        return {'schema_version': 5, 'method': 'scene_photographic_edit', 'handoff': record(self.p / 'handoff.json'),
                'camera_id': 'front', 'baseline_review': record(self.p / 'finish-review.json'),
                'candidate': proof['candidate'], 'scene_edit_proof': record(self.p / out / 'scene-edit-proof.json'),
                'model': dict(MODEL),
                'scene_review': {'reviewer': 'vision-capable-agent', 'source_and_final_loaded': True, 'native_size_crops_loaded': True,
                                 'observations': 'Synthetic fixture truth, not actual photo recognition.',
                                 'paired_evidence': [proof['candidate'], proof['compare']], 'check_set': CHECK_SET,
                                 'checks': dict(ALL_PASS), 'defects': [], 'defect_count': 0,
                                 'attempts': len(proof['attempts']['jobs'][0]['attempts']),
                                 'selected_attempt': proof['attempts']['jobs'][0]['selected']}}

    def rewrite(self, manifest, name):
        save(self.raw / name, manifest); return self.raw / name

    def compose_with(self, manifest, name):
        return compose(self.p / 'scene/plan.json', self.rewrite(manifest, name + '.json'), self.p / name)

    def test_exact_outside_region_and_edit_inside(self):
        self.assertEqual(self.proof['operation'], OPERATION)
        self.assertEqual(self.proof['output'], {'raw_size': [24, 24], 'delivered_size': [32, 32], 'upscale': 1.3333})
        final, _ = load(self.proof['candidate']['path'])
        previous, _ = load(self.plan['composite']['path'])
        cleaned, _ = load(self.plan['cleaned_photo']['path'])
        region = gray_of(self.plan['region']['path']) > 0
        self.assertTrue(np.array_equal(final[~region], previous[~region]))
        self.assertTrue(np.array_equal(final[~region], cleaned[~region]))
        self.assertFalse(np.array_equal(final[region], previous[region]))
        weight = gray_of(self.proof['region_weight']['path'])
        self.assertTrue(np.all(weight[~region] == 0))
        full = weight == 255
        self.assertGreater(int(full.sum()), 4)
        edit = resized(self.raw / 'full-1.png', self.job['box'])
        diff = np.abs(final[full].astype(np.int32) - edit[full].astype(np.int32) * 257)
        self.assertLessEqual(int(diff.max()), 257)  # the whole region is the one edit, no tiles
        verify(self.proof, self.p)

    def test_compare_is_the_full_view_side_by_side(self):
        with Image.open(self.proof['compare']['path']) as im:
            self.assertEqual(im.format, 'JPEG'); self.assertEqual(im.width, COMPARE_WIDTH)
            self.assertGreater(im.height, COMPARE_WIDTH // 2)  # a square view: each half is 1200x1200 plus labels
        a = np.zeros((20, 30, 3), np.uint8); b = np.full((20, 30, 3), 255, np.uint8)
        pair = np.array(side_by_side(a, b, ('left', 'right'), width=600))
        self.assertEqual(pair.shape[1], 600)
        self.assertLess(int(pair[-1, 100].max()), 10); self.assertGreater(int(pair[-1, 500].min()), 245)
        with self.assertRaisesRegex(ValueError, 'same size'): side_by_side(a, b[:10])

    def test_review_accepts_and_always_delivers(self):
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(result['method'], 'scene_photographic_edit')
        self.assertEqual(result['delivered'], 'scene_edit_composite'); self.assertTrue(result['photo_background_composited'])
        self.assertEqual(result['model']['agent_model'], 'gpt-6-astra'); self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['check_set'], CHECK_SET); self.assertEqual(result['defect_count'], 0)
        self.assertEqual(result['real_failure_observations'], dict(ALL_PASS))
        self.assertEqual(result['output']['upscale'], 1.3333); self.assertEqual(result['compare'], self.proof['compare'])
        self.assertEqual(check(self.fin.review, self.p)['method'], 'photo_finish')  # previous route unchanged

    def test_real_failure_concerns_and_defects_never_block(self):
        review = copy.deepcopy(self.review)
        review['scene_review']['checks']['plants_rooted_in_beds'] = {
            'result': 'concern', 'where': 'front step, left', 'observation': 'Synthetic flower growing from the tread.'}
        review['scene_review']['defects'] = [
            {'type': 'extent_change', 'where': 'left agave', 'observation': 'crown about 10% wider'},
            {'type': 'invented_plant', 'where': 'bed edge, centre', 'observation': 'synthetic'}]
        review['scene_review']['defect_count'] = 2
        result = check(review, self.p)
        self.assertEqual(result['status'], 'completed_with_limitations')
        self.assertEqual(result['delivered'], 'scene_edit_composite'); self.assertEqual(result['scene_edit_status'], 'delivered')
        self.assertEqual(result['real_failure_observations']['plants_rooted_in_beds'], 'concern')
        self.assertEqual(result['defect_count'], 2)
        self.assertEqual(result['limitations'][0], {'check': 'plants_rooted_in_beds', 'blocking': False,
                                                    'observations': 'front step, left: Synthetic flower growing from the tread.'})
        self.assertEqual(result['limitations'][1], {'check': 'defect: extent_change', 'blocking': False,
                                                    'observations': 'left agave: crown about 10% wider'})
        self.assertTrue(all(r['blocking'] is False for r in result['non_blocking_observations']))

    def test_review_schema_is_required(self):
        for change, message in ((lambda v: v.pop('defects'), 'defects list'),
                                (lambda v: v.update(defect_count=3), 'defect_count'),
                                (lambda v: v.update(defects=[{'type': 'wrong_species', 'where': 'x', 'observation': 'y'}], defect_count=1), 'type'),
                                (lambda v: v.update(defects=[{'type': 'movement', 'observation': 'y'}], defect_count=1), 'where'),
                                (lambda v: v.update(check_set='real_failures_v1'), 'check_set'),
                                (lambda v: v['checks'].pop('hardscape_unchanged'), 'exactly the real-failure'),
                                (lambda v: v['checks'].update(hardscape_unchanged='fail'), 'is "pass", or'),
                                (lambda v: v.update(selected_attempt=2), 'recorded attempts'),
                                (lambda v: v.update(native_size_crops_loaded=False), 'native-size crops')):
            review = copy.deepcopy(self.review); change(review['scene_review'])
            with self.assertRaisesRegex(ValueError, message): check(review, self.p)
        review = copy.deepcopy(self.review); review['release_policy'] = 'fallback_previous_composite'
        with self.assertRaisesRegex(ValueError, 'no fallback'): check(review, self.p)
        review = copy.deepcopy(self.review); review['schema_version'] = 4
        with self.assertRaisesRegex(ValueError, 'Only schema 5'): check(review, self.p)
        review = copy.deepcopy(self.review); review['candidate'] = self.fin.review['candidate']
        with self.assertRaisesRegex(ValueError, 'candidate differs'): check(review, self.p)

    def test_model_and_tool_are_required(self):
        for field, value in [('agent_model', 'gpt-5'), ('reasoning_effort', 'medium'), ('image_tool', 'api')]:
            manifest = copy.deepcopy(self.manifest); manifest[field] = value
            with self.assertRaisesRegex(ValueError, 'gpt-6-astra|built-in image tool'): self.compose_with(manifest, f'bad-{field}')
        bad = copy.deepcopy(self.review); bad['model']['reasoning_effort'] = 'low'
        with self.assertRaisesRegex(ValueError, 'gpt-6-astra/high'): check(bad, self.p)

    def test_attempt_rules(self):
        planned = text_of(self.job['prompt']['path'])
        (self.raw / 'changed.txt').write_text(planned + ' improvised', encoding='utf-8')
        (self.raw / 'second.txt').write_text(planned + '\nSECOND ATTEMPT: edit the ORIGINAL input; keep every agave rosette.', encoding='utf-8')
        solid(self.raw / 'fake-input.png', (32, 32), [1, 2, 3])
        ok = self.manifest['jobs'][JOB_ID]['attempts'][0]
        bare = {k: v for k, v in ok.items() if k != 'inspection'}
        cases = [
            ([dict(ok, prompt='changed.txt')], 1, 'exact planned prompt'),
            ([bare, dict(bare, prompt='second.txt')], 2, 'recorded inspection'),
            ([ok, bare], 2, 'adds a correction'),
            ([ok] * 3, 1, 'Record 1-2 attempts'),
            ([], None, 'Record 1-2 attempts'),
            ([ok], None, 'always delivers one'),
            ([{k: v for k, v in ok.items() if k != 'reference'}], 1, 'must attach the planned reference sheet'),
            ([dict(ok, reference='full-1.png')], 1, 'must attach the planned reference sheet'),
            ([dict(ok, input='fake-input.png')], 1, 'not the planned full-view input')]
        for k, (attempts, selected, message) in enumerate(cases):
            manifest = copy.deepcopy(self.manifest); manifest['jobs'][JOB_ID] = {'attempts': attempts, 'selected': selected}
            with self.subTest(message), self.assertRaisesRegex(ValueError, message): self.compose_with(manifest, f'bad-{k}')
        manifest = copy.deepcopy(self.manifest); manifest['jobs']['tile-01'] = manifest['jobs'][JOB_ID]
        with self.assertRaisesRegex(ValueError, 'one planned full-view job'): self.compose_with(manifest, 'extra-job')
        manifest = copy.deepcopy(self.manifest)
        manifest['jobs'][JOB_ID] = {'attempts': [dict(ok, inspection='Agave turned into pads.'), dict(bare, prompt='second.txt')],
                                    'selected': 2}
        proof = self.compose_with(manifest, 'two')
        self.assertEqual(len(proof['attempts']['jobs'][0]['attempts']), 2)
        review = self.make_review(proof, 'two')
        self.assertEqual(check(review, self.p)['selected'], 2)

    def test_aspect_mismatch_rejected(self):
        solid(self.raw / 'wide.png', (48, 24), [0, 0, 0])
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][JOB_ID]['attempts'][0]['raw'] = 'wide.png'
        with self.assertRaisesRegex(ValueError, 'aspect ratio'): self.compose_with(manifest, 'wide')

    def test_changed_candidate_output_or_raw_rejected(self):
        proof = copy.deepcopy(self.proof); proof['output']['upscale'] = 1.0
        with self.assertRaisesRegex(ValueError, 'Recompute the output'): verify(proof, self.p)
        proof = copy.deepcopy(self.proof); proof['operation'] = 'scene_edit_composite_tiles_seam_v3'
        with self.assertRaisesRegex(ValueError, 'tiled operations were removed'): verify(proof, self.p)
        final, _ = load(self.proof['candidate']['path']); final[16, 16] = 0
        save_png(self.proof['candidate']['path'], final)
        proof = copy.deepcopy(self.proof); proof['candidate'] = record(self.proof['candidate']['path'])
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'): verify(proof, self.p)
        solid(self.raw / 'full-1.png', RAW_SIZE, [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Missing or changed'): verify(self.proof, self.p)


if __name__ == '__main__':
    unittest.main()
