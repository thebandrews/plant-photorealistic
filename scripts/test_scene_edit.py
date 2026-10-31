"""Synthetic scene photographic edit tests: layout, prompts, blending and review gates; no recognition claims."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from appearance_guard import check
from handoff import read, record
from raster import load, save as save_png
from scene_edit_compose import SCENE_CHECKS, compose, verify
from scene_edit_plan import prepare, reference, tiles, validate, edit_region, region_weight
from test_contract import save
import test_finishing

PARAMS = dict(tile=[16, 12], overlap=.3, margin_px=2, feather_px=2)
RECORDS = {'species': {
    'Agave victoriae-reginae': {'common_name': 'Queen Victoria agave',
                                'description': 'compact dense rosette of narrow stiff dark green leaves with white markings',
                                'bloom_exempt': 'monocarpic agave; flowering ends the rosette'},
    'Synthetic tiny cultivar': {'common_name': 'tiny daisy', 'description': 'low mound of small leaves',
                                'flowers': 'many small white daisies with yellow centres'}}}


def text_of(path):
    return Path(path).read_text(encoding='utf-8')


def gray_of(path):
    with Image.open(path) as im:
        return np.array(im)


def solid(path, size, color):
    Image.fromarray(np.full((size[1], size[0], 3), color, np.uint8)).save(path)


class PlanFixture(unittest.TestCase):
    def setUp(self):
        self.fin = test_finishing.PhotoFinishing(); self.fin.setUp(); self.addCleanup(self.fin.doCleanups)
        self.p = self.fin.p
        save(self.p / 'finish-review.json', self.fin.review)
        save(self.p / 'records.json', RECORDS)
        self.plan = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene', **PARAMS)


class SceneEditPlan(PlanFixture):
    def test_layout_covers_region_with_thirty_percent_overlap(self):
        plan = self.plan
        self.assertEqual(plan['jobs'][0]['id'], 'full'); self.assertEqual(plan['jobs'][0]['box'], [0, 0, 32, 32])
        region = gray_of(plan['region']['path']) > 0
        covered = np.zeros_like(region)
        boxes = [j['box'] for j in plan['jobs'] if j['kind'] == 'tile']
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
        self.assertIn("Agave victoriae-reginae (Queen Victoria agave; compact dense rosette", text)
        self.assertIn('flowers: none required; preserve foliage. Instances: agave-1 at [50, 50]%, size [20, 20]px', text)
        self.assertIn('flowers: many small white daisies with yellow centres. Instances: tiny at [11, 11]%, size [1, 1]px', text)
        tile = next(j for j in self.plan['jobs'] if j['kind'] == 'tile')
        self.assertIn('matching crop of the already-edited whole view', text_of(tile['prompt']['path']))

    def test_plan_is_deterministic_and_frozen(self):
        again = prepare(self.p / 'finish-review.json', self.p / 'records.json', self.p / 'scene-2', **PARAMS)
        strip = lambda jobs: [{k: v for k, v in j.items() if k not in ('input', 'prompt')} for j in jobs]
        self.assertEqual(strip(again['jobs']), strip(self.plan['jobs']))
        validate(self.plan, self.p)
        tampered = copy.deepcopy(self.plan); tampered['jobs'][1]['inventory'][0]['size_px'] = [99, 99]
        with self.assertRaisesRegex(ValueError, 'differs'): validate(tampered, self.p)
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


class SceneEditCompose(PlanFixture):
    def setUp(self):
        super().setUp()
        self.raw = self.p / 'raw'; self.raw.mkdir()
        manifest = {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen', 'jobs': {}}
        solid(self.raw / 'full-1.png', (16, 16), [200, 40, 40])  # half-size return, same aspect
        (self.raw / 'full-1.txt').write_text(text_of(self.plan['jobs'][0]['prompt']['path']), encoding='utf-8')
        manifest['jobs']['full'] = {'attempts': [{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Synthetic.'}], 'selected': 1}
        for i, job in enumerate(j for j in self.plan['jobs'] if j['kind'] == 'tile'):
            solid(self.raw / f"{job['id']}-1.png", (19, 14), [40, 40 + 20 * i, 200])  # 1.357 vs 1.333 aspect: within 3%
            (self.raw / f"{job['id']}-1.txt").write_text(text_of(job['prompt']['path']), encoding='utf-8')
            reference(self.p / 'scene/plan.json', self.raw / 'full-1.png', job['id'], self.raw / f"{job['id']}-ref.png")
            manifest['jobs'][job['id']] = {'attempts': [{'raw': f"{job['id']}-1.png", 'prompt': f"{job['id']}-1.txt",
                                           'reference': f"{job['id']}-ref.png", 'reference_from': 1, 'inspection': 'Synthetic.'}], 'selected': 1}
        self.manifest = manifest; save(self.raw / 'attempts.json', manifest)
        self.proof = compose(self.p / 'scene/plan.json', self.raw / 'attempts.json', self.p / 'scene-out')
        self.review = {'schema_version': 4, 'method': 'scene_photographic_edit', 'handoff': record(self.p / 'handoff.json'),
                       'camera_id': 'front', 'baseline_review': record(self.p / 'finish-review.json'),
                       'candidate': self.proof['candidate'], 'scene_edit_proof': record(self.p / 'scene-out/scene-edit-proof.json'),
                       'model': {'agent_model': 'gpt-6-astra', 'reasoning_effort': 'high', 'image_tool': 'image_gen.imagegen'},
                       'scene_review': {'reviewer': 'vision-capable-agent', 'source_and_final_loaded': True, 'native_size_crops_loaded': True,
                           'observations': 'Synthetic fixture truth, not actual photo recognition.',
                           'paired_evidence': [self.proof['candidate']], 'checks': dict.fromkeys(SCENE_CHECKS, 'pass'),
                           'jobs': [{'id': j['id'], 'attempts': 1, 'selected_attempt': 1, 'species_fidelity': 'pass',
                                     'invented_or_hybrid_plants': 'none', 'observations': 'Synthetic.',
                                     'evidence': self.proof['job_proofs'][j['id']]} for j in self.plan['jobs']]}}

    def rewrite(self, manifest, name):
        save(self.raw / name, manifest); return self.raw / name

    def test_blend_is_exact_outside_region_and_tiled_inside(self):
        final, _ = load(self.proof['candidate']['path'])
        previous, _ = load(self.plan['composite']['path'])
        cleaned, _ = load(self.plan['cleaned_photo']['path'])
        region = gray_of(self.plan['region']['path']) > 0
        self.assertTrue(np.any(~region))
        self.assertTrue(np.array_equal(final[~region], previous[~region]))
        self.assertTrue(np.array_equal(final[~region], cleaned[~region]))
        self.assertFalse(np.array_equal(final[region], previous[region]))
        self.assertEqual(int(final[16, 16, 0]) // 257, 40)  # tiles (blue), not the red full view, own the interior
        weight = gray_of(self.proof['region_weight']['path'])
        self.assertTrue(np.all(weight[~region] == 0)); self.assertEqual(weight[16, 16], 255)
        verify(self.proof, self.p)

    def test_full_view_fills_where_no_tile_is_selected(self):
        manifest = copy.deepcopy(self.manifest)
        for job_id in manifest['jobs']:
            if job_id != 'full': manifest['jobs'][job_id]['selected'] = None
        proof = compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'full-only.json'), self.p / 'full-only')
        final, _ = load(proof['candidate']['path'])
        self.assertEqual(int(final[16, 16, 0]) // 257, 200)

    def test_review_accepts_and_guard_routes_scene_edit(self):
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'accepted'); self.assertEqual(result['method'], 'scene_photographic_edit')
        self.assertEqual(result['delivered'], 'scene_edit_composite'); self.assertTrue(result['photo_background_composited'])
        self.assertEqual(result['model']['agent_model'], 'gpt-6-astra'); self.assertEqual(result['attempts']['full'], 1)
        self.assertEqual(check(self.fin.review, self.p)['method'], 'photo_finish')  # previous route unchanged

    def test_failed_findings_fall_back_to_previous_composite(self):
        bad = copy.deepcopy(self.review); bad['scene_review']['checks']['plants_rooted_in_beds'] = 'fail'
        with self.assertRaisesRegex(ValueError, 'fall back'): check(bad, self.p)
        bad['release_policy'] = 'fallback_previous_composite'
        bad['limitations'] = [{'check': 'plants_rooted_in_beds', 'observations': 'Synthetic plant drawn on the step.'}]
        with self.assertRaisesRegex(ValueError, 'previous photo_finish composite'): check(bad, self.p)
        bad['candidate'] = self.fin.review['candidate']
        result = check(bad, self.p)
        self.assertEqual(result['status'], 'completed_with_limitations')
        self.assertEqual(result['delivered'], 'previous_photo_finish_composite')
        self.assertEqual(result['nonpassing_checks'], ['plants_rooted_in_beds'])
        bad['limitations'] = []
        with self.assertRaisesRegex(ValueError, 'Describe each'): check(bad, self.p)

    def test_tile_species_failure_blocks_acceptance(self):
        bad = copy.deepcopy(self.review); bad['scene_review']['jobs'][1]['invented_or_hybrid_plants'] = 'present'
        with self.assertRaisesRegex(ValueError, 'species fidelity'): check(bad, self.p)
        bad = copy.deepcopy(self.review); bad['scene_review']['jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'every scene edit job'): check(bad, self.p)

    def test_fallback_without_any_scene_edit_is_disclosed(self):
        review = {k: v for k, v in self.review.items() if k not in ('scene_edit_proof', 'scene_review', 'model')}
        review.update(candidate=self.fin.review['candidate'], release_policy='fallback_previous_composite')
        with self.assertRaisesRegex(ValueError, 'disclosed fallback'): check(review, self.p)
        review['limitations'] = [{'check': 'scene_edit_not_run', 'observations': 'Synthetic: image tool unavailable.'}]
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
        cases = [
            ([{'raw': 'full-1.png', 'prompt': 'changed.txt', 'inspection': 'x'}], 'exact planned prompt'),
            ([{'raw': 'full-1.png', 'prompt': 'full-1.txt'}, {'raw': 'full-1.png', 'prompt': 'second.txt'}], 'recorded inspection'),
            ([{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'x'}, {'raw': 'full-1.png', 'prompt': 'full-1.txt'}], 'adds a correction'),
            ([{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'x'}] * 3, 'At most 2')]
        for attempts, message in cases:
            manifest = copy.deepcopy(self.manifest); manifest['jobs']['full']['attempts'] = attempts
            with self.assertRaisesRegex(ValueError, message):
                compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'attempts-bad.json'), self.p / 'bad')
        manifest = copy.deepcopy(self.manifest)
        manifest['jobs']['full'] = {'attempts': [{'raw': 'full-1.png', 'prompt': 'full-1.txt', 'inspection': 'Too red.'},
                                                 {'raw': 'full-1.png', 'prompt': 'second.txt'}], 'selected': 2}
        proof = compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'two.json'), self.p / 'two')
        self.assertEqual(len(proof['attempts']['jobs'][0]['attempts']), 2)

    def test_aspect_mismatch_and_wrong_reference_rejected(self):
        solid(self.raw / 'wide.png', (32, 16), [0, 0, 0])
        manifest = copy.deepcopy(self.manifest); manifest['jobs']['full']['attempts'][0]['raw'] = 'wide.png'
        with self.assertRaisesRegex(ValueError, 'aspect ratio'):
            compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'wide.json'), self.p / 'bad')
        tile = next(k for k in self.manifest['jobs'] if k != 'full')
        solid(self.raw / 'fake-ref.png', (16, 12), [1, 2, 3])
        manifest = copy.deepcopy(self.manifest); manifest['jobs'][tile]['attempts'][0]['reference'] = 'fake-ref.png'
        with self.assertRaisesRegex(ValueError, 'full-view crop'):
            compose(self.p / 'scene/plan.json', self.rewrite(manifest, 'ref.json'), self.p / 'bad2')

    def test_changed_candidate_or_raw_rejected(self):
        final, _ = load(self.proof['candidate']['path']); final[0, 31] = 0
        save_png(self.proof['candidate']['path'], final)
        proof = copy.deepcopy(self.proof); proof['candidate'] = record(self.proof['candidate']['path'])
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'): verify(proof, self.p)
        solid(self.raw / 'full-1.png', (16, 16), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Missing or changed'): verify(self.proof, self.p)


if __name__ == '__main__':
    unittest.main()
