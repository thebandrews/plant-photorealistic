"""Synthetic behavioral tests; annotations below are test ground truth only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from appearance_guard import GEOMETRY, check as guard
from appearance_plan import prepare, prepare_all, validate
from compose_candidate import compose
from handoff import record, read
from make_edit_masks import make
from raster import load, save as save_png
from review_pairs import prepare as prepare_pairs
from test_contract import fixture, save


class EvidencePaths(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'test-output'
        root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        self.data, _ = fixture(self.p)
        ids = np.load(self.p / 'ids.npy')
        ids[3, 3] = 8  # Only one visible pixel; no diagnostic anatomy.
        np.save(self.p / 'ids.npy', ids)
        ledger = read(self.p / 'ledger.json')
        ledger += [{'id': 'tiny', 'species': 'Synthetic tiny cultivar'},
                   {'id': 'hidden', 'species': 'Synthetic hidden species'}]
        self.data['plant_ledger'] = save(self.p / 'ledger.json', ledger)
        projection = read(self.p / 'projected.json')
        projection['plants'] += [
            {'id': 'tiny', 'species': ledger[1]['species'], 'pass_index': 8,
             'root_pixel': [3, 3], 'visible_pixels': 1},
            {'id': 'hidden', 'species': ledger[2]['species'], 'pass_index': 9,
             'root_pixel': None, 'visible_pixels': 0}]
        self.data['views'][0]['projected_plants'] = save(self.p / 'projected.json', projection)
        self.refresh()

    def refresh(self):
        view = self.data['views'][0]
        ids = np.load(self.p / 'ids.npy')
        projection = read(self.p / 'projected.json')
        rgb, _ = load(self.p / 'native.png')
        image_record = record(self.p / 'native.png')
        projection['image_sha256'] = image_record['sha256']
        view['projected_plants'] = save(self.p / 'projected.json', projection)
        view['image'] = image_record
        view['object_ids'] = record(self.p / 'ids.npy')
        e, k = make(ids, np.load(self.p / 'depth.npy'), [7, 8],
                    root_pixels=[p['root_pixel'] for p in projection['plants']])
        Image.fromarray(e.astype('uint8') * 255).save(self.p / 'editable.png')
        Image.fromarray(k.astype('uint8') * 255).save(self.p / 'protected.png')
        view['editable_mask'] = record(self.p / 'editable.png')
        view['protected_mask'] = record(self.p / 'protected.png')
        proof = read(self.p / 'mask-proof.json')
        proof.update(object_ids=view['object_ids'], projected_plants=view['projected_plants'],
                     allowed_material_indices=[7, 8])
        view['mask_provenance']['evidence'] = save(self.p / 'mask-proof.json', proof)
        validation = read(self.p / 'native-validation.json')
        validation.update(plant_ledger_sha256=self.data['plant_ledger']['sha256'],
                          render_sha256_by_camera={'front': image_record['sha256']},
                          evidence=[self.data['plant_ledger']])
        self.data['native_validation'] = save(self.p / 'native-validation.json', validation)
        save(self.p / 'handoff.json', self.data)

    def review(self, method='image_edit', retain=()):
        plan = prepare(self.p / 'handoff.json', 'front', self.p / 'plan', method)
        rows = [{k: p[k] for k in ('id', 'expected_species', 'identity_source', 'evidence_path')}
                for p in plan['plants']]
        for row in rows:
            if row['id'] in retain:
                row['evidence_path'] = 'preserved'
            if row['evidence_path'] == 'visual_review':
                row.update(appearance_assessment='consistent', observations='Synthetic fixed rosette outline retained.',
                           count_assessment='unchanged', occlusion_assessment='consistent',
                           measurement_evidence=record(self.p / 'candidate.png'))
        review = {'schema_version': 2, 'handoff': record(self.p / 'handoff.json'),
                  'plan': record(self.p / 'plan/plan.json'), 'camera_id': 'front',
                  'baseline_sha256': self.data['views'][0]['image']['sha256'], 'method': method,
                  'recognition': {'reviewer': 'vision-capable-agent', 'baseline_and_candidate_loaded': True,
                                  'original_pixel_crops_loaded': True, 'observations': 'Synthetic known material change.',
                                  'paired_crop_evidence': [record(self.p / 'candidate.png')],
                                  'geometry': dict.fromkeys(GEOMETRY, 'pass'), 'plants': rows,
                                  'image_quality': 'improved', 'quality_observations': 'Synthetic interior detail improved.'}}
        if method == 'image_edit':
            proof = compose(self.p / 'plan/plan.json', self.p / 'candidate.png', self.p / 'composite', retain)
            review.update(candidate=proof['candidate'], composite_proof=record(self.p / 'composite/composite-proof.json'))
        else:
            instances = [{'id': p['id'], 'species': p['expected_species'], 'pass_index': p['pass_index'],
                          'asset_id': 'synthetic-asset', 'geometry_sha256': 'fixed', 'seed': 1} for p in plan['plants']]
            before = {'exporter': 'material_invariants_v1', 'native_scene': self.data['native_scene'],
                      'invariants': {'plants': instances, 'objects': ['fixed'], 'cameras': ['fixed'], 'settings': {'lighting': 'fixed'}}}
            (self.p / 'material-scene.blend').write_bytes(b'Synthetic material scene')
            after = copy.deepcopy(before)
            after['native_scene'] = record(self.p / 'material-scene.blend')
            proof = {'provenance': 'renderer_export', 'camera_id': 'front',
                     'native_scene_sha256': after['native_scene']['sha256'],
                     'image_sha256': record(self.p / 'candidate.png')['sha256'],
                     'object_ids': record(self.p / 'ids.npy')}
            review.update(candidate=record(self.p / 'candidate.png'),
                          native_before=save(self.p / 'before.json', before),
                          native_after=save(self.p / 'after.json', after),
                          render_proof=save(self.p / 'render-proof.json', proof))
        return review

    def test_one_pixel_and_occluded_plants_do_not_block_improvement(self):
        result = guard(self.review(), self.p)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['evidence_counts'], {'preserved': 1, 'occluded': 1, 'visual_review': 1, 'native_invariants': 0})

    def test_batch_planning_includes_all_cameras_and_canonical_ids(self):
        plans=prepare_all(self.p/'handoff.json',self.p/'all-plans')
        self.assertEqual([p['camera_id'] for p in plans],['front'])
        self.assertEqual({p['id'] for p in plans[0]['plants']},{'agave-1','tiny','hidden'})

    def test_raw_hardscape_hallucination_is_excluded_from_final(self):
        raw = np.array(Image.open(self.p / 'candidate.png'))
        raw[0, 0] = [255, 0, 255]
        raw[3, 3] = [255, 0, 255]
        Image.fromarray(raw).save(self.p / 'candidate.png')
        review = self.review()
        final = load(review['candidate']['path'])[0]
        baseline = load(self.p / 'native.png')[0]
        self.assertTrue(np.array_equal(final[0, 0], baseline[0, 0]))
        self.assertTrue(np.array_equal(final[3, 3], baseline[3, 3]))
        self.assertEqual(guard(review, self.p)['status'], 'accepted')

    def test_final_protected_pixel_tampering_rejected(self):
        review = self.review()
        rgb, alpha = load(review['candidate']['path'])
        rgb[0, 0] = 0
        save_png(review['candidate']['path'], rgb, alpha)
        review['candidate'] = record(review['candidate']['path'])
        proof = read(self.p / 'composite/composite-proof.json')
        proof['candidate'] = review['candidate']
        review['composite_proof'] = save(self.p / 'composite/composite-proof.json', proof)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            guard(review, self.p)

    def test_widened_plan_mask_rejected_even_with_new_hash(self):
        review = self.review()
        mask = np.full((32, 32), 255, np.uint8)
        Image.fromarray(mask).save(self.p / 'plan/editable.png')
        plan = read(self.p / 'plan/plan.json')
        plan['editable_mask'] = record(self.p / 'plan/editable.png')
        with self.assertRaisesRegex(ValueError, 'permissions differ'):
            validate(plan, self.p / 'plan')

    def test_unreviewable_region_cannot_be_marked_visual(self):
        self.review()
        plan = read(self.p / 'plan/plan.json')
        plan['plants'][1]['evidence_path'] = 'visual_review'
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            validate(plan, self.p / 'plan')

    def test_preserved_region_cannot_be_claimed_without_preservation(self):
        review = self.review()
        review['recognition']['plants'][0]['evidence_path'] = 'preserved'
        with self.assertRaisesRegex(ValueError, 'evidence path differs'):
            guard(review, self.p)

    def test_local_retention_keeps_other_material_improvements(self):
        # Add a second editable material surface, so the first can be retained.
        ids = np.load(self.p / 'ids.npy'); ids[12:22, 22:30] = 10
        np.save(self.p / 'ids.npy', ids)
        projection = read(self.p / 'projected.json')
        projection['plants'][0]['visible_pixels'] = int(np.count_nonzero(ids == 7))
        save(self.p / 'projected.json', projection)
        self.refresh()
        proof = read(self.p / 'mask-proof.json');proof['allowed_material_indices'].append(10)
        view = self.data['views'][0]
        e,k = make(ids,np.ones(ids.shape),[7,8,10],root_pixels=[[16,25],[3,3]])
        for name,mask in [('editable',e),('protected',k)]:
            Image.fromarray(mask.astype('uint8')*255).save(self.p/f'{name}.png');view[f'{name}_mask']=record(self.p/f'{name}.png')
        view['mask_provenance']['evidence']=save(self.p/'mask-proof.json',proof);save(self.p/'handoff.json',self.data)
        raw = np.array(Image.open(self.p / 'candidate.png'));raw[17,26]=[200,150,80]
        Image.fromarray(raw).save(self.p / 'candidate.png')
        result = guard(self.review(retain=['agave-1']), self.p)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['evidence_counts']['preserved'], 2)

    def test_uncertain_edited_appearance_requires_local_retention(self):
        review = self.review()
        review['recognition']['plants'][0]['appearance_assessment'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'retain this region'):
            guard(review, self.p)

    def test_all_canonical_rows_and_species_bindings_required(self):
        review = self.review()
        bad = copy.deepcopy(review);bad['recognition']['plants'].pop()
        with self.assertRaisesRegex(ValueError, 'every canonical'):
            guard(bad, self.p)
        bad = copy.deepcopy(review);bad['recognition']['plants'][1]['expected_species'] = 'Substitute'
        with self.assertRaisesRegex(ValueError, 'species/identity'):
            guard(bad, self.p)

    def test_native_identity_and_footprints_without_fragment_recognition(self):
        self.assertEqual(guard(self.review('native_material_only'), self.p)['evidence_counts']['native_invariants'], 2)

    def test_native_geometry_or_asset_change_rejected(self):
        review = self.review('native_material_only')
        after = read(self.p / 'after.json');after['invariants']['plants'][0]['asset_id'] = 'replacement'
        review['native_after'] = save(self.p / 'after.json', after)
        with self.assertRaisesRegex(ValueError, 'invariants changed'):
            guard(review, self.p)

    def test_native_visibility_change_rejected(self):
        review = self.review('native_material_only')
        ids = np.load(self.p / 'ids.npy');ids[3,3]=0;np.save(self.p/'candidate-ids.npy',ids)
        proof = read(self.p/'render-proof.json');proof['object_ids']=record(self.p/'candidate-ids.npy')
        review['render_proof']=save(self.p/'render-proof.json',proof)
        with self.assertRaisesRegex(ValueError, 'visibility/footprints'):
            guard(review,self.p)

    def test_native_output_opacity_change_rejected(self):
        rgb,_=load(self.p/'native.png')
        save_png(self.p/'native.png',rgb,np.full(rgb.shape[:2],255,np.uint8));self.refresh()
        rgb,_=load(self.p/'candidate.png')
        save_png(self.p/'candidate.png',rgb,np.zeros(rgb.shape[:2],np.uint8))
        with self.assertRaisesRegex(ValueError,'opacity changed'):
            guard(self.review('native_material_only'),self.p)

    def test_supplied_independent_drift_measurements_still_enforced(self):
        review=self.review();row=review['recognition']['plants'][0]
        row.update(root_visible=True,candidate_root=[19,25])
        with self.assertRaisesRegex(ValueError,'root moved'):
            guard(review,self.p)
        row.update(root_visible=False,candidate_anchor=[19,15],baseline_anchor=[15,15])
        with self.assertRaisesRegex(ValueError,'anchor moved'):
            guard(review,self.p)

    def test_explicit_fallback_has_a_valid_nonaccepted_status(self):
        review={'schema_version':2,'handoff':record(self.p/'handoff.json'),'camera_id':'front',
                'baseline_sha256':self.data['views'][0]['image']['sha256'],
                'candidate':self.data['views'][0]['image'],'method':'unchanged_baseline'}
        self.assertFalse(guard(review,self.p)['accepted_appearance'])

    def test_unchanged_composite_cannot_claim_improvement(self):
        Image.open(self.p/'native.png').save(self.p/'candidate.png')
        with self.assertRaisesRegex(ValueError,'fallback, not improved'):
            guard(self.review(),self.p)

    def test_pending_visual_review_and_wrong_camera_rejected(self):
        review=self.review()
        pending=prepare_pairs(self.p/'handoff.json','front',review['candidate']['path'],self.p/'pairs',self.p/'plan/plan.json')
        pending['composite_proof']=review['composite_proof']
        with self.assertRaisesRegex(ValueError,'Actual image-recognition'):
            guard(pending,self.p)
        self.assertEqual({p['id'] for p in pending['recognition']['plants']},{'agave-1','tiny','hidden'})
        self.assertNotIn('identity_confidence',pending['recognition']['plants'][1])
        with self.assertRaisesRegex(ValueError,'another handoff/camera'):
            prepare_pairs(self.p/'handoff.json','wrong',review['candidate']['path'],self.p/'wrong-pairs',self.p/'plan/plan.json')

    def test_native_16bit_rgb_and_alpha_are_preserved(self):
        rgb,_=load(self.p/'native.png');rgb=rgb.astype(np.uint16)*257+1
        alpha=np.full(rgb.shape[:2],60001,np.uint16)
        save_png(self.p/'native.png',rgb,alpha);self.refresh()
        review=self.review()
        final,final_alpha=load(review['candidate']['path'])
        self.assertEqual(final.dtype,np.uint16)
        self.assertTrue(np.array_equal(final[3,3],rgb[3,3]))
        self.assertTrue(np.array_equal(final_alpha,alpha))
        self.assertEqual(guard(review,self.p)['status'],'accepted')


if __name__ == '__main__':
    unittest.main()
