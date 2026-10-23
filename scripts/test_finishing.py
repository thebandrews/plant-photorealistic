"""Synthetic operation/permission tests; no semantic recognition claims."""
import copy
import unittest

import numpy as np
from PIL import Image

from appearance_guard import check, GEOMETRY
from botanical_refinement import compare, prepare as prepare_botany
from handoff import record, read
from photo_composite import prepare as prepare_photo
from photo_finish import FINISH_CHECKS, blend, compose, prepare, preview, verify, context
from raster import load, save as save_png
from test_contract import save
import test_evidence_paths as fixtures


class PhotoFinishing(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.EvidencePaths(); self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.p = self.case.p
        ids = np.load(self.p / 'ids.npy'); ids[25:30, 2:8] = 10; np.save(self.p / 'ids.npy', ids)
        self.ids = ids
        projection = read(self.p / 'projected.json')
        for plant in projection['plants']: plant['visible_pixels'] = int((ids == plant['pass_index']).sum())
        save(self.p / 'projected.json', projection)
        Image.fromarray(np.full((32, 32, 3), [20, 70, 130], np.uint8)).save(self.p / 'photo.png')
        self.case.data['views'][0]['source_photo'] = record(self.p / 'photo.png'); self.case.refresh()
        self.stage = self.case.review(); save(self.p / 'stage.json', self.stage)
        objects = [{'name': name, 'type': 'MESH', 'kind': 'plant', 'pass_index': index}
                   for name, index in [('agave-1', 7), ('tiny', 8), ('hidden', 9)]]
        objects.append({'name': 'ground', 'type': 'MESH', 'kind': 'bed', 'pass_index': 10})
        save(self.p / 'catalog.json', {'provenance': 'renderer_object_catalog', 'native_scene': self.case.data['native_scene'], 'objects': objects})
        save(self.p / 'fit.json', {'camera_id': 'front', 'photo': record(self.p / 'photo.png')})
        prepare_photo(self.p / 'handoff.json', 'front', self.p / 'catalog.json', self.p / 'fit.json', self.p / 'photo-plan')
        alpha = np.isin(ids, [7, 8, 9, 10]).astype(np.uint8) * 255
        alpha[3, 3] = 128  # Genuine synthetic subpixel coverage of the tiny plant.
        Image.fromarray(alpha).save(self.p / 'matte.png')
        matte = {'provenance': 'renderer_landscape_matte', 'camera_id': 'front',
                 'native_scene_sha256': self.case.data['native_scene']['sha256'],
                 'appearance_image_sha256': self.stage['candidate']['sha256'], 'rgb_encoding': 'straight_srgb',
                 'object_catalog': record(self.p / 'catalog.json'), 'object_ids': record(self.p / 'ids.npy'),
                 'depth': record(self.p / 'depth.npy'), 'straight_rgb': self.stage['candidate'], 'alpha': record(self.p / 'matte.png')}
        save(self.p / 'matte.json', matte)
        cleanup = np.zeros((32, 32), np.uint8); cleanup[:2, :2] = 255
        Image.fromarray(cleanup).save(self.p / 'cleanup.png')
        save(self.p / 'inventory.json', {'regions': [{'id': 'worker', 'kind': 'person', 'description': 'Synthetic worker and shadow', 'mask': record(self.p / 'cleanup.png')}]})
        transitions = np.zeros((32, 32), np.uint8); transitions[27, 4] = 128
        Image.fromarray(transitions).save(self.p / 'transition.png')
        self.plan = prepare(self.p / 'photo-plan/plan.json', self.p / 'stage.json', self.p / 'matte.json',
                            self.p / 'inventory.json', self.p / 'transition.png', self.p / 'finish-plan.json')
        Image.fromarray(np.full((32, 32, 3), [100, 120, 140], np.uint8)).save(self.p / 'cleanup-raw.png')
        Image.fromarray(np.full((32, 32, 3), [60, 80, 100], np.uint8)).save(self.p / 'transition-raw.png')
        self.proof = compose(self.p / 'finish-plan.json', self.p / 'final', self.p / 'cleanup-raw.png', self.p / 'transition-raw.png')
        self.review = {'schema_version': 4, 'method': 'photo_finish', 'handoff': record(self.p / 'handoff.json'),
                       'camera_id': 'front', 'candidate': self.proof['candidate'], 'finish_proof': record(self.p / 'final/finish-proof.json'),
                       'photo_review': {'reviewer': 'vision-capable-agent', 'source_and_final_loaded': True,
                            'native_pixel_edges_loaded': True, 'observations': 'Synthetic fixture truth, not actual photo recognition.',
                            'paired_evidence': [self.proof['candidate']], 'checks': dict.fromkeys(FINISH_CHECKS, 'pass'),
                            'cleanup': [{'id': 'worker', 'status': 'removed', 'observations': 'Synthetic cleanup region replaced.',
                                         'paired_evidence': self.proof['candidate']}]}}

    def test_completion_discloses_semantic_limits_without_upgrading_acceptance(self):
        self.review['release_policy'] = 'complete_with_limitations'
        self.review['photo_review']['checks']['camera_alignment'] = 'estimated'
        self.review['limitations'] = [{'check':'camera_alignment', 'observations':'Synthetic source pose estimate remains unverified.'}]
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'completed_with_limitations')
        self.assertEqual(result['nonpassing_checks'], ['camera_alignment'])
        self.assertTrue(result['photo_background_composited'])

    def test_completion_never_bypasses_native_or_pixel_integrity(self):
        self.review['release_policy'] = 'complete_with_limitations'; self.review['limitations'] = []
        rgb, aa = load(self.p/'final/candidate.png'); rgb[0,10] = 0
        save_png(self.p/'final/candidate.png', rgb)
        self.proof['candidate'] = record(self.p/'final/candidate.png'); save(self.p/'final/finish-proof.json', self.proof)
        self.review['candidate'] = self.proof['candidate']; self.review['finish_proof'] = record(self.p/'final/finish-proof.json')
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'): check(self.review, self.p)

    def test_completion_can_truthfully_retain_an_unresolved_cleanup_target(self):
        self.review['release_policy'] = 'complete_with_limitations'
        self.review['photo_review']['checks']['people_removed'] = 'fail'
        self.review['limitations'] = [{'check':'people_removed', 'observations':'Synthetic target remains visible after targeted recovery.'}]
        self.review['photo_review']['cleanup'][0]['status'] = 'partial'
        result = check(self.review, self.p)
        self.assertEqual(result['status'], 'completed_with_limitations')
        self.assertEqual(result['unresolved_cleanup_targets'], ['worker'])
        self.review['photo_review']['checks']['people_removed'] = 'pass'; self.review['limitations'] = []
        with self.assertRaisesRegex(ValueError, 'Unresolved cleanup target'): check(self.review, self.p)

    def test_completion_requires_each_nonpass_finding(self):
        self.review['release_policy'] = 'complete_with_limitations'
        self.review['photo_review']['checks']['camera_alignment'] = 'estimated'; self.review['limitations'] = []
        with self.assertRaisesRegex(ValueError, 'Describe each'): check(self.review, self.p)

    def test_cleanup_alpha_blend_transition_and_exact_protection(self):
        result = check(self.review, self.p)
        self.assertTrue(result['photo_background_composited']); self.assertEqual(result['cleanup_regions'], 1)
        final = load(self.proof['candidate']['path'])[0]
        self.assertTrue(np.array_equal(final[0, 0], np.array([100, 120, 140]) * 257))
        self.assertTrue(np.array_equal(final[0, 10], np.array([20, 70, 130]) * 257))
        fg = load(self.stage['candidate']['path'])[0].astype(np.uint16) * 257
        plate = load(self.p / 'photo-plan/photo-plate.png')[0]
        self.assertTrue(np.array_equal(final[3, 3], blend(fg, plate, np.full((32, 32), 128, np.uint8))[3, 3]))
        self.assertTrue(np.array_equal(final[16, 16], fg[16, 16]))
        self.assertFalse(np.array_equal(final[27, 4], fg[27, 4]))

    def test_preview_skips_transition_and_cannot_claim_completion(self):
        result = preview(self.p / 'finish-plan.json', self.p / 'preview', self.p / 'cleanup-raw.png')
        self.assertEqual(result['status'], 'preview_only')
        self.assertFalse((self.p / 'preview/finish-proof.json').exists())
        self.assertFalse(np.array_equal(load(result['image']['path'])[0][27, 4], load(self.proof['candidate']['path'])[0][27, 4]))

    def test_no_cleanup_targets_supported_after_inspection(self):
        self.plan['cleanup_regions'] = []; self.plan.pop('cleanup_inventory', None); save(self.p / 'empty-plan.json', self.plan)
        proof = compose(self.p / 'empty-plan.json', self.p / 'empty-final', transition_raw=self.p / 'transition-raw.png')
        verify(proof, self.p)

    def test_raw_edits_required_when_permissions_nonempty(self):
        with self.assertRaisesRegex(ValueError, 'Raw photo cleanup'):
            compose(self.p / 'finish-plan.json', self.p / 'bad')
        with self.assertRaisesRegex(ValueError, 'Raw transition'):
            compose(self.p / 'finish-plan.json', self.p / 'bad', self.p / 'cleanup-raw.png')

    def test_transition_cannot_paint_plants_or_distant_house(self):
        for point in [(3, 3), (0, 31)]:
            mask = np.zeros((32, 32), np.uint8); mask[point] = 255
            Image.fromarray(mask).save(self.p / 'bad-transition.png')
            plan = copy.deepcopy(self.plan); plan['transition_alpha'] = record(self.p / 'bad-transition.png')
            with self.assertRaisesRegex(ValueError, 'protected plant/background'):
                context(plan, self.p)

    def test_matte_cannot_erase_tiny_plant_or_expand_into_house(self):
        original = np.array(Image.open(self.p / 'matte.png'))
        for point, value, message in [((3, 3), 0, 'erased visible plant'), ((0, 31), 255, 'native edge band')]:
            alpha = original.copy(); alpha[point] = value
            Image.fromarray(alpha).save(self.p / 'bad-matte.png')
            matte = read(self.p / 'matte.json'); matte['alpha'] = record(self.p / 'bad-matte.png')
            plan = copy.deepcopy(self.plan); plan['renderer_matte'] = save(self.p / 'bad-matte.json', matte)
            with self.assertRaisesRegex(ValueError, message): context(plan, self.p)

    def test_actual_candidate_ids_required(self):
        ids = self.ids.copy(); ids[2, 2] = 7; np.save(self.p / 'fake-ids.npy', ids)
        matte = read(self.p / 'matte.json'); matte['object_ids'] = record(self.p / 'fake-ids.npy')
        plan = copy.deepcopy(self.plan); plan['renderer_matte'] = save(self.p / 'bad-matte.json', matte)
        with self.assertRaisesRegex(ValueError, 'actual candidate renderer IDs'): context(plan, self.p)

    def test_updated_hash_cannot_hide_changed_final_background(self):
        final, _ = load(self.proof['candidate']['path']); final[0, 10] = 0
        save_png(self.proof['candidate']['path'], final); self.proof['candidate'] = record(self.proof['candidate']['path'])
        with self.assertRaisesRegex(ValueError, 'reconstruction differs'): verify(self.proof, self.p)

    def test_unresolved_clutter_or_missing_semantic_check_fails(self):
        bad = copy.deepcopy(self.review); bad['photo_review']['cleanup'][0]['status'] = 'retained'
        with self.assertRaisesRegex(ValueError, 'Unresolved cleanup'): check(bad, self.p)
        bad = copy.deepcopy(self.review); bad['photo_review']['checks']['natural_plant_anatomy'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'anatomy or photo integration'): check(bad, self.p)

    def test_full_frame_cleanup_mask_rejected(self):
        Image.fromarray(np.full((32, 32), 255, np.uint8)).save(self.p / 'full.png')
        plan = copy.deepcopy(self.plan); plan['cleanup_regions'][0]['mask'] = record(self.p / 'full.png')
        with self.assertRaisesRegex(ValueError, 'Localized binary|Cleanup inventory permissions changed'): context(plan, self.p)


class BotanicalRefinement(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.EvidencePaths(); self.case.setUp(); self.addCleanup(self.case.doCleanups); self.p = self.case.p
        projection = read(self.p / 'projected.json')['plants']
        matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        plants = [dict(id=p['id'], species=p['species'], pass_index=p['pass_index'], asset_id='asset', seed=1,
                       matrix_world=matrix, root_world=[0, 0, 0], root_collar={'vertices': 2, 'sha256': 'fixed', 'radius_world': .05},
                       geometry_sha256='old', opacity_mapping_sha256='fixed') for p in projection]
        objects = [dict(id=p['id'], kind='plant', geometry_sha256='old', matrix_world=matrix) for p in plants]
        objects.append({'id': 'wall', 'kind': 'retaining_wall', 'geometry_sha256': 'fixed'})
        self.before = {'exporter': 'botanical_invariants_v1', 'native_scene': self.case.data['native_scene'],
                       'invariants': {'plants': plants, 'objects': objects, 'cameras': {'front': 'fixed'},
                                      'settings': {'lights': 'fixed', 'film_transparent': False}}}
        save(self.p / 'botanical-before.json', self.before)
        prepare_botany(self.p / 'handoff.json', self.p / 'botanical-before.json', ['agave-1'], self.p / 'botanical-plan.json')
        self.after = copy.deepcopy(self.before)
        (self.p / 'revised.blend').write_bytes(b'Synthetic revised native scene')
        self.after['native_scene'] = record(self.p / 'revised.blend')
        self.after['invariants']['plants'][0]['geometry_sha256'] = 'new'
        self.after['invariants']['objects'][0]['geometry_sha256'] = 'new'
        self.after['invariants']['settings']['film_transparent'] = True
        save(self.p / 'botanical-after.json', self.after)
        ids = np.load(self.p / 'ids.npy'); ids[2, 2] = 7; np.save(self.p / 'candidate-ids.npy', ids)
        proof = {'provenance': 'renderer_export', 'camera_id': 'front', 'native_scene_sha256': self.after['native_scene']['sha256'],
                 'image_sha256': record(self.p / 'candidate.png')['sha256'], 'object_ids': record(self.p / 'candidate-ids.npy'), 'depth': record(self.p / 'depth.npy')}
        save(self.p / 'botanical-render.json', proof)
        rows = [dict(id=p['id'], expected_species=p['species'], identity_source='validated_handoff',
                     appearance_assessment='consistent', observations='Synthetic planned leaf geometry refinement.',
                     measurement_evidence=record(self.p / 'candidate.png')) for p in projection]
        extras = ('species_habit', 'branch_taper_attachment', 'leaf_scale_distribution', 'presentation_size', 'root_positions')
        self.review = {'schema_version': 4, 'method': 'native_botanical_refinement', 'handoff': record(self.p / 'handoff.json'),
                       'camera_id': 'front', 'candidate': record(self.p / 'candidate.png'), 'refinement_plan': record(self.p / 'botanical-plan.json'),
                       'native_after': record(self.p / 'botanical-after.json'), 'render_proof': record(self.p / 'botanical-render.json'),
                       'recognition': {'reviewer': 'vision-capable-agent', 'baseline_and_candidate_loaded': True,
                            'original_pixel_crops_loaded': True, 'observations': 'Synthetic fixture truth.', 'paired_crop_evidence': [record(self.p / 'candidate.png')],
                            'geometry': dict.fromkeys(GEOMETRY + extras, 'pass'), 'plants': rows, 'image_quality': 'improved', 'quality_observations': 'Synthetic anatomy refined.'}}

    def test_planned_geometry_and_new_crown_pixels_accepted(self):
        result = check(self.review, self.p)
        self.assertEqual(result['changed_geometry_ids'], ['agave-1']); self.assertFalse(result['photo_background_composited'])

    def test_species_location_root_and_hardscape_remain_locked(self):
        mutations = [('plants', 0, 'species', 'Other species'), ('plants', 0, 'root_world', [1, 0, 0]),
                     ('plants', 0, 'root_collar', {'vertices': 0}), ('objects', 3, 'geometry_sha256', 'moved-wall')]
        for section, index, field, value in mutations:
            after = copy.deepcopy(self.after); after['invariants'][section][index][field] = value
            with self.assertRaisesRegex(ValueError, 'changed'): compare(self.before, after, ['agave-1'])

    def test_undeclared_plant_geometry_rejected(self):
        after = copy.deepcopy(self.after); after['invariants']['plants'][1]['geometry_sha256'] = 'new'
        with self.assertRaisesRegex(ValueError, 'Undeclared'): compare(self.before, after, ['agave-1'])

    def test_geometry_change_requires_visual_anatomy_evidence(self):
        self.review['recognition']['plants'][0]['appearance_assessment'] = 'uncertain'
        with self.assertRaisesRegex(ValueError, 'changed botanical anatomy'): check(self.review, self.p)

    def test_camera_and_opacity_changes_rejected(self):
        after = copy.deepcopy(self.after); after['invariants']['cameras']['front'] = 'shifted'
        with self.assertRaisesRegex(ValueError, 'changed'): compare(self.before, after, ['agave-1'])
        after = copy.deepcopy(self.after); after['invariants']['plants'][0]['opacity_mapping_sha256'] = 'new'
        with self.assertRaisesRegex(ValueError, 'changed'): compare(self.before, after, ['agave-1'])


if __name__ == '__main__':
    unittest.main()
