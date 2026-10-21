"""Synthetic photo/landscape ground truth tests, not semantic recognition."""
import copy
import json
import unittest

import numpy as np
from PIL import Image

from appearance_guard import check
from handoff import record, read
from photo_composite import PHOTO_CHECKS, compose, prepare, verify, resolve_surface_overlap
from raster import load, save as save_png
from test_contract import save
import test_evidence_paths as evidence_fixture


class PhotoComposite(unittest.TestCase):
    def setUp(self):
        self.case = evidence_fixture.EvidencePaths(); self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.p = self.case.p
        photo = np.full((32, 32, 3), [21, 71, 131], dtype=np.uint8)
        photo[:8, :, 0] = np.arange(32, dtype=np.uint8)
        Image.fromarray(photo).save(self.p / 'photo.png')
        self.case.data['views'][0]['source_photo'] = record(self.p / 'photo.png')
        self.case.refresh()
        stage = self.case.review()
        self.stage = save(self.p / 'landscape-review.json', stage)
        self.catalog = save(self.p / 'catalog.json', {
            'provenance': 'renderer_object_catalog', 'native_scene': self.case.data['native_scene'],
            'objects': [{'name': name, 'type': 'MESH', 'kind': 'plant', 'pass_index': i}
                        for name, i in [('agave-1', 7), ('tiny', 8), ('hidden', 9)]]})
        self.fit = save(self.p / 'fit.json', {'camera_id': 'front', 'photo': record(self.p / 'photo.png')})
        self.plan = prepare(self.p / 'handoff.json', 'front', self.p / 'catalog.json', self.p / 'fit.json', self.p / 'photo-plan')
        self.proof = compose(self.p / 'photo-plan/plan.json', stage['candidate']['path'], self.p / 'final')
        self.review = {'schema_version': 3, 'method': 'photo_composite',
                       'camera_id': 'front', 'handoff': record(self.p / 'handoff.json'),
                       'candidate': self.proof['candidate'], 'landscape_review': self.stage,
                       'photo_composite_proof': record(self.p / 'final/composite-proof.json'),
                       'photo_review': {'reviewer': 'vision-capable-agent', 'source_and_final_loaded': True,
                                        'native_pixel_edges_loaded': True, 'observations': 'Synthetic photo/foreground test truth.',
                                        'paired_evidence': [self.proof['candidate']],
                                        'checks': dict.fromkeys(PHOTO_CHECKS, 'pass')}}

    def test_photo_background_and_all_native_plants_preserved(self):
        result = check(self.review, self.p)
        self.assertTrue(result['photo_background_composited'])
        self.assertEqual(result['landscape_evidence_counts']['preserved'], 1)
        final = load(self.proof['candidate']['path'])[0]
        plate = load(self.plan['photo_plate']['path'])[0]
        self.assertTrue(np.array_equal(final[0, 0], plate[0, 0]))
        stage = load(read(self.p / 'landscape-review.json')['candidate']['path'])[0]
        self.assertTrue(np.array_equal(final[3, 3], stage[3, 3].astype(np.uint16) * 257))

    def test_material_acceptance_does_not_claim_photo_completion(self):
        stage = check(read(self.p / 'landscape-review.json'), self.p)
        self.assertFalse(stage['photo_background_composited'])
        self.assertEqual(stage['completion_scope'], 'landscape_appearance_stage')

    def test_photo_pixel_tampering_rejected_even_with_updated_hash(self):
        rgb, alpha = load(self.proof['candidate']['path']); rgb[0, 0] = 0
        save_png(self.p / 'final/candidate.png', rgb, alpha)
        self.proof['candidate'] = record(self.p / 'final/candidate.png')
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            verify(self.proof, self.p)

    def test_full_frame_mask_rejected(self):
        Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(self.p / 'photo-plan/foreground-mask.png')
        self.plan['foreground_mask'] = record(self.p / 'photo-plan/foreground-mask.png')
        self.proof['plan'] = save(self.p / 'photo-plan/plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'native object selection'):
            verify(self.proof, self.p)

    def test_missing_source_photo_fails(self):
        del self.case.data['views'][0]['source_photo']
        save(self.p / 'handoff.json', self.case.data)
        with self.assertRaisesRegex(ValueError, 'matching source photograph'):
            prepare(self.p / 'handoff.json', 'front', self.p / 'catalog.json', self.p / 'fit.json', self.p / 'missing')

    def test_camera_swap_rejected(self):
        self.plan['camera_fit'] = save(self.p / 'fit.json', {'camera_id': 'other', 'photo': record(self.p / 'photo.png')})
        self.proof['plan'] = save(self.p / 'photo-plan/plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'Camera fit'):
            verify(self.proof, self.p)

    def test_context_ground_or_architecture_cannot_be_added(self):
        catalog = read(self.p / 'catalog.json')
        catalog['objects'].append({'name': 'proxy-house', 'kind': 'existing_context', 'type': 'MESH', 'pass_index': 10})
        self.plan['object_catalog'] = save(self.p / 'catalog.json', catalog)
        self.plan['selected_object_indices'].append(10)
        self.proof['plan'] = save(self.p / 'photo-plan/plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'includes context'):
            verify(self.proof, self.p)

    def test_omitted_tiny_plant_rejected(self):
        self.plan['selected_object_indices'].remove(8)
        self.proof['plan'] = save(self.p / 'photo-plan/plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'omits native landscape'):
            verify(self.proof, self.p)

    def test_missing_visual_photo_check_rejected(self):
        self.review['photo_review']['checks']['house_neighborhood_retained'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'Photo-background integration'):
            check(self.review, self.p)

    def test_unchecked_foreground_cannot_be_substituted(self):
        self.review['landscape_review'] = save(self.p / 'bad-stage.json', {**read(self.p / 'landscape-review.json'), 'camera_id': 'other'})
        with self.assertRaisesRegex(ValueError, 'native-camera landscape'):
            check(self.review, self.p)

    def test_changed_photo_plate_rejected(self):
        plate, _ = load(self.plan['photo_plate']['path']); plate[0, 0] = 0
        save_png(self.p / 'photo-plan/photo-plate.png', plate)
        self.plan['photo_plate'] = record(self.p / 'photo-plan/photo-plate.png')
        self.proof['plan'] = save(self.p / 'photo-plan/plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'documented original-photo preparation'):
            verify(self.proof, self.p)

    def test_original_photo_crop_provenance_is_enforced(self):
        provenance = {'original': record(self.p / 'photo.png'),
                      'working_photo': record(self.p / 'photo.png'),
                      'operations': [{'crop_box': [0, 0, 32, 32]}]}
        save(self.p / 'provenance.json', provenance)
        plan = prepare(self.p / 'handoff.json', 'front', self.p / 'catalog.json', self.p / 'fit.json',
                       self.p / 'original-plan', self.p / 'provenance.json')
        proof = compose(self.p / 'original-plan/plan.json', read(self.p / 'landscape-review.json')['candidate']['path'], self.p / 'original-final')
        verify(proof, self.p)
        plan['source_crop'] = [1, 1, 31, 31]
        proof['plan'] = save(self.p / 'original-plan/plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'Source crop differs'):
            verify(proof, self.p)

    def test_unrelated_text_is_not_photo_comparison_evidence(self):
        (self.p / 'unrelated.txt').write_text('Not image evidence')
        self.review['photo_review']['paired_evidence'] = [record(self.p / 'unrelated.txt')]
        with self.assertRaisesRegex(ValueError, 'readable comparison images'):
            check(self.review, self.p)

    def test_surface_recovery_requires_actual_projected_depth(self):
        ids = np.full((4,4), 10, dtype=int); depth = np.full((4,4), 100.)
        triangle = [{'pixel':[0,0],'camera_xyz':[0,0,2]},
                    {'pixel':[4,0],'camera_xyz':[4,0,2]},
                    {'pixel':[0,4],'camera_xyz':[0,4,2]}]
        projection = {'overlapping_context_indices':[10], 'depth_convention':'ray_distance', 'surfaces':[{'name':'paving','pass_index':7,'triangles':[triangle]}]}
        catalog = {'objects':[{'name':'paving','type':'MESH','kind':'sidewalk','pass_index':7},
                              {'name':'context','type':'MESH','kind':'existing_ground_context','pass_index':10}]}
        self.assertFalse(resolve_surface_overlap(ids, depth, projection, [7], catalog).any())
        depth[0,0] = np.sqrt(.5**2+.5**2+2**2)
        result = resolve_surface_overlap(ids, depth, projection, [7], catalog)
        self.assertTrue(result[0,0]); self.assertEqual(result.sum(), 1)
        with self.assertRaisesRegex(ValueError, 'cannot be widened'):
            resolve_surface_overlap(ids, depth, projection, [7], catalog, tolerance=1)


if __name__ == '__main__':
    unittest.main()
