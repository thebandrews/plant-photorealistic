"""Bed-ground pass on a synthetic top-down bed: bounded pixels, mean-preserving grain, clean skips."""
import copy
import unittest

import cv2
import numpy as np

import bed_ground as bg
from handoff import read
from photo_finish import verify
from test_contract import save
import test_finishing

N = 160


def manifest(with_matrix=False):
    """Camera 2 m above a flat bed at z = 0, looking straight down, 60 degree field of view."""
    mw = np.eye(4); mw[2, 3] = 2.0
    f, near, far = 1 / np.tan(np.radians(30)), 0.1, 100.0
    proj = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])
    m = {'world_to_clip': (proj @ np.linalg.inv(mw)).tolist()}
    if with_matrix:
        m['camera'] = {'matrix_world': mw.ravel().tolist()}
    return m


class BedGround(unittest.TestCase):
    def setUp(self):
        self.ids = np.full((N, N), 10, np.int32)
        self.ids[60:100, 60:100] = 7                                    # one plant
        self.plants = self.ids == 7
        self.bed = self.ids == 10
        self.depth = np.full((N, N), 2.0)                               # planar depth of the bed
        self.img = np.zeros((N, N, 3), np.uint16)
        self.img[...] = np.array([90, 70, 50]) * 257
        self.img[self.plants] = np.array([40, 120, 50]) * 257
        self.spec = bg.plan_spec()

    def run_pass(self, **kw):
        args = dict(final=self.img, ids=self.ids, depth=self.depth, plants=self.plants, bed_area=self.bed,
                    manifest=manifest(), spec=self.spec, base='.')
        args.update(kw)
        return bg.apply(**args)

    def test_camera_from_world_to_clip_matches_matrix_world(self):
        a, b = bg.camera(manifest()), bg.camera(manifest(True))
        self.assertTrue(np.allclose(a[1], b[1]) and np.allclose(a[2], b[2]))
        self.assertTrue(np.allclose(a[1], [0, 0, 2]) and np.allclose(a[2], [0, 0, -1]))

    def test_plants_ring_and_non_bed_pixels_keep_exact_samples(self):
        bed = self.bed.copy(); bed[:, :12] = False                      # a path strip that is not bed
        out, info = self.run_pass(bed_area=bed)
        self.assertEqual(info['status'], 'applied')
        self.assertEqual(info['depth_convention'], 'planar')
        ring = cv2.dilate(self.plants.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
        changed = np.any(out != self.img, axis=2)
        self.assertFalse(np.any(changed & ring))
        self.assertFalse(np.any(changed & ~bed))
        self.assertGreater(changed.sum(), 1000)
        self.assertEqual(info['guard_changed_px'] + info['outside_bed_changed_px'], 0)

    def test_steep_bed_faces_are_left_alone(self):
        depth = self.depth.copy()
        depth[:24] = 2.0 + 0.05 * np.arange(24)[:, None]              # a face dropping 5 cm per pixel row
        out, info = self.run_pass(depth=depth)
        self.assertEqual(info['status'], 'applied')
        changed = np.any(out != self.img, axis=2)
        self.assertFalse(changed[:22].any())
        self.assertTrue(changed[40:50].any())

    def test_grain_preserves_the_ground_tone(self):
        out, _ = self.run_pass()
        changed = np.any(out != self.img, axis=2)
        luma = lambda a: (a.astype(np.float64) / 257) @ bg.LUMA
        before, after = luma(self.img)[changed].mean(), luma(out)[changed].mean()
        self.assertLess(abs(after - before) / before, 0.04)
        self.assertGreater(luma(out)[changed].std(), 3)                 # real grain, not a flat shift

    def test_deterministic(self):
        self.assertTrue(np.array_equal(self.run_pass()[0], self.run_pass()[0]))

    def test_skips_cleanly(self):
        for kw, reason in [({'bed_area': np.zeros((N, N), bool)}, 'no_bed_ground'),
                           ({'manifest': None}, 'no_renderer_camera'),
                           ({'bed_area': (cv2.dilate(self.plants.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) &
                             ~self.plants}, 'no_open_bed_ground')]:                # only the ring round the plant
            out, info = self.run_pass(**kw)
            self.assertEqual((info['status'], info['reason']), ('skipped', reason))
            self.assertTrue(np.array_equal(out, self.img))

    def test_plan_entry(self):
        self.assertEqual(bg.plan_spec(False)['operation'], 'off')
        self.assertFalse(bg.check_spec(bg.plan_spec(False), '.'))
        self.assertTrue(bg.check_spec(self.spec, '.'))
        bad = copy.deepcopy(self.spec); bad['exemplars'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Missing or changed'):
            bg.check_spec(bad, '.')


class FinishRecordsBedGround(test_finishing.PhotoFinishing):
    """The finishing plan freezes the pass, the proof records its outcome, and verification recomputes it."""
    def runTest(self):
        self.assertEqual(self.plan['bed_ground']['operation'], bg.OPERATION)
        self.assertEqual(self.proof['bed_ground'], {'status': 'skipped', 'reason': 'no_renderer_camera'})
        proof = read(self.p / 'final/finish-proof.json')
        proof['bed_ground'] = {'status': 'applied'}
        with self.assertRaisesRegex(ValueError, 'Bed ground record differs'):
            verify(proof, self.p / 'final')


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(BedGround))
    suite.addTest(FinishRecordsBedGround())
    return suite


if __name__ == '__main__':
    unittest.main()
