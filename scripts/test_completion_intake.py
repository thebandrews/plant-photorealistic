"""Portable synthetic intake regressions; never production certification evidence."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from handoff import check, read, record
from test_contract import fixture, save


def completion_fixture(p):
    data, _ = fixture(p)
    view = data['views'][0]
    scene = data['native_scene']
    projected = read(p/'projected.json')
    projected['plants'][0]['root_xyz'] = [0, 0, 0]
    view['projected_plants'] = save(p/'projected.json', projected)
    proof = read(p/'mask-proof.json')
    proof['projected_plants'] = view['projected_plants']
    view['mask_provenance']['evidence'] = save(p/'mask-proof.json', proof)
    native_plant = copy.deepcopy(projected['plants'][0])
    native_plant.pop('visible_pixels')
    snapshot = save(p/'snapshot.json', {'native_scene': scene, 'instances': [native_plant]})
    state = save(p/'state.json', {'native_scene': scene, 'settings': {'fixed': True}})
    data.update(kind='native_3d_completion_handoff', production_status='completed_with_assumptions',
                certification='not_certified', assumptions=[{'id':'synthetic-estimate'}],
                frozen_snapshot=snapshot, rendered_snapshot=snapshot,
                frozen_scene_state=state, rendered_scene_state=state)
    view['source_photo'] = view['image']
    data['source_inventory'] = save(p/'inventory.json', {'sources':[dict(view['image'],role='site_photo')]})
    export = save(p/'native-export.json', {'native_scene':scene, 'camera_id':'front',
                  'dimensions':view['dimensions'], 'image':view['image'], 'roots':[native_plant]})
    (p/'synthetic.exr').write_bytes(b'Synthetic fixture, not real render data')
    manifest = {'finalization':'renderer_output', 'native_scene':scene, 'camera_id':'front',
                'dimensions':view['dimensions'], 'image':view['image'], 'frozen_snapshot':snapshot,
                'scene_state':state, 'render_evidence':{k:view[k] for k in ('structural','object_ids','depth')},
                'native_export':export, 'source_render_exr':record(p/'synthetic.exr')}
    view['renderer_manifest'] = save(p/'renderer-manifest.json', manifest)
    data['review'] = {'evidence':[data['planting_notes']]}
    val = read(data['native_validation']['path'])
    val.update(kind='native_3d_completion_validation', status='completed_with_assumptions',
               certification={'status':'not_certified'}, assumptions=data['assumptions'],
               native_invariants_passed=True, renderer_pixels_unchanged=True, photo_coverage_passed=True,
               expected_views=1, rendered_views=1, delivered_views=1, plants=1,
               dimensions_by_camera={'front':view['dimensions']})
    data['native_validation'] = save(p/'completion-validation.json', val)
    return data


class CompletionIntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        self.data = completion_fixture(self.p)

    def rewrite_projection(self, key, value):
        view = self.data['views'][0]
        projected = read(view['projected_plants']['path'])
        projected['plants'][0][key] = value
        view['projected_plants'] = save(self.p/'projected.json', projected)
        proof = read(view['mask_provenance']['evidence']['path'])
        proof['projected_plants'] = view['projected_plants']
        view['mask_provenance']['evidence'] = save(self.p/'mask-proof.json', proof)

    def test_completed_with_assumptions_retained(self):
        result = check(self.data,self.p)
        self.assertEqual(result['certification'],'not_certified')
        self.assertEqual(result['production_status'],'completed_with_assumptions')
        self.assertEqual(result['assumptions'],self.data['assumptions'])

    def test_missing_camera_rejected(self):
        self.data['views']=[]
        with self.assertRaises(ValueError):check(self.data,self.p)

    def test_certification_cannot_upgrade(self):
        self.data['certification']='certified'
        with self.assertRaisesRegex(ValueError,'certification'):check(self.data,self.p)

    def render_skill_inventory(self, unproven=True):
        """The render skill's intake form: 'files', paths relative to the inventory, plus a photo with no view."""
        prod = self.p/'production'; prod.mkdir(exist_ok=True)
        (self.p/'other.jpg').write_bytes(b'Synthetic second site photo')
        files = [{'id': 'front-photo', 'role': 'site_photo', 'path': '../native.png',
                  'sha256': self.data['views'][0]['image']['sha256']},
                 {'id': 'other', 'role': 'site_photo', 'path': '../other.jpg', 'sha256': record(self.p/'other.jpg')['sha256']},
                 {'id': 'plan', 'role': 'plan_and_schedule', 'path': '../scene.blend',
                  'sha256': record(self.p/'scene.blend')['sha256']}]
        self.data['source_inventory'] = save(prod/'source-inventory.json', {'schema_version': 1, 'files': files})
        if unproven:
            self.data['unproven_photos'] = ['other']

    def test_render_skill_inventory_with_unproven_photo_accepted(self):
        self.render_skill_inventory()
        self.assertEqual(check(self.data,self.p)['certification'],'not_certified')

    def test_site_photo_without_view_rejected(self):
        self.render_skill_inventory(unproven=False)
        with self.assertRaisesRegex(ValueError,'completion view'):check(self.data,self.p)

    def test_unknown_unproven_photo_rejected(self):
        self.render_skill_inventory()
        self.data['unproven_photos'] = ['other', 'not-in-inventory']
        with self.assertRaisesRegex(ValueError,'source inventory'):check(self.data,self.p)

    def test_manifest_with_relative_paths_accepted(self):
        """Renderer manifests reference files relative to themselves; the handoff uses absolute paths."""
        view = self.data['views'][0]
        mpath = Path(view['renderer_manifest']['path'])
        manifest = read(mpath)
        rel = lambda ref: dict(ref, path=os.path.relpath(ref['path'], mpath.parent))
        for key in ('native_scene', 'image', 'frozen_snapshot', 'scene_state'):
            manifest[key] = rel(manifest[key])
        manifest['render_evidence'] = {k: rel(v) for k, v in manifest['render_evidence'].items()}
        view['renderer_manifest'] = save(mpath, manifest)
        self.assertEqual(check(self.data,self.p)['certification'],'not_certified')
        manifest['image'] = dict(manifest['image'], sha256='0' * 64)
        view['renderer_manifest'] = save(mpath, manifest)
        with self.assertRaises(ValueError):check(self.data,self.p)

    def completion_render_manifest(self, pixel_offset=0.0):
        """The render skill's native_completion_render manifest: roots come from the frozen transforms through
        the renderer's world_to_clip; there is no native_export. pixel_offset moves the frozen root sideways."""
        view = self.data['views'][0]
        w, h = view['dimensions']
        projected = read(view['projected_plants']['path'])
        px, py = projected['plants'][0]['root_pixel']
        snap_path = Path(self.data['frozen_snapshot']['path'])
        snapshot = read(snap_path)
        x, y = 2 * (px + pixel_offset) / w - 1, 1 - 2 * py / h
        snapshot['instances'][0]['matrix_world'] = [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, 2, 0, 0, 0, 1]
        self.data['frozen_snapshot'] = self.data['rendered_snapshot'] = save(snap_path, snapshot)
        projected.update(camera_id='front', image_sha256=view['image']['sha256'])
        view['projected_plants'] = save(Path(view['projected_plants']['path']), projected)
        proof = read(view['mask_provenance']['evidence']['path'])
        proof['projected_plants'] = view['projected_plants']
        view['mask_provenance']['evidence'] = save(self.p/'mask-proof.json', proof)
        mpath = Path(view['renderer_manifest']['path'])
        manifest = read(mpath)
        manifest.pop('native_export')
        manifest.update(kind='native_completion_render', frozen_snapshot=self.data['frozen_snapshot'],
                        world_to_clip=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        view['renderer_manifest'] = save(mpath, manifest)

    def test_completion_render_manifest_accepted(self):
        self.completion_render_manifest()
        self.assertEqual(check(self.data,self.p)['certification'],'not_certified')

    def test_completion_render_moved_root_rejected(self):
        self.completion_render_manifest(pixel_offset=3.0)
        with self.assertRaisesRegex(ValueError,'Root projection'):check(self.data,self.p)

    def test_changed_native_state_rejected(self):
        self.data['rendered_scene_state']=save(self.p/'changed-state.json',{'native_scene':self.data['native_scene'],'settings':{'fixed':False}})
        with self.assertRaisesRegex(ValueError,'Frozen native state'):check(self.data,self.p)

    def test_rehashed_projection_identity_rejected(self):
        self.rewrite_projection('pass_index',8)
        with self.assertRaisesRegex(ValueError,'frozen native identity'):check(self.data,self.p)

    def test_rehashed_projection_root_rejected(self):
        self.rewrite_projection('root_xyz',[.25,0,0])
        with self.assertRaisesRegex(ValueError,'frozen native identity'):check(self.data,self.p)

    def test_rehashed_projection_pixel_root_rejected(self):
        self.rewrite_projection('root_pixel',[15,25])
        with self.assertRaisesRegex(ValueError,'frozen native identity'):check(self.data,self.p)

    def test_native_export_hash_checked(self):
        (self.p/'native-export.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'changed evidence'):check(self.data,self.p)

    def test_native_export_camera_binding_checked(self):
        view=self.data['views'][0]
        manifest=read(view['renderer_manifest']['path'])
        export=read(manifest['native_export']['path']);export['camera_id']='wrong'
        manifest['native_export']=save(self.p/'native-export.json',export)
        view['renderer_manifest']=save(self.p/'renderer-manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'Native export camera/image'):check(self.data,self.p)


if __name__=='__main__':unittest.main()
