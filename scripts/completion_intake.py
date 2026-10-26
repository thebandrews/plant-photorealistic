"""Verify completed native production without converting uncertainty to certification."""
from pathlib import Path

import numpy as np

from handoff import checked, read, require


def same(ref, ref_base, expected, expected_base):
    """Hashed references to the same content. Each record resolves against its own file (renderer manifests and
    exports use paths relative to themselves, the handoff absolute ones), must verify there, and hashes must match."""
    if not (isinstance(ref, dict) and isinstance(expected, dict)):
        return False
    checked(ref, ref_base); checked(expected, expected_base)
    return ref['sha256'] == expected['sha256']


def check_projected_roots(manifest, view, base, frozen, ledger):
    """The render skill's completion manifest (native_completion_render) carries no root export. Its plant roots
    are the frozen instances' origins through the renderer's own world_to_clip, as native_completion.py checks."""
    require(manifest.get('kind') == 'native_completion_render', 'Native export or completion render manifest required')
    doc = read(checked(view['projected_plants'], base))
    require(doc.get('camera_id') == view['camera_id'] and doc.get('image_sha256') == view['image']['sha256'],
            'Plant projection is not bound to the native image')
    w2c = np.asarray(manifest.get('world_to_clip'), dtype=float)
    require(w2c.shape == (4, 4) and np.isfinite(w2c).all(), 'Renderer world-to-clip matrix required')
    width, height = view['dimensions']
    passes = {p['id']: p.get('pass_index') for p in ledger}
    rows = doc['plants']
    require(len(rows) == len(frozen) and {r['id'] for r in rows} == set(frozen), 'Native root population differs')
    for row in rows:
        fixed = frozen[row['id']]
        require(row.get('species') == fixed.get('species') and
                (passes.get(row['id']) is None or row.get('pass_index') == passes[row['id']]),
                'Projected plant differs from frozen native identity')
        m = fixed.get('matrix_world')
        require(isinstance(m, list) and len(m) == 16, 'Frozen plant transform required')
        clip = w2c @ np.array([m[3], m[7], m[11], 1.0])
        pixel = row.get('root_pixel')
        if clip[3] > 1e-12:
            expected = [(clip[0] / clip[3] + 1) * width / 2, (1 - clip[1] / clip[3]) * height / 2]
            require(isinstance(pixel, list) and len(pixel) == 2 and
                    np.linalg.norm(np.asarray(pixel, dtype=float) - expected) <= .5,
                    'Root projection differs from frozen native transform: ' + row['id'])
        else:
            require(pixel is None, 'Behind-camera root must have null projection: ' + row['id'])


def verify(data, base, validation, ledger):
    require(data.get('production_status') == 'completed_with_assumptions' and
            data.get('certification') == 'not_certified', 'Completion certification must remain explicit')
    require(validation.get('kind') == 'native_3d_completion_validation' and
            validation.get('status') == 'completed_with_assumptions', 'Matching native completion validation required')
    for key in ('native_invariants_passed', 'renderer_pixels_unchanged', 'photo_coverage_passed'):
        require(validation.get(key) is True, 'Unresolved native integrity failure: ' + key)
    require(validation.get('certification', {}).get('status') == 'not_certified' and
            data.get('assumptions') == validation.get('assumptions') and data.get('assumptions'),
            'Carry the original assumptions and certification outcome')
    rows = data['views']; cids = [v['camera_id'] for v in rows]
    require(len(cids) == len(set(cids)) and cids, 'Unique completion cameras required')
    require(all(validation.get(k) == len(cids) for k in ('expected_views', 'rendered_views', 'delivered_views')),
            'Native completion camera coverage differs')
    require(validation.get('plants') == len(ledger) and
            validation.get('dimensions_by_camera') == {v['camera_id']: v['dimensions'] for v in rows},
            'Completion plant population or dimensions differ')
    for before, after in (('frozen_snapshot', 'rendered_snapshot'), ('frozen_scene_state', 'rendered_scene_state')):
        a = read(checked(data[before], base)); b = read(checked(data[after], base))
        require(a == b and a.get('native_scene', {}).get('sha256') == data['native_scene']['sha256'],
                'Frozen native state changed or belongs to another scene')
    snapshot = read(checked(data['frozen_snapshot'], base))
    frozen = {p['id']: p for p in snapshot['instances']}
    require({p['id']: p['species'] for p in snapshot['instances']} == {p['id']: p['species'] for p in ledger},
            'Completion snapshot/ledger bindings differ')
    # The render skill's intake inventory lists files under 'files', with paths relative to the inventory
    # ('sources' is the older form). Photos the user excluded or the camera proof rejected have no view; the
    # upstream completion validator checked both (photo_coverage_passed above).
    inv_path = checked(data['source_inventory'], base)
    inventory = read(inv_path)
    listed = inventory.get('files', inventory.get('sources'))
    require(isinstance(listed, list) and listed, 'Source inventory lists no files')
    photos = {}
    for source in listed:
        if source.get('role') == 'site_photo':
            sid = source.get('id') or Path(source['path']).stem
            require(sid not in photos, 'Duplicate source-photo ID: ' + sid)
            photos[sid] = source
    skipped = set(data.get('excluded_photos', [])) | set(data.get('unproven_photos', []))
    require(skipped <= set(photos), 'Excluded or unproven photos must be in the source inventory')
    require({photos[k]['sha256'] for k in set(photos) - skipped} == {v['source_photo']['sha256'] for v in rows},
            'Every supplied site photo must have its completion view, unless excluded or not proven')
    for source in photos.values(): checked(source, inv_path.parent)
    for view in rows:
        path = checked(view['renderer_manifest'], base); manifest = read(path); mdir = path.parent
        require(manifest.get('finalization') == 'renderer_output' and
                same(manifest.get('native_scene'), mdir, data['native_scene'], base) and
                manifest.get('camera_id') == view['camera_id'] and manifest.get('dimensions') == view['dimensions'] and
                same(manifest.get('image'), mdir, view['image'], base), 'Actual renderer manifest binding differs')
        require(same(manifest.get('frozen_snapshot'), mdir, data['frozen_snapshot'], base) and
                same(manifest.get('scene_state'), mdir, data['rendered_scene_state'], base),
                'Renderer frozen-state binding differs')
        for key in ('structural', 'object_ids', 'depth'):
            require(same(manifest['render_evidence'][key], mdir, view[key], base), 'Renderer evidence differs: ' + key)
        if 'native_export' in manifest:
            ep = checked(manifest['native_export'], mdir); export = read(ep)
            require(same(export.get('native_scene'), ep.parent, data['native_scene'], base) and
                    export.get('camera_id') == view['camera_id'] and
                    same(export.get('image'), ep.parent, view['image'], base) and
                    export.get('dimensions') == view['dimensions'], 'Native export camera/image binding differs')
            roots = {p['id']: p for p in export['roots']}
            projected = read(checked(view['projected_plants'], base))['plants']
            require(len(roots) == len(export['roots']) and set(roots) == set(frozen), 'Native root population differs')
            for row in projected:
                native = roots.get(row['id'], {}); fixed = frozen.get(row['id'], {})
                require(row.get('pass_index') == native.get('pass_index') == fixed.get('pass_index') and
                        row.get('species') == native.get('species') == fixed.get('species') and
                        row.get('root_pixel') == native.get('root_pixel') and row.get('root_xyz') == native.get('root_xyz'),
                        'Projected plant differs from frozen native identity/root export')
        else:
            check_projected_roots(manifest, view, base, frozen, ledger)
        checked(manifest['source_render_exr'], path.parent)
    require(data.get('review', {}).get('evidence'), 'Actual upstream review evidence required')
    for ref in data['review']['evidence']: checked(ref, base)
    # Only old completion exports lack requested_cameras. Coverage above supplies independent authority.
    return data.get('requested_cameras', cids)
