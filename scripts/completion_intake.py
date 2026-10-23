"""Verify completed native production without converting uncertainty to certification."""
from handoff import checked, read, require


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
    inventory = read(checked(data['source_inventory'], base))
    sources = [s for s in inventory['sources'] if s.get('role') == 'site_photo']
    require({s['sha256'] for s in sources} == {v['source_photo']['sha256'] for v in rows},
            'Every supplied site photo must have its completion view')
    for source in sources: checked(source, base)
    for view in rows:
        path = checked(view['renderer_manifest'], base); manifest = read(path)
        require(manifest.get('finalization') == 'renderer_output' and
                manifest.get('native_scene') == data['native_scene'] and manifest.get('camera_id') == view['camera_id'] and
                manifest.get('dimensions') == view['dimensions'] and manifest.get('image') == view['image'],
                'Actual renderer manifest binding differs')
        require(manifest.get('frozen_snapshot') == data['frozen_snapshot'] and
                manifest.get('scene_state') == data['rendered_scene_state'], 'Renderer frozen-state binding differs')
        for key in ('structural', 'object_ids', 'depth'):
            require(manifest['render_evidence'][key] == view[key], 'Renderer evidence differs: ' + key)
        ep = checked(manifest['native_export'], path.parent); export = read(ep)
        require(export.get('native_scene') == data['native_scene'] and
                export.get('camera_id') == view['camera_id'] and export.get('image') == view['image'] and
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
        checked(manifest['source_render_exr'], path.parent)
    require(data.get('review', {}).get('evidence'), 'Actual upstream review evidence required')
    for ref in data['review']['evidence']: checked(ref, base)
    # Only old completion exports lack requested_cameras. Coverage above supplies independent authority.
    return data.get('requested_cameras', cids)
