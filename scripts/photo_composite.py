"""Freeze and verify an original-photo background plus native landscape overlay."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from handoff import checked, check as check_handoff, read, record, require
from raster import load, save, scaled_like

LANDSCAPE_KINDS = {'plant', 'bed', 'planter', 'retaining_wall', 'edging',
                   'stepping_stone', 'step', 'walk_slab', 'joint_substrate',
                   'accent_boulder', 'paver_crossing', 'paver_band', 'porch_slab',
                   'curb', 'sidewalk', 'nonliving_surface_detail'}
PHOTO_CHECKS = ('source_photo_background', 'camera_alignment', 'landscape_placement',
                'house_neighborhood_retained', 'occlusion_edges', 'ground_contacts',
                'lighting_shadows', 'construction_transitions', 'cross_view_consistency')


def landscape_kind(obj):
    return obj.get('photo_kind', obj.get('kind'))


def verify_roles(catalog, base):
    roles = [o for o in catalog['objects'] if 'photo_kind' in o]
    if not roles: return
    rp = checked(catalog['photo_role_map'], base); rolemap = read(rp)
    require(rolemap.get('native_scene') == catalog['native_scene'] and rolemap.get('observations') and rolemap.get('evidence'), 'Reviewed scene-bound photo role map required')
    for ref in rolemap['evidence']: checked(ref, rp.parent)
    expected = {o['name']: o for o in rolemap['objects']}
    require(len(expected) == len(roles) and set(expected) == {o['name'] for o in roles}, 'Role map/catalog membership differs')
    for obj in roles:
        row = expected[obj['name']]
        require(obj['type'] == 'MESH' and row['native_kind'] == obj['kind'] and row['photo_kind'] == obj['photo_kind'] and obj['photo_kind'] in LANDSCAPE_KINDS, 'Invalid proposed landscape role')


def resolve_surface_overlap(ids, depth, projection, allowed, catalog, tolerance=.01):
    """Only recover actual projected landscape within 0.01 native units of context."""
    require(tolerance == .01, 'Surface overlap tolerance cannot be widened')
    background = projection['overlapping_context_indices']
    convention = projection.get('depth_convention')
    require(convention in ('camera_z', 'ray_distance'), 'Recorded native depth convention required')
    objects = {o['pass_index']: o for o in catalog['objects'] if o['type'] == 'MESH'}
    require(all(objects[i]['kind'] == 'existing_ground_context' for i in background), 'Only coincident context ground can be recovered')
    eligible = np.isin(ids, background); recovered = np.zeros(ids.shape, dtype=bool); height, width = ids.shape
    for surface in projection['surfaces']:
        require(surface['pass_index'] in allowed and objects[surface['pass_index']]['kind'] != 'plant', 'Only selected nonplant native surfaces can recover overlap')
        require(objects[surface['pass_index']]['name'] == surface['name'], 'Surface/catalog binding differs')
        for triangle in surface['triangles']:
            xy = np.asarray([v['pixel'] for v in triangle], dtype=float); xyz = np.asarray([v['camera_xyz'] for v in triangle], dtype=float)
            require(xy.shape == (3, 2) and xyz.shape == (3, 3) and np.all(np.isfinite(xy)) and np.all(np.isfinite(xyz)) and np.all(xyz[:, 2] > 0), 'Finite front-facing surface triangle required')
            x0, y0 = np.maximum(np.floor(xy.min(0)).astype(int), 0); x1, y1 = np.minimum(np.ceil(xy.max(0)).astype(int) + 1, [width, height])
            if x1 <= x0 or y1 <= y0 or not np.any(eligible[y0:y1, x0:x1]):
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]; xx = xx + .5; yy = yy + .5
            a, b, c = xy; den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(den) < 1e-12:
                continue
            u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
            v = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
            w = 1-u-v; inside = (u >= -1e-8)&(v >= -1e-8)&(w >= -1e-8)
            reciprocal = u/xyz[0,2]+v/xyz[1,2]+w/xyz[2,2]
            points = [(u*xyz[0,j]/xyz[0,2]+v*xyz[1,j]/xyz[1,2]+w*xyz[2,j]/xyz[2,2])/np.maximum(reciprocal,1e-30) for j in range(3)]
            distance = points[2] if convention == 'camera_z' else np.sqrt(sum(point*point for point in points))
            recovered[y0:y1,x0:x1] |= inside & eligible[y0:y1,x0:x1] & (np.abs(distance-depth[y0:y1,x0:x1]) <= tolerance)
    return recovered


def selected_mask(plan, base, handoff, hp, view, ids, allowed, catalog):
    mask = np.isin(ids, allowed)
    if plan.get('surface_projection'):
        sp = checked(plan['surface_projection'], base); projection = read(sp)
        require(projection.get('provenance') == 'renderer_surface_projection' and projection.get('native_scene') == handoff['native_scene'], 'Actual validated-scene surface projection required')
        match = next((v for v in projection['views'] if v['camera_id'] == view['camera_id']), None)
        require(match is not None and match['dimensions'] == view['dimensions'], 'Surface projection camera differs')
        depth = np.load(checked(view['depth'], hp.parent), allow_pickle=False)
        mask |= resolve_surface_overlap(ids, depth, match, allowed, catalog)
    return mask


def plate_pixels(source, dimensions, crop=None):
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
        if crop is not None:
            require(len(crop) == 4 and all(type(x) is int for x in crop), 'Integer source crop required')
            x0, y0, x1, y1 = crop
            require(0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height, 'Invalid source crop')
            image = image.crop(crop)
        require(abs(image.width / image.height - dimensions[0] / dimensions[1]) < .001,
                'Photo crop aspect differs; do not warp the photo or landscape')
        image = image.resize(tuple(dimensions), Image.Resampling.LANCZOS)
        return np.array(image).astype(np.uint16) * 257


def context(plan, base):
    require(plan.get('schema_version') == 1 and plan.get('kind') == 'photo_composite_plan',
            'Frozen original-photo plan required')
    hp = checked(plan['handoff'], base)
    handoff = read(hp)
    view = next((v for v in handoff['views'] if v['camera_id'] == plan['camera_id']), None)
    require(view is not None and view.get('source_photo'), 'Matching original source photograph required')
    require(plan['source_photo'] == view['source_photo'] and plan['dimensions'] == view['dimensions'],
            'Photo plan differs from handoff camera/source/dimensions')
    checked(view['source_photo'], hp.parent)
    fit_path = checked(plan['camera_fit'], base)
    fit = read(fit_path)
    require(fit.get('camera_id') == plan['camera_id'] and
            fit.get('photo', {}).get('sha256') == view['source_photo']['sha256'],
            'Camera fit is not bound to this camera and photograph')
    cat_path = checked(plan['object_catalog'], base)
    catalog = read(cat_path)
    verify_roles(catalog, cat_path.parent)
    require(catalog.get('provenance') == 'renderer_object_catalog' and
            catalog.get('native_scene', {}).get('sha256') == handoff['native_scene']['sha256'],
            'Actual catalog from the validated native scene required')
    checked(catalog['native_scene'], cat_path.parent)
    objects = [o for o in catalog['objects'] if o['type'] == 'MESH']
    indices = [o['pass_index'] for o in objects]
    require(len(indices) == len(set(indices)) and all(type(i) is int and i > 0 for i in indices),
            'Unique native mesh indices required')
    allowed = sorted(o['pass_index'] for o in objects if landscape_kind(o) in LANDSCAPE_KINDS)
    require(plan['selected_object_indices'] == allowed, 'Foreground selection includes context or omits native landscape')
    ids = np.load(checked(view['object_ids'], hp.parent), allow_pickle=False)
    projection = read(checked(view['projected_plants'], hp.parent))
    require(all(p['pass_index'] in allowed for p in projection['plants']), 'Every native plant must be in the overlay')
    mask = selected_mask(plan, base, handoff, hp, view, ids, allowed, catalog)
    require(np.any(mask) and np.any(~mask), 'Foreground mask must retain a real photo background')
    actual = np.array(Image.open(checked(plan['foreground_mask'], base)).convert('L'))
    require(np.array_equal(actual, mask.astype(np.uint8) * 255), 'Foreground mask differs from native object selection')
    source = checked(plan['plate_source'], base)
    if plan.get('photo_provenance'):
        provenance_path = checked(plan['photo_provenance'], base)
        provenance = read(provenance_path)
        require(provenance.get('original') == plan['plate_source'] and
                provenance.get('working_photo') == view['source_photo'], 'Original photo provenance differs')
        crops = [op['crop_box'] for op in provenance.get('operations', []) if 'crop_box' in op]
        require(len(crops) == 1 and plan.get('source_crop') == crops[0], 'Source crop differs from camera plate provenance')
    else:
        require(plan['plate_source'] == view['source_photo'] and plan.get('source_crop') is None,
                'Alternate photo source/crop requires original-photo provenance')
    plate = plate_pixels(source, view['dimensions'], plan.get('source_crop'))
    actual_plate, alpha = load(checked(plan['photo_plate'], base))
    require(alpha is None and np.array_equal(actual_plate, plate), 'Photo plate is not the documented original-photo preparation')
    return hp, handoff, view, mask, plate


def prepare(handoff_path, cid, catalog_path, fit_path, out, original_provenance=None, validated=False, surface_projection=None):
    hp = Path(handoff_path).resolve(); handoff = read(hp)
    if not validated:
        check_handoff(handoff, hp.parent)
    view = next(v for v in handoff['views'] if v['camera_id'] == cid)
    require(view.get('source_photo'), 'Photo compositing requires the matching source photograph')
    out = Path(out).resolve(); require(not out.exists(), 'Use a new photo plan directory'); out.mkdir(parents=True)
    catalog_path = Path(catalog_path).resolve(); catalog = read(catalog_path)
    allowed = sorted(o['pass_index'] for o in catalog['objects'] if o['type'] == 'MESH' and landscape_kind(o) in LANDSCAPE_KINDS)
    ids = np.load(checked(view['object_ids'], hp.parent), allow_pickle=False)
    source = view['source_photo']; crop = None
    if original_provenance:
        provenance = read(original_provenance); source = provenance['original']
        crop = next(op['crop_box'] for op in provenance['operations'] if 'crop_box' in op)
    save(out / 'photo-plate.png', plate_pixels(checked(source, hp.parent), view['dimensions'], crop))
    plan = {'schema_version': 1, 'kind': 'photo_composite_plan', 'handoff': record(hp),
            'camera_id': cid, 'dimensions': view['dimensions'], 'source_photo': view['source_photo'],
            'plate_source': source, 'source_crop': crop, 'photo_plate': record(out / 'photo-plate.png'),
            'plate_operation': 'EXIF orientation, documented crop, uniform Lanczos resize, RGB8 to RGB16 by 257',
            'camera_fit': record(fit_path), 'object_catalog': record(catalog_path),
            'selected_object_indices': allowed,
            'geometry_operation': 'none; original native camera, positions, dimensions and full canvas retained'}
    if original_provenance:
        plan['photo_provenance'] = record(original_provenance)
    if surface_projection:
        plan['surface_projection'] = record(surface_projection)
    mask = selected_mask(plan, out, handoff, hp, view, ids, allowed, catalog)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / 'foreground-mask.png')
    plan['foreground_mask'] = record(out / 'foreground-mask.png')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    context(plan, out)
    return plan


def compose(plan_path, foreground_path, out):
    pp = Path(plan_path).resolve(); plan = read(pp)
    _, _, _, mask, plate = context(plan, pp.parent)
    foreground, _ = load(foreground_path)
    require(foreground.shape == plate.shape, 'Landscape canvas differs; do not resize/warp the planting')
    foreground = scaled_like(foreground, plate)
    final = np.where(mask[..., None], foreground, plate)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new photo composite directory'); out.mkdir(parents=True)
    save(out / 'candidate.png', final)
    proof = {'schema_version': 1, 'kind': 'original_photo_landscape_composite',
             'plan': record(pp), 'foreground': record(foreground_path), 'candidate': record(out / 'candidate.png'),
             'operation': 'native_landscape_rgb_over_original_photo_v1',
             'photo_background_pixels': int((~mask).sum()), 'landscape_pixels': int(mask.sum())}
    (out / 'composite-proof.json').write_text(json.dumps(proof, indent=2))
    return proof


def verify(proof, base):
    require(proof.get('kind') == 'original_photo_landscape_composite' and
            proof.get('operation') == 'native_landscape_rgb_over_original_photo_v1', 'Photo composite proof required')
    pp = checked(proof['plan'], base); plan = read(pp)
    hp, handoff, view, mask, plate = context(plan, pp.parent)
    fg, _ = load(checked(proof['foreground'], base)); final, alpha = load(checked(proof['candidate'], base))
    require(fg.shape == plate.shape == final.shape and final.dtype == plate.dtype and alpha is None,
            'Photo composite dimensions, precision or opacity differ')
    expected = np.where(mask[..., None], scaled_like(fg, plate), plate)
    require(np.array_equal(expected, final), 'Final differs from original-photo/native-landscape reconstruction')
    require(proof['photo_background_pixels'] == int((~mask).sum()) and proof['landscape_pixels'] == int(mask.sum()),
            'Photo/landscape coverage differs')
    return hp, handoff, view, mask


def check_review(data, base):
    from appearance_guard import check
    require(data.get('method') == 'photo_composite', 'Photo composite completion method required')
    proof_path = checked(data['photo_composite_proof'], base); proof = read(proof_path)
    hp, _, view, mask = verify(proof, proof_path.parent)
    require(checked(data['handoff'], base) == hp and data['camera_id'] == view['camera_id'] and
            data['candidate'] == proof['candidate'], 'Final review/photo/foreground camera bindings differ')
    material_path = checked(data['landscape_review'], base); material = read(material_path)
    require(material.get('schema_version') == 2 and material['camera_id'] == view['camera_id'] and
            material['handoff'] == data['handoff'] and material['candidate'] == proof['foreground'],
            'Photo overlay must use the checked native-camera landscape stage')
    stage = check(material, material_path.parent)
    require(stage['status'] == 'accepted', 'Landscape appearance stage has not passed')
    vision = data.get('photo_review', {})
    require(vision.get('reviewer') == 'vision-capable-agent' and vision.get('source_and_final_loaded') is True and
            vision.get('native_pixel_edges_loaded') is True and vision.get('observations'), 'Actual photo/final and edge inspection required')
    require(all(vision.get('checks', {}).get(k) == 'pass' for k in PHOTO_CHECKS), 'Photo-background integration check failed or absent')
    require(vision.get('paired_evidence'), 'Photo/landscape comparison evidence required')
    for item in vision['paired_evidence']:
        path = checked(item, base)
        try:
            with Image.open(path) as image:
                require(image.width > 1 and image.height > 1, 'Comparison image is empty')
                image.verify()
        except OSError as exc:
            raise ValueError('Paired photo evidence must be readable comparison images') from exc
    return {'status': 'accepted', 'camera_id': view['camera_id'], 'method': 'photo_composite',
            'photo_background_composited': True, 'photo_background_pixels_preserved': int((~mask).sum()),
            'native_landscape_pixels': int(mask.sum()), 'landscape_evidence_counts': stage['evidence_counts'],
            'scope': 'Original photograph with validated landscape overlay; semantic integration is visually reviewed',
            'placement_authority': 'Original validated native handoff; no new synthesized-pixel placement certification'}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan'); a.add_argument('handoff'); a.add_argument('--camera', required=True)
    a.add_argument('--catalog', required=True); a.add_argument('--camera-fit', required=True)
    a.add_argument('--original-provenance'); a.add_argument('--surface-projection'); a.add_argument('--out', required=True)
    a = sub.add_parser('compose'); a.add_argument('plan'); a.add_argument('--foreground', required=True); a.add_argument('--out', required=True)
    a = p.parse_args()
    result = prepare(a.handoff, a.camera, a.catalog, a.camera_fit, a.out, a.original_provenance, surface_projection=a.surface_projection) if a.command == 'plan' else compose(a.plan, a.foreground, a.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
