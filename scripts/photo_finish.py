"""Verified photo cleanup, straight-alpha foliage and bounded ground transitions.

Plans are frozen before image generation. Visual findings remain agent judgments;
this helper proves the declared pixel operations and native evidence bindings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from appearance_plan import expand
import bed_ground
from botanical_refinement import images
from handoff import checked, read, record, require
from photo_composite import PHOTO_CHECKS, LANDSCAPE_KINDS, context as photo_context, resolve_surface_overlap, landscape_kind, verify_roles
from raster import load, save, scaled_like

LEGACY_FINISH_CHECKS = PHOTO_CHECKS + ('people_removed', 'construction_clutter_removed',
    'clean_background_repairs', 'natural_plant_anatomy', 'no_cutout_fringes', 'surface_transitions')
SURFACE_CHECKS = ('hardscape_surface_realism', 'natural_leaf_litter')
FINISH_CHECKS = LEGACY_FINISH_CHECKS + SURFACE_CHECKS


def context_ground_occlusion(matte, mp, stage, sp, catalog, ids, depth):
    """Depth-order buried nonplant faces against the unchanged actual context ground."""
    result = np.zeros(ids.shape, bool)
    op = matte.get('context_ground_occlusion')
    if not op: return result
    require(op.get('operation') == 'native_context_ground_depth_v1' and
            op.get('depth_tolerance') == .01 and op.get('depth_convention') == 'ray_distance', 'Bounded native context-depth operation required')
    hp = checked(stage['handoff'], sp.parent); handoff = read(hp)
    view = next(v for v in handoff['views'] if v['camera_id'] == stage['camera_id'])
    require(op.get('baseline_object_ids') == view['object_ids'] and op.get('baseline_depth') == view['depth'],
            'Context depth must use the immutable handoff passes')
    baseline_ids = np.load(checked(op['baseline_object_ids'], hp.parent), allow_pickle=False)
    baseline_depth = np.load(checked(op['baseline_depth'], hp.parent), allow_pickle=False)
    require(baseline_ids.shape == baseline_depth.shape == ids.shape, 'Context depth canvas differs')
    ground = sorted(o['pass_index'] for o in catalog['objects'] if o['type'] == 'MESH' and
                    o.get('kind') in ('existing_ground_context', 'terrain', 'driveway', 'walkway') and
                    landscape_kind(o) not in LANDSCAPE_KINDS)
    require(op.get('occluding_indices') == ground, 'Only actual unselected context ground may occlude buried faces')
    candidates = [o['pass_index'] for o in catalog['objects'] if o['type'] == 'MESH' and
                  landscape_kind(o) in LANDSCAPE_KINDS and o.get('kind') != 'plant']
    result = np.isin(baseline_ids, ground) & np.isin(ids, candidates) & np.isfinite(depth) & np.isfinite(baseline_depth) & (depth > baseline_depth + .01)
    require(op.get('occluded_samples') == int(result.sum()), 'Recompute actual context occlusion count')
    return result


def verify_renderer_rgba(matte, mp, stage, sp, catalog, alpha, straight, ids=None, depth=None):
    """Bind filtered coverage to actual native output; IndexOB includes leaf holes."""
    require(stage['method'] == 'native_botanical_refinement', 'Native botanical stage required for renderer RGBA')
    path = checked(matte['renderer_rgba'], mp.parent)
    require(path == checked(stage['candidate'], sp.parent), 'Renderer RGBA differs from checked native candidate')
    rgb, native_alpha = load(path)
    require(native_alpha is not None and np.array_equal(rgb, straight), 'Renderer straight RGB differs')
    quantized = np.ceil(native_alpha.astype(np.float64) * 255 / np.iinfo(native_alpha.dtype).max).astype(np.uint8)
    if matte.get('context_ground_occlusion'):
        require(ids is not None and depth is not None, 'Actual candidate passes required for context occlusion')
        occluded = context_ground_occlusion(matte, mp, stage, sp, catalog, ids, depth)
    else:
        occluded = np.zeros(alpha.shape, bool)
    quantized[occluded] = 0
    require(np.array_equal(alpha, quantized), 'Renderer alpha coverage was modified')
    after = read(checked(stage['native_after'], sp.parent))
    require(after['invariants']['settings']['film_transparent'] is True, 'Transparent native world required')
    proof = read(checked(stage['render_proof'], sp.parent))
    excluded = {o['name'] for o in catalog['objects'] if o['type'] == 'MESH' and landscape_kind(o) not in LANDSCAPE_KINDS}
    overrides = proof.get('camera_only_export_overrides', [])
    require(len(overrides) == len(excluded) and {o['name'] for o in overrides} == excluded and
            all(o.get('visible_camera') is False or o.get('holdout') is True for o in overrides),
            'Log every excluded context mesh camera override')
    return occluded


def gray(ref, base, shape):
    with Image.open(checked(ref, base)) as image:
        require(image.mode == 'L', 'Use 8-bit grayscale permission/matte PNGs')
        result = np.array(image)
    require(result.shape == shape, 'Permission/matte camera dimensions differ')
    return result


def _blend_block(fg, bg, alpha):
    """Straight (unassociated) sRGB, blended in linear light; exact endpoints."""
    def linear(rgb):
        c = rgb.astype(np.float64) / 65535
        return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    a = alpha[..., None].astype(np.float64) / 255
    c = linear(fg) * a + linear(bg) * (1 - a)
    srgb = np.where(c <= .0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - .055)
    out = np.rint(np.clip(srgb, 0, 1) * 65535).astype(np.uint16)
    out[alpha == 0] = bg[alpha == 0]; out[alpha == 255] = fg[alpha == 255]
    return out


def blend(fg, bg, alpha):
    # Identical per-sample arithmetic, bounded temporary memory at native size.
    out = np.empty(fg.shape, dtype=np.uint16)
    for y in range(0, len(fg), 128):
        out[y:y+128] = _blend_block(fg[y:y+128], bg[y:y+128], alpha[y:y+128])
    return out


def context(plan, base):
    from appearance_guard import check
    require(plan.get('kind') == 'photo_finish_plan' and plan.get('schema_version') == 1, 'Frozen finishing plan required')
    pp = checked(plan['photo_plan'], base); photo_plan = read(pp)
    hp, handoff, view, _, plate = photo_context(photo_plan, pp.parent)
    sp = checked(plan['landscape_review'], base); stage = read(sp)
    require(stage.get('schema_version') in (2, 4) and stage.get('method') in
            ('image_edit', 'native_material_only', 'native_botanical_refinement') and
            checked(stage['handoff'], sp.parent) == hp and stage['camera_id'] == view['camera_id'],
            'Matching checked landscape appearance stage required')
    result = check(stage, sp.parent); require(result['status'] == 'accepted', 'Landscape stage has not passed')
    foreground, _ = load(checked(stage['candidate'], sp.parent)); foreground = scaled_like(foreground, plate)
    require(foreground.shape == plate.shape, 'Landscape/photo canvas differs')
    scene = handoff['native_scene']; native_ids_ref = view['object_ids']; native_ids_base = hp.parent
    if stage['method'] != 'image_edit':
        ap = checked(stage['native_after'], sp.parent); scene = read(ap)['native_scene']
        rp = checked(stage['render_proof'], sp.parent); native_ids_ref = read(rp)['object_ids']; native_ids_base = rp.parent
    mp = checked(plan['renderer_matte'], base); matte = read(mp)
    require(matte.get('provenance') == 'renderer_landscape_matte' and matte.get('camera_id') == view['camera_id'] and
            matte.get('native_scene_sha256') == scene['sha256'] and matte.get('rgb_encoding') == 'straight_srgb' and
            matte.get('appearance_image_sha256') == stage['candidate']['sha256'], 'Actual matching straight-alpha renderer matte required')
    checked(scene, ap.parent if stage['method'] != 'image_edit' else hp.parent)
    cp = checked(matte['object_catalog'], mp.parent); catalog = read(cp)
    verify_roles(catalog, cp.parent)
    require(catalog.get('provenance') == 'renderer_object_catalog' and catalog.get('native_scene', {}).get('sha256') == scene['sha256'],
            'Matte catalog belongs to another native scene')
    checked(catalog['native_scene'], cp.parent)
    objects = [o for o in catalog['objects'] if o['type'] == 'MESH']
    indices = [o['pass_index'] for o in objects]
    require(len(indices) == len(set(indices)) and all(type(i) is int and i > 0 for i in indices), 'Unique renderer mesh indices required')
    # A different catalog must not quietly turn a building into landscape.
    old_cp = checked(photo_plan['object_catalog'], pp.parent); old = read(old_cp)
    require(sorted(objects, key=lambda x:x['name']) == sorted([o for o in old['objects'] if o['type'] == 'MESH'], key=lambda x:x['name']),
            'Renderer catalog membership/kinds/indices changed')
    ids = np.load(checked(matte['object_ids'], mp.parent), allow_pickle=False)
    expected_ids = np.load(checked(native_ids_ref, native_ids_base), allow_pickle=False)
    depth = np.load(checked(matte['depth'], mp.parent), allow_pickle=False)
    require(ids.shape == depth.shape == plate.shape[:2] and np.issubdtype(ids.dtype, np.integer) and
            np.array_equal(ids, expected_ids) and np.issubdtype(depth.dtype, np.number), 'Matte must use actual candidate renderer IDs/depth')
    require(set(np.unique(ids)) <= {0, *indices}, 'Unknown native renderer object index')
    if stage['method'] == 'native_botanical_refinement':
        require(matte['depth']['sha256'] == read(rp)['depth']['sha256'], 'Matte depth differs from candidate render')
    else:
        require(matte['depth']['sha256'] == view['depth']['sha256'], 'Unchanged geometry matte depth differs')
    selected = [o['pass_index'] for o in objects if landscape_kind(o) in LANDSCAPE_KINDS]
    plant_indices = [o['pass_index'] for o in objects if o.get('kind') == 'plant']
    mask = np.isin(ids, selected); plants = np.isin(ids, plant_indices)
    if matte.get('surface_projection'):
        xp = checked(matte['surface_projection'], mp.parent); projection = read(xp)
        require(projection.get('provenance') == 'renderer_surface_projection' and
                projection.get('native_scene', {}).get('sha256') == scene['sha256'], 'Candidate surface projection scene differs')
        match = next(v for v in projection['views'] if v['camera_id'] == view['camera_id'])
        require(match['dimensions'] == view['dimensions'], 'Projection camera differs')
        mask |= resolve_surface_overlap(ids, depth, match, selected, catalog)
    require(np.any(mask) and np.any(~mask), 'Real photographic background required')
    alpha = gray(matte['alpha'], mp.parent, mask.shape)
    core = ~expand(~mask, 2)
    straight, _ = load(checked(matte['straight_rgb'], mp.parent)); require(straight.shape == plate.shape, 'Straight RGB canvas differs')
    if matte.get('renderer_rgba'):
        occluded = verify_renderer_rgba(matte, mp, stage, sp, catalog, alpha, straight, ids, depth)
    else:
        occluded = np.zeros(mask.shape, bool)
        require(not np.any((alpha > 0) & ~expand(mask, 2)) and np.all(alpha[core] == 255), 'Matte exceeds native edge band or erodes opaque interiors')
        require(np.all(alpha[plants] > 0), 'Matte erased visible plant fragments')
    straight = scaled_like(straight, plate)
    # Checked appearance owns every opaque interior; renderer RGB removes old
    # background contamination only in the two-pixel antialiasing boundary band.
    foreground[~core] = straight[~core]
    cleanup = np.zeros(mask.shape, bool)
    regions = plan.get('cleanup_regions', [])
    if plan.get('cleanup_inventory'):
        ip = checked(plan['cleanup_inventory'], base); inventory = read(ip)
        expected = inventory['regions']
        for region in expected: region['mask'] = record(checked(region['mask'], ip.parent))
        require(expected == regions, 'Cleanup inventory permissions changed')
    require(len({r['id'] for r in regions}) == len(regions), 'Duplicate cleanup region')
    for region in regions:
        require(region.get('kind') in ('person', 'tool', 'debris', 'equipment', 'temporary_staging') and region.get('description'), 'Describe temporary cleanup target')
        region_mask = gray(region['mask'], base, mask.shape)
        require(set(np.unique(region_mask)) <= {0, 255} and np.any(region_mask) and np.any(region_mask == 0), 'Localized binary cleanup masks required')
        cleanup |= region_mask > 0
    require(not np.all(cleanup), 'Full-frame photo reconstruction is forbidden')
    transition = gray(plan['transition_alpha'], base, mask.shape)
    # Native ground interiors and a narrow local contact band. Plant pixels and
    # their three-pixel margin are never painted over by the ground repair pass.
    ground_kinds = {'bed', 'walk_slab', 'porch_slab', 'joint_substrate', 'sidewalk',
                    'stepping_stone', 'step', 'paver_crossing', 'paver_band', 'nonliving_surface_detail'}
    ground = np.isin(ids, [o['pass_index'] for o in objects if landscape_kind(o) in ground_kinds])
    ground |= mask & ~np.isin(ids, selected)  # Bounded coincident-ground recovery.
    nonplants = mask & ~plants
    eligible = (ground | (expand(nonplants, 12) & ~mask)) & ~expand(plants, 3) & ~occluded
    require(not np.any((transition > 0) & ~eligible), 'Transition edit reaches protected plant/background regions')
    bed_on = bed_ground.check_spec(plan.get('bed_ground'), base)
    layers = {'ids': ids, 'depth': depth, 'plants': plants, 'occluded': occluded, 'bed_on': bed_on,
              'bed': np.isin(ids, [o['pass_index'] for o in objects if landscape_kind(o) == 'bed']),
              'manifest': read(checked(view['renderer_manifest'], hp.parent)) if view.get('renderer_manifest') else None}
    return hp, view, plate, foreground, alpha, cleanup, transition, result, layers


def prepare(photo_plan, stage, matte, cleanup_inventory, transition_alpha, out, bed=True, soil_image=None, soil_box=None):
    inventory_path = Path(cleanup_inventory).resolve(); inventory = read(inventory_path)
    regions = inventory['regions']
    for region in regions:
        region['mask'] = record(checked(region['mask'], inventory_path.parent))
    plan = {'schema_version': 1, 'kind': 'photo_finish_plan', 'photo_plan': record(photo_plan),
            'landscape_review': record(stage), 'renderer_matte': record(matte),
            'cleanup_regions': regions, 'cleanup_inventory': record(inventory_path), 'transition_alpha': record(transition_alpha),
            'surface_realism_review': True,
            'bed_ground': bed_ground.plan_spec(bed, soil_image, soil_box),
            'operation': 'cleanup_then_straight_srgb_linear_overlay_then_bounded_ground_blend_v1'}
    context(plan, Path.cwd())
    out = Path(out).resolve(); require(not out.exists(), 'Use a new pre-generation finishing plan')
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def cleaned_plate(plate, cleanup, cleanup_raw, base):
    if np.any(cleanup):
        require(cleanup_raw is not None, 'Raw photo cleanup edit required')
        raw, _ = load(checked(cleanup_raw, base)); require(raw.shape == plate.shape, 'Raw cleanup canvas differs')
        plate = np.where(cleanup[..., None], scaled_like(raw, plate), plate)
    return plate


def reconstruct(plan, base, cleanup_raw=None, transition_raw=None, *, preview=False):
    ctx = context(plan, base); _, _, plate, fg, alpha, cleanup, transition, _, layers = ctx
    plate = cleaned_plate(plate, cleanup, cleanup_raw, base)
    final = blend(fg, plate, alpha)
    if np.any(transition) and not preview:
        require(transition_raw is not None, 'Raw transition edit required')
        raw, _ = load(checked(transition_raw, base)); require(raw.shape == final.shape, 'Raw transition canvas differs')
        final = blend(scaled_like(raw, final), final, transition)
    if layers['bed_on'] and not preview:
        # Open bed ground the other permissions don't own: opaque landscape, no cleanup/transition/occlusion.
        area = layers['bed'] & (alpha == 255) & ~cleanup & (transition == 0) & ~layers['occluded']
        final, layers['bed_ground_info'] = bed_ground.apply(final, layers['ids'], layers['depth'], layers['plants'],
                                                            area, layers['manifest'], plan['bed_ground'], base)
    return final, ctx


def preview(plan_path, out, cleanup_raw=None):
    pp = Path(plan_path).resolve(); raw = record(cleanup_raw) if cleanup_raw else None
    image, ctx = reconstruct(read(pp), pp.parent, raw, preview=True)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new preview directory'); out.mkdir(parents=True)
    save(out / 'overlay-preview.png', image)
    save(out / 'cleaned-photo.png', cleaned_plate(ctx[2], ctx[5], raw, pp.parent))
    return {'status': 'preview_only', 'image': record(out / 'overlay-preview.png'), 'cleaned_photo': record(out / 'cleaned-photo.png')}


def compose(plan_path, out, cleanup_raw=None, transition_raw=None):
    pp = Path(plan_path).resolve(); plan = read(pp)
    proof = {'kind': 'photo_finish_proof', 'operation': 'cleanup_then_straight_srgb_linear_overlay_then_bounded_ground_blend_v1',
             'plan': record(pp)}
    for key, path in [('cleanup_raw', cleanup_raw), ('transition_raw', transition_raw)]:
        if path: proof[key] = record(path)
    final, ctx = reconstruct(plan, pp.parent, proof.get('cleanup_raw'), proof.get('transition_raw'))
    if 'bed_ground_info' in ctx[8]: proof['bed_ground'] = ctx[8]['bed_ground_info']
    out = Path(out).resolve(); require(not out.exists(), 'Use a new final directory'); out.mkdir(parents=True)
    save(out / 'candidate.png', final); proof['candidate'] = record(out / 'candidate.png')
    save(out / 'cleaned-photo.png', cleaned_plate(ctx[2], ctx[5], proof.get('cleanup_raw'), pp.parent))
    proof['cleaned_photo'] = record(out / 'cleaned-photo.png')
    (out / 'finish-proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def verify(proof, base):
    require(proof.get('kind') == 'photo_finish_proof' and proof.get('operation') ==
            'cleanup_then_straight_srgb_linear_overlay_then_bounded_ground_blend_v1', 'Finishing operation proof required')
    pp = checked(proof['plan'], base); plan = read(pp)
    cleanup_raw = record(checked(proof['cleanup_raw'], base)) if proof.get('cleanup_raw') else None
    transition_raw = record(checked(proof['transition_raw'], base)) if proof.get('transition_raw') else None
    expected, ctx = reconstruct(plan, pp.parent, cleanup_raw, transition_raw)
    final, alpha = load(checked(proof['candidate'], base))
    require(alpha is None and final.dtype == np.uint16 and np.array_equal(final, expected), 'Final photo finishing reconstruction differs')
    require(proof.get('bed_ground') == ctx[8].get('bed_ground_info'), 'Bed ground record differs from its reconstruction')
    cleaned, cleaned_alpha = load(checked(proof['cleaned_photo'], base))
    require(cleaned_alpha is None and np.array_equal(cleaned, cleaned_plate(ctx[2], ctx[5], cleanup_raw, pp.parent)),
            'Cleaned photo reconstruction differs')
    return plan, ctx


def check_review(data, base):
    return review_context(data, base)[0]


def review_context(data, base):
    """Verify a schema 4 photo_finish review; also return its plan, reconstruction context and proof."""
    fp = checked(data['finish_proof'], base); proof = read(fp)
    plan, ctx = verify(proof, fp.parent); hp, view, _, _, alpha, cleanup, transition, stage, layers = ctx
    require(checked(data['handoff'], base) == hp and data['camera_id'] == view['camera_id'] and
            checked(data['candidate'], base) == checked(proof['candidate'], fp.parent), 'Final review camera/candidate differs')
    vision = data.get('photo_review', {})
    require(vision.get('reviewer') == 'vision-capable-agent' and vision.get('source_and_final_loaded') is True and
            vision.get('native_pixel_edges_loaded') is True and vision.get('observations'), 'Actual source/final/native edge review required')
    checks = FINISH_CHECKS if plan.get('surface_realism_review') else LEGACY_FINISH_CHECKS
    outcomes = vision.get('checks', {})
    nonpassing = [k for k in checks if outcomes.get(k) != 'pass']
    if data.get('release_policy') == 'complete_with_limitations':
        require(all(outcomes.get(k) in ('pass', 'fail', 'estimated') for k in checks), 'Every finishing check needs an actual disposition')
        limitations = data.get('limitations', [])
        require({r.get('check') for r in limitations} == set(nonpassing) and
                len(limitations) == len(nonpassing) and all(r.get('observations') for r in limitations),
                'Describe each nonpassing finishing check without relabeling it passed')
    else:
        require(not nonpassing, 'Cleanup, anatomy or photo integration review failed')
    images(vision.get('paired_evidence'), base)
    rows = vision.get('cleanup', [])
    require(len(rows) == len(plan['cleanup_regions']) and {r['id'] for r in rows} == {r['id'] for r in plan['cleanup_regions']}, 'Review every cleanup target')
    for row in rows:
        require(row.get('observations'), 'Cleanup target needs actual observations')
        if row.get('status') != 'removed':
            target = next(t for t in plan['cleanup_regions'] if t['id'] == row['id'])
            relevant = 'people_removed' if target['kind'] == 'person' else 'construction_clutter_removed'
            require(data.get('release_policy') == 'complete_with_limitations' and row.get('status') in ('retained', 'partial') and relevant in nonpassing,
                    'Unresolved cleanup target must be disclosed under the matching nonpassing check')
        images([row['paired_evidence']], base)
    return {'status': 'completed_with_limitations' if nonpassing else 'accepted', 'nonpassing_checks': nonpassing, 'limitations': data.get('limitations', []), 'camera_id': view['camera_id'], 'method': 'photo_finish', 'photo_background_composited': True,
            'cleanup_regions': len(rows), 'unresolved_cleanup_targets': [r['id'] for r in rows if r.get('status') != 'removed'], 'cleanup_pixels': int(cleanup.sum()), 'transition_pixels': int((transition > 0).sum()),
            'photo_background_pixels_preserved': int(((alpha == 0) & ~cleanup & (transition == 0)).sum()),
            'bed_ground': layers.get('bed_ground_info', {'status': 'off'}),
            'landscape_evidence_counts': stage['evidence_counts'], 'scope': 'Cleaned original photo with checked landscape and reviewed integration',
            'upstream_handoff_kind': read(hp)['kind'], 'upstream_certification': read(hp).get('certification', 'inherited_strict_validation'), 'upstream_production_status': read(hp).get('production_status'), 'inherited_assumptions': read(hp).get('assumptions', []),
            'limitation': 'Native evidence and reproducible operations do not independently establish semantic realism; finishing never upgrades upstream source/camera certification'}, plan, ctx, proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan'); a.add_argument('--photo-plan', required=True); a.add_argument('--landscape-review', required=True)
    a.add_argument('--renderer-matte', required=True); a.add_argument('--cleanup-inventory', required=True)
    a.add_argument('--transition-alpha', required=True); a.add_argument('--out', required=True)
    a.add_argument('--no-bed-ground', action='store_true', help='Leave open bed ground as rendered (no mulch grain)')
    a.add_argument('--soil-reference', help='Site photo showing bare bed soil; its colour guides the mulch')
    a.add_argument('--soil-box', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='Bare-soil box in that photo')
    a = sub.add_parser('preview'); a.add_argument('plan'); a.add_argument('--out', required=True); a.add_argument('--cleanup-raw')
    a = sub.add_parser('compose'); a.add_argument('plan'); a.add_argument('--out', required=True)
    a.add_argument('--cleanup-raw'); a.add_argument('--transition-raw'); a = p.parse_args()
    if a.command == 'plan':
        result = prepare(a.photo_plan, a.landscape_review, a.renderer_matte, a.cleanup_inventory, a.transition_alpha, a.out,
                         not a.no_bed_ground, a.soil_reference, a.soil_box)
    elif a.command == 'preview':
        result = preview(a.plan, a.out, a.cleanup_raw)
    else:
        result = compose(a.plan, a.out, a.cleanup_raw, a.transition_raw)
    print(json.dumps(result, indent=2))
