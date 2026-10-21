"""Planned native branch/leaf refinement with species and placement locked."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from handoff import checked, check as check_handoff, read, record, require
from raster import load


def images(refs, base):
    require(refs, 'Readable visual evidence required')
    for ref in refs:
        with Image.open(checked(ref, base)) as image:
            require(image.width > 1 and image.height > 1, 'Empty visual evidence')
            image.verify()


def mapping(rows):
    require(isinstance(rows, list) and rows and all(isinstance(p, dict) and p.get('id') for p in rows),
            'Actual native object/plant rows required')
    result = {p['id']: p for p in rows}
    require(len(result) == len(rows), 'Duplicate native identity')
    return result


def compare(before, after, editable):
    """Only declared plant mesh geometry may differ. Return actual changed IDs."""
    a, b = copy.deepcopy(before['invariants']), copy.deepcopy(after['invariants'])
    original_a, original_b = mapping(before['invariants']['plants']), mapping(after['invariants']['plants'])
    plants_a, plants_b = mapping(a['plants']), mapping(b['plants'])
    require(plants_a.keys() == plants_b.keys() and set(editable) <= plants_a.keys(), 'Plant population or plan IDs changed')
    changed = []
    for name in plants_a:
        pa, pb = plants_a[name], plants_b[name]
        require(pa.get('matrix_world') and pa.get('root_world') is not None and pa.get('root_collar'),
                'Root transform/collar exports required')
        if pa.get('geometry_sha256') != pb.get('geometry_sha256'):
            require(name in editable, 'Undeclared plant geometry changed')
            changed.append(name)
        if name in editable:
            pa.pop('geometry_sha256', None); pb.pop('geometry_sha256', None)
    oa, ob = mapping(a['objects']), mapping(b['objects'])
    require(oa.keys() == ob.keys(), 'Native object membership changed')
    for name in editable:
        require(name in oa and oa[name].get('kind') == ob[name].get('kind') == 'plant', 'Only plant geometry can be refined')
        require(original_a[name]['geometry_sha256'] == oa[name].get('geometry_sha256') and
                original_b[name]['geometry_sha256'] == ob[name].get('geometry_sha256'),
                'Plant/object geometry exports differ')
        oa[name].pop('geometry_sha256', None); ob[name].pop('geometry_sha256', None)
    # Film transparency is a render presentation choice; cameras/lights stay fixed.
    for value in (a, b):
        value['settings'].pop('film_transparent', None)
    require(a == b, 'Species, asset, root/collar, transform, camera, hardscape, opacity or lighting changed')
    return changed


def prepare(handoff_path, before_path, ids, out):
    hp = Path(handoff_path).resolve(); data = read(hp); check_handoff(data, hp.parent)
    bp = Path(before_path).resolve(); before = read(bp)
    require(before.get('exporter') == 'botanical_invariants_v1' and
            before['native_scene']['sha256'] == data['native_scene']['sha256'], 'Validated botanical baseline export required')
    checked(before['native_scene'], bp.parent)
    plants = mapping(before['invariants']['plants'])
    require(ids and len(ids) == len(set(ids)) and set(ids) <= plants.keys(), 'Unique canonical refinement IDs required')
    plan = {'kind': 'botanical_refinement_plan', 'schema_version': 1, 'handoff': record(hp),
            'native_before': record(bp), 'editable_plant_ids': list(ids),
            'permission': 'Branch/leaf mesh and UV refinement only; preserve species, assets, roots, size class and design.'}
    out = Path(out).resolve(); require(not out.exists(), 'Use a new refinement plan'); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def check_review(data, base):
    hp = checked(data['handoff'], base); handoff = read(hp); check_handoff(handoff, hp.parent)
    view = next(v for v in handoff['views'] if v['camera_id'] == data['camera_id'])
    pp = checked(data['refinement_plan'], base); plan = read(pp)
    require(plan.get('kind') == 'botanical_refinement_plan' and plan.get('schema_version') == 1 and
            checked(plan['handoff'], pp.parent) == hp, 'Frozen botanical refinement plan required')
    bp = checked(plan['native_before'], pp.parent); ap = checked(data['native_after'], base)
    before, after = read(bp), read(ap)
    require(before.get('exporter') == after.get('exporter') == 'botanical_invariants_v1' and
            before['native_scene']['sha256'] == handoff['native_scene']['sha256'], 'Actual validated botanical snapshots required')
    checked(before['native_scene'], bp.parent); checked(after['native_scene'], ap.parent)
    editable = plan['editable_plant_ids']
    require(editable and len(editable) == len(set(editable)), 'Unique refinement IDs required')
    changed = compare(before, after, editable)
    ledger = {p['id']: p['species'] for p in read(checked(handoff['plant_ledger'], hp.parent))}
    native = mapping(before['invariants']['plants'])
    projected = read(checked(view['projected_plants'], hp.parent))['plants']
    require(set(native) == set(ledger) and all(native[p['id']]['species'] == ledger[p['id']] and
            native[p['id']]['pass_index'] == p['pass_index'] for p in projected), 'Native species/ID ledger binding differs')
    rp = checked(data['render_proof'], base); proof = read(rp)
    require(proof.get('provenance') == 'renderer_export' and proof.get('camera_id') == view['camera_id'] and
            proof.get('native_scene_sha256') == after['native_scene']['sha256'] and
            proof.get('image_sha256') == data['candidate']['sha256'], 'Actual candidate renderer binding required')
    rgb, _ = load(checked(data['candidate'], base))
    require(list(rgb.shape[1::-1]) == view['dimensions'], 'Candidate camera/canvas differs')
    ids = np.load(checked(proof['object_ids'], rp.parent), allow_pickle=False)
    depth = np.load(checked(proof['depth'], rp.parent), allow_pickle=False)
    require(ids.shape == depth.shape == rgb.shape[:2] and np.issubdtype(ids.dtype, np.integer) and
            np.issubdtype(depth.dtype, np.number), 'Actual candidate ID/depth dimensions differ')
    vision = data.get('recognition', {})
    require(vision.get('reviewer') == 'vision-capable-agent' and vision.get('baseline_and_candidate_loaded') is True and
            vision.get('original_pixel_crops_loaded') is True and vision.get('observations'), 'Actual botanical visual review required')
    images(vision.get('paired_crop_evidence'), base)
    from appearance_guard import GEOMETRY
    extra = ('species_habit', 'branch_taper_attachment', 'leaf_scale_distribution', 'presentation_size', 'root_positions')
    require(all(vision.get('geometry', {}).get(k) == 'pass' for k in GEOMETRY + extra), 'Botanical geometry/placement review failed')
    rows = vision.get('plants', [])
    require(len(rows) == len(ledger) and {p['id'] for p in rows} == set(ledger), 'Account for every canonical plant')
    for row in rows:
        require(row.get('expected_species') == ledger[row['id']] and row.get('identity_source') == 'validated_handoff', 'Species binding changed')
        if row['id'] in changed:
            require(row.get('appearance_assessment') == 'consistent' and row.get('observations'), 'Inspect changed botanical anatomy')
            images([row['measurement_evidence']], base)
    require(vision.get('image_quality') == 'improved' and vision.get('quality_observations'), 'Photographic improvement must be inspected')
    require(data['candidate']['sha256'] != view['image']['sha256'], 'Unchanged baseline is not improved appearance')
    return {'status': 'accepted', 'camera_id': view['camera_id'], 'method': 'native_botanical_refinement',
            'completion_scope': 'landscape_appearance_stage', 'photo_background_composited': False,
            'changed_geometry_ids': changed, 'evidence_counts': {'native_botanical_refinement': len(changed),
            'native_invariants': len(ledger) - len(changed)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('handoff'); p.add_argument('--before', required=True)
    p.add_argument('--plants', nargs='+', required=True); p.add_argument('--out', required=True); args = p.parse_args()
    print(json.dumps(prepare(args.handoff, args.before, args.plants, args.out), indent=2))
