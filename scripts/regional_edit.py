"""Freeze and stitch native-pixel photo edits; no generated-image resizing."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from handoff import checked, read, record, require
from raster import load, save, scaled_like
from photo_finish import blend


def permission(ref, base, shape):
    with Image.open(checked(ref, base)) as im:
        require(im.mode == 'L' and im.size == shape[::-1], 'Native grayscale permission required')
        return np.array(im)


def prepare(source, mask, box, out, feather=32):
    source = Path(source).resolve(); rgb, alpha = load(source)
    require(alpha is None, 'Use the RGB photo or overlay preview, not a foliage matte')
    h, w = rgb.shape[:2]
    require(len(box) == 4 and all(type(v) is int for v in box) and
            0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h, 'Native crop bounds required')
    require(type(feather) is int and 0 <= feather < min(box[2]-box[0], box[3]-box[1])/2, 'Invalid inward crop feather')
    mr = record(mask); allowed = permission(mr, Path.cwd(), (h, w))
    require(np.any(allowed[box[1]:box[3], box[0]:box[2]]), 'Crop has no editable permission')
    out = Path(out).resolve(); require(not out.exists(), 'Use a new regional plan directory'); out.mkdir(parents=True)
    save(out/'target.png', rgb[box[1]:box[3], box[0]:box[2]])
    plan = {'kind':'native_region_edit_plan', 'source':record(source), 'permission':mr,
            'native_box':box, 'crop_feather_px':feather, 'target':record(out/'target.png')}
    (out/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def reconstruct(entries, base):
    require(entries, 'At least one frozen region and raw generation required')
    result = None; source_ref = None
    for entry in entries:
        pp = checked(entry['plan'], base); plan = read(pp)
        require(plan.get('kind') == 'native_region_edit_plan', 'Frozen native region plan required')
        source = checked(plan['source'], pp.parent)
        if result is None:
            original, source_alpha = load(source)
            require(source_alpha is None, 'RGB source required')
            source_ref = plan['source']; result = scaled_like(original, np.empty(0, np.uint16)).copy()
        require(plan['source'] == source_ref, 'Regions must share the exact same source')
        allowed = permission(plan['permission'], pp.parent, result.shape[:2]); box = plan['native_box']
        x0,y0,x1,y1 = box; h,w = y1-y0,x1-x0
        require(0 <= x0 < x1 <= result.shape[1] and 0 <= y0 < y1 <= result.shape[0], 'Native crop bounds required')
        target, ta = load(checked(plan['target'], pp.parent))
        require(ta is None and np.array_equal(target, original[y0:y1,x0:x1]), 'Frozen target differs from native crop')
        raw, ra = load(checked(entry['raw'], base))
        require(ra is None and raw.shape == (h,w,3), 'Raw generation dimensions/alpha differ; never resize')
        # The final finishing operation applies grayscale transition strength.
        # Regional assembly uses membership only, avoiding a squared blend weight.
        weight = (allowed[y0:y1,x0:x1] > 0).astype(np.uint8) * 255
        feather = plan['crop_feather_px']
        require(type(feather) is int and 0 <= feather < min(h,w)/2, 'Invalid inward crop feather')
        if feather:
            y,x = np.indices((h,w)); edge = np.minimum.reduce([x,y,w-1-x,h-1-y])
            weight = np.rint(weight * np.minimum(edge/feather,1)).astype(np.uint8)
        result[y0:y1,x0:x1] = blend(scaled_like(raw, result), result[y0:y1,x0:x1], weight)
    return result


def stitch(pairs, out):
    entries = [{'plan':record(plan), 'raw':record(raw)} for plan,raw in pairs]
    rgb = reconstruct(entries, Path.cwd())
    out = Path(out).resolve(); require(not out.exists(), 'Use a new stitched output directory'); out.mkdir(parents=True)
    save(out/'raw.png', rgb)
    proof = {'kind':'native_region_stitch_proof', 'operation':'ordered_native_regions_linear_srgb_v1',
             'regions':entries, 'candidate':record(out/'raw.png')}
    (out/'stitch-proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def verify(proof, base):
    require(proof.get('kind') == 'native_region_stitch_proof' and proof.get('operation') ==
            'ordered_native_regions_linear_srgb_v1', 'Regional stitch proof required')
    expected = reconstruct(proof['regions'], base); actual, alpha = load(checked(proof['candidate'], base))
    require(alpha is None and np.array_equal(expected, actual), 'Regional reconstruction differs')
    return {'status':'verified', 'regions':len(proof['regions'])}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('plan'); a.add_argument('--source',required=True); a.add_argument('--permission',required=True)
    a.add_argument('--box',nargs=4,type=int,required=True); a.add_argument('--feather',type=int,default=32); a.add_argument('--out',required=True)
    a=sub.add_parser('stitch'); a.add_argument('--region',nargs=2,action='append',required=True,metavar=('PLAN','RAW')); a.add_argument('--out',required=True)
    a=sub.add_parser('verify'); a.add_argument('proof'); args=p.parse_args()
    if args.command=='plan': result=prepare(args.source,args.permission,args.box,args.out,args.feather)
    elif args.command=='stitch': result=stitch(args.region,args.out)
    else:
        path=Path(args.proof).resolve(); result=verify(read(path),path.parent)
    print(json.dumps(result,indent=2))
