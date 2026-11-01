"""Compose the built-in image tool's full-view scene edit back into the finished photo, and check its review.

Plan schema 5, operation scene_edit_full_view_v1: one image-tool edit of the whole previous photo_finish composite
(reduced to the tool's size, followed by the view's numbered plant map and its labelled reference sheet when the plan
has them). At most two attempts: the second edits the original input again with the retry prompt built from the
first attempt's review notes, and the attempt with fewer real-failure concerns, then fewer defects, is selected. The
selected output is resized to the view (Lanczos; the aspect must match within 3%, nothing is warped) and replaces the
previous composite inside the landscape edit region through the plan's raised-cosine inward feather, in linear
light. Outside the region every sample stays exactly the previous composite, which there is exactly the cleaned
original photo; both are asserted. There is no alignment rejection and no partial (tiled) result: the whole
landscape comes from the one edit, and the pass always delivers it. The review records the four real-failure
observations and the defects without blocking; a method is adopted by comparing whole runs against the baseline
side by side, not by failing single renders. `compare.jpg` (previous composite | scene edit, labelled) is the
full-view side-by-side for that comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handoff import checked, read, record, require
from raster import load, save, scaled_like
# The four real failures (REAL_FAILURE_CHECKS) are observed and recorded on the whole delivered view; with the defects
# they never block delivery.
from scene_edit_plan import (CHECK_SET, IMAGE_TOOLS, JOB_ID, MAX_ATTEMPTS, MODEL, REAL_FAILURE_CHECKS,
                             best_attempt, check_notes, region_weight, resized, retry_items, retry_prompt_text, score,
                             sheet_font, to8, validate)

OPERATION = 'scene_edit_full_view_v1'
REVIEW_SCHEMA = 5
COMPARE_WIDTH = 2400
COMPARE_LABELS = ('Before: previous photo composite', 'After: scene photographic edit')
BLOCK = 256

_SRGB8 = np.arange(256, dtype=np.float64) / 255
_LIN8 = np.where(_SRGB8 <= .04045, _SRGB8 / 12.92, ((_SRGB8 + .055) / 1.055) ** 2.4).astype(np.float32)
_SRGB16 = np.arange(65536, dtype=np.float64) / 65535
_LIN16 = np.where(_SRGB16 <= .04045, _SRGB16 / 12.92, ((_SRGB16 + .055) / 1.055) ** 2.4).astype(np.float32)


def linear(rgb):
    return (_LIN8 if rgb.dtype == np.uint8 else _LIN16)[rgb]


def encode16(lin):
    c = np.clip(lin.astype(np.float64), 0, 1)
    return np.rint(np.where(c <= .0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - .055) * 65535).astype(np.uint16)


def text(ref, base):
    return checked(ref, base).read_text(encoding='utf-8')


def normalize(manifest_path):
    """Hash the agent's attempt manifest (plain relative paths) into proof records."""
    mp = Path(manifest_path).resolve(); m = read(mp)
    jobs = []
    for job_id, job in m.get('jobs', {}).items():
        attempts = []
        for n, a in enumerate(job.get('attempts', []), 1):
            row = {'attempt': n, 'raw': record(mp.parent / a['raw']), 'prompt': record(mp.parent / a['prompt']),
                   'inspection': a.get('inspection', '')}
            for key in ('labels', 'reference', 'input', 'notes'):
                if a.get(key):
                    row[key] = record(mp.parent / a[key])
            attempts.append(row)
        jobs.append({'id': job_id, 'attempts': attempts, 'selected': job.get('selected')})
    return {'agent_model': m.get('agent_model'), 'reasoning_effort': m.get('reasoning_effort'),
            'image_tool': m.get('image_tool'), 'manifest': record(mp), 'jobs': jobs}


def check_attempts(plan, attempts, base):
    """Model, attempt count, exact prompts, the input, label map and reference bindings, the returned aspect, the
    defect-driven second attempt and the selection. Returns the job row with `scores` (concerns and defects per
    attempt that recorded review notes)."""
    require(attempts.get('agent_model') == MODEL['agent_model'] and attempts.get('reasoning_effort') == MODEL['reasoning_effort'],
            'Scene edits must come from gpt-6-astra with high reasoning')
    require(attempts.get('image_tool') in IMAGE_TOOLS, 'Scene edits must use the built-in image tool')
    require([j['id'] for j in attempts['jobs']] == [JOB_ID], f'Record the one planned full-view job ({JOB_ID})')
    job = plan['jobs'][0]; row = attempts['jobs'][0]; tries = row['attempts']
    require(1 <= len(tries) <= MAX_ATTEMPTS, f'Record 1-{MAX_ATTEMPTS} attempts of the full-view edit')
    require([a['attempt'] for a in tries] == list(range(1, len(tries) + 1)), 'Attempts are numbered in order')
    require(row['selected'] in range(1, len(tries) + 1), 'Select the attempt to deliver; the full-view pass always delivers one')
    planned = text(job['prompt'], base)
    require(text(tries[0]['prompt'], base) == planned, 'Attempt 1 must use the exact planned prompt')
    notes = {a['attempt']: read(checked(a['notes'], base)) for a in tries if a.get('notes')}
    scores = [{'attempt': n, **score(v)} for n, v in sorted(notes.items())]
    if len(tries) == 2:
        require(set(notes) == {1, 2}, 'With two attempts, record the review notes of both (notes)')
        records = read(checked(plan['records'], base))
        want = retry_prompt_text(planned, retry_items(notes[1], job['inventory'], records))
        require(text(tries[1]['prompt'], base) == want,
                "Attempt 2 uses the retry prompt built from attempt 1's review notes (scene_edit_plan.py retry_prompt)")
        require(row['selected'] == best_attempt(scores),
                'Select the attempt with fewer real-failure concerns, then fewer defects (a tie keeps attempt 1)')
    for name, key, path in (('label map', 'labels', 'labels.png'), ('reference sheet', 'reference', 'references.jpg')):
        want = job.get('labels') if key == 'labels' else job['references'].get('sheet')
        for a in tries:
            if want:
                require(a.get(key) and checked(a[key], base) and a[key]['sha256'] == want['sha256'],
                        f"Attempt {a['attempt']} must attach the planned {name} (jobs/{JOB_ID}/{path})")
            else:
                require(not a.get(key), f'The view has no planned {name}; attach none')
    for a in tries:
        if a.get('input'):  # attempt 2 edits the original input again, never a previous output
            got, _ = load(checked(a['input'], base)); want, _ = load(checked(job['input'], base))
            require(np.array_equal(got, want), 'The attempt input is not the planned full-view input')
        resized(checked(a['raw'], base), job['box'])
    return {**row, 'scores': scores}


def reconstruct(plan, region, composite, row, base):
    """The selected edit, resized to the view, replaces the composite inside the feathered region, in linear light."""
    job = plan['jobs'][0]
    raw = checked(row['attempts'][row['selected'] - 1]['raw'], base)
    edit = resized(raw, job['box'])
    weight = region_weight(region, plan['params']['feather_px'])
    h, w = composite.shape[:2]
    final = composite.copy()
    for b0 in range(0, h, BLOCK):
        b1 = min(h, b0 + BLOCK)
        wr = weight[b0:b1]
        if not np.any(wr > 0):
            continue
        mix = wr[..., None] * linear(edit[b0:b1]) + (1 - wr[..., None]) * linear(composite[b0:b1])
        block = encode16(mix)
        if composite.dtype != np.uint16:
            block = scaled_like(block, composite)
        keep = wr == 0
        block[keep] = composite[b0:b1][keep]
        final[b0:b1] = block
    require(np.array_equal(final[~region], composite[~region]), 'Scene edit changed pixels outside the landscape edit region')
    with Image.open(raw) as im:
        raw_size = [im.width, im.height]
    output = {'raw_size': raw_size, 'delivered_size': [w, h], 'upscale': round(w / raw_size[0], 4)}
    return final, weight, output


def outside_is_cleaned_photo(region, composite, cleaned):
    require(np.array_equal(composite[~region], scaled_like(cleaned, composite)[~region]),
            'Previous composite differs from the cleaned photo outside the edit region')


def side_by_side(left, right, labels=COMPARE_LABELS, width=COMPARE_WIDTH):
    """Full-view comparison: left | right, each half the width, labelled above. Returns a PIL image."""
    left, right = to8(left), to8(right)
    require(left.shape == right.shape, 'Side-by-side images must be the same size')
    h, w = left.shape[:2]
    half = max(1, width // 2); ph = max(1, round(h * half / w))
    size = max(14, half // 40); bar = int(size * 1.8)
    out = Image.new('RGB', (2 * half, ph + bar), (255, 255, 255))
    draw = ImageDraw.Draw(out); font = sheet_font(size)
    for k, (image, label) in enumerate(zip((left, right), labels)):
        out.paste(Image.fromarray(image).resize((half, ph), Image.LANCZOS), (k * half, bar))
        draw.text((k * half + size // 2, (bar - size) // 2), label, fill=(0, 0, 0), font=font)
    return out


def compose(plan_path, manifest, out):
    pp = Path(plan_path).resolve(); plan = read(pp)
    plan, region, composite, cleaned, _ = validate(plan, pp.parent)
    outside_is_cleaned_photo(region, composite, cleaned)
    attempts = normalize(manifest)
    row = check_attempts(plan, attempts, pp.parent)
    final, weight, output = reconstruct(plan, region, composite, row, pp.parent)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new scene edit output directory'); out.mkdir(parents=True)
    save(out / 'candidate.png', final)
    Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(out / 'region-weight.png')
    side_by_side(composite, final).save(out / 'compare.jpg', quality=90)
    proof = {'kind': 'scene_edit_proof', 'operation': OPERATION, 'plan': record(pp), 'attempts': attempts,
             'candidate': record(out / 'candidate.png'), 'region_weight': record(out / 'region-weight.png'),
             'compare': record(out / 'compare.jpg'), 'output': output, 'region_pixels': int(region.sum()),
             'outside_region_pixels_preserved': int((~region).sum())}
    (out / 'scene-edit-proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def verify(proof, base):
    require(proof.get('kind') == 'scene_edit_proof', 'Scene edit proof required')
    require(proof.get('operation') == OPERATION,
            f'Only {OPERATION} scene edit proofs are supported; the tiled operations were removed')
    pp = checked(proof['plan'], base)
    plan, region, composite, cleaned, baseline = validate(read(pp), pp.parent)
    outside_is_cleaned_photo(region, composite, cleaned)
    attempts = proof['attempts']
    checked(attempts['manifest'], base)
    row = check_attempts(plan, attempts, pp.parent)
    final, weight, output = reconstruct(plan, region, composite, row, pp.parent)
    actual, alpha = load(checked(proof['candidate'], base))
    require(alpha is None and np.array_equal(actual, final), 'Scene edit reconstruction differs')
    with Image.open(checked(proof['region_weight'], base)) as im:
        require(np.array_equal(np.array(im), np.rint(weight * 255).astype(np.uint8)), 'Scene edit region_weight differs')
    checked(proof['compare'], base)
    require(proof.get('output') == output, 'Recompute the output and delivered resolution')
    require(proof.get('outside_region_pixels_preserved') == int((~region).sum()), 'Recompute preserved pixel count')
    return plan, row, baseline


def observations(vision):
    """The four real-failure observations ('pass', or a concern naming where and what) and the defect list."""
    concerns, defects = check_notes(vision)
    return [{'check': c['check'], 'observations': f"{c['where']}: {c['observation']}", 'blocking': False}
            for c in concerns], defects


def defect_rows(defects):
    return [{'check': f"defect: {d['type']}", 'observations': f"{d['where']}: {d['observation']}", 'blocking': False}
            for d in defects]


def check_review(data, base):
    """Schema 5 scene_photographic_edit review. The full-view edit is always delivered; real-failure concerns and
    defects are recorded (status completed_with_limitations), never a reason to fall back. Only integrity failures
    (missing or changed evidence, wrong model, an attempt rule broken) raise."""
    require(data.get('schema_version') == REVIEW_SCHEMA,
            'Only schema 5 full-view scene edit reviews are supported; the tiled schema 4 route was removed')
    require('release_policy' not in data, 'The full-view scene edit has no fallback release policy; it is always delivered')
    brp = checked(data['baseline_review'], base); previous = read(brp)
    require(previous.get('schema_version') == 4 and previous.get('method') == 'photo_finish', 'Previous composite needs its schema 4 photo_finish review')
    require(checked(data['handoff'], base) == checked(previous['handoff'], brp.parent) and data['camera_id'] == previous['camera_id'],
            'Scene edit review camera/handoff differs from the previous composite')
    sp = checked(data['scene_edit_proof'], base); proof = read(sp)
    plan, row, baseline = verify(proof, sp.parent)
    require(checked(plan['review'], sp.parent) == brp, 'Scene edit plan was made from a different previous composite')
    require(checked(data['candidate'], base) == checked(proof['candidate'], sp.parent), 'Scene edit review candidate differs from its proof')
    model = data.get('model', {})
    require(model.get('agent_model') == proof['attempts']['agent_model'] == MODEL['agent_model'] and
            model.get('reasoning_effort') == proof['attempts']['reasoning_effort'] == MODEL['reasoning_effort'] and
            model.get('image_tool') == proof['attempts']['image_tool'], 'Record the gpt-6-astra/high model and image tool used')
    vision = data.get('scene_review', {})
    require(vision.get('reviewer') == 'vision-capable-agent' and vision.get('source_and_final_loaded') is True and
            vision.get('native_size_crops_loaded') is True and vision.get('observations'),
            'Actual scene edit review of source, final and native-size crops required')
    require(vision.get('attempts') == len(row['attempts']) and vision.get('selected_attempt') == row['selected'],
            'The review must match the recorded attempts and selection')
    from botanical_refinement import images
    images(vision.get('paired_evidence'), base)
    concerns, defects = observations(vision)
    observed = concerns + defect_rows(defects)
    limited = bool(observed) or baseline['status'] == 'completed_with_limitations'
    return {'status': 'completed_with_limitations' if limited else 'accepted', 'method': 'scene_photographic_edit',
            'camera_id': data['camera_id'], 'delivered': 'scene_edit_composite', 'scene_edit_status': 'delivered',
            'limitations': baseline.get('limitations', []) + observed,
            'previous_composite_status': baseline['status'], 'previous_composite_limitations': baseline.get('limitations', []),
            'model': data.get('model'), 'attempts': len(row['attempts']), 'selected': row['selected'],
            'attempt_scores': row['scores'],
            'check_set': CHECK_SET,
            'real_failure_observations': {k: 'pass' if vision['checks'][k] == 'pass' else 'concern' for k in REAL_FAILURE_CHECKS},
            'defects': defects, 'defect_count': len(defects), 'non_blocking_observations': observed,
            'output': proof['output'], 'compare': proof['compare'],
            'photo_background_composited': True, 'bed_ground': baseline.get('bed_ground'),
            'landscape_evidence_counts': baseline.get('landscape_evidence_counts'),
            'upstream_handoff_kind': baseline.get('upstream_handoff_kind'), 'upstream_certification': baseline.get('upstream_certification'),
            'scope': 'Built-in image-tool photographic edit of the whole landscape region of a checked photo composite',
            'limitation': 'Generated plant pixels are agent-reviewed derivatives; the native scene stays the placement and species authority'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('compose'); a.add_argument('plan'); a.add_argument('--attempts', required=True, help='Attempt manifest JSON')
    a.add_argument('--out', required=True)
    a = sub.add_parser('verify'); a.add_argument('proof')
    a = sub.add_parser('side_by_side', help='Full-view comparison of two same-size images (left | right), labelled')
    a.add_argument('left'); a.add_argument('right'); a.add_argument('--out', required=True)
    a.add_argument('--labels', nargs=2, default=list(COMPARE_LABELS)); a.add_argument('--width', type=int, default=COMPARE_WIDTH)
    a = p.parse_args()
    if a.command == 'compose':
        r = compose(a.plan, a.attempts, a.out)
        result = {'candidate': r['candidate'], 'compare': r['compare'], 'output': r['output'],
                  'proof': str(Path(a.out).resolve() / 'scene-edit-proof.json'), 'region_pixels': r['region_pixels']}
    elif a.command == 'verify':
        path = Path(a.proof).resolve(); plan, row, _ = verify(read(path), path.parent)
        result = {'status': 'verified', 'attempts': len(row['attempts']), 'selected': row['selected']}
    else:
        out = Path(a.out).resolve(); require(not out.exists(), 'Use a new side-by-side path')
        side_by_side(load(a.left)[0], load(a.right)[0], tuple(a.labels), a.width).save(out, quality=90)
        result = {'compare': str(out)}
    print(json.dumps(result, indent=2))
