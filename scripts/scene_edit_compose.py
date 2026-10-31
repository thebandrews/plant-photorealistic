"""Compose built-in image-tool scene edits back into the finished photo, and check their review.

Each selected generation is resized back to its frozen box (aspect must already match; nothing is warped).
Tiles blend with separable raised-cosine weights across their overlaps; the upscaled full-view result fills
where tiles do not reach (or the previous composite, when no full-view attempt was selected). The scene then
replaces the previous composite only inside the landscape edit region, through a raised-cosine inward feather,
in linear light. Whole tiles / the whole region are blended: generated pixels are never pasted into plant
outlines. Outside the region every sample stays exactly the previous composite, which there is exactly the
cleaned original photo; both are asserted. Visual findings remain the agent's judgments.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from handoff import checked, read, record, require
from raster import load, save, scaled_like
from scene_edit_plan import IMAGE_TOOLS, MAX_ATTEMPTS, MODEL, reference_pixels, region_weight, resized, validate

OPERATION = 'scene_edit_raised_cosine_tiles_full_view_linear_v1'
SCENE_CHECKS = ('species_fidelity', 'no_invented_or_hybrid_plants', 'plants_rooted_in_beds',
                'hardscape_geometry_preserved', 'flowering_in_bloom', 'no_visible_seams', 'house_background_retained')
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


def raised_cosine(t):
    return (.5 - .5 * np.cos(np.pi * np.clip(t, 0, 1))).astype(np.float32)


def window(start, size, limit, ramp):
    """1D tile weight: ramps to zero at tile edges inside the canvas; flat at the canvas border."""
    k = np.arange(size, dtype=np.float32) + .5
    w = np.ones(size, np.float32)
    if start > 0:
        w *= raised_cosine(k / ramp)
    if start + size < limit:
        w *= raised_cosine((size - k) / ramp)
    return w


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
            if a.get('reference'):
                row['reference'] = record(mp.parent / a['reference']); row['reference_from'] = a.get('reference_from')
            attempts.append(row)
        jobs.append({'id': job_id, 'attempts': attempts, 'selected': job.get('selected')})
    return {'agent_model': m.get('agent_model'), 'reasoning_effort': m.get('reasoning_effort'),
            'image_tool': m.get('image_tool'), 'manifest': record(mp), 'jobs': jobs}


def check_attempts(plan, attempts, base):
    """Model, attempt count, exact prompts, references and returned aspect for every planned job."""
    require(attempts.get('agent_model') == MODEL['agent_model'] and attempts.get('reasoning_effort') == MODEL['reasoning_effort'],
            'Scene edits must come from gpt-6-astra with high reasoning')
    require(attempts.get('image_tool') in IMAGE_TOOLS, 'Scene edits must use the built-in image tool')
    planned = {j['id']: j for j in plan['jobs']}
    rows = {j['id']: j for j in attempts['jobs']}
    require(len(rows) == len(attempts['jobs']) and set(rows) == set(planned), 'Record every planned scene edit job exactly once')
    full = next(j for j in plan['jobs'] if j['kind'] == 'full_view')
    for job_id, row in rows.items():
        job = planned[job_id]; tries = row['attempts']
        require(len(tries) <= MAX_ATTEMPTS, f'At most {MAX_ATTEMPTS} attempts per scene edit job')
        require([a['attempt'] for a in tries] == list(range(1, len(tries) + 1)), 'Attempts are numbered in order')
        require(row['selected'] is None or row['selected'] in range(1, len(tries) + 1), 'Selected attempt does not exist')
        planned_prompt = text(job['prompt'], base)
        for a in tries:
            used = text(a['prompt'], base)
            if a['attempt'] == 1:
                require(used == planned_prompt, f'Attempt 1 of {job_id} must use the exact planned prompt')
            else:
                require(tries[0].get('inspection'), 'Attempt 2 follows a recorded inspection of attempt 1')
                require(used.startswith(planned_prompt) and len(used[len(planned_prompt):].strip()) >= 20 and
                        used != text(tries[0]['prompt'], base), 'Attempt 2 keeps the planned prompt and adds a correction')
            resized(checked(a['raw'], base), job['box'])
            if job['kind'] == 'tile':
                require(a.get('reference') and a.get('reference_from') in range(1, MAX_ATTEMPTS + 1), 'Tiles take the full-view result crop as reference')
                source = rows['full']['attempts']
                require(a['reference_from'] <= len(source), 'Tile reference names a missing full-view attempt')
                ref, _ = load(checked(a['reference'], base))
                want = reference_pixels(checked(source[a['reference_from'] - 1]['raw'], base), full['box'], job['box'])
                require(np.array_equal(ref, want), f'Tile reference is not the full-view crop: {job_id}')
            else:
                require(not a.get('reference'), 'The full view is edited from its own input only')
    return rows


def reconstruct(plan, region, composite, rows, base):
    h, w = composite.shape[:2]
    weight = region_weight(region, plan['params']['feather_px'])
    full_job = next(j for j in plan['jobs'] if j['kind'] == 'full_view')
    full = None
    if rows['full']['selected']:
        full = resized(checked(rows['full']['attempts'][rows['full']['selected'] - 1]['raw'], base), full_job['box'])
    rx, ry = plan['ramps_px']
    tiles = []
    for job in plan['jobs']:
        row = rows[job['id']]
        if job['kind'] != 'tile' or not row['selected']:
            continue
        x0, y0, x1, y1 = job['box']
        tiles.append((job['box'], linear(resized(checked(row['attempts'][row['selected'] - 1]['raw'], base), job['box'])),
                      window(x0, x1 - x0, w, rx), window(y0, y1 - y0, h, ry)))
    final = composite.copy(); coverage = np.zeros((h, w), np.float32)
    for b0 in range(0, h, BLOCK):
        b1 = min(h, b0 + BLOCK)
        wr = weight[b0:b1]
        if not np.any(wr > 0):
            continue
        num = np.zeros((b1 - b0, w, 3), np.float32); den = np.zeros((b1 - b0, w), np.float32)
        for (x0, y0, x1, y1), lin, wx, wy in tiles:
            r0, r1 = max(b0, y0), min(b1, y1)
            if r0 >= r1:
                continue
            tw = wy[r0 - y0:r1 - y0, None] * wx[None, :]
            num[r0 - b0:r1 - b0, x0:x1] += lin[r0 - y0:r1 - y0] * tw[..., None]
            den[r0 - b0:r1 - b0, x0:x1] += tw
        base_lin = linear(composite[b0:b1])
        fill = linear(full[b0:b1]) if full is not None else base_lin
        c = np.minimum(den, 1)[..., None]
        tiled = np.divide(num, den[..., None], out=np.zeros_like(num), where=den[..., None] > 0)
        scene = c * tiled + (1 - c) * fill
        mix = wr[..., None] * scene + (1 - wr[..., None]) * base_lin
        block = encode16(mix)
        if composite.dtype != np.uint16:
            block = scaled_like(block, composite)
        keep = wr == 0
        block[keep] = composite[b0:b1][keep]
        final[b0:b1] = block
        coverage[b0:b1] = np.minimum(den, 1)
    require(np.array_equal(final[~region], composite[~region]), 'Scene edit changed pixels outside the landscape edit region')
    return final, weight, coverage


def outside_is_cleaned_photo(region, composite, cleaned):
    require(np.array_equal(composite[~region], scaled_like(cleaned, composite)[~region]),
            'Previous composite differs from the cleaned photo outside the edit region')


def proof_panel(job, rows, composite, final, base):
    x0, y0, x1, y1 = job['box']
    to8 = lambda a: a if a.dtype == np.uint8 else np.rint(a / 257).astype(np.uint8)
    row = rows[job['id']]
    gen = (resized(checked(row['attempts'][row['selected'] - 1]['raw'], base), job['box']) if row['selected']
           else np.full((y1 - y0, x1 - x0, 3), 128, np.uint8))
    panel = np.concatenate([to8(composite[y0:y1, x0:x1]), gen, to8(final[y0:y1, x0:x1])], axis=1)
    image = Image.fromarray(panel)
    if image.height > 1024:
        image = image.resize((round(image.width * 1024 / image.height), 1024), Image.LANCZOS)
    return image


def compose(plan_path, manifest, out):
    pp = Path(plan_path).resolve(); plan = read(pp)
    plan, region, composite, cleaned, _ = validate(plan, pp.parent)
    outside_is_cleaned_photo(region, composite, cleaned)
    attempts = normalize(manifest)
    rows = check_attempts(plan, attempts, pp.parent)
    require(any(r['selected'] for r in rows.values()), 'Select at least one scene edit, or record a fallback review instead')
    final, weight, coverage = reconstruct(plan, region, composite, rows, pp.parent)
    out = Path(out).resolve(); require(not out.exists(), 'Use a new scene edit output directory'); (out / 'proofs').mkdir(parents=True)
    save(out / 'candidate.png', final)
    Image.fromarray(np.rint(weight * 255).astype(np.uint8)).save(out / 'region-weight.png')
    Image.fromarray(np.rint(coverage * 255).astype(np.uint8)).save(out / 'tile-coverage.png')
    proofs = {}
    for job in plan['jobs']:
        proof_panel(job, rows, composite, final, pp.parent).save(out / 'proofs' / f"{job['id']}.png")
        proofs[job['id']] = record(out / 'proofs' / f"{job['id']}.png")
    proof = {'kind': 'scene_edit_proof', 'operation': OPERATION, 'plan': record(pp), 'attempts': attempts,
             'candidate': record(out / 'candidate.png'), 'region_weight': record(out / 'region-weight.png'),
             'tile_coverage': record(out / 'tile-coverage.png'), 'job_proofs': proofs,
             'region_pixels': int(region.sum()), 'outside_region_pixels_preserved': int((~region).sum())}
    (out / 'scene-edit-proof.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    return proof


def verify(proof, base):
    require(proof.get('kind') == 'scene_edit_proof' and proof.get('operation') == OPERATION, 'Scene edit proof required')
    pp = checked(proof['plan'], base)
    plan, region, composite, cleaned, baseline = validate(read(pp), pp.parent)
    outside_is_cleaned_photo(region, composite, cleaned)
    attempts = proof['attempts']
    checked(attempts['manifest'], base)
    rows = check_attempts(plan, attempts, pp.parent)
    final, weight, coverage = reconstruct(plan, region, composite, rows, pp.parent)
    actual, alpha = load(checked(proof['candidate'], base))
    require(alpha is None and np.array_equal(actual, final), 'Scene edit reconstruction differs')
    for key, want in (('region_weight', np.rint(weight * 255)), ('tile_coverage', np.rint(coverage * 255))):
        with Image.open(checked(proof[key], base)) as im:
            require(np.array_equal(np.array(im), want.astype(np.uint8)), f'Scene edit {key} differs')
    require(set(proof['job_proofs']) == {j['id'] for j in plan['jobs']}, 'Every scene edit job needs a proof image')
    for ref in proof['job_proofs'].values():
        checked(ref, base)
    require(proof.get('outside_region_pixels_preserved') == int((~region).sum()), 'Recompute preserved pixel count')
    return plan, rows, baseline


def check_review(data, base):
    """Schema 4 scene_photographic_edit review: accepted edit, or explicit fallback to the previous composite."""
    from photo_finish import check_review as check_finish
    brp = checked(data['baseline_review'], base); previous = read(brp)
    require(previous.get('schema_version') == 4 and previous.get('method') == 'photo_finish', 'Previous composite needs its schema 4 photo_finish review')
    require(checked(data['handoff'], base) == checked(previous['handoff'], brp.parent) and data['camera_id'] == previous['camera_id'],
            'Scene edit review camera/handoff differs from the previous composite')
    vision = data.get('scene_review', {})
    fallback = data.get('release_policy') == 'fallback_previous_composite'
    require(data.get('release_policy') in (None, 'fallback_previous_composite'), 'Unknown scene edit release policy')
    limitations = data.get('limitations', [])
    require(all(r.get('check') and r.get('observations') for r in limitations), 'Each limitation needs its check and observations')
    if not data.get('scene_edit_proof'):
        require(fallback and limitations, 'Without a scene edit proof only a disclosed fallback is possible')
        baseline = check_finish(previous, brp.parent); rows = {}; checks = {}; nonpassing = [r['check'] for r in limitations]
    else:
        sp = checked(data['scene_edit_proof'], base); proof = read(sp)
        plan, rows, baseline = verify(proof, sp.parent)
        require(checked(plan['review'], sp.parent) == brp, 'Scene edit plan was made from a different previous composite')
        model = data.get('model', {})
        require(model.get('agent_model') == proof['attempts']['agent_model'] == MODEL['agent_model'] and
                model.get('reasoning_effort') == proof['attempts']['reasoning_effort'] == MODEL['reasoning_effort'] and
                model.get('image_tool') == proof['attempts']['image_tool'], 'Record the gpt-6-astra/high model and image tool used')
        require(vision.get('reviewer') == 'vision-capable-agent' and vision.get('source_and_final_loaded') is True and
                vision.get('native_size_crops_loaded') is True and vision.get('observations'), 'Actual scene edit review of source, final and native-size crops required')
        from botanical_refinement import images
        images(vision.get('paired_evidence'), base)
        checks = vision.get('checks', {})
        require(all(checks.get(k) in ('pass', 'fail', 'estimated') for k in SCENE_CHECKS), 'Every scene edit check needs an actual disposition')
        nonpassing = [k for k in SCENE_CHECKS if checks[k] != 'pass']
        job_rows = vision.get('jobs', [])
        require(len(job_rows) == len(rows) and {r.get('id') for r in job_rows} == set(rows), 'Review every scene edit job exactly once')
        for r in job_rows:
            want = rows[r['id']]
            require(r.get('attempts') == len(want['attempts']) and r.get('selected_attempt') == want['selected'] and r.get('observations'),
                    'Job review must match its recorded attempts and selection')
            if want['selected']:
                require(r.get('species_fidelity') in ('pass', 'fail') and r.get('invented_or_hybrid_plants') in ('none', 'present'),
                        'Selected job needs species fidelity and invented/hybrid plant findings')
                images([r.get('evidence')], base)
                if r['species_fidelity'] != 'pass' or r['invented_or_hybrid_plants'] != 'none':
                    require(fallback, f"Job {r['id']} fails species fidelity; deselect it and recompose, or fall back")
                    if 'species_fidelity' not in nonpassing:
                        nonpassing.append('species_fidelity')
        if fallback:
            require(nonpassing, 'A fallback needs the failed scene edit findings')
            require({r['check'] for r in limitations} == set(nonpassing) and len(limitations) == len(nonpassing),
                    'Describe each nonpassing scene edit check')
        else:
            require(not nonpassing, 'Scene edit review failed: fall back to the previous composite with disclosed limitations')
            require(checked(data['candidate'], base) == checked(proof['candidate'], sp.parent), 'Scene edit review candidate differs from its proof')
    if fallback:
        require(checked(data['candidate'], base) == checked(previous['candidate'], brp.parent),
                'A fallback delivers the previous photo_finish composite unchanged')
    attempts = {k: len(v['attempts']) for k, v in rows.items()}
    status = 'completed_with_limitations' if fallback else baseline['status']
    return {'status': status, 'method': 'scene_photographic_edit', 'camera_id': data['camera_id'],
            'delivered': 'previous_photo_finish_composite' if fallback else 'scene_edit_composite',
            'scene_edit_status': 'rejected' if fallback else 'accepted', 'nonpassing_checks': nonpassing if fallback else [],
            'limitations': limitations if fallback else baseline.get('limitations', []),
            'previous_composite_status': baseline['status'], 'previous_composite_limitations': baseline.get('limitations', []),
            'model': data.get('model'), 'attempts': attempts, 'selected': {k: v['selected'] for k, v in rows.items()},
            'photo_background_composited': True, 'bed_ground': baseline.get('bed_ground'),
            'landscape_evidence_counts': baseline.get('landscape_evidence_counts'),
            'upstream_handoff_kind': baseline.get('upstream_handoff_kind'), 'upstream_certification': baseline.get('upstream_certification'),
            'scope': 'Built-in image-tool photographic edit of the landscape region of a checked photo composite',
            'limitation': 'Generated plant pixels are agent-reviewed derivatives; the native scene stays the placement and species authority'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('compose'); a.add_argument('plan'); a.add_argument('--attempts', required=True, help='Attempt manifest JSON')
    a.add_argument('--out', required=True)
    a = sub.add_parser('verify'); a.add_argument('proof'); a = p.parse_args()
    if a.command == 'compose':
        r = compose(a.plan, a.attempts, a.out)
        result = {'candidate': r['candidate'], 'proof': str(Path(a.out).resolve() / 'scene-edit-proof.json'), 'region_pixels': r['region_pixels']}
    else:
        path = Path(a.proof).resolve(); plan, rows, _ = verify(read(path), path.parent)
        result = {'status': 'verified', 'jobs': len(plan['jobs']), 'selected': {k: v['selected'] for k, v in rows.items()}}
    print(json.dumps(result, indent=2))
