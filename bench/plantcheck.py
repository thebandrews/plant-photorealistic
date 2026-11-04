"""Per-plant check of a delivered view against the render, in Ben's ranked defect terms.

  species_change        the plant no longer has its form: its render crop clearly reads as one plant form (fern,
                        agave/yucca rosette, cactus, grass, conifer, strap leaves, shrub, flowering mound, succulent
                        groundcover, or bare mulch) and its output crop reads as another (BioCLIP zero-shot over
                        those forms, from words only; never the reference photos, so a reference-sheet experiment
                        can't grade itself). Exact-species ranks are kept per plant for information: zero-shot
                        species within one form (ostrich vs lady fern) is not reliable enough to count.
  significant_movement  the plant's colour mass (blurred Lab of its own render pixels) is found in the output,
                        clearly better than in place and shifted by a large share of its size. Leaf texture is
                        not compared: the edit re-textures every plant.
  invented_plant        new vegetation on open bed ground well away from every designed plant, in pieces the
                        size of a plant (fuller foliage spilling into small gaps is allowed)

Crops dim everything but the plant's own render pixels, so neighbours don't decide its form. Counts are per
plant ID (and per new-vegetation component), not review observations. Deterministic: one fixed model on CPU,
no sampling. Only plants big enough to judge are checked (`min_side`, `min_pixels`).
"""
import numpy as np

import cv2

T = {
    'min_side': 64,          # px: plants smaller than this on either side are not checked
    'min_pixels': 2500,      # own visible render pixels
    'pad': 0.15,             # crop margin, share of the plant's box
    'search': 0.5,           # movement search radius, share of the plant's box side
    'blur': 0.06,            # colour-mass blur, share of the plant's box side
    'match_min': 0.60,       # colour-mass match below this: the plant is not found
    'match_gain': 0.15,      # a shifted match must beat the in-place match by this much
    'move_rel': 0.15,        # shift / box diagonal that counts as significant movement
    'dim': 0.25,             # crop pixels away from the plant's own pixels are dimmed to this
    'form_min': 0.40,        # the render crop must read as one form at least this clearly to be judged
    'form_ratio': 0.30,      # species_change: the output keeps less than this share of that form's probability
    'newveg_exg': 0.08,      # as the new_plants gate
    'plant_clear_px': 41,    # new vegetation must be this far (kernel) from every designed plant
    'addition_px': 15000,    # a new-vegetation component at least this big counts as an added plant
}

FORMS = {  # plant forms, as BioCLIP prompts; a "species change" is a plant that leaves its form
    'fern': 'a fern with feathery divided fronds',
    'rosette': 'an agave or yucca, a rosette of stiff thick sword-shaped leaves',
    'cactus': 'a cactus with spines',
    'grass': 'an ornamental grass, a clump of thin grass blades',
    'conifer': 'a conifer with needle foliage',
    'strap': 'a clump of broad strap-shaped leaves',
    'shrub': 'a leafy broadleaf shrub',
    'flowers': 'a mound of flowering perennials covered in flowers',
    'groundcover': 'a low succulent groundcover',
    'mulch': 'bare bark mulch with no plant',
}


def checked_plants(view):
    out = []
    for p in view.plants:
        x0, y0, x1, y1 = p['box']
        if min(x1 - x0, y1 - y0) >= T['min_side'] and p['pixels'] >= T['min_pixels']:
            out.append(p)
    return out


def _lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def movement(ids, ref_lab, out_lab, p):
    """Where the plant's colour mass is in the output: (dx, dy, shift / diagonal, match, in-place match)."""
    h, w = ids.shape
    bw, bh = p['box'][2] - p['box'][0], p['box'][3] - p['box'][1]
    e = 4  # the template reaches a little past the plant's box, so its outline is part of it
    x0, y0, x1, y1 = max(0, p['box'][0] - e), max(0, p['box'][1] - e), min(w, p['box'][2] + e), min(h, p['box'][3] + e)
    r = int(T['search'] * max(bw, bh)) + 2
    sx0, sy0, sx1, sy1 = max(0, x0 - r), max(0, y0 - r), min(w, x1 + r), min(h, y1 + r)
    sigma = max(1.5, T['blur'] * max(bw, bh))
    templ = cv2.GaussianBlur(ref_lab[y0:y1, x0:x1], (0, 0), sigma)
    search = cv2.GaussianBlur(out_lab[sy0:sy1, sx0:sx1], (0, 0), sigma)
    own = (ids[y0:y1, x0:x1] == p['pass_index']).astype(np.uint8)
    own = cv2.dilate(own, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
    if own.sum() < 16 or templ[own > 0].std() < 1e-3:
        return 0, 0, 0.0, 0.0, 0.0
    mask = np.repeat(own[..., None].astype(np.float32), 3, axis=2)
    res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED, mask=mask)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, peak, _, (px, py) = cv2.minMaxLoc(res)
    ix, iy = x0 - sx0, y0 - sy0
    here = float(res[iy, ix])
    dx, dy = px - ix, py - iy
    return int(dx), int(dy), float(np.hypot(dx, dy) / float(np.hypot(bw, bh))), float(peak), here


def crop_box(p, shape, dx=0, dy=0):
    """Square crop around the plant's box (with margin), shifted by (dx, dy), inside the image."""
    x0, y0, x1, y1 = p['box']
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    side = max(x1 - x0, y1 - y0) * (1 + 2 * T['pad'])
    h, w = shape[:2]
    side = min(side, w, h)
    cx0 = int(round(min(max(cx - side / 2, 0), w - side)))
    cy0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return [cx0, cy0, cx0 + int(side), cy0 + int(side)]


def focus(img, ids, p, b, dx=0, dy=0):
    """The crop b with everything but the plant's own render pixels (grown a little, moved by dx, dy) dimmed."""
    h, w = ids.shape
    own = np.zeros((h, w), np.uint8)
    ys, xs = np.nonzero(ids == p['pass_index'])
    ys, xs = ys + dy, xs + dx
    keep = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    own[ys[keep], xs[keep]] = 1
    g = max(3, int(0.04 * max(p['box'][2] - p['box'][0], p['box'][3] - p['box'][1])))
    own = cv2.dilate(own[b[1]:b[3], b[0]:b[2]], np.ones((2 * g + 1, 2 * g + 1), np.uint8)).astype(np.float32)
    own = cv2.GaussianBlur(own, (0, 0), g / 2)[..., None]
    crop = img[b[1]:b[3], b[0]:b[2]].astype(np.float32)
    return np.clip(crop * (T['dim'] + (1 - T['dim']) * own) + 0.5, 0, 255).astype(np.uint8)


def _exg(rgb):
    f = rgb.astype(np.float32)
    return (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(axis=2) + 1.0)


def additions(view, ref, out):
    """New vegetation on open ground (beds, terrain) well away from designed plants, in plant-sized pieces."""
    from benchlib import GROUND_KINDS
    ground = view.kind_mask(GROUND_KINDS) & (view.alpha > 0.5)
    k = T['plant_clear_px']
    ground &= ~(cv2.dilate(view.plant_mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)
    new = ground & ((_exg(out) - _exg(ref)) > T['newveg_exg']) & (_exg(out) > 0.05)
    n, _, stats, cents = cv2.connectedComponentsWithStats(new.astype(np.uint8), 8)
    h, w = new.shape
    rows = []
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        if a >= T['addition_px']:
            cx, cy = cents[k]
            rows.append({'pixels': a, 'centre_percent': [round(100 * cx / w), round(100 * cy / h)],
                         'box': [int(stats[k, 0]), int(stats[k, 1]), int(stats[k, 0] + stats[k, 2]),
                                 int(stats[k, 1] + stats[k, 3])]})
    rows.sort(key=lambda r: -r['pixels'])
    return rows


def _rank(probs, k):
    order = np.argsort(-probs)
    return int(np.nonzero(order == k)[0][0]) + 1


def species_names(view):
    return sorted({p['species'] for p in view.plants})


class BioClip:
    """Zero-shot scores from BioCLIP (imageomics/bioclip, open_clip) on CPU, for plant forms or species names."""
    MODEL = 'hf-hub:imageomics/bioclip'

    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.MODEL)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.MODEL)
        self._text = {}

    @staticmethod
    def label(name):
        if name in FORMS:
            return f'a photo of {FORMS[name]}.'
        return f"a photo of {name.split(chr(39))[0].strip()}."  # cultivar names mean nothing to the model

    def text(self, names):
        key = tuple(names)
        if key not in self._text:
            with self.torch.no_grad():
                t = self.model.encode_text(self.tokenizer([self.label(n) for n in names]))
                self._text[key] = t / t.norm(dim=-1, keepdim=True)
        return self._text[key]

    def probs(self, crops, names):
        """crops: list of RGB uint8 arrays -> (len(crops), len(names)) probabilities."""
        from PIL import Image
        text = self.text(names)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops), 32):
                batch = self.torch.stack([self.preprocess(Image.fromarray(c)) for c in crops[i:i + 32]])
                f = self.model.encode_image(batch)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append((100.0 * f @ text.T).softmax(dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(names)))


def check(view, out, classifier, ref=None, ref_probs=None):
    """All checked plants of one view: per-plant rows and the ranked-defect summary. `ref_probs` caches the
    render crops' scores between images of the same view."""
    ref = view.naive_composite() if ref is None else ref
    ref_lab, out_lab = _lab(ref), _lab(out)
    names, forms = species_names(view), list(FORMS)
    plants = checked_plants(view)
    moves = [movement(view.ids, ref_lab, out_lab, p) for p in plants]
    if ref_probs is None:
        crops = [focus(ref, view.ids, p, crop_box(p, ref.shape)) for p in plants]
        ref_probs = (classifier.probs(crops, forms), classifier.probs(crops, names))
    out_crops = []
    for p, (dx, dy, rel, peak, here) in zip(plants, moves):
        found = peak >= T['match_min'] and peak - here >= T['match_gain']
        dx, dy = (dx, dy) if found else (0, 0)
        out_crops.append(focus(out, view.ids, p, crop_box(p, out.shape, dx, dy), dx, dy))
    out_forms, out_names = classifier.probs(out_crops, forms), classifier.probs(out_crops, names)
    rows = []
    for i, (p, (dx, dy, rel, peak, here)) in enumerate(zip(plants, moves)):
        f = int(np.argmax(ref_probs[0][i]))
        pf_ref, pf_out = float(ref_probs[0][i][f]), float(out_forms[i][f])
        judged = pf_ref >= T['form_min']
        species_change = judged and int(np.argmax(out_forms[i])) != f and pf_out < T['form_ratio'] * pf_ref
        moved = peak >= T['match_min'] and peak - here >= T['match_gain'] and rel >= T['move_rel']
        k = names.index(p['species'])
        rows.append({'id': p['id'], 'species': p['species'], 'pixels': p['pixels'], 'box': p['box'],
                     'shift_px': [dx, dy], 'shift_rel': round(rel, 3), 'match': round(peak, 3),
                     'match_in_place': round(here, 3), 'form_render': forms[f], 'p_form_render': round(pf_ref, 3),
                     'form_output': forms[int(np.argmax(out_forms[i]))], 'p_form_output': round(pf_out, 3),
                     'species_rank_render': _rank(ref_probs[1][i], k), 'species_rank_output': _rank(out_names[i], k),
                     'species_judged': judged, 'species_change': bool(species_change),
                     'significant_movement': bool(moved)})
    added = additions(view, ref, out)
    summary = {'checked': len(rows), 'species_judged': sum(r['species_judged'] for r in rows),
               'species_change': sum(r['species_change'] for r in rows),
               'significant_movement': sum(r['significant_movement'] for r in rows),
               'invented_plant': len(added)}
    return {'summary': summary, 'plants': rows, 'additions': added, 'thresholds': dict(T)}, ref_probs
