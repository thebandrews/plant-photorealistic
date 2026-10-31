"""Load a frozen photoreal bench (see bench/README.md).

A bench is a read-only folder made from one finished pipeline run:
  bench-manifest.json            views, files and hashes
  handoff/                       render-handoff.json, plant ledger, candidate-catalog-v3.json (object kinds)
  views/<view>/                  source.jpg, cleaned-photo.png, native.png, object-ids.npy, depth.npy,
                                 structural.png, projected-plants.json, baseline/final.png,
                                 baseline-run/ (alpha.png, straight_rgb.png, transition_alpha.png, cleanup-masks/)
  crops.json                     fixed judge crops (bench.py init-crops)
"""
import hashlib
import json
import os
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PLANT_KINDS = {'plant'}
GROUND_KINDS = {'bed', 'terrain'}
HARDSCAPE_KINDS = {'hardscape', 'driveway', 'boulder'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def load_gray(path):
    return np.asarray(Image.open(path).convert('L'))


def save_rgb(path, arr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


class View:
    def __init__(self, bench, entry):
        self.bench = bench
        self.id = entry['id']
        self.camera_id = entry['camera_id']
        self.dir = bench.root / 'views' / self.id
        w, h = entry.get('dimensions') or (None, None)
        self.size = (w, h)

    def path(self, rel):
        return self.dir / rel

    # -- images -------------------------------------------------------------
    @cached_property
    def cleaned(self):
        """The source photo with people and construction clutter removed (frozen from the baseline run)."""
        return load_rgb(self.path('cleaned-photo.png'))

    @cached_property
    def native(self):
        return load_rgb(self.path('native.png'))

    @cached_property
    def baseline_final(self):
        return load_rgb(self.path('baseline/final.png'))

    @cached_property
    def straight_rgb(self):
        return load_rgb(self.path('baseline-run/straight_rgb.png'))

    @cached_property
    def alpha(self):
        """Landscape matte of the baseline run's refined render, 0..1."""
        return load_gray(self.path('baseline-run/alpha.png')).astype(np.float32) / 255.0

    @cached_property
    def transition(self):
        p = self.path('baseline-run/transition_alpha.png')
        return load_gray(p).astype(np.float32) / 255.0 if p.exists() else np.zeros(self.ids.shape, np.float32)

    @cached_property
    def cleanup(self):
        """Union of the cleanup masks (areas where the photo itself was repaired)."""
        out = np.zeros(self.ids.shape, bool)
        for p in sorted(self.path('baseline-run/cleanup-masks').glob('*.png')):
            out |= load_gray(p) > 0
        return out

    # -- geometry -----------------------------------------------------------
    @cached_property
    def ids(self):
        """Handoff object-ID pass (pass_index per pixel): the placement authority."""
        return np.load(self.path('object-ids.npy'))

    @cached_property
    def depth(self):
        return np.load(self.path('depth.npy'))

    @cached_property
    def plants(self):
        """Visible plants: [{id, species, pass_index, pixels, box (x0,y0,x1,y1), centroid (x,y)}], largest first."""
        pp = read_json(self.path('projected-plants.json'))['plants']
        ids = self.ids
        counts = np.bincount(ids.ravel().clip(0), minlength=int(ids.max()) + 1)
        out = []
        for p in pp:
            k = p.get('pass_index')
            if k is None or k >= len(counts) or counts[k] == 0:
                continue
            ys, xs = np.nonzero(ids == k)
            out.append({'id': p['id'], 'species': p['species'], 'pass_index': int(k), 'pixels': int(counts[k]),
                        'box': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                        'centroid': [float(xs.mean()), float(ys.mean())]})
        from segment import seed_points
        for p in out:
            p['points'] = seed_points(ids, p)
        out.sort(key=lambda p: -p['pixels'])
        return out

    def kind_mask(self, kinds):
        lut = self.bench.kind_lut(int(self.ids.max()) + 1)
        sel = np.isin(lut, list(kinds))
        return sel[self.ids.clip(0)]

    @cached_property
    def plant_mask(self):
        return self.kind_mask(PLANT_KINDS)

    @cached_property
    def edit_region(self):
        """Pixels an experiment may change: the landscape matte and transition band (dilated), plus cleanup areas."""
        import cv2
        base = (self.alpha > 0.002) | (self.transition > 0.002) | self.cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
        return cv2.dilate(base.astype(np.uint8), kernel) > 0

    def naive_composite(self):
        """Straight-alpha overlay of the baseline render onto the cleaned photo: the raw 3D look, no finishing."""
        a = self.alpha[..., None]
        out = self.straight_rgb.astype(np.float32) * a + self.cleaned.astype(np.float32) * (1 - a)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class Bench:
    def __init__(self, root=None):
        root = root or os.environ.get('PHOTOREAL_BENCH')
        if not root:
            raise SystemExit('Set PHOTOREAL_BENCH or pass --bench to the bench folder (the one with bench-manifest.json)')
        self.root = Path(root)
        self.manifest = read_json(self.root / 'bench-manifest.json')
        self.views = [View(self, v) for v in self.manifest['views']]
        self.name = self.manifest.get('bench', self.root.name)

    def view(self, view_id):
        for v in self.views:
            if v.id == view_id:
                return v
        raise KeyError(view_id)

    @cached_property
    def catalog(self):
        cat = read_json(self.root / 'handoff' / 'candidate-catalog-v3.json')
        return {int(o['pass_index']): o for o in cat['objects'] if o.get('pass_index') is not None}

    def kind_lut(self, n):
        lut = np.array([''] * max(n, max(self.catalog) + 1), dtype=object)
        for k, o in self.catalog.items():
            lut[k] = o.get('kind') or ''
        return lut

    @property
    def crops(self):
        p = self.root / 'crops.json'
        return read_json(p) if p.exists() else None

    @property
    def bloom(self):
        """bloom.json: {'rule', 'bloom': {species: expected flowers}, 'not_required': {species: why}} or None."""
        p = self.root / 'bloom.json'
        return read_json(p) if p.exists() else None

    def verify(self):
        """Check every frozen file against its recorded hash. Returns the list of mismatches."""
        bad = []
        for f in self.manifest['files']:
            p = self.root / f['path']
            if not p.exists() or sha256(p) != f['sha256']:
                bad.append(f['path'])
        return bad
