"""Bed ground: photographic fine bark-mulch grain on open bed ground of the finished composite.

Renderer bed soil reads as flat brown paint. This pass keeps the composite's own low-frequency tone, light, shadows and
colour and adds only zero-mean grain from packaged photographic exemplars, mapped in world space through the renderer
camera and depth so the shreds have the right size at every distance. It is deterministic and bounded:

* only open bed ground changes: near-horizontal bed-kind renderer surfaces (not the sides of raised beds), fully
  opaque landscape alpha, outside cleanup, transition and context-occlusion permissions, more than GAP px from every
  plant pixel;
* every plant pixel and its GAP px ring, and every pixel outside open bed ground, keep their exact input samples;
* grain and colour fade in over FEATHER px from that ring and EDGE px from bed borders; crushed render shade is lifted
  softly over LIFT_FEATHER px (contact shade, not a black outline); the render's small bright leaf decals on the soil
  are inpainted inside the zone first.

Soil colour stays the render's own, optionally pulled CHROMA_SOIL toward bare soil the agent pointed at in a site photo
(`soil_reference`). Ported from the photoreal lab's round-5 H1 experiment (Reddy bloom bench: beat the shipped image on
91% of judged crops with every plant pixel unchanged).
"""
from pathlib import Path

import cv2
import numpy as np

from handoff import checked, record, require
from raster import load

OPERATION = 'bed_ground_mulch_v1'
ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'bed-ground'
EXEMPLARS = ('mulch-0.png', 'mulch-1.png', 'mulch-2.png')
GAP = 8              # px: plants dilated by this keep their exact samples
SAFE = 1.5           # px of zero weight beyond the ring (ellipse dilation vs distance transform rounding)
FEATHER = 8.0        # px over which grain and colour fade in from the ring
LIFT_FEATHER = 32.0  # px over which the crushed-shade lift fades in
EDGE = 3.0           # px fade at bed borders away from plants
TEX_M = 0.22         # metres covered by one exemplar (shreds 1-3 cm)
GRAIN = 1.6          # contrast of the exemplar's relative luminance
CLUMP = 0.14         # +- brightness of 7 / 19 cm clumps
CHROMA_TEX = 0.12    # share of the exemplar's own chroma
CHROMA_SOIL = 0.30   # pull toward the soil reference chroma, when one is given
SHADE_FLOOR = 20.0   # soft-knee floor (8-bit units) for crushed-black render shade
ANISO = 1.5          # max footprint anisotropy before blurring
MAX_SLOPE = 0.36     # rise over run (~20 degrees): steeper bed faces (raised-bed sides) are not mulched
MIN_OPEN_PX = 500    # less open bed ground than this is skipped
LUMA = np.array([0.299, 0.587, 0.114])


# -- plan ---------------------------------------------------------------------------------------------------------
def plan_spec(enabled=True, soil_image=None, soil_box=None):
    """The frozen plan entry. Off: {'operation': 'off'}; on: packaged exemplars (hashed) and an optional soil patch."""
    if not enabled:
        return {'operation': 'off', 'reason': 'disabled for this plan'}
    spec = {'operation': OPERATION, 'exemplars': [record(ASSETS / n) for n in EXEMPLARS]}
    if soil_image:
        require(soil_box and len(soil_box) == 4 and soil_box[0] < soil_box[2] and soil_box[1] < soil_box[3],
                'Soil reference needs a box x0 y0 x1 y1')
        spec['soil_reference'] = {'image': record(soil_image), 'box': [int(v) for v in soil_box]}
    return spec


def check_spec(spec, base):
    """Validate a plan entry; returns True when the pass is on."""
    if spec is None or spec.get('operation') == 'off':
        return False
    require(spec.get('operation') == OPERATION and len(spec.get('exemplars', [])) == len(EXEMPLARS),
            'Unknown bed-ground operation')
    for rec in spec['exemplars']:
        checked(rec, base)
    if spec.get('soil_reference'):
        checked(spec['soil_reference']['image'], base)
    return True


def soil_chroma(spec, base):
    """rgb / luma of the mean of the reference patch's mid-tone pixels, or None (keep the render's chroma)."""
    ref = spec.get('soil_reference')
    if not ref:
        return None
    rgb, _ = load(checked(ref['image'], base))
    x0, y0, x1, y1 = ref['box']
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64) * (255 / np.iinfo(rgb.dtype).max)
    lum = patch @ LUMA
    patch = patch[(lum > 25) & (lum < 160)]
    require(len(patch) >= 100, 'Soil reference box holds too little mid-tone soil')
    mean = patch.mean(0)
    return mean / (mean @ LUMA)


# -- camera -------------------------------------------------------------------------------------------------------
def camera(manifest):
    """world_to_clip, camera origin and forward axis from a renderer manifest."""
    w2c = np.asarray(manifest['world_to_clip'], np.float64)
    require(w2c.shape == (4, 4) and np.all(np.isfinite(w2c)), 'Renderer world_to_clip required')
    mw = (manifest.get('camera') or {}).get('matrix_world')
    if mw is not None:
        mw = np.asarray(mw, np.float64).reshape(4, 4)
        fwd, origin = -mw[:3, 2], mw[:3, 3]
    else:   # clip x, y and w vanish at the camera centre; w grows along the view axis
        origin = np.linalg.solve(w2c[[0, 1, 3], :3], -w2c[[0, 1, 3], 3])
        fwd = w2c[3, :3]
    return w2c, origin, fwd / np.linalg.norm(fwd)


def world_points(cam, depth, ys, xs, planar):
    w2c, origin, fwd = cam
    H, W = depth.shape
    inv = np.linalg.inv(w2c)
    xn = (xs.astype(np.float64) + 0.5) / W * 2 - 1
    yn = 1 - (ys.astype(np.float64) + 0.5) / H * 2
    ends = []
    for z in (-1.0, 1.0):
        p = inv @ np.stack([xn, yn, np.full_like(xn, z), np.ones_like(xn)])
        ends.append(p[:3] / p[3])
    d = ends[1] - ends[0]
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    t = depth[ys, xs].astype(np.float64)
    if planar:
        t = t / (fwd @ d)
    return origin[:, None] + d * t


def depth_convention(cam, depth, ids, bed_indices, usable):
    """'planar' or 'ray_distance': whichever makes the bed surfaces flatter (beds are near-horizontal)."""
    rng = np.random.default_rng(0)
    spread = {True: [], False: []}
    for k in bed_indices:
        ys, xs = np.nonzero((ids == k) & usable)
        if len(ys) < 2000:
            continue
        sel = rng.choice(len(ys), 1500, replace=False)
        for planar in (True, False):
            z = world_points(cam, depth, ys[sel], xs[sel], planar)[2]
            spread[planar].append(float(np.percentile(z, 90) - np.percentile(z, 10)))
    if not spread[True]:
        return 'planar'
    return 'planar' if np.median(spread[True]) <= np.median(spread[False]) else 'ray_distance'


# -- texture ------------------------------------------------------------------------------------------------------
def tileable(img):
    h, w = img.shape[:2]
    wy = np.minimum(np.arange(h), h - 1 - np.arange(h)) / (h / 2)
    wx = np.minimum(np.arange(w), w - 1 - np.arange(w)) / (w / 2)
    wgt = np.clip(np.minimum(wy[:, None], wx[None, :]) * 3, 0, 1)[..., None]
    rolled = np.roll(np.roll(img, h // 2, 0), w // 2, 1)
    return (img * wgt + rolled * (1 - wgt)).astype(np.float32)


def pyramid(img, levels=9):
    out = [img]
    for _ in range(levels):
        a = out[-1]
        h, w = a.shape[:2]
        if h < 4:
            break
        pad = cv2.copyMakeBorder(a, 2, 2, 2, 2, cv2.BORDER_WRAP)
        pad = cv2.GaussianBlur(pad, (5, 5), 1.0)[2:-2, 2:-2]
        out.append(cv2.resize(pad, (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    return out


def sample(pyr, u, v, level):
    """Sample a tileable pyramid at texture coords (level-0 px) and a fractional mip level per point."""
    level = np.clip(level, 0, len(pyr) - 1.001)
    l0 = np.floor(level).astype(int)
    f = (level - l0)[:, None]
    out = np.zeros((len(u), 3), np.float32)
    for L in np.unique(l0):
        sel = l0 == L
        acc = []
        for LL in (L, L + 1):
            t = pyr[LL]
            h, w = t.shape[:2]
            s = 2.0 ** LL
            mx = np.mod(u[sel] / s, w).astype(np.float32)
            my = np.mod(v[sel] / s, h).astype(np.float32)
            k = len(mx)
            pad = (-k) % 1024               # remap maps must stay under 32767 px a side
            mx = np.pad(mx, (0, pad)).reshape(-1, 1024)
            my = np.pad(my, (0, pad)).reshape(-1, 1024)
            acc.append(cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).reshape(-1, 3)[:k])
        out[sel] = acc[0] * (1 - f[sel]) + acc[1] * f[sel]
    return out


def value_noise(u, v, period, seed):
    """Smooth noise in world metres, 0..1."""
    g = np.random.default_rng(seed).random((64, 64)).astype(np.float64)
    x, y = u / period, v / period
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - xi, y - yi
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = g[yi % 64, xi % 64], g[yi % 64, (xi + 1) % 64]
    c, d = g[(yi + 1) % 64, xi % 64], g[(yi + 1) % 64, (xi + 1) % 64]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def steep(cam, depth, usable, planar):
    """Bed pixels on faces steeper than MAX_SLOPE (rise over run to a usable right/lower neighbour), widened 1 px."""
    H, W = depth.shape
    ys, xs = np.nonzero(usable)
    P = world_points(cam, depth, ys, xs, planar)
    out = np.zeros((H, W), bool)
    for ny, nx in ((ys, np.minimum(xs + 1, W - 1)), (np.minimum(ys + 1, H - 1), xs)):
        ok = usable[ny, nx] & ((ny != ys) | (nx != xs))
        Q = world_points(cam, depth, ny, nx, planar)
        rise = np.abs(Q[2] - P[2])
        run = np.hypot(Q[0] - P[0], Q[1] - P[1])
        bad = ok & (rise > MAX_SLOPE * np.maximum(run, 1e-6))
        out[ys[bad], xs[bad]] = True
    return cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


# -- pass ---------------------------------------------------------------------------------------------------------
def zones(plants, bed_area):
    """guard ring, hard zone, grain weight, lift weight, distance to plants."""
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GAP + 1, 2 * GAP + 1))
    guard = cv2.dilate(plants.astype(np.uint8), disk) > 0
    dist = cv2.distanceTransform((~plants).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    hard = bed_area & ~guard & (dist > GAP + SAFE)
    other = ~bed_area & ~plants
    a_edge = np.clip(cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / EDGE, 0, 1)

    def ramp(width):
        t = np.clip((dist - GAP - SAFE) / width, 0, 1)
        return t * t * (3 - 2 * t) * a_edge * hard
    return guard, hard, ramp(FEATHER), ramp(LIFT_FEATHER), dist


def decals(img8, hard, soil, dist):
    """Small bright leaf decals on the render soil, inside the hard zone, away from the plant ring."""
    lum = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
    H, W = lum.shape
    small = cv2.resize(lum, (max(W // 4, 1), max(H // 4, 1)), interpolation=cv2.INTER_AREA)
    k = min(31, (min(small.shape) // 2) * 2 - 1)
    med = small if k < 3 else cv2.medianBlur(small, k)
    med = cv2.resize(med, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    lf = lum.astype(np.float32)
    speck = soil & (dist > 3) & (lf - med > np.maximum(8, 0.25 * med))
    n, lab, st, _ = cv2.connectedComponentsWithStats(speck.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] < 6000           # decals are small; large bright areas are sun patches
    keep[np.unique(lab[speck & (dist <= GAP + SAFE + 7)])] = False   # whole leaves only, never half of one
    speck = cv2.dilate(keep[lab].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    return speck & hard


def apply(final, ids, depth, plants, bed_area, manifest, spec, base):
    """Return (image, info). Pixels outside open bed ground keep their exact input samples."""
    if not bed_area.any():
        return final, {'status': 'skipped', 'reason': 'no_bed_ground'}
    if manifest is None:
        return final, {'status': 'skipped', 'reason': 'no_renderer_camera'}
    usable = bed_area & np.isfinite(depth) & (depth > 0)
    cam = camera(manifest)
    bed_indices = [int(k) for k in np.unique(ids[bed_area])]
    convention = depth_convention(cam, depth, ids, bed_indices, usable)
    planar = convention == 'planar'
    usable &= ~steep(cam, depth, usable, planar)
    guard, hard, a_grain, a_lift, dist = zones(plants, usable)
    if hard.sum() < MIN_OPEN_PX:
        return final, {'status': 'skipped', 'reason': 'no_open_bed_ground', 'open_ground_px': int(hard.sum())}
    scale = np.iinfo(final.dtype).max / 255
    x = final.astype(np.float64) / scale                          # 8-bit units
    img8 = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    speck = decals(img8, hard, usable & ~plants, dist)
    base_img = final.copy()
    if speck.any():
        m = speck.astype(np.uint8)
        for ch in range(3):
            base_img[..., ch] = cv2.inpaint(np.ascontiguousarray(final[..., ch]), m, 7, cv2.INPAINT_TELEA)
    bx = base_img.astype(np.float64) / scale
    H, W = ids.shape
    ys, xs = np.nonzero(hard)
    P = world_points(cam, depth, ys, xs, planar)
    xs1, ys1 = np.minimum(xs + 1, W - 1), np.minimum(ys + 1, H - 1)
    Px = world_points(cam, depth, ys, xs1, planar)
    Py = world_points(cam, depth, ys1, xs, planar)
    # neighbours off the bed (other depth) fall back to the other axis' footprint
    fx = np.hypot(*(Px[:2] - P[:2]))
    fy = np.hypot(*(Py[:2] - P[:2]))
    fx = np.where(hard[ys, xs1], fx, fy)
    fy = np.where(hard[ys1, xs], fy, fx)
    foot = np.minimum(np.maximum(fx, fy), ANISO * np.minimum(fx, fy))
    foot = np.clip(foot, 1e-4, max(np.percentile(foot, 99), 1e-4))
    pyrs = []
    for rec in spec['exemplars']:
        t, _ = load(checked(rec, base))
        pyrs.append(pyramid(tileable(t.astype(np.float32) / (np.iinfo(t.dtype).max / 255))))
    tex_px = TEX_M / pyrs[0][0].shape[1]
    level = np.log2(np.maximum(foot / tex_px, 1.0))
    u, v = P[0], P[1]
    layers, weights = [], []
    for i, pyr in enumerate(pyrs):
        a = (0.0, 2.1, 4.2)[i]
        ur = (np.cos(a) * u - np.sin(a) * v) / tex_px + 977 * i
        vr = (np.sin(a) * u + np.cos(a) * v) / tex_px + 531 * i
        layers.append(sample(pyr, ur, vr, level).astype(np.float64))
        weights.append(value_noise(u, v, 0.35, 11 + i) ** 3 + 1e-3)
    wsum = np.sum(weights, 0)
    tex = np.sum([l * (w / wsum)[:, None] for l, w in zip(layers, weights)], 0)
    tl = tex @ LUMA
    # relative grain, normalised per half mip level so its mean is exactly 1 at every distance (no tonal drift)
    lv = np.round(level * 2).astype(int)
    mean_l = np.bincount(lv, weights=tl) / np.maximum(np.bincount(lv), 1)
    rel = 1 + (tl / (mean_l[lv] + 1e-6) - 1) * GRAIN
    rel *= 1 + CLUMP * (value_noise(u, v, 0.07, 41) + value_noise(u, v, 0.19, 43) - 1)
    rel = np.clip(rel, 0.2, 2.2)
    aG, aL = a_grain[ys, xs], a_lift[ys, xs]
    px = bx[ys, xs]
    bl = px @ LUMA
    L1 = bl + aL * (bl + SHADE_FLOOR * np.exp(-bl / SHADE_FLOOR) - bl)
    lum = L1 * (1 + aG * (rel - 1))
    sm = cv2.GaussianBlur(bx.astype(np.float32), (0, 0), 3)[ys, xs].astype(np.float64)
    sml = sm @ LUMA
    trust = np.clip((sml - 6) / 20, 0, 1)[:, None]
    soil = soil_chroma(spec, base)
    c_sm = sm / (sml[:, None] + 1e-3)
    fallback = soil if soil is not None else (np.median(c_sm[trust[:, 0] > 0.5], 0) if (trust > 0.5).any()
                                              else np.ones(3))
    c_x0 = c_sm * trust + fallback * (1 - trust)
    c = c_x0 if soil is None else c_x0 * (1 - CHROMA_SOIL) + soil * CHROMA_SOIL
    c = c * (1 - CHROMA_TEX) + tex / (tl[:, None] + 1e-3) * CHROMA_TEX
    c_px = px / (bl[:, None] + 1e-3)
    col = c_px * (1 - aG[:, None]) + c * aG[:, None]
    col = np.where((bl < 6)[:, None], c_x0 * (1 - aG[:, None]) + c * aG[:, None], col)
    rgb = np.clip(col * lum[:, None], 0, 255)
    w = np.maximum(aG, aL)[:, None]
    out = final.copy()
    out[speck] = base_img[speck]
    write = w[:, 0] > 0
    mixed = np.clip(np.rint((px * (1 - w) + rgb * w) * scale), 0, np.iinfo(final.dtype).max).astype(final.dtype)
    out[ys[write], xs[write]] = mixed[write]
    changed = np.any(out != final, axis=2)
    info = {'status': 'applied', 'operation': OPERATION, 'depth_convention': convention,
            'open_ground_px': int(hard.sum()), 'decal_px': int(speck.sum()), 'changed_px': int(changed.sum()),
            'footprint_mm_p10_p90': [round(float(np.percentile(foot, q)) * 1000, 2) for q in (10, 90)],
            'soil_chroma': None if soil is None else [round(float(s), 4) for s in soil],
            'guard_changed_px': int((changed & guard).sum()), 'outside_bed_changed_px': int((changed & ~usable).sum())}
    require(info['guard_changed_px'] == 0 and info['outside_bed_changed_px'] == 0,
            'Bed ground pass changed plant or non-bed pixels')
    return out, info
