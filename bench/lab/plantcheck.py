"""Per-plant placement check, for generate-and-check loops (the same test as the bench's plants gate).

Regenerate a plant, paste it into the full-size composite, and ask whether that plant still passes before
moving on, instead of finding out from the whole-image gate at the end:

    import sys; sys.path.insert(0, r'<worktree>\\bench')
    from lab.plantcheck import PlantChecker
    pc = PlantChecker('p49')                      # loads SAM and the render's reference masks once (cached)
    rows = pc.check(composite, ['REDDY-S26-004'])  # composite: HxWx3 uint8 at bench size
    rows['REDDY-S26-004']['ok'], rows['REDDY-S26-004'].get('iou'), .get('shift'), .get('height_change')

A plant passes when its SAM mask matches the render's (IoU >= 0.70, median shift <= 3% of its box diagonal,
height within 10%; small plants IoU >= 0.40). `hard` marks a hard fail (IoU < 0.60 or shift > 5%), which
disqualifies a whole run. Only plants with at least 400 visible pixels are tested.
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench as bb  # noqa: E402
import benchlib as bl  # noqa: E402
import gates as gt  # noqa: E402

_SEG = None


class _Subset:
    """A view whose plant list is limited to some plants; everything else is the view's."""

    def __init__(self, view, ids):
        self._view, self.plants = view, [p for p in view.plants if p['id'] in ids]

    def __getattr__(self, name):
        return getattr(self._view, name)


class PlantChecker:
    def __init__(self, view_id, bench_root=None):
        global _SEG
        self.bench = bl.Bench(bench_root or os.environ['PHOTOREAL_BENCH'])
        self.view = self.bench.view(view_id)
        _SEG = _SEG or bb.segmenter()
        self.seg = _SEG
        self.native = bb._native_mask_cache(self.bench, self.view, self.seg)

    @property
    def plants(self):
        """Tested plants: id, species, pass_index, box [x0, y0, x1, y1], pixels."""
        return [p for p in self.view.plants if p['pixels'] >= gt.T['plant_small_px']]

    def check(self, image, ids=None):
        """{plant id: row} for the given plants (default: all tested plants) in a full-size composite."""
        ids = set(ids) if ids else {p['id'] for p in self.plants}
        sub = _Subset(self.view, ids)
        r = gt.gate_plants(sub, image, self.seg, {k: v for k, v in self.native.items() if k in ids})
        hard = set(r['hard_fails'])
        return {row['id']: dict(row, hard=row['id'] in hard) for row in r['rows']}
