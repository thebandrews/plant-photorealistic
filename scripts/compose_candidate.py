"""Apply a raw image edit only through a frozen native-derived appearance plan.

Local retries may retain additional native plant regions, never widen permissions.
Keep the raw candidate and proof; the final composite still requires visual review.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from handoff import checked, pixels, read, record, require
from appearance_plan import alpha_for, effective_mask, validate
from raster import load, save, scaled_like


def blend(before, raw, alpha):
    a = alpha.astype(np.uint32)[..., None]
    raw = scaled_like(raw, before)
    return ((before.astype(np.uint32) * (255 - a) + raw.astype(np.uint32) * a + 127) // 255).astype(before.dtype)


def expected(plan, base, raw, retained_ids=()):
    hp, _, view, ids, rows, editable = validate(plan, base)
    require(plan["method"] == "image_edit", "Compositing requires an image-edit plan")
    before, original_alpha = load(checked(view["image"], hp.parent))
    require(raw.shape == before.shape, "Raw candidate camera dimensions/crop differ; do not resize or warp")
    alpha = alpha_for(effective_mask(ids, editable, rows, retained_ids))
    return blend(before, raw, alpha), alpha, original_alpha


def compose(plan_path, raw_path, out, retained_ids=()):
    plan_path = Path(plan_path).resolve()
    raw_path = Path(raw_path).resolve()
    plan = read(plan_path)
    final, alpha, baseline_alpha = expected(plan, plan_path.parent, pixels(record(raw_path), raw_path.parent), retained_ids)
    out = Path(out).resolve()
    require(not out.exists(), "Use a new candidate output directory")
    out.mkdir(parents=True)
    save(out / "candidate.png", final, baseline_alpha)
    Image.fromarray(alpha).save(out / "applied-alpha.png")
    proof = {"schema_version": 2, "kind": "native_mask_composite",
             "plan": record(plan_path), "raw_candidate": record(raw_path),
             "candidate": record(out / "candidate.png"), "applied_alpha": record(out / "applied-alpha.png"),
             "retained_ids": sorted(set(retained_ids)), "operation": "inward_feathered_integer_rgb_blend_v1"}
    (out / "composite-proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return proof


def verify(proof, base, plan_record, candidate_record):
    require(proof.get("kind") == "native_mask_composite" and proof.get("schema_version") == 2 and
            proof.get("operation") == "inward_feathered_integer_rgb_blend_v1", "Unknown composite proof")
    pp = checked(proof["plan"], base)
    require(proof["plan"]["sha256"] == plan_record["sha256"] and
            proof["candidate"]["sha256"] == candidate_record["sha256"], "Composite plan/output binding differs")
    checked(proof["candidate"], base)
    raw = pixels(proof["raw_candidate"], base)
    final, alpha, original_alpha = expected(read(pp), pp.parent, raw, proof.get("retained_ids", []))
    require(np.array_equal(alpha, np.array(Image.open(checked(proof["applied_alpha"], base)).convert("L"))),
            "Composite alpha differs from frozen permissions/local retention")
    actual, final_alpha = load(checked(candidate_record, base))
    require(final.dtype == actual.dtype and np.array_equal(final, actual), "Candidate does not match recorded native-mask composite")
    require((original_alpha is None and final_alpha is None) or
            (original_alpha is not None and final_alpha is not None and
             np.array_equal(original_alpha, final_alpha)), "Composite opacity changed")
    return proof.get("retained_ids", [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retain", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(compose(args.plan, args.raw, args.out, args.retain), indent=2))


if __name__ == "__main__":
    main()
