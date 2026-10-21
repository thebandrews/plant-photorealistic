"""Prepare a native-ID plant map and editing permissions BEFORE generation.

This is routing/provenance, not botanical recognition. No pixel-count cutoff is
used: fragments without a safe editable interior are preserved automatically.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from handoff import check, checked, read, record, require


def expand(mask, margin=3):
    """Dilate without wrapping image edges."""
    result = mask.copy()
    for _ in range(margin):
        old = result.copy()
        result[1:] |= old[:-1]
        result[:-1] |= old[1:]
        result[:, 1:] |= old[:, :-1]
        result[:, :-1] |= old[:, 1:]
    return result


def alpha_for(editable):
    # Feather inward only. Protected pixels always get zero weight.
    image = Image.fromarray(editable.astype("uint8") * 255)
    alpha = np.array(image.filter(ImageFilter.GaussianBlur(2)))
    alpha[~editable] = 0
    return alpha


def view_data(handoff_path, camera_id, _validated_data=None):
    hp = Path(handoff_path).resolve()
    data = read(hp) if _validated_data is None else _validated_data
    if _validated_data is None:
        check(data, hp.parent)
    views = [v for v in data["views"] if v["camera_id"] == camera_id]
    require(len(views) == 1, "Unknown camera")
    view = views[0]
    ids = np.load(checked(view["object_ids"], hp.parent), allow_pickle=False)
    projection = read(checked(view["projected_plants"], hp.parent))
    editable = np.array(Image.open(checked(view["editable_mask"], hp.parent)).convert("L")) > 0
    return hp, data, view, ids, projection, editable


def mapped_plants(ids, projected, editable):
    """Compute bounds in one image traversal, including every canonical ID."""
    yy, xx = np.nonzero(ids > 0)
    values = ids[yy, xx]
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    bounds = np.zeros((len(unique), 4), dtype=np.int64)
    bounds[:, :2] = np.iinfo(np.int64).max
    np.minimum.at(bounds[:, 0], inverse, xx)
    np.minimum.at(bounds[:, 1], inverse, yy)
    np.maximum.at(bounds[:, 2], inverse, xx + 1)
    np.maximum.at(bounds[:, 3], inverse, yy + 1)
    lookup = {int(n): (int(counts[i]), bounds[i].tolist()) for i, n in enumerate(unique)}
    interior_ids = set(int(n) for n in np.unique(ids[editable]))
    result = []
    for p in projected["plants"]:
        count, box = lookup.get(p["pass_index"], (0, None))
        result.append({"id": p["id"], "expected_species": p["species"],
                       "pass_index": p["pass_index"], "visible_pixels": count,
                       "bounds": box, "root_pixel": p.get("root_pixel"),
                       "has_editable_interior": count > 0 and p["pass_index"] in interior_ids})
    return result


def effective_mask(ids, editable, plants, retained_ids=()):
    retained = set(retained_ids)
    names = {p["id"] for p in plants if p["visible_pixels"] > 0}
    require(retained <= names, "Retained regions must name visible native plants")
    indices = [p["pass_index"] for p in plants
               if p["evidence_path"] == "preserved" or p["id"] in retained]
    if indices:
        editable = editable & ~expand(np.isin(ids, indices))
    return editable


def prepare(handoff, camera_id, out, method="image_edit", preserve_ids=(), _validated_data=None):
    require(method in ("image_edit", "native_material_only"), "Unknown appearance method")
    hp, _, v, ids, projection, editable = view_data(handoff, camera_id, _validated_data)
    rows = mapped_plants(ids, projection, editable)
    require(set(preserve_ids) <= {p["id"] for p in rows}, "Unknown preserve ID")
    for p in rows:
        if not p["visible_pixels"]:
            route, reason = "occluded", "No native visible pixels; identity inherited from validated scene."
        elif method == "native_material_only":
            route, reason = "native_invariants", "Verify unchanged native geometry, asset binding and opacity."
        elif p["id"] in preserve_ids or not p["has_editable_interior"]:
            route, reason = "preserved", "Keep native pixels and margin; no independent species recognition required."
        else:
            route, reason = "visual_review", "Review changed material appearance using native identity/location."
        p.update(evidence_path=route, reason=reason, identity_source="validated_handoff")
    out = Path(out).resolve()
    require(not out.exists(), "Use a new pre-generation plan directory")
    out.mkdir(parents=True)
    mask = effective_mask(ids, editable, rows)
    Image.fromarray(mask.astype("uint8") * 255).save(out / "editable.png")
    Image.fromarray(alpha_for(mask)).save(out / "alpha.png")
    plan = {"schema_version": 2, "kind": "appearance_pre_generation_plan",
            "handoff": record(hp), "camera_id": camera_id,
            "baseline_sha256": v["image"]["sha256"], "method": method,
            "dimensions": v["dimensions"], "preservation_margin_px": 3,
            "identity_source": "validated_handoff", "plants": rows,
            "editable_mask": record(out / "editable.png"), "edit_alpha": record(out / "alpha.png")}
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def prepare_all(handoff, out, method="image_edit", preserve_ids=()):
    """Validate the whole immutable handoff once, then prepare every camera."""
    hp = Path(handoff).resolve()
    data = read(hp)
    check(data, hp.parent)
    out = Path(out).resolve()
    require(not out.exists(), "Use a new batch plan directory")
    out.mkdir(parents=True)
    # Numbered directories avoid treating arbitrary camera labels as paths.
    return [prepare(hp, view['camera_id'], out / f'view-{i:03d}', method,
                    preserve_ids, _validated_data=data)
            for i, view in enumerate(data['views'])]


def validate(plan, base):
    require(plan.get("schema_version") == 2 and plan.get("kind") == "appearance_pre_generation_plan",
            "Version 2 pre-generation plan required")
    hp, data, v, ids, projection, editable = view_data(checked(plan["handoff"], base), plan["camera_id"])
    require(plan.get("baseline_sha256") == v["image"]["sha256"] and
            plan.get("dimensions") == v["dimensions"], "Plan baseline/camera binding differs")
    require(plan.get("method") in ("image_edit", "native_material_only") and
            plan.get("preservation_margin_px") == 3, "Invalid plan method or preservation margin")
    expected = mapped_plants(ids, projection, editable)
    rows = plan.get("plants", [])
    require(len(rows) == len(expected) and len({p["id"] for p in rows}) == len(rows),
            "Plan must include every native plant exactly once")
    lookup = {p["id"]: p for p in rows}
    for p in expected:
        row = lookup.get(p["id"], {})
        require(all(row.get(k) == val for k, val in p.items()), "Plan native plant map differs")
        require(row.get("reason") and row.get("identity_source") == "validated_handoff", "Plan evidence source/reason required")
        if not p["visible_pixels"]:
            allowed = {"occluded"}
        elif plan["method"] == "native_material_only":
            allowed = {"native_invariants"}
        elif not p["has_editable_interior"]:
            allowed = {"preserved"}
        else:
            allowed = {"preserved", "visual_review"}
        require(row.get("evidence_path") in allowed, "Plant evidence path conflicts with native visibility/method")
    mask = effective_mask(ids, editable, rows)
    actual_mask = np.array(Image.open(checked(plan["editable_mask"], base)).convert("L")) > 0
    alpha = np.array(Image.open(checked(plan["edit_alpha"], base)).convert("L"))
    require(np.array_equal(mask, actual_mask) and np.array_equal(alpha, alpha_for(mask)),
            "Plan permissions differ from native material interiors/preserved regions")
    return hp, data, v, ids, rows, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--camera")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["image_edit", "native_material_only"], default="image_edit")
    parser.add_argument("--preserve", nargs="*", default=[])
    args = parser.parse_args()
    plans = prepare_all(args.handoff, args.out, args.method, args.preserve) if args.all else [prepare(args.handoff, args.camera, args.out, args.method, args.preserve)]
    summary = []
    for plan in plans:
        routes = {route: sum(p['evidence_path'] == route for p in plan['plants'])
                  for route in {p['evidence_path'] for p in plan['plants']}}
        summary.append({'camera_id': plan['camera_id'], 'plants': len(plan['plants']), 'routes': routes})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
