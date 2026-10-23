# Original-photo integration and final release

The photograph supplies existing architecture, neighborhood, background trees and sky. The validated native scene supplies proposed planting, beds and hardscape. Preserve native species, roots and camera; botanical geometry refinement follows its separate planned route. Do not generate a replacement background. This reference covers plate preparation, bounded surface recovery and legacy schema 3 binary composites. **New deliveries use the cleanup, genuine alpha and schema 4 review in [finishing](finishing.md).**

## Evidence and preparation

Require the handoff view's hashed `source_photo`, matching camera-fit JSON (`camera_id` and `photo.sha256`), and an object catalog exported from the unchanged validated scene. Camera-fit numerical evidence is inherited; visually check the actual final alignment and occlusion. A passed upstream validation never implies that a photographic background has already been composited.

Run Blender against the validated scene with `--python scripts/export_photo_catalog.py -- CATALOG.json`. It writes actual mesh names, kinds, integer object indices and scene hash. `photo_composite.py` selects proposed landscape kinds from that catalog and reconstructs the selection from the baseline renderer ID pass. It includes every native plant, including one-pixel fragments, and excludes existing context, proxy architecture, shadow proxies and context ground planes. The latter can project across photographed walls or distant neighborhood features; do not include them simply to hide construction activity.

Prepare one frozen plan per camera:

```text
python scripts/photo_composite.py plan HANDOFF --camera CAMERA --catalog CATALOG --camera-fit FIT --out PHOTO_PLAN_DIR
```

Use `--original-provenance PROVENANCE.json` when the upstream working-photo provenance binds `original`, `working_photo` and an explicit `operations[].crop_box`. The helper verifies those hashes and crop. Otherwise use the verified working photograph. It applies only EXIF orientation, the documented crop and uniform Lanczos resampling to the original native canvas. Record enlargement when the available photo is smaller than the render. No perspective warp, generative reconstruction, exposure change or global background replacement occurs. RGB8 photograph samples map to the RGB16 output by multiplication by 257.

For a batch, Python callers may perform one full `handoff.check` before `prepare(..., validated=True)` for each view. Never skip intake validation. Plans and masks must precede the composite; reused plans must retain hashes and source/camera bindings.

## Compose and inspect

```text
python scripts/photo_composite.py compose PHOTO_PLAN/plan.json --foreground CHECKED_LANDSCAPE.png --out FINAL_DIR
```

This legacy/diagnostic composite uses the exact accepted schema 2 foreground at original dimensions. It uses a binary native-ID matte, retaining selected native samples and photographic samples elsewhere. It can reveal old-background fringes and abrupt joins; use `photo_finish.py` with actual renderer straight-alpha evidence for final delivery. Do not claim cleanup or alpha blending used this unchanged binary recipe.

Coincident native ground surfaces can interleave foreground IDs with a context ground proxy and produce a stippled photo/render transition. Resolve that from actual geometry, not by including the entire context plane. Run Blender with `--python scripts/export_surface_projection.py -- --handoff HANDOFF --surfaces SURFACE_NAME --overlap-with CONTEXT_GROUND_NAME --out PROJECTION.json`, then add `--surface-projection PROJECTION.json` to photo planning. The exporter preserves actual triangles/camera, clips triangles crossing the near plane, and records camera-Z depth convention. Verify the convention against the actual baseline depth; a ray-distance dataset needs a corresponding actual exporter convention.

The helper recovers only pixels inside those actual projected nonplant landscape triangles, currently labeled as the explicitly listed existing-ground context, whose baseline depth agrees within 0.01 native scene units. It cannot change plant selection or include buildings/sky. The guard recomputes that bounded recovery. Keep the projection and depth hashes, inspect the repaired native-size seam, and do not widen the tolerance or draw a replacement footprint if the recovery fails.

Inspect source/native/final overviews, native-pixel edges, ground contact and the camera set together. Remove people and temporary construction clutter through the bounded cleanup stage, including outside the proposed footprint. Preserve permanent photographed features and all proposed plants. Mismatched alignment, geometry-like seams or lighting discontinuity require correction or an incomplete disposition; reconstruction alone cannot establish realism.

## Schema 3 final review

Top level: `schema_version: 3`, `method: photo_composite`, hashed `handoff`, `camera_id`, hashed final `candidate`, hashed `landscape_review` (schema 2), and hashed `photo_composite_proof`.

`photo_review` contains `reviewer: vision-capable-agent`, actual `source_and_final_loaded: true`, `native_pixel_edges_loaded: true`, specific `observations`, hashed `paired_evidence`, and `checks` with pass/fail for:

- `source_photo_background`, `camera_alignment`, `landscape_placement`
- `house_neighborhood_retained`, `occlusion_edges`, `ground_contacts`
- `lighting_shadows`, `construction_transitions`, `cross_view_consistency`

`appearance_guard.py` reconstructs the original plate and native selection, verifies exact photo samples outside the overlay and exact foreground samples within it, binds the foreground to the accepted plant review, and requires actual visual findings. It returns `photo_background_composited: true` only for the final photo path. Legacy/schema 2 results remain compatible but are explicitly landscape-stage acceptances.

Paired evidence must be readable images of the actual source/native/final comparisons and native-pixel edge inspection. Hashing an unrelated text file is not photographic review evidence. Image validity and hashes do not substitute for the agent's actual inspection or independently establish the semantic content of a comparison.

Schema 3 remains verifiable for existing artifacts but cannot satisfy the new cleanup/anatomy/edge release criteria. The final gallery must show schema 4 finished photo composites and source/cleaned/native/final comparisons. Report photographic completion independently of material-stage improvement, with cleanup/refined plant regions, dimensions/resampling, background preservation outside declared edits and inherited warnings.

When an older exporter uses coarse kinds such as terrain or walkway, export a reviewed scene-bound photo role map rather than selecting every context plane or changing native kinds. Pass `--photo-roles MAP.json` after the catalog output path. The map binds native_scene, observations, hashed evidence and objects [{name,native_kind,photo_kind}], with photo_kind from LANDSCAPE_KINDS. Actual kind/name/index metadata stays intact. The checker verifies the map and its selected members; copy the same role selection to the checked candidate catalog, binding it to the actual candidate scene. Never include a building, distant context ground or neighborhood proxy merely to hide a mismatch.
