# Cleanup, botanical realism and photo transitions

Use this flow for new deliveries. Schema 2 material/image-interior checks and schema 3 binary composites remain compatible, but do not bypass the finishing review below.

## Botanical geometry route

Refine the native scene when material detail cannot correct crude anatomy. Keep the baseline immutable. Work on the same permanent plant objects, exact species/cultivar and source asset binding. Improve tapered branches, plausible junctions, petioles, leaf curvature, natural spacing/orientation and species-specific sprays. Remove accidental pole-like mesh artifacts only after distinguishing them from photographed construction equipment. Preserve real species habit, flowering state, root position and intended presentation size; do not replace unusual plants with generic attractive shrubs.

For lifeless sparse specimens, add `--foliage-fill` to the planning command. Additional attached branchlets/leaves are allowed inside the existing presentation envelope of the named specimen; maintain its expected open/dense habit, size class, bloom state, root and all world transforms. Fill accidental bald patches with layered, varied orientations and plausible attachment. Give leaves volume, curl, thickness/translucency and restrained color variation; keep conifer sprays from reading as flat sheets and flower heads from reading as smooth primitives. Do not create new permanent plant objects or substitute species. Record before/after density, volume and asymmetry observations in the pilot, then apply the same shared anatomy to every camera.

Export the saved baseline using Blender `--python scripts/botanical_snapshot.py -- BEFORE.json`. Freeze the permission plan before editing:

```text
python scripts/botanical_refinement.py HANDOFF --before BEFORE.json --plants ID ID --foliage-fill --out REFINEMENT_PLAN.json
```

Join edited branch/leaf meshes into their original per-plant objects. Mesh topology and UVs may change only for the named plants. Preserve the root collar within 5 cm of the actual object root, object world transforms, IDs, species/asset/seed bindings, materials' opacity behavior, nonplant geometry, cameras and lighting. Empty root-collar exports still require visual ground-contact review; a fixed origin alone does not prove the mesh stayed planted. Export `botanical_snapshot.py` from the actual saved candidate. Film transparency may differ for foreground rendering; other snapshot settings remain fixed.

Use the same revised scene for all supplied cameras; render real candidate beauty, integer object IDs, depth and a transparent landscape layer. Do not copy old IDs onto changed crowns. Silhouettes and occlusions can change naturally; baseline IoU/centroid/pixel-count equality is inappropriate for this route. Never describe renderer IDs as AI recognition.

Schema 4 landscape review:

- `method: native_botanical_refinement`, `handoff`, `camera_id`, `candidate`, `refinement_plan`, `native_after`, `render_proof` (hashed references).
- Render proof: `provenance: renderer_export`, actual `camera_id`, saved `native_scene_sha256`, `image_sha256`, hashed actual `object_ids` and `depth`.
- `recognition`: `reviewer: vision-capable-agent`, `baseline_and_candidate_loaded: true`, `original_pixel_crops_loaded: true`, observations, readable hashed `paired_crop_evidence`, `image_quality: improved`, `quality_observations`.
- `recognition.geometry`: every existing appearance geometry check plus `species_habit`, `branch_taper_attachment`, `leaf_scale_distribution`, `presentation_size`, `root_positions`, all pass after actual inspection. Inspect the camera set together for habit, scale, occlusion and consistent shared anatomy.
- A `--foliage-fill` plan additionally requires `foliage_fullness`, `natural_asymmetry`, `leaf_volume_detail` to pass; inspect actual before/after crops rather than equating a mesh-hash change with photographic improvement.
- `recognition.plants`: one row for every canonical ID with `id`, exact `expected_species`, `identity_source: validated_handoff`. Each geometry-changed plant additionally needs `appearance_assessment: consistent`, specific observations and readable hashed `measurement_evidence` showing before/after anatomy. Hidden plants inherit native identity; do not invent visible diagnostic traits. Include clearer-view evidence for edited specimens that are hidden in the current view.

Run `appearance_guard.py REVIEW.json`. It allows only planned mesh changes; species, root collar/transforms, cameras, hardscape and other invariants must match. Material-only and frozen image-interior routes keep their schema 2 contracts unchanged.

## Photo cleanup before overlay

Inspect source overviews and detail crops, including behind proposed foliage. Inventory people, tools, wheelbarrows, ladders, cones, rubble, bags, loose construction materials, temporary wrapping/reels and associated shadows/reflections. Distinguish temporary clutter from permanent fixtures and intentional finished landscape. Never remove existing vegetation or a permanent fixture simply because it resembles clutter.

For each target create a localized binary 8-bit grayscale mask at the output camera dimensions. Cover its visible fragments and attributable shadow, leaving unrelated context protected. Inventory JSON is `{"regions":[{"id":"cleanup-1","kind":"equipment","description":"temporary wheelbarrow and shadow","mask":{"path":"mask.png","sha256":"..."}}]}`. Kinds are `person`, `tool`, `debris`, `equipment`, `temporary_staging`. Use an empty regions list only after actually finding no cleanup targets.

Use the built-in image-generation tool to edit the original photographic plate, with this inventory and frozen masks as references. Reconstruct obscured ground, fence or wall only from nearby photo evidence; keep architectural landmarks, existing vegetation, perspective and light consistent. Preserve raw edits and exact prompts including permission-plan/reference hashes. A raw full-frame generation is not the deliverable; the helper copies its samples only inside cleanup masks, restoring exact original-photo samples everywhere else. Do not solve clutter by covering it with expanded proposed planting or a context plane.

## Renderer matte and edge integration

Prepare the original photo plate/camera plan with `photo_composite.py plan` as described in [photo integration](photo-integration.md); it uses the validated baseline catalog. This supplies source/crop provenance, not the final binary composite.

Render/export the actual checked foreground as a transparent landscape layer. Hide context proxies from camera visibility while retaining their required lighting/occlusion contribution; log export overrides. Keep photographed foreground occluders properly masked and depth ordered; no generated replacement architecture. Export straight/unassociated RGB and a genuine antialiased alpha PNG, **not a blurred ID mask or an RGB beauty still contaminated by the old background**. Export an actual candidate object catalog using `export_photo_catalog.py` from the saved candidate scene. The catalog metadata (names, kinds, indices) must match the baseline. For image-interior refinement use native baseline geometry passes; for botanical refinement use fresh candidate passes.

Matte JSON: `provenance: renderer_landscape_matte`, `camera_id`, saved `native_scene_sha256`, exact checked stage `appearance_image_sha256`, `rgb_encoding: straight_srgb`, hashed `object_catalog`, `object_ids`, `depth`, `straight_rgb` and `alpha`. All arrays/images retain the camera canvas. Alpha is grayscale 8-bit; RGB may be 8/16-bit. RGB must use the same display transform as the checked beauty. For the legacy ID-support route, the selection bounds alpha to two native pixels, opaque interiors stay opaque and visible plant ID samples retain coverage.

For botanical native RGBA output, add hashed `renderer_rgba`, identical to the checked stage candidate. The helper compares straight RGB and every alpha sample to that native file, quantizing alpha upward to 8-bit so tiny nonzero fragments survive. The saved scene must have transparent film and its render proof must list `camera_only_export_overrides` for every excluded context mesh exactly once, with `name` and `visible_camera: false` or `holdout: true`. This route handles unfiltered IndexOB samples in transparent leaf-card holes without filling them or forcing an ID heuristic onto actual filtered coverage. Do not fabricate alpha or silently use a raw AI image as renderer output. Fix buried faces or dark holdout contamination using actual scene/depth evidence and a verified operation, not an ad hoc painted matte.

The helper uses checked appearance RGB in opaque interiors, renderer straight RGB at edges, and blends straight sRGB colors in linear light. It preserves plant alpha and avoids white/dark cutout fringes. It does not create a matte or infer foreground color from a contaminated beauty. Inspect thin leaves, needles, branch gaps and high-contrast edges at native size.

If coincident ground causes stippling, export actual surface projection/depth evidence as in photo integration. Optional matte `surface_projection` must be from the candidate scene/camera. The existing 0.01-unit depth recovery applies; do not include whole context planes.

Match ground material scale, hue, texture, grain, exposure and shadow softness to the photograph. Prepare an 8-bit `transition_alpha` before generating a local integration edit: zero except reviewed ground joins/contact-shadow regions, with a soft inward falloff. The helper permits native ground interiors and a maximum 12-pixel exterior contact band around nonplant landscape, excluding every plant plus a three-pixel margin. It cannot paint over plants or entire walls. Larger shadow work requires a genuine renderer shadow layer and a separately verified operation; do not stretch the permissions. Use an all-zero alpha only when joins/shadows already pass actual review.

Include the interiors of visibly artificial paving in this surface pass, not just a thin seam. Concrete needs fine aggregate at photographic scale, subtle nonrepeating wear/color variation and coherent grain/light. Add a restrained irregular scatter of thin fallen leaves, varied in orientation and scale, with plausible small contact shadows, matching the source season and adjacent vegetation. Keep paths usable and planting intact. Preserve every slab outline, joint, riser, wall and bed border; reject invented scoring lines, new cracks/borders, thick litter piles, rubble or props. Restore native samples locally if generation invents topology; fix only the failed area. Whole-wall texture changes require a separate verified material-interior operation; ground permissions do not authorize painting over walls.

Freeze the photographic plate, cleanup inventory and any regional cleanup permissions before cleanup generation. Cleanup may overlap native rendering. Freeze the combined plan below before transition generation, binding the identical cleanup inventory and masks; changing cleanup permissions requires a new plan and affected edit:

```text
python scripts/photo_finish.py plan --photo-plan PHOTO_PLAN/plan.json --landscape-review STAGE_REVIEW.json --renderer-matte MATTE.json --cleanup-inventory CLEANUP.json --transition-alpha TRANSITIONS.png --out FINISH_PLAN.json
```

Save its hash in prompts. Generate the raw photo cleanup first, then prepare the exact cleaned photo and landscape preview:

```text
python scripts/photo_finish.py preview FINISH_PLAN.json --cleanup-raw CLEANUP_RAW.png --out PREVIEW_DIR
```

This preview skips the transition edit and cannot claim acceptance. Edit `overlay-preview.png` for local ground texture/shadow integration, keeping plants and physical borders fixed. Compose after both raw edits:

```text
python scripts/photo_finish.py compose FINISH_PLAN.json --cleanup-raw CLEANUP_RAW.png --transition-raw TRANSITION_RAW.png --out FINAL_DIR
```

Omit a raw argument only when its corresponding permission mask is empty. Inputs must match dimensions; never crop/warp/resize a generation to force acceptance. The proof reconstructs cleanup, linear-alpha overlay and bounded transition blending exactly. Retain raw results and proof even when a local edit is rejected. Correct abrupt physical geometry mismatches in the scene, not by smearing pixels.

Use [native regional edits and bounded parallel work](efficient-execution.md) when full-frame dimensions are unsupported or independent views can overlap. Crop the source **before** generation with a frozen plan; the resulting generation must match that crop exactly. This differs from resizing/cropping an incorrectly sized result afterward.

## Final review (schema 4)

Top level: `schema_version: 4`, `method: photo_finish`, hashed `handoff`, `candidate`, `finish_proof`, and `camera_id`. `finish_proof` is `FINAL_DIR/finish-proof.json`.

`photo_review` requires actual `reviewer: vision-capable-agent`, `source_and_final_loaded: true`, `native_pixel_edges_loaded: true`, specific observations, readable hashed `paired_evidence`, and pass/fail `checks` for all prior photo checks plus `people_removed`, `construction_clutter_removed`, `clean_background_repairs`, `natural_plant_anatomy`, `no_cutout_fringes`, `surface_transitions`. Each planned cleanup target needs a `cleanup` row with `id`, `status: removed`, observations and hashed readable before/after `paired_evidence`. No targets means an empty cleanup list with actual whole-frame no-clutter observations, not fabricated removals.

New plans set `surface_realism_review: true` and also require `hardscape_surface_realism` and `natural_leaf_litter`. Inspect texture scale, nonrepetition, real-looking concrete, leaf placement/contact shadows and unchanged paving joints in source/native/final crops. If a view contains no visible paving/litter opportunity, record that actual finding and preserve it; do not add litter to roofs, plants or invisible surfaces simply to tick a check. Old plans remain verifiable under their original check list.

Inspect the entire final frame, not just listed masks: missed people, tools, detached shadows, ghost silhouettes and repeated/invented background patches fail review. Inspect source/cleaned/native/final comparisons, high-contrast foliage edges and long rendered-ground/photo boundaries. Check species habit, branch/leaf realism, planting positions, foreground occlusion, wall/path geometry, light direction and contacts across every camera. Automated reconstruction alone cannot pass these semantic judgments.

Run `appearance_guard.py FINAL_REVIEW.json`. New delivery completion requires accepted schema 4 `photo_finish`. Report unresolved cleanup/seams/anatomy honestly and correct the affected region; legacy binary acceptance does not waive them. Keep all client photos, masks, native assets and run logs outside the reusable public skill repository.

## Completion with disclosed limitations

Keep production completion separate from photographic acceptance. After the permitted targeted corrections, finish every recoverable camera and retain its best checked composite, comparisons, actual review and operation proof. A new schema 4 photo_finish review may set release_policy: complete_with_limitations. Give every check an actual pass, fail or estimated finding and list limitations [{check,observations}] for exactly the nonpassing checks. The guard still reconstructs all pixels and enforces native/permission/evidence checks, but returns completed_with_limitations rather than accepted when semantic findings remain unresolved. Missing review evidence, corrupt hashes and native integrity defects still need repair; the completion policy never fabricates a pass. Carry those outcomes into the gallery and final report, distinguishing expected/delivered/accepted/limited/incomplete views. An ordinary camera estimate or residual material seam does not stop the whole run.

For buried nonplant foundations exposed by hiding context, matte.context_ground_occlusion may use operation native_context_ground_depth_v1. Bind immutable baseline_object_ids/depth from the handoff, ray_distance convention, depth_tolerance exactly 0.01, the exact occluding_indices of unselected actual context ground, and recomputed occluded_samples. The helper hides only candidate nonplant landscape samples farther than that unchanged ground; every canonical plant alpha remains untouched. It verifies the actual renderer RGBA before this explicit depth-order operation, excludes hidden faces from transition permissions, and never permits architecture or arbitrary painted masks as ground occluders. Retain both actual passes and the unmodified renderer RGBA.
