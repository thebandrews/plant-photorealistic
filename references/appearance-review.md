# Appearance evidence and release

Use schema version 2 for the landscape appearance stage and schema version 3 for the required final original-photo composite; see [photo integration](photo-integration.md). A landscape-stage acceptance is not final completion. Legacy reviews retain their strict recognition contract for compatibility. The upstream schema version 1 handoff remains unchanged.

## Identity and visibility

Validated ledger/asset bindings establish species and native IDs, depth and projections establish location/occlusion. Appearance review tests changes against that authority; it does not re-identify every cultivar from a few pixels. Use source assets or other supplied views to resolve a visible appearance question. Do not claim a hidden diagnostic trait was observed.

`appearance_plan.py` writes a frozen, hashed pre-generation plan, effective editable mask, inward-feathered alpha and one row for every canonical plant. `--all` creates numbered per-view directories, each plan retaining its original camera ID. Each row contains `id`, `expected_species`, `pass_index`, `visible_pixels`, `bounds`, `root_pixel`, `has_editable_interior`, `identity_source: validated_handoff`, `evidence_path`, and a routing reason. The helper recomputes the map from actual native arrays. There is no minimum pixel count for accepting a view.

Evidence paths:

- `occluded`: zero visible native ID pixels. Inherit identity/location; do not require a crop, visible root or candidate mask.
- `native_invariants`: native material-only mode. Scene invariants and the actual candidate renderer ID pass preserve species/asset bindings, geometry, transforms, opacity and visibility. No image-recognition segmentation is required for native pixels.
- `preserved`: image mode retains the original visible specimen plus a three-pixel, nonwrapping protective margin. Exact sample comparison replaces independent recognition, including for one-pixel fragments. Larger overlapping/uncertain plants may also be deliberately preserved.
- `visual_review`: image mode permits material-interior edits. Review exposed anatomy, habit, color/variegation and bloom state only to the extent visible. Identity comes from the native map. Require an appearance-consistency finding, unchanged count/occlusion and paired crop evidence. Safe boundaries/root contacts remain protected by native-derived masks. Do not demand an unreliable new segmentation as a blanket condition.

The helper initially routes visible image plants with no safe editable interior to preservation. Other specimens are review candidates; the agent may choose preservation based on actual overlap/exposed structure. All IDs remain accounted for even when they need no independent recognition.

## Controlled compositing

Freeze `plan.json` and its permission masks **before** generation; retain their hashes in the run log/prompt record. The raw generation is a candidate only. `compose_candidate.py` reconstructs the final image by integer RGB blending through the plan's inward-feathered alpha; alpha is zero outside permitted material interiors. It preserves baseline bit depth and opacity. An 8-bit raw edit may supply color to a 16-bit baseline, but protected 16-bit samples remain exact. Raw images need matching dimensions/crop; no geometric registration, resizing or warping is performed.

Save the raw image, final image and composite proof: plan, raw, final and applied-alpha hashed references, exact operation and locally `retained_ids`. Retention can only remove permissions and restores a three-pixel margin around the actual native footprint. Never enlarge the upstream editable mask or use a new mask drawn around hallucinated output. Preserve masks and raw pixels as separate provenance, not as proof that the raw generation itself was faithful. Inspect the final composite for seams and semantic drift; numerical protection alone does not prove realism.

Minimal blending of a limited specimen is optional only within a pre-authorized, reviewable material interior. Otherwise retain it unchanged. Preservation is not deletion or permission to replace it with surrounding soil. Report retained regions within accepted views separately from whole-view fallback.

## Review JSON

Top-level fields: `schema_version: 2`, hashed `handoff`, `candidate`, `plan`, `camera_id`, immutable `baseline_sha256`, and `method` (`image_edit` or `native_material_only`). File references resolve against the JSON record containing them.

Image mode adds hashed `composite_proof`. The guard reconstructs the effective permissions, alpha and final image from the frozen plan/raw and declared retention; mismatched hashes, widened permissions, changed opacity, protected samples or output reconstruction fail.

Native mode adds hashed `native_before`, `native_after` exports from `material_snapshot.py`, plus hashed `render_proof`. The latter has `provenance: renderer_export`, `camera_id`, actual revised `native_scene_sha256`, `image_sha256`, and hashed `object_ids` for the real candidate renderer ID array. Invariants must match and bind every canonical plant's species/ID; the candidate integer ID array must match the baseline footprints/occlusion. These are native renderer records, not synthesized-pixel segmentation. Save snapshots per matching render setup when camera resolutions differ.

`recognition` contains:

- `reviewer: vision-capable-agent`, actual `baseline_and_candidate_loaded: true`, `original_pixel_crops_loaded: true`, specific `observations`, and hashed `paired_crop_evidence`.
- `geometry`: pass/fail for `wall_runs_returns`, `bed_borders`, `paved_landings`, `stairs_pavers`, `ground_contact`, `architecture`, `camera_crop`, `no_added_or_missing_plants`, `cross_view_consistency`. Inspect the full frame/camera set and changed materials; the script does not fill semantic answers.
- `plants`: one row for **every canonical ID**, including hidden plants. Each row has `id`, exact `expected_species`, `identity_source: validated_handoff`, and its `evidence_path`. Locally retained IDs use `preserved`. Native/preserved/occluded rows do not need fictional confidence, anchors or diagnostic traits.
- A `visual_review` row additionally needs `appearance_assessment: consistent`, specific `observations`, `count_assessment: unchanged`, `occlusion_assessment: consistent`, and hashed `measurement_evidence` (a paired native-size crop). If uncertain/changed, retain/correct that region instead of blocking unrelated edits.
- `image_quality: improved` and specific `quality_observations`, including whether blending/preserved regions look coherent. Byte-identical output cannot be accepted as an improvement.

`review_pairs.py --plan` creates a pending template, edited-plant crops, representative native asset crops, a whole-frame pair and a native-size changed-material sample. It does not perform recognition. For a locally retained composite, update the corresponding row to `preserved` and attach the final composite proof. Supply native proof fields in native mode.

Independent recognition measurements are optional when reliable and still enforced when supplied: `candidate_mask` must be full-size genuine image-recognition segmentation with `segmentation_source: image_recognition`; IoU >= .98, area change <= 2%, centroid shift <= 2 native pixels. Recognized `baseline_anchor`/`candidate_anchor` must differ by <= 2 pixels; if `root_visible` is true, `candidate_root` must stay within 1 pixel of the projected root. Do not call an occluded root visible, copy native IDs as recognition masks, or use these uncertainty ceilings to permit movement.

## Release and fallback

Run the guard only after actual review. It checks evidence and limits, not taxonomic truth or photographic quality itself. An accepted result includes counts for all four evidence paths. Keep original native certification distinct from a synthesized appearance derivative.

An entirely unchanged fallback uses `schema_version: 2`, hashed `handoff` and exact original-file `candidate`, `camera_id`, `baseline_sha256`, `method: unchanged_baseline`. The guard returns `status: unchanged_baseline`, `accepted_appearance: false`; this is a supported disposition, not an unsupported-method error. Record the actual failed attempt/reason. Do not convert successful intake with tiny plants into an all-blocked, zero-generation batch. Try appearance work using native invariants or planned preservation.
