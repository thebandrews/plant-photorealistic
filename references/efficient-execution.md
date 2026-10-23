# Efficient production

Reduce repeated work before increasing concurrency. Record rough elapsed time for intake, scene refinement/rendering, photo edits and final checks so the next run can address the real bottleneck. These are production decisions, not relaxed acceptance rules.

## Resolve quality before the full render queue

Inspect a contact sheet and select a representative view plus native-size crops of the worst sparse foliage, close branches and long paving/photo join. Build a fast preview of the proposed planting over the original photograph, including a genuine alpha sample. Test foliage fullness, real concrete/leaf litter and a difficult cleanup region here. Reduced-resolution or border renders are previews only, never certified final-camera evidence. Keep any temporary preview settings out of the final saved scene and passes.

If foliage is still a flat curtain or paving is still a smooth generated plane, fix that cause before queuing every full-resolution camera. A small taper or shader adjustment does not meet the appearance target just because native invariants pass. Inspect the pilot after a targeted correction; use the skill's correction limit and report residual failures honestly.

Refine a shared species asset once and keep its instance identities/transforms. Reuse the shared result across supplied cameras. Export baseline/candidate snapshots once per actual scene revision, then render beauty, IDs, depth and transparent foreground together where the renderer supports it. Do not rerender unchanged views for photo-only corrections. Reuse an existing photo-fit plan only after checking its source/crop/proof hashes; final guards still verify current inputs.

## Overlap independent stages

After source photos, cleanup inventories and native crop permissions are frozen, dispatch independent built-in image-generation calls in small bounded batches (start with 2–4). Each call gets one view or region and its exact prompt/reference hashes. Use each result as it completes; persist its record immediately, including failures. Do not make a whole batch's records depend on its slowest call, or lose completed work when one call fails.

Photo cleanup can run while the renderer works. CPU mask preparation/comparison generation can run beside the render queue when memory permits. Locally, start with one render worker on a single GPU; increase only after measuring memory and throughput. With the `runpod` render backend (`scripts/blender_run.py`, see SKILL.md), each camera render gets its own remote GPU, so up to three independent camera renders may run at once. More concurrent full scenes can slow the run or exhaust memory. Shared-scene mutations and dependent steps stay ordered: freeze plan → edit/render → check stage → compose → review. Concurrency never authorizes modifying shared outputs or bypassing validation.

Use `appearance_plan.py --all` for a camera set. Compose/verification helpers already invoke their upstream checks: avoid adding redundant manual guard calls around every preview, compose and final review. Do not monkeypatch validators or reuse an old pass merely because file names match. Completed immutable stage outputs can be resumed when their input, plan, tool/code and output hashes still match; dependent outputs become stale after a change. Keep a compact per-stage status/hash manifest rather than repeatedly rebuilding reports, ZIPs or contact sheets during iteration. Package once after final review.

## Native-size regional edits

Do not request an unsupported 4K image repeatedly or resize a smaller returned image. Prefer a few useful native-pixel crops, sized for the image tool's actual supported dimensions, with enough surrounding context and margin around the complete object. Check full heads, carried tools, shadows and reflected flags before freezing masks. Inspect every input crop before editing it.

Use `regional_edit.py` to avoid writing client-specific crop assembly code:

```text
python scripts/regional_edit.py plan --source PHOTO_OR_OVERLAY.png --permission FROZEN_MASK.png --box X0 Y0 X1 Y1 --feather 32 --out REGION_PLAN
# Generate REGION_PLAN/target.png with the built-in image tool, exact dimensions.
python scripts/regional_edit.py stitch --region REGION_PLAN/plan.json RAW.png --region OTHER_PLAN/plan.json OTHER_RAW.png --out STITCHED
python scripts/regional_edit.py verify STITCHED/stitch-proof.json
```

Save plan hashes in prompts. The helper checks each target against its exact source crop, rejects mismatched dimensions/alpha, preserves 16-bit source samples outside permissions, and records ordered overlap plus inward crop feathering. Regional assembly uses nonzero mask membership; final `photo_finish.py` applies grayscale transition strength once. It never resizes or warps. Use the resulting full-canvas `raw.png` as the cleanup or transition input to `photo_finish.py`. The regional stitch and final finishing proof are both retained; the stitch is not a substitute for final acceptance. Permissions may be narrowed to retain source pixels after a bad edit; wider permissions need a new pre-generation plan and edit.

For surface edits, use the exact cleaned-photo overlay preview and the frozen nonplant transition mask. Group nearby soil and sidewalk joins into one useful crop instead of one call per plant or tiny seam. Keep plants, all physical boundaries and the photographic background protected. Do not generate another surface edit when the current one already passes.

## Helper costs

Install `requirements.txt` into the helper runtime once. `raster.py` uses native PNG decoding with exact 8/16-bit samples, with a portable fallback; it does not truncate RGB16 through Pillow. Pixel visibility is counted once per ID pass, invariant comparison avoids copying entire scene exports, and linear-light blending uses bounded row blocks. These optimizations preserve the checks and pixel arithmetic. Compare actual inputs/outputs and use meaningful regression tests; do not turn a performance cache into new placement authority.

Cover each contiguous visibly artificial material surface, rather than stopping a texture edit halfway across a slab or bed. Tile with overlap exceeding twice the inward crop feather; inspect the union against native material permissions and record any deliberately retained baseline area. Adjacent crops that merely touch leave an unchanged band. Test difficult surface joins in the pilot and reuse only hash-bound unaffected regions after a local correction.

For a hard-edged cleanup patch, regional_edit.cleanup_edges(STITCH_PROOF, INVENTORY, OUT, feather=32) can narrow the generated changes inward inside the existing permission. It preserves every original photograph sample outside that inventory and records the stitch, source, inventory, alpha and candidate hashes. A visually traced narrower inventory may restore source pixels after a failed patch; never enlarge permissions using this operation. Keep the original generations and stitch proof, inspect reconstructed surfaces and report partial cleanup honestly.
