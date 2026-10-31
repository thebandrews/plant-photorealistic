# Scene photographic edit (default photo appearance route)

The finished `photo_finish` composite keeps the designed species, roots and hardscape but still reads as rendered plants. This route makes the landscape part of that photograph an ordinary real photograph with the built-in image tool, then blends the result back only inside the landscape edit region. In the lab (experiment A1) it held species, layout and hardscape far better than any diffusion route. Image-interior refinement and the native routes stay available; when this route fails, the previous composite is delivered with disclosed limitations.

## Requirements

- The Codex session runs on **gpt-6-astra with high reasoning** (the pipeline launches it that way). Use the built-in image tool (`image_gen.imagegen`, through the imagegen skill; ChatGPT sign-in, no API key). Record both in the attempt manifest and the review; the guard rejects any other model or effort.
- Input: a view whose schema 4 `photo_finish` review passes `appearance_guard.py` (`accepted` or `completed_with_limitations`). Its composite stays immutable, and so does the original photo.
- Pixels outside the landscape edit region stay exactly the cleaned original photo. The helper asserts it.
- Never paste generated pixels into plant outlines (that leaves fake dirt patches). The helper gives each tile a whole area chosen by tile centre, never by plant outline, and blends the whole region with feathered weights.

## What counts as a failure

Only real failures block delivery:

1. **Plants growing out of hardscape:** a plant, foliage base or flower growing out of concrete, paving, steps, walls or other hardscape. Foliage overhanging an edge from a plant rooted in its bed is fine.
2. **Changed hardscape:** slab outlines, joints, steps, walls, planters, edging, driveway, house or other structures with changed geometry. Making materials look more photographic is fine. Added or removed joints, new gravel strips, and moved or doubled edges fail.
3. **Visible seams or double images:** seams, halos at the edit region, or ghosted/doubled leaves, spikes, flowers or edges.
4. **Species changed beyond recognition:** a plant no longer reads as its own form, or reads as a different kind of plant (a mound pine turned into grass tufts, a cactus flower turned into a daisy, an agave turned into a fern). Changes in size, outline, density, a little position, extra foliage, flower count or cultivar detail (where the variegation sits, the exact colour shade) are not species failures.

Everything else is a **non-blocking defect**. Record it, because fewer defects is better and runs are compared on them, but it never stops delivery on its own. Iterate on the prompts and the attempts to reduce them.

## 1. Plant records

Write `RECORDS.json` from the plant ledger, planting notes (design intent, bloom statement and exemptions) and the library assets' research. Give one entry per species visible in the view:

```json
{"species": {
  "Agave parryi var. truncata": {"common_name": "artichoke agave",
    "description": "compact round rosette of broad short powder blue-grey fleshy leaves with dark terminal points",
    "bloom_exempt": "monocarpic agave; flowering ends the rosette"},
  "Armeria maritima 'Dusseldorf Pride'": {"common_name": "sea thrift",
    "description": "low dense tuft of narrow grass-like green leaves",
    "flowers": "wine-red pompom flower heads on short wiry stems above the grassy tuft"}},
 "hardscape": "concrete, steps, walls, corten planters and house",
 "materials": "realistic concrete texture and weathered corten"}
```

Each entry needs `common_name`, a plain-words `description` of habit and leaves, and exactly one of `flowers` (how the cultivar looks in full bloom: colour, form, how the flowers are carried) or `bloom_exempt` (the reason recorded in the planting notes). `hardscape` and `materials` are optional project wording. Describe what the plant looks like; don't repeat the botanical name.

## 2. Freeze the plan

```text
python scripts/scene_edit_plan.py plan --finish-review FINAL_REVIEW.json --records RECORDS.json --out SCENE_DIR
```

The plan (schema 2) re-verifies the previous composite and writes `SCENE_DIR/plan.json`, `region.png` (landscape alpha and transition pixels grown by `--margin` 24 px), `region-weight.png` (raised-cosine inward feather, `--feather` 24 px), and one folder per job under `jobs/`:

- `full`: the whole view, with `input.png` (the 8-bit composite to edit) and `prompt.txt`. This edit is the **anchor**: it decides the layout and every silhouette.
- `tile-NN`: 1152x1024 native-resolution tiles (`--tile`) with at least 30% overlap (`--overlap`), covering the edit region plus one ramp width. Tiles without region pixels are skipped. Each has `geometry.png` (the composite crop, a secondary geometry reference only) and `prompt.txt`. Its edit target is written later from the accepted anchor.

The full-view prompt holds only what makes a real failure: same camera, crop, perspective and light; hardscape kept in place with identical geometry (no added, removed, moved or doubled joints, edges, steps, walls or gravel strips); bed ground shredded bark mulch when the bed-ground pass is on; every plant rooted in its bed, never growing out of concrete, paving, steps or walls; no new pots, steps, paving, structures, text or plants in empty bed ground; each plant clearly its listed species and form at about its listed position and size, where natural movement, fuller foliage and a natural outline are fine; no invented hybrid foliage; every flowering plant in full bloom. The tile prompt asks only for native-resolution photographic detail over the anchor crop, without changing any shape, outline, position, size or species, so neighbouring tiles line up. One line per species follows: botanical name (common name; description); flowers; `Instances: ID at [x, y]%, size [w, h]px`. Positions are % of that crop and sizes are projected pixels, computed from the renderer object IDs under the matte. Edge fragments down to one pixel are included. Prompts are deterministic, and `validate` re-derives them, so don't edit a planned prompt. `--seam-feather` (default 20 px) sets the feather at tile ownership boundaries.

## 3. Generate

For each view, do the **full view first**, then the tiles:

1. **Anchor:** edit `jobs/full/input.png` with `jobs/full/prompt.txt` exactly. Save the returned image unchanged (about 1536x1024 for a 3:2 view). Inspect it at full size against the four blocking checks and record `blocking` on that attempt. If it fails one, make attempt 2 (see below). The selected full-view attempt must pass all four. If neither attempt passes, **stop**: don't edit tiles, and deliver the previous composite through a fallback review without a proof (one limitation row per failing check).
2. Write each tile's edit target: `python scripts/scene_edit_plan.py tile-input SCENE_DIR/plan.json --full-raw FULL_RAW.png --job tile-NN --out INPUT.png`. This is the matching crop of the selected anchor, upscaled to the photo (it looks soft).
3. **Tile:** edit `INPUT.png` (the first image) with its planned prompt, attaching `jobs/tile-NN/geometry.png` as the second image (geometry and plant identity only). Save the raw output unchanged (about 1330x1182). Independent tiles may run concurrently.
4. Inspect every result against its input at full size. For a second attempt, edit the **original input** again with the planned prompt plus an appended correction (`SECOND ATTEMPT: ...`) that names what went wrong and what to keep (which plant, its correct leaf and flower form, roots and hardscape edges; for a tile, "keep every outline exactly as in the first image"). Each job gets at most 2 attempts. Never edit a previous output. A tile that still fails a blocking check gets `selected: null`; its area keeps the anchor.

Watch for the A1 failure modes: plants or flowers on concrete, steps or paving; invented steps, pots, joints or paving; moved or doubled hardscape edges; a plant turned into a different kind of plant (cactus cups into daisies, agaves into ferns); and redrawn house or street details inside the margin. Also note, as defects, hybrid foliage where two species overlap, enlarged flower masses, filled-in gaps and variegation in the wrong place.

Record everything in `ATTEMPTS.json`, with paths relative to it:

```json
{"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen",
 "jobs": {
  "full": {"attempts": [
     {"raw": "full-1.png", "prompt": "full-prompt-1.txt", "inspection": "what attempt 1 got right/wrong",
      "blocking": {"plants_rooted_in_beds": "pass", "hardscape_unchanged": "fail",
                   "no_seams_or_double_images": "pass", "species_form_recognizable": "pass"}},
     {"raw": "full-2.png", "prompt": "full-prompt-2.txt",
      "blocking": {"plants_rooted_in_beds": "pass", "hardscape_unchanged": "pass",
                   "no_seams_or_double_images": "pass", "species_form_recognizable": "pass"}}],
   "selected": 2},
  "tile-01": {"attempts": [
     {"raw": "tile-01-1.png", "prompt": "tile-01-prompt-1.txt", "input": "tile-01-input.png", "anchor_from": 2, "inspection": "..."}],
   "selected": 1}}}
```

List every planned job. `selected` is the attempt to use, or `null` to leave that area to the anchor. Attempt 1 must use the planned prompt exactly. Attempt 2 must start with it, add a correction, and follow a recorded `inspection`. `anchor_from` must name the selected full-view attempt, and `input` must be exactly its `tile-input` crop.

## 4. Compose

```text
python scripts/scene_edit_compose.py compose SCENE_DIR/plan.json --attempts ATTEMPTS.json --out OUT
```

Compose resizes each selected output back to its box (Lanczos). The aspect must match within 3%, and nothing is warped to fit.

- **Anchor:** the selected full view, upscaled to the photo, is the base of the scene.
- **Alignment check:** each selected tile is compared with the same crop of the upscaled anchor. Both are converted to luminance, reduced to the anchor's own resolution (at least 64 px on the short side), blurred (sigma 1) and turned into Sobel edge maps. The check computes their normalized cross-correlation over the tile's edit-region pixels, for the whole tile and in blocks of about 48 px. A tile is aligned when the global NCC is at least **0.55** and at most **10%** of its structured blocks score below **0.4**. In synthetic calibration at production scale, re-detailing with 2-6 px silhouette jitter scored 0.71-0.93, silhouettes moved 12-30 px scored 0.20-0.46, and a 25 px shift of one quarter of the tile passed the global test but failed the block test. A tile that fails is not used, and its area takes the anchor pixels. Per-tile values are recorded in `scene-edit-proof.json` under `alignment.tiles`.
- **Ownership:** each pixel belongs to exactly one tile, the one whose centre is nearest among the tiles covering it (`ownership.png` stores the tile index + 1, and 0 where no tile covers). Where an owner is unused, the anchor supplies its pixels. Sources cross-fade only in a raised-cosine band `seam_feather_px` (20 px) wide centred on each ownership boundary. Two tiles never blend across their wide overlaps, so a disagreement cannot turn into doubled leaves.
- **Seam metric:** along every tile-tile ownership boundary, the two neighbouring sources are compared in a band 2 x `seam_feather_px` on each side, using the same edge NCC plus the mean absolute luminance difference. Pairs below 0.4 are flagged as `likely_ghosting` under `seams.pairs`, with a `bbox` to crop. This informs the visual check; it does not replace it.
- **Region:** the scene goes into the photo inside the edit region through the feathered region weight, in linear light. Outside the region the result equals the previous composite and the cleaned photo exactly.

Outputs: `candidate.png`, `region-weight.png`, `ownership.png`, `proofs/<job>.png` (input, which for tiles is the anchor crop | resized output | final crop) and `scene-edit-proof.json`. `scene_edit_compose.py verify OUT/scene-edit-proof.json` recomputes all of it, including the metrics.

## 5. Review (schema 4 `scene_photographic_edit`, check set `real_failures_v1`)

```json
{"schema_version": 4, "method": "scene_photographic_edit", "handoff": {...}, "camera_id": "...",
 "baseline_review": {"path": "FINAL_REVIEW.json", "sha256": "..."},
 "candidate": {"path": "OUT/candidate.png", "sha256": "..."},
 "scene_edit_proof": {"path": "OUT/scene-edit-proof.json", "sha256": "..."},
 "model": {"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen"},
 "scene_review": {"reviewer": "vision-capable-agent", "source_and_final_loaded": true, "native_size_crops_loaded": true,
   "observations": "...", "paired_evidence": [{...}], "check_set": "real_failures_v1",
   "checks": {"plants_rooted_in_beds": "pass",
              "hardscape_unchanged": "pass",
              "no_seams_or_double_images": {"result": "fail", "where": "tile-05/06 boundary, lower bed", "observation": "doubled agave spikes"},
              "species_form_recognizable": "pass"},
   "defects": [{"type": "extent_change", "where": "left agave", "observation": "crown about 10% wider"},
               {"type": "cultivar_detail", "where": "fern, lower right", "observation": "cream stripe near the leaf edge, not the centre"}],
   "defect_count": 2, "metrics_inspected": true,
   "jobs": [{"id": "tile-01", "attempts": 1, "selected_attempt": 1, "blocking": "pass", "observations": "...",
             "evidence": {"path": "OUT/proofs/tile-01.png", "sha256": "..."}}]}}
```

Load the source photo, the previous composite and the final, plus native-size crops of every tile and of every flagged seam band. Judge the four blocking checks on the **whole delivered view at native size**. Each check is `"pass"`, or a fail object with `where` (which plant, bed, edge or tile boundary) and `observation`.

`defects` is required, even when empty, and `defect_count` must equal its length. Each defect is `{type, where, observation}`, with these types:

- `movement`: a plant moved a little.
- `extent_change`: a crown or outline grew or shrank.
- `extra_foliage`: extra foliage or filled-in gaps.
- `invented_plant`: a new plant in empty bed ground.
- `cultivar_detail`: variegation placement or the exact shade.
- `bloom_missing`: a flowering plant not in bloom.
- `colour_shift`.
- `other`.

When the proof flags any seam band, set `metrics_inspected: true` after looking at those bands. Write one job row for every planned job. Each selected job needs `blocking` (`pass` or `fail`) and a readable evidence image. A selected tile marked `fail` is an error: deselect it (`selected: null`; its area takes the anchor) and recompose. A single tile never fails the view. Run `appearance_guard.py SCENE_REVIEW.json`:

- **All four blocking checks pass:** the scene edit composite is delivered. The status is the previous composite's (`accepted`, or `completed_with_limitations` with its limitations carried). Defects go into the guard result as `defects`, `defect_count` and `non_blocking_observations`. Any `invented_plant` or `bloom_missing` defect downgrades `accepted` to `completed_with_limitations`, because they contradict the design and the in-bloom rule, and all defects are then listed in `limitations` with `blocking: false`. The other defect types are recorded only. The result also carries `tile_alignment`, `used_tiles` and `seams_flagged`.
- **Any blocking check fails:** set `release_policy: fallback_previous_composite`, set `candidate` to the previous `photo_finish` candidate, and list `limitations` [{check, observations}] for exactly the failing blocking checks. The guard returns `completed_with_limitations` with `delivered: previous_photo_finish_composite`.
- **The tool was unavailable, the anchor failed, or no attempt was usable:** a fallback review without `scene_edit_proof` needs only the limitation rows, for example `scene_edit_not_run` or the anchor's failing blocking checks.

Keep the plan, records, manifest, raw outputs, prompts, tile inputs, proofs and the review with the delivery even when the edit is rejected.

Reviews of schema 1 plans (tiles edited from the composite crop and blended over wide overlaps, with seven checks: `species_fidelity`, `no_invented_or_hybrid_plants`, `plants_rooted_in_beds`, `hardscape_geometry_preserved`, `flowering_in_bloom`, `no_visible_seams`, `house_background_retained`, plus per-tile species findings) stay verifiable under that original contract. New plans are always schema 2.
