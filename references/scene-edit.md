# Scene photographic edit (default photo appearance route)

The finished `photo_finish` composite keeps the designed species, roots and hardscape but still reads as rendered plants. This route makes the landscape part of that photograph an ordinary real photograph with the built-in image tool, then blends the result back only inside the landscape edit region. In the lab (experiment A1) it held species, layout and hardscape far better than any diffusion route. Image-interior refinement and the native routes stay available; when this route fails, the previous composite is delivered with disclosed limitations.

## Requirements

- The Codex session runs on **gpt-6-astra with high reasoning** (the pipeline launches it that way). Use the built-in image tool (`image_gen.imagegen`, through the imagegen skill; ChatGPT sign-in, no API key). Record both in the attempt manifest and the review; the guard rejects any other model or effort.
- Input: a view whose schema 4 `photo_finish` review passes `appearance_guard.py` (`accepted` or `completed_with_limitations`). Its composite stays immutable, and so does the original photo.
- Pixels outside the landscape edit region stay exactly the cleaned original photo. The helper asserts it.
- Never paste generated pixels into plant outlines (that leaves fake dirt patches). The helper blends whole tiles and the whole region with feathered weights.

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

The plan re-verifies the previous composite and writes `SCENE_DIR/plan.json`, `region.png` (landscape alpha and transition pixels grown by `--margin` 24 px), `region-weight.png` (raised-cosine inward feather, `--feather` 24 px), and one folder per job under `jobs/`, each with `input.png` (the 8-bit crop to edit) and `prompt.txt`:

- `full`: the whole view.
- `tile-NN`: 1152x1024 native-resolution tiles (`--tile`) with at least 30% overlap (`--overlap`), covering the edit region plus one ramp width. Tiles without region pixels are skipped.

Each prompt is the fixed A1 preamble (same camera, crop, light, plant identities, roots, sizes and overlap; hardscape fixed; bed ground shredded bark mulch when the bed-ground pass is on; plants rooted in beds, never in concrete, paving or steps; no new plants, pots, steps, paving, structures or text; no invented hybrid foliage; every flowering plant in full bloom). One line per species follows: botanical name (common name; description); flowers; `Instances: ID at [x, y]%, size [w, h]px`. Positions are % of that crop and sizes are projected pixels, computed from the renderer object IDs under the matte. Edge fragments down to one pixel are included. Prompts are deterministic, and `validate` re-derives them, so don't edit a planned prompt.

## 3. Generate

For each view, do the **full view first**, then every tile:

1. Full view: edit `jobs/full/input.png` with `jobs/full/prompt.txt` exactly. Save the returned image unchanged (about 1536x1024 for a 3:2 view).
2. Write each tile's reference: `python scripts/scene_edit_plan.py reference SCENE_DIR/plan.json --full-raw FULL_RAW.png --job tile-NN --out REF.png`. This is the matching crop of the upscaled full-view result.
3. Tile: edit `jobs/tile-NN/input.png` (the edit target, first image) with its planned prompt, attaching `REF.png` as the style and consistency reference (second image). Save the raw output unchanged (about 1330x1182). Independent tiles may run concurrently.
4. Inspect every result against its input at full size. For a second attempt, edit the **original input** again with the planned prompt plus an appended correction (`SECOND ATTEMPT: ...`) that names what went wrong: which plant, its correct leaf and flower form, and keeping its extent, crowns, roots and hardscape edges. Each job gets at most 2 attempts. Never edit a previous output.

Watch for the A1 failure modes: hybrid foliage where two species overlap; flowers turned into another type (cactus cups into daisies or pompoms); variegation in the wrong place (edges instead of the centre stripe); repeated generic leaflets; agaves merged into chains; dense cushions turned into twiggy brooms; flower masses enlarged into new focal clumps; plants or flowers on concrete, steps or paving; invented steps, pots or paving; and redrawn house or street details inside the margin.

Record everything in `ATTEMPTS.json`, with paths relative to it:

```json
{"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen",
 "jobs": {
  "full": {"attempts": [{"raw": "full-1.png", "prompt": "full-prompt-1.txt", "inspection": "what attempt 1 got right/wrong"}], "selected": 1},
  "tile-01": {"attempts": [
     {"raw": "tile-01-1.png", "prompt": "tile-01-prompt-1.txt", "reference": "tile-01-ref.png", "reference_from": 1, "inspection": "..."},
     {"raw": "tile-01-2.png", "prompt": "tile-01-prompt-2.txt", "reference": "tile-01-ref.png", "reference_from": 1}],
   "selected": 2}}}
```

List every planned job. `selected` is the attempt to use, or `null` to leave that area to the full view (or, for the full view, to the previous composite). Attempt 1 must use the planned prompt exactly. Attempt 2 must start with it, add a correction, and follow a recorded `inspection`. `reference_from` names the full-view attempt the reference was cropped from.

## 4. Compose

```text
python scripts/scene_edit_compose.py compose SCENE_DIR/plan.json --attempts ATTEMPTS.json --out OUT
```

Compose resizes each selected output back to its box (Lanczos). The aspect must match within 3%, and nothing is warped to fit. Tiles are blended with separable raised-cosine weights over their overlaps, the upscaled full view fills where tiles don't reach, and the scene goes into the photo inside the edit region through the feathered region weight, in linear light. Outside the region the result equals the previous composite and the cleaned photo exactly. Outputs: `candidate.png`, `region-weight.png`, `tile-coverage.png`, `proofs/<job>.png` (input | resized output | final crop) and `scene-edit-proof.json`. `scene_edit_compose.py verify OUT/scene-edit-proof.json` recomputes all of it.

If a tile invents or hybridizes plants after 2 attempts, set its `selected` to `null` and recompose; the full view covers it. Do the same with the full view when it redesigns the scene.

## 5. Review (schema 4 `scene_photographic_edit`)

```json
{"schema_version": 4, "method": "scene_photographic_edit", "handoff": {...}, "camera_id": "...",
 "baseline_review": {"path": "FINAL_REVIEW.json", "sha256": "..."},
 "candidate": {"path": "OUT/candidate.png", "sha256": "..."},
 "scene_edit_proof": {"path": "OUT/scene-edit-proof.json", "sha256": "..."},
 "model": {"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen"},
 "scene_review": {"reviewer": "vision-capable-agent", "source_and_final_loaded": true, "native_size_crops_loaded": true,
   "observations": "...", "paired_evidence": [{...}],
   "checks": {"species_fidelity": "pass", "no_invented_or_hybrid_plants": "pass", "plants_rooted_in_beds": "pass",
              "hardscape_geometry_preserved": "pass", "flowering_in_bloom": "pass", "no_visible_seams": "pass",
              "house_background_retained": "pass"},
   "jobs": [{"id": "tile-01", "attempts": 2, "selected_attempt": 2, "species_fidelity": "pass",
             "invented_or_hybrid_plants": "none", "observations": "...", "evidence": {"path": "OUT/proofs/tile-01.png", "sha256": "..."}}]}}
```

Load the source photo, the previous composite and the final, plus native-size crops of every tile. Judge each check from what you actually see:

- **Species fidelity:** each designed species is recognizable with its own leaf, habit and flower form.
- **No invented or hybrid plants.**
- **Rooted in beds:** no plant or flower on concrete, steps or paving.
- **Hardscape geometry:** slab outlines, steps, walls, planters and joints are unchanged.
- **Bloom:** every flowering plant is in bloom, true to its cultivar.
- **No seams:** at tile overlaps and the region feather.
- **House and background retained** inside the margin.

A little movement or extra foliage is acceptable. A changed species, a plant on hardscape or a moved structure is not. Write one job row for every planned job. Selected jobs also need `species_fidelity`, `invented_or_hybrid_plants` and a readable evidence image. Run `appearance_guard.py SCENE_REVIEW.json`:

- **All checks pass:** the status is the previous composite's (`accepted`, or `completed_with_limitations` with its limitations carried), and the scene edit composite is delivered.
- **Any check or selected job fails:** set `release_policy: fallback_previous_composite`, set `candidate` to the previous `photo_finish` candidate, and list `limitations` [{check, observations}] for exactly the nonpassing checks. The guard returns `completed_with_limitations` with `delivered: previous_photo_finish_composite`.
- **The tool was unavailable or no attempt was usable:** a fallback review without `scene_edit_proof` needs only the limitation rows, for example `scene_edit_not_run`.

Keep the plan, records, manifest, raw outputs, prompts, references, proofs and the review with the delivery even when the edit is rejected.
