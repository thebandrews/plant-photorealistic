# Scene photographic edit (default photo appearance route)

The finished `photo_finish` composite keeps the designed species, roots and hardscape but still reads as rendered plants. This route makes the landscape part of that photograph an ordinary real photograph with the built-in image tool, then blends the result back only inside the landscape edit region. Each native-resolution tile of the composite is edited directly, with a labelled sheet of real photographs of the species in that tile as a second image. In the lab (experiments A1 and P1) tiles edited this way held species, layout and hardscape far better than any diffusion route, and far better than tiles anchored to a full-view edit (P2). In lab R1 (3 tiles x 3 arms x 2 attempts) species-form failures fell from 8 with the text inventory alone, and 7 with an added numbered label map, to 1 with one real reference photo per species (fern anatomy, cultivar banding, a mixed bed). The label map did not help, so the plan does not make one. Image-interior refinement and the native routes stay available; when this route fails, the previous composite is delivered with disclosed limitations.

## Requirements

- The Codex session runs on **gpt-6-astra with high reasoning** (the pipeline launches it that way). Use the built-in image tool (`image_gen.imagegen`, through the imagegen skill; ChatGPT sign-in, no API key). Record both in the attempt manifest and the review; the guard rejects any other model or effort.
- Input: a view whose schema 4 `photo_finish` review passes `appearance_guard.py` (`accepted` or `completed_with_limitations`). Its composite stays immutable, and so does the original photo.
- Pixels outside the landscape edit region stay exactly the cleaned original photo. The helper asserts it.
- Never paste generated pixels into plant outlines (that leaves fake dirt patches). The helper gives each tile a whole area bounded by seams through the tile overlaps, never by plant outline, and blends the whole region with feathered weights.

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

### Reference photos

Each species that is **at least 48 px on both sides** (an instance's visible bounding box, `--reference-min-px`) in some planned tile needs `reference_photos` or `reference_missing`. The plan says which species these are (it fails with their names); smaller fragments may go without. Any entry that is given is checked.

```json
"Agave parryi var. truncata": {"common_name": "artichoke agave", "description": "...", "bloom_exempt": "...",
  "reference_photos": [
    {"path": "references/agave-parryi-truncata-mobot.jpg",
     "source_page": "https://www.missouribotanicalgarden.org/PlantFinder/...",
     "image_url": "https://www.missouribotanicalgarden.org/.../agave.jpg",
     "accessed": "2026-10-31",
     "identification": "Missouri Botanical Garden, Agave parryi var. truncata",
     "cultivar_match": "exact",
     "usage_note": "mature rosette in full sun; leaf shape and colour, ignore the gravel bed"}]},
"Lomandra 'Platinum Beauty'": {"...": "...",
  "reference_missing": "no identified photo of the cultivar or species found at a garden, university, herbarium or nursery"}
```

- Find a real photograph of the plant whose identification you can cite: a botanical garden, university or extension service, herbarium, or a nursery that names the exact plant. Prefer the exact cultivar (`cultivar_match: "exact"`); otherwise use the species (`"species"`, captioned "species photo" on the sheet so cultivar colour and banding still come from the records). Prefer a whole plant in its usual habit, in flower for flowering plants, with little else in the frame. The plant library's research records (`references[].image_url` in a `pacifica-planting-renders` bundle's research JSON) are a good first source.
- Download it into the scene work folder, next to `RECORDS.json` (for example `WORK/references/`), and record `path` relative to `RECORDS.json`. The plan rejects absolute paths, paths outside that folder and paths inside the shared plant library (`PACIFICA_ASSETS_DIR`).
- Record `source_page` and `image_url` (http/https), `accessed` (YYYY-MM-DD), `identification` (who identified it, and as what), `cultivar_match` and a short `usage_note`.
- The photos are **model input only**. They stay in the scene work folder with their provenance. Never copy them into deliverables, galleries or the shared plant library.
- No usable photo: record `reference_missing` with the reason. It is not a failure; that species is described by its records text only.

When a species has several photos, its sheet shows the first `exact` one (else the first).

## 2. Freeze the plan

```text
python scripts/scene_edit_plan.py plan --finish-review FINAL_REVIEW.json --records WORK/RECORDS.json --out WORK/scene
```

The plan (schema 4) re-verifies the previous composite and writes `SCENE_DIR/plan.json`, `region.png` (landscape alpha and transition pixels grown by `--margin` 24 px), `region-weight.png` (raised-cosine inward feather, `--feather` 24 px), `structure.png` (the alignment classes, below) and one folder per tile under `jobs/`:

- `tile-NN`: 1152x1024 native-resolution tiles (`--tile`) with at least 30% overlap (`--overlap`), covering the edit region plus one ramp width. Tiles without region pixels are skipped. Each has `input.png`, its crop of the 8-bit `photo_finish` composite (the CG render composited into the photo), `prompt.txt`, and `references.jpg` when any species in it has a reference photo. **The tile edits that crop directly.** At native resolution each plant is large in its tile, which is what keeps its species: in lab P2 a full-view edit (about 1536x1024 for a 3840x2560 photo) shrank small plants to a few dozen pixels and swapped species, and tiles anchored to it inherited those errors.
- `references.jpg`: a labelled contact sheet of the tile's reference photos, one per species, each captioned with its botanical and common name. It lists the species at least `reference_min_px` (48) px on both sides in that tile that have `reference_photos`, largest visible pixel area first, at most 8, with a long edge of at most 1536 px (one image avoids tool input limits). The plan job's `references` records the threshold, each shown species (`species`, `common_name`, `pixel_area`, the hashed `photo`, `identification`, `cultivar_match`, `source_page`, `image_url`, `accessed`), the species left off (`omitted`, with reason `below_threshold`, `reference_missing` or `sheet_cap`) and the hashed `sheet`. `validate` re-derives the selection and checks the sheet against its photos.
- `structure.png` (0 outside the region): `1` plants, and bed ground within 64 px of a plant (lenient); `2` open bed ground more than 64 px from any plant (must not gain new structure); `3` hardscape, house and other photo pixels more than 48 px from any plant (strict). Plant and bed pixels come from the renderer IDs.

The tile prompt holds only what makes a real failure: this exact crop, because neighbouring tiles are edited separately and must line up; same camera, perspective and light; hardscape kept in place with identical geometry (no added, removed, moved or doubled joints, edges, steps, walls or gravel strips); bed ground shredded bark mulch when the bed-ground pass is on; every plant rooted in its bed, never growing out of concrete, paving, steps or walls; no new pots, steps, paving, structures, text or plants in empty bed ground; every plant in place as its listed species and form at its listed position and size, where natural leaf movement, fuller foliage and a natural outline are fine but a plant is never moved, removed, replaced or merged; plants cut by the tile edge stay cut; no invented hybrid foliage; every flowering plant in full bloom. One line per species follows: botanical name (common name; description); flowers; `Instances: ID at [x, y]%, size [w, h]px`. Positions are % of the tile and sizes are projected pixels, computed from the renderer object IDs under the matte. Edge fragments down to one pixel are included. When the tile has a reference sheet, the prompt calls the composite crop the first supplied image and adds: "The second supplied image is a labelled sheet of real photographs of the species in this tile, for identification only: match each listed plant's leaves, habit and flowers to its photo; do not copy the photo's composition, background or extra plants into the scene." Prompts are deterministic, and `validate` re-derives them, so don't edit a planned prompt. `--seam-feather` (default 20 px) sets the feather at tile ownership boundaries.

**Optional label map:** not built. In R1 a numbered, species-coloured outline map from the renderer IDs gave no clear gain over the text inventory (7 species-form failures against 8), and an extra image adds tool input, so the plan has no `--label-map`.

**Optional anchor (`--anchor full_view`).** The plan adds a `full` job (the whole view, `input.png` + `prompt.txt`) whose accepted result is the anchor; tiles then edit `scene_edit_plan.py tile-input` crops of the upscaled anchor with a detail-only prompt, with `geometry.png` (the composite crop) as a geometry reference. Use it only when a view's tiles disagree in a way a whole-view edit would settle and its plants are large in the full view; P2 showed it is the weak link for small plants. Anchored plans get no reference sheets. The rest of this page describes the default (`anchor: none`); the anchored route keeps the schema 2 rules (full view first, its blocking findings recorded on the attempt, stop and fall back without tiles if it fails twice).

## 3. Generate

For each tile:

1. Edit `jobs/tile-NN/input.png` with `jobs/tile-NN/prompt.txt` exactly. When the tile has `jobs/tile-NN/references.jpg`, supply it as the second image, unchanged; otherwise attach no second image. Save the returned image unchanged (about 1330x1182). Independent tiles may run concurrently.
2. Inspect every result against its input at full size, against the four blocking checks. For a second attempt, edit the **original input** again with the planned prompt plus an appended correction (`SECOND ATTEMPT: ...`) that names what went wrong and what to keep (which plant, its correct leaf and flower form, roots and hardscape edges, and that every plant and edge stays where it is in the input). Each tile gets at most 2 attempts. Never edit a previous output. A tile that still fails a blocking check gets `selected: null`; its area keeps the previous composite.

Watch for the A1 and P2 failure modes: plants or flowers on concrete, steps or paving; invented steps, pots, joints, slab grids or paving; moved or doubled hardscape edges; a plant turned into a different kind of plant (a hedgehog cactus into pads, an agave into prickly-pear pads, cactus cups into daisies, agaves into ferns); and redrawn house or street details inside the margin. Also note, as defects, hybrid foliage where two species overlap, enlarged flower masses, filled-in gaps and variegation in the wrong place.

Record everything in `ATTEMPTS.json`, with paths relative to it:

```json
{"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen",
 "jobs": {
  "tile-01": {"attempts": [
     {"raw": "tile-01-1.png", "prompt": "tile-01-prompt-1.txt", "reference": "../scene/jobs/tile-01/references.jpg",
      "inspection": "agave turned into prickly-pear pads"},
     {"raw": "tile-01-2.png", "prompt": "tile-01-prompt-2.txt", "reference": "../scene/jobs/tile-01/references.jpg",
      "inspection": "agave rosette kept; joints unchanged"}],
   "selected": 2},
  "tile-02": {"attempts": [{"raw": "tile-02-1.png", "prompt": "tile-02-prompt-1.txt", "inspection": "..."}],
   "selected": null}}}
```

List every planned job. `selected` is the attempt to use, or `null` to leave that area to the previous composite. Attempt 1 must use the planned prompt exactly. Attempt 2 must start with it, add a correction, and follow a recorded `inspection`. An attempt may record `input`; it must then be exactly the planned composite crop. Every attempt of a tile with a planned sheet records `"reference": "../scene/jobs/tile-01/references.jpg"` (the path relative to `ATTEMPTS.json`); it must be byte-identical to the sheet in the plan, and a tile without a planned sheet records no reference. Compose and verify check both bindings.

## 4. Compose

```text
python scripts/scene_edit_compose.py compose SCENE_DIR/plan.json --attempts ATTEMPTS.json --out OUT
```

Compose resizes each selected output back to its box (Lanczos). The aspect must match within 3%, and nothing is warped to fit.

- **Base:** the previous `photo_finish` composite is the base of the scene, at full precision.
- **Structure check (`composite_structure_v1`):** each selected tile is compared with its composite crop. A photographic re-render of CG replaces every leaf, mulch and concrete texture and lets plants move a little and fill out, so the check works on coarse luminance edge maps (area reduction, Gaussian sigma 1.5, Sobel magnitude), per `structure.png` class, in blocks of about 128 native px:
  - *lenient* (plants and nearby bed, at 1/8 scale, about 12 px of blur): the global edge NCC must be at least **0.20**. This only catches a tile whose plants no longer correspond to the composite at all. Plants moving a little, filling out or changing outline are defects, not failures (Ben's rule), and real Astra re-renders that kept the hardscape intact scored 0.24-0.50 here (P3), so a higher floor rejects good tiles. Shifts are caught by the strict hardscape checks; a re-laid-out planting by the visual review.
  - *open bed ground* (1/8 scale): in each block, the edge energy the tile adds where the composite has none (tile edges minus the 5 px dilated composite edges, relative to the tile's typical composite block) must be at most **0.30**. This catches a plant, pot or path invented in empty ground.
  - *strict* (hardscape and photo pixels away from plants, 1/4 scale, about 6 px of blur): global NCC at least **0.60**; every block where the composite has structure keeps NCC at least **0.30** (lines and edges stay put); added edge energy per block at most **0.45** (no new joints, slab grids or edges).

  Synthetic calibration at production scale (1152x1024 tiles; 8 garden layouts of 16 leaf-rosette plants on mulch with a jointed slab and a wall; the "photo" replaces every texture, shifts colour, moves each plant by up to sigma 12 px and adds 40% more leaves): normal re-renders scored lenient 0.76-0.97, open-ground added at most 0.16, strict global at least 0.97, strict worst block at least 0.51 and strict added at most 0.39, so none was rejected. Whole-content shifts of 24-40 px scored strict global at most 0.35; a plant added in open ground scored 0.34-0.80 added; a joint moved 24 px scored strict block at most 0.07; the slab edge moved 30 px at most 0.17; an added joint 0.49-0.61 strict added. All were rejected. Drift tolerance: 2-3 px with 0.5% zoom rejected 1 of 8 tiles, and 4 px with 1% zoom (about 10 px at the tile edge, enough to kink a joint at a seam) 7 of 8. A **single** missing, swapped or moved plant inside dense planting is not reliably measurable this way; the visual review catches it (`species_form_recognizable`, `invented_plant`).

  A tile that fails is not used, and its area keeps the previous composite. Per-tile values and the failed tests (`failed`) are recorded in `scene-edit-proof.json` under `alignment.tiles`.
- **Ownership:** each pixel belongs to exactly one tile (`ownership.png` stores the tile index + 1, and 0 where no tile covers). Neighbouring tiles in a row or column split their overlap along a **minimum-cost seam**: a path across the overlap, at least `seam_feather_px` inside both tiles, through the pixels where the two sources agree most (Gaussian-blurred absolute luminance difference, zero outside the edit region, plus a small pull toward the overlap midline). Where an owner is unused, the composite is its source, so the seam runs where the used tile agrees with the composite. Sources cross-fade only in a raised-cosine band `seam_feather_px` (20 px) wide centred on each boundary. Two tiles never blend across their wide overlaps, so a disagreement cannot turn into doubled leaves.
- **Seam metric:** along every tile-tile ownership boundary, the two neighbouring sources are compared in a band 2 x `seam_feather_px` on each side, using edge NCC at 1/4 scale (at least 64 px across the band) plus the mean absolute luminance difference. Pairs whose NCC is below 0.4 and whose sources differ by at least 2 levels on average are flagged as `likely_ghosting` under `seams.pairs`, with a `bbox` to crop. This informs the visual check; it does not replace it.
- **Region:** the scene goes into the photo inside the edit region through the feathered region weight, in linear light. Outside the region the result equals the previous composite and the cleaned photo exactly.

Outputs: `candidate.png`, `region-weight.png`, `ownership.png`, `proofs/<job>.png` (composite crop | resized output | final crop) and `scene-edit-proof.json` (`base`, `alignment`, `ownership_boundary`, `seams`, `used_tiles`). `scene_edit_compose.py verify OUT/scene-edit-proof.json` recomputes all of it, including the metrics.

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

Load the source photo, the previous composite and the final, plus native-size crops of every tile (its proof panel shows composite crop, output and final) and of every flagged seam band. Judge the four blocking checks on the **whole delivered view at native size**. Each check is `"pass"`, or a fail object with `where` (which plant, bed, edge or tile boundary) and `observation`.

`defects` is required, even when empty, and `defect_count` must equal its length. Each defect is `{type, where, observation}`, with these types:

- `movement`: a plant moved a little.
- `extent_change`: a crown or outline grew or shrank.
- `extra_foliage`: extra foliage or filled-in gaps.
- `invented_plant`: a new plant in empty bed ground.
- `cultivar_detail`: variegation placement or the exact shade.
- `bloom_missing`: a flowering plant not in bloom.
- `colour_shift`.
- `other`.

When the proof flags any seam band, set `metrics_inspected: true` after looking at those bands. Write one job row for every planned job. Each selected job needs `blocking` (`pass` or `fail`) and a readable evidence image. A selected tile marked `fail` is an error: deselect it (`selected: null`; its area keeps the previous composite) and recompose. A single tile never fails the view. Run `appearance_guard.py SCENE_REVIEW.json`:

- **All four blocking checks pass:** the scene edit composite is delivered. The status is the previous composite's (`accepted`, or `completed_with_limitations` with its limitations carried). Defects go into the guard result as `defects`, `defect_count` and `non_blocking_observations`. Any `invented_plant` or `bloom_missing` defect downgrades `accepted` to `completed_with_limitations`, because they contradict the design and the in-bloom rule, and all defects are then listed in `limitations` with `blocking: false`. The other defect types are recorded only. The result also carries `tile_alignment`, `used_tiles` and `seams_flagged`.
- **Any blocking check fails:** set `release_policy: fallback_previous_composite`, set `candidate` to the previous `photo_finish` candidate, and list `limitations` [{check, observations}] for exactly the failing blocking checks. The guard returns `completed_with_limitations` with `delivered: previous_photo_finish_composite`.
- **The tool was unavailable or no attempt was usable** (or, in the anchored route, the anchor failed): a fallback review without `scene_edit_proof` needs only the limitation rows, for example `scene_edit_not_run`.

Keep the plan, records, manifest, raw outputs, prompts, tile inputs, proofs and the review with the delivery even when the edit is rejected. Reference photos and `references.jpg` sheets stay in the scene work folder only: the delivery carries their hashes and provenance (in the plan and records), not the third-party images.

Schema 3 plans (one image per tile, no reference photos; an attempt `reference` may only be the composite crop) stay verifiable. Schema 2 plans (lab P2: always anchored, nearest-tile-centre ownership, operation `scene_edit_anchor_ownership_v2`) and their reviews stay verifiable. Reviews of schema 1 plans (tiles edited from the composite crop and blended over wide overlaps, with seven checks: `species_fidelity`, `no_invented_or_hybrid_plants`, `plants_rooted_in_beds`, `hardscape_geometry_preserved`, `flowering_in_bloom`, `no_visible_seams`, `house_background_retained`, plus per-tile species findings) stay verifiable under that original contract. New plans are always schema 4.
