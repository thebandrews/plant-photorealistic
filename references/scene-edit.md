# Scene photographic edit (default photo appearance route)

The finished `photo_finish` composite keeps the designed species, roots and hardscape but still reads as rendered plants. This route is **one final pass over the entire render**: the built-in image tool edits the whole composite into an ordinary real photograph, and the result is blended back inside the landscape edit region. There are no tiles and no partial results. The pass never fails a render and never falls back: its result is always delivered, with what the review saw recorded beside it.

Whether a method is good enough is decided per experiment, not per render. Run it on the bench, put each view's result beside the baseline (`compare.jpg`, full view, previous composite | scene edit) and adopt it only if it looks better. If it doesn't, change the method and run again.

History: tiled routes (lab P1-P4, plan schemas 1-4) were removed. Independent tiles blended across overlaps doubled leaves (P1). Tiles anchored to a low-resolution full view inherited its species swaps (P2). Strict per-tile gates dropped every tile (P3), and with looser gates the delivered views mixed photographic and rendered patches (P4). Real reference photos per species cut species-form failures from 8 to 1 (R1), so the full-view pass keeps them. The full-view pass (P5) is the production baseline for realness; the numbered plant map and the defect-driven second attempt aim to cut its defects (missing bloom, invented plants, species-form drift) without losing that realness.

## Requirements

- The Codex session runs on **gpt-6-astra with high reasoning** (the pipeline launches it that way). Use the built-in image tool (`image_gen.imagegen`, through the imagegen skill; ChatGPT sign-in, no API key). Record both in the attempt manifest and the review; the guard rejects any other model or effort.
- Input: a view whose schema 4 `photo_finish` review passes `appearance_guard.py` (`accepted` or `completed_with_limitations`). Its composite stays immutable, and so does the original photo.
- Pixels outside the landscape edit region stay exactly the cleaned original photo (house, sky, street, neighbourhood). The helper asserts it.
- Never paste generated pixels into plant outlines. The whole region takes the one edit through a feathered weight.

## What the review records

The review looks for the four real failures on the whole delivered view and records each as `pass` or a `concern` naming where and what. They are observations, not gates:

1. **Plants growing out of hardscape:** a plant, foliage base or flower growing out of concrete, paving, steps, walls or other hardscape. Foliage overhanging an edge from a plant rooted in its bed is fine.
2. **Changed hardscape:** slab outlines, joints, steps, walls, planters, edging, driveway, house or other structures with changed geometry. More photographic materials are fine; added or removed joints, new gravel strips, and moved or doubled edges are concerns.
3. **Visible seams or double images:** seams, halos at the edit region, or ghosted/doubled leaves, spikes, flowers or edges.
4. **Species changed beyond recognition:** a plant no longer reads as its own form, or reads as a different kind of plant. Changes in size, outline, density, a little position, extra foliage, flower count or cultivar detail are not species concerns.

Everything else is a **defect** (types below). Record them all honestly: runs are compared on concerns and defects, and fewer is better.

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

Each species with an instance **at least 48 px on both sides** in the view (its visible bounding box, `--reference-min-px`) needs `reference_photos` or `reference_missing`. The plan fails with the names of any that lack both; smaller fragments may go without. Any entry that is given is checked.

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

- Find a real photograph whose identification you can cite: a botanical garden, university or extension service, herbarium, or a nursery that names the exact plant. Prefer the exact cultivar (`cultivar_match: "exact"`); otherwise the species (`"species"`, captioned "species photo" on the sheet). Prefer a whole plant in its usual habit, in flower for flowering plants, with little else in the frame. The plant library's research records (`references[].image_url`) are a good first source.
- Download it into the scene work folder next to `RECORDS.json` (for example `WORK/references/`) and record `path` relative to `RECORDS.json`. The plan rejects absolute paths, paths outside that folder and paths inside the shared plant library (`PACIFICA_ASSETS_DIR`).
- Record `source_page` and `image_url` (http/https), `accessed` (YYYY-MM-DD), `identification`, `cultivar_match` and a short `usage_note`.
- The photos are **model input only**. Never copy them into deliverables, galleries or the shared plant library.
- No usable photo: record `reference_missing` with the reason. That species is described by its records text only.

When a species has several photos, the sheet shows the first `exact` one (else the first).

## 2. Freeze the plan

```text
python scripts/scene_edit_plan.py plan --finish-review FINAL_REVIEW.json --records WORK/RECORDS.json --out WORK/scene
```

The plan (schema 5) re-verifies the previous composite and writes `SCENE_DIR/plan.json`, `region.png` (landscape alpha and transition pixels grown by `--margin` 24 px), `region-weight.png` (raised-cosine inward feather, `--feather` 24 px) and one job, `jobs/full/`:

- `input.png`: the whole 8-bit composite, Lanczos-reduced to fit the image tool's largest size for the view's orientation (`--input-size`, default 1536x1024 landscape, 1024x1536 portrait; never upscaled). The job records `input_size`.
- `prompt.txt`: the whole-view prompt (below).
- `labels.png`: the numbered plant map, the size of `input.png`: the input dimmed and desaturated, every plant with at least `label_min_px` (`--label-min-px`, default 400) visible native pixels thinly outlined from the renderer IDs and numbered in a black box at its visible centroid (nudged apart so numbers never overlap). Plants are numbered in prompt order (species largest first, then their locations, then left to right); each inventory row records its `label` (`null` for smaller fragments). The plan hashes the map and `validate` redraws it exactly.
- `references.jpg`: a labelled contact sheet of the view's reference photos, one per species, captioned with the botanical and common name. It lists the species at least `reference_min_px` (48) px on both sides that have `reference_photos`, largest visible pixel area first, at most 16, with a long edge of at most 2048 px. The job's `references` records the threshold, each shown species (`species`, `common_name`, `pixel_area`, the hashed `photo`, `identification`, `cultivar_match`, `source_page`, `image_url`, `accessed`), the species left off (`omitted`, with reason `below_threshold`, `reference_missing` or `sheet_cap`) and the hashed `sheet`.

The prompt holds what keeps the design: one pass over the entire view; same camera, framing, perspective and light; hardscape kept in place with identical geometry (no added, removed, moved or doubled joints, edges, steps, walls or gravel strips); bed ground shredded bark mulch when the bed-ground pass is on; every plant rooted in its bed; no new pots, steps, paving, structures, text or plants in empty bed ground; the house, sky, street and neighbourhood unchanged; every plant where it is as its listed species and form (natural leaf movement, fuller foliage and a natural outline are fine; never remove, replace or merge a plant); no invented hybrid foliage; every flowering plant in full bloom. Then one line per species, largest first: botanical name (common name; description); flowers; how many plants are visible and where, on a 3x3 grid of the view (`left`, `centre`, `right` × `background`, `middle ground`, `foreground`, from each plant's renderer-ID bounding-box centre), with the plants' numbers on the map, for example `3 plants: left foreground 2 (#1, #2), centre middle ground 1 (#3)`. The images go in this order: the composite (first, the edit target), the label map, then the reference sheet, and the prompt names each one before the species list: "The second supplied image is a numbered map of every plant in the view, for identification only: each number marks one plant of the species listed for it; keep each numbered plant where it is as that species; do not draw numbers, outlines or labels." and "The third supplied image is a labelled sheet of real photographs of the species in this view, for identification only: match each listed plant's leaves, habit and flowers to its photo; do not copy the photo's composition, background or extra plants into the scene." A view without a map or sheet shifts the ordinals. Prompts are deterministic and `validate` re-derives them, so don't edit a planned prompt.

## 3. Generate

1. Edit `jobs/full/input.png` with `jobs/full/prompt.txt` exactly, attaching `jobs/full/labels.png` and then `jobs/full/references.jpg` when the plan made them, in that order. Save the returned image unchanged.
2. Review attempt 1 as in step 5 and save its review notes (`check_set`, the four `checks`, `defects`, `defect_count`) as a JSON file, for example `full-notes-1.json`.
3. Make a second attempt **only** when those notes record a real-failure concern, a `bloom_missing` or `invented_plant` defect, or species-form drift (a `cultivar_detail` or `other` defect that names a plant of the view). Build its prompt with `python scripts/scene_edit_plan.py retry_prompt SCENE_DIR/plan.json --review-notes full-notes-1.json --out full-prompt-2.txt`: the planned prompt plus a frozen block of `Fix: <where> — <observation>; keep everything else as in your previous result's realism` lines (the tool refuses when nothing qualifies). Edit the **original input** again with the same attachments; never edit a previous output. Review attempt 2 the same way (`full-notes-2.json`).
4. Select the attempt with fewer real-failure concerns, then fewer defects; a tie keeps attempt 1. The pass always delivers one.

Record everything in `ATTEMPTS.json`, with paths relative to it:

```json
{"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen",
 "jobs": {"full": {"attempts": [
     {"raw": "full-1.png", "prompt": "full-prompt-1.txt", "labels": "../scene/jobs/full/labels.png",
      "reference": "../scene/jobs/full/references.jpg", "input": "../scene/jobs/full/input.png",
      "notes": "full-notes-1.json", "inspection": "agave #3 in the left foreground turned into prickly-pear pads"},
     {"raw": "full-2.png", "prompt": "full-prompt-2.txt", "labels": "../scene/jobs/full/labels.png",
      "reference": "../scene/jobs/full/references.jpg", "input": "../scene/jobs/full/input.png",
      "notes": "full-notes-2.json", "inspection": "agave rosette kept; joints unchanged"}],
   "selected": 2}}}
```

Attempt 1 uses the planned prompt exactly. With two attempts, both record their review `notes`, attempt 2's prompt must be exactly the `retry_prompt` output for attempt 1's notes, and `selected` must follow the selection rule; the guard result carries each attempt's `attempt_scores` (`concerns`, `defects`). With a planned label map or sheet, every attempt records it as `labels` / `reference` (byte-identical); without one, none. A recorded `input` must be the planned input exactly.

## 4. Compose

```text
python scripts/scene_edit_compose.py compose SCENE_DIR/plan.json --attempts ATTEMPTS.json --out OUT
```

The selected output is resized to the view (Lanczos; the aspect must match within 3%, nothing is warped) and replaces the previous composite inside the edit region through the feathered region weight, in linear light. Outside the region the result equals the previous composite and the cleaned photo exactly. There is no alignment check and nothing is dropped.

Outputs: `candidate.png`, `region-weight.png`, `compare.jpg` (the full-view side-by-side, previous composite | scene edit, labelled, 2400 px wide) and `scene-edit-proof.json` (`output`: the tool's `raw_size`, the `delivered_size` and the `upscale` factor, reported honestly). `scene_edit_compose.py verify OUT/scene-edit-proof.json` recomputes all of it. `scene_edit_compose.py side_by_side LEFT RIGHT --out OUT.jpg [--labels L R]` makes the same full-view comparison of any two same-size images (for example the bench baseline and a run).

Show people full-view side-by-sides only, never crops of parts of the view.

## 5. Review (schema 5 `scene_photographic_edit`, check set `real_failures_observed_v1`)

```json
{"schema_version": 5, "method": "scene_photographic_edit", "handoff": {...}, "camera_id": "...",
 "baseline_review": {"path": "FINAL_REVIEW.json", "sha256": "..."},
 "candidate": {"path": "OUT/candidate.png", "sha256": "..."},
 "scene_edit_proof": {"path": "OUT/scene-edit-proof.json", "sha256": "..."},
 "model": {"agent_model": "gpt-6-astra", "reasoning_effort": "high", "image_tool": "image_gen.imagegen"},
 "scene_review": {"reviewer": "vision-capable-agent", "source_and_final_loaded": true, "native_size_crops_loaded": true,
   "observations": "...", "paired_evidence": [{...}], "check_set": "real_failures_observed_v1",
   "attempts": 2, "selected_attempt": 2,
   "checks": {"plants_rooted_in_beds": "pass",
              "hardscape_unchanged": {"result": "concern", "where": "front steps", "observation": "a new joint on the second tread"},
              "no_seams_or_double_images": "pass",
              "species_form_recognizable": "pass"},
   "defects": [{"type": "extent_change", "where": "left agave", "observation": "crown about 10% wider"}],
   "defect_count": 1}}
```

Load the source photo, the previous composite and the final, and look at native-size crops across the whole view (for your own review; they are not deliverables). Each real-failure check is `"pass"` or a concern object with `where` and `observation`. `defects` is required, even when empty, and `defect_count` must equal its length. Defect types: `movement`, `extent_change`, `extra_foliage`, `invented_plant`, `cultivar_detail`, `bloom_missing`, `colour_shift`, `other`.

Run `appearance_guard.py SCENE_REVIEW.json`. The scene edit is always delivered (`delivered: scene_edit_composite`). The status is `accepted` when nothing was recorded and the previous composite was accepted; otherwise `completed_with_limitations`, with the previous composite's limitations followed by every concern and defect as non-blocking rows. The result carries `real_failure_observations`, `defects`, `defect_count`, `attempt_scores`, `output` and `compare`. Only integrity problems raise: missing or changed evidence, the wrong model, a broken attempt rule.

Keep the plan, records, manifest, raw outputs, prompts, attempt notes, input, label map, compare and the review with the delivery. Reference photos and `references.jpg` stay in the scene work folder only: the delivery carries their hashes and provenance, not the third-party images.

Plans, proofs and reviews of the removed tiled routes (schemas 1-4) are not verifiable with this version; check them with the skill commit that made them.
