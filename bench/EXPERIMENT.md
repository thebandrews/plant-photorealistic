# Running one photoreal experiment

You are running **one** experiment from the photoreal lab backlog. Your job is to produce two composites,
`out/p49.png` and `out/p51.png`, that look like ordinary photographs while keeping the design exactly.

## Hard rules

1. **Plants stay put.** No plant may move, disappear, be added, change species or cultivar, or change
   presentation size. Leaf, petal, bark and texture detail may change. The bench gates check this; a run
   that fails a gate is disqualified however good it looks.
2. **Only the landscape changes.** Pixels outside the edit region (landscape matte, transition band, cleanup
   areas) must equal `cleaned-photo.png`. Composite your result over it.
3. **Code changes go in your branch only** (`exp/<id>` of plant-photorealistic, in your own worktree).
   Change nothing in `bench/` (gates, judge, crops): those are locked. Never write into the bench folder.
4. **Budget.** Stay inside `budget_gpu_seconds` in `experiment.json` (render-remote exits 5 when spent).
   Codex image generation (the image pass) counts as turns; keep a count. Don't loop renders; don't raise samples or resolution
   beyond the bench's 3840×2560.
5. **Client data stays out of git.** Outputs, crops and intermediate images live only in the experiment folder.
6. **Never print or store SAS URLs, keys or tokens.**
7. **Flowering plants are shown in full bloom.** `bloom.json` in the bench lists every flowering species with the
   flowers it should show (colour, form, how they are carried) and the species that are exempt (tall flower stalks,
   conifers, ferns). Flowers stay within the plant's own footprint and height, so the plants gate still passes.
   The judge checks the largest plant of each flowering species in both views; under 90% in bloom and true to the
   description disqualifies the run. The shipped output shows 5 of 22.

## Inputs

`PHOTOREAL_BENCH` points at the bench (read the README table). Most useful:
`baseline-run/straight_rgb.png` + `alpha.png` (the landscape layer and its matte), `cleaned-photo.png`,
`object-ids.npy` + `handoff/candidate-catalog-v3.json` (what each pixel is), `depth.npy`, `structural.png`,
`projected-plants.json` (species per plant). The Blender scene is on the VM at the path in
`bench-manifest.json` (`scene`); save any changed scene as a new file in your experiment folder.

## Tools

- `bench/lab/comfy.py`: diffusion on the lab's RunPod endpoint (Qwen-Image, Qwen-Image-Edit 2509/2511, InstantX
  ControlNet Union and Inpainting, Lightning and Anything2Real LoRAs). `comfy.run(workflow, images)`; `comfy.qwen_edit`.
- `bench/lab/plantcheck.py`: the plants gate for chosen plants, so a generate-and-check loop can test each plant
  as soon as it is regenerated: `PlantChecker('p49').check(composite, ids)`. About 15 s per call; check several
  plants per call.

## Scoring

The goal metric is `impression_share`: the share of the fixed crops that give a homeowner a realistic impression
(score 3 or more of 5 - real living plants of believable species; small AI or CG traces are fine). The goal is 80%
with no crop scoring as a plastic toy, every gate passing and bloom at 90%. The shipped output scores 1.8 of 5 (4%
realistic): its plants read as plastic toys. The judge's giveaways name what holds a crop back - mostly plant
anatomy (cloned leaflets, fused fronds, waxy leaves), plastic material and flat ground. `win_vs_shipped` (side by
side with the shipped output) and the strict forensic `realism` are reported for reference.

## Deliverables

- `out/p49.png`, `out/p51.png`: 3840×2560 RGB, the full composite.
- `run.json`: `{"gpu_seconds": n, "codex_turns": n, "wall_minutes": n, "notes": "...", "steps": [...]}`.
- `NOTES.md`: what you did, what you saw, what you'd try next. Include failed attempts.

Then run `python bench/bench.py gate <id>` yourself before finishing. If a gate fails, fix it (two attempts at
most) or report it. Don't run `judge`; the orchestrator scores every experiment the same way.

## Image pass model (Ben, 2026-10-30)
The AI image pass uses GPT-6 Astra on high reasoning through Codex's built-in image generation (ChatGPT sign-in, no API key). Lab Codex runs default to `-m gpt-6-astra -c model_reasoning_effort="high"` (a spec's `codex_args` overrides it). Don't use Qwen/ComfyUI diffusion for the image pass unless Ben asks.
