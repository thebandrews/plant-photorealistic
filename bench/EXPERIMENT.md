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
   Codex image generation counts as turns; keep a count. Don't loop renders; don't raise samples or resolution
   beyond the bench's 3840×2560.
5. **Client data stays out of git.** Outputs, crops and intermediate images live only in the experiment folder.
6. **Never print or store SAS URLs, keys or tokens.**

## Inputs

`PHOTOREAL_BENCH` points at the bench (read the README table). Most useful:
`baseline-run/straight_rgb.png` + `alpha.png` (the landscape layer and its matte), `cleaned-photo.png`,
`object-ids.npy` + `handoff/candidate-catalog-v3.json` (what each pixel is), `depth.npy`, `structural.png`,
`projected-plants.json` (species per plant). The Blender scene is on the VM at the path in
`bench-manifest.json` (`scene`); save any changed scene as a new file in your experiment folder.

## Deliverables

- `out/p49.png`, `out/p51.png`: 3840×2560 RGB, the full composite.
- `run.json`: `{"gpu_seconds": n, "codex_turns": n, "wall_minutes": n, "notes": "...", "steps": [...]}`.
- `NOTES.md`: what you did, what you saw, what you'd try next. Include failed attempts.

Then run `python bench/bench.py gate <id>` yourself before finishing. If a gate fails, fix it (two attempts at
most) or report it. Don't run `judge`; the orchestrator scores every experiment the same way.
