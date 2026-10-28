# Photoreal lab bench

A frozen copy of one finished pipeline run, plus a harness that scores experiments on the photoreal stage.
The goal is composites that read as ordinary photographs **without moving any plant or changing any species**.
The plan behind it is the Photoreal Lab Plan page (private artifact).

## Data (never in this repo)

The bench folder holds client imagery, so it lives outside every repo:

- VM: `C:\Workstation\artifacts\photoreal-bench\reddy-20261025\`
- blob: `pipeline/bench/photoreal-reddy-20261025*.zip`

| Path | What |
|---|---|
| `bench-manifest.json` | views, files and SHA-256 (`Bench.verify()` checks them) |
| `handoff/` | render handoff, plant ledger, `candidate-catalog-v3.json` (object kind per pass index) |
| `views/<v>/source.jpg` | original site photo |
| `views/<v>/cleaned-photo.png` | photo with people and clutter removed (frozen; cleanup is not under test) |
| `views/<v>/native.png`, `object-ids.npy`, `depth.npy`, `structural.png` | handoff render passes: the placement authority |
| `views/<v>/projected-plants.json` | every plant: id, species, pass index |
| `views/<v>/baseline-run/` | the shipped run's refined render: `straight_rgb.png` + `alpha.png` (landscape layer), its IDs/depth, `transition_alpha.png`, cleanup masks |
| `views/<v>/baseline/final.png` | the shipped composite (X0-final) |
| `anchors/real/` | crops of the project's other site photos: real photographs for judge calibration |
| `bloom.json` | flowering species and the flowers each should show; exempt species and why |
| `crops.json` | the fixed judge crops (frozen; never re-chosen once experiments are scored) |

Experiments live next to the bench in `experiments/` (or `PHOTOREAL_EXPERIMENTS`).

## Running

```
set PHOTOREAL_BENCH=C:\Workstation\artifacts\photoreal-bench\reddy-20261025
python bench\bench.py new B2a --layer B --hypothesis "Depth + line conditioning keeps geometry" --method "FLUX.2-dev + ControlNet union, denoise 0.45"
...  the experiment writes experiments\B2a\out\p49.png and p51.png (3840x2560 RGB) and run.json ...
python bench\bench.py score B2a        # gates + judge -> scores.json
python bench\bench.py board            # experiments\board\index.html
```

Requirements: `numpy pillow opencv-python-headless`; the plants gate also needs `torch` (CPU is fine) and
`transformers` (SAM ViT-B weights download once from Hugging Face). The judge runs `claude -p` (Read only);
set `PHOTOREAL_JUDGE_CLI` to use another judge, `PHOTOREAL_JUDGE_MODEL` to pin a model.

## Gates (hard; any failure disqualifies)

| Gate | Test | Threshold |
|---|---|---|
| background | pixels outside the edit region (landscape matte + transition band + cleanup areas, dilated 24 px) equal the cleaned photo | ≤ 0.1% differ by > 3 |
| hardscape | render edges of walls, steps, paving (away from plants) vs output edges | mean distance ≤ 4 px |
| plants | every plant ≥ 400 visible px: SAM mask of the output vs SAM mask of the render (box + 4 points from its own pixels). ≥ 2500 px: IoU ≥ 0.70, median shift ≤ 3% of its box diagonal, height (2–98th pct) within ±10%. Smaller: IoU ≥ 0.40. Plants SAM can't isolate in the render fall back to gradient alignment (shift ≤ 3%, correlation ≥ 0.12) | ≥ 97% of plants pass, none with IoU < 0.30 |
| new plants | green vegetation on open ground (beds, terrain) away from plants, beyond the raw render | ≤ 0.2% of the image |
| species | (judge) the 12 largest plants per view: render crop vs output crop, "same plant and species?" and "any plant, or flowers, not carried by a plant in the render?" (flowers growing from a render plant are not extra). A pair flagged either way is asked twice more; the majority stands | no confident "different" or "extra" |
| bloom | (judge) the largest plant of each flowering species in `bloom.json`, per view, in a crop centred on its own pixels (object ID), zoomed to 768 px with everything else dimmed: open flowers visible, and their colour family and flower type match the description (finer detail - exact shade, ageing tints, leaves, habit - is noted, not failed). Majority of 3 votes; a plant whose own pixels equal X0-final's keeps X0-final's verdict | ≥ 90% |

Controls: `X0-final` (shipped output), `X0-raw3d` (render over the photo, no finishing), `X0-broken`
(plants moved, blobs painted, sky changed: must fail).

## Judge

`judge.py` sends anonymized crops in random order to a fresh `claude -p` session that may only read them.
Each batch mixes the experiment's crops with known real crops (`anchors/real`, plus cleaned-photo crops outside
the landscape) and known synthetic crops (the shipped final), so every score comes with the judge's accuracy on
anchors. Two passes in different orders. Realism = weighted mean `p_photo` over the fixed crops
(overview 2, plants 1.5, ground and hardscape 1). Giveaways are tagged by type to steer the next round.

**Goal metric: impression.** The same crops and anchors go to a second prompt (`judge.impression`) that scores the
impression a homeowner gets, 1-5: 1 toy (plastic, clay, video-game plants), 2 render, 3 realistic (real living
plants; small AI or CG traces are fine), 4 near photo, 5 photo. A crop is *realistic* at a mean of 2.75 or more over
the passes. `impression_share` = weighted share of realistic crops; `goal_met` = qualified and `impression_share`
>= 0.8 and no crop below 1.5 (`impression_toys`). Calibration on reddy-20261025 (2 passes): real anchors 5.0 (100%
realistic), raw 3D 1.5 (0%), shipped final 1.8 (4%), D1a 2.4 (31%). The forensic `realism` / `photo_share` (unedited
photo vs any generated content) stays as a reference; it calls nearly any generated content synthetic.
`bench.py impression <id>` adds the impression judge to an already scored experiment.

## Rules for experiment agents

See `EXPERIMENT.md`.
