# Plant Photorealistic

A Codex skill for appearance-only refinement of validated planting renders.

The upstream `pacifica-planting-renders` skill ends with high-fidelity 3D images and exact placement validations. This skill separately consumes its `render-handoff.json`, keeps the baseline immutable, and refines plant/material appearance.

## Install

Install this repository's root as the skill folder `plant-photorealistic` under your Codex skills directory. It contains `SKILL.md`, `agents/`, `references/` and `scripts/`. Python helpers need the packages in `requirements.txt`; native material snapshots also require Blender. No API key is required for the built-in image-generation tool.

Invoke: `Use $plant-photorealistic on this render-handoff.json.`

## Validation

```text
python scripts/handoff.py check /path/to/render-handoff.json
python scripts/review_pairs.py /path/to/render-handoff.json --camera front --candidate /path/to/candidate.png --out /path/to/new-review
python scripts/appearance_guard.py /path/to/review.json
python -m unittest discover -s scripts -p "test_*.py" -v
```

The agent must actually recognize and compare the plants, species traits, positions, silhouettes and hardscape in both images. The scripts recompute mask drift and protected-pixel checks and validate evidence; they do not perform semantic recognition themselves. Unknown species identity, missing plants, changed wall routes, paved borders or planted landings reject a candidate. After bounded retries, the original validated 3D view is retained as the fallback.

Only native 3D outputs retain exact placement certification. Visually checked image edits remain appearance derivatives. See the [handoff contract](references/handoff.md) and [recognition protocol](references/appearance-review.md).

The repository contains reusable workflow code and synthetic tests only, with no client plans, site photos, render assets or planting logs.
