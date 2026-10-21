# Plant Photorealistic

A Codex skill for placing validated 3D planting into original site photographs, retaining the real house and neighborhood.

The upstream `pacifica-planting-renders` skill ends with high-fidelity 3D images and placement validations. This skill consumes its `render-handoff.json`, keeps the baseline immutable, refines landscape appearance, then composites that checked landscape into each matching source photograph. A refined render with a rendered background is an intermediate stage, not a completed photographic delivery.

## Install

Install this repository's root as the skill folder `plant-photorealistic` under your Codex skills directory. It contains `SKILL.md`, `agents/`, `references/` and `scripts/`. Python helpers need the packages in `requirements.txt`; native material snapshots also require Blender. No API key is required for the built-in image-generation tool.

Invoke: `Use $plant-photorealistic on this render-handoff.json.`

## Validation

```text
python scripts/handoff.py check /path/to/render-handoff.json
python scripts/appearance_plan.py /path/to/render-handoff.json --camera front --method image_edit --out /path/to/new-plan
# Generate an edit of the native render after freezing the plan, then:
python scripts/compose_candidate.py /path/to/new-plan/plan.json --raw /path/to/raw.png --out /path/to/new-candidate
python scripts/review_pairs.py /path/to/render-handoff.json --camera front --candidate /path/to/new-candidate/candidate.png --plan /path/to/new-plan/plan.json --out /path/to/new-review
# Complete actual visual review and attach composite-proof.json before checking:
python scripts/appearance_guard.py /path/to/review.json
# Export the object catalog in Blender, prepare the original-photo plan, and composite:
python scripts/photo_composite.py plan /path/to/render-handoff.json --camera front --catalog /path/to/catalog.json --camera-fit /path/to/fit.json --out /path/to/photo-plan
python scripts/photo_composite.py compose /path/to/photo-plan/plan.json --foreground /path/to/new-candidate/candidate.png --out /path/to/final
# Record actual photo/native/final and edge observations in a schema 3 final review:
python scripts/appearance_guard.py /path/to/final-review.json
python -m unittest discover -s scripts -p "test_*.py" -v
```

Version 2 reviews inherit species and location from validated native evidence. Every plant is accounted for through native invariants, exact preserved pixels, visual review of edited appearance, or full occlusion. Tiny/overlapping fragments do not require impossible independent cultivar recognition. Image composites use frozen native-derived masks, preserve 8/16-bit protected samples and opacity, and retain raw generation/proof. Local failures can retain individual native regions while other improvements proceed. The agent still inspects changed anatomy/materials and the whole frame for added/missing plants or altered hardscape; scripts do not perform semantic recognition. See the recognition protocol for native-render proof and local `--retain` recovery.

Original native renders remain the exact placement authority. Visually checked image edits remain appearance derivatives; unchanged whole-view fallbacks are explicitly nonaccepted. Legacy reviews retain their original contract. See the [handoff contract](references/handoff.md) and [recognition protocol](references/appearance-review.md).

Final schema 3 checks reconstruct the original photographic plate and native landscape matte, preserve photo samples outside the overlay, retain all visible native plants and bind the foreground to its schema 2 appearance review. Proxy architecture and context ground planes are excluded. Visual checks separately cover alignment, house/neighborhood retention, edges, ground contact, shadows and construction transitions. See [photo integration](references/photo-integration.md). Material-stage acceptance alone never counts as final completion.

The repository contains reusable workflow code and synthetic tests only, with no client plans, site photos, render assets or planting logs.
