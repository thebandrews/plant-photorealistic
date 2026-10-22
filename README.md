# Plant Photorealistic

A Codex skill for placing validated 3D planting into original site photographs, with people and construction clutter removed, natural plant anatomy, and blended foliage/ground transitions.

The upstream `pacifica-planting-renders` skill supplies the validated handoff and placement authority. This skill keeps originals immutable, refines appearance, cleans the source photo, and overlays the checked landscape while retaining the real house and neighborhood. Branch and leaf mesh refinement is supported without changing species, root positions or design.

Sparse foliage can be filled within the existing presentation envelope using a native `--foliage-fill` plan. Surface finishing adds photographic concrete detail and restrained natural fallen leaves without moving structures or inventing paving joints. New plans require explicit fullness/surface realism findings. See [efficient execution](references/efficient-execution.md) for early visual pilots, bounded parallel work, resumable stages and verified native-size regional edits. Native PNG decoding, one-pass visibility counts and bounded-memory comparison/blending reduce helper overhead without skipping checks.

Install this repository root as `plant-photorealistic` under the Codex skills directory. Python helpers use `requirements.txt`; native snapshots/mattes require Blender. Built-in image generation needs no separate API key. Invoke: `Use $plant-photorealistic on this render-handoff.json.`

Candidate re-renders run on the local Blender by default. On a workstation with the `render-remote` GPU client, `scripts/blender_run.py --render-backend runpod` (or `PLANT_RENDER_BACKEND=runpod`) runs them on remote GPUs with byte-identical inputs and outputs. `scripts/blender_backend.py` is shared unchanged with the plant-render skill. When `PACIFICA_ASSETS_DIR` points at the shared plant library, `scripts/library_writeback.py` saves accepted refined specimens there as new versions.

Read [SKILL.md](SKILL.md) and [the finishing workflow](references/finishing.md) for the three appearance routes, cleanup inventory, genuine renderer alpha, local transition editing and schema 4 release evidence. Material/image-interior routes retain schema 2 checks. Legacy schema 3 binary composites remain verifiable; new deliveries require the cleanup/anatomy/edge checks in schema 4 `photo_finish`. Scripts verify evidence and reproducible operations; actual visual review establishes photographic coherence.

```text
python scripts/handoff.py check HANDOFF
python scripts/appearance_guard.py LANDSCAPE_REVIEW.json
python scripts/photo_finish.py plan --photo-plan PHOTO_PLAN/plan.json --landscape-review LANDSCAPE_REVIEW.json --renderer-matte MATTE.json --cleanup-inventory CLEANUP.json --transition-alpha TRANSITIONS.png --out FINISH_PLAN.json
# Generate the photo cleanup through the built-in image-generation tool, then:
python scripts/photo_finish.py preview FINISH_PLAN.json --cleanup-raw CLEANUP_RAW.png --out PREVIEW_DIR
# Generate a bounded ground-transition edit of that preview, then:
python scripts/photo_finish.py compose FINISH_PLAN.json --cleanup-raw CLEANUP_RAW.png --transition-raw TRANSITION_RAW.png --out FINAL_DIR
python scripts/appearance_guard.py FINAL_REVIEW.json
python -m unittest discover -s scripts -p "test_*.py" -v
```

Permission plans precede generation. Raw images, immutable masks, actual native passes and reconstruction proofs remain separate. Tiny/hidden plants inherit native identity and remain accounted for without fictional cultivar recognition. Ground repair cannot paint over plants. Cleanup preserves exact original-photo samples outside declared masks, and final compositing preserves all other untouched photo samples. Entirely unchanged landscape fallbacks remain explicit.

The repository contains reusable workflow code and synthetic tests only. Client photos, plans, planting assets and run logs stay outside it.
