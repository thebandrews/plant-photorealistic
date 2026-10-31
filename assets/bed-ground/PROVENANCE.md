# Bed-ground mulch exemplars

`mulch-0.png`, `mulch-1.png`, `mulch-2.png`: original generated images, not photographs of any site or client.

- Model: Qwen-Image 2512 text-to-image (fp8), Lightning 8-step LoRA, ComfyUI on the photoreal lab's RunPod endpoint;
  seed 5, four images at 1024 px; the fourth was rejected because it showed a green sprout.
- Made by the photoreal lab experiment G3 (Reddy bloom bench, 2026-10-26) and reused unchanged by H1, which beat the
  shipped image on 91% of judged crops. Here each is centre-cropped square and resized to 512 px (the pass samples
  mip level 2 or coarser at bench distances, so nothing is lost).
- Prompt: "A real photograph looking straight down at a garden bed covered with fine shredded bark mulch. The whole
  frame is filled edge to edge with mulch: small, thin, fibrous shreds of dark reddish-brown fir bark, 1 to 3 cm long,
  randomly piled, natural variation in colour from dark brown to warm red-brown, a few darker gaps between shreds.
  Soft even daylight, no strong shadows, sharp focus everywhere, shot on a DSLR. No plants, no leaves, no stones, no
  wood chips larger than 3 cm, no objects, no text."
- Inspected: fibrous bark shreds only; no plants, sprouts, stones or text.

`scripts/bed_ground.py` uses only their relative luminance (zero-mean grain) and 12% of their chroma; tone, light and
colour come from the composite. Replace them only with new hashed files and a new provenance entry.
