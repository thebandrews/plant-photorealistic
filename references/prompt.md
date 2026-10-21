# Image appearance edit template

Use this route when image-space refinement is useful; prefer native material refinement when available. First inspect the validated render and freeze its appearance plan. Use one view per built-in image-generation call.

Primary edit target: the validated native render. Preserve its camera, framing, population, exact species, root positions, sizes, silhouettes, branching, hardscape topology, narrow borders, empty paved landings and current bloom state. Additional native location maps, same-specimen crops, other supplied views and exact-asset references explain identity/materials; they do not authorize a different design.

Improve photographic surface detail only within the plan's permitted material interiors: original leaf surface color/detail, bark and fine soil/mulch or material grain. Match existing lighting/exposure/shadows. Do not enlarge/fill crowns, add plants/flowers, erase unusual anatomy, straighten walls, pave borders, alter stairs/paver gaps or close physical fence gaps.

Insert relevant planting notes and the native ID/species/location map. List regions marked preserved: leave tiny or occluded fragments unchanged; do not invent anatomy to identify them or blend them away into the background. Their identity is inherited from the validated native scene. Reference clearer supplied views only when they show the same specimen.

Return one full-frame candidate at the original dimensions, no labels. Save the exact prompt and reference hashes, including the pre-generation plan hash. Retain this raw image. Compose through that plan with `compose_candidate.py`, then inspect/validate the final composite. Protected output samples come from the native baseline; fidelity of the raw generation is not assumed. No crop/resize/warp to force a mismatched generation to fit. Local correction/retention can only narrow permissions.
