# Image appearance and cleanup prompts

The first template below is only for the frozen image-interior route. It does not restrict the separate native branch/leaf geometry refinement route in [finishing](finishing.md).

Use this route for the landscape foreground stage when image-space refinement is useful; prefer native material refinement when available. First inspect the validated render and freeze its appearance plan. Use one view per built-in image-generation call. This stage does not complete the skill: subsequently composite the checked landscape over the original site photograph using the photo-integration workflow. Never generate a replacement house or neighborhood.

Primary edit target: the validated native render. Preserve its camera, framing, population, exact species, root positions, sizes, silhouettes, branching, hardscape topology, narrow borders, empty paved landings and current bloom state. Additional native location maps, same-specimen crops, other supplied views and exact-asset references explain identity/materials; they do not authorize a different design.

Improve photographic surface detail only within the plan's permitted material interiors: original leaf surface color/detail, bark and fine soil/mulch or material grain. Match existing lighting/exposure/shadows. Do not enlarge/fill crowns, add plants/flowers, erase unusual anatomy, straighten walls, pave borders, alter stairs/paver gaps or close physical fence gaps.

Insert relevant planting notes and the native ID/species/location map. List regions marked preserved: leave tiny or occluded fragments unchanged; do not invent anatomy to identify them or blend them away into the background. Their identity is inherited from the validated native scene. Reference clearer supplied views only when they show the same specimen.

Return one full-frame candidate at the original dimensions, no labels. Save the exact prompt and reference hashes, including the pre-generation plan hash. Retain this raw image. Compose through that plan with `compose_candidate.py`, then inspect/validate the final composite. Protected output samples come from the native baseline; fidelity of the raw generation is not assumed. No crop/resize/warp to force a mismatched generation to fit. Local correction/retention can only narrow permissions.

## Source-photo cleanup

Primary target: the original photographic plate, before proposed landscape overlay. Remove the specifically inventoried people, tools, debris and temporary construction equipment, including visible fragments and attributable shadows/reflections. Reconstruct only the obscured nearby surfaces using the photographed fence, ground, wall or vegetation as context. Preserve permanent architecture, fixtures, existing vegetation, camera, perspective and lighting. Do not add proposed plants or invent a different neighborhood. Reference the frozen cleanup masks/inventory and finishing-plan hash. Return the original full canvas without labels. Keep the raw result; final pixels outside the cleanup union are restored from the original photo.

## Local photo/landscape joins

Primary target: the exact cleaned-photo/checked-landscape overlay preview. Match local soil/paving texture, grain, hue and contact shadows to adjacent photo context inside the frozen transition permissions. Keep every plant, species, root position, physical border, wall, step and paving joint fixed. Do not blur foliage, fill gaps, expand beds or invent structures. Reference the transition alpha and plan hash; return the original full canvas. The final helper applies only bounded nonplant transition pixels. Foliage antialiasing comes from the genuine renderer matte, not this generation.
