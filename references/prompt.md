# Image appearance and cleanup prompts

## Scene photographic edit (default)

`scene_edit_plan.py` writes these prompts: the fixed A1 preamble plus one inventory line per species, computed from renderer IDs and the plant records. Use them exactly for attempt 1. A second attempt keeps the planned prompt and appends a `SECOND ATTEMPT:` correction written after inspecting attempt 1, naming the plant, its correct leaf/flower form and the extents, crowns, roots and hardscape edges to keep, and it edits the original input again. See [scene edit](scene-edit.md).

## Frozen image-interior route

The first template below is only for the frozen image-interior route. It does not restrict the separate native branch/leaf geometry refinement route in [finishing](finishing.md).

Use this route for the landscape foreground stage when image-space refinement is useful. Use native botanical refinement when fuller foliage or different branch/leaf geometry is needed; the frozen image-interior route cannot fill a crown. First inspect the validated render and freeze its appearance plan. Use one view or frozen native crop per built-in image-generation call. This stage does not complete the skill: subsequently composite the checked landscape over the original site photograph using the photo-integration workflow. Never generate a replacement house or neighborhood.

Primary edit target: the validated native render. Preserve its camera, framing, population, exact species, root positions, sizes, silhouettes, branching, hardscape topology, narrow borders, empty paved landings and current bloom state. Additional native location maps, same-specimen crops, other supplied views and exact-asset references explain identity/materials; they do not authorize a different design.

Improve photographic surface detail only within the plan's permitted material interiors: original leaf surface color/detail, bark and fine soil/mulch or material grain. Match existing lighting/exposure/shadows. Do not enlarge/fill crowns, add plants/flowers, erase unusual anatomy, straighten walls, pave borders, alter stairs/paver gaps or close physical fence gaps.

Insert relevant planting notes and the native ID/species/location map. List regions marked preserved: leave tiny or occluded fragments unchanged; do not invent anatomy to identify them or blend them away into the background. Their identity is inherited from the validated native scene. Reference clearer supplied views only when they show the same specimen.

Return one full-frame candidate at the original dimensions, no labels. Save the exact prompt and reference hashes, including the pre-generation plan hash. Retain this raw image. Compose through that plan with `compose_candidate.py`, then inspect/validate the final composite. Protected output samples come from the native baseline; fidelity of the raw generation is not assumed. No crop/resize/warp to force a mismatched generation to fit. Local correction/retention can only narrow permissions.

## Source-photo cleanup

Primary target: the original photographic plate, before proposed landscape overlay. Remove the specifically inventoried people, tools, debris and temporary construction equipment, including visible fragments and attributable shadows/reflections. Reconstruct only the obscured nearby surfaces using the photographed fence, ground, wall or vegetation as context. Preserve permanent architecture, fixtures, existing vegetation, camera, perspective and lighting. Do not add proposed plants or invent a different neighborhood. Reference the frozen cleanup masks/inventory and finishing-plan hash. Return the original full canvas without labels. Keep the raw result; final pixels outside the cleanup union are restored from the original photo.

For a native-region target, explicitly name the edit target, exact crop dimensions and regional plan hash. Cover the complete target and its shadows before freezing permissions. Reconstruct surrounding texture with matched exposure so rectangular patch boundaries do not remain. Preserve natural fallen leaves that belong to the finished scene; they are not construction debris.

## Local photo/landscape joins

Primary target: the exact cleaned-photo/checked-landscape overlay preview or its frozen native-region crop. Make permitted concrete/sidewalk interiors look photographed: fine aggregate, subtle nonrepeating wear and tonal variation at the scene's scale, coherent grain and light. Add a restrained natural scatter of fallen leaves consistent with the source season/nearby trees, varied in orientation with believable small contact shadows. Match soil/mulch grain, hue and contacts to adjacent photo context. Keep every plant, species, root position, physical border, wall, slab outline, step and EXISTING paving joint fixed. Uninterrupted concrete remains uninterrupted: do not add scoring lines, joints, borders, tiles or structural cracks. No thick leaf piles, construction debris or props. Do not blur foliage, fill planting gaps or expand beds. Reference the transition alpha, finishing plan and regional plan hash; return the target's exact dimensions. The final helper applies only bounded nonplant transition pixels. Foliage antialiasing comes from the genuine renderer matte, not this generation.

## Native foliage refinement brief

For the separately planned native `--foliage-fill` route: preserve each named plant's exact species/cultivar, permanent ID, root/world transform, source asset binding, size class and existing presentation envelope. Fill accidental sparse/bald regions with attached species-appropriate branchlets and layered leaves. Keep characteristic openness where the species requires it; do not turn every specimen into a dense ball. Vary leaf orientation, curvature and overlap naturally, with credible attachment and volume. Make sprays three-dimensional instead of flat curtains and flower heads botanically plausible instead of smooth primitives. Keep bloom state and the planted design. No new plants, relocated roots, substituted species, expanded maturity or moved hardscape. Compare actual before/after native crops and the photo-overlay pilot before the full render queue.
