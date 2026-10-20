# Input from the 3D render skill

Run the render skill first. Its final delivery is high-fidelity 3D, not an AI-finished photograph. This skill accepts render-handoff.json using schema version 1, kind validated_3d_render_handoff.

File references are objects with path and sha256. Relative paths resolve against the containing record's documented base; use an intact project package or intentionally remap/revalidate copied evidence. Absolute user-machine paths are not portable by themselves. No client files or logs belong in the reusable skill repository.

Top level: project, revision, requested_cameras (unique IDs); native_scene, plant_ledger, planting_notes, native_validation (hashed files); views, exactly one per requested camera.

The plant ledger is a nonempty list of unique permanent id and exact species; retain roots, sizes, asset binding, rotation and seed in the upstream ledger as well. The notes object contains design_intent, species_cautions, hardscape_constraints and warnings. Carry designer notes, specimen exceptions, growth/flowering state, nearby same-color species, groundcover versus individual counts, occlusion, earlier errors and areas that must remain unplanted. Preserve source references and distinguish estimated dimensions from measured ones.

The native validation report has kind native_3d_validation, status pass or completed_with_issues, placement_passed/geometry_passed/coverage_passed true, native_scene_sha256, plant_ledger_sha256, render_sha256_by_camera, and evidence references to actual scene/instance/geometry/camera/coverage checks. Draft aesthetic quality can remain a warning; fidelity cannot. The report is generated only after running the render skill's native delivery checker. A status field alone is not evidence.

Each view:
- camera_id, dimensions [width,height], provenance renderer_output.
- image (delivered lossless native beauty), structural (hardscape-only image), object_ids (integer HxW NumPy array), depth (numeric HxW array).
- projected_plants JSON with camera_id, image_sha256, plants [{id,species,pass_index,root_pixel,visible_pixels,...}]. Include all canonical plants even when visible_pixels is zero. Counts are recomputed from the real ID pass.
- editable_mask/protected_mask: grayscale PNGs partitioning the frame. Editable regions are permitted material interiors; protect crown silhouettes, root contacts, wall caps/toes/returns, borders, landing outlines, steps, paving joints and architecture/context. Derive masks from native IDs/depth/boundary evidence with a protected boundary band. Never use a full-frame edit mask.
- mask_provenance {method:native_id_depth_boundaries,evidence:{path,sha256}}.
- optional source_photo and exact-species appearance references.

Use scripts/make_edit_masks.py with actual renderer ID/depth arrays and explicitly selected material-interior object indices. Its default boundary margin is three pixels. Keep paving/landing, borders and roots protected; expanding material-interior permissions requires actual geometry review. The helper cannot infer semantic surface permissions from an integer ID alone.

Do not accept an earlier AI illustration by relabeling its provenance or attaching native IDs. Return incomplete guides or illustrations to the render workflow.

handoff.py verifies hashes, dimensions, coverage, species bindings, actual ID visibility, mask partition and validation bindings. It does not authenticate a dishonest exporter or independently interpret PDF geometry; inspect upstream evidence.
