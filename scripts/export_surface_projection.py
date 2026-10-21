"""Blender: export actual surface triangles for coincident context/landscape recovery."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def record(p):
    p = Path(p).resolve()
    with p.open('rb') as stream:
        return {'path': str(p), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


p = argparse.ArgumentParser(); p.add_argument('--handoff', required=True); p.add_argument('--surfaces', nargs='+', required=True)
p.add_argument('--overlap-with', nargs='+', required=True); p.add_argument('--out', required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:]); handoff = json.loads(Path(a.handoff).read_text())
scene = bpy.context.scene; result = {'provenance': 'renderer_surface_projection', 'native_scene': record(bpy.data.filepath), 'views': []}
background = []
for name in a.overlap_with:
    ob = bpy.data.objects[name]
    if ob.get('kind') != 'existing_ground_context':
        raise ValueError('Only existing ground context can be resolved by surface overlap')
    background.append(ob.pass_index)
for view in handoff['views']:
    width, height = view['dimensions']; scene.camera = bpy.data.objects[view['camera_id']]
    scene.render.resolution_x = width; scene.render.resolution_y = height; scene.render.resolution_percentage = 100
    inv = scene.camera.matrix_world.inverted(); surfaces = []
    for name in a.surfaces:
        ob = bpy.data.objects[name]
        if ob.get('kind') in ('plant', 'existing_context', 'existing_ground_context') or ob.type != 'MESH':
            raise ValueError('A real nonplant landscape surface is required')
        evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = evaluated.to_mesh(); mesh.calc_loop_triangles(); triangles = []
        for tri in mesh.loop_triangles:
            poly = []
            for i in tri.vertices:
                cam = inv @ (evaluated.matrix_world @ mesh.vertices[i].co)
                poly.append(Vector((cam.x, cam.y, -cam.z)))
            # A ground triangle can cross behind the camera. Clip it rather than
            # dropping its entire visible front portion.
            clipped = []; near = max(.01, scene.camera.data.clip_start)
            for first, second in zip(poly, poly[1:] + poly[:1]):
                ina, inb = first.z >= near, second.z >= near
                if ina:
                    clipped.append(first)
                if ina != inb:
                    clipped.append(first + (second-first) * ((near-first.z)/(second.z-first.z)))
            projected = []
            for cam in clipped:
                world = scene.camera.matrix_world @ Vector((cam.x, cam.y, -cam.z))
                ndc = world_to_camera_view(scene, scene.camera, world)
                projected.append({'pixel': [ndc.x*width, (1-ndc.y)*height], 'camera_xyz': list(cam)})
            for i in range(1, len(projected)-1):
                triangles.append([projected[0], projected[i], projected[i+1]])
        evaluated.to_mesh_clear()
        surfaces.append({'name': name, 'pass_index': ob.pass_index, 'kind': ob.get('kind'), 'triangles': triangles})
    result['views'].append({'camera_id': view['camera_id'], 'dimensions': [width, height], 'depth_convention': 'camera_z',
                            'overlapping_context_indices': background, 'surfaces': surfaces})
Path(a.out).write_text(json.dumps(result, indent=2)); print('Native surface projections exported', len(result['views']))
