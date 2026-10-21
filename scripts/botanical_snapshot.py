"""Blender: -b saved.blend --python botanical_snapshot.py -- snapshot.json.

Extends the material snapshot with an immutable world-space root collar. Mesh
changes are authorized separately by a botanical refinement plan, never globally.
"""
import hashlib
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from material_snapshot import snapshot


def export(out):
    result = snapshot(out)
    graph = bpy.context.evaluated_depsgraph_get()
    plants = {p['id']: p for p in result['invariants']['plants']}
    for name, row in plants.items():
        ob = bpy.context.scene.objects[name]
        root = ob.matrix_world.translation
        evaluated = ob.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        # Five centimetres in world units, accounting for the scene unit scale.
        radius = .05 / bpy.context.scene.unit_settings.scale_length
        collar = sorted(tuple(float(x) for x in ob.matrix_world @ v.co)
                        for v in mesh.vertices if (ob.matrix_world @ v.co - root).length <= radius)
        evaluated.to_mesh_clear()
        row['root_world'] = list(root)
        row['root_collar'] = {'radius_world': radius, 'vertices': len(collar),
                              'sha256': hashlib.sha256(json.dumps(collar).encode()).hexdigest()}
    result['exporter'] = 'botanical_invariants_v1'
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    export(Path(sys.argv[sys.argv.index('--') + 1]).resolve())
