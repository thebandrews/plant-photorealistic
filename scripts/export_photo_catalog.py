"""Run in Blender against the unchanged validated scene to bind overlay objects."""
import json
import hashlib
import sys
from pathlib import Path
import bpy

def record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': sha}

out = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
objects = [{'name': o.name, 'kind': o.get('kind'), 'pass_index': o.pass_index, 'type': o.type}
           for o in bpy.context.scene.objects]
out.write_text(json.dumps({'provenance': 'renderer_object_catalog',
                           'native_scene': record(bpy.data.filepath), 'objects': objects}, indent=2))
print('Photo object catalog exported', len(objects))
