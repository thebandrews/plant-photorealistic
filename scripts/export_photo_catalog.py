"""Run in Blender against the saved baseline or checked candidate scene."""
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
catalog = {'provenance': 'renderer_object_catalog', 'native_scene': record(bpy.data.filepath), 'objects': objects}
args = sys.argv[sys.argv.index('--') + 2:]
if args:
    assert len(args) == 2 and args[0] == '--photo-roles', 'Use --photo-roles reviewed-map.json'
    rp = Path(args[1]).resolve(); roles = json.loads(rp.read_text(encoding='utf-8-sig'))
    assert roles['native_scene'] == catalog['native_scene']
    lookup = {o['name']: o for o in objects}
    for row in roles['objects']:
        ob = lookup[row['name']]; assert ob['kind'] == row['native_kind'] and ob['type'] == 'MESH'
        ob['photo_kind'] = row['photo_kind']
    catalog['photo_role_map'] = record(rp)
out.write_text(json.dumps(catalog, indent=2), encoding='utf-8')
print('Photo object catalog exported', len(objects))
