"""Blender: -b frozen.blend --python this.py -- output.json

Run before material finishing and again from the scene actually rendered.
The native scene must first be saved. A scene/hash alone is not a final-image proof.
"""
import bpy,json,sys,hashlib,array
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_contracts import camera_contract, node_settings, opacity_sensitive_rgb

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def mesh_digest(me):
    h=hashlib.sha256()
    for data,prop,n,kind in [(me.vertices,'co',3,'f'),(me.loops,'vertex_index',1,'i'),(me.polygons,'loop_total',1,'i'),(me.polygons,'material_index',1,'i')]:
        values=array.array(kind,[0])*(len(data)*n);data.foreach_get(prop,values);h.update(values.tobytes())
    for uv in me.uv_layers:
        values=array.array('f',[0])*(len(uv.data)*2);uv.data.foreach_get('uv',values);h.update(uv.name.encode());h.update(values.tobytes())
    return h.hexdigest()

def material_contract(mat,images):
    """Allow surface color/roughness changes, freeze opacity and mapping behavior."""
    if not mat:return None
    if not mat.use_nodes:return {'alpha':mat.diffuse_color[3],'nodes':False}
    rows=[];appearance={'Base Color','Roughness','Metallic','Subsurface Weight','Coat Weight'}
    for n in mat.node_tree.nodes:
        assert n.type!='GROUP','Expand material node groups before strict snapshot: '+mat.name
        if n.type=='OUTPUT_MATERIAL':assert not n.inputs['Displacement'].is_linked,'True displacement is not allowed in the locked material pass'
        row={'name':n.name,'type':n.bl_idname,'inputs':{}}
        row['settings']=node_settings(n)
        for sock in n.inputs:
            if not hasattr(sock,'default_value') or (n.type=='BSDF_PRINCIPLED' and sock.name in appearance):continue
            v=sock.default_value
            try:v=list(v)
            except TypeError:pass
            if isinstance(v,(str,int,float,bool,list)):row['inputs'][sock.identifier]=v
        for attr in ('operation','blend_type','interpolation','projection','extension','vector_type','uv_map'):
            if hasattr(n,attr):row[attr]=getattr(n,attr)
        if n.type=='TEX_IMAGE' and n.image:
            im=n.image;key=im.as_pointer()
            if key not in images:
                values=array.array('f',[0])*len(im.pixels);im.pixels.foreach_get(values)
                images[key]={'size':list(im.size),'alpha_sha256':hashlib.sha256(values[3::4].tobytes()).hexdigest(),
                             'all_channels_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
            row['image_alpha']={k:v for k,v in images[key].items() if k!='all_channels_sha256'}
            if opacity_sensitive_rgb(n):
                row['opacity_or_lighting_rgb_sha256']=images[key]['all_channels_sha256']
        rows.append(row)
    links=sorted((l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in mat.node_tree.links)
    return {'nodes':sorted(rows,key=lambda r:r['name']),'links':links,'backface_culling':mat.use_backface_culling}

def export(out):
    path=Path(bpy.data.filepath);assert path.is_file(),'Save native scene first'
    deps=bpy.context.evaluated_depsgraph_get();cache={};materials={};images={};rows=[]
    for ob in bpy.context.scene.objects:
        if ob.get('kind')!='plant':continue
        assert ob.type=='MESH' and not ob.hide_render,'Missing/hidden native plant: '+ob.name
        if ob.modifiers:
            evaluated=ob.evaluated_get(deps);me=evaluated.to_mesh();geometry=mesh_digest(me);evaluated.to_mesh_clear()
        else:
            key=ob.data.as_pointer()
            if key not in cache:cache[key]=mesh_digest(ob.data)
            geometry=cache[key]
        contracts=[]
        for slot in ob.material_slots:
            mat=slot.material;key=mat.as_pointer() if mat else 0
            if key not in materials:materials[key]=material_contract(mat,images)
            contracts.append(materials[key])
        opacity=hashlib.sha256(json.dumps(contracts,sort_keys=True).encode()).hexdigest()
        rows.append({'id':ob.name,'species':ob.get('species'),'asset_id':ob.get('asset_id'),'matrix_world':[x for row in ob.matrix_world for x in row],'geometry_sha256':geometry,'opacity_mapping_sha256':opacity,'pass_index':ob.pass_index,'seed':ob.get('seed')})
    cameras={o.name:camera_contract(o) for o in bpy.context.scene.objects if o.type=='CAMERA'}
    record={'exporter':'blender_instance_snapshot_v1','native_scene':{'path':str(path),'sha256':digest(path)},'instances':sorted(rows,key=lambda r:r['id']),'cameras':cameras}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2));print('Native plant snapshot:',len(rows),flush=True)

if __name__=='__main__':export(Path(sys.argv[sys.argv.index('--')+1]).resolve())
