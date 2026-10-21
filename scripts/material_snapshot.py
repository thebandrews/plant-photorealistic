"""Blender: --background saved.blend --python material_snapshot.py -- snapshot.json.
Only RGB/surface shader refinements may differ; geometry, opacity and lighting freeze.
"""
import bpy,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_plant_snapshot import export,mesh_digest,material_contract
def world_contract(world):
    if world is None:return None
    rows=[]
    if world.use_nodes:
        for n in world.node_tree.nodes:
            vals={}
            for s in n.inputs:
                if hasattr(s,"default_value"):
                    v=s.default_value
                    try:v=list(v)
                    except TypeError:pass
                    if isinstance(v,(str,int,float,bool,list)):vals[s.identifier]=v
            row={"name":n.name,"type":n.bl_idname,"values":vals}
            if hasattr(n,"image") and n.image:
                import array
                values=array.array("f",[0])*len(n.image.pixels);n.image.pixels.foreach_get(values)
                row["lighting_image_sha256"]=hashlib.sha256(values.tobytes()).hexdigest()
            rows.append(row)
        links=sorted((x.from_node.name,x.from_socket.identifier,x.to_node.name,x.to_socket.identifier) for x in world.node_tree.links)
    else:links=[]
    return {"color":list(world.color),"nodes":sorted(rows,key=lambda x:x["name"]),"links":links}
def snapshot(out):
    out=Path(out);export(out)
    plants=json.loads(out.read_text());graph=bpy.context.evaluated_depsgraph_get()
    objects=[];images={};mats={};geometries={}
    for ob in sorted(bpy.context.scene.objects,key=lambda o:o.name):
        row={"id":ob.name,"type":ob.type,"parent":ob.parent.name if ob.parent else None,"matrix_world":[x for r in ob.matrix_world for x in r],"hide_render":ob.hide_render,"visible":ob.visible_get(),"kind":ob.get("kind"),"species":ob.get("species"),"seed":ob.get("seed")}
        row['ray_visibility']={k:getattr(ob,k) for k in ('visible_camera','visible_diffuse','visible_glossy','visible_transmission','visible_shadow','visible_volume_scatter','is_holdout') if hasattr(ob,k)}
        if ob.type=="MESH":
            evaluated=ob.evaluated_get(graph);me=evaluated.to_mesh()
            row["geometry_sha256"]=mesh_digest(me);evaluated.to_mesh_clear()
            contracts=[]
            for slot in ob.material_slots:
                mat=slot.material;key=mat.as_pointer() if mat else 0
                if key not in mats:mats[key]=material_contract(mat,images)
                contracts.append(mats[key])
            row["opacity_mapping"]=contracts
        elif ob.type=="LIGHT":
            row["lighting"]={"type":ob.data.type,"energy":ob.data.energy,"color":list(ob.data.color),"use_shadow":ob.data.use_shadow}
            for attr in ("angle","shadow_soft_size","shape","size","size_y","spread"):
                if hasattr(ob.data,attr):row["lighting"][attr]=getattr(ob.data,attr)
            row["lighting"]["nodes"]=world_contract(ob.data)
        elif ob.type not in ("CAMERA","EMPTY"):
            raise ValueError("Convert geometry to evaluated meshes before material-only refinement: "+ob.name)
        objects.append(row)
    sc=bpy.context.scene
    settings={"world":world_contract(sc.world),"engine":sc.render.engine,"frame":sc.frame_current,"resolution":[sc.render.resolution_x,sc.render.resolution_y,sc.render.resolution_percentage],"pixel_aspect":[sc.render.pixel_aspect_x,sc.render.pixel_aspect_y],"view_transform":sc.view_settings.view_transform,"look":sc.view_settings.look,"exposure":sc.view_settings.exposure,"gamma":sc.view_settings.gamma,"film_transparent":sc.render.film_transparent,"use_border":sc.render.use_border,"border":[sc.render.border_min_x,sc.render.border_min_y,sc.render.border_max_x,sc.render.border_max_y],"collections":{c.name:{"hide_render":c.hide_render,"hide_viewport":c.hide_viewport} for c in bpy.data.collections}}
    if hasattr(sc,'cycles'):
        settings['sampling']={k:getattr(sc.cycles,k) for k in ('seed','use_animated_seed','samples','use_adaptive_sampling','adaptive_threshold','use_denoising') if hasattr(sc.cycles,k)}
    result={"exporter":"material_invariants_v1","native_scene":plants["native_scene"],"invariants":{"plants":plants["instances"],"cameras":plants["cameras"],"objects":objects,"settings":settings}}
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("Material invariants exported:",len(objects),flush=True)
    return result
if __name__=="__main__":snapshot(Path(sys.argv[sys.argv.index("--")+1]).resolve())
