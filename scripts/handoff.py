"""Validate immutable 3D handoff evidence; no image generation or semantic recognition."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from PIL import Image
def require(ok, message):
    if not ok: raise ValueError(message)
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def digest(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def record(path):
    path=Path(path).resolve()
    return {"path":str(path),"sha256":digest(path)}
def checked(rec, base):
    require(isinstance(rec,dict) and rec.get("path") and rec.get("sha256"),"Hashed file reference required")
    path=(Path(base)/rec["path"]).resolve()
    require(path.is_file() and digest(path)==rec["sha256"],"Missing or changed evidence: "+str(path))
    return path
def pixels(rec,base):
    from raster import load
    return load(checked(rec,base))[0]
def check(data,base):
    require(data.get("schema_version")==1 and data.get("kind")=="validated_3d_render_handoff","Expected validated 3D render handoff, not an AI illustration")
    require(data.get("project") and data.get("revision"),"Project and revision required")
    scene=checked(data["native_scene"],base)
    ledger=read(checked(data["plant_ledger"],base))
    require(isinstance(ledger,list) and ledger,"Nonempty canonical plant ledger required")
    pids=[p.get("id") for p in ledger]
    require(len(pids)==len(set(pids)) and all(pids),"Unique plant IDs required")
    require(all(p.get("species") for p in ledger),"Every plant requires an exact scheduled species")
    notes=read(checked(data["planting_notes"],base))
    require(isinstance(notes,dict) and all(k in notes for k in ("design_intent","species_cautions","hardscape_constraints","warnings")),"Planting/geometry notes and warnings required, even when lists are empty")
    vp=checked(data["native_validation"],base);validation=read(vp)
    require(validation.get("kind")=="native_3d_validation" and validation.get("status") in ("pass","completed_with_issues"),"Upstream native validation has not passed")
    require(validation.get("placement_passed") is True and validation.get("geometry_passed") is True and validation.get("coverage_passed") is True,"Upstream placement/geometry/coverage must pass")
    require(validation.get("native_scene_sha256")==digest(scene) and validation.get("plant_ledger_sha256")==data["plant_ledger"]["sha256"],"Validation scene or ledger binding differs")
    require(validation.get("evidence"),"Native validation evidence is missing")
    for rec in validation["evidence"]:checked(rec,vp.parent)
    wanted=data.get("requested_cameras",[])
    rows=data.get("views",[]);cids=[v.get("camera_id") for v in rows]
    require(wanted and len(wanted)==len(set(wanted)) and len(cids)==len(set(cids)) and set(cids)==set(wanted),"All requested cameras must appear exactly once")
    require(validation.get("render_sha256_by_camera")=={v["camera_id"]:v["image"]["sha256"] for v in rows},"Validation is not bound to current camera images")
    for view in rows:
        require(view.get("provenance")=="renderer_output","Only native renderer output can be the baseline")
        with Image.open(checked(view['image'],base)) as image:
            w,h=image.size
        require(view.get("dimensions")==[w,h],"Actual image dimensions differ")
        for key in ("structural","object_ids","depth","projected_plants","editable_mask","protected_mask"):
            checked(view[key],base)
        ids=np.load(checked(view["object_ids"],base),allow_pickle=False)
        depth=np.load(checked(view["depth"],base),allow_pickle=False)
        require(ids.shape==(h,w) and np.issubdtype(ids.dtype,np.integer),"Object IDs must be integer HxW renderer data")
        require(depth.shape==(h,w) and np.issubdtype(depth.dtype,np.number),"Depth must match image")
        projected=read(checked(view["projected_plants"],base))
        plants=projected.get("plants",[]);seen=[p.get("id") for p in plants]
        require(projected.get("camera_id")==view["camera_id"] and projected.get("image_sha256")==view["image"]["sha256"],"Projected plants belong to another camera/render")
        require(len(seen)==len(set(seen)) and set(seen)==set(pids),"Projection must include every plant, including occluded IDs")
        indices=[p.get("pass_index") for p in plants]
        require(all(type(i) is int and i>0 for i in indices) and len(indices)==len(set(indices)),"Unique positive plant render indices required")
        species={p["id"]:p["species"] for p in ledger}
        values,counts=np.unique(ids,return_counts=True)
        visibility=dict(zip(map(int,values),map(int,counts)))
        for p in plants:
            require(p.get("species")==species[p["id"]],"Projection/ledger species mismatch")
            count=visibility.get(p["pass_index"],0)
            require(p.get("visible_pixels")==count,"Visibility does not match actual ID pixels")
            if count:
                require(isinstance(p.get("root_pixel"),list) and len(p["root_pixel"])==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p["root_pixel"]),"Visible plant needs its native root projection")
        editable=np.array(Image.open(checked(view["editable_mask"],base)).convert("L"))>0
        protected=np.array(Image.open(checked(view["protected_mask"],base)).convert("L"))>0
        require(editable.shape==(h,w) and protected.shape==(h,w),"Mask dimensions differ")
        require(not np.any(editable & protected) and np.all(editable | protected),"Editable/protected masks must partition the image without overlaps or gaps")
        require(np.any(protected),"A full-frame edit mask does not protect geometry")
        require(view.get("mask_provenance") and view["mask_provenance"].get("method")=="native_id_depth_boundaries","Masks must derive from real renderer IDs/depth/boundaries")
        proof_path=checked(view["mask_provenance"]["evidence"],base);proof=read(proof_path)
        require(proof.get("object_ids",{}).get("sha256")==view["object_ids"]["sha256"] and proof.get("depth",{}).get("sha256")==view["depth"]["sha256"] and proof.get("projected_plants",{}).get("sha256")==view["projected_plants"]["sha256"],"Mask derivation uses different native arrays/projections")
        checked(proof["object_ids"],proof_path.parent);checked(proof["depth"],proof_path.parent)
        from make_edit_masks import make
        checked(proof["projected_plants"],proof_path.parent)
        expected_edit,expected_protect=make(ids,depth,proof["allowed_material_indices"],proof.get("boundary_margin_px",3),proof.get("protected_indices",[]),[p.get("root_pixel") for p in plants])
        require(np.array_equal(editable,expected_edit) and np.array_equal(protected,expected_protect),"Masks differ from recorded native boundary derivation")
        if view.get("source_photo"):checked(view["source_photo"],base)
    return {"status":"pass","project":data["project"],"views":cids,"plants":len(ledger),"scope":"Hashed upstream native-render handoff; not independent semantic verification"}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["check"]);parser.add_argument("handoff")
    args=parser.parse_args();p=Path(args.handoff).resolve()
    try: print(json.dumps(check(read(p),p.parent),indent=2))
    except (ValueError,KeyError,OSError,TypeError) as exc: parser.exit(1,"FAIL: "+str(exc)+"\n")
if __name__=="__main__":main()
