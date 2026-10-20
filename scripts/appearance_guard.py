"""Check drift/protection evidence recorded after actual agent image recognition.
This script does NOT recognize species or segment plants by itself.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from handoff import read,record,checked,pixels,check as check_handoff,require
GEOMETRY=("wall_runs_returns","bed_borders","paved_landings","stairs_pavers","ground_contact","architecture","camera_crop","no_added_or_missing_plants","cross_view_consistency")
def distance(a,b):
    require(isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==2,"Two-dimensional recognized anchors required")
    require(all(type(x) in (int,float) and math.isfinite(x) for x in a+b),"Finite image anchors required")
    return math.dist(a,b)
def check(data,base):
    hp=checked(data["handoff"],base);handoff=read(hp);check_handoff(handoff,hp.parent)
    cid=data["camera_id"];matches=[v for v in handoff["views"] if v["camera_id"]==cid]
    require(len(matches)==1,"Unknown handoff camera");v=matches[0]
    require(data.get("baseline_sha256")==v["image"]["sha256"],"Comparison is not bound to the immutable native baseline")
    before=pixels(v["image"],hp.parent);after=pixels(data["candidate"],base)
    require(before.shape==after.shape,"Candidate camera dimensions/crop differ")
    method=data.get("method");require(method in ("image_edit","native_material_only"),"Appearance method required")
    if method=="image_edit":
        mask=np.array(Image.open(checked(v["protected_mask"],hp.parent)).convert("L"))>0
        require(np.array_equal(before[mask],after[mask]),"Protected pixels changed: reject candidate; do not move borders, walls, landing or plant silhouettes")
    else:
        a=read(checked(data["native_before"],base));b=read(checked(data["native_after"],base))
        require(a.get("exporter")==b.get("exporter")=="material_invariants_v1","Actual native material invariant exports required")
        require(a.get("native_scene",{}).get("sha256")==handoff["native_scene"]["sha256"],"Native material baseline differs from handoff")
        checked(a["native_scene"],Path(checked(data["native_before"],base)).parent)
        checked(b["native_scene"],Path(checked(data["native_after"],base)).parent)
        require(a.get("invariants") and a["invariants"]==b.get("invariants"),"Native plant/hardscape/camera/lighting invariants changed")
        proof=read(checked(data["render_proof"],base))
        require(proof.get("provenance")=="renderer_export" and proof.get("camera_id")==cid and proof.get("native_scene_sha256")==b["native_scene"]["sha256"] and proof.get("image_sha256")==data["candidate"]["sha256"],"Candidate not bound to revised native material render")
    vision=data.get("recognition",{})
    require(vision.get("reviewer")=="vision-capable-agent" and vision.get("baseline_and_candidate_loaded") is True and vision.get("original_pixel_crops_loaded") is True,"Actual image-recognition inspection of both images and native-size crops required")
    require(vision.get("observations") and vision.get("paired_crop_evidence"),"Record specific recognition observations and paired crops")
    for crop in vision["paired_crop_evidence"]:checked(crop,base)
    checks=vision.get("geometry",{})
    require(all(checks.get(k)=="pass" for k in GEOMETRY),"Hardscape/camera/cross-view recognition check failed or absent")
    projected=read(checked(v["projected_plants"],hp.parent))
    visible={p["id"]:p for p in projected["plants"] if p["visible_pixels"]>0}
    native_ids=np.load(checked(v["object_ids"],hp.parent),allow_pickle=False)
    rows=vision.get("plants",[]);names=[p.get("id") for p in rows]
    require(len(names)==len(set(names)) and set(names)==set(visible),"Compare every visible plant exactly once; do not waive small or awkward plants")
    for row in rows:
        p=visible[row["id"]]
        require(row.get("expected_species")==p["species"],"Recognition row uses the wrong scheduled species")
        require(row.get("species_assessment")=="consistent" and row.get("identity_confidence")=="high" and row.get("diagnostic_traits"),"Changed or uncertain species identity blocks acceptance")
        require(row.get("count_assessment")=="unchanged" and row.get("occlusion_assessment")=="consistent","Plant count/occlusion changed or uncertain")
        require(distance(row["baseline_anchor"],row["candidate_anchor"])<=2,"Recognized plant anchor moved by more than 2 native pixels")
        if row.get("root_visible"):
            require(distance(p["root_pixel"],row["candidate_root"])<=1,"Visible plant root moved by more than 1 native pixel")
        baseline_mask=native_ids==p["pass_index"]
        with Image.open(checked(row["candidate_mask"],base)) as im:
            recognized=np.array(im.convert("L"))>0
        require(recognized.shape==baseline_mask.shape,"Candidate recognition mask dimensions differ")
        union=np.count_nonzero(baseline_mask | recognized)
        iou=np.count_nonzero(baseline_mask & recognized)/max(1,union)
        require(iou>=.98,"Recognized crown silhouette differs beyond tolerance")
        area_change=np.count_nonzero(recognized)/np.count_nonzero(baseline_mask)-1
        require(abs(area_change)<=.02,"Recognized plant size changed by more than 2 percent")
        ay,ax=np.where(baseline_mask);by,bx=np.where(recognized)
        require(len(bx)>0 and math.hypot(float(ax.mean()-bx.mean()),float(ay.mean()-by.mean()))<=2,"Recognized plant centroid moved beyond tolerance")
        require(row.get("measurement_evidence"),"Recognition/measurement evidence required per plant")
        checked(row["measurement_evidence"],base)
    require(vision.get("image_quality")=="improved" and vision.get("quality_observations"),"Actual photographic improvement must be inspected")
    return {"status":"accepted","camera_id":cid,"method":method,"visible_plants_checked":len(rows),"classification":"appearance_derivative_visually_checked","exact_placement_certification":"retained only by original native baseline","limitation":"Semantic findings are the recorded agent's visual judgments, not a mathematical species guarantee"}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("review");a=p.parse_args();path=Path(a.review).resolve()
    try:print(json.dumps(check(read(path),path.parent),indent=2))
    except (ValueError,KeyError,OSError,TypeError) as e:p.exit(1,"FAIL: "+str(e)+"\n")
if __name__=="__main__":main()
