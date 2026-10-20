"""Derive conservative material-interior masks from real native IDs/depth."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from handoff import require,record
def make(ids,depth,allowed,margin=3,protected_ids=(),root_pixels=()):
    require(ids.ndim==2 and depth.shape==ids.shape,"Matching ID/depth arrays required")
    require(np.issubdtype(ids.dtype,np.integer),"Integer renderer ID array required")
    require(type(margin) is int and margin>=3,"Protect at least three boundary pixels")
    require(allowed and not set(allowed)&set(protected_ids),"Select explicit nonprotected material indices")
    editable=np.isin(ids,allowed)
    boundary=np.zeros(ids.shape,bool)
    for axis in (0,1):
        shifted=np.roll(ids,1,axis);other=np.roll(depth,1,axis)
        jumps=(shifted!=ids) | ((np.isfinite(depth)!=np.isfinite(other))) | (np.abs(np.nan_to_num(depth)-np.nan_to_num(other))>.01)
        boundary|=jumps
    for _ in range(margin):
        boundary=boundary|np.roll(boundary,1,0)|np.roll(boundary,-1,0)|np.roll(boundary,1,1)|np.roll(boundary,-1,1)
    editable &= ~boundary
    for root in root_pixels:
        if not isinstance(root,list) or len(root)!=2 or not all(isinstance(x,(int,float)) and np.isfinite(x) for x in root):continue
        x,y=map(lambda n:int(round(n)),root)
        x0=max(0,x-margin);x1=min(ids.shape[1],x+margin+1);y0=max(0,y-margin);y1=min(ids.shape[0],y+margin+1)
        if x1>x0 and y1>y0:editable[y0:y1,x0:x1]=False
    editable[:margin]=False;editable[-margin:]=False;editable[:,:margin]=False;editable[:,-margin:]=False
    return editable,~editable
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--ids",required=True);p.add_argument("--depth",required=True);p.add_argument("--projected-plants",required=True);p.add_argument("--allow",type=int,nargs="+",required=True);p.add_argument("--protect",type=int,nargs="*",default=[]);p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);require(not out.exists(),"Use a new mask output directory");out.mkdir(parents=True)
    projection=json.loads(Path(a.projected_plants).read_text(encoding="utf-8-sig"))
    e,k=make(np.load(a.ids,allow_pickle=False),np.load(a.depth,allow_pickle=False),a.allow,protected_ids=a.protect,root_pixels=[p.get("root_pixel") for p in projection["plants"]])
    Image.fromarray(e.astype("uint8")*255).save(out/"editable.png");Image.fromarray(k.astype("uint8")*255).save(out/"protected.png")
    report={"method":"native_id_depth_boundaries","object_ids":record(a.ids),"depth":record(a.depth),"projected_plants":record(a.projected_plants),"allowed_material_indices":a.allow,"protected_indices":a.protect,"boundary_margin_px":3,"editable_pixels":int(e.sum()),"editable_mask":record(out/"editable.png"),"protected_mask":record(out/"protected.png"),"semantic_permissions":"Caller must inspect roots, architecture, borders and landing protection; integer indices alone cannot infer roles."}
    (out/"mask-provenance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
if __name__=="__main__":main()
