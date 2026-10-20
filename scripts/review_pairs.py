"""Prepare native-pixel crop pairs and an intentionally pending recognition template."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from handoff import read,checked,record,check,require
from appearance_guard import GEOMETRY
def prepare(path,cid,candidate,out):
    path=Path(path).resolve();data=read(path);check(data,path.parent)
    views=[v for v in data["views"] if v["camera_id"]==cid];require(len(views)==1,"Unknown camera")
    v=views[0];im=Image.open(checked(v["image"],path.parent)).convert("RGB");ai=Image.open(candidate).convert("RGB")
    require(ai.size==im.size,"Candidate dimensions differ; do not silently resize")
    ids=np.load(checked(v["object_ids"],path.parent),allow_pickle=False);proj=read(checked(v["projected_plants"],path.parent))
    out=Path(out);require(not out.exists(),"Use a new review output directory");out.mkdir(parents=True)
    plants=[];evidence=[]
    for n,p in enumerate(proj["plants"]):
        yy,xx=np.where(ids==p["pass_index"])
        if not len(xx):continue
        box=[max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(im.width,int(xx.max())+13),min(im.height,int(yy.max())+13)]
        a=im.crop(box);b=ai.crop(box);pair=Image.new("RGB",(a.width*2,a.height));pair.paste(a,(0,0));pair.paste(b,(a.width,0))
        file=out/f"plant-{n:05d}-baseline-candidate.png";pair.save(file);evidence.append(record(file))
        plants.append({"id":p["id"],"expected_species":p["species"],"species_assessment":"uncertain","identity_confidence":"low","diagnostic_traits":"","count_assessment":"uncertain","occlusion_assessment":"uncertain","baseline_anchor":[float(xx.mean()),float(yy.mean())],"candidate_anchor":None,"candidate_mask":None,"root_visible":False,"measurement_evidence":record(file)})
    template={"handoff":record(path),"camera_id":cid,"candidate":record(candidate),"baseline_sha256":v["image"]["sha256"],"method":"image_edit","recognition":{"reviewer":"vision-capable-agent","baseline_and_candidate_loaded":False,"original_pixel_crops_loaded":False,"observations":"","paired_crop_evidence":evidence,"geometry":dict.fromkeys(GEOMETRY,"pending"),"plants":plants,"image_quality":"pending","quality_observations":""}}
    (out/"review-pending.json").write_text(json.dumps(template,indent=2),encoding="utf-8");return template
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("handoff");p.add_argument("--camera",required=True);p.add_argument("--candidate",required=True);p.add_argument("--out",required=True);a=p.parse_args();prepare(a.handoff,a.camera,a.candidate,a.out)
if __name__=="__main__":main()
