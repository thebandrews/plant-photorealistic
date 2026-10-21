"""Prepare native-pixel crop pairs and an intentionally pending recognition template."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from handoff import read,checked,record,check,require
from appearance_guard import GEOMETRY
def prepare(path,cid,candidate,out,plan_path=None):
    if plan_path is not None:
        return prepare_v2(path,cid,candidate,out,plan_path)
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

def prepare_v2(path,cid,candidate,out,plan_path):
    from appearance_plan import validate
    from raster import load, save as save_png
    path=Path(path).resolve();plan_path=Path(plan_path).resolve();plan=read(plan_path)
    hp,_,v,ids,planned,_=validate(plan,plan_path.parent)
    require(hp==path and plan['camera_id']==cid,'Plan belongs to another handoff/camera')
    before,_=load(checked(v['image'],hp.parent));after,_=load(candidate)
    require(before.shape==after.shape and before.dtype==after.dtype,'Candidate dimensions/precision differ; do not silently resize')
    display=lambda a:Image.fromarray((a>>8).astype('uint8') if a.dtype==np.uint16 else a)
    im=display(before);ai=display(after);diff=np.any(before!=after,axis=2)
    out=Path(out).resolve();require(not out.exists(),'Use a new review output directory');out.mkdir(parents=True)
    # Native material invariants cover every instance. Representative visible
    # assets/materials need appearance review, not taxonomic re-identification.
    representatives={}
    for p in planned:
        if p['visible_pixels']>representatives.get(p['expected_species'],{}).get('visible_pixels',0):
            representatives[p['expected_species']]=p
    native_samples={p['id'] for p in representatives.values()}
    rows=[];evidence=[]
    for n,p in enumerate(planned):
        row={k:p[k] for k in ('id','expected_species','identity_source','evidence_path')}
        route=p['evidence_path']
        sample=route=='visual_review' or (route=='native_invariants' and p['id'] in native_samples)
        if sample:
            visible=ids==p['pass_index'];sample_mask=visible&diff
            yy,xx=np.where(sample_mask if np.any(sample_mask) else visible);mid=len(xx)//2;x,y=int(xx[mid]),int(yy[mid])
            x0,y0,x1,y1=max(0,x-256),max(0,y-256),min(im.width,x+257),min(im.height,y+257)
            file=out/f'plant-{n:05d}-baseline-candidate.png'
            save_png(file,np.concatenate([before[y0:y1,x0:x1],after[y0:y1,x0:x1]],axis=1));evidence.append(record(file))
            row['measurement_evidence']=record(file)
        if route=='visual_review':
            row.update(appearance_assessment='pending',observations='',count_assessment='pending',occlusion_assessment='pending',root_visible=False)
        rows.append(row)
    # Always include a whole-frame comparison and at least one native-size
    # changed-material crop, even when no plant is edited.
    a=im.copy();b=ai.copy();a.thumbnail((1440,1080));b.thumbnail((1440,1080))
    pair=Image.new('RGB',(a.width*2,a.height));pair.paste(a,(0,0));pair.paste(b,(a.width,0));pair.save(out/'whole-frame.png')
    evidence.append(record(out/'whole-frame.png'))
    yy,xx=np.where(diff)
    if len(xx):
        mid=len(xx)//2;x,y=int(xx[mid]),int(yy[mid]);box=[max(0,x-128),max(0,y-128),min(im.width,x+129),min(im.height,y+129)]
        x0,y0,x1,y1=box;save_png(out/'changed-material.png',np.concatenate([before[y0:y1,x0:x1],after[y0:y1,x0:x1]],axis=1))
        evidence.append(record(out/'changed-material.png'))
    template={'schema_version':2,'handoff':record(path),'plan':record(plan_path),'camera_id':cid,'candidate':record(candidate),'baseline_sha256':v['image']['sha256'],'method':plan['method'],
              'recognition':{'reviewer':'vision-capable-agent','baseline_and_candidate_loaded':False,'original_pixel_crops_loaded':False,'observations':'','paired_crop_evidence':evidence,'geometry':dict.fromkeys(GEOMETRY,'pending'),'plants':rows,'image_quality':'pending','quality_observations':''}}
    (out/'review-pending.json').write_text(json.dumps(template,indent=2),encoding='utf-8');return template
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("handoff");p.add_argument("--camera",required=True);p.add_argument("--candidate",required=True);p.add_argument("--out",required=True);p.add_argument("--plan");a=p.parse_args();prepare(a.handoff,a.camera,a.candidate,a.out,a.plan)
if __name__=="__main__":main()
