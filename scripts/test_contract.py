"""Synthetic behavioral regressions: tests are not botanical-recognition evaluations."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from handoff import record,check,read
from appearance_guard import check as appearance,GEOMETRY
from make_edit_masks import make
from review_pairs import prepare
def save(p,data):
    p.write_text(json.dumps(data),encoding="utf-8");return record(p)
def fixture(p):
    (p/"scene.blend").write_bytes(b"Synthetic fixture, not a production native scene")
    image=np.full((32,32,3),100,np.uint8);image[6:26,6:26]=[30,120,40]
    Image.fromarray(image).save(p/"native.png");Image.fromarray(image).save(p/"structural.png")
    ids=np.zeros((32,32),np.int32);ids[6:26,6:26]=7;np.save(p/"ids.npy",ids);np.save(p/"depth.npy",np.ones((32,32)))
    editable,protected=make(ids,np.ones((32,32)),[7],root_pixels=[[16,25]])
    Image.fromarray(editable.astype("uint8")*255).save(p/"editable.png");Image.fromarray(protected.astype("uint8")*255).save(p/"protected.png")
    Image.fromarray((ids==7).astype("uint8")*255).save(p/"recognition-mask.png")
    notes=save(p/"notes.json",dict(design_intent=[],species_cautions=[],hardscape_constraints=[],warnings=[]))
    ledger=save(p/"ledger.json",[{"id":"agave-1","species":"Agave victoriae-reginae","root_xyz":[0,0,0]}])
    projections=save(p/"projected.json",{"camera_id":"front","image_sha256":record(p/"native.png")["sha256"],"plants":[{"id":"agave-1","species":"Agave victoriae-reginae","pass_index":7,"root_pixel":[16,25],"visible_pixels":400}]})
    provenance=save(p/"mask-proof.json",{"test_fixture":True,"object_ids":record(p/"ids.npy"),"depth":record(p/"depth.npy"),"projected_plants":projections,"allowed_material_indices":[7],"protected_indices":[],"boundary_margin_px":3})
    validation=save(p/"native-validation.json",{"kind":"native_3d_validation","status":"pass","placement_passed":True,"geometry_passed":True,"coverage_passed":True,"native_scene_sha256":record(p/"scene.blend")["sha256"],"plant_ledger_sha256":ledger["sha256"],"render_sha256_by_camera":{"front":record(p/"native.png")["sha256"]},"evidence":[ledger]})
    view={"camera_id":"front","provenance":"renderer_output","dimensions":[32,32],"image":record(p/"native.png"),"structural":record(p/"structural.png"),"object_ids":record(p/"ids.npy"),"depth":record(p/"depth.npy"),"projected_plants":projections,"editable_mask":record(p/"editable.png"),"protected_mask":record(p/"protected.png"),"mask_provenance":{"method":"native_id_depth_boundaries","evidence":provenance}}
    data={"schema_version":1,"kind":"validated_3d_render_handoff","project":"Synthetic","revision":"1","native_scene":record(p/"scene.blend"),"plant_ledger":ledger,"planting_notes":notes,"native_validation":validation,"requested_cameras":["front"],"views":[view]}
    save(p/"handoff.json",data)
    candidate=image.copy();candidate[15,15]=[40,130,50];Image.fromarray(candidate).save(p/"candidate.png")
    # These are known synthetic ground-truth annotations for script testing, not a claimed real agent recognition.
    row={"id":"agave-1","expected_species":"Agave victoriae-reginae","species_assessment":"consistent","identity_confidence":"high","diagnostic_traits":"Synthetic rosette label; no real botanical claim","count_assessment":"unchanged","occlusion_assessment":"consistent","baseline_anchor":[15,15],"candidate_anchor":[15,15],"root_visible":True,"candidate_root":[16,25],"candidate_mask":record(p/"recognition-mask.png"),"measurement_evidence":record(p/"candidate.png")}
    review={"handoff":record(p/"handoff.json"),"camera_id":"front","baseline_sha256":view["image"]["sha256"],"candidate":record(p/"candidate.png"),"method":"image_edit","recognition":{"reviewer":"vision-capable-agent","baseline_and_candidate_loaded":True,"original_pixel_crops_loaded":True,"observations":"Synthetic ground truth only","paired_crop_evidence":[record(p/"candidate.png")],"geometry":dict.fromkeys(GEOMETRY,"pass"),"plants":[row],"image_quality":"improved","quality_observations":"Synthetic permitted interior color change"}}
    return data,review
class ContractTests(unittest.TestCase):
    def setUp(self):
        root=(Path(__file__).resolve().parents[1]/"test-output").resolve();root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.p=Path(self.tmp.name).resolve()
        assert self.p.is_relative_to(root) and self.p!=root
        self.addCleanup(self.tmp.cleanup);self.data,self.review=fixture(self.p)
    def test_valid_handoff_and_material_interior(self):
        self.assertEqual(check(self.data,self.p)["plants"],1)
        self.assertEqual(appearance(self.review,self.p)["status"],"accepted")
    def test_missing_camera_and_ai_provenance_rejected(self):
        bad=copy.deepcopy(self.data);bad["requested_cameras"].append("side")
        with self.assertRaisesRegex(ValueError,"cameras"):check(bad,self.p)
        bad=copy.deepcopy(self.data);bad["views"][0]["provenance"]="ai_illustration"
        with self.assertRaisesRegex(ValueError,"native renderer"):check(bad,self.p)
    def test_changed_ledger_and_species_projection_rejected(self):
        (self.p/"ledger.json").write_text("[]")
        with self.assertRaisesRegex(ValueError,"changed evidence"):check(self.data,self.p)
    def test_changed_wall_or_landing_pixels_rejected(self):
        im=np.array(Image.open(self.p/"candidate.png"));im[0,0]=[0,200,0];Image.fromarray(im).save(self.p/"candidate.png")
        self.review["candidate"]=record(self.p/"candidate.png")
        with self.assertRaisesRegex(ValueError,"Protected pixels"):appearance(self.review,self.p)
    def test_species_uncertainty_or_substitution_rejected(self):
        for value in ["changed","uncertain"]:
            bad=copy.deepcopy(self.review);bad["recognition"]["plants"][0]["species_assessment"]=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"uncertain species"):appearance(bad,self.p)
    def test_movement_visible_root_and_missing_plant_rejected(self):
        bad=copy.deepcopy(self.review);bad["recognition"]["plants"][0]["candidate_anchor"]=[19,15]
        with self.assertRaisesRegex(ValueError,"anchor moved"):appearance(bad,self.p)
        bad=copy.deepcopy(self.review);bad["recognition"]["plants"][0]["candidate_root"]=[19,25]
        with self.assertRaisesRegex(ValueError,"root moved"):appearance(bad,self.p)
        bad=copy.deepcopy(self.review);bad["recognition"]["plants"]=[]
        with self.assertRaisesRegex(ValueError,"every visible plant"):appearance(bad,self.p)
    def test_crown_change_recomputed_not_self_reported(self):
        mask=np.array(Image.open(self.p/"recognition-mask.png"));mask=np.roll(mask,2,1);Image.fromarray(mask).save(self.p/"recognition-mask.png")
        self.review["recognition"]["plants"][0]["candidate_mask"]=record(self.p/"recognition-mask.png")
        with self.assertRaisesRegex(ValueError,"silhouette differs"):appearance(self.review,self.p)
    def test_pending_recognition_cannot_be_accepted(self):
        pending=prepare(self.p/"handoff.json","front",self.p/"candidate.png",self.p/"pairs")
        with self.assertRaisesRegex(ValueError,"Actual image-recognition"):appearance(pending,self.p)
        self.assertTrue(list((self.p/"pairs").glob("*baseline-candidate.png")))
    def test_masks_partition_and_protect_boundaries(self):
        ids=np.zeros((32,32),np.int32);ids[6:26,6:26]=7
        e,k=make(ids,np.ones(ids.shape),[7])
        self.assertFalse(e[6,10]);self.assertTrue(k[0,0]);self.assertTrue(np.all(e|k));self.assertFalse(np.any(e&k));self.assertTrue(e[15,15])
    def test_native_geometry_change_rejected(self):
        before={"exporter":"material_invariants_v1","native_scene":record(self.p/"scene.blend"),"invariants":{"plants":["fixed"],"wall":[0,1]}}
        after=copy.deepcopy(before);after["invariants"]["wall"]=[0,2]
        self.review.update(method="native_material_only",native_before=save(self.p/"before.json",before),native_after=save(self.p/"after.json",after))
        with self.assertRaisesRegex(ValueError,"invariants changed"):appearance(self.review,self.p)
    def test_missing_vision_geometry_check_rejected(self):
        del self.review["recognition"]["geometry"]["paved_landings"]
        with self.assertRaisesRegex(ValueError,"recognition check failed"):appearance(self.review,self.p)
if __name__=="__main__":unittest.main()
