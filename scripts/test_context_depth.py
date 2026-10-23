"""Actual-array occlusion regressions, synthetic geometry and no visual claims."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from handoff import record
from photo_finish import context_ground_occlusion

class GroundDepth(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'test-output';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        np.save(self.p/'ids.npy',np.array([[11,11],[12,0]],np.int32))
        np.save(self.p/'depth.npy',np.ones((2,2),np.float32))
        view={'camera_id':'c','object_ids':record(self.p/'ids.npy'),'depth':record(self.p/'depth.npy')}
        (self.p/'handoff.json').write_text(json.dumps({'views':[view]}))
        self.stage={'camera_id':'c','handoff':record(self.p/'handoff.json')}
        self.catalog={'objects':[{'type':'MESH','name':'p','kind':'plant','pass_index':7},
            {'type':'MESH','name':'wall','kind':'retaining_wall','pass_index':10},
            {'type':'MESH','name':'ground','kind':'terrain','pass_index':11},
            {'type':'MESH','name':'house','kind':'building','pass_index':12}]}
        self.op={'operation':'native_context_ground_depth_v1','baseline_object_ids':view['object_ids'],
            'baseline_depth':view['depth'],'occluding_indices':[11],'depth_tolerance':.01,
            'depth_convention':'ray_distance','occluded_samples':1}
        self.ids=np.array([[10,7],[10,10]],np.int32);self.depth=np.full((2,2),2,np.float32)
    def run_op(self,op=None):
        return context_ground_occlusion({'context_ground_occlusion':op or self.op}, self.p/'matte.json',self.stage,self.p/'stage.json',self.catalog,self.ids,self.depth)
    def test_only_buried_nonplant_faces_are_occluded(self):
        self.assertTrue(np.array_equal(self.run_op(),[[True,False],[False,False]]))
    def test_no_ground_tolerance_expansion_or_architecture_occlusion(self):
        for key,value in [('depth_tolerance',1),('occluding_indices',[11,12]),('occluded_samples',2)]:
            op=copy.deepcopy(self.op);op[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_op(op)
    def test_changed_or_alternate_baseline_pass_is_rejected(self):
        (self.p/'ids.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Missing or changed'):self.run_op()
    def test_proposed_lawn_is_not_treated_as_context_ground(self):
        self.catalog['objects'][2]['photo_kind']='bed';op=copy.deepcopy(self.op);op['occluding_indices']=[];op['occluded_samples']=0
        self.assertFalse(self.run_op(op).any())

if __name__=='__main__':unittest.main()
