"""Synthetic photographic edge reconciliation and immutable source checks."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from handoff import record
from raster import load,save
from regional_edit import prepare,stitch,cleanup_edges

class CleanupEdges(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'test-output';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        self.original=np.full((64,64,3),12345,np.uint16);save(self.p/'source.png',self.original)
        mask=np.zeros((64,64),np.uint8);mask[12:52,12:52]=255;self.mask=mask>0;Image.fromarray(mask).save(self.p/'mask.png')
        prepare(self.p/'source.png',self.p/'mask.png',[8,8,56,56],self.p/'region',feather=0)
        Image.fromarray(np.full((48,48,3),240,np.uint8)).save(self.p/'generation.png')
        stitch([(self.p/'region/plan.json',self.p/'generation.png')],self.p/'stitched')
        (self.p/'inventory.json').write_text(json.dumps({'regions':[{'id':'tool','kind':'tool','description':'Synthetic cleanup target','mask':record(self.p/'mask.png')}]}))
    def test_softening_preserves_exact_source_and_raw_generation(self):
        before=record(self.p/'generation.png');result=cleanup_edges(self.p/'stitched/stitch-proof.json',self.p/'inventory.json',self.p/'out',4)
        final,alpha=load(result['candidate']['path']);self.assertIsNone(alpha)
        self.assertTrue(np.array_equal(final[~self.mask],self.original[~self.mask]))
        self.assertTrue(np.all(final[32,32]==240*257));self.assertTrue(np.all(final[12,32]>12345));self.assertTrue(np.all(final[12,32]<240*257))
        self.assertEqual(record(self.p/'generation.png'),before)
    def test_changed_stitch_and_unbounded_feather_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Bounded inward'):cleanup_edges(self.p/'stitched/stitch-proof.json',self.p/'inventory.json',self.p/'bad',65)
        (self.p/'generation.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Missing or changed'):cleanup_edges(self.p/'stitched/stitch-proof.json',self.p/'inventory.json',self.p/'bad',4)

if __name__=='__main__':unittest.main()
