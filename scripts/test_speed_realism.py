"""Precision and permission regressions for faster photographic finishing."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import png
from PIL import Image

import raster
import regional_edit
import botanical_refinement
import photo_finish
from handoff import read, record
from test_contract import save
import test_finishing as fixtures


def local_temp():
    root=Path(__file__).resolve().parents[1]/'test-output';root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class RasterPrecision(unittest.TestCase):
    def test_native_and_portable_decoders_keep_all_rgb16_and_hidden_rgb(self):
        rng=np.random.default_rng(13)
        with local_temp() as tmp:
            p=Path(tmp)/'non-ascii-葉.png'
            for dtype in [np.uint8,np.uint16]:
                rgb=rng.integers(0,np.iinfo(dtype).max+1,(19,23,3),dtype=dtype)
                for alpha in [None,rng.integers(0,np.iinfo(dtype).max+1,(19,23),dtype=dtype)]:
                    if alpha is not None: alpha[0,0]=0
                    raster.save(p,rgb,alpha)
                    for decoder in [raster.cv2,None]:
                        with patch.object(raster,'cv2',decoder): actual,aa=raster.load(p)
                        self.assertTrue(np.array_equal(actual,rgb));self.assertEqual(actual.dtype,dtype)
                        if alpha is None:self.assertIsNone(aa)
                        else:self.assertTrue(np.array_equal(aa,alpha))

    def test_gray_alpha16_keeps_precision(self):
        with local_temp() as tmp:
            p=Path(tmp)/'gray.png';g=np.array([[1,65534],[32001,12345]],np.uint16);a=np.array([[0,32767],[65535,1]],np.uint16)
            with p.open('wb') as f:png.Writer(2,2,greyscale=True,alpha=True,bitdepth=16).write(f,np.stack([g,a],-1).reshape(2,-1))
            for decoder in [raster.cv2,None]:
                with patch.object(raster,'cv2',decoder):rgb,aa=raster.load(p)
                self.assertTrue(np.array_equal(rgb,np.repeat(g[...,None],3,2)));self.assertTrue(np.array_equal(aa,a))

    def test_block_blend_matches_monolithic_arithmetic(self):
        rng=np.random.default_rng(2);fg=rng.integers(0,65536,(279,73,3),dtype=np.uint16);bg=rng.integers(0,65536,fg.shape,dtype=np.uint16);a=rng.integers(0,256,fg.shape[:2],dtype=np.uint8)
        self.assertTrue(np.array_equal(photo_finish.blend(fg,bg,a),photo_finish._blend_block(fg,bg,a)))

    def test_low_bit_palette_retains_legacy_rgb8_expansion(self):
        with local_temp() as tmp:
            p=Path(tmp)/'palette.png'
            with p.open('wb') as f:png.Writer(2,1,palette=[(1,2,3,0),(4,5,6,255)],bitdepth=1).write(f,[[0,1]])
            rgb,alpha=raster.load(p)
            self.assertTrue(np.array_equal(rgb,np.array([[[1,2,3],[4,5,6]]],np.uint8)))
            self.assertTrue(np.array_equal(alpha,[[0,255]]))


class RegionalPermissions(unittest.TestCase):
    def setUp(self):
        self.tmp=local_temp();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        self.rgb=np.full((16,20,3),12345,np.uint16);raster.save(self.p/'source.png',self.rgb)
        self.mask=np.zeros((16,20),np.uint8);self.mask[4:12,5:15]=128;Image.fromarray(self.mask).save(self.p/'mask.png')
        regional_edit.prepare(self.p/'source.png',self.p/'mask.png',[3,2,17,14],self.p/'plan',feather=0)
        Image.fromarray(np.full((12,14,3),240,np.uint8)).save(self.p/'raw.png')

    def test_transition_strength_applied_once_and_background_exact(self):
        proof=regional_edit.stitch([(self.p/'plan/plan.json',self.p/'raw.png')],self.p/'stitched');self.assertEqual(regional_edit.verify(proof,self.p)['status'],'verified')
        raw,_=raster.load(proof['candidate']['path']);self.assertTrue(np.array_equal(raw[self.mask==0],self.rgb[self.mask==0]));self.assertTrue(np.all(raw[self.mask>0]==240*257))
        final=photo_finish.blend(raw,self.rgb,self.mask);expected=photo_finish.blend(np.full_like(self.rgb,240*257),self.rgb,self.mask)
        self.assertTrue(np.array_equal(final,expected))

    def test_wrong_size_and_tampered_protected_sample_rejected(self):
        Image.new('RGB',(13,12)).save(self.p/'wrong.png')
        with self.assertRaisesRegex(ValueError,'never resize'):regional_edit.stitch([(self.p/'plan/plan.json',self.p/'wrong.png')],self.p/'bad')
        proof=regional_edit.stitch([(self.p/'plan/plan.json',self.p/'raw.png')],self.p/'stitched');raw,_=raster.load(proof['candidate']['path']);raw[0,0,0]+=1;raster.save(proof['candidate']['path'],raw);proof['candidate']=record(proof['candidate']['path'])
        with self.assertRaisesRegex(ValueError,'reconstruction differs'):regional_edit.verify(proof,self.p)


class RealismContracts(unittest.TestCase):
    def test_new_surfaces_require_realism_and_litter_but_legacy_remains_verifiable(self):
        c=fixtures.PhotoFinishing();c.setUp();self.addCleanup(c.doCleanups)
        for name in photo_finish.SURFACE_CHECKS:
            bad=copy.deepcopy(c.review);bad['photo_review']['checks'].pop(name)
            with self.assertRaisesRegex(ValueError,'review failed'):photo_finish.check_review(bad,c.p)
        legacy=read(c.p/'finish-plan.json');legacy.pop('surface_realism_review');save(c.p/'legacy-plan.json',legacy)
        proof=photo_finish.compose(c.p/'legacy-plan.json',c.p/'legacy-final',c.p/'cleanup-raw.png',c.p/'transition-raw.png')
        review=copy.deepcopy(c.review);review['candidate']=proof['candidate'];review['finish_proof']=record(c.p/'legacy-final/finish-proof.json')
        for name in photo_finish.SURFACE_CHECKS:review['photo_review']['checks'].pop(name)
        self.assertEqual(photo_finish.check_review(review,c.p)['status'],'accepted')

    def test_foliage_fill_requires_volume_review_and_keeps_roots(self):
        c=fixtures.BotanicalRefinement();c.setUp();self.addCleanup(c.doCleanups)
        botanical_refinement.prepare(c.p/'handoff.json',c.p/'botanical-before.json',['agave-1'],c.p/'fullness-plan.json',foliage_fill=True)
        review=copy.deepcopy(c.review);review['refinement_plan']=record(c.p/'fullness-plan.json')
        with self.assertRaisesRegex(ValueError,'Foliage fullness'):botanical_refinement.check_review(review,c.p)
        review['recognition']['geometry'].update(dict.fromkeys(('foliage_fullness','natural_asymmetry','leaf_volume_detail'),'pass'))
        self.assertEqual(botanical_refinement.check_review(review,c.p)['status'],'accepted')
        original=copy.deepcopy(c.before);after=copy.deepcopy(c.after);botanical_refinement.compare(c.before,c.after,['agave-1']);self.assertEqual(c.before,original);self.assertEqual(c.after,after)
        after['invariants']['plants'][0]['root_world'][0]=1
        with self.assertRaisesRegex(ValueError,'changed'):botanical_refinement.compare(c.before,after,['agave-1'])

    def test_genuine_renderer_alpha_accepts_leaf_holes_rejects_changed_coverage(self):
        c=fixtures.BotanicalRefinement();c.setUp();self.addCleanup(c.doCleanups)
        rgb=np.full((32,32,3),20123,np.uint16);alpha=np.full((32,32),65535,np.uint16);alpha[6,6]=0;alpha[6,7]=1
        raster.save(c.p/'native-rgba.png',rgb,alpha);stage=copy.deepcopy(c.review);stage['candidate']=record(c.p/'native-rgba.png')
        after=read(c.p/'botanical-after.json');after['invariants']['settings']['film_transparent']=True;save(c.p/'rgba-after.json',after);stage['native_after']=record(c.p/'rgba-after.json')
        proof=read(c.p/'botanical-render.json');proof['camera_only_export_overrides']=[{'name':'house','visible_camera':False}];save(c.p/'rgba-render.json',proof);stage['render_proof']=record(c.p/'rgba-render.json')
        catalog={'objects':[{'name':'plant','type':'MESH','kind':'plant'},{'name':'house','type':'MESH','kind':'existing_context'}]};matte={'renderer_rgba':stage['candidate']};q=np.ceil(alpha.astype(float)*255/65535).astype(np.uint8)
        photo_finish.verify_renderer_rgba(matte,c.p/'matte.json',stage,c.p/'stage.json',catalog,q,rgb)
        q[6,6]=1
        with self.assertRaisesRegex(ValueError,'coverage was modified'):photo_finish.verify_renderer_rgba(matte,c.p/'matte.json',stage,c.p/'stage.json',catalog,q,rgb)
        q[6,6]=0;proof['camera_only_export_overrides']=[];save(c.p/'rgba-render.json',proof);stage['render_proof']=record(c.p/'rgba-render.json')
        with self.assertRaisesRegex(ValueError,'every excluded'):photo_finish.verify_renderer_rgba(matte,c.p/'matte.json',stage,c.p/'stage.json',catalog,q,rgb)


if __name__=='__main__':unittest.main()
