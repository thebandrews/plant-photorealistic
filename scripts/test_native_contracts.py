"""Native graph/camera regressions using Blender-shaped synthetic objects."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from native_contracts import camera_contract, node_settings, opacity_sensitive_rgb


class Node:
    def __init__(self, kind, outputs=()):
        self.type=kind
        self.outputs=[types.SimpleNamespace(name=name,links=[]) for name in outputs]


def link(source, target, socket, output='Color'):
    edge=types.SimpleNamespace(to_node=target,to_socket=types.SimpleNamespace(name=socket))
    next(s for s in source.outputs if s.name==output).links.append(edge)


class NativeContracts(unittest.TestCase):
    def test_only_surface_rgb_can_change(self):
        image=Node('TEX_IMAGE',['Color','Alpha']);shader=Node('BSDF_PRINCIPLED',['BSDF'])
        link(image,shader,'Base Color');self.assertFalse(opacity_sensitive_rgb(image))
        link(image,shader,'Alpha');self.assertTrue(opacity_sensitive_rgb(image))

    def test_rgb_through_color_nodes_into_opacity_is_frozen(self):
        image=Node('TEX_IMAGE',['Color']);math=Node('MATH',['Value']);shader=Node('BSDF_PRINCIPLED')
        link(image,math,'Value');link(math,shader,'Alpha',output='Value')
        self.assertTrue(opacity_sensitive_rgb(image))

    def test_alpha_mapping_remains_separate_from_permitted_color(self):
        image=Node('TEX_IMAGE',['Color','Alpha']);mix=Node('MIX_SHADER')
        link(image,mix,'Fac',output='Alpha')
        self.assertFalse(opacity_sensitive_rgb(image))

    def test_rgb_driving_transparency_or_uv_mapping_is_frozen(self):
        for kind,socket in [('BSDF_TRANSPARENT','Color'),('MIX_SHADER','Fac'),('TEX_IMAGE','Vector'),('BSDF_PRINCIPLED','Emission Color')]:
            with self.subTest(kind=kind,socket=socket):
                image=Node('TEX_IMAGE',['Color']);link(image,Node(kind),socket)
                self.assertTrue(opacity_sensitive_rgb(image))

    def test_actual_material_contract_detects_opacity_rgb_changes(self):
        with patch.dict(sys.modules,{'bpy':types.ModuleType('bpy')}):
            spec=importlib.util.spec_from_file_location('snapshot_test',Path(__file__).with_name('native_plant_snapshot.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class Pixels(list):
            def foreach_get(self, target):
                for i,value in enumerate(self):target[i]=value
        image=Node('TEX_IMAGE',['Color']);image.name='Image';image.bl_idname='ShaderNodeTexImage';image.inputs=[]
        pixels=Pixels([.1,.2,.3,1]);image.image=types.SimpleNamespace(size=[1,1],pixels=pixels,as_pointer=lambda:1)
        shader=Node('BSDF_PRINCIPLED');shader.name='Shader';shader.bl_idname='ShaderNodeBsdfPrincipled';shader.inputs=[]
        link(image,shader,'Alpha')
        mat=types.SimpleNamespace(use_nodes=True,node_tree=types.SimpleNamespace(nodes=[image,shader],links=[]),use_backface_culling=False)
        before=module.material_contract(mat,{})
        pixels[0]=.9
        after=module.material_contract(mat,{})
        self.assertNotEqual(before,after)

    def test_camera_dof_sensor_fit_and_clip_changes_are_detected(self):
        dof=types.SimpleNamespace(use_dof=False,focus_distance=5,aperture_fstop=8,focus_object=None)
        camera=types.SimpleNamespace(matrix_world=[[1,0],[0,1]],data=types.SimpleNamespace(type='PERSP',lens=50,sensor_fit='HORIZONTAL',clip_start=.1,dof=dof))
        before=camera_contract(camera);dof.use_dof=True;dof.focus_distance=.01;dof.aperture_fstop=.1
        self.assertNotEqual(before,camera_contract(camera))
        before=camera_contract(camera);camera.data.sensor_fit='VERTICAL'
        self.assertNotEqual(before,camera_contract(camera))

    def test_color_ramp_alpha_controls_are_frozen(self):
        ramp=types.SimpleNamespace(interpolation='LINEAR',color_mode='RGB',hue_interpolation='NEAR',
                                   elements=[types.SimpleNamespace(position=0,color=[1,1,1,.8])])
        node=types.SimpleNamespace(color_ramp=ramp)
        before=node_settings(node);ramp.elements[0].color[3]=.5
        self.assertNotEqual(before,node_settings(node))

    def test_shader_boolean_settings_are_frozen(self):
        prop=types.SimpleNamespace(identifier='use_clamp',type='BOOLEAN',is_array=False)
        node=types.SimpleNamespace(bl_rna=types.SimpleNamespace(properties=[prop]),use_clamp=False)
        before=node_settings(node);node.use_clamp=True
        self.assertNotEqual(before,node_settings(node))


if __name__=='__main__':unittest.main()
