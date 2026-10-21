"""Pure helpers for Blender snapshot contracts, also testable without Blender."""


def scalar_properties(value, exclude=()):
    result = {}
    for prop in getattr(getattr(value, 'bl_rna', None), 'properties', ()):
        key = prop.identifier
        if key in exclude or key == 'rna_type' or getattr(prop, 'type', None) in ('POINTER', 'COLLECTION'):
            continue
        item = getattr(value, key)
        if isinstance(item, (str, bool, int, float)):
            result[key] = item
        elif isinstance(item, set):
            result[key] = sorted(item)
        elif getattr(prop, 'is_array', False):
            result[key] = list(item)
    return result


def node_settings(node):
    # Sockets alone omit controls such as ColorRamp alpha and Math.use_clamp.
    ignored = ('name', 'label', 'location', 'select', 'show_options', 'show_preview',
               'show_texture', 'color', 'use_custom_color', 'width', 'height', 'dimensions')
    result = {'properties': scalar_properties(node, ignored)}
    if hasattr(node, 'color_ramp'):
        ramp = node.color_ramp
        result['color_ramp'] = {'interpolation': ramp.interpolation,
                                'color_mode': ramp.color_mode, 'hue_interpolation': ramp.hue_interpolation,
                                'elements': [{'position': e.position, 'color': list(e.color)} for e in ramp.elements]}
    if hasattr(node, 'mapping'):
        mapping = node.mapping
        result['curve_mapping'] = {'properties': scalar_properties(mapping),
                                   'curves': [[{'location': list(p.location), 'handle_type': p.handle_type}
                                               for p in curve.points] for curve in mapping.curves]}
    for key in ('texture_mapping', 'image_user'):
        if hasattr(node, key):
            result[key] = scalar_properties(getattr(node, key))
    return result


def opacity_sensitive_rgb(image_node):
    """Freeze RGB when it influences opacity, texture mapping or emission.

    Follow Color links through arithmetic/color nodes, but stop at opaque shader
    color/normal inputs. Image Alpha is always frozen separately.
    """
    pending = [link for socket in image_node.outputs if socket.name == 'Color'
               for link in socket.links]
    seen = set()
    while pending:
        link = pending.pop()
        node, socket = link.to_node, link.to_socket
        key = (id(node), socket.name)
        if key in seen:
            continue
        seen.add(key)
        if ((node.type == 'BSDF_PRINCIPLED' and socket.name in
             ('Alpha', 'Transmission Weight', 'Emission Color', 'Emission Strength')) or
                (node.type == 'BSDF_TRANSPARENT') or
                (node.type == 'MIX_SHADER' and socket.name == 'Fac') or
                (node.type == 'TEX_IMAGE' and socket.name == 'Vector') or
                node.type == 'EMISSION'):
            return True
        if node.type.startswith('BSDF_') or node.type in ('OUTPUT_MATERIAL', 'MIX_SHADER', 'ADD_SHADER'):
            continue
        pending.extend(edge for output in node.outputs for edge in output.links)
    return False


def camera_contract(ob):
    data = ob.data
    fields = ('lens', 'sensor_width', 'sensor_height', 'sensor_fit', 'shift_x', 'shift_y',
              'type', 'ortho_scale', 'clip_start', 'clip_end', 'panorama_type',
              'fisheye_fov', 'fisheye_lens', 'latitude_min', 'latitude_max',
              'longitude_min', 'longitude_max')
    result = {'matrix_world': [x for row in ob.matrix_world for x in row]}
    result.update({name: getattr(data, name) for name in fields if hasattr(data, name)})
    if hasattr(data, 'dof'):
        dof = data.dof
        result['depth_of_field'] = {name: getattr(dof, name) for name in
                                    ('use_dof', 'focus_distance', 'aperture_fstop',
                                     'aperture_blades', 'aperture_rotation', 'aperture_ratio')
                                    if hasattr(dof, name)}
        result['depth_of_field']['focus_object'] = dof.focus_object.name if dof.focus_object else None
    return result
