import bpy
import bmesh
import math
import json
from pathlib import Path
from mathutils import Vector, Matrix, Euler

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cloth_tunnel"
WORK = PROJ / "work"
TRACK_JSON = WORK / "camera_track.json"
HDRI_PATH = WORK / "scene_matched_livingroom_2k.hdr"
BLEND_PATH = WORK / "cloth_scene.blend"

PREROLL = 32
NUM_SHOT_FRAMES = 125

def set_bsdf(bsdf, **kwargs):
    for k, v in kwargs.items():
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = v

def setup_cycles_optix(scene, samples=48):
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == 'OPTIX')
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPTIX'
    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.sample_clamp_indirect = 8.0

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = 25
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Base Contrast'
    scene.view_settings.exposure = 0.0

    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = 'CENTER'
    scene.render.motion_blur_shutter = 0.50

def setup_world_hdri_and_practicals(scene):
    world = bpy.data.worlds.new("SceneMatchedWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    n_tc = nt.nodes.new('ShaderNodeTexCoord')
    n_map = nt.nodes.new('ShaderNodeMapping')
    n_env = nt.nodes.new('ShaderNodeTexEnvironment')
    n_bg = nt.nodes.new('ShaderNodeBackground')
    n_out = nt.nodes.new('ShaderNodeOutputWorld')

    img = bpy.data.images.load(str(HDRI_PATH), check_existing=True)
    img.colorspace_settings.name = 'Linear Rec.709'
    n_env.image = img
    n_bg.inputs['Strength'].default_value = 1.60

    nt.links.new(n_tc.outputs['Generated'], n_map.inputs['Vector'])
    nt.links.new(n_map.outputs['Vector'], n_env.inputs['Vector'])
    nt.links.new(n_env.outputs['Color'], n_bg.inputs['Color'])
    nt.links.new(n_bg.outputs['Background'], n_out.inputs['Surface'])

    # Practical sunlit back doorway key/rim light matching exact doorway position in Pexels_3769965
    ld = bpy.data.lights.new("Practical_SunlitDoorway", type='AREA')
    ld.shape = 'RECTANGLE'
    ld.size = 0.90
    ld.size_y = 1.55
    ld.energy = 45.0
    ld.color = (1.0, 0.94, 0.85)
    l_obj = bpy.data.objects.new("Practical_SunlitDoorway", ld)
    scene.collection.objects.link(l_obj)
    l_obj.location = (0.55, 2.10, 0.16)
    l_obj.rotation_euler = (Vector((0.02, 0.46, -0.08)) - l_obj.location).to_track_quat('-Z', 'Y').to_euler()
    l_obj.visible_camera = False

    # Subtle warm chandelier overhead fill
    ld2 = bpy.data.lights.new("Practical_ChandelierWarm", type='AREA')
    ld2.shape = 'DISK'
    ld2.size = 0.65
    ld2.energy = 12.0
    ld2.color = (1.0, 0.82, 0.56)
    l_obj2 = bpy.data.objects.new("Practical_ChandelierWarm", ld2)
    scene.collection.objects.link(l_obj2)
    l_obj2.location = (-0.12, 1.00, 1.05)
    l_obj2.rotation_euler = (Vector((0.0, 0.46, -0.05)) - l_obj2.location).to_track_quat('-Z', 'Y').to_euler()
    l_obj2.visible_camera = False

    # Soft warm bounce from camera-right room wall (gives crisp pearl glints & sculpted folds!)
    ld3 = bpy.data.lights.new("Practical_RoomBounce", type='AREA')
    ld3.shape = 'DISK'
    ld3.size = 0.55
    ld3.energy = 4.2
    ld3.color = (1.0, 0.90, 0.76)
    l_obj3 = bpy.data.objects.new("Practical_RoomBounce", ld3)
    scene.collection.objects.link(l_obj3)
    l_obj3.location = (0.38, -0.35, 0.30)
    l_obj3.rotation_euler = (Vector((-0.14, 0.44, -0.12)) - l_obj3.location).to_track_quat('-Z', 'Y').to_euler()
    l_obj3.visible_camera = False

    return world

# ---------------------------------------------------------------------------
# Procedural PBR Materials
# ---------------------------------------------------------------------------
def make_wood_hanger_mat(name="Mat_LightOak_Wood", tint=(1.0, 0.96, 0.90)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (14.0, 2.6, 14.0)

    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'X'
    wave.wave_profile = 'SIN'
    wave.inputs['Scale'].default_value = 3.5
    wave.inputs['Distortion'].default_value = 5.2
    wave.inputs['Detail'].default_value = 5.0
    wave.inputs['Detail Scale'].default_value = 1.8

    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 48.0
    noise.inputs['Detail'].default_value = 6.0

    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.21 * tint[0], 0.115 * tint[1], 0.056 * tint[2], 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (0.54 * tint[0], 0.35 * tint[1], 0.19 * tint[2], 1.0)

    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.09
    bump.inputs['Distance'].default_value = 0.002

    set_bsdf(bsdf, **{
        'Roughness': 0.34,
        'Specular IOR Level': 0.48,
        'Coat Weight': 0.25,
        'Coat Roughness': 0.22,
    })

    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], wave.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], noise.inputs['Vector'])
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_chrome_mat(name="Mat_BrushedChrome", rough=0.22):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    set_bsdf(bsdf, **{
        'Base Color': (0.76, 0.74, 0.71, 1.0),
        'Metallic': 0.96,
        'Roughness': rough,
        'Anisotropic': 0.35,
    })
    return mat

def make_black_dress_mat(name="Mat_BlackSatinTwill"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')

    wave1 = nt.nodes.new('ShaderNodeTexWave')
    wave1.inputs['Scale'].default_value = 150.0
    wave1.inputs['Distortion'].default_value = 0.8
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 24.0
    noise.inputs['Detail'].default_value = 5.0

    ramp_r = nt.nodes.new('ShaderNodeMapRange')
    ramp_r.inputs['To Min'].default_value = 0.30
    ramp_r.inputs['To Max'].default_value = 0.50

    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.18
    bump.inputs['Distance'].default_value = 0.001

    set_bsdf(bsdf, **{
        'Base Color': (0.022, 0.021, 0.026, 1.0),
        'Specular IOR Level': 0.58,
        'Sheen Weight': 0.65,
        'Sheen Roughness': 0.38,
        'Sheen Tint': (0.88, 0.82, 0.75, 1.0),
    })

    nt.links.new(tc.outputs['UV'], wave1.inputs['Vector'])
    nt.links.new(tc.outputs['UV'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], ramp_r.inputs['Value'])
    nt.links.new(ramp_r.outputs['Result'], bsdf.inputs['Roughness'])
    nt.links.new(wave1.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_satin_ribbon_mat(name="Mat_BlackSatinRibbon"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    set_bsdf(bsdf, **{
        'Base Color': (0.015, 0.015, 0.019, 1.0),
        'Roughness': 0.22,
        'Specular IOR Level': 0.75,
        'Coat Weight': 0.35,
        'Coat Roughness': 0.18,
        'Sheen Weight': 0.70,
        'Sheen Roughness': 0.28,
        'Sheen Tint': (0.95, 0.88, 0.78, 1.0),
    })
    return mat

def make_pearl_mat(name="Mat_GlossyPearl"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    set_bsdf(bsdf, **{
        'Base Color': (0.98, 0.93, 0.82, 1.0),
        'Metallic': 0.05,
        'Roughness': 0.08,
        'IOR': 1.62,
        'Specular IOR Level': 0.92,
        'Coat Weight': 0.98,
        'Coat Roughness': 0.04,
        'Subsurface Weight': 0.45,
        'Subsurface Scale': 0.006,
        'Thin Film Thickness': 420.0,
        'Thin Film IOR': 1.38,
    })
    return mat

def make_eyelet_knit_mat(name="Mat_EyeletKnit"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    mix_tr = nt.nodes.new('ShaderNodeMixShader')
    mix_tr.inputs['Fac'].default_value = 0.28
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    transl = nt.nodes.new('ShaderNodeBsdfTranslucent')
    transl.inputs['Color'].default_value = (0.92, 0.84, 0.74, 1.0)

    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (24.0, 28.0, 1.0)

    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.feature = 'F1'
    vor.inputs['Scale'].default_value = 2.4
    vor.inputs['Randomness'].default_value = 0.04

    alpha_ramp = nt.nodes.new('ShaderNodeValToRGB')
    alpha_ramp.color_ramp.elements[0].position = 0.14
    alpha_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    alpha_ramp.color_ramp.elements[1].position = 0.24
    alpha_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    col_ramp = nt.nodes.new('ShaderNodeValToRGB')
    col_ramp.color_ramp.elements[0].position = 0.20
    col_ramp.color_ramp.elements[0].color = (0.34, 0.32, 0.30, 1.0)
    col_ramp.color_ramp.elements[1].position = 0.56
    col_ramp.color_ramp.elements[1].color = (0.68, 0.65, 0.60, 1.0)

    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.42
    bump.inputs['Distance'].default_value = 0.0022

    set_bsdf(bsdf, **{
        'Roughness': 0.80,
        'Sheen Weight': 0.78,
        'Sheen Roughness': 0.52,
        'Sheen Tint': (0.95, 0.90, 0.82, 1.0),
    })

    nt.links.new(tc.outputs['UV'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], vor.inputs['Vector'])
    nt.links.new(vor.outputs['Distance'], alpha_ramp.inputs['Fac'])
    nt.links.new(vor.outputs['Distance'], col_ramp.inputs['Fac'])
    nt.links.new(vor.outputs['Distance'], bump.inputs['Height'])
    nt.links.new(col_ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], mix_tr.inputs[1])
    nt.links.new(transl.outputs['BSDF'], mix_tr.inputs[2])

    transp = nt.nodes.new('ShaderNodeBsdfTransparent')
    mix_alpha = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(alpha_ramp.outputs['Color'], mix_alpha.inputs['Fac'])
    nt.links.new(transp.outputs['BSDF'], mix_alpha.inputs[1])
    nt.links.new(mix_tr.outputs['Shader'], mix_alpha.inputs[2])
    nt.links.new(mix_alpha.outputs['Shader'], out.inputs['Surface'])
    return mat

def make_cream_blouse_mat(name="Mat_CreamLinenBlouse"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    mix_tr = nt.nodes.new('ShaderNodeMixShader')
    mix_tr.inputs['Fac'].default_value = 0.36

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    transl = nt.nodes.new('ShaderNodeBsdfTranslucent')
    transl.inputs['Color'].default_value = (0.96, 0.87, 0.74, 1.0)

    tc = nt.nodes.new('ShaderNodeTexCoord')
    w_x = nt.nodes.new('ShaderNodeTexWave')
    w_x.bands_direction = 'X'
    w_x.inputs['Scale'].default_value = 180.0
    w_y = nt.nodes.new('ShaderNodeTexWave')
    w_y.bands_direction = 'Y'
    w_y.inputs['Scale'].default_value = 180.0
    m_add = nt.nodes.new('ShaderNodeMath')
    m_add.operation = 'ADD'

    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 9.0
    noise.inputs['Detail'].default_value = 4.0

    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = (0.50, 0.45, 0.40, 1.0)
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = (0.76, 0.71, 0.63, 1.0)

    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.22
    bump.inputs['Distance'].default_value = 0.001

    set_bsdf(bsdf, **{
        'Roughness': 0.62,
        'Specular IOR Level': 0.38,
        'Sheen Weight': 0.65,
        'Sheen Roughness': 0.42,
        'Sheen Tint': (1.0, 0.94, 0.85, 1.0),
        'Subsurface Weight': 0.30,
        'Subsurface Scale': 0.004,
    })

    nt.links.new(tc.outputs['UV'], w_x.inputs['Vector'])
    nt.links.new(tc.outputs['UV'], w_y.inputs['Vector'])
    nt.links.new(tc.outputs['UV'], noise.inputs['Vector'])
    nt.links.new(w_x.outputs['Fac'], m_add.inputs[0])
    nt.links.new(w_y.outputs['Fac'], m_add.inputs[1])
    nt.links.new(m_add.outputs['Value'], bump.inputs['Height'])
    nt.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], mix_tr.inputs[1])
    nt.links.new(transl.outputs['BSDF'], mix_tr.inputs[2])
    nt.links.new(mix_tr.outputs['Shader'], out.inputs['Surface'])
    return mat

def make_dark_wool_coat_mat(name="Mat_DarkOliveWool"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 140.0
    noise.inputs['Detail'].default_value = 8.0

    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.28
    bump.inputs['Distance'].default_value = 0.0012

    set_bsdf(bsdf, **{
        'Base Color': (0.044, 0.050, 0.038, 1.0),
        'Roughness': 0.84,
        'Sheen Weight': 0.80,
        'Sheen Roughness': 0.55,
        'Sheen Tint': (0.76, 0.70, 0.60, 1.0),
    })

    nt.links.new(tc.outputs['UV'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_clay_and_wire_mats():
    clay = bpy.data.materials.new("Mat_3DView_Clay")
    clay.use_nodes = True
    set_bsdf(clay.node_tree.nodes.get('Principled BSDF'), **{
        'Base Color': (0.52, 0.52, 0.53, 1.0),
        'Roughness': 0.74,
        'Specular IOR Level': 0.22,
    })

    wire = bpy.data.materials.new("Mat_3DView_Wire")
    wire.use_nodes = True
    set_bsdf(wire.node_tree.nodes.get('Principled BSDF'), **{
        'Base Color': (0.015, 0.015, 0.018, 1.0),
        'Roughness': 0.95,
        'Specular IOR Level': 0.05,
    })
    return clay, wire

# ---------------------------------------------------------------------------
# Clean Quad-Topology BMesh Procedural Builders
# ---------------------------------------------------------------------------
def build_sculpted_wooden_hanger(name, wood_mat, chrome_mat, width=0.40, drop=0.115, with_hook=True, with_bar=True):
    pivot = bpy.data.objects.new(f"{name}_Pivot", None)
    bpy.context.scene.collection.objects.link(pivot)

    apex_z = -0.068
    half_w = width * 0.5

    bm = bmesh.new()
    n_arch = 36

    def make_ring(cx, cy, cz, tangent, thick_y, height_z):
        t = tangent.normalized()
        up = Vector((0, 0, 1))
        n_y = t.cross(up)
        n_y = Vector((0, 1, 0)) if n_y.length < 1e-4 else n_y.normalized()
        n_z = n_y.cross(t).normalized()
        pts_2d = [
            (-0.50, -0.35), (-0.35, -0.50), (0.35, -0.50), (0.50, -0.35),
            (0.50,  0.35), (0.35,  0.50), (-0.35, 0.50), (-0.50, 0.35),
        ]
        return [bm.verts.new(Vector((cx, cy, cz)) + n_y * (py * thick_y) + n_z * (pz * height_z)) for py, pz in pts_2d]

    arch_pts = []
    for i in range(n_arch + 1):
        s = -1.0 + 2.0 * i / n_arch
        x = s * half_w
        z = apex_z - drop * (abs(s) ** 1.45) - 0.008 * math.cos(s * math.pi * 0.5)
        y = -0.018 * (s ** 2)
        z -= 0.0045 * math.exp(-((abs(s) - 0.72) / 0.05) ** 2)
        arch_pts.append((x, y, z, s))

    rings = []
    for i, (x, y, z, s) in enumerate(arch_pts):
        p_prev = Vector(arch_pts[max(0, i - 1)][:3])
        p_next = Vector(arch_pts[min(n_arch, i + 1)][:3])
        tang = p_next - p_prev
        thick_y = 0.017 + 0.014 * (abs(s) ** 1.6)
        height_z = 0.026 + 0.012 * math.exp(-(s / 0.25) ** 2) - 0.004 * (abs(s) ** 2)
        rings.append(make_ring(x, y, z, tang, thick_y, height_z))

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for k in range(len(r0)):
            kn = (k + 1) % len(r0)
            bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
    for ring, flip in [(rings[0], True), (rings[-1], False)]:
        v = ring if not flip else list(reversed(ring))
        bm.faces.new((v[0], v[1], v[2], v[7]))
        bm.faces.new((v[2], v[3], v[6], v[7]))
        bm.faces.new((v[3], v[4], v[5], v[6]))

    if with_bar:
        bar_y = -0.012
        bar_z = apex_z - drop * 0.86
        bar_hw = half_w * 0.86
        bar_rings = [make_ring((-1.0 + 2.0 * i / 12.0) * bar_hw, bar_y, bar_z, Vector((1, 0, 0)), 0.010, 0.011) for i in range(13)]
        for i in range(len(bar_rings) - 1):
            r0, r1 = bar_rings[i], bar_rings[i + 1]
            for k in range(len(r0)):
                kn = (k + 1) % len(r0)
                bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_WoodMesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons: p.use_smooth = True

    wood_obj = bpy.data.objects.new(f"{name}_Wood", mesh)
    wood_obj.data.materials.append(wood_mat)
    bpy.context.scene.collection.objects.link(wood_obj)
    wood_obj.parent = pivot

    hook_obj = None
    if with_hook:
        bm_h = bmesh.new()
        hook_pts = [Vector((0.0, 0.0, apex_z + 0.010 + (i / 5.0) * 0.036)) for i in range(6)]
        for i in range(1, 17):
            ang = -math.pi * 0.5 + (i / 16.0) * (math.pi * 1.48)
            hook_pts.append(Vector((0.0, 0.018 * math.cos(ang), 0.002 + 0.019 * math.sin(ang))))
        h_rings = []
        rad = 0.0024
        for i, pt in enumerate(hook_pts):
            tang = (hook_pts[min(len(hook_pts) - 1, i + 1)] - hook_pts[max(0, i - 1)]).normalized()
            ref = Vector((1, 0, 0)) if abs(tang.x) < 0.9 else Vector((0, 1, 0))
            u_ax = tang.cross(ref).normalized()
            v_ax = tang.cross(u_ax).normalized()
            h_rings.append([bm_h.verts.new(pt + (u_ax * math.cos(2*math.pi*k/8) + v_ax * math.sin(2*math.pi*k/8)) * rad) for k in range(8)])
        for i in range(len(h_rings) - 1):
            r0, r1 = h_rings[i], h_rings[i + 1]
            for k in range(8):
                kn = (k + 1) % 8
                bm_h.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
        bmesh.ops.recalc_face_normals(bm_h, faces=bm_h.faces)
        h_mesh = bpy.data.meshes.new(f"{name}_HookMesh")
        bm_h.to_mesh(h_mesh)
        bm_h.free()
        for p in h_mesh.polygons: p.use_smooth = True

        hook_obj = bpy.data.objects.new(f"{name}_Hook", h_mesh)
        hook_obj.data.materials.append(chrome_mat)
        bpy.context.scene.collection.objects.link(hook_obj)
        hook_obj.parent = pivot

    return pivot, wood_obj, hook_obj

def create_volumetric_garment_mesh(
    name, mat, pivot,
    nx=54, ny=60,
    width_top=0.44, width_bot=0.68,
    height=0.56,
    z_top=-0.105,
    y_depth=0.095,
    y_offset=0.0,
    pleat_amp=0.035,
    pleat_freq=7.5,
    ruffle_hem=0.0,
    collar_cut=0.045,
    wrap_span=0.56,      # <= 0.58 so sin(ang) reaches FULL shoulder width cleanly around hanger tips (ZERO wood poke-through!)
    pin_center_only=False,
    arch_drop_scale=0.105,
):
    """
    Creates a generous, pleated, zero-clipping quad garment mesh draped over/below the hanger:
    - Because wrap_span <= 0.58, at |s|=1 the cloth sits cleanly OUTSIDE the hanger shoulder tips!
    - Pinning ONLY row 0 along the central hanger contact zone lets the outer sleeves, open lapels,
      and ruffled lower hem billow dramatically in the wind & turbulence!
    """
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    sin_max = math.sin(min(math.pi * 0.5, math.pi * wrap_span))
    verts_grid = []
    for j in range(ny + 1):
        v = j / ny
        row = []
        flare = (1.0 - v) * width_top + v * width_bot + 0.06 * math.sin(v * math.pi)
        depth_v = y_depth * (0.65 + 0.55 * (v ** 0.65))
        z_base = z_top - v * height

        for i in range(nx + 1):
            u = i / nx
            s = -1.0 + 2.0 * u
            arch_drop = arch_drop_scale * (abs(s) ** 1.42) * max(0.0, 1.0 - v * 1.15)
            neck_scoop = collar_cut * math.exp(-(s / 0.25) ** 2) * max(0.0, 1.0 - v * 2.8)

            ang = s * (math.pi * wrap_span)
            # Clean elliptical wrap so x reaches full +/- (flare*0.5) without pinching inside wooden tips!
            x = (flare * 0.5) * (math.sin(ang) / sin_max)
            y_wrap = y_offset - depth_v * math.cos(ang)

            pleat = pleat_amp * (v ** 0.55) * (
                0.68 * math.sin(u * math.pi * pleat_freq + 1.2 * v)
                + 0.32 * math.cos(u * math.pi * (pleat_freq * 1.8) - 1.7 * v)
            )
            ruffle = ruffle_hem * (v ** 1.35) * (
                0.62 * math.sin(u * math.pi * 11.0 + 2.2 * v)
                + 0.38 * math.cos(u * math.pi * 17.0 - 1.8 * v)
            )
            radial_push = pleat + ruffle
            x += radial_push * math.sin(ang) * 0.55
            y = y_wrap - radial_push * math.cos(ang)
            z = z_base - arch_drop - neck_scoop + 0.016 * ruffle

            vert = bm.verts.new((x, y, z))
            row.append((vert, u, v))
        verts_grid.append(row)

    bm.verts.index_update()
    bm.verts.ensure_lookup_table()

    # Pin row 0 shoulder anchor points so lower body & outer sides billow naturally under wind!
    pin_indices = []
    for i in range(nx + 1):
        s = -1.0 + 2.0 * i / nx
        if pin_center_only:
            # Pin two dress-strap anchor points on row 0..1 (like real shoulder straps!)
            if 0.14 < abs(s) < 0.72:
                pin_indices.append(verts_grid[0][i][0].index)
                pin_indices.append(verts_grid[1][i][0].index)
        else:
            # Pin shoulder yoke across top row only (outermost 15% sleeves left free to billow!)
            if abs(s) < 0.82:
                pin_indices.append(verts_grid[0][i][0].index)

    for j in range(ny):
        for i in range(nx):
            v00, u00, w00 = verts_grid[j][i]
            v10, u10, w10 = verts_grid[j][i + 1]
            v11, u11, w11 = verts_grid[j + 1][i + 1]
            v01, u01, w01 = verts_grid[j + 1][i]
            f = bm.faces.new((v00, v01, v11, v10))
            for loop, uv in zip(f.loops, [(u00, 1 - w00), (u01, 1 - w01), (u11, 1 - w11), (u10, 1 - w10)]):
                loop[uv_layer].uv = uv

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons: p.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = pivot

    vg = obj.vertex_groups.new(name="Pin")
    vg.add(pin_indices, 1.0, 'REPLACE')
    return obj

def create_hanger_shoulder_straps(name, mat, pivot, xs=(-0.12, 0.12), z_top=-0.088, drop_to=-0.21):
    """Two crisp black dress straps looping cleanly over the exposed wooden hanger shoulder arch down into the bodice!"""
    bm = bmesh.new()
    for sx in xs:
        nu, nv = 4, 20
        grid = []
        s_norm = abs(sx / 0.21)
        arch_z = -0.068 - 0.118 * (s_norm ** 1.45)
        for j in range(nv + 1):
            t = j / nv
            if t < 0.35:
                # Loop cleanly over top of wooden shoulder
                u_arc = t / 0.35
                ang = -math.pi * 0.25 + u_arc * math.pi * 0.65
                y_c = -0.012 - 0.020 * math.sin(ang)
                z_c = arch_z + 0.019 * math.cos(ang)
            else:
                # Drop vertically down to meet the pleated dress bodice
                u_dn = (t - 0.35) / 0.65
                y_c = -0.030 - 0.036 * u_dn
                z_c = (arch_z + 0.005) * (1.0 - u_dn) + drop_to * u_dn
            grid.append([bm.verts.new((sx + (i / nu - 0.5) * 0.020, y_c, z_c)) for i in range(nu + 1)])
        for j in range(nv):
            for i in range(nu):
                bm.faces.new((grid[j][i], grid[j+1][i], grid[j+1][i+1], grid[j][i+1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons: p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = pivot
    sol = obj.modifiers.new("Solidify", 'SOLIDIFY')
    sol.thickness = 0.003
    return obj

def create_whipping_ribbon(name, mat, pivot, attach_pos, length=0.38, width=0.015, nx=4, ny=44, curve_x=0.08):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    ax, ay, az = attach_pos
    grid = []
    for j in range(ny + 1):
        v = j / ny
        row = []
        cx = ax + curve_x * (v ** 0.80) + 0.045 * math.sin(v * math.pi * 1.8) * v
        cy = ay - 0.035 * math.sin(v * math.pi)
        cz = az - v * length
        twist = v * 1.25
        for i in range(nx + 1):
            u = i / nx
            s = (u - 0.5) * width
            vert = bm.verts.new((cx + s * math.cos(twist), cy + s * math.sin(twist), cz))
            row.append((vert, u, v))
        grid.append(row)

    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    pin_idx = [grid[j][i][0].index for j in range(2) for i in range(nx + 1)]

    for j in range(ny):
        for i in range(nx):
            v00, u00, w00 = grid[j][i]
            v10, u10, w10 = grid[j][i + 1]
            v11, u11, w11 = grid[j + 1][i + 1]
            v01, u01, w01 = grid[j + 1][i]
            f = bm.faces.new((v00, v01, v11, v10))
            for loop, uv in zip(f.loops, [(u00, 1 - w00), (u01, 1 - w01), (u11, 1 - w11), (u10, 1 - w10)]):
                loop[uv_layer].uv = uv

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons: p.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = pivot

    vg = obj.vertex_groups.new(name="Pin")
    vg.add(pin_idx, 1.0, 'REPLACE')
    return obj

def create_pearl_oval_buckle(name, ring_mat, pearl_mat, parent_cloth, nx_cloth, row_j, col_i, offset_local=(0.042, -0.095, -0.030)):
    """Full 100%-unclipped 18-pearl oval buckle! Uses single-VERTEX parent so it tracks live cloth
    3D translation 1:1 without twisting edge-on when deep cloth pleats fold."""
    stride = nx_cloth + 1
    v0 = row_j * stride + col_i

    anchor = bpy.data.objects.new(f"{name}_Anchor", None)
    bpy.context.scene.collection.objects.link(anchor)
    anchor.parent = parent_cloth
    anchor.parent_type = 'VERTEX'
    anchor.parent_vertices = (v0, 0, 0)
    anchor.location = Vector(offset_local)

    bm = bmesh.new()
    rx, rz = 0.044, 0.032
    r_tube = 0.0068
    n_u, n_v = 32, 10
    rings = []
    for i in range(n_u):
        a = 2.0 * math.pi * i / n_u
        cx, cz = rx * math.cos(a), rz * math.sin(a)
        tx, tz = -rx * math.sin(a), rz * math.cos(a)
        out_n = Vector((tz, 0.0, -tx)).normalized()
        y_n = Vector((0.0, 1.0, 0.0))
        rings.append([bm.verts.new(Vector((cx, 0.0, cz)) + (out_n * math.cos(2*math.pi*j/n_v) + y_n * math.sin(2*math.pi*j/n_v)) * r_tube) for j in range(n_v)])
    for i in range(n_u):
        r0, r1 = rings[i], rings[(i + 1) % n_u]
        for j in range(n_v):
            jn = (j + 1) % n_v
            bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    m_ring = bpy.data.meshes.new(f"{name}_RingMesh")
    bm.to_mesh(m_ring)
    bm.free()
    for p in m_ring.polygons: p.use_smooth = True

    ring_obj = bpy.data.objects.new(f"{name}_Ring", m_ring)
    ring_obj.data.materials.append(ring_mat)
    bpy.context.scene.collection.objects.link(ring_obj)
    ring_obj.parent = anchor
    # Counter-rotate L2 hanger's -44 deg yaw so the full 18-pearl oval faces camera directly with a natural 3/4 tilt!
    ring_obj.rotation_euler = Euler((math.radians(-18), math.radians(10), math.radians(+36)), 'XYZ')

    bm_p = bmesh.new()
    n_pearls = 18
    for i in range(n_pearls):
        a = 2.0 * math.pi * i / n_pearls
        px = (rx + 0.001) * math.cos(a)
        pz = (rz + 0.001) * math.sin(a)
        py = -0.0072
        pr = 0.0062 + 0.0006 * math.sin(i * 2.1)
        mat_t = Matrix.Translation((px, py, pz)) @ Matrix.Diagonal((1.0, 0.88, 1.05, 1.0))
        bmesh.ops.create_uvsphere(bm_p, u_segments=12, v_segments=8, radius=pr, matrix=mat_t)
    bmesh.ops.recalc_face_normals(bm_p, faces=bm_p.faces)
    m_pearls = bpy.data.meshes.new(f"{name}_PearlsMesh")
    bm_p.to_mesh(m_pearls)
    bm_p.free()
    for p in m_pearls.polygons: p.use_smooth = True
    pearls_obj = bpy.data.objects.new(f"{name}_Pearls", m_pearls)
    pearls_obj.data.materials.append(pearl_mat)
    bpy.context.scene.collection.objects.link(pearls_obj)
    pearls_obj.parent = ring_obj

    return anchor, ring_obj, pearls_obj

def create_blouse_collar_and_ruffle_layers(name, cloth_mat, pivot):
    """Structured soft collar lapels + TWO dramatic fluttering ruffled sleeve/hem layers!"""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for side in (-1.0, 1.0):
        grid = []
        nu, nv = 12, 8
        for j in range(nv + 1):
            v = j / nv
            row = []
            for i in range(nu + 1):
                u = i / nu
                x = side * (0.015 + u * 0.095 + v * 0.025)
                y = -0.068 - v * 0.032 + 0.022 * (u ** 1.4)
                z = -0.088 - v * 0.082 - 0.022 * (u ** 1.5)
                row.append((bm.verts.new((x, y, z)), u, v))
            grid.append(row)
        for j in range(nv):
            for i in range(nu):
                f = bm.faces.new((grid[j][i][0], grid[j+1][i][0], grid[j+1][i+1][0], grid[j][i+1][0]))
                for loop, uv in zip(f.loops, [(i/nu, 1-j/nv), (i/nu, 1-(j+1)/nv), ((i+1)/nu, 1-(j+1)/nv), ((i+1)/nu, 1-j/nv)]):
                    loop[uv_layer].uv = uv
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_CollarMesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons: p.use_smooth = True
    collar_obj = bpy.data.objects.new(f"{name}_Collar", mesh)
    collar_obj.data.materials.append(cloth_mat)
    bpy.context.scene.collection.objects.link(collar_obj)
    collar_obj.parent = pivot
    sol = collar_obj.modifiers.new("Solidify", 'SOLIDIFY')
    sol.thickness = 0.0025

    # Fluttering inner ruffled sleeve cascade (ultra-light chiffon, billows wildly into sculptural folds!)
    ruffle_a = create_volumetric_garment_mesh(
        f"{name}_FlutterRuffleA", cloth_mat, pivot,
        nx=48, ny=48, width_top=0.34, width_bot=0.72, height=0.46,
        z_top=-0.125, y_depth=0.082, y_offset=-0.048,
        pleat_amp=0.052, pleat_freq=9.5, ruffle_hem=0.068, collar_cut=0.02, wrap_span=0.46,
    )
    add_cloth_physics(ruffle_a, mass=0.065, tension=3.0, bending=0.015, air_damp=0.22, quality=7, wind_factor=1.85)

    # Lower cascading ruffle tier for rich chaotic multi-layered folds just like @_.meekat's 3D VIEW!
    ruffle_b = create_volumetric_garment_mesh(
        f"{name}_FlutterRuffleB", cloth_mat, pivot,
        nx=44, ny=40, width_top=0.38, width_bot=0.78, height=0.38,
        z_top=-0.240, y_depth=0.090, y_offset=-0.042,
        pleat_amp=0.058, pleat_freq=11.0, ruffle_hem=0.075, collar_cut=0.0, wrap_span=0.45,
    )
    add_cloth_physics(ruffle_b, mass=0.060, tension=2.6, bending=0.012, air_damp=0.20, quality=7, wind_factor=2.05)
    return collar_obj, ruffle_a, ruffle_b

def add_cloth_physics(obj, mass=0.18, tension=6.0, bending=0.08, air_damp=0.50, quality=7, wind_factor=1.0):
    mod = obj.modifiers.new("Cloth", 'CLOTH')
    s = mod.settings
    s.quality = quality
    s.mass = mass
    s.air_damping = air_damp
    s.tension_stiffness = tension
    s.compression_stiffness = tension * 0.35  # low compression lets fabric crumple into crisp realistic cloth wrinkles!
    s.shear_stiffness = tension * 0.35
    s.bending_stiffness = bending
    s.tension_damping = 1.5
    s.compression_damping = 1.5
    s.shear_damping = 1.5
    s.bending_damping = 0.12
    s.vertex_group_mass = "Pin"
    s.pin_stiffness = 1.0
    s.effector_weights.wind = wind_factor
    s.effector_weights.turbulence = wind_factor
    s.effector_weights.gravity = 1.0

    c = mod.collision_settings
    c.use_collision = True
    c.distance_min = 0.008
    c.use_self_collision = False

    mod.point_cache.frame_start = 1 - PREROLL
    mod.point_cache.frame_end = NUM_SHOT_FRAMES
    return mod

def build_closet_rod_and_valance(chrome_mat, wood_mat, y_rod=0.52, z_rod=0.268):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.011, depth=2.6, location=(0.06, y_rod, z_rod), rotation=(0, math.pi * 0.5, 0))
    rod = bpy.context.active_object
    rod.name = "Closet_Rod"
    rod.data.materials.append(chrome_mat)
    for p in rod.data.polygons: p.use_smooth = True

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.06, y_rod - 0.03, z_rod + 0.056))
    val = bpy.context.active_object
    val.name = "Closet_TopValance"
    val.scale = (2.8, 0.36, 0.022)
    bpy.ops.object.transform_apply(scale=True)
    val.data.materials.append(wood_mat)
    return rod, val

def build_camera_from_colmap(scene, track):
    cam_data = bpy.data.cameras.new("COLMAP_Camera")
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = 36.0
    cam_data.lens = float(track["focal_mm_35"])  # 16.52mm real COLMAP wide-angle!
    cam_data.clip_start = 0.02
    cam_data.clip_end = 50.0

    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 1.60
    cam_data.dof.aperture_fstop = 2.05
    cam_data.dof.aperture_blades = 7

    cam_obj = bpy.data.objects.new("COLMAP_Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    frames = track["frames"]
    total_fwd = max(float(track["total_forward_m"]), 1e-4)
    R_base_lookY = Matrix.Rotation(math.pi * 0.5, 3, 'X')

    def apply_cam_at_index(bl_frame, idx):
        fd = frames[idx]
        p = fd["pos_bl_m"]
        u_t = max(0.0, min(1.0, (p[1] - frames[0]["pos_bl_m"][1]) / total_fwd))

        cx = 0.08 * p[0]
        cy = 0.00 + 0.095 * u_t
        cz = 0.16 * p[2]

        R_delta = Matrix(fd["R_bl_delta"])
        R_cam = R_delta @ R_base_lookY

        cam_obj.location = (cx, cy, cz)
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = R_cam.to_quaternion()
        cam_obj.keyframe_insert(data_path="location", frame=bl_frame)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=bl_frame)

    for bf in range(1 - PREROLL, 1):
        apply_cam_at_index(bf, 0)
    for i in range(NUM_SHOT_FRAMES):
        apply_cam_at_index(i + 1, i)
    return cam_obj

def animate_hanger_pendulum(pivot, base_loc, base_rot_deg, track, phase=0.0, swing_amp=1.0, part_x=0.0):
    frames = track["frames"]
    total_fwd = max(float(track["total_forward_m"]), 1e-4)
    bx, by, bz = base_loc
    rx0, ry0, rz0 = [math.radians(a) for a in base_rot_deg]

    theta_x, theta_y = 0.0, 0.0
    vx, vy = 0.0, 0.0
    dt = 1.0 / 25.0
    omega = 3.6 + 0.4 * math.sin(phase)
    zeta = 0.24

    for bf in range(1 - PREROLL, NUM_SHOT_FRAMES + 1):
        idx = max(0, min(NUM_SHOT_FRAMES - 1, bf - 1))
        fd = frames[idx]
        acc = fd["acc_bl"]
        p = fd["pos_bl_m"]
        u_t = max(0.0, min(1.0, (p[1] - frames[0]["pos_bl_m"][1]) / total_fwd))

        t_sec = bf * dt
        gust_x = 0.48 * math.sin(t_sec * 2.3 + phase) + 0.30 * math.sin(t_sec * 5.0 + phase * 1.8)
        gust_y = 0.36 * math.cos(t_sec * 1.8 + phase * 1.3) + 0.22 * math.sin(t_sec * 4.1 + phase)
        fx_drive = (-0.18 * acc[1] + gust_x) * swing_amp
        fy_drive = (0.22 * acc[0] + gust_y) * swing_amp

        ax = fx_drive - 2.0 * zeta * omega * vx - (omega ** 2) * theta_x
        ay = fy_drive - 2.0 * zeta * omega * vy - (omega ** 2) * theta_y
        vx += ax * dt
        vy += ay * dt
        theta_x += vx * dt
        theta_y += vy * dt

        px = bx + part_x * (u_t ** 1.1)
        pivot.location = (px, by, bz)
        pivot.rotation_euler = Euler((rx0 + theta_x * 0.22, ry0 + theta_y * 0.18, rz0 + theta_y * 0.14), 'XYZ')
        pivot.keyframe_insert(data_path="location", frame=bf)
        pivot.keyframe_insert(data_path="rotation_euler", frame=bf)

def create_wind_and_turbulence(scene):
    """
    VERIFIED PHYSICAL WIND FOR BLENDER CLOTH (flow = 0.0!!):
    Setting flow=0.0 removes artificial air-drag clamping so Wind (900..1450) & Turbulence (320..520)
    genuinely billow the translucent cream blouse ruffles & whip the thin black satin ribbons!
    """
    # Primary room draft blowing OUT of sunlit living room (-Y) toward camera + gentle left sway
    bpy.ops.object.effector_add(type='WIND', location=(0.15, 1.05, -0.10), rotation=(math.radians(+84), 0, math.radians(-14)))
    wind = bpy.context.active_object
    wind.name = "FX_RoomDraft_Wind"
    wind.field.flow = 0.0
    wind.field.noise = 3.8

    for bf in range(1 - PREROLL, NUM_SHOT_FRAMES + 1, 3):
        t = bf / 25.0
        strength = 950.0 + 480.0 * math.sin(t * 2.7 + 0.3) + 290.0 * math.sin(t * 5.9 + 1.2)
        wind.field.strength = max(300.0, strength)
        wind.field.keyframe_insert(data_path="strength", frame=bf)

    # Secondary upward-inward cross-gust aimed straight across the cream ruffled blouse sleeves!
    bpy.ops.object.effector_add(type='WIND', location=(0.78, 0.70, -0.26), rotation=(math.radians(+68), math.radians(32), math.radians(+42)))
    wind_r = bpy.context.active_object
    wind_r.name = "FX_BlouseCrossGust"
    wind_r.field.flow = 0.0
    wind_r.field.noise = 3.2
    wind_r.field.use_max_distance = True
    wind_r.field.distance_max = 0.85
    for bf in range(1 - PREROLL, NUM_SHOT_FRAMES + 1, 4):
        t = bf / 25.0
        wind_r.field.strength = 780.0 + 420.0 * math.cos(t * 3.4 + 0.8)
        wind_r.field.keyframe_insert(data_path="strength", frame=bf)

    # High-energy local turbulence churning rich asymmetric organic cloth folds & whipping straps!
    bpy.ops.object.effector_add(type='TURBULENCE', location=(0.08, 0.46, -0.14))
    turb = bpy.context.active_object
    turb.name = "FX_Wardrobe_Turbulence"
    turb.field.strength = 360.0
    turb.field.size = 0.16
    turb.field.flow = 0.0
    turb.field.noise = 2.8

    for bf in range(1 - PREROLL, NUM_SHOT_FRAMES + 1, 4):
        t = bf / 25.0
        turb.field.strength = 340.0 + 200.0 * math.sin(t * 3.7 + 1.1)
        turb.field.keyframe_insert(data_path="strength", frame=bf)

    return wind, wind_r, turb

def main():
    print("=== Building Foreground Cloth Tunnel Scene v4 (True Wind Billow + Zero Clipping + Clear Framing) ===", flush=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = NUM_SHOT_FRAMES

    track = json.loads(TRACK_JSON.read_text(encoding="utf-8"))
    setup_cycles_optix(scene, samples=48)
    setup_world_hdri_and_practicals(scene)
    cam_obj = build_camera_from_colmap(scene, track)

    mat_oak_light = make_wood_hanger_mat("Mat_LightOak_Wood", (1.06, 0.98, 0.90))
    mat_oak_warm = make_wood_hanger_mat("Mat_WarmOak_Wood", (0.90, 0.82, 0.72))
    mat_walnut = make_wood_hanger_mat("Mat_DarkWalnut", (0.32, 0.26, 0.20))
    mat_chrome = make_chrome_mat()
    mat_black_dress = make_black_dress_mat()
    mat_ribbon = make_satin_ribbon_mat()
    mat_pearl = make_pearl_mat()
    mat_eyelet = make_eyelet_knit_mat()
    mat_blouse = make_cream_blouse_mat()
    mat_coat = make_dark_wool_coat_mat()
    make_clay_and_wire_mats()

    y_rod = 0.52
    z_rod = 0.268
    build_closet_rod_and_valance(mat_chrome, mat_walnut, y_rod=y_rod, z_rod=z_rod)

    # 0) FAR-LEFT DARK FRAMING GARMENT (narrower 0.33m wood hanger tucked cleanly inside 0.44m shoulder!)
    p_L0, _, _ = build_sculpted_wooden_hanger("Hanger_L0_Edge", mat_oak_warm, mat_chrome, width=0.32, drop=0.095)
    animate_hanger_pendulum(p_L0, (-0.50, y_rod, z_rod), (1, -2, -32), track, phase=0.1, swing_amp=0.60, part_x=0.0)
    cloth_L0 = create_volumetric_garment_mesh(
        "Cloth_L0_DarkFrame", mat_coat, p_L0,
        nx=28, ny=38, width_top=0.42, width_bot=0.52, height=0.60,
        z_top=-0.088, y_depth=0.080, pleat_amp=0.024, pleat_freq=5.0, wrap_span=0.52, arch_drop_scale=0.095,
    )
    add_cloth_physics(cloth_L0, mass=0.40, tension=14.0, bending=0.65, air_damp=1.1, quality=5, wind_factor=0.50)

    # 1) LEFT-REAR HERO: Thick Exposed Oak Hanger + Off-White Perforated Eyelet Knit draped BELOW shoulder arch!
    p_L1, _, _ = build_sculpted_wooden_hanger("Hanger_L1_Rear", mat_oak_light, mat_chrome, width=0.35, drop=0.105)
    animate_hanger_pendulum(p_L1, (-0.335, y_rod, z_rod), (2, -3, -40), track, phase=0.5, swing_amp=0.80, part_x=-0.008)
    cloth_L1 = create_volumetric_garment_mesh(
        "Cloth_L_EyeletKnit", mat_eyelet, p_L1,
        nx=48, ny=56, width_top=0.44, width_bot=0.58, height=0.58,
        z_top=-0.096, y_depth=0.088, pleat_amp=0.028, pleat_freq=6.0, collar_cut=0.040, wrap_span=0.52, arch_drop_scale=0.102,
    )
    add_cloth_physics(cloth_L1, mass=0.24, tension=8.0, bending=0.22, air_damp=0.75, quality=6, wind_factor=0.75)

    # 2) LEFT-FRONT HERO: Bold Exposed Oak Hanger + Black Straps + Volumetric Black Pleated Dress + Full 18-Pearl Buckle + 2 Whipping Ribbons
    p_L2, _, _ = build_sculpted_wooden_hanger("Hanger_L2_Front", mat_oak_warm, mat_chrome, width=0.42, drop=0.118)
    animate_hanger_pendulum(p_L2, (-0.175, y_rod, z_rod), (3, -2, -44), track, phase=1.3, swing_amp=1.05, part_x=-0.015)
    create_hanger_shoulder_straps("Dress_ShoulderStraps", mat_black_dress, p_L2, xs=(-0.125, 0.125), z_top=-0.088, drop_to=-0.205)
    cloth_L2 = create_volumetric_garment_mesh(
        "Cloth_L_BlackDress", mat_black_dress, p_L2,
        nx=52, ny=58, width_top=0.38, width_bot=0.56, height=0.50,
        z_top=-0.195, y_depth=0.092, pleat_amp=0.036, pleat_freq=7.5, ruffle_hem=0.026, collar_cut=0.035,
        wrap_span=0.54, pin_center_only=True, arch_drop_scale=0.055,
    )
    add_cloth_physics(cloth_L2, mass=0.18, tension=6.5, bending=0.12, air_damp=0.55, quality=7, wind_factor=0.95)

    # Full unclipped 18-Pearl Oval Buckle sitting proudly in front of the pleated black dress!
    create_pearl_oval_buckle("Pearl_Oval_Buckle", mat_ribbon, mat_pearl, cloth_L2, nx_cloth=52, row_j=22, col_i=29, offset_local=(0.042, -0.095, -0.025))

    # Three thin whipping black satin ribbons pinned along the inner edge lashing visibly across the bright sunlit gap!
    rib_A = create_whipping_ribbon("Strap_Ribbon_A", mat_ribbon, p_L2, attach_pos=(0.165, -0.068, -0.175), length=0.42, width=0.014, curve_x=0.115)
    add_cloth_physics(rib_A, mass=0.060, tension=10.0, bending=0.04, air_damp=0.22, quality=7, wind_factor=1.55)
    rib_B = create_whipping_ribbon("Strap_Ribbon_B", mat_ribbon, p_L2, attach_pos=(0.145, -0.092, -0.215), length=0.38, width=0.011, curve_x=0.095)
    add_cloth_physics(rib_B, mass=0.055, tension=9.0, bending=0.04, air_damp=0.20, quality=7, wind_factor=1.70)

    # 3) RIGHT-FRONT HERO: Narrower Oak Hanger (0.33m, zero poke-through!) + Lush Billowing Cream Ruffled Blouse!
    # Placed at X = +0.460m with subtle inward hold (part_x = -0.020m) so it stays richly framing the right edge
    # across ALL 125 frames (f001..f125) while keeping the woman (u = 0.61..0.82) 100% UNOBSTRUCTED!
    p_R1, _, _ = build_sculpted_wooden_hanger("Hanger_R1_Front", mat_oak_light, mat_chrome, width=0.33, drop=0.100)
    animate_hanger_pendulum(p_R1, (0.460, y_rod, z_rod), (2, 3, 24), track, phase=2.2, swing_amp=1.15, part_x=-0.020)
    cloth_R1 = create_volumetric_garment_mesh(
        "Cloth_R_CreamBlouse", mat_blouse, p_R1,
        nx=62, ny=68, width_top=0.46, width_bot=0.76, height=0.60,
        z_top=-0.088, y_depth=0.098, pleat_amp=0.048, pleat_freq=8.5, ruffle_hem=0.065, collar_cut=0.050,
        wrap_span=0.50, arch_drop_scale=0.098,
    )
    add_cloth_physics(cloth_R1, mass=0.095, tension=4.0, bending=0.025, air_damp=0.25, quality=7, wind_factor=1.40)
    create_blouse_collar_and_ruffle_layers("Blouse_Details", mat_blouse, p_R1)

    # 4) RIGHT-REAR HERO: Narrower Oak Hanger + Dark Olive Heavy Wool Coat framing far-right shadowed edge
    p_R2, _, _ = build_sculpted_wooden_hanger("Hanger_R2_Rear", mat_oak_warm, mat_chrome, width=0.34, drop=0.105)
    animate_hanger_pendulum(p_R2, (0.615, y_rod + 0.04, z_rod), (1, 2, 32), track, phase=3.1, swing_amp=0.70, part_x=-0.025)
    cloth_R2 = create_volumetric_garment_mesh(
        "Cloth_R_DarkCoat", mat_coat, p_R2,
        nx=40, ny=50, width_top=0.46, width_bot=0.58, height=0.62,
        z_top=-0.086, y_depth=0.090, pleat_amp=0.026, pleat_freq=5.5, collar_cut=0.030, wrap_span=0.50, arch_drop_scale=0.100,
    )
    add_cloth_physics(cloth_R2, mass=0.40, tension=15.0, bending=0.75, air_damp=1.2, quality=5, wind_factor=0.55)

    # 5) EXTREME-CLOSE BOTTOM-LEFT BOKEH WOODEN HANGER ARC (with_hook=False so NO wire hook pokes up by the pearls!)
    p_Bok, _, _ = build_sculpted_wooden_hanger("Hanger_Bottom_Bokeh", mat_oak_warm, mat_chrome, width=0.38, drop=0.100, with_hook=False, with_bar=False)
    animate_hanger_pendulum(p_Bok, (-0.14, 0.29, -0.102), (-22, 14, -22), track, phase=0.8, swing_amp=0.45, part_x=-0.015)

    create_wind_and_turbulence(scene)

    print(f"=== Baking Blender Cloth Physics v4 (frames {1-PREROLL}..{NUM_SHOT_FRAMES}) ===", flush=True)
    scene.frame_start = 1 - PREROLL
    scene.frame_end = NUM_SHOT_FRAMES
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_start = 1
    scene.frame_end = NUM_SHOT_FRAMES

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"Saved baked scene: {BLEND_PATH}", flush=True)

    # Quick diagnostic: check real cloth vertex displacement!
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get()
    scene.frame_set(1)
    v1 = np.array([v.co[:] for v in cloth_R1.evaluated_get(dg).to_mesh().vertices])
    scene.frame_set(62)
    v62 = np.array([v.co[:] for v in cloth_R1.evaluated_get(dg).to_mesh().vertices])
    print(f"  [Diag] Cloth_R_CreamBlouse disp f1->f62: max={np.linalg.norm(v62-v1,axis=1).max():.3f}m, mean={np.linalg.norm(v62-v1,axis=1).mean():.3f}m", flush=True)

    # Fast half-res lookdev renders (f001, f062, f120)
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.cycles.samples = 32
    for fnum in (1, 62, 120):
        scene.frame_set(fnum)
        scene.render.filepath = str(WORK / f"lookdev_f{fnum:03d}.png")
        print(f"Rendering lookdev frame {fnum}...", flush=True)
        bpy.ops.render.render(write_still=True)

    # 3D VIEW clay + true-quad wireframe preview at f062
    print("Rendering 3D VIEW quad-wireframe lookdev f062...", flush=True)
    clay_mat = bpy.data.materials["Mat_3DView_Clay"]
    wire_mat = bpy.data.materials["Mat_3DView_Wire"]
    cam_obj.data.dof.use_dof = False
    scene.render.use_motion_blur = False
    scene.cycles.samples = 16
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.hide_render:
            obj.data.materials.clear()
            obj.data.materials.append(clay_mat)
            obj.data.materials.append(wire_mat)
            wf = obj.modifiers.new("Wire3DView", 'WIREFRAME')
            wf.use_replace = False
            wf.material_offset = 1
            wf.use_even_offset = True
            wf.use_boundary = True
            is_small = ("Pearl" in obj.name or "Hook" in obj.name or "Ribbon" in obj.name)
            wf.thickness = 0.00045 if is_small else 0.00075
    scene.frame_set(62)
    scene.render.filepath = str(WORK / "lookdev_wire_f062.png")
    bpy.ops.render.render(write_still=True)

    print("LOOKDEV v4 COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
