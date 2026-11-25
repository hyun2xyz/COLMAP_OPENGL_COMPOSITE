import bpy
import bmesh
import math
import json
from pathlib import Path
from mathutils import Vector, Matrix, Euler

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cafe_peek"
WORK = PROJ / "work"
FRAMES_DIR = WORK / "frames"
TRACK_JSON = WORK / "camera_track.json"
HDRI_PATH = WORK / "scene_matched_cafe_2k.hdr"
BLEND_PATH = WORK / "cafe_peek_scene.blend"

NUM_FRAMES = 150
FPS = 25

def set_bsdf(bsdf, **kwargs):
    for k, v in kwargs.items():
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = v

def setup_cycles_optix(scene, samples=64):
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == 'OPTIX')
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.012
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPTIX'
    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 10
    scene.cycles.transparent_max_bounces = 10
    scene.cycles.sample_clamp_indirect = 10.0

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.film_transparent = True
    # Keep glass opaque in alpha so refracted live-action plate inside glass renders cleanly in beauty!
    scene.cycles.film_transparent_glass = False

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Base Contrast'
    scene.view_settings.exposure = 0.0

    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = 'CENTER'
    scene.render.motion_blur_shutter = 0.45

def setup_world_hdri_and_lights(scene):
    world = bpy.data.worlds.new("CafeMatchedWorld")
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
    n_bg.inputs['Strength'].default_value = 0.90

    nt.links.new(n_tc.outputs['Generated'], n_map.inputs['Vector'])
    nt.links.new(n_map.outputs['Vector'], n_env.inputs['Vector'])
    nt.links.new(n_env.outputs['Color'], n_bg.inputs['Color'])
    nt.links.new(n_bg.outputs['Background'], n_out.inputs['Surface'])

    # 1) Large cool overcast right café window key light (matched accurately to plate exposure!)
    ld_win = bpy.data.lights.new("Practical_RightCafeWindow", type='AREA')
    ld_win.shape = 'RECTANGLE'
    ld_win.size = 1.8
    ld_win.size_y = 2.0
    ld_win.energy = 56.0
    ld_win.color = (0.85, 0.93, 1.0)
    o_win = bpy.data.objects.new("Practical_RightCafeWindow", ld_win)
    scene.collection.objects.link(o_win)
    o_win.location = (1.65, 0.95, 0.25)
    o_win.rotation_euler = (Vector((0.0, 0.42, -0.06)) - o_win.location).to_track_quat('-Z', 'Y').to_euler()
    o_win.visible_camera = False

    # 2) Warm Edison bulb overhead practical fill (matches the café's hanging Edison filament lamps!)
    ld_ed = bpy.data.lights.new("Practical_OverheadEdison", type='POINT')
    ld_ed.energy = 10.5
    ld_ed.shadow_soft_size = 0.08
    ld_ed.color = (1.0, 0.64, 0.26)
    o_ed = bpy.data.objects.new("Practical_OverheadEdison", ld_ed)
    scene.collection.objects.link(o_ed)
    o_ed.location = (0.18, 0.36, 0.19)
    o_ed.visible_camera = False

    # 3) Subtle warm terracotta brick bounce from center-left interior (keeps camera-facing shadows rich & deep!)
    ld_brk = bpy.data.lights.new("Practical_BrickWarmFill", type='AREA')
    ld_brk.shape = 'DISK'
    ld_brk.size = 1.1
    ld_brk.energy = 3.2
    ld_brk.color = (1.0, 0.74, 0.50)
    o_brk = bpy.data.objects.new("Practical_BrickWarmFill", ld_brk)
    scene.collection.objects.link(o_brk)
    o_brk.location = (-0.85, 1.15, 0.15)
    o_brk.rotation_euler = (Vector((0.0, 0.42, -0.05)) - o_brk.location).to_track_quat('-Z', 'Y').to_euler()
    o_brk.visible_camera = False

    return world

def build_live_refraction_card(scene, cam_obj, track):
    """
    Feature-film VFX trick:
    Places an invisible-to-camera image plane driven by the live-action café PNG sequence at Y = 3.2m
    aligned 1:1 with the camera frustum, visible ONLY to Transmission & Glossy rays!
    Result: Every fluted glass bottle, cut-crystal glass, & brass surface physically refracts & reflects
    the REAL moving café scene & window behind it, while keeping the empty gap 100% alpha=0!
    """
    dist = 3.2
    fx = float(track["focal_px"])
    w_plane = dist * (1920.0 / fx) * 1.04
    h_plane = dist * (1080.0 / fx) * 1.04

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    card = bpy.context.active_object
    card.name = "LiveAction_RefractionCard"
    card.scale = (w_plane, h_plane, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # Parent to camera in local camera space: camera looks down -Z, +X right, +Y up!
    card.parent = cam_obj
    card.location = (0.0, 0.0, -dist)
    card.rotation_euler = (0.0, 0.0, 0.0)

    # Invisible to direct camera & diffuse lighting; ONLY visible to glass transmission & specular glossy rays!
    card.visible_camera = False
    card.visible_diffuse = False
    card.visible_shadow = False
    card.visible_volume_scatter = False
    card.visible_transmission = True
    card.visible_glossy = True

    mat = bpy.data.materials.new("Mat_LiveRefractionCard")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.05
    tex = nt.nodes.new('ShaderNodeTexImage')
    img = bpy.data.images.load(str(FRAMES_DIR / "frame_0000.png"), check_existing=True)
    img.source = 'SEQUENCE'
    tex.image = img
    tex.image_user.frame_duration = NUM_FRAMES
    tex.image_user.frame_start = 1
    tex.image_user.frame_offset = -1  # Blender frame 1 -> frame_0000.png!
    tex.image_user.use_auto_refresh = True
    tex.extension = 'EXTEND'

    nt.links.new(tex.outputs['Color'], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    card.data.materials.append(mat)
    return card

# ---------------------------------------------------------------------------
# Ultra-Detailed Procedural PBR Materials (Glass, Brass, Stoneware, Botanical, Wood, Linen)
# ---------------------------------------------------------------------------
def make_amber_glass_mat(name="Mat_AmberApothecaryGlass"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    set_bsdf(bsdf, **{
        'Base Color': (0.76, 0.34, 0.09, 1.0),
        'Roughness': 0.035,
        'IOR': 1.52,
        'Transmission Weight': 0.96,
        'Coat Weight': 0.35,
        'Coat Roughness': 0.02,
    })
    vol = nt.nodes.new('ShaderNodeVolumeAbsorption')
    vol.inputs['Color'].default_value = (0.72, 0.28, 0.06, 1.0)
    vol.inputs['Density'].default_value = 18.0

    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    nt.links.new(vol.outputs['Volume'], out.inputs['Volume'])
    return mat

def make_clear_crystal_glass_mat(name="Mat_ClearCrystalGlass", rough=0.02, tint=(0.97, 0.99, 0.98, 1.0)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    set_bsdf(bsdf, **{
        'Base Color': tint,
        'Roughness': rough,
        'IOR': 1.53,
        'Transmission Weight': 0.98,
        'Coat Weight': 0.40,
        'Coat Roughness': 0.015,
    })
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_iced_coffee_liquid_mat(name="Mat_IcedCoffeeLiquid"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    set_bsdf(bsdf, **{
        'Base Color': (0.24, 0.11, 0.04, 1.0),
        'Roughness': 0.03,
        'IOR': 1.333,
        'Transmission Weight': 0.88,
    })
    vol = nt.nodes.new('ShaderNodeVolumeAbsorption')
    vol.inputs['Color'].default_value = (0.32, 0.14, 0.05, 1.0)
    vol.inputs['Density'].default_value = 32.0
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    nt.links.new(vol.outputs['Volume'], out.inputs['Volume'])
    return mat

def make_cork_mat(name="Mat_NaturalCork"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.inputs['Scale'].default_value = 85.0
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = (0.16, 0.10, 0.055, 1.0)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (0.44, 0.30, 0.17, 1.0)
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.35
    bump.inputs['Distance'].default_value = 0.002
    set_bsdf(bsdf, **{'Roughness': 0.88, 'Specular IOR Level': 0.2})
    nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
    nt.links.new(vor.outputs['Distance'], ramp.inputs['Fac'])
    nt.links.new(vor.outputs['Distance'], bump.inputs['Height'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_kraft_label_mat(name="Mat_KraftBottleLabel"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.bands_direction = 'Y'
    wave.inputs['Scale'].default_value = 18.0
    wave.inputs['Distortion'].default_value = 0.3
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 140.0
    noise.inputs['Detail'].default_value = 6.0
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.08
    ramp.color_ramp.elements[0].color = (0.12, 0.10, 0.08, 1.0)  # subtle printed dark ink stripes
    ramp.color_ramp.elements[1].position = 0.22
    ramp.color_ramp.elements[1].color = (0.64, 0.56, 0.44, 1.0)  # warm handmade cotton kraft paper
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.12
    bump.inputs['Distance'].default_value = 0.001
    set_bsdf(bsdf, **{'Roughness': 0.84, 'Specular IOR Level': 0.25})
    nt.links.new(tc.outputs['UV'], wave.inputs['Vector'])
    nt.links.new(tc.outputs['UV'], noise.inputs['Vector'])
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_matte_black_metal_mat(name="Mat_MatteBlackKettle"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 280.0
    noise.inputs['Detail'].default_value = 6.0
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.06
    bump.inputs['Distance'].default_value = 0.0005
    set_bsdf(bsdf, **{
        'Base Color': (0.025, 0.026, 0.028, 1.0),
        'Metallic': 0.65,
        'Roughness': 0.34,
        'Anisotropic': 0.25,
        'Coat Weight': 0.25,
        'Coat Roughness': 0.22,
    })
    nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_antique_brass_mat(name="Mat_AntiqueBrass"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 35.0
    noise.inputs['Detail'].default_value = 5.0
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value = 0.16
    mr.inputs['To Max'].default_value = 0.32
    set_bsdf(bsdf, **{
        'Base Color': (0.72, 0.52, 0.24, 1.0),
        'Metallic': 0.96,
        'Anisotropic': 0.35,
    })
    nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], bsdf.inputs['Roughness'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_speckled_stoneware_mat(name="Mat_SpeckledStoneware"):
    """Handmade matte-cream glazed stoneware with organic dark iron speckles & wheel-throwing grooves!"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')

    # Dark iron speckles via Voronoi F1 threshold
    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.inputs['Scale'].default_value = 115.0
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.13
    ramp.color_ramp.elements[0].color = (0.09, 0.07, 0.055, 1.0)
    ramp.color_ramp.elements[1].position = 0.25
    ramp.color_ramp.elements[1].color = (0.48, 0.43, 0.37, 1.0)

    # Subtle horizontal potter's wheel grooves
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.bands_direction = 'Z'
    wave.inputs['Scale'].default_value = 28.0
    wave.inputs['Distortion'].default_value = 1.6
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.16
    bump.inputs['Distance'].default_value = 0.0014

    set_bsdf(bsdf, **{
        'Roughness': 0.22,
        'Specular IOR Level': 0.58,
        'Coat Weight': 0.75,
        'Coat Roughness': 0.11,
        'Subsurface Weight': 0.06,
        'Subsurface Scale': 0.003,
    })

    nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
    nt.links.new(tc.outputs['Object'], wave.inputs['Vector'])
    nt.links.new(vor.outputs['Distance'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(wave.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_botanical_leaf_mat(name="Mat_BotanicalLeaf", base=(0.11, 0.24, 0.11), trans=(0.30, 0.52, 0.18)):
    """Realistic translucent veined botanical leaf catching sunlit window backlighting!"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = 0.30  # real translucent sunlit green glow!

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    transl = nt.nodes.new('ShaderNodeBsdfTranslucent')
    transl.inputs['Color'].default_value = (*trans, 1.0)

    tc = nt.nodes.new('ShaderNodeTexCoord')
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.inputs['Scale'].default_value = 14.0
    wave.inputs['Distortion'].default_value = 2.5
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.14
    bump.inputs['Distance'].default_value = 0.001

    set_bsdf(bsdf, **{
        'Base Color': (*base, 1.0),
        'Roughness': 0.26,
        'Specular IOR Level': 0.52,
        'Coat Weight': 0.48,
        'Coat Roughness': 0.18,
        'Subsurface Weight': 0.22,
        'Subsurface Scale': 0.003,
    })

    nt.links.new(tc.outputs['UV'], wave.inputs['Vector'])
    nt.links.new(wave.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'], mix.inputs[1])
    nt.links.new(transl.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat

def make_walnut_wood_mat(name="Mat_WalnutWood"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.inputs['Scale'].default_value = 6.0
    wave.inputs['Distortion'].default_value = 5.5
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = (0.075, 0.042, 0.022, 1.0)
    ramp.color_ramp.elements[1].position = 0.80
    ramp.color_ramp.elements[1].color = (0.22, 0.125, 0.065, 1.0)
    set_bsdf(bsdf, **{'Roughness': 0.38, 'Coat Weight': 0.25, 'Coat Roughness': 0.24})
    nt.links.new(tc.outputs['Object'], wave.inputs['Vector'])
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_linen_book_mat(name="Mat_LinenBook", col=(0.18, 0.21, 0.19), sheen=0.45, rough=0.82):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    set_bsdf(bsdf, **{
        'Base Color': (*col, 1.0),
        'Roughness': rough,
        'Specular IOR Level': 0.35,
        'Sheen Weight': sheen,
        'Sheen Roughness': 0.55,
    })
    return mat

def make_edison_filament_mat(name="Mat_EdisonFilament"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (1.0, 0.44, 0.09, 1.0)
    em.inputs['Strength'].default_value = 24.0
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    return mat

def make_edison_glass_mat(name="Mat_EdisonAmberGlass"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    set_bsdf(bsdf, **{
        'Base Color': (0.95, 0.72, 0.36, 1.0),
        'Roughness': 0.03,
        'IOR': 1.48,
        'Transmission Weight': 0.94,
    })
    return mat

# ---------------------------------------------------------------------------
# High-Detail Procedural BMesh Asset Builders
# ---------------------------------------------------------------------------
def build_lathe_mesh(name, profile_rz, n_seg=64, flute_count=0, flute_amp=0.0, flute_z_range=(0.0, 1.0), mat=None):
    """Builds a smooth radially lathed vessel (with optional crisp vertical glass flutes!)."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rings = []
    n_p = len(profile_rz)
    for j, (r, z) in enumerate(profile_rz):
        v = j / max(n_p - 1, 1)
        row = []
        # Smooth flute envelope along height
        in_f = 0.0
        if flute_count > 0 and flute_z_range[0] < z < flute_z_range[1]:
            t_f = (z - flute_z_range[0]) / (flute_z_range[1] - flute_z_range[0])
            in_f = math.sin(min(1.0, max(0.0, t_f)) * math.pi) ** 0.45
        for i in range(n_seg):
            u = i / n_seg
            ang = 2.0 * math.pi * u
            rf = r + flute_amp * in_f * math.cos(flute_count * ang)
            row.append((bm.verts.new((rf * math.cos(ang), rf * math.sin(ang), z)), u, v))
        rings.append(row)

    for j in range(n_p - 1):
        for i in range(n_seg):
            inext = (i + 1) % n_seg
            v00, u00, w00 = rings[j][i]
            v10, u10, w10 = rings[j][inext]
            v11, u11, w11 = rings[j + 1][inext]
            v01, u01, w01 = rings[j + 1][i]
            f = bm.faces.new((v00, v10, v11, v01))
            for loop, uv in zip(f.loops, [(u00, w00), (u10, w10), (u11, w11), (u01, w01)]):
                loop[uv_layer].uv = uv

    # Close bottom center cleanly
    if profile_rz[0][0] > 0.001:
        vc = bm.verts.new((0.0, 0.0, profile_rz[0][1]))
        for i in range(n_seg):
            bm.faces.new((vc, rings[0][(i + 1) % n_seg][0], rings[0][i][0]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons: p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    if mat: obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def build_amber_apothecary_bottle(name, loc, glass_mat, cork_mat, brass_mat, label_mat):
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = loc

    # Thick-walled fluted amber bottle profile (outer + inner cavity for true physical refraction!)
    outer = [
        (0.040, 0.000), (0.044, 0.006), (0.044, 0.145),
        (0.042, 0.165), (0.032, 0.185), (0.017, 0.205),
        (0.016, 0.238), (0.019, 0.244), (0.019, 0.252), (0.014, 0.252),
        # Inner glass cavity going back down
        (0.012, 0.245), (0.013, 0.205), (0.037, 0.162), (0.038, 0.014), (0.002, 0.014),
    ]
    bottle = build_lathe_mesh(f"{name}_Glass", outer, n_seg=72, flute_count=20, flute_amp=0.0018, flute_z_range=(0.012, 0.158), mat=glass_mat)
    bottle.parent = root

    # Cork stopper sticking gracefully out of the neck
    cork_p = [(0.0125, 0.230), (0.0140, 0.255), (0.0155, 0.278), (0.0145, 0.282), (0.001, 0.282)]
    cork = build_lathe_mesh(f"{name}_Cork", cork_p, n_seg=32, mat=cork_mat)
    cork.parent = root

    # Brushed brass collar ring around bottle neck
    ring_p = [(0.0165, 0.220), (0.0188, 0.221), (0.0188, 0.234), (0.0165, 0.235)]
    ring = build_lathe_mesh(f"{name}_BrassCollar", ring_p, n_seg=40, mat=brass_mat)
    ring.parent = root

    # Kraft cotton paper label band wrapping cleanly around upper belly
    lbl_p = [(0.0462, 0.068), (0.0462, 0.132)]
    lbl = build_lathe_mesh(f"{name}_Label", lbl_p, n_seg=48, mat=label_mat)
    lbl.parent = root
    return root

def build_cut_crystal_glass(name, loc, glass_mat, liquid_mat):
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = loc

    # Heavy-base 16-facet fluted crystal café tumbler!
    prof = [
        (0.031, 0.000), (0.034, 0.004), (0.037, 0.065), (0.040, 0.132),
        (0.039, 0.135), (0.0365, 0.135), (0.0355, 0.130), (0.0315, 0.022), (0.002, 0.020),
    ]
    gl = build_lathe_mesh(f"{name}_Glass", prof, n_seg=64, flute_count=16, flute_amp=0.0016, flute_z_range=(0.008, 0.120), mat=glass_mat)
    gl.parent = root

    # Dark iced coffee liquid inside (slightly overlapping inner wall for correct Cycles glass-liquid interface!)
    liq_p = [(0.002, 0.021), (0.0325, 0.022), (0.0352, 0.082), (0.0340, 0.084), (0.001, 0.0825)]
    liq = build_lathe_mesh(f"{name}_Coffee", liq_p, n_seg=40, mat=liquid_mat)
    liq.parent = root

    # Two beveled translucent ice cubes floating gently at top of coffee!
    for k, (ix, iy, iz, rot) in enumerate([(-0.009, -0.004, 0.076, (0.2, 0.3, 0.5)), (0.011, 0.007, 0.074, (-0.3, 0.15, 1.1))]):
        bpy.ops.mesh.primitive_cube_add(size=0.021, location=(ix, iy, iz), rotation=rot)
        ice = bpy.context.active_object
        ice.name = f"{name}_Ice_{k}"
        bev = ice.modifiers.new("Bevel", 'BEVEL')
        bev.width = 0.0035
        bev.segments = 3
        ice.data.materials.append(glass_mat)
        for p in ice.data.polygons: p.use_smooth = True
        ice.parent = root
    return root

def build_speckled_ceramic_pitcher(name, loc, rot_z_deg, stoneware_mat):
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = loc
    root.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))

    # Handmade bulbous stoneware coffee pitcher
    prof = [
        (0.034, 0.000), (0.036, 0.005), (0.044, 0.035), (0.043, 0.075),
        (0.037, 0.105), (0.039, 0.122), (0.036, 0.124), (0.033, 0.102),
        (0.037, 0.035), (0.030, 0.009), (0.002, 0.009),
    ]
    body = build_lathe_mesh(f"{name}_Body", prof, n_seg=48, mat=stoneware_mat)
    body.parent = root

    # Swept sculpted ear handle on +X side
    bm = bmesh.new()
    pts = [
        Vector((0.037, 0.0, 0.102)), Vector((0.056, 0.0, 0.105)),
        Vector((0.068, 0.0, 0.088)), Vector((0.066, 0.0, 0.056)),
        Vector((0.055, 0.0, 0.034)), Vector((0.041, 0.0, 0.028)),
    ]
    # Catmull-Rom-like dense smooth curve via Bezier/subdiv interpolation
    dense = []
    for s_i in range(25):
        t = s_i / 24.0
        # Smooth Bernstein cubic-ish curve for graceful mug handle
        ang = -math.pi * 0.45 + t * math.pi * 1.15
        hx = 0.038 + 0.028 * math.cos(ang) * (1.0 - 0.18 * t)
        hz = 0.068 + 0.037 * math.sin(-ang)
        dense.append(Vector((hx, 0.0, hz)))
    rings = []
    for i, pt in enumerate(dense):
        tang = (dense[min(len(dense)-1, i+1)] - dense[max(0, i-1)]).normalized()
        n1 = Vector((0, 1, 0))
        n2 = tang.cross(n1).normalized()
        rings.append([bm.verts.new(pt + n1 * (0.0062 * math.cos(2*math.pi*k/10)) + n2 * (0.0040 * math.sin(2*math.pi*k/10))) for k in range(10)])
    for i in range(len(rings)-1):
        for k in range(10):
            kn = (k+1)%10
            bm.faces.new((rings[i][k], rings[i][kn], rings[i+1][kn], rings[i+1][k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    hm = bpy.data.meshes.new(f"{name}_HandleMesh")
    bm.to_mesh(hm)
    bm.free()
    for p in hm.polygons: p.use_smooth = True
    h_obj = bpy.data.objects.new(f"{name}_Handle", hm)
    h_obj.data.materials.append(stoneware_mat)
    bpy.context.scene.collection.objects.link(h_obj)
    h_obj.parent = root
    return root

def build_gooseneck_kettle(name, loc, rot_z_deg, black_mat, brass_mat, wood_mat):
    """Artisan matte-black & antique-brass gooseneck pour-over kettle with iconic swan-neck spout!"""
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = loc
    root.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))

    # Tapered minimalist kettle body
    body_p = [
        (0.062, 0.000), (0.068, 0.006), (0.069, 0.025),
        (0.060, 0.115), (0.052, 0.155), (0.048, 0.159), (0.001, 0.163),
    ]
    body = build_lathe_mesh(f"{name}_Body", body_p, n_seg=56, mat=black_mat)
    body.parent = root

    # Brushed antique-brass collar ring around lid rim
    collar_p = [(0.0525, 0.152), (0.0545, 0.155), (0.0510, 0.162), (0.0475, 0.161)]
    collar = build_lathe_mesh(f"{name}_BrassRim", collar_p, n_seg=48, mat=brass_mat)
    collar.parent = root

    # Turned walnut wood lid knob with brass stem
    stem_p = [(0.005, 0.162), (0.006, 0.172), (0.005, 0.174)]
    stem = build_lathe_mesh(f"{name}_KnobStem", stem_p, n_seg=24, mat=brass_mat)
    stem.parent = root
    knob_p = [(0.010, 0.172), (0.018, 0.179), (0.018, 0.188), (0.012, 0.193), (0.001, 0.194)]
    knob = build_lathe_mesh(f"{name}_WoodKnob", knob_p, n_seg=32, mat=wood_mat)
    knob.parent = root

    # Signature S-curved Swan-Neck Spout extending cleanly toward -X (pointing gracefully into the sunlit gap!)
    bm = bmesh.new()
    n_s = 34
    s_pts = []
    s_rads = []
    for i in range(n_s + 1):
        t = i / n_s
        # Graceful swan S-curve: emerges from lower left (-X), sweeps out, climbs gracefully up, tips gently outward!
        x = -0.060 - 0.072 * (t ** 0.72) - 0.026 * math.sin(t * math.pi * 1.45)
        z = 0.024 + 0.150 * (t ** 1.28) + 0.016 * math.sin(t * math.pi)
        r = 0.0125 * (1.0 - 0.54 * (t ** 0.85))  # tapers smoothly from 12.5mm base to sleek 5.7mm pour lip!
        s_pts.append(Vector((x, 0.0, z)))
        s_rads.append(r)

    rings = []
    for i, (pt, rad) in enumerate(zip(s_pts, s_rads)):
        tang = (s_pts[min(n_s, i + 1)] - s_pts[max(0, i - 1)]).normalized()
        n1 = Vector((0, 1, 0))
        n2 = tang.cross(n1).normalized()
        rings.append([bm.verts.new(pt + (n1 * math.cos(2*math.pi*k/16) + n2 * math.sin(2*math.pi*k/16)) * rad) for k in range(16)])
    for i in range(n_s):
        for k in range(16):
            kn = (k + 1) % 16
            bm.faces.new((rings[i][k], rings[i][kn], rings[i+1][kn], rings[i+1][k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    sm = bpy.data.meshes.new(f"{name}_SpoutMesh")
    bm.to_mesh(sm)
    bm.free()
    for p in sm.polygons: p.use_smooth = True
    spout = bpy.data.objects.new(f"{name}_Spout", sm)
    spout.data.materials.append(black_mat)
    bpy.context.scene.collection.objects.link(spout)
    spout.parent = root
    return root

def build_botanical_vine_cluster(name, anchor_loc, stem_configs, leaf_mat, stem_mat):
    """
    Creates sculpted hanging/rising botanical eucalyptus & ivy stems with individually curved veined leaves!
    Animates ultra-gentle subtle indoor breeze sway across the 150 frames!
    """
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = anchor_loc

    # Reusable curved leaf BMesh helper
    def add_leaf(bm, uv_layer, base_pos, dir_vec, side_vec, length=0.052, width=0.028):
        d = dir_vec.normalized()
        s = side_vec.normalized()
        n = d.cross(s).normalized()
        # 5 rows x 3 cols cupped almond leaf with central midrib crease!
        nu, nv = 4, 6
        grid = []
        for j in range(nv + 1):
            v = j / nv
            # Almond/eucalyptus leaf width profile
            w_v = math.sin(v * math.pi) ** 0.78 * width * 0.5
            droop = -0.012 * (v ** 1.8)
            row = []
            for i in range(nu + 1):
                u = i / nu
                su = (u - 0.5) * 2.0  # -1..+1
                cup = 0.0045 * (abs(su) ** 1.4)  # gentle V-midrib cup!
                pos = base_pos + d * (v * length) + s * (su * w_v) + n * (droop + cup)
                row.append((bm.verts.new(pos), u, v))
            grid.append(row)
        for j in range(nv):
            for i in range(nu):
                f = bm.faces.new((grid[j][i][0], grid[j+1][i][0], grid[j+1][i+1][0], grid[j][i+1][0]))
                for loop, uv in zip(f.loops, [(i/nu, j/nv), (i/nu, (j+1)/nv), ((i+1)/nu, (j+1)/nv), ((i+1)/nu, j/nv)]):
                    loop[uv_layer].uv = uv

    for idx, cfg in enumerate(stem_configs):
        piv = bpy.data.objects.new(f"{name}_StemPiv_{idx}", None)
        bpy.context.scene.collection.objects.link(piv)
        piv.parent = root
        piv.location = cfg["offset"]

        # Build curved stem tube + leaves along stem
        bm_s = bmesh.new()
        bm_l = bmesh.new()
        uv_l = bm_l.loops.layers.uv.new("UVMap")

        n_pts = 22
        pts = []
        length = cfg["length"]
        dir_base = Vector(cfg["dir"]).normalized()
        curve_vec = Vector(cfg["curve"])
        for i in range(n_pts + 1):
            t = i / n_pts
            p = dir_base * (t * length) + curve_vec * (t ** 1.6) + Vector((0.008 * math.sin(t * 5.0 + idx), 0.006 * math.cos(t * 4.0), 0.0)) * t
            pts.append(p)

        # Stem tube
        s_rings = []
        for i, pt in enumerate(pts):
            tang = (pts[min(n_pts, i + 1)] - pts[max(0, i - 1)]).normalized()
            ref = Vector((1, 0, 0)) if abs(tang.x) < 0.85 else Vector((0, 1, 0))
            u_ax = tang.cross(ref).normalized()
            v_ax = tang.cross(u_ax).normalized()
            rad = 0.0022 * (1.0 - 0.55 * (i / n_pts))
            s_rings.append([bm_s.verts.new(pt + (u_ax * math.cos(2*math.pi*k/6) + v_ax * math.sin(2*math.pi*k/6)) * rad) for k in range(6)])
        for i in range(n_pts):
            for k in range(6):
                kn = (k + 1) % 6
                bm_s.faces.new((s_rings[i][k], s_rings[i][kn], s_rings[i+1][kn], s_rings[i+1][k]))

        # Attach lush alternating botanical leaves along stem
        for i in range(3, n_pts, 2):
            t = i / n_pts
            pt = pts[i]
            tang = (pts[min(n_pts, i + 1)] - pts[max(0, i - 1)]).normalized()
            side_sign = 1.0 if (i % 4 == 1) else -1.0
            # Fan leaves cleanly toward camera-visible angles!
            l_dir = (tang * 0.55 + Vector((side_sign * 0.72, -0.35 + 0.25 * math.sin(i * 1.7), -0.25))).normalized()
            l_side = l_dir.cross(Vector((0, -1, 0.2))).normalized()
            if l_side.length < 0.1: l_side = Vector((1, 0, 0))
            lf_len = cfg.get("leaf_len", 0.052) * (0.72 + 0.42 * math.sin(t * math.pi))
            lf_wid = lf_len * 0.56
            add_leaf(bm_l, uv_l, pt, l_dir, l_side, length=lf_len, width=lf_wid)

        bmesh.ops.recalc_face_normals(bm_s, faces=bm_s.faces)
        ms = bpy.data.meshes.new(f"{name}_StemMesh_{idx}")
        bm_s.to_mesh(ms); bm_s.free()
        for p in ms.polygons: p.use_smooth = True
        o_s = bpy.data.objects.new(f"{name}_Stem_{idx}", ms)
        o_s.data.materials.append(stem_mat)
        bpy.context.scene.collection.objects.link(o_s)
        o_s.parent = piv

        bmesh.ops.recalc_face_normals(bm_l, faces=bm_l.faces)
        ml = bpy.data.meshes.new(f"{name}_LeavesMesh_{idx}")
        bm_l.to_mesh(ml); bm_l.free()
        for p in ml.polygons: p.use_smooth = True
        o_l = bpy.data.objects.new(f"{name}_Leaves_{idx}", ml)
        o_l.data.materials.append(leaf_mat)
        bpy.context.scene.collection.objects.link(o_l)
        o_l.parent = piv

        # Ultra-gentle natural indoor air-draft sway (~1.5 degrees slow breath!)
        phase = idx * 1.4 + 0.3
        for fr in range(1, NUM_FRAMES + 1, 5):
            sec = fr / float(FPS)
            rx = math.radians(1.4 * math.sin(sec * 0.95 + phase) + 0.5 * math.sin(sec * 2.1 + phase))
            ry = math.radians(1.1 * math.cos(sec * 0.80 + phase * 1.3))
            piv.rotation_euler = (rx, ry, 0.0)
            piv.keyframe_insert(data_path="rotation_euler", frame=fr)
    return root

def build_foreground_edison_pendant(name, loc, brass_mat, glass_mat, filament_mat, cord_mat):
    """Out-of-focus foreground vintage Edison amber glass pendant echoing the real café lamps!"""
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = loc

    # Braided black fabric hanging cord rising straight up past top border
    cord_p = [(0.0032, 0.055), (0.0032, 0.45)]
    cord = build_lathe_mesh(f"{name}_Cord", cord_p, n_seg=16, mat=cord_mat)
    cord.parent = root

    # Vintage knurled antique-brass socket
    sock_p = [(0.015, 0.000), (0.017, 0.005), (0.017, 0.048), (0.013, 0.056), (0.005, 0.058)]
    sock = build_lathe_mesh(f"{name}_BrassSocket", sock_p, n_seg=32, mat=brass_mat)
    sock.parent = root

    # Teardrop amber Edison glass bulb hanging below socket (z = 0..-0.095)
    bulb_p = [
        (0.014, 0.000), (0.018, -0.018), (0.032, -0.048), (0.033, -0.072),
        (0.024, -0.092), (0.010, -0.100), (0.001, -0.102),
    ]
    bulb = build_lathe_mesh(f"{name}_AmberBulb", bulb_p, n_seg=40, mat=glass_mat)
    bulb.parent = root

    # Glowing warm vertical hairpin tungsten filaments inside bulb (creates dreamy warm bokeh orb!)
    bm = bmesh.new()
    for k in range(4):
        ang = 2.0 * math.pi * k / 4.0
        r0 = 0.007
        pts = [
            Vector((r0 * math.cos(ang), r0 * math.sin(ang), -0.025)),
            Vector((r0 * 1.15 * math.cos(ang), r0 * 1.15 * math.sin(ang), -0.075)),
            Vector((r0 * 0.4 * math.cos(ang + 0.5), r0 * 0.4 * math.sin(ang + 0.5), -0.082)),
        ]
        for a, b in zip(pts[:-1], pts[1:]):
            tang = (b - a).normalized()
            ref = Vector((1, 0, 0)) if abs(tang.x) < 0.9 else Vector((0, 1, 0))
            u_ax = tang.cross(ref).normalized() * 0.0011
            v_ax = tang.cross(u_ax).normalized() * 0.0011
            r_a = [bm.verts.new(a + u_ax * math.cos(2*math.pi*m/6) + v_ax * math.sin(2*math.pi*m/6)) for m in range(6)]
            r_b = [bm.verts.new(b + u_ax * math.cos(2*math.pi*m/6) + v_ax * math.sin(2*math.pi*m/6)) for m in range(6)]
            for m in range(6):
                mn = (m + 1) % 6
                bm.faces.new((r_a[m], r_a[mn], r_b[mn], r_b[m]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    fm = bpy.data.meshes.new(f"{name}_FilamentMesh")
    bm.to_mesh(fm); bm.free()
    fil = bpy.data.objects.new(f"{name}_Filament", fm)
    fil.data.materials.append(filament_mat)
    bpy.context.scene.collection.objects.link(fil)
    fil.parent = root
    return root

def build_foreground_book_stack(name, loc, rot_z_deg, cover_mat, page_mat, brass_mat):
    """Extreme-close dark espresso-leather notebook with cream page block & brushed-brass fountain pen!"""
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    root.location = loc
    root.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))

    # Bottom leather cover
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.002))
    c_bot = bpy.context.active_object
    c_bot.scale = (0.165, 0.215, 0.004)
    bpy.ops.object.transform_apply(scale=True)
    c_bot.data.materials.append(cover_mat)
    c_bot.parent = root

    # Warm ivory paper page block tucked cleanly inside leather covers
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.002, 0.0, 0.011))
    pgs = bpy.context.active_object
    pgs.scale = (0.156, 0.205, 0.015)
    bpy.ops.object.transform_apply(scale=True)
    pgs.data.materials.append(page_mat)
    pgs.parent = root

    # Top dark espresso-leather cover with subtle bevel
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.020))
    bk = bpy.context.active_object
    bk.name = f"{name}_Cover"
    bk.scale = (0.165, 0.215, 0.0045)
    bpy.ops.object.transform_apply(scale=True)
    bev = bk.modifiers.new("Bevel", 'BEVEL')
    bev.width = 0.0018; bev.segments = 3
    bk.data.materials.append(cover_mat)
    bk.parent = root

    # Brushed antique-brass fountain pen resting diagonally on top cover
    pen_p = [(0.001, 0.0), (0.0052, 0.018), (0.0056, 0.115), (0.0046, 0.138), (0.001, 0.142)]
    pen = build_lathe_mesh(f"{name}_BrassPen", pen_p, n_seg=24, mat=brass_mat)
    pen.parent = root
    pen.location = (0.025, -0.055, 0.027)
    pen.rotation_euler = (math.radians(90), 0.0, math.radians(-35))
    return root

def build_camera_from_track(scene, track):
    cam_data = bpy.data.cameras.new("Cafe_Tracked_Camera")
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = 36.0
    cam_data.lens = float(track["focal_mm_35"])  # 24.0mm wide-angle matching track!
    cam_data.clip_start = 0.02
    cam_data.clip_end = 50.0

    # Authentic cinema shallow DoF focused on the man reading across the café (~3.2m):
    # f/2.45 keeps glass flutes, ceramic speckles, brass gooseneck spout, & leaf silhouettes richly readable
    # while preserving lush cinema optical foreground bokeh!
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 3.20
    cam_data.dof.aperture_fstop = 2.45
    cam_data.dof.aperture_blades = 7

    cam_obj = bpy.data.objects.new("Cafe_Tracked_Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    R_base_lookY = Matrix.Rotation(math.pi * 0.5, 3, 'X')
    for fd in track["frames"]:
        bf = fd["frame"] + 1
        rx, ry, rz = fd["rot_xyz_bl"]
        R_delta = Euler((rx, ry, rz), 'XYZ').to_matrix()
        R_cam = R_delta @ R_base_lookY

        cam_obj.location = tuple(fd["pos_bl_m"])
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = R_cam.to_quaternion()
        cam_obj.keyframe_insert(data_path="location", frame=bf)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=bf)

    return cam_obj

def render_hdri_probe(scene, cam_obj, card_obj):
    hidden = []
    for obj in scene.objects:
        if obj.type == 'MESH' and obj != card_obj and not obj.hide_render:
            obj.hide_render = True
            hidden.append(obj)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.065, location=(-0.085, 0.50, 0.0))
    s_ch = bpy.context.active_object
    for p in s_ch.data.polygons: p.use_smooth = True
    m_ch = bpy.data.materials.new("_ProbeCh")
    m_ch.use_nodes = True
    set_bsdf(m_ch.node_tree.nodes.get('Principled BSDF'), **{'Base Color': (0.98, 0.98, 0.98, 1.0), 'Metallic': 1.0, 'Roughness': 0.02})
    s_ch.data.materials.append(m_ch)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.065, location=(0.085, 0.50, 0.0))
    s_df = bpy.context.active_object
    for p in s_df.data.polygons: p.use_smooth = True
    m_df = bpy.data.materials.new("_ProbeDf")
    m_df.use_nodes = True
    set_bsdf(m_df.node_tree.nodes.get('Principled BSDF'), **{'Base Color': (0.5, 0.5, 0.5, 1.0), 'Metallic': 0.0, 'Roughness': 0.85})
    s_df.data.materials.append(m_df)

    old = (cam_obj.data.dof.use_dof, scene.render.use_motion_blur, scene.render.film_transparent, scene.cycles.samples, scene.render.resolution_x, scene.render.resolution_y)
    cam_obj.data.dof.use_dof = False
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    scene.cycles.samples = 24
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.frame_set(1)
    scene.render.filepath = str(WORK / "lookdev_hdri_probe.png")
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(s_ch, do_unlink=True)
    bpy.data.objects.remove(s_df, do_unlink=True)
    for obj in hidden: obj.hide_render = False
    cam_obj.data.dof.use_dof, scene.render.use_motion_blur, scene.render.film_transparent, scene.cycles.samples, scene.render.resolution_x, scene.render.resolution_y = old

def main():
    print("=== Building Café Extreme-Foreground Peeking Scene in Blender 5.1.2 ===", flush=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES

    track = json.loads(TRACK_JSON.read_text(encoding="utf-8"))
    setup_cycles_optix(scene, samples=64)
    setup_world_hdri_and_lights(scene)
    cam_obj = build_camera_from_track(scene, track)
    refrac_card = build_live_refraction_card(scene, cam_obj, track)

    # Materials
    mat_amber = make_amber_glass_mat()
    mat_crystal = make_clear_crystal_glass_mat("Mat_ClearCrystalGlass", rough=0.02)
    mat_vase_glass = make_clear_crystal_glass_mat("Mat_RibbedVaseGlass", rough=0.03, tint=(0.94, 0.98, 0.95, 1.0))
    mat_coffee = make_iced_coffee_liquid_mat()
    mat_cork = make_cork_mat()
    mat_label = make_kraft_label_mat()
    mat_black = make_matte_black_metal_mat()
    mat_brass = make_antique_brass_mat()
    mat_stoneware = make_speckled_stoneware_mat()
    mat_leaf_dark = make_botanical_leaf_mat("Mat_EucalyptusLeaf", base=(0.032, 0.085, 0.035), trans=(0.14, 0.30, 0.10))
    mat_leaf_bright = make_botanical_leaf_mat("Mat_SunlitIvyLeaf", base=(0.048, 0.115, 0.042), trans=(0.24, 0.44, 0.14))
    mat_stem = make_walnut_wood_mat("Mat_PlantStem")
    mat_walnut = make_walnut_wood_mat("Mat_WalnutWood")
    mat_book = make_linen_book_mat("Mat_DarkLeatherBook", col=(0.020, 0.012, 0.008), sheen=0.0, rough=0.42)
    mat_cord = make_linen_book_mat("Mat_BlackBraidedCord", col=(0.02, 0.02, 0.022), sheen=0.2, rough=0.75)
    mat_ed_glass = make_edison_glass_mat()
    mat_ed_fil = make_edison_filament_mat()

    # Tabletop Z plane aligned to the real foreground café table edge (v_scr ≈ 0.95 at y ≈ 0.44m!)
    Z_TABLE = -0.172

    # 1) LEFT-OUTER HERO: Fluted Amber Glass Apothecary Bottle with Cork, Brass Collar & Kraft Label
    build_amber_apothecary_bottle("AmberBottle_L", (-0.265, 0.46, Z_TABLE), mat_amber, mat_cork, mat_brass, mat_label)

    # 2) LEFT-INNER HERO: 16-Facet Cut-Crystal Glass Tumbler with Iced Coffee & Ice Cubes
    build_cut_crystal_glass("CrystalGlass_L", (-0.172, 0.43, Z_TABLE), mat_crystal, mat_coffee)

    # 3) UPPER-LEFT HERO: Trailing Botanical Eucalyptus & Ivy Vines draping gently down from top-left
    build_botanical_vine_cluster(
        "BotanicalVines_L",
        anchor_loc=(-0.22, 0.38, 0.19),
        stem_configs=[
            {"offset": (-0.04, 0.00, 0.02), "dir": (0.25, 0.05, -0.95), "curve": (0.06, -0.02, -0.02), "length": 0.25, "leaf_len": 0.052},
            {"offset": (0.02, -0.02, 0.01), "dir": (0.46, 0.02, -0.87), "curve": (0.06, -0.02, -0.04), "length": 0.20, "leaf_len": 0.044},
            {"offset": (-0.08, 0.04, 0.00), "dir": (0.12, -0.04, -0.98), "curve": (0.04, -0.03, 0.00), "length": 0.28, "leaf_len": 0.056},
        ],
        leaf_mat=mat_leaf_dark, stem_mat=mat_stem,
    )

    # 4) BOTTOM-LEFT EXTREME-CLOSE BOKEH: Dark Espresso-Leather Notebook + Brass Pen tucked cleanly along bottom-left lip!
    build_foreground_book_stack("ForegroundBook_BL", (-0.170, 0.31, -0.168), rot_z_deg=26, cover_mat=mat_book, page_mat=mat_label, brass_mat=mat_brass)

    # 5) RIGHT-INNER HERO: Handmade Speckled Stoneware Ceramic Pitcher
    pitcher = build_speckled_ceramic_pitcher("CeramicPitcher_R", (0.182, 0.43, Z_TABLE - 0.008), rot_z_deg=-22, stoneware_mat=mat_stoneware)
    pitcher.scale = (0.90, 0.90, 0.88)

    # 6) RIGHT-HERO: Matte-Black & Antique-Brass Gooseneck Pour-Over Kettle (swan-neck spout clearly silhouetting against bright window!)
    build_gooseneck_kettle("GooseneckKettle_R", (0.275, 0.48, Z_TABLE), rot_z_deg=-4, black_mat=mat_black, brass_mat=mat_brass, wood_mat=mat_walnut)

    # 7) RIGHT-REAR HERO: Tall Fluted Clear Glass Vase + Sunlit Backlit Eucalyptus Stems in front of bright window!
    vase_p = [
        (0.034, 0.000), (0.038, 0.008), (0.037, 0.190), (0.031, 0.215),
        (0.034, 0.228), (0.031, 0.230), (0.028, 0.214), (0.032, 0.015), (0.002, 0.015),
    ]
    vase = build_lathe_mesh("RibbedVase_R", vase_p, n_seg=64, flute_count=22, flute_amp=0.0018, flute_z_range=(0.015, 0.205), mat=mat_vase_glass)
    vase.location = (0.325, 0.56, Z_TABLE + 0.01)

    build_botanical_vine_cluster(
        "VaseEucalyptus_R",
        anchor_loc=(0.325, 0.56, Z_TABLE + 0.18),
        stem_configs=[
            {"offset": (0.00, 0.00, 0.00), "dir": (-0.34, -0.05, 0.94), "curve": (-0.08, -0.02, -0.02), "length": 0.30, "leaf_len": 0.054},
            {"offset": (0.01, 0.01, 0.00), "dir": (-0.12, -0.02, 0.99), "curve": (-0.05, -0.02, 0.00), "length": 0.33, "leaf_len": 0.048},
        ],
        leaf_mat=mat_leaf_bright, stem_mat=mat_stem,
    )

    # 8) UPPER-RIGHT EXTREME-CLOSE PRACTICAL: Cropped Glowing Vintage Amber Edison Pendant Lamp!
    build_foreground_edison_pendant("ForegroundEdison_UR", (0.215, 0.39, 0.194), mat_brass, mat_ed_glass, mat_ed_fil, mat_cord)

    # Save master .blend
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"Saved café foreground scene: {BLEND_PATH}", flush=True)

    # Render HDRI Orientation Probe + 3 half-res Lookdev frames (f001, f075, f145)
    print("=== Rendering HDRI orientation probe ===", flush=True)
    render_hdri_probe(scene, cam_obj, refrac_card)

    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.cycles.samples = 40
    for fnum in (1, 75, 145):
        scene.frame_set(fnum)
        scene.render.filepath = str(WORK / f"lookdev_f{fnum:03d}.png")
        print(f"Rendering café lookdev frame {fnum}...", flush=True)
        bpy.ops.render.render(write_still=True)

    print("CAFE LOOKDEV BUILD COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
