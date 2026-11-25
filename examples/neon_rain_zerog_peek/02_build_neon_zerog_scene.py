import bpy
import bmesh
import math
import json
import numpy as np
from pathlib import Path
from mathutils import Vector, Matrix, Euler, Quaternion

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "neon_rain_zerog_peek"
WORK = PROJ / "work"
FRAMES_DIR = WORK / "frames"
TRACK_JSON = WORK / "camera_track.json"
HDRI_PATH = WORK / "scene_matched_neon_rain_2k.hdr"
BLEND_PATH = WORK / "neon_zerog_scene.blend"

NUM_FRAMES = 120
FPS = 24

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
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 10
    scene.cycles.sample_clamp_indirect = 12.0

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.film_transparent = True
    scene.cycles.film_transparent_glass = False

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = 0.0

    # True 180-deg cinema shutter motion blur for fast colliding shards & spraying impact micro-droplets!
    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = 'CENTER'
    scene.render.motion_blur_shutter = 0.50

def setup_world_hdri_and_dynamic_lights(scene, track):
    world = bpy.data.worlds.new("NeonRainWorld")
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
    n_bg.inputs['Strength'].default_value = 0.76

    nt.links.new(n_tc.outputs['Generated'], n_map.inputs['Vector'])
    nt.links.new(n_map.outputs['Vector'], n_env.inputs['Vector'])
    nt.links.new(n_env.outputs['Color'], n_bg.inputs['Color'])
    nt.links.new(n_bg.outputs['Background'], n_out.inputs['Surface'])

    # 1) Blazing Left Vertical Chinatown Neon Sign Key Light (Crimson-Red + Warm Pink-White)
    ld_n1 = bpy.data.lights.new("Practical_ChinatownNeon_Red", type='AREA')
    ld_n1.shape = 'RECTANGLE'
    ld_n1.size = 0.35
    ld_n1.size_y = 1.45
    ld_n1.energy = 42.0
    ld_n1.color = (1.0, 0.14, 0.20)
    o_n1 = bpy.data.objects.new("Practical_ChinatownNeon_Red", ld_n1)
    scene.collection.objects.link(o_n1)
    o_n1.location = (-0.85, 1.35, 0.18)
    o_n1.rotation_euler = (Vector((0.0, 0.42, 0.0)) - o_n1.location).to_track_quat('-Z', 'Y').to_euler()
    o_n1.visible_camera = False

    # Hot pink-white inner neon tube accent right along left neon core
    ld_n2 = bpy.data.lights.new("Practical_ChinatownNeon_Core", type='AREA')
    ld_n2.shape = 'RECTANGLE'
    ld_n2.size = 0.16
    ld_n2.size_y = 1.10
    ld_n2.energy = 21.0
    ld_n2.color = (1.0, 0.70, 0.74)
    o_n2 = bpy.data.objects.new("Practical_ChinatownNeon_Core", ld_n2)
    scene.collection.objects.link(o_n2)
    o_n2.location = (-0.72, 1.40, 0.22)
    o_n2.rotation_euler = (Vector((-0.10, 0.42, 0.0)) - o_n2.location).to_track_quat('-Z', 'Y').to_euler()
    o_n2.visible_camera = False

    # 2) Right Cobalt-Blue Wet Streetlamp Rim Light
    ld_b = bpy.data.lights.new("Practical_CobaltStreetlamp", type='AREA')
    ld_b.shape = 'DISK'
    ld_b.size = 0.75
    ld_b.energy = 24.0
    ld_b.color = (0.22, 0.46, 1.0)
    o_b = bpy.data.objects.new("Practical_CobaltStreetlamp", ld_b)
    scene.collection.objects.link(o_b)
    o_b.location = (1.15, 1.15, 0.42)
    o_b.rotation_euler = (Vector((0.08, 0.42, 0.0)) - o_b.location).to_track_quat('-Z', 'Y').to_euler()
    o_b.visible_camera = False

    # 3) DYNAMIC SYNCHRONIZED RIGHT CAR HEADLIGHT FLARE LIGHT!
    ld_fl = bpy.data.lights.new("Practical_DynamicHeadlightFlare", type='AREA')
    ld_fl.shape = 'DISK'
    ld_fl.size = 0.35
    ld_fl.color = (0.76, 0.87, 1.0)
    o_fl = bpy.data.objects.new("Practical_DynamicHeadlightFlare", ld_fl)
    scene.collection.objects.link(o_fl)
    o_fl.location = (0.95, 1.45, -0.05)
    o_fl.rotation_euler = (Vector((0.05, 0.40, 0.0)) - o_fl.location).to_track_quat('-Z', 'Y').to_euler()
    o_fl.visible_camera = False

    for fd in track["frames"]:
        bf = fd["frame"] + 1
        fl = float(fd["right_flare"])
        ld_fl.energy = 3.0 + (fl ** 1.3) * 95.0
        ld_fl.keyframe_insert(data_path="energy", frame=bf)

    # 4) Subtle warm golden streetlamp rear kick
    ld_g = bpy.data.lights.new("Practical_WarmStreetBokeh", type='POINT')
    ld_g.energy = 4.8
    ld_g.shadow_soft_size = 0.14
    ld_g.color = (1.0, 0.65, 0.25)
    o_g = bpy.data.objects.new("Practical_WarmStreetBokeh", ld_g)
    scene.collection.objects.link(o_g)
    o_g.location = (0.35, -0.45, 0.30)
    o_g.visible_camera = False

    return world

def build_live_refraction_card(scene, cam_obj, track):
    """
    Invisible-to-camera live-action PNG sequence plane aligned 1:1 with camera frustum at Y=3.2m.
    Visible ONLY to Transmission & Glossy rays so EVERY water orb, liquid ribbon, and shattered glass shard
    physically refracts & bends the REAL moving Chinatown neon sign, rain, and headlight flare!
    """
    dist = 3.2
    fx = float(track["focal_px"])
    w_plane = dist * (1920.0 / fx) * 1.06
    h_plane = dist * (1080.0 / fx) * 1.06

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    card = bpy.context.active_object
    card.name = "LiveAction_RefractionCard"
    card.scale = (w_plane, h_plane, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    card.parent = cam_obj
    card.location = (0.0, 0.0, -dist)
    card.rotation_euler = (0.0, 0.0, 0.0)

    card.visible_camera = False
    card.visible_diffuse = False
    card.visible_shadow = False
    card.visible_volume_scatter = False
    card.visible_transmission = True
    card.visible_glossy = True

    mat = bpy.data.materials.new("Mat_LiveNeonRefractionCard")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 1.05  # Natural filmic neon refraction brightness inside water & glass!
    tex = nt.nodes.new('ShaderNodeTexImage')
    img = bpy.data.images.load(str(FRAMES_DIR / "frame_0000.png"), check_existing=True)
    img.source = 'SEQUENCE'
    tex.image = img
    tex.image_user.frame_duration = NUM_FRAMES
    tex.image_user.frame_start = 1
    tex.image_user.frame_offset = -1
    tex.image_user.use_auto_refresh = True
    tex.extension = 'EXTEND'

    nt.links.new(tex.outputs['Color'], em.inputs['Color'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    card.data.materials.append(mat)
    return card

# ---------------------------------------------------------------------------
# Photorealistic Cinema Materials (Zero-G Rainwater, Spectral Crystal Glass, Dark Chrome, Aged Brass)
# ---------------------------------------------------------------------------
def make_zerog_water_mat(name="Mat_ZeroGRainwater"):
    """Physically accurate IOR 1.333 pure rainwater with crystal specular highlights & subtle iridescent rim."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    set_bsdf(bsdf, **{
        'Base Color': (0.97, 0.99, 1.0, 1.0),
        'Roughness': 0.008,
        'IOR': 1.333,
        'Transmission Weight': 1.0,
        'Specular IOR Level': 0.65,
        'Coat Weight': 0.45,
        'Coat Roughness': 0.005,
    })
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_spectral_crystal_glass_mat(name="Mat_SpectralCrystalGlass"):
    """
    True optical RGB spectral dispersion glass shader (R=1.515, G=1.540, B=1.572) + crisp specular coat!
    Splits crimson neon & blue headlight refractions along beveled glass shard edges like real optical quartz!
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    g_r = nt.nodes.new('ShaderNodeBsdfGlass')
    g_g = nt.nodes.new('ShaderNodeBsdfGlass')
    g_b = nt.nodes.new('ShaderNodeBsdfGlass')
    g_r.inputs['Color'].default_value = (1.0, 0.06, 0.04, 1.0)
    g_g.inputs['Color'].default_value = (0.04, 1.0, 0.06, 1.0)
    g_b.inputs['Color'].default_value = (0.04, 0.08, 1.0, 1.0)
    g_r.inputs['IOR'].default_value = 1.510
    g_g.inputs['IOR'].default_value = 1.545
    g_b.inputs['IOR'].default_value = 1.585
    for g in (g_r, g_g, g_b):
        g.inputs['Roughness'].default_value = 0.018

    add1 = nt.nodes.new('ShaderNodeAddShader')
    add2 = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(g_r.outputs['BSDF'], add1.inputs[0])
    nt.links.new(g_g.outputs['BSDF'], add1.inputs[1])
    nt.links.new(add1.outputs['Shader'], add2.inputs[0])
    nt.links.new(g_b.outputs['BSDF'], add2.inputs[1])

    # Blend subtle Fresnel glossy coat for bright neon edge glints
    gloss = nt.nodes.new('ShaderNodeBsdfGlossy')
    gloss.inputs['Color'].default_value = (0.98, 0.98, 1.0, 1.0)
    gloss.inputs['Roughness'].default_value = 0.025
    fres = nt.nodes.new('ShaderNodeFresnel')
    fres.inputs['IOR'].default_value = 1.58
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(fres.outputs['Fac'], mix.inputs['Fac'])
    nt.links.new(add2.outputs['Shader'], mix.inputs[1])
    nt.links.new(gloss.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    return mat

def make_smoked_glass_mat(name="Mat_SmokedNightGlass"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    set_bsdf(bsdf, **{
        'Base Color': (0.62, 0.68, 0.78, 1.0),
        'Roughness': 0.04,
        'IOR': 1.54,
        'Transmission Weight': 0.94,
        'Coat Weight': 0.55,
        'Coat Roughness': 0.02,
    })
    return mat

def make_dark_titanium_chrome_mat(name="Mat_DarkTitaniumChrome"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 65.0
    noise.inputs['Detail'].default_value = 5.0
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value = 0.08
    mr.inputs['To Max'].default_value = 0.22
    set_bsdf(bsdf, **{
        'Base Color': (0.28, 0.30, 0.35, 1.0),
        'Metallic': 0.98,
        'Anisotropic': 0.45,
        'Coat Weight': 0.35,
        'Coat Roughness': 0.08,
    })
    nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], bsdf.inputs['Roughness'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def make_brushed_gold_brass_mat(name="Mat_BrushedGoldBrass"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 50.0
    noise.inputs['Detail'].default_value = 5.0
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value = 0.10
    mr.inputs['To Max'].default_value = 0.24
    set_bsdf(bsdf, **{
        'Base Color': (0.82, 0.60, 0.28, 1.0),
        'Metallic': 0.98,
        'Anisotropic': 0.42,
    })
    nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], bsdf.inputs['Roughness'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

# ---------------------------------------------------------------------------
# High-Detail Sculpted 3D Asset Builders
# ---------------------------------------------------------------------------
def build_fractured_glass_shard(name, mat, rng, size_scale=1.0):
    """Creates a realistic beveled jagged thick fractured crystal glass shard with faceted fracture planes."""
    bm = bmesh.new()
    n_pts = int(rng.integers(5, 8))
    angles = np.sort(rng.uniform(0, 2.0 * math.pi, size=n_pts))
    # Elongated dramatic jagged shard aspect ratio!
    rx = rng.uniform(0.022, 0.048) * size_scale
    rz = rng.uniform(0.010, 0.024) * size_scale
    thick = rng.uniform(0.0045, 0.0095) * size_scale

    front_v = []
    back_v = []
    for a in angles:
        rad_jit = rng.uniform(0.68, 1.25)
        x = math.cos(a) * rx * rad_jit
        z = math.sin(a) * rz * rad_jit
        # Slight wedge taper across shard for authentic glass fracture thickness!
        y_f = -thick * (0.45 + 0.25 * math.cos(a))
        y_b = +thick * (0.45 + 0.20 * math.sin(a))
        front_v.append(bm.verts.new((x, y_f, z)))
        back_v.append(bm.verts.new((x, y_b, z)))

    # Add central subtle faceted ridge point on front & back so glass refracts prismatically!
    cf = bm.verts.new((rng.uniform(-0.004, 0.004)*size_scale, -thick * 0.68, rng.uniform(-0.003, 0.003)*size_scale))
    cb = bm.verts.new((rng.uniform(-0.004, 0.004)*size_scale, +thick * 0.65, rng.uniform(-0.003, 0.003)*size_scale))

    for i in range(n_pts):
        inext = (i + 1) % n_pts
        bm.faces.new((cf, front_v[i], front_v[inext]))
        bm.faces.new((cb, back_v[inext], back_v[i]))
        bm.faces.new((front_v[i], back_v[i], back_v[inext], front_v[inext]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # Crisp micro-bevel along all sharp fractured edges to catch blazing crimson & blue specular glints!
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.0007 * size_scale, segments=2, affect='EDGES')

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh); bm.free()
    # Keep facets crisp/flat-shaded with sharp bevel highlights (true fractured crystal glass look!)
    for p in mesh.polygons: p.use_smooth = False
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def build_machined_precision_ring(name, metal_mat, accent_mat, radius=0.068, tube_w=0.007, tube_h=0.0055, with_notches=True):
    """Machined chamfered metallic gyroscopic ring with recessed grooves & precision gear-like inner accent."""
    bm = bmesh.new()
    n_u = 64
    # Chamfered octagonal cross-section with subtle center groove
    prof = [
        (-0.50, -0.35), (-0.32, -0.50), (0.32, -0.50), (0.50, -0.35),
        (0.50, -0.10), (0.38, 0.00), (0.50, 0.10),
        (0.50, 0.35), (0.32, 0.50), (-0.32, 0.50), (-0.50, 0.35),
    ]
    rings = []
    for i in range(n_u):
        a = 2.0 * math.pi * i / n_u
        r_mod = radius + (0.0014 * math.cos(16 * a) if with_notches and (i % 4 < 2) else 0.0)
        ca, sa = math.cos(a), math.sin(a)
        rad_dir = Vector((ca, 0.0, sa))
        y_dir = Vector((0.0, 1.0, 0.0))
        rings.append([bm.verts.new(rad_dir * (r_mod + pr * tube_w) + y_dir * (py * tube_h)) for pr, py in prof])

    n_p = len(prof)
    for i in range(n_u):
        r0, r1 = rings[i], rings[(i + 1) % n_u]
        for k in range(n_p):
            kn = (k + 1) % n_p
            bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh); bm.free()
    for p in mesh.polygons: p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(metal_mat)
    bpy.context.scene.collection.objects.link(obj)

    # Inner thin delicate accent ring offset slightly for tactile depth
    bm2 = bmesh.new()
    r_in = radius * 0.80
    r2_rings = []
    for i in range(48):
        a = 2.0 * math.pi * i / 48.0
        rad_dir = Vector((math.cos(a), 0.0, math.sin(a)))
        y_dir = Vector((0.0, 1.0, 0.0))
        r2_rings.append([bm2.verts.new(rad_dir * (r_in + 0.0018 * math.cos(2*math.pi*k/8)) + y_dir * (0.0018 * math.sin(2*math.pi*k/8))) for k in range(8)])
    for i in range(48):
        r0, r1 = r2_rings[i], r2_rings[(i + 1) % 48]
        for k in range(8):
            kn = (k + 1) % 8
            bm2.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    m2 = bpy.data.meshes.new(f"{name}_InnerMesh")
    bm2.to_mesh(m2); bm2.free()
    for p in m2.polygons: p.use_smooth = True
    inner = bpy.data.objects.new(f"{name}_Inner", m2)
    inner.data.materials.append(accent_mat)
    bpy.context.scene.collection.objects.link(inner)
    inner.parent = obj
    inner.rotation_euler = (math.radians(18), math.radians(12), 0.0)
    return obj

# ---------------------------------------------------------------------------
# 240Hz Physical 3D Zero-G Collision & Spin-Transfer Simulator!
# ---------------------------------------------------------------------------
def simulate_zero_g_collisions(bodies, n_frames=NUM_FRAMES, fps=FPS, substeps=10):
    """
    True 3D elastic rigid-body zero-G simulation with:
    - Real pairwise sphere-collision impulses + restitution (e=0.88)
    - Real off-center angular momentum transfer (collisions visibly kick objects into faster tumbling spin!)
    - Soft toroidal framing bounds keeping left/right clusters lushly around the protagonist gap
    - Logs exact 3D collision impact points & times to spawn high-speed micro-droplet splash bursts!
    """
    dt = (1.0 / fps) / substeps
    N = len(bodies)
    pos = np.array([b["pos"] for b in bodies], dtype=np.float64)
    vel = np.array([b["vel"] for b in bodies], dtype=np.float64)
    rad = np.array([b["rad"] for b in bodies], dtype=np.float64)
    mass = np.array([b["mass"] for b in bodies], dtype=np.float64)
    omega = np.array([b["omega"] for b in bodies], dtype=np.float64)
    quats = [Quaternion(b["quat"]) for b in bodies]
    home = np.array([b["home"] for b in bodies], dtype=np.float64)
    stiff = np.array([b["stiff"] for b in bodies], dtype=np.float64)
    side = [b["side"] for b in bodies]

    traj_pos = np.zeros((n_frames, N, 3), dtype=np.float64)
    traj_quat = [[None] * N for _ in range(n_frames)]
    impacts = []  # list of (frame_float, contact_pos_3d, rel_speed)
    last_hit_pair = {}

    for fr in range(n_frames):
        traj_pos[fr] = pos.copy()
        for i in range(N):
            traj_quat[fr][i] = quats[i].copy()

        for sub in range(substeps):
            t_cur = fr + sub / float(substeps)
            # Gentle harmonic tether to each cluster's side framing zone + swirling zero-G vortex eddy
            for i in range(N):
                dp = home[i] - pos[i]
                # Soft pull back if drifting too far from framing edge
                acc = dp * stiff[i]
                # Subtle swirling zero-G current around cluster center
                rel = pos[i] - home[i]
                swirl = np.array([-rel[2] * 0.45, math.sin(t_cur * 0.08 + i) * 0.06, rel[0] * 0.45])
                acc += swirl * (0.6 if bodies[i]["kind"] == "shard" else 0.25)

                # Hard invisible corridor barrier around protagonist face (x in [-0.115..+0.135] at y~0.42!)
                if side[i] == "L" and pos[i, 0] > -0.108:
                    acc[0] -= 42.0 * (pos[i, 0] + 0.108)
                elif side[i] == "R" and pos[i, 0] < 0.138:
                    acc[0] += 42.0 * (0.138 - pos[i, 0])

                vel[i] += acc * dt
                vel[i] *= (1.0 - 0.10 * dt)  # whisper-light air drag

            # Pairwise true 3D elastic collisions + spin transfer!
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pos[j] - pos[i]
                    dist = float(np.linalg.norm(diff))
                    min_d = rad[i] + rad[j]
                    if 1e-6 < dist < min_d:
                        n = diff / dist
                        # Positional separation
                        overlap = min_d - dist
                        inv_m1, inv_m2 = 1.0 / mass[i], 1.0 / mass[j]
                        inv_sum = inv_m1 + inv_m2
                        pos[i] -= n * (overlap * (inv_m1 / inv_sum))
                        pos[j] += n * (overlap * (inv_m2 / inv_sum))

                        rv = vel[j] - vel[i]
                        vn = float(np.dot(rv, n))
                        if vn < 0.0:
                            e = 0.90
                            j_imp = -(1.0 + e) * vn / inv_sum
                            vel[i] -= (j_imp * inv_m1) * n
                            vel[j] += (j_imp * inv_m2) * n

                            # Off-center rotational torque impulse (tumbles glass shards & rocks chrome rings dramatically!)
                            tang = rv - vn * n
                            spin_kick = np.cross(n, tang) * 14.0 + np.cross(n, np.array([0.2, 0.7, -0.3])) * (abs(vn) * 18.0)
                            omega[i] += spin_kick * (inv_m1 / inv_sum)
                            omega[j] -= spin_kick * (inv_m2 / inv_sum)

                            # Clamp max spin for crisp legible motion blur
                            for idx_k in (i, j):
                                sp = float(np.linalg.norm(omega[idx_k]))
                                if sp > 8.5:
                                    omega[idx_k] *= (8.5 / sp)

                            # Log energetic collisions to trigger live micro-water droplet spray bursts!
                            pair_key = (i, j)
                            if abs(vn) > 0.09 and (t_cur - last_hit_pair.get(pair_key, -99.0)) > 8.0:
                                last_hit_pair[pair_key] = t_cur
                                contact = pos[i] + n * rad[i]
                                impacts.append((t_cur, contact.copy(), abs(vn)))

            # Integrate positions & quaternion orientations
            pos += vel * dt
            for i in range(N):
                w = omega[i]
                wn = float(np.linalg.norm(w))
                if wn > 1e-6:
                    dq = Quaternion(Vector(w / wn), wn * dt)
                    quats[i] = (dq @ quats[i]).normalized()
                omega[i] *= (1.0 - 0.14 * dt)

    print(f"  [Physics] Simulated {N} rigid bodies across {n_frames}f -> {len(impacts)} energetic 3D collisions recorded!", flush=True)
    return traj_pos, traj_quat, impacts

def build_zero_g_metaball_water_system(scene, water_mat, rng):
    """
    Creates a HIGH-RES animated Blender Metaball Zero-G Liquid Rainwater System:
    - Real surface-tension merging, liquid neck stretching, coalescing jelly wobble, & splitting!
    - Left Neon Cluster: two large hero water globules collide -> stretch a shimmering liquid bridge ->
      fuse into a giant crimson-lit wobbling orb -> pinch apart into orbiting liquid streams!
    - Right Cobalt Cluster: spiraling rainwater ribbon weaves right through the machined dark-chrome ring,
      coalescing & splitting under the booming blue-white car headlight flare!
    - Extreme-close lens droplets (~0.23m) gliding slowly past top/bottom edges for creamy cinema bokeh!
    """
    mball_data = bpy.data.metaballs.new("ZeroG_LiquidWater_MB")
    mball_data.resolution = 0.0032
    mball_data.render_resolution = 0.0024  # Ultra-smooth glossy fluid surface!
    mball_data.threshold = 0.62

    mb_obj = bpy.data.objects.new("ZeroG_LiquidWater", mball_data)
    mb_obj.data.materials.append(water_mat)
    scene.collection.objects.link(mb_obj)

    # Define choreographed dynamic liquid blobs (positions animated per frame via real spring/merge/split equations!)
    elems = []

    def add_blob(r0):
        el = mball_data.elements.new(type='BALL')
        el.radius = r0
        el.stiffness = 2.0
        elems.append(el)
        return el

    # --- LEFT CLUSTER (18 dynamic blobs right in front of crimson Chinatown neon! x=-0.27..-0.12, y=0.33..0.48) ---
    # 2 giant merging/splitting hero globules + 4 liquid-bridge droplets + 12 orbiting swirling rainwater pearls!
    left_heroes = [add_blob(0.032), add_blob(0.027)]
    left_bridge = [add_blob(0.017) for _ in range(4)]
    left_stream = [add_blob(rng.uniform(0.011, 0.020)) for _ in range(12)]
    left_stream_params = [(rng.uniform(0, 6.28), rng.uniform(0.045, 0.11), rng.uniform(-0.12, 0.14), rng.uniform(1.4, 2.8)) for _ in range(12)]

    # --- RIGHT CLUSTER (18 dynamic blobs in front of cobalt lamppost & exploding headlight! x=+0.15..+0.32, y=0.33..0.50) ---
    right_heroes = [add_blob(0.030), add_blob(0.025)]
    right_bridge = [add_blob(0.016) for _ in range(4)]
    right_stream = [add_blob(rng.uniform(0.011, 0.020)) for _ in range(12)]
    right_stream_params = [(rng.uniform(0, 6.28), rng.uniform(0.045, 0.105), rng.uniform(-0.12, 0.14), rng.uniform(1.5, 3.0)) for _ in range(12)]

    # --- EXTREME-CLOSE LENS BOKEH DROPLETS (8 blobs at y=0.22..0.28m framing top/bottom corners!) ---
    close_blobs = [add_blob(rng.uniform(0.012, 0.019)) for _ in range(8)]
    close_specs = [
        (-0.11, 0.24, -0.105, 0.018, 0.010, 0.0),
        (-0.04, 0.23, -0.115, 0.024, 0.008, 1.2),
        ( 0.08, 0.25, -0.118, -0.020, 0.009, 2.4),
        ( 0.15, 0.24, -0.100, -0.015, 0.012, 3.1),
        (-0.14, 0.26,  0.115, 0.020, -0.012, 0.7),
        (-0.07, 0.25,  0.122, 0.015, -0.008, 1.9),
        ( 0.13, 0.26,  0.115, -0.022, -0.010, 4.0),
        ( 0.18, 0.27,  0.095, -0.016, -0.014, 5.1),
    ]

    # Keyframe every metaball element cleanly across frames 1..120!
    for fr in range(1, NUM_FRAMES + 1):
        t = (fr - 1) / float(FPS)         # 0.0 .. 5.0s
        u = (fr - 1) / float(NUM_FRAMES - 1)

        # 1) LEFT MERGE -> WOBBLE -> SPLIT CHOREOGRAPHY:
        # Separation distance: starts at 0.085m -> rushes together & fuses at t=1.6..2.8s -> splits sharply at t=3.2..5.0s!
        sep_L = 0.042 * (1.0 - math.exp(-((t - 2.15) / 0.85) ** 2) * 0.92) + 0.014 * math.sin(t * 6.5) * math.exp(-((t - 2.2) / 0.9) ** 2)
        cx_L = -0.205 + 0.022 * math.sin(t * 1.1)
        cy_L = 0.405 + 0.020 * math.cos(t * 1.3)
        cz_L = 0.015 + 0.028 * math.sin(t * 0.9 + 0.4)
        ang_L = t * 1.15

        pL0 = Vector((cx_L - sep_L * math.cos(ang_L), cy_L - 0.015 * math.sin(ang_L), cz_L + sep_L * math.sin(ang_L) * 0.85))
        pL1 = Vector((cx_L + sep_L * math.cos(ang_L), cy_L + 0.015 * math.sin(ang_L), cz_L - sep_L * math.sin(ang_L) * 0.85))
        left_heroes[0].co = pL0
        left_heroes[1].co = pL1
        # Surface-tension jelly breathing during coalescence!
        wob_L = 1.0 + 0.16 * math.sin(t * 11.0) * math.exp(-((t - 2.2) / 1.1) ** 2)
        left_heroes[0].radius = 0.032 * wob_L
        left_heroes[1].radius = 0.027 / (wob_L ** 0.7)

        # Liquid neck bridge beads connecting pL0 & pL1 (creates real stretching water filaments when splitting!)
        for k, el in enumerate(left_bridge):
            frac = (k + 1) / 5.0
            mid = pL0.lerp(pL1, frac)
            mid.x += 0.010 * math.sin(t * 4.5 + k * 1.4) * (0.4 + sep_L * 8.0)
            mid.z += 0.012 * math.cos(t * 4.0 + k * 1.7) * (0.4 + sep_L * 8.0)
            el.co = mid
            el.radius = 0.0155 + 0.004 * math.sin(t * 6.0 + k)

        # Swirling zero-G satellite water stream weaving around left neon cluster
        for el, (ph, rad_orb, z0, spd) in zip(left_stream, left_stream_params):
            a = ph + t * spd
            x = -0.215 + rad_orb * 0.65 * math.cos(a)
            y = 0.415 + rad_orb * 0.55 * math.sin(a)
            z = z0 + 0.045 * math.sin(a * 0.8 + ph)
            x = min(x, -0.115)  # never cross protagonist's face!
            el.co = (x, y, z)

        # 2) RIGHT SPIRALING WATER RIBBON WEAVING THROUGH CHROME RING + COALESCING UNDER HEADLIGHT FLARE!
        sep_R = 0.044 * (1.0 - math.exp(-((t - 3.35) / 0.80) ** 2) * 0.90)
        cx_R = 0.235 + 0.020 * math.cos(t * 1.2)
        cy_R = 0.410 + 0.022 * math.sin(t * 1.0)
        cz_R = 0.005 + 0.030 * math.cos(t * 0.85)
        ang_R = -t * 1.25 + 0.8
        pR0 = Vector((cx_R - sep_R * math.cos(ang_R), cy_R + 0.018 * math.sin(ang_R), cz_R + sep_R * math.sin(ang_R) * 0.90))
        pR1 = Vector((cx_R + sep_R * math.cos(ang_R), cy_R - 0.018 * math.sin(ang_R), cz_R - sep_R * math.sin(ang_R) * 0.90))
        pR0.x = max(pR0.x, 0.150)
        pR1.x = max(pR1.x, 0.150)
        right_heroes[0].co = pR0
        right_heroes[1].co = pR1
        wob_R = 1.0 + 0.18 * math.sin(t * 12.0) * math.exp(-((t - 3.35) / 1.0) ** 2)
        right_heroes[0].radius = 0.030 * wob_R
        right_heroes[1].radius = 0.025 / (wob_R ** 0.7)

        for k, el in enumerate(right_bridge):
            frac = (k + 1) / 5.0
            mid = pR0.lerp(pR1, frac)
            mid.x = max(0.150, mid.x + 0.010 * math.cos(t * 4.8 + k * 1.3))
            mid.z += 0.012 * math.sin(t * 4.2 + k * 1.6)
            el.co = mid
            el.radius = 0.015 + 0.004 * math.cos(t * 6.5 + k)

        for el, (ph, rad_orb, z0, spd) in zip(right_stream, right_stream_params):
            a = ph - t * spd
            x = max(0.148, 0.240 + rad_orb * 0.68 * math.cos(a))
            y = 0.415 + rad_orb * 0.58 * math.sin(a)
            z = z0 + 0.050 * math.cos(a * 0.85 + ph)
            el.co = (x, y, z)

        # 3) EXTREME-CLOSE LENS BOKEH DROPLETS drifting gently past camera edges
        for el, (x0, y0, z0, vx, vz, ph) in zip(close_blobs, close_specs):
            el.co = (
                x0 + vx * math.sin(t * 0.75 + ph),
                y0 + 0.012 * math.cos(t * 0.9 + ph),
                z0 + vz * math.sin(t * 1.1 + ph * 0.7),
            )

        # Insert keyframes on every metaball element
        for el in elems:
            el.keyframe_insert(data_path="co", frame=fr)
            el.keyframe_insert(data_path="radius", frame=fr)

    print(f"  [Liquid] Animated {len(elems)} high-res zero-G metaball water elements across {NUM_FRAMES} frames!", flush=True)
    return mb_obj

def spawn_collision_spray_droplets(scene, impacts, water_mat, rng):
    """
    Spawns high-speed refractive micro-water droplets radiating dynamically from the EXACT 3D impact
    points & times of our rigid-body glass/ring collisions -> razor-real 180-deg 3D shutter motion blur!
    """
    # Select up to 6 strongest impact moments across the 120 frames
    impacts_sorted = sorted(impacts, key=lambda x: -x[2])
    chosen = []
    for imp in impacts_sorted:
        if all(abs(imp[0] - c[0]) > 10.0 for c in chosen):
            chosen.append(imp)
        if len(chosen) >= 6:
            break
    chosen.sort(key=lambda x: x[0])
    print(f"  [Spray] Spawning radial micro-droplet bursts at {len(chosen)} collision beats: {[round(c[0],1) for c in chosen]}", flush=True)

    # Share one clean UV sphere mesh across all micro-droplets for lightning-fast OptiX BVH!
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=1.0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    drop_mesh = bpy.data.meshes.new("MicroDroplet_SharedMesh")
    bm.to_mesh(drop_mesh); bm.free()
    for p in drop_mesh.polygons: p.use_smooth = True
    drop_mesh.materials.append(water_mat)

    count = 0
    for b_idx, (t_hit, c_pos, spd) in enumerate(chosen):
        n_drops = 11
        for k in range(n_drops):
            obj = bpy.data.objects.new(f"ImpactDrop_{b_idx}_{k}", drop_mesh)
            scene.collection.objects.link(obj)
            # Random outward radial burst velocity (0.35 .. 1.05 m/s -> awesome 180-deg shutter motion blur streaks!)
            v_dir = rng.normal(size=3)
            v_dir /= max(np.linalg.norm(v_dir), 1e-5)
            v_dir[1] *= 0.45  # keep mostly within dramatic foreground depth plane
            speed = rng.uniform(0.32, 0.92) * (0.7 + min(spd * 2.0, 1.0))
            v_vec = v_dir * speed
            r_drop = rng.uniform(0.0024, 0.0052)

            f_hit = int(round(t_hit)) + 1
            for fr in range(1, NUM_FRAMES + 1):
                dt_since = (fr - f_hit) / float(FPS)
                if dt_since < 0.0:
                    # Hide cleanly at tiny scale inside collision cluster before impact
                    obj.location = tuple(c_pos)
                    obj.scale = (0.00001, 0.00001, 0.00001)
                else:
                    # Decelerate gently in zero-G air drag + tiny stretch along velocity direction!
                    drag = (1.0 - math.exp(-1.6 * dt_since)) / 1.6
                    p = c_pos + v_vec * drag
                    fade = max(0.00001, 1.0 - max(0.0, (dt_since - 1.4) / 0.8))
                    s = r_drop * fade
                    obj.location = tuple(p)
                    obj.scale = (s, s, s)
                obj.keyframe_insert(data_path="location", frame=fr)
                obj.keyframe_insert(data_path="scale", frame=fr)
            count += 1
    return count

def build_camera_from_track(scene, track):
    cam_data = bpy.data.cameras.new("NeonRain_CinemaCam")
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = 36.0
    cam_data.lens = float(track["focal_mm_35"])  # 24.0mm wide-angle cinema prime!
    cam_data.clip_start = 0.02
    cam_data.clip_end = 50.0

    # Authentic cinema shallow DoF focused on protagonist (~2.8m):
    # f/2.80 keeps glass fracture facets, ring bevels, & stretching water filaments sharp & tactile while giving lush bokeh!
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 2.80
    cam_data.dof.aperture_fstop = 2.80
    cam_data.dof.aperture_blades = 7

    cam_obj = bpy.data.objects.new("NeonRain_CinemaCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    R_base_lookY = Matrix.Rotation(math.pi * 0.5, 3, 'X')
    for fd in track["frames"]:
        bf = fd["frame"] + 1
        rx, ry, rz = fd["rot_xyz_bl"]
        R_cam = Euler((rx, ry, rz), 'XYZ').to_matrix() @ R_base_lookY
        cam_obj.location = tuple(fd["pos_bl_m"])
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = R_cam.to_quaternion()
        cam_obj.keyframe_insert(data_path="location", frame=bf)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=bf)

    return cam_obj

def render_hdri_probe(scene, cam_obj, card_obj):
    hidden = []
    for obj in scene.objects:
        if obj.type in ('MESH', 'META') and obj != card_obj and not obj.hide_render:
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
    scene.frame_set(60)
    scene.render.filepath = str(WORK / "lookdev_hdri_probe.png")
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(s_ch, do_unlink=True)
    bpy.data.objects.remove(s_df, do_unlink=True)
    for obj in hidden: obj.hide_render = False
    cam_obj.data.dof.use_dof, scene.render.use_motion_blur, scene.render.film_transparent, scene.cycles.samples, scene.render.resolution_x, scene.render.resolution_y = old

def main():
    print("=== Building Neon Rain Zero-G Interactive Foreground Scene in Blender 5.1.2 ===", flush=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = NUM_FRAMES

    rng = np.random.default_rng(7062407)
    track = json.loads(TRACK_JSON.read_text(encoding="utf-8"))
    setup_cycles_optix(scene, samples=64)
    setup_world_hdri_and_dynamic_lights(scene, track)
    cam_obj = build_camera_from_track(scene, track)
    refrac_card = build_live_refraction_card(scene, cam_obj, track)

    # Materials
    mat_water = make_zerog_water_mat()
    mat_crystal = make_spectral_crystal_glass_mat()
    mat_smoked = make_smoked_glass_mat()
    mat_chrome = make_dark_titanium_chrome_mat()
    mat_brass = make_brushed_gold_brass_mat()

    # Build 4 hero machined metallic gyroscopic rings + 24 jagged fractured crystal glass shards
    # All simulated together in our true 240Hz 3D elastic collision + spin-transfer solver!
    rigid_objs = []
    bodies = []

    # 1) 4 HERO KINETIC METALLIC RINGS (2 left framing crimson neon, 2 right framing cobalt lamppost/headlight!)
    ring_specs = [
        ("Ring_L_HeroChrome", mat_chrome, mat_brass, 0.068, "L", [-0.235, 0.42,  0.045], [ 0.05, -0.02, -0.04], [ 1.1, 1.8, 0.7], 0.35),
        ("Ring_L_BrassArc",   mat_brass,  mat_chrome, 0.052, "L", [-0.185, 0.37, -0.075], [-0.04,  0.03,  0.06], [-1.5, 0.9, 1.4], 0.25),
        ("Ring_R_HeroChrome", mat_chrome, mat_brass, 0.072, "R", [ 0.255, 0.43,  0.035], [-0.05, -0.02, -0.03], [-1.2, -1.6, 0.8], 0.38),
        ("Ring_R_BrassArc",   mat_brass,  mat_chrome, 0.054, "R", [ 0.215, 0.38, -0.082], [ 0.04,  0.02,  0.05], [ 1.4, -0.8, -1.3], 0.26),
    ]
    for rname, m1, m2, r_rad, sd, p0, v0, om0, mass in ring_specs:
        r_obj = build_machined_precision_ring(rname, m1, m2, radius=r_rad, tube_w=0.0075, tube_h=0.0055)
        rigid_objs.append(r_obj)
        bodies.append({
            "kind": "ring", "side": sd, "pos": p0, "vel": v0, "rad": r_rad * 0.58,
            "mass": mass, "omega": om0, "quat": Euler((rng.uniform(-0.8,0.8), rng.uniform(-0.8,0.8), rng.uniform(-0.8,0.8))).to_quaternion(),
            "home": p0, "stiff": 2.4,
        })

    # 2) 24 JAGGED FRACTURED CRYSTAL GLASS SHARDS (12 Left + 12 Right, launched into rich colliding orbits!)
    for idx in range(24):
        sd = "L" if idx < 12 else "R"
        k = idx if idx < 12 else idx - 12
        mat_s = mat_crystal if (k % 3 != 2) else mat_smoked
        sz = float(rng.uniform(0.75, 1.45))
        s_obj = build_fractured_glass_shard(f"GlassShard_{sd}_{k:02d}", mat_s, rng, size_scale=sz)
        rigid_objs.append(s_obj)

        # Place inside dramatic left/right framing tunnels with targeted opposing velocities for real collisions!
        if sd == "L":
            px = float(rng.uniform(-0.30, -0.14))
            py = float(rng.uniform(0.32, 0.50))
            pz = float(rng.uniform(-0.145, 0.145))
            home = [-0.22, 0.41, pz * 0.7]
        else:
            px = float(rng.uniform(0.165, 0.33))
            py = float(rng.uniform(0.32, 0.50))
            pz = float(rng.uniform(-0.145, 0.145))
            home = [0.245, 0.41, pz * 0.7]

        # Give energetic initial velocities pointed gently toward cluster center so real collisions trigger cleanly!
        to_ctr = np.array(home) - np.array([px, py, pz])
        to_ctr /= max(np.linalg.norm(to_ctr), 1e-4)
        vel0 = (to_ctr * rng.uniform(0.12, 0.32) + rng.normal(0, 0.08, 3)).tolist()
        om0 = rng.uniform(-3.2, 3.2, 3).tolist()
        bodies.append({
            "kind": "shard", "side": sd, "pos": [px, py, pz], "vel": vel0,
            "rad": 0.026 * sz, "mass": 0.04 * (sz ** 2), "omega": om0,
            "quat": Euler(tuple(rng.uniform(0, 6.28, 3))).to_quaternion(),
            "home": home, "stiff": 3.2,
        })

    # Run 240Hz true 3D elastic collision & angular momentum physics!
    traj_pos, traj_quat, impacts = simulate_zero_g_collisions(bodies, n_frames=NUM_FRAMES, fps=FPS, substeps=10)

    # Bake keyframes onto all 28 rigid bodies (4 rings + 24 crystal glass shards)
    for i, obj in enumerate(rigid_objs):
        obj.rotation_mode = 'QUATERNION'
        for fr in range(NUM_FRAMES):
            bf = fr + 1
            obj.location = tuple(traj_pos[fr, i])
            obj.rotation_quaternion = traj_quat[fr][i]
            obj.keyframe_insert(data_path="location", frame=bf)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=bf)

    # 3) Build high-res Zero-G Morphing/Coalescing/Splitting Liquid Rainwater Metaballs!
    build_zero_g_metaball_water_system(scene, mat_water, rng)

    # 4) Spawn high-speed micro-droplet radial spray bursts at exact 3D collision points!
    spawn_collision_spray_droplets(scene, impacts, mat_water, rng)

    # Save master .blend
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"Saved Neon Rain Zero-G scene: {BLEND_PATH}", flush=True)

    # Render HDRI Orientation Probe + 3 half-res Lookdev frames (f010, f060, f112 -> showing crimson neon + peak blue headlight flare!)
    print("=== Rendering HDRI orientation probe ===", flush=True)
    render_hdri_probe(scene, cam_obj, refrac_card)

    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.cycles.samples = 40
    for fnum in (10, 60, 112):
        scene.frame_set(fnum)
        scene.render.filepath = str(WORK / f"lookdev_f{fnum:03d}.png")
        print(f"Rendering Neon Rain lookdev frame {fnum}...", flush=True)
        bpy.ops.render.render(write_still=True)

    print("NEON RAIN ZERO-G LOOKDEV COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
