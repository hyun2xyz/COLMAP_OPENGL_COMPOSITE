import bpy
import time
from pathlib import Path

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cloth_tunnel"
WORK = PROJ / "work"
BLEND_PATH = WORK / "cloth_scene.blend"
BEAUTY_DIR = WORK / "render_beauty"
WIRE_DIR = WORK / "render_wire"

NUM_SHOT_FRAMES = 125

def configure_optix(scene, samples=64, adaptive=0.012):
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == 'OPTIX')
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = adaptive
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPTIX'
    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'

def main():
    BEAUTY_DIR.mkdir(parents=True, exist_ok=True)
    WIRE_DIR.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.context.scene
    cam_obj = scene.camera

    # ---------------- PASS A: BEAUTY (RENDER ONLY, DoF + 3D Motion Blur + Scene HDRI) ----------------
    print("=== PASS A: Rendering 125 full-HD Beauty frames (Cycles OptiX 64spp) ===", flush=True)
    configure_optix(scene, samples=64, adaptive=0.012)
    cam_obj.data.dof.use_dof = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.50
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Base Contrast'

    t0 = time.time()
    for fnum in range(1, NUM_SHOT_FRAMES + 1):
        out_p = BEAUTY_DIR / f"frame_{fnum-1:04d}.png"
        if out_p.exists() and out_p.stat().st_size > 10000:
            continue
        scene.frame_set(fnum)
        scene.render.filepath = str(out_p)
        bpy.ops.render.render(write_still=True)
        if fnum % 15 == 0 or fnum == 1 or fnum == NUM_SHOT_FRAMES:
            el = time.time() - t0
            print(f"  [Beauty] {fnum:03d}/{NUM_SHOT_FRAMES} ({el:.1f}s elapsed)", flush=True)

    # ---------------- PASS B: 3D VIEW (Shaded Clay + Clean True-Quad Wireframe) ----------------
    print("=== PASS B: Rendering 125 full-HD 3D VIEW Clay+Wireframe frames (Cycles OptiX 16spp) ===", flush=True)
    configure_optix(scene, samples=16, adaptive=0.03)
    scene.render.image_settings.color_depth = '8'
    cam_obj.data.dof.use_dof = False
    scene.render.use_motion_blur = False

    clay_mat = bpy.data.materials.get("Mat_3DView_Clay") or bpy.data.materials.new("Mat_3DView_Clay")
    clay_mat.use_nodes = True
    bsdf_c = clay_mat.node_tree.nodes.get("Principled BSDF")
    bsdf_c.inputs["Base Color"].default_value = (0.52, 0.52, 0.53, 1.0)
    bsdf_c.inputs["Roughness"].default_value = 0.74
    if "Specular IOR Level" in bsdf_c.inputs:
        bsdf_c.inputs["Specular IOR Level"].default_value = 0.22

    wire_mat = bpy.data.materials.get("Mat_3DView_Wire") or bpy.data.materials.new("Mat_3DView_Wire")
    wire_mat.use_nodes = True
    bsdf_w = wire_mat.node_tree.nodes.get("Principled BSDF")
    bsdf_w.inputs["Base Color"].default_value = (0.015, 0.015, 0.018, 1.0)
    bsdf_w.inputs["Roughness"].default_value = 0.95
    if "Specular IOR Level" in bsdf_w.inputs:
        bsdf_w.inputs["Specular IOR Level"].default_value = 0.05

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
            wf.thickness = 0.00050 if is_small else 0.00095

    t1 = time.time()
    for fnum in range(1, NUM_SHOT_FRAMES + 1):
        out_p = WIRE_DIR / f"frame_{fnum-1:04d}.png"
        scene.frame_set(fnum)
        scene.render.filepath = str(out_p)
        bpy.ops.render.render(write_still=True)
        if fnum % 25 == 0 or fnum == 1 or fnum == NUM_SHOT_FRAMES:
            el = time.time() - t1
            print(f"  [3DView] {fnum:03d}/{NUM_SHOT_FRAMES} ({el:.1f}s elapsed, {el/fnum:.2f}s/f)", flush=True)

    print(f"ALL RENDERS DONE! Total elapsed: {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
