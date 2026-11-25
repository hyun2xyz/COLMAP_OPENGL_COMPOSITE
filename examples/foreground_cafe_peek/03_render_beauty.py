import bpy
import time
from pathlib import Path

ROOT = Path(r"D:\VFX_Work_Claude\_repo\COLMAP_OPENGL_COMPOSITE")
PROJ = ROOT / "examples" / "foreground_cafe_peek"
WORK = PROJ / "work"
BLEND_PATH = WORK / "cafe_peek_scene.blend"
BEAUTY_DIR = WORK / "render_beauty"

NUM_FRAMES = 150

def main():
    BEAUTY_DIR.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    scene = bpy.context.scene
    cam_obj = scene.camera

    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == 'OPTIX')

    scene.cycles.device = 'GPU'
    scene.cycles.samples = 96
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.010
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPTIX'
    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 10
    scene.cycles.transparent_max_bounces = 10

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.cycles.film_transparent_glass = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '16'
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Base Contrast'
    scene.view_settings.exposure = 0.0
    cam_obj.data.dof.use_dof = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.45

    print(f"=== Rendering {NUM_FRAMES} Full-HD 16-bit Beauty frames (Cycles OptiX 96spp) ===", flush=True)
    t0 = time.time()
    for fnum in range(1, NUM_FRAMES + 1):
        out_p = BEAUTY_DIR / f"frame_{fnum-1:04d}.png"
        if out_p.exists() and out_p.stat().st_size > 15000:
            continue
        scene.frame_set(fnum)
        scene.render.filepath = str(out_p)
        bpy.ops.render.render(write_still=True)
        if fnum % 15 == 0 or fnum == 1 or fnum == NUM_FRAMES:
            el = time.time() - t0
            print(f"  [Beauty] {fnum:03d}/{NUM_FRAMES} ({el:.1f}s elapsed, {el/fnum:.2f}s/f)", flush=True)

    print(f"ALL {NUM_FRAMES} BEAUTY FRAMES RENDERED in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
