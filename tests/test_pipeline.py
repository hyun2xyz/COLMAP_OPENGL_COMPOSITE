"""Fast checks that run anywhere (no footage, no GPU): camera math, settings, CLI."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tracking"))

from pipeline.common import projection, shot_config  # noqa: E402
from pipeline.__main__ import parser, STEPS  # noqa: E402
import export_colmap  # noqa: E402
sys.path.insert(0, str(ROOT/"scripts"))
import find_materials  # noqa: E402


class CameraMath(unittest.TestCase):
    def test_colmap_to_three_roundtrip(self):
        """A point seen by the COLMAP camera lands on the same pixel through matrixWorld + GL projection."""
        rng = np.random.default_rng(1)
        cam = dict(width=1280, height=720, fx=850., fy=850., cx=640., cy=360.)
        P = projection(cam)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            t = rng.normal(size=3)
            world = np.array(export_colmap.matrix_world([*q, *t])).reshape(4, 4).T
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            optical = np.array([rng.uniform(-.5, .5), rng.uniform(-.3, .3), rng.uniform(2, 6)])
            X = R.T@(optical-t)  # world point in front of the COLMAP camera
            expected = optical[:2]/optical[2]*[cam["fx"], cam["fy"]]+[cam["cx"], cam["cy"]]
            clip = P@np.linalg.inv(world)@np.r_[X, 1]
            ndc = clip[:2]/clip[3]
            uv = [(ndc[0]+1)*cam["width"]/2, (1-ndc[1])*cam["height"]/2]
            np.testing.assert_allclose(uv, expected, atol=1e-6)


class Settings(unittest.TestCase):
    def test_shots_are_complete(self):
        for path in sorted((ROOT/"shots").glob("*.json")):
            cfg = shot_config(path.stem)
            with self.subTest(shot=path.stem):
                self.assertIn(cfg["kind"], ("water", "rigid_spheres", "cloth"))
                for key in ("download", "sha256", "license", "creator", "page"):
                    self.assertTrue(cfg["source"][key])
                for key in ("file", "download", "sha256", "license"):
                    self.assertTrue(cfg["hdri"][key])
                self.assertEqual(len(cfg["plate"]["size"]), 2)
                self.assertGreater(cfg["camera"]["focal_px"], 0)

    def test_cli_knows_every_step(self):
        for step in [*STEPS, "composite"]:
            args = parser().parse_args([step, "hall_water"])
            self.assertEqual(args.step, step)


class Materials(unittest.TestCase):
    """Search results are parsed offline from the documented API response shapes."""

    def test_pexels_keeps_long_landscape_clips_and_a_1920_file(self):
        data = {"videos": [
            {"id": 1, "url": "https://www.pexels.com/video/a-1/", "duration": 12, "width": 3840, "height": 2160,
             "user": {"name": "A"}, "image": "p.jpg", "video_files": [
                 {"file_type": "video/mp4", "width": 3840, "height": 2160, "fps": 30, "link": "uhd"},
                 {"file_type": "video/mp4", "width": 1920, "height": 1080, "fps": 30, "link": "hd"},
                 {"file_type": "video/mp4", "width": 1080, "height": 1920, "fps": 30, "link": "portrait"}]},
            {"id": 2, "url": "u", "duration": 3, "width": 1920, "height": 1080, "user": {}, "video_files": []},
            {"id": 3, "url": "u", "duration": 20, "width": 1080, "height": 1920, "user": {}, "video_files": []}]}
        found = find_materials.pexels_candidates(data, min_seconds=6)
        self.assertEqual([c["id"] for c in found], ["pexels:1"])
        self.assertEqual((found[0]["download"], found[0]["license"]), ("hd", "pexels"))

    def test_pixabay_picks_widest_when_nothing_reaches_1920(self):
        data = {"hits": [{"id": 9, "pageURL": "https://pixabay.com/videos/x-9/", "duration": 8, "user": "B", "videos": {
            "medium": {"url": "m", "width": 1280, "height": 720}, "small": {"url": "s", "width": 960, "height": 540}}}]}
        found = find_materials.pixabay_candidates(data, min_seconds=6)
        self.assertEqual((found[0]["download"], found[0]["license"]), ("m", "pixabay"))

    def test_hdri_search_ranks_matches_and_filters_category(self):
        assets = {
            "a_forest": {"name": "A Forest", "tags": ["trees"], "categories": ["outdoor", "overcast"], "date_published": 1},
            "b_hall": {"name": "B Hall", "tags": ["forest mural"], "categories": ["indoor"], "date_published": 2},
            "c_sky": {"name": "C Sky", "tags": [], "categories": ["skies"], "date_published": 3}}
        ids = [r[2] for r in find_materials.hdri_matches(assets, ["forest", "overcast"])]
        self.assertEqual(ids, ["a_forest", "b_hall"])
        ids = [r[2] for r in find_materials.hdri_matches(assets, ["forest"], category="indoor")]
        self.assertEqual(ids, ["b_hall"])

    def test_every_license_has_terms(self):
        for key, terms in find_materials.LICENSES.items():
            with self.subTest(license=key):
                self.assertTrue(terms["license"] and terms["license_url"].startswith("https://") and terms["permissions"])


class Lighting(unittest.TestCase):
    def test_hdri_solid_angles_cover_the_sphere(self):
        from pipeline.light import hdri_directions
        dirs, omega = hdri_directions(64, 128)
        self.assertAlmostEqual(float(omega.sum()), 4*np.pi, places=2)
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=-1), 1, atol=1e-9)

    def test_two_groups_separates_two_lamps(self):
        from pipeline.light import two_groups
        rng = np.random.default_rng(1)
        a = np.array([0, .8, .6])+rng.normal(0, .02, (50, 3))
        b = np.array([.9, .2, -.4])+rng.normal(0, .02, (20, 3))
        dirs = np.concatenate([a, b])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        groups = two_groups(dirs, np.ones(len(dirs)))
        self.assertEqual([int(g[2].sum()) for g in groups], [50, 20])
        self.assertGreater(float(groups[0][1]@np.array([0, .8, .6])), .99)

    def test_sphere_rotations_stay_orthonormal(self):
        from pipeline.relight_spheres import rotations
        t = np.linspace(0, 1, 40)[:, None, None]
        positions = np.concatenate([np.stack([t[:, 0]*3, .3+0*t[:, 0], t[:, 0]], -1)]*2, axis=1)
        rots = rotations(positions, np.array([.2, .3]), seed=3)
        for r in rots.reshape(-1, 3, 3):
            np.testing.assert_allclose(r@r.T, np.eye(3), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
