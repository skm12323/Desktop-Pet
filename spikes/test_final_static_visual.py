"""Rendered FINAL regressions: floor, mood alignment, rest fidelity and eye return."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from render_rig_rest import RigRenderer
from pet.asset_provider import SpriteRef
from pet.rig.motion import MotionEngine, MotionInputs


def premul_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a[:, :, :3]*a[:, :, 3:] - b[:, :, :3]*b[:, :, 3:]).max(-1)/255


class FinalStaticVisual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = RigRenderer(stage="final")

    @classmethod
    def tearDownClass(cls):
        cls.r.close()

    def setUp(self):
        self.r.win.set_sprite(SpriteRef(str(ROOT / "assets/ai/final_healthy_neutral.png"), 320, 320))

    def test_all_ten_mood_poses_have_a_common_floor_and_size(self):
        heights = []
        for branch in ("healthy", "neglected"):
            for mood in ("neutral", "happy", "sad", "hungry", "sleepy"):
                with self.subTest(branch=branch, mood=mood):
                    self.r.win.set_sprite(SpriteRef(str(ROOT / f"assets/ai/final_{branch}_{mood}.png"), 320, 320))
                    im = self.r.render(self.r.rest_frame(), size=(320, 320), ground_shift=True)
                    yy, xx = np.where(np.asarray(im)[:, :, 3] >= 24)
                    self.assertEqual(int(yy.max()), 319, "pose does not meet the window floor")
                    heights.append(int(yy.max()-yy.min()+1))
        self.assertLessEqual(max(heights)-min(heights), 2)
        self.assertTrue(all(305 <= h <= 309 for h in heights))

    def test_waiting_keeps_feet_stable_and_upper_body_alive(self):
        engine = MotionEngine(self.r.spec)
        bottoms, chests = [], []
        for _ in range(75):
            f = engine.step(MotionInputs(grounded=True, walking=False), 66)
            im = self.r.render(f, size=(320, 320), ground_shift=True)
            alpha = np.asarray(im)[280:, :, 3]
            bottoms.append(int(np.where(alpha > 128)[0].max())+280)
            chests.append(f.bone_angles["chest"])
        self.assertEqual(set(bottoms), {319})
        self.assertGreater(max(chests)-min(chests), .8)

    def test_final_rest_preserves_the_key_art(self):
        rest = self.r.render(self.r.rest_frame())
        key = Image.open(ROOT / "assets/rig_final/references/front_key.png").convert("RGBA")
        diff = premul_difference(rest, key)
        foreground = np.asarray(key)[:, :, 3] > 0
        self.assertLess(float(diff[foreground].mean()), .03)
        self.assertLess(float((diff[foreground] > 24).mean()), .0002)

    def test_closed_eyes_and_gaze_restore_without_residue(self):
        rest = self.r.render(self.r.rest_frame())
        open_a = np.asarray(rest)[215:285, 430:610, :3].astype(int)
        # Closed lids expose light warm skin; it is not blue-white sclera.
        white = lambda a: int(((a.min(-1) > 225) & (np.ptp(a, axis=-1) < 18)
                               & (a[:, :, 2] >= a[:, :, 0]-3)).sum())
        for look in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            f = self.r.rest_frame(); f.look_at = look
            self.r.render(f)
        for _ in range(2):
            f = self.r.rest_frame(); f.blink_progress = 1
            closed = np.asarray(self.r.render(f))[215:285, 430:610, :3].astype(int)
            self.assertLess(white(closed), white(open_a)*.15)
            reopened = self.r.render(self.r.rest_frame())
            np.testing.assert_array_equal(np.asarray(reopened), np.asarray(rest))

    def test_action_frames_keep_their_original_framing(self):
        path = ROOT / "assets/frames/final_stretch_0.png"
        self.r.win.set_sprite(SpriteRef(str(path), 320, 320))
        self.r.render(self.r.rest_frame(), size=(320, 320))
        self.assertFalse(self.r.win._root.property("staticAlignEnabled"))
        self.assertFalse(self.r.win._root.property("skinnedMeshVisible"))

    def test_old_neutral_parts_restore_the_original_without_seams(self):
        pkg = ROOT / "assets/rig/final"
        manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
        raw = Image.open(ROOT / "assets/ai/final_neglected_neutral.png").convert("RGBA")
        under = Image.new("RGBA", raw.size)
        for p in manifest["parts"]:
            if p["source_figure"] == "neglected_neutral" and p.get("kind") != "blink":
                under.alpha_composite(Image.open(pkg / p["file"]).convert("RGBA"), tuple(p["px_rect"][:2]))
        under.alpha_composite(Image.open(pkg / manifest["figures"]["neglected_neutral"]).convert("RGBA"))
        d = premul_difference(under, raw)
        self.assertEqual(int((d > 24).sum()), 0)
        self.assertLess(float(d.mean()), .03)
        from qa_rig_composite import evaluate
        self.assertTrue(evaluate("final", "neglected", "neutral", write_qa=False)["ok"])

    def test_legacy_parts_are_composited_before_display_reduction(self):
        # Full-resolution alpha restoration alone missed the grid and cut
        # rectangles exposed by independently downsampling core and parts.
        from PySide6.QtCore import QUrl
        path = ROOT / "assets/ai/final_neglected_neutral.png"
        self.r.win.set_sprite(SpriteRef(str(path), 320, 320))
        actual = self.r.render(self.r.rest_frame(), size=(320, 320), ground_shift=True)
        root = self.r.win._root
        parts, source = root.property("partsModel"), root.property("figASrc")
        try:
            root.setProperty("partsModel", [])
            root.setProperty("figASrc", QUrl.fromLocalFile(str(path)))
            expected = self.r.render(self.r.rest_frame(), size=(320, 320), ground_shift=True)
        finally:
            root.setProperty("partsModel", parts)
            root.setProperty("figASrc", source)
        diff = premul_difference(actual, expected)
        foreground = np.asarray(expected)[:, :, 3] > 0
        self.assertLess(float(diff[foreground].mean()), .8)
        self.assertLess(float((diff[foreground] > 16).mean()), .005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
