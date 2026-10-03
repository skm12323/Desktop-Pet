"""Rendered regression: front ADULT blink must close the sclera and restore irises."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
OUT = ROOT / "spikes/_qa/adult_blink_return_2026-09-30"


def eye_counts(im):
    a = np.asarray(im)[305:380, 365:558, :3].astype(int)
    white = (a.min(-1) > 235) & (np.ptp(a, axis=-1) < 15)
    blue = (a[..., 2] > a[..., 0]+25) & (a[..., 2] > a[..., 1]+5) & (a[..., 2] > 90)
    return int(white.sum()), int(blue.sum())


class BlinkReturn(unittest.TestCase):
    def test_generator_keeps_every_eye_boundary(self):
        from mesh_generator import generate_layer_mesh
        spec = json.loads((ROOT / "assets/rig_adult/spec.json").read_text(encoding="utf-8"))
        layer = next(x for x in spec["layers"] if x["id"] == "head_base")
        mesh = generate_layer_mesh(layer, spec["skeleton"], str(ROOT / "assets/rig_adult/layers/head_base.png"), (960,1696))
        xs = {v[0] for v in mesh["vertices"]}
        ys = {v[1] for v in mesh["vertices"]}
        for cx, top, bottom, radius in layer["blink_zones"]:
            for x in (cx-radius, cx, cx+radius):
                self.assertIn(x, xs)
            for y in (top, bottom):
                self.assertIn(y, ys)

    def test_full_blink_and_reopen(self):
        from render_rig_rest import RigRenderer
        r = RigRenderer()
        try:
            open_im = r.render(r.rest_frame("relaxed"))
            open_white, open_blue = eye_counts(open_im)
            self.assertGreater(open_blue, 500)
            for cycle in range(3):
                for b in (.25, .75, 1, 1, .75, .25, 0):
                    f = r.rest_frame("relaxed")
                    f.blink_progress = b
                    im = r.render(f)
                    white, blue = eye_counts(im)
                    if b == 1:
                        self.assertLess(white, open_white*.15, "closed eye still shows a white sclera")
                    if b == 0:
                        self.assertGreaterEqual(blue, open_blue*.99, "iris did not return")
                        np.testing.assert_array_equal(np.asarray(im)[305:380, 365:558],
                                                      np.asarray(open_im)[305:380, 365:558])
        finally:
            r.close()

    def test_first_and_second_blink_after_turn_back(self):
        from render_rig_rest import RigRenderer
        from pet.rig.motion import MotionInputs
        from PySide6.QtGui import QImage
        r = RigRenderer()
        w = r.win
        w.resize(960, 1696)
        r._pump()
        self.assertTrue(w.enable_side_locomotion(str(ROOT / "assets/rig_adult_walk_v1")))
        w._loco.scale = 256/1696
        w._root.setProperty("skinnedGroundYPx", 0.0)
        w.move(200, 300)
        inputs = MotionInputs(grounded=True, facing=1, source_facing=1, cursor_pos=None, walk_hz=1.2)
        w._motion_inputs = inputs
        OUT.mkdir(parents=True, exist_ok=True)
        returned = False
        prev = 0.0
        cycles, completed = 0, 0
        baseline_white = baseline_blue = None
        records, frames = [], []
        try:
            for i in range(26*60):
                t = (i+1)/60
                vx = 120 if 2 <= t < 8 else 0
                inputs.walking = bool(vx)
                w.set_locomotion_intent(vx)
                frame = w._engine.step(inputs, 1000/60)
                lf = w._loco.update(1/60, vx, w.x())
                if lf.mode == "front":
                    w._push_frame(w._settle_frame(frame, lf.settle))
                w._apply_loco(lf, frame)
                w._root.setProperty("bodyAngle", 0.0)
                r._pump(2)
                if not returned and t > 8 and lf.state.value == "front":
                    returned = True
                    # Schedule the first complete blink just after the 300-ms
                    # return fade; the second uses the shipped deterministic RNG.
                    w._engine._blink_open_until = w._engine.t_ms
                    w._engine._blink_next_ms = w._engine.t_ms + 350
                if not returned:
                    continue
                b = w._skinned_item._blink
                q = w._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
                a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine()//4, 4)
                im = Image.fromarray(a[:, :q.width()].copy())
                white, blue = eye_counts(im)
                if baseline_white is None and b == 0:
                    baseline_white, baseline_blue = white, blue
                if prev == 0 and b > 0:
                    cycles += 1
                if b >= .999:
                    self.assertLess(white, baseline_white*.15, "white-eye flash after turn back")
                if prev > 0 and b == 0:
                    self.assertGreater(blue, baseline_blue*.9, "post-turn iris failed to restore")
                    completed += 1
                if b > 0 or prev > 0:
                    crop = im.crop((365,285,558,385))
                    bg = Image.new("RGBA", crop.size, (239,242,247,255))
                    bg.alpha_composite(crop)
                    frames.append((t, b, bg.convert("RGB")))
                records.append({"t": round(t,4), "blink": b, "white": white, "blue": blue})
                prev = b
                if completed >= 2:
                    break
            self.assertTrue(returned)
            self.assertEqual(completed, 2)
            self.assertEqual(cycles, 2)
            cols, cell_w, cell_h = 8, 260, 165
            sheet = Image.new("RGB", (cols*cell_w, ((len(frames)+cols-1)//cols)*cell_h), (239,242,247))
            draw = ImageDraw.Draw(sheet)
            for j,(t,b,crop) in enumerate(frames):
                x,y = j%cols*cell_w,j//cols*cell_h
                sheet.paste(crop.resize((260,135)),(x,y+28))
                draw.text((x+7,y+8),f"{t:.3f}s blink {b:.2f}",fill="black")
            sheet.save(OUT / "two_post_return_blinks_after.png")
            animation = [im.resize((579,300),Image.Resampling.LANCZOS) for _,_,im in frames]
            animation[0].save(OUT / "post_return_blink_after.gif", save_all=True,
                              append_images=animation[1:], duration=80, loop=0)
            (OUT / "post_return_blink_results.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
