"""G0 diagnosis (QA only, not product code): production ADULT idle + walk.

Renders the real Qt skinned path (assets/rig/adult -> assets/rig_adult) the way
app.py drives it at behavior.walk_speed=120 px/s (walk_hz = 0.9 + v/400 = 1.2),
composites the window onto a desktop strip that moves at that speed, and
measures shoe world-x motion to check whether any foot is ever planted.
Evidence for docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md.

Run: D:\\anaconda3\\python.exe -X utf8 spikes/qa_adult_walk_diagnosis.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("QT_QUICK_BACKEND", "rhi")
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from pet.asset_provider import SpriteRef
from pet.rig.motion import MotionEngine, MotionInputs
from pet.rig.presenter import build_rig_window
from pet.rig.spec import load_rig_spec
from pet.window import WindowBase

OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g0_diagnosis"
WIN = 512                      # render at 2x of the 256 default for legibility
SCALE_TO_256 = 256 / WIN
SPEED_256 = 120.0              # config.json behavior.walk_speed
HZ = 0.9 + SPEED_256 / 400.0   # app.py mapping
TILT = max(-9.0, min(9.0, SPEED_256 / 140.0))
FPS = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    spec = load_rig_spec(str(ROOT / "assets/rig/adult"), "adult")
    win = build_rig_window(WindowBase, SpriteRef(spec.figures["healthy_neutral"], WIN, WIN), "adult")
    win._motion_timer.stop()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(WIN, WIN)
    win.show()
    for _ in range(4):
        app.processEvents()
    assert win._root.property("skinnedMeshVisible"), "skinned mesh not visible"
    rt = win._skinned_item._rt
    names = [b.name for b in rt.bones]
    layers = {l.layer_id: l for l in rt.layers}

    def grab(frame):
        win._push_frame(frame)
        win._root.setProperty("bodyAngle", 0.0)      # bypass QML 90 ms Behavior
        app.processEvents()
        img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
        pil = Image.fromarray(arr[:, :img.width()].copy(), "RGBA").resize((WIN, WIN), Image.Resampling.LANCZOS)
        # re-apply the body rotation exactly like QML: origin (w/2, h), clockwise-positive
        if abs(frame.body_angle) > 1e-4:
            big = Image.new("RGBA", (WIN, WIN * 2), (0, 0, 0, 0))
            big.paste(pil, (0, 0))
            big = big.rotate(-frame.body_angle, resample=Image.Resampling.BICUBIC, center=(WIN / 2, WIN))
            pil = big.crop((0, 0, WIN, WIN))
        return pil

    fit = min(WIN / rt.img_w, WIN / rt.img_h)
    offx = (WIN - rt.img_w * fit) / 2
    ground_shift = WIN - ((WIN - rt.img_h * fit) / 2 + spec.ground_anchor_y_px * fit)

    def shoes(frame):
        ang = np.array([frame.bone_angles.get(n, 0.0) for n in names], np.float32)
        tx = np.array([frame.bone_tx.get(n, 0.0) for n in names], np.float32)
        ty = np.array([frame.bone_ty.get(n, 0.0) for n in names], np.float32)
        rt.skinning_matrices(ang, tx, ty, 0.0, 0.0)
        res = {}
        for side in ("l", "r"):
            layer = layers["leg_" + side]
            xy = rt.deform(layer, rt.effective_rest(layer, 0)).copy()
            m = layer.rest[:, 1] >= 1490
            res[side] = (offx + float(xy[m, 0].mean()) * fit,
                         float(xy[m, 1].max()) * fit + ground_shift + frame.body_y)
        return res

    # ---------------- idle (3 s) ----------------
    eng = MotionEngine(spec)
    idle_in = MotionInputs(walking=False, grounded=True, facing=1, source_facing=1, wind_gain=1.0)
    for _ in range(30):
        eng.step(idle_in, 1000 / FPS)
    idle = []
    for i in range(3 * FPS):
        idle.append(grab(eng.step(idle_in, 1000 / FPS)))
    bg = (236, 240, 246, 255)
    frames = []
    for im in idle:
        c = Image.new("RGBA", im.size, bg)
        c.alpha_composite(im)
        frames.append(c.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS))
    frames[0].save(OUT / "idle_256.gif", save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
    picks = [0, 22, 45, 67, 89]
    idle_sheet = Image.new("RGB", (256 * len(picks), 256), bg[:3])
    for j, idx in enumerate(picks):
        idle_sheet.paste(frames[idx], (j * 256, 0))
    idle_sheet.save(OUT / "idle_sheet_256.png")

    # ---------------- walk (window moves at 120 px/s on a 256-scale desktop) ----------------
    eng = MotionEngine(spec)
    walk_in = MotionInputs(walking=True, walk_hz=HZ, tilt_deg=TILT, grounded=True,
                           facing=1, source_facing=1, wind_gain=1.0)
    for _ in range(FPS):
        eng.step(walk_in, 1000 / FPS)
    n = int(2.5 * FPS)
    strip_w = 256 + int(SPEED_256 * n / FPS) + 40
    walk_frames, recs = [], []
    for i in range(n):
        fr = eng.step(walk_in, 1000 / FPS)
        im = grab(fr)
        win_x = 20 + SPEED_256 * i / FPS                     # 256-scale world px
        s = shoes(fr)
        recs.append({"t": i / FPS, "phase": eng.gait_phase, "win_x": win_x,
                     **{f"{k}_world_x": win_x + v[0] * SCALE_TO_256 for k, v in s.items()},
                     **{f"{k}_sole_y": v[1] * SCALE_TO_256 for k, v in s.items()}})
        canvas = Image.new("RGBA", (strip_w, 300), bg)
        d = ImageDraw.Draw(canvas)
        floor = 20 + 256
        d.line([(0, floor), (strip_w, floor)], fill=(120, 130, 150, 255), width=1)
        for x in range(0, strip_w, 16):                      # floor ticks to read sliding
            d.line([(x, floor), (x, floor + 6)], fill=(150, 160, 180, 255), width=1)
        small = im.resize((256, 256), Image.Resampling.LANCZOS)
        canvas.alpha_composite(small, (int(round(win_x)), 20))
        for k, col in (("l", (220, 60, 60, 255)), ("r", (40, 120, 220, 255))):
            wx = recs[-1][f"{k}_world_x"]
            d.ellipse([wx - 3, floor + 10, wx + 3, floor + 16], fill=col)
        walk_frames.append(canvas.convert("RGB"))
    walk_frames[0].save(OUT / "walk_desktop_strip.gif", save_all=True, append_images=walk_frames[1:],
                        duration=int(1000 / FPS), loop=0)
    # keyframe sheet at display size (256) over one gait cycle
    per = int(round(FPS / HZ))
    picks = [int(round(k * per / 8)) for k in range(8)]
    sheet = Image.new("RGB", (8 * 180, 300), bg[:3])
    for j, idx in enumerate(picks):
        fr = walk_frames[idx]
        wx = int(round(recs[idx]["win_x"]))
        sheet.paste(fr.crop((wx + 38, 0, wx + 218, 300)), (j * 180, 0))
        ImageDraw.Draw(sheet).text((j * 180 + 6, 4), f"phase {recs[idx]['phase']:.2f}", fill=(30, 40, 55))
    sheet.save(OUT / "walk_cycle_256.png")

    # ---------------- sliding metrics ----------------
    out = {"speed_px_s_at_256": SPEED_256, "walk_hz": HZ, "fps": FPS}
    for k in ("l", "r"):
        xs = np.array([r[f"{k}_world_x"] for r in recs])
        v = np.diff(xs) * FPS
        out[f"foot_{k}"] = {
            "world_speed_min_px_s": float(v.min()), "world_speed_max_px_s": float(v.max()),
            "frac_frames_planted(|v|<12px/s)": float(np.mean(np.abs(v) < 12.0)),
            "sole_y_range_px": [float(min(r[f"{k}_sole_y"] for r in recs)), float(max(r[f"{k}_sole_y"] for r in recs))],
        }
    gaps = [r["r_world_x"] - r["l_world_x"] for r in recs]
    out["shoe_gap_r_minus_l_px"] = [float(min(gaps)), float(max(gaps))]
    (OUT / "walk_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    win.close()


if __name__ == "__main__":
    main()
