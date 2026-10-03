"""FINAL side walk preview: real GaitSolver on assets/rig_final_walk_v1 (it drives the long-skirt
panels itself since F6). Renders through Qt (Python 3.12 + D3D11).

Writes spikes/_qa/final_side_f4/walk.gif (2 cycles, 1/3 scale) and walk_sheet.png (8 phases).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "spikes")]
PKG = ROOT / "assets" / "rig_final_walk_v1"
OUT = ROOT / "spikes" / "_qa" / "final_side_f4"


def main() -> None:
    from pet.rig.gait import GaitSolver
    from pet.rig.motion import MotionFrame
    from render_rig_rest import RigRenderer
    spec = json.loads((PKG / "spec.json").read_text(encoding="utf-8"))
    W, H = spec["skeleton"]["source_reference"]["image_size_px"]
    g = GaitSolver(spec, 256 / H)
    r = RigRenderer("final", str(PKG / "final"))
    frames = []
    dt = 1 / 60
    try:
        for i in range(90 + int(2 / spec["gait"]["frequency_hz"] * 60)):
            o = g.update(dt, 120.0, (round(g.window_x_float), 0))
            ang = {b: math.degrees(v) for b, v in o.bone_rotations.items()}
            tx, ty = {"root_hip": o.pelvis_offset[0]}, {"root_hip": o.pelvis_offset[1]}
            for b, (ox, oy) in o.bone_offsets.items():
                tx[b] = tx.get(b, 0) + ox
                ty[b] = ty.get(b, 0) + oy
            if i < 90 or i % 2:
                continue
            im = r.render(MotionFrame(bone_angles=ang, bone_tx=tx, bone_ty=ty, blink_progress=0.0, look_at=(0.0, 0.0)))
            bg = Image.new("RGBA", im.size, (240, 242, 246, 255))
            bg.alpha_composite(im)
            frames.append(bg.convert("RGB"))
    finally:
        r.close()
    OUT.mkdir(parents=True, exist_ok=True)
    small = [f.resize((341, 608), Image.Resampling.LANCZOS) for f in frames]
    small[0].save(OUT / "walk.gif", save_all=True, append_images=small[1:], duration=33, loop=0)
    pick = np.linspace(0, len(frames) // 2 - 1, 8).astype(int)
    s = Image.new("RGB", (512 * 8, 912))
    for k, j in enumerate(pick):
        s.paste(frames[j].resize((512, 912), Image.Resampling.LANCZOS), (512 * k, 0))
    s.save(OUT / "walk_sheet.png")
    print(f"[OK] {len(frames)} frames")


if __name__ == "__main__":
    main()
