"""Measure/render the leg attachment relative to the pelvis over complete walk cycles."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="after")
    args = ap.parse_args()
    from render_rig_rest import RigRenderer
    from pet.rig.gait import GaitSolver
    from pet.rig.motion import MotionFrame
    pkg = ROOT / "assets/rig_adult_walk_v1"
    out = ROOT / "spikes/_qa/side_leg_alignment"
    out.mkdir(exist_ok=True)
    spec = json.loads((pkg / "spec.json").read_text(encoding="utf-8"))
    r = RigRenderer("adult", str(pkg / "adult"))
    g = GaitSolver(spec, 256 / 1696)
    rt = r.rt
    legs = [l for l in rt.layers if l.layer_id in ("leg_l", "leg_r")]
    rows, frames = [], []
    for i in range(750):
        o = g.update(1 / 120, 120, (round(g.window_x_float), 0))
        if i < 450:
            continue
        angles = {name: float(np.degrees(rad)) for name, rad in o.bone_rotations.items()}
        a = np.array([angles.get(b.name, 0) for b in rt.bones], np.float32)
        tx, ty = np.zeros_like(a), np.zeros_like(a)
        tx[rt.bone_index["root_hip"]], ty[rt.bone_index["root_hip"]] = o.pelvis_offset
        rt.skinning_matrices(a, tx, ty, 0, 0)
        row = dict(phase=g.phase)
        for layer in legs:
            xy = rt.deform(layer, layer.rest).copy()
            attached = layer.rest @ rt.M[rt.bone_index["root_hip"]].T
            mask = (layer.rest[:, 1] >= 1100) & (layer.rest[:, 1] <= 1160)
            row[layer.layer_id] = float(np.mean(xy[mask, 0] - attached[mask, 0])) * g.scale
        rows.append(row)
        if i % 5 == 0:
            f = MotionFrame(bone_angles=angles, bone_tx={"root_hip": o.pelvis_offset[0]},
                            bone_ty={"root_hip": o.pelvis_offset[1]})
            im = r.render(f, (512, 512), ground_shift=True)
            bg = Image.new("RGBA", (512, 512), (245, 246, 250, 255))
            bg.alpha_composite(im)
            frames.append(bg.convert("RGB"))
    r.close()
    metrics = {sd: {"mean_root_shift_px": float(np.mean([x[sd] for x in rows])),
                    "max_abs_root_shift_px": float(max(abs(x[sd]) for x in rows))}
               for sd in ("leg_l", "leg_r")}
    (out / f"{args.tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    frames[0].save(out / f"{args.tag}.gif", save_all=True, append_images=frames[1:], duration=42, loop=0)
    strip = Image.new("RGB", (240 * 6, 212), "white")
    for n, i in enumerate(np.linspace(0, 15, 6).astype(int)):
        strip.paste(frames[i].crop((145, 300, 385, 512)), (n * 240, 0))
    strip.save(out / f"{args.tag}.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
