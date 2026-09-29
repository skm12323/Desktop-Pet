"""Review actual side presenter poses, not only contact markers / clip seams.

Run with --out baseline or --out fixed; evidence includes full-session video,
walk/stop strips, signed knee bend from actual FK and solver contact drift.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fixed")
    ap.add_argument("--speed", type=float, default=120)
    ap.add_argument("--hz", type=float, default=60)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--jitter", action="store_true")
    args = ap.parse_args()
    from render_rig_rest import RigRenderer
    from pet.rig import presenter
    from PySide6.QtGui import QImage

    class Clock:
        t = 1000.0
        def perf_counter(self):
            return self.t

    clock = Clock()
    presenter.time = clock
    r = RigRenderer()
    win = r.win
    win.resize(args.size, args.size)
    r._pump()
    assert win.enable_side_locomotion(str(ROOT / "assets/rig_adult_walk_v1"))
    win.move(200, 300)
    out = ROOT / "spikes/_qa/side_gait_review" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_bytes((ROOT / "assets/rig_adult_walk_v1/spec.json").read_bytes())
    scale = args.size / 1696
    frames, records = [], []
    rng = np.random.default_rng(23)
    t = 0.0
    while t < 12:
        dt = (rng.uniform(.5, 1.5) if args.jitter else 1.0) / args.hz
        t += dt
        clock.t += dt
        speed = args.speed if .5 <= t < 5.0 else 0.0
        win.set_locomotion_intent(speed)
        win._motion_inputs.walking = bool(speed)
        win._motion_inputs.walk_hz = 1.2
        win._motion_inputs.grounded = True
        win._motion_tick()
        r._pump(2)
        lf = win._loco_last
        rec = dict(t=t, state=lf.state.value, x=win.x(), mode=lf.mode)
        if lf.mode == "side" and lf.gait:
            solver = win._loco._solver
            item = win._side_item
            rt = item._rt
            arrays = [np.array([p.get(b.name, 0) for b in rt.bones], np.float32)
                      for p in (item._pose_angle, item._pose_tx, item._pose_ty)]
            rt.skinning_matrices(*arrays, 0, 0)
            rec.update(gait=solver.state.value, phase=solver.phase,
                       dip=lf.gait.pelvis_offset[1] * scale, drift=lf.gait.foot_slide_drift_px)
            for side in ("l", "r"):
                leg = solver.legs[side]
                pts = []
                for name, rest in ((leg.hip_bone, leg.hip_rest), (leg.knee_bone, leg.knee_rest),
                                   (leg.foot_bone, leg.ankle_rest)):
                    pts.append((rt.M[rt.bone_index[name]] @ np.r_[rest, 1])[:2])
                hip, knee, ankle = pts
                chord_x = hip[0] + (ankle[0] - hip[0]) * (knee[1] - hip[1]) / (ankle[1] - hip[1])
                rec[f"knee_forward_{side}"] = float(knee[0] - chord_x) * scale
                rec[f"ankle_{side}"] = (ankle * scale).tolist()
                rec[f"knee_angle_{side}"] = item._pose_angle[leg.knee_bone]
                rec[f"foot_pitch_{side}"] = math.degrees(solver._feet[side].pitch_now)
        records.append(rec)
        if len(records) % max(1, round(args.hz / 30)) == 0:
            img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
            arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
            im = Image.fromarray(arr[:, :img.width()].copy()).resize((256, 256))
            bg = Image.new("RGBA", (256, 278), (245, 246, 250, 255))
            bg.alpha_composite(im, (0, 22))
            ImageDraw.Draw(bg).text((4, 3), f"{t:.2f} {rec.get('gait', lf.state.value)}", fill="black")
            frames.append((t, bg.convert("RGB")))
    r.close()
    for name, times in (("walk", np.linspace(3.0, 3.83, 8)),
                        ("stop", np.linspace(5.0, 6.4, 8)),
                        ("settle", np.linspace(9.5, 10.3, 8))):
        strip = Image.new("RGB", (8 * 170, 278), "white")
        for i, tt in enumerate(times):
            _, im = min(frames, key=lambda x: abs(x[0] - tt))
            strip.paste(im.crop((43, 0, 213, 278)), (170 * i, 0))
        strip.save(out / f"{name}.png")
    frames[0][1].save(out / "session.gif", save_all=True, append_images=[im for _, im in frames[1:]],
                      duration=33, loop=0)
    active = [x for x in records if x.get("gait") == "walk_loop"]
    stats = {"speed": args.speed, "hz": args.hz, "size": args.size,
             "min_knee_forward_px": {s: min(x[f"knee_forward_{s}"] for x in active) for s in ("l", "r")},
             "max_knee_forward_px": {s: max(x[f"knee_forward_{s}"] for x in active) for s in ("l", "r")},
             "pelvis_dip_range_px": [min(x["dip"] for x in active), max(x["dip"] for x in active)],
             "max_solver_drift": max(x.get("drift", 0) for x in records)}
    (out / "trace.json").write_text(json.dumps(records), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
