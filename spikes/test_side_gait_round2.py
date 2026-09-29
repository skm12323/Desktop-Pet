"""Side-walk round-2 regressions (docs/ADULT侧身步态复查-2026-09-28.md, "第二轮复查").

1 single bend  - on the actual skinned leg mesh, every walking frame: the shin (below the knee)
                 is straight (its bands agree within 8 deg) and the angle between thigh and shin
                 bands matches the solver's knee angle within 10 deg (no false joint mid-calf).
2 stop         - speeds 60/120/200 px/s x 3 stop moments x 30 / 60 / 120 Hz / jittered dt:
                 idle within 1.6 s of the stop command; no foot contact point moves backwards
                 more than 1 display px after the command; planted feet are flat (|pitch| < 1 deg)
                 once the park flatten has finished (no held tiptoe / toes-up); final stance
                 within 1 canvas px of the rest stance; solver drift <= 0.05 px.
Evidence (--evidence): spikes/_qa/side_gait_review2/{walk_cycle_legs,stop_legs}.png
  D:\\anaconda3\\python.exe -X utf8 spikes/test_side_gait_round2.py [--evidence]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
PKG = ROOT / "assets" / "rig_adult_walk_v1"
S = 256 / 1696
results: list = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def simulate(spec, speed, stop_at, dts, t_end=None):
    from pet.rig.gait import GaitSolver
    sol = GaitSolver(spec, window_scale=S)
    wx, t, i = 200.0, 0.0, 0
    t_end = t_end or stop_at + 2.5
    frames = []
    while t < t_end:
        dt = dts[i % len(dts)]
        i += 1
        t += dt
        v = speed if 0.5 <= t < stop_at else 0.0
        o = sol.update(dt, v, (round(wx), 0))
        wx = sol.window_x_float
        feet = {}
        for s in "lr":
            f = sol._feet[s]
            leg = sol.legs[s]
            off = leg.marker_sole
            c, sn = math.cos(f.pitch_now), math.sin(f.pitch_now)
            sole_x = wx + S * float(f.ankle_now[0] + c * off[0] - sn * off[1])
            feet[s] = {"sole_world": sole_x, "pitch": math.degrees(f.pitch_now),
                       "locked": f.contact.is_locked, "rel": float(f.ankle_now[0] - leg.ankle_rest[0])}
        frames.append({"t": t, "state": sol.state.value, "rot": {k: math.degrees(a) for k, a in o.bone_rotations.items()},
                       "pelvis": o.pelvis_offset, "drift": o.foot_slide_drift_px, "feet": feet,
                       "park_flat_after": sol._park_flatten_s})
    frames[-1]["knee_max"] = sol.knee_rate_max_observed
    return frames


def leg_centrelines():
    """Rest centre-line x(y) of each leg from its full layer (visible leg + hidden thigh continuation)."""
    from PIL import Image
    out = {}
    for s in "lr":
        own = np.asarray(Image.open(PKG / "layers_full" / f"leg_{s}.png"))[..., 3] > 127

        def cx(y, own=own):
            xs = np.where(own[int(round(y))])[0]
            return np.array([(xs.min() + xs.max()) / 2.0, float(y)])
        out[s] = cx
    return out


def deform_point(cache, deformed, q):
    """Skinned position of rest point q via its containing mesh triangle (barycentric)."""
    rest, tris = cache
    a, b, c = rest[tris[:, 0]], rest[tris[:, 1]], rest[tris[:, 2]]
    v0, v1, v2 = b - a, c - a, q - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
    k = int(np.where(inside)[0][0]) if inside.any() else int(np.argmin(np.abs(u) + np.abs(v)))
    d = deformed[:, :2]
    return d[tris[k, 0]] * (1 - u[k] - v[k]) + d[tris[k, 1]] * u[k] + d[tris[k, 2]] * v[k]


def single_bend(spec, frames):
    cl = leg_centrelines()
    from pet.rig.skinned_mesh_item import RigRuntime
    rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh" / "mesh_data.json"), str(PKG / "layers"))
    bones = {b["bone_name"]: b for b in spec["skeleton"]["bones"]}
    worst_shin, worst_knee = 0.0, 0.0
    for side in "lr":
        leg = next(l for l in rt.layers if l.layer_id == f"leg_{side}")
        knee_y = bones[f"lower_leg_{side}"]["joint_pos"][1] * 1696
        ankle_y = bones[f"foot_{side}"]["joint_pos"][1] * 1696
        thigh_band = (950.0, knee_y - 40)   # thigh incl. its continuation under the skirt
        shin_bands = [(knee_y + 25, knee_y + 25 + (ankle_y - 45 - knee_y - 25) * k / 3) for k in (1, 2, 3)]
        ri = rt.bone_index["root_hip"]
        tri_cache = (leg.rest[:, :2].astype(np.float64), leg.triangles.reshape(-1, 3))
        rest_bend: list = []
        for fr in frames:
            if fr["state"] not in ("walk_loop",):
                continue
            pa = np.zeros(len(rt.bones), np.float32)
            tx, ty = pa.copy(), pa.copy()
            for n, v in fr["rot"].items():
                if n in rt.bone_index:
                    pa[rt.bone_index[n]] = v
            tx[ri], ty[ri] = fr["pelvis"]
            rt.skinning_matrices(pa, tx, ty, 0.0, 0.0)
            p = rt.deform(leg, leg.rest)

            def centre(y):
                return deform_point(tri_cache, p, cl[side](y))

            def direction(a, b):
                d = centre(b) - centre(a)
                return math.degrees(math.atan2(-d[0], d[1]))
            if not rest_bend:                             # rest-pose bend of the art (subtract it)
                p_saved = p
                p = leg.rest.copy()
                rb_th = direction(*thigh_band)
                prev_r, rb_sh = knee_y + 25, []
                for _, b in shin_bands:
                    rb_sh.append(direction(prev_r, b))
                    prev_r = b
                rest_bend.append(float(np.mean(rb_sh)) - rb_th)
                p = p_saved
            th_dir = direction(*thigh_band)
            prev = knee_y + 25
            shin_dirs = []
            for _, b in shin_bands:
                shin_dirs.append(direction(prev, b))
                prev = b
            worst_shin = max(worst_shin, max(shin_dirs) - min(shin_dirs))
            mesh_knee = float(np.mean(shin_dirs)) - th_dir - rest_bend[0]
            bone_knee = fr["rot"][f"lower_leg_{side}"]           # + = clockwise = shin swings back
            worst_knee = max(worst_knee, abs(mesh_knee - bone_knee))
    return worst_shin, worst_knee


def main():
    evidence = "--evidence" in sys.argv
    spec = json.loads((PKG / "spec.json").read_text(encoding="utf-8"))
    print("== 1. single bend on the skinned leg mesh ==")
    walk = simulate(spec, 120.0, 99.0, [1 / 60], t_end=4.0)
    ws, wk = single_bend(spec, walk)
    check("shin is straight (bands agree)", ws <= 8.0, f"max spread {ws:.1f} deg")
    check("mesh knee angle = solver knee angle (bend only at the knee)", wk <= 10.0, f"max error {wk:.1f} deg")

    print("== 2. stop (3 speeds x 3 stop moments x 3 dt modes) ==")
    rng = np.random.default_rng(3)
    modes = {"30Hz": [1 / 30], "60Hz": [1 / 60], "120Hz": [1 / 120],
             "jitter": list(rng.uniform(0.5, 1.5, 997) / 30)}
    worst = {"idle": 0.0, "back": 0.0, "pitch": 0.0, "final": 0.0, "drift": 0.0, "knee": 0.0}
    for speed in (60.0, 120.0, 200.0):
        for stop_at in (2.3, 2.47, 2.61):
            for mode, dts in modes.items():
                fr = simulate(spec, speed, stop_at, dts)
                t_idle = next((f["t"] for f in fr if f["t"] > stop_at and f["state"] == "idle_side"), 99.0)
                worst["idle"] = max(worst["idle"], t_idle - stop_at)
                after = [f for f in fr if f["t"] > stop_at]
                sgn = 1.0
                for a, b in zip(after, after[1:]):
                    for s in "lr":
                        if not b["feet"][s]["locked"]:        # stepping / swinging feet (planted: drift check)
                            worst["back"] = max(worst["back"], sgn * (a["feet"][s]["sole_world"] - b["feet"][s]["sole_world"]))
                park = [f for f in after if f["state"] == "walk_park"]
                if park:
                    t0 = park[0]["t"] + park[0]["park_flat_after"] + 0.02
                    for f in after:
                        if f["t"] >= t0 and f["state"] in ("walk_park", "idle_side"):
                            for s in "lr":
                                if f["feet"][s]["locked"]:
                                    worst["pitch"] = max(worst["pitch"], abs(f["feet"][s]["pitch"]))
                worst["final"] = max(worst["final"], max(abs(fr[-1]["feet"][s]["rel"]) for s in "lr"))
                worst["drift"] = max(worst["drift"], max(f["drift"] for f in fr))
                worst["knee"] = max(worst["knee"], fr[-1]["knee_max"])
    check("idle within 1.6 s of the stop command", worst["idle"] <= 1.6, f"worst {worst['idle']:.2f} s")
    check("no backward foot motion after the stop command", worst["back"] <= 1.0,
          f"worst {worst['back']:.2f} display px")
    check("planted feet flat once parked (no held tiptoe / toes-up)", worst["pitch"] <= 1.0,
          f"worst {worst['pitch']:.2f} deg")
    check("final stance = rest stance", worst["final"] <= 1.0, f"worst {worst['final']:.2f} canvas px")
    check("knee angular speed incl. stop steps (no lift-off jump; plan <= 20 rad/s)", worst["knee"] <= 20.0,
          f"worst {worst['knee']:.1f} rad/s over 30/60/120 Hz + jitter")
    check("solver contact drift (plan G5 foot lock <= 0.5 px)", worst["drift"] <= 0.5, f"worst {worst['drift']:.3f} px")

    if evidence:
        render_evidence(spec)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n== 门禁结果：{n_ok} 通过 / {len(results) - n_ok} 失败 ==")
    return 0 if n_ok == len(results) else 1


def render_evidence(spec):
    from PIL import Image, ImageDraw
    from PySide6.QtGui import QImage
    from render_rig_rest import RigRenderer
    from pet.rig.motion import MotionFrame
    out = ROOT / "spikes" / "_qa" / "side_gait_review2"
    out.mkdir(parents=True, exist_ok=True)
    r = RigRenderer("adult", str(PKG / "adult"))
    rest = r.rest_frame("zero")
    r.render(rest)
    fr = simulate(spec, 120.0, 3.5, [1 / 30], t_end=6.0)

    def shot(f):
        ang = dict(rest.bone_angles)
        ang.update(f["rot"])
        r.win._push_frame(MotionFrame(bone_angles=ang, blink_progress=0, look_at=(0, 0)))
        r.win._skinned_item.setBonePose("root_hip", f["rot"].get("root_hip", 0.0),
                                        tx=f["pelvis"][0], ty=f["pelvis"][1])
        r._pump()
        img = r.win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()].copy()
        im = Image.fromarray(arr, "RGBA")
        bg = Image.new("RGBA", im.size, (245, 246, 250, 255))
        bg.alpha_composite(im)
        c = bg.convert("RGB").crop((240, 1000, 780, 1640))
        c = c.resize((int(c.width * 0.5), int(c.height * 0.5)), Image.LANCZOS)
        ImageDraw.Draw(c).text((3, 3), f"{f['t']:.2f}s {f['state']}", fill=(0, 0, 0))
        return c

    def sheet(sel, name):
        tiles = [shot(f) for f in sel]
        w, h = tiles[0].size
        cols = 8
        s = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)), "white")
        for k, tl in enumerate(tiles):
            s.paste(tl, ((k % cols) * w, (k // cols) * h))
        s.save(out / name)
    sheet([f for f in fr if 2.0 <= f["t"] < 2.66], "walk_cycle_legs.png")
    sheet([f for f in fr if 3.4 <= f["t"] < 5.2][::2], "stop_legs.png")
    r.close()


if __name__ == "__main__":
    raise SystemExit(main())
