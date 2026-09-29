"""G5: side-view ADULT walk through the real RigWindow path (offscreen), with mesh-level checks.

Per frame (fixed or jittered dt): MotionEngine.step -> _push_frame (idle bones) -> GaitSolver.update
-> window move to the solver's float accumulator (int) -> gait bones pushed EVERY frame (legs,
pelvis, arms) -> framebuffer grab. Foot lock is measured on the skinned sole vertices of the
stance foot (RigRuntime LBS of the pushed pose + view fit + window x), per contact segment,
as §5.1 rule 6 requires - not on bone markers.

  D:\\anaconda3\\python.exe -X utf8 spikes/qa_side_walk.py [--hz 30] [--jitter] [--gif]
Evidence -> spikes/_qa/adult_walk_v1/g5_side_walk/<tag>/
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
PKG = ROOT / "assets" / "rig_adult_walk_v1"
OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g5_side_walk"
WIN = 256


def script(t: float, v: float) -> float:
    """idle 0.8 s -> walk right 4.2 s -> stop -> idle."""
    return v if 0.8 <= t < 5.0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--jitter", action="store_true", help="random dt in [0.5, 1.5]/hz")
    ap.add_argument("--speed", type=float, default=120.0)
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    tag = f"{int(a.hz)}hz{'_jitter' if a.jitter else ''}_v{int(a.speed)}"
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    from render_rig_rest import RigRenderer
    from PySide6.QtGui import QImage
    from pet.rig.gait import GaitSolver, ContactType
    from pet.rig.skinned_mesh_item import RigRuntime

    r = RigRenderer("adult", str(PKG / "adult"))
    win = r.win
    win.resize(WIN, WIN)
    r._pump()
    spec_data = json.loads((PKG / "spec.json").read_text(encoding="utf-8"))
    solver = GaitSolver(spec_data, window_scale=WIN / 1696.0)
    win._gait = solver
    win.move(200, 300)
    r._pump()
    rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh" / "mesh_data.json"), str(PKG / "layers"))
    legs = {s: next(l for l in rt.layers if l.layer_id == f"leg_{s}") for s in ("l", "r")}
    fit = min(WIN / rt.img_w, WIN / rt.img_h)
    off_x = (WIN - rt.img_w * fit) * 0.5
    root = win._root
    ground_shift = float(root.property("skinnedGroundShift") or 0.0)

    def marker_rest(side: str, kind: ContactType) -> np.ndarray:
        leg = solver.legs[side]
        off = solver._marker_offset(side, kind)
        return leg.ankle_rest + off

    # nearest sole vertex to each marker (rest), per side & kind
    vsel = {}
    for s, layer in legs.items():
        for kind in (ContactType.HEEL, ContactType.FLAT_SOLE, ContactType.FOREFOOT):
            m = marker_rest(s, kind)
            d = np.linalg.norm(layer.rest[:, :2] - m, axis=1)
            vsel[(s, kind)] = int(d.argmin())

    rng = np.random.default_rng(7)
    t, frames, recs = 0.0, [], []
    seg = {"l": None, "r": None}
    drift = {"l": 0.0, "r": 0.0}
    penetration, lift_mid = 0.0, {"l": 0.0, "r": 0.0}
    xs_track = []
    item = win._skinned_item
    while t < a.seconds:
        dt = (1.0 / a.hz) * (rng.uniform(0.5, 1.5) if a.jitter else 1.0)
        t += dt
        v = script(t, a.speed)
        win.set_gait_command(v)
        frame = win._engine.step(win._motion_inputs, dt * 1000.0)
        win._push_frame(frame)
        o = solver.update(dt, v, (win.x(), win.y()), is_grounded=True, is_dragged=False)
        win.move(int(round(solver.window_x_float)), win.y())
        for bone, rad in o.bone_rotations.items():
            item.setBonePose(bone, math.degrees(rad))
        sway, dip = o.pelvis_offset
        item.setBonePose("root_hip", math.degrees(o.bone_rotations.get("root_hip", 0.0)), tx=sway, ty=dip)
        root.setProperty("bodyAngle", 0.0)
        root.setProperty("bodyY", 0.0)
        r._pump(2)

        # mesh-level contact measurement (pushed pose -> LBS -> view -> desktop)
        pa = np.zeros(len(rt.bones), np.float32)
        ptx = np.zeros_like(pa)
        pty = np.zeros_like(pa)
        for name, val in item._pose_angle.items():
            if name in rt.bone_index:
                pa[rt.bone_index[name]] = val
        for name, val in item._pose_tx.items():
            if name in rt.bone_index:
                ptx[rt.bone_index[name]] = val
        for name, val in item._pose_ty.items():
            if name in rt.bone_index:
                pty[rt.bone_index[name]] = val
        rt.skinning_matrices(pa, ptx, pty, 0.0, 0.0)
        rec = {"t": round(t, 4), "v_cmd": v, "win_x": win.x(), "win_x_float": solver.window_x_float,
               "state": solver.state.value, "solver_drift": round(o.foot_slide_drift_px, 4)}
        for s in ("l", "r"):
            p = rt.deform(legs[s], legs[s].rest).copy()
            c = solver._feet[s].contact
            lay = legs[s]
            # shoe vertices are 100 % foot-bound: every contact point moves with the foot matrix
            vi = vsel[(s, ContactType.FLAT_SOLE)]
            Mv = np.einsum("k,kij->ij", lay.weights[vi], rt.M[lay.bone_idx])
            pts = {k: Mv @ np.array([*marker_rest(s, k), 1.0])
                   for k in (ContactType.HEEL, ContactType.FLAT_SOLE, ContactType.FOREFOOT)}
            ground = fit * float(solver._ground_side.get(s, solver.ground_y))
            low = max(fit * float(q[1]) for q in pts.values())       # lowest contact point
            penetration = max(penetration, low - ground)
            rec[f"low_minus_ground_{s}"] = round(low - ground, 3)
            if c.is_locked:
                q = pts[c.contact_type]
                wx = solver.window_x_float + off_x + fit * float(q[0])     # float window (sub-px)
                rec[f"int_round_{s}"] = round(win.x() - solver.window_x_float, 3)
                key = (c.contact_type, round(c.world_anchor_x, 4))
                if seg[s] is None or seg[s][0] != key:
                    seg[s] = (key, wx)
                else:
                    drift[s] = max(drift[s], abs(wx - seg[s][1]))
                    rec[f"drift_{s}"] = round(abs(wx - seg[s][1]), 3)
            else:
                seg[s] = None
                lift_mid[s] = max(lift_mid[s], ground - low)
            rec[f"contact_{s}"] = c.contact_type.name if c.is_locked else "SWING"
        recs.append(rec)
        xs_track.append((t, solver.window_x_float, v))
        img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()].copy()
        frames.append((win.x(), arr))
    r.close()

    walk = [rr for rr in recs if rr["state"] == "walk_loop"]
    speed_meas = None
    if len(walk) > 10:
        speed_meas = (walk[-1]["win_x_float"] - walk[0]["win_x_float"]) / (walk[-1]["t"] - walk[0]["t"])
    stop_t = next((rr["t"] for rr in recs if rr["t"] >= 5.0 and rr["state"] in ("idle_side", "idle_front")), None)
    metrics = {
        "dt_mode": tag, "frames": len(recs),
        "foot_lock_drift_disp_px": {k: round(v_, 3) for k, v_ in drift.items()},
        "foot_lock_pass": bool(max(drift.values()) <= 0.5),
        "penetration_disp_px": round(penetration, 3), "penetration_pass": bool(penetration <= 0.5),
        "swing_clearance_disp_px": {k: round(v_, 2) for k, v_ in lift_mid.items()},
        "clearance_pass": bool(min(lift_mid.values()) >= 1.5),
        "speed_measured": speed_meas, "speed_cmd": a.speed,
        "speed_expected_from_gait": solver.speed_target,
        "stop_settled_at_s": stop_t,
        "knee_rate_max_rad_s": round(solver.knee_rate_max_observed, 2),
    }
    (out / "metrics.json").write_bytes(json.dumps(metrics, indent=2).encode("utf-8"))
    (out / "trace.json").write_bytes(json.dumps(recs).encode("utf-8"))

    # evidence: desktop strip GIF (window composited at its x) + filmstrip
    x0 = min(x for x, _ in frames) - 20
    x1 = max(x for x, _ in frames) + WIN + 20
    gif = []
    for (wx, arr) in frames:
        bg = Image.new("RGBA", (x1 - x0, WIN + 10), (245, 246, 250, 255))
        dr = ImageDraw.Draw(bg)
        for gx in range(0, x1 - x0, 40):
            dr.line([(gx, WIN - 2), (gx, WIN + 10)], fill=(200, 200, 210, 255))
        bg.alpha_composite(Image.fromarray(arr, "RGBA"), (wx - x0, 0))
        gif.append(bg.convert("RGB"))
    step = max(1, int(a.hz / 30))
    if a.gif:
        gif[0].save(out / "walk_1x.gif", save_all=True, append_images=gif[1::step], duration=int(1000 / 30), loop=0)
        gif[0].save(out / "walk_quarter.gif", save_all=True, append_images=gif[1::step], duration=int(4000 / 30), loop=0)
    idx = np.linspace(int(1.5 * a.hz), int(2.6 * a.hz), 8).astype(int)
    crops = []
    for i in idx:
        wx, arr = frames[min(i, len(frames) - 1)]
        im = Image.new("RGBA", (WIN, WIN), (245, 246, 250, 255))
        im.alpha_composite(Image.fromarray(arr, "RGBA"))
        crops.append(im.convert("RGB").crop((50, 0, 206, WIN)))
    strip = Image.new("RGB", (156 * len(crops), WIN), "white")
    for k, c in enumerate(crops):
        strip.paste(c, (k * 156, 0))
    strip.resize((strip.width * 2, WIN * 2), Image.Resampling.NEAREST).save(out / "cycle_strip_2x.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
