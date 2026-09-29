"""Diagnostic capture: real side presenter walk at a large window, per-frame world joints + frames."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 640
SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
OUT = ROOT / ".scratch" / "diag" / f"walk_{SIZE}_{int(SPEED)}"


def main():
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
    win.resize(SIZE, SIZE)
    r._pump()
    assert win.enable_side_locomotion(str(ROOT / "assets/rig_adult_walk_v1"))
    win.move(200, 300)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    hz, t, k = 60.0, 0.0, 0
    recs = []
    names = ["root_hip", "spine", "chest", "neck", "head", "upper_leg_l", "lower_leg_l", "foot_l",
             "upper_leg_r", "lower_leg_r", "foot_r", "upper_arm_l", "forearm_l", "hand_l",
             "upper_arm_r", "forearm_r", "hand_r", "skirt_hem_l", "skirt_hem_r"]
    while t < 6.0:
        t += 1 / hz
        clock.t += 1 / hz
        speed = SPEED if .5 <= t < 4.5 else 0.0
        win.set_locomotion_intent(speed)
        win._motion_inputs.walking = bool(speed)
        win._motion_inputs.walk_hz = 1.2
        win._motion_inputs.grounded = True
        win._motion_tick()
        r._pump(2)
        lf = win._loco_last
        rec = dict(t=t, state=lf.state.value, mode=lf.mode, x=win.x())
        if lf.mode == "side" and lf.gait:
            solver = win._loco._solver
            item = win._side_item
            rt = item._rt
            arrays = [np.array([p.get(b.name, 0) for b in rt.bones], np.float32)
                      for p in (item._pose_angle, item._pose_tx, item._pose_ty)]
            rt.skinning_matrices(*arrays, 0, 0)
            rest = {b.name: np.array(b.joint_px if hasattr(b, "joint_px") else b.pivot, float) for b in rt.bones} \
                if hasattr(rt.bones[0], "joint_px") or hasattr(rt.bones[0], "pivot") else None
            world = {}
            for b in rt.bones:
                if b.name in names:
                    i = rt.bone_index[b.name]
                    p = rest[b.name] if rest else None
                    if p is not None:
                        world[b.name] = (rt.M[i] @ np.r_[p, 1])[:2].tolist()
            rec.update(pose={"a": {k: float(v) for k, v in item._pose_angle.items()},
                             "tx": {k: float(v) for k, v in item._pose_tx.items()},
                             "ty": {k: float(v) for k, v in item._pose_ty.items()}})
            rec.update(gait=solver.state.value, phase=solver.phase, stride_hz=solver.stride_hz,
                       angles={n: float(item._pose_angle.get(n, 0.0)) for n in names},
                       world=world, pelvis=list(lf.gait.pelvis_offset),
                       pitch={s: math.degrees(solver._feet[s].pitch_now) for s in ("l", "r")})
        recs.append(rec)
        k += 1
        if k % 2 == 0:
            img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
            arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
            Image.fromarray(arr[:, :img.width()].copy()).save(OUT / "frames" / f"{k:04d}.png")
    r.close()
    (OUT / "trace.json").write_text(json.dumps(recs), encoding="utf-8")
    print("frames", k // 2, "->", OUT)


if __name__ == "__main__":
    main()
