"""Capture current ADULT visual review evidence without changing production assets.

Run: python -X utf8 spikes/qa_adult_visual_report.py
Uses the real Qt/D3D11 presenter at 256 px; high-resolution replays keep the
same 256-px gait poses (the solver is never re-run at a larger window size).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
OUT = ROOT / "spikes/_qa/adult_visual_review_2026-09-30"
SCALE = 256 / 1696


def bg(im, dark=False):
    canvas = Image.new("RGBA", im.size, (32, 36, 45, 255) if dark else (245, 246, 250, 255))
    canvas.alpha_composite(im)
    return canvas.convert("RGB")


def sheet(items, path, width=256, columns=4):
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (width * columns, (width + 28) * rows), (245, 246, 250))
    for i, (label, im) in enumerate(items):
        x, y = (i % columns) * width, (i // columns) * (width + 28)
        canvas.paste(bg(im).resize((width, width)), (x, y + 28))
        ImageDraw.Draw(canvas).text((x + 6, y + 8), label, fill="black")
    canvas.save(path)


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
    w = r.win
    w.move(200, 300)
    assert w.enable_side_locomotion(str(ROOT / "assets/rig_adult_walk_v1"))
    w._last_tick_s = clock.t
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)

    def grab():
        q = w._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        assert not q.isNull()
        a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine() // 4, 4)
        return Image.fromarray(a[:, :q.width()].copy()).resize((w.width(), w.height()))

    props = ("locoMode", "facing", "bodyAngle", "bodyY", "bodyScaleX", "bodyScaleY",
             "clipCanvasX", "clipCanvasY", "clipCanvasW", "clipCanvasH", "clipFrameSrc",
             "clipOpacity", "clipUnder", "clipUnderOpacity")
    frames, records, snapshots = [], [], []
    transitions = []
    last_state = None
    deform = {}
    rt = w._side_item._rt
    pre = {}
    for layer in rt.layers:
        v = layer.rest[:, :2].astype(float)
        tri = layer.triangles.reshape(-1, 3)
        edge = np.stack((v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), -1)
        valid = np.abs(np.linalg.det(edge)) > 1e-6
        pre[layer.layer_id] = tri[valid], np.linalg.inv(edge[valid])
        deform[layer.layer_id] = {"sigma_max": 1.0, "sigma_min": 1.0, "fold_frames": 0}

    # A long steady walk ensures strips show WALK_LOOP rather than its entry envelope.
    for i in range(34 * 60):
        t = (i + 1) / 60
        vx = 120.0 if 2 <= t < 9 or 17 <= t < 24 else (-120.0 if 24 <= t < 31 else 0.0)
        w.set_locomotion_intent(vx)
        w._motion_inputs.walking = bool(vx)
        w._motion_inputs.grounded = True
        clock.t += 1 / 60
        w._motion_tick()
        r._pump(2)
        lf = w._loco_last
        rec = {"t": t, "state": lf.state.value, "mode": lf.mode, "x": w.x(), "facing": lf.facing}
        key = (lf.state.value, w._loco._solver.state.value if lf.gait else "")
        if key != last_state:
            transitions.append({**rec, "gait": key[1]})
            last_state = key
        if lf.mode == "side" and lf.gait:
            solver, item = w._loco._solver, w._side_item
            rec.update(gait=solver.state.value, phase=solver.phase, stride_hz=solver.stride_hz,
                       drift=lf.gait.foot_slide_drift_px, pelvis=list(lf.gait.pelvis_offset),
                       angles=dict(item._pose_angle))
            for side in "lr":
                f = solver._feet[side]
                rec[f"foot_{side}"] = {"locked": f.contact.is_locked, "pitch": math.degrees(f.pitch_now)}
            if i % 2 == 1:
                arrays = [np.array([p.get(b.name, 0) for b in rt.bones], np.float32)
                          for p in (item._pose_angle, item._pose_tx, item._pose_ty)]
                rt.skinning_matrices(*arrays, 0, 0)
                for layer in rt.layers:
                    tri, inv = pre[layer.layer_id]
                    if not len(tri):
                        continue
                    xy = rt.deform(layer, rt.effective_rest(layer, item._blink))[:, :2].astype(float)
                    f = np.stack((xy[tri[:, 1]] - xy[tri[:, 0]], xy[tri[:, 2]] - xy[tri[:, 0]]), -1) @ inv
                    det = np.linalg.det(f)
                    sig = np.linalg.svd(f, compute_uv=False)
                    d = deform[layer.layer_id]
                    d["fold_frames"] += int(np.any(det < 0))
                    ok = det > 0
                    if np.any(ok):
                        d["sigma_max"] = max(d["sigma_max"], float(sig[ok, 0].max()))
                        d["sigma_min"] = min(d["sigma_min"], float(sig[ok, 1].min()))
        records.append(rec)
        if i % 2 == 1:
            im = grab()
            idx = len(frames)
            im.save(OUT / "frames" / f"{idx:04d}.png")
            frames.append((t, im))
            snap = {"props": {p: w._root.property(p) for p in props}, "items": []}
            for item in (w._skinned_item, w._side_item):
                snap["items"].append({"a": dict(item._pose_angle), "tx": dict(item._pose_tx),
                                     "ty": dict(item._pose_ty), "blink": item._blink,
                                     "look": (item._look_x, item._look_y)})
            snapshots.append(snap)

    def nearest(tt):
        return min(range(len(frames)), key=lambda k: abs(frames[k][0] - tt))

    scenes = [("front_idle", [0.5, 1, 1.5, 1.9]),
              ("turn_out", [2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6]),
              ("walk_cycle", np.linspace(6, 6 + 1 / 1.2, 8, endpoint=False)),
              ("stop", np.linspace(9, 10.4, 8)),
              ("turn_in", np.linspace(14.8, 16.2, 8)),
              ("reverse", np.linspace(24, 28.2, 12)),
              ("walk_left", np.linspace(29, 29 + 1 / 1.2, 8, endpoint=False))]
    for name, times in scenes:
        indices = [nearest(float(tt)) for tt in times]
        sheet([(f"{frames[k][0]:.2f}s", frames[k][1]) for k in indices], OUT / f"{name}_256.png")
    animated = [bg(im) for _, im in frames[:17 * 30]]
    animated[0].save(OUT / "session_256.gif", save_all=True, append_images=animated[1:], duration=33, loop=0)
    indices = [k for k, (tt, _) in enumerate(frames) if 5.8 <= tt < 7.5]
    animated = [bg(frames[k][1]) for k in indices]
    animated[0].save(OUT / "walk_loop_256.gif", save_all=True, append_images=animated[1:], duration=33, loop=0)

    # Replay snapshots at 3x render resolution, preserving all bone poses captured at 256.
    w.resize(768, 768)
    r._pump()
    chosen = [0.5, 1.5, 3.4, *np.linspace(6, 6 + 1 / 1.2, 8, endpoint=False), 10.6, 15.4]
    hi = []
    for j, tt in enumerate(chosen):
        k = nearest(float(tt))
        snap = snapshots[k]
        for p, value in snap["props"].items():
            w._root.setProperty(p, value)
        for item, pose in zip((w._skinned_item, w._side_item), snap["items"]):
            for b in item._rt.bone_index:
                item.setBonePose(b, pose["a"].get(b, 0), pose["tx"].get(b, 0), pose["ty"].get(b, 0))
            item.setBlink(pose["blink"])
            item.setLookAt(*pose["look"])
        r._pump()
        im = grab()
        im.save(OUT / f"high_{j:02d}_{frames[k][0]:.2f}.png")
        hi.append((frames[k][0], im))
    for part, box in (("legs", (290, 545, 560, 768)), ("arms", (230, 275, 590, 565)),
                      ("tail_skirt", (150, 465, 610, 680)), ("head", (280, 50, 540, 300))):
        picks = hi[3:11]
        cw, ch = box[2] - box[0], box[3] - box[1]
        canvas = Image.new("RGB", (cw * 4, (ch + 26) * 2), (245, 246, 250))
        for j, (tt, im) in enumerate(picks):
            x, y = (j % 4) * cw, (j // 4) * (ch + 26)
            canvas.paste(bg(im).crop(box), (x, y + 26))
            ImageDraw.Draw(canvas).text((x + 5, y + 6), f"{tt:.2f}s; 3x render", fill="black")
        canvas.save(OUT / f"detail_{part}.png")
    sheet([(f"{tt:.2f}s light", im) for tt, im in (hi[0], hi[4])], OUT / "front_side_light.png", 512, 2)
    contrast = Image.new("RGB", (768 * 2, 768 * 2))
    for j, (_, im) in enumerate((hi[0], hi[4])):
        contrast.paste(bg(im), (j * 768, 0))
        contrast.paste(bg(im, True), (j * 768, 768))
    contrast.save(OUT / "light_dark_3x.png")

    # Capture actual sprite backends for mood/branch switches.
    w.resize(256, 256)
    w.disable_side_locomotion()
    moods, modes = [], []
    from pet.asset_provider import SpriteRef
    for name in ("healthy_neutral", "healthy_happy", "healthy_sad", "healthy_hungry", "healthy_sleepy", "neglected_neutral"):
        w.set_sprite(SpriteRef(str(ROOT / "assets/ai" / f"adult_{name}.png"), 256, 256))
        r._pump()
        moods.append((name, grab()))
        modes.append({"figure": name, "skinned": bool(w._root.property("skinnedMeshVisible"))})
    sheet(moods, OUT / "mood_backends_256.png", columns=3)
    r.close()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metrics = {"revision": revision, "render_size": 256, "sample_hz": 60, "capture_hz": 30,
               "note": "Mesh sigma metrics include transparent mesh cells; appearance assessed in Qt renders. Stress poses excluded here.",
               "transitions": transitions, "actual_session_deformation": deform, "mood_backends": modes,
               "max_solver_drift": max(x.get("drift", 0) for x in records)}
    (OUT / "trace.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Evidence: {OUT}")


if __name__ == "__main__":
    main()
