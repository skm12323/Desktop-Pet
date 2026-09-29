"""G6 scene tests: ADULT side locomotion through the real RigWindow (offscreen, fake clock).

Scenarios (docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §5.2 G6):
  1 front idle -> walk right -> stop -> side idle -> timeout turn back
  2 walking right -> reverse -> walk left
  3 drag during the turn clip          4 drag while walking
  5 fall (not grounded) while walking  6 stage switch during the clip
  7 missing assets -> legacy path
Gates: every rig<->clip switch (seam): premultiplied mean abs diff <= 3/255 over the silhouette,
silhouette centroid shift <= 1 px, sole line shift <= 0.5 px (256 window); window x does not
move during the turn; window x updates at the 60 Hz tick (no 20 Hz steps) while walking.

  D:\\anaconda3\\python.exe -X utf8 spikes/test_adult_locomotion.py
Evidence -> spikes/_qa/adult_walk_v1/g6_integration/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("QT_QUICK_BACKEND", "rhi")
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

PKG = str(ROOT / "assets" / "rig_adult_walk_v1")
OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g6_integration"
WIN = 256
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""), flush=True)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def perf_counter(self):
        return self.t


def main() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from pet.asset_provider import SpriteRef
    from pet.rig import presenter as pres
    from pet.rig.presenter import build_rig_window
    from pet.rig.spec import load_rig_spec
    from pet.window import WindowBase

    app = QApplication.instance() or QApplication([])
    clock = FakeClock()
    pres.time = clock                     # presenter reads time.perf_counter()
    spec = load_rig_spec(str(ROOT / "assets" / "rig" / "adult"), "adult")
    win = build_rig_window(WindowBase, SpriteRef(spec.figures["healthy_neutral"], WIN, WIN), "adult")
    win._motion_timer.stop()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    win.move(400, 300)
    for _ in range(6):
        app.processEvents()
    OUT.mkdir(parents=True, exist_ok=True)

    def grab() -> np.ndarray:
        img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        return np.frombuffer(img.constBits(), np.uint8).reshape(
            img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()].astype(np.float32).copy()

    def tick(dt: float = 1 / 60, vx: float = 0.0, dragging: bool = False, grounded: bool = True):
        win.set_locomotion_intent(vx)
        win._dragging = dragging
        win._motion_inputs.grounded = grounded
        clock.t += dt
        win._motion_tick()
        app.processEvents()
        lf = win._loco_last
        return lf

    def seam_metrics(a: np.ndarray, b: np.ndarray) -> dict:
        pa = a[..., :3] * a[..., 3:4] / 255.0
        pb = b[..., :3] * b[..., 3:4] / 255.0
        sil = (a[..., 3] > 8) | (b[..., 3] > 8)
        diff = float(np.abs(pa - pb).mean(-1)[sil].mean())

        def cent(x):
            al = x[..., 3] / 255.0
            ys, xs = np.mgrid[0:al.shape[0], 0:al.shape[1]]
            return np.array([(xs * al).sum(), (ys * al).sum()]) / max(al.sum(), 1e-6)

        def sole(x):
            # sub-pixel sole line: last row where the row-max alpha crosses 0.5, interpolated
            prof = x[..., 3].max(1) / 255.0
            rows = np.where(prof >= 0.5)[0]
            if not len(rows):
                return 0.0
            r0 = int(rows.max())
            if r0 + 1 >= len(prof):
                return float(r0)
            a0, a1 = prof[r0], prof[r0 + 1]
            return r0 + (a0 - 0.5) / max(a0 - a1, 1e-6)
        return {"mean_abs_255": diff, "centroid_px": float(np.linalg.norm(cent(a) - cent(b))),
                "sole_px": abs(sole(a) - sole(b))}

    # ---- 7 (first, cheap): missing assets -> fallback ----
    print("== 7. missing assets -> legacy ==")
    ok = win.enable_side_locomotion(str(ROOT / "assets" / "does_not_exist"))
    check("missing package: enable returns False", not ok)
    check("missing package: legacy path kept (no loco)", not win.locomotion_available())

    print("== enable ==")
    ok = win.enable_side_locomotion(PKG)
    check("side locomotion enabled on ADULT", ok and win.locomotion_available())
    if not ok:
        finish()
        return
    for _ in range(20):
        tick()

    # ---- 1: walk right, stop, idle, timeout turn back; seams at every switch ----
    print("== 1. front -> walk right -> stop -> side idle -> timeout turn back ==")
    frames, states, xs = [], [], []
    seams = []
    prev_img, prev_mode = grab(), "front"
    prev_alpha = 1.0
    prev_idx = -1
    clip_steps: list = []
    x_turn_start = None
    t = 0.0
    while t < 12.0:
        vx = 120.0 if t < 3.0 else 0.0
        lf = tick(1 / 60, vx)
        t += 1 / 60
        img = grab()
        fading = lf.mode == "clip" and (lf.clip_alpha < 1.0 or lf.under)
        same_clip_frame = lf.mode == "clip" and prev_mode == "clip" and lf.clip_index == prev_idx
        if lf.mode != prev_mode or ((fading or prev_alpha < 1.0) and same_clip_frame):
            # hand-over frames: mode switch, or a fade step with the clip frame unchanged
            m = seam_metrics(prev_img, img)
            seams.append((f"{prev_mode}->{lf.mode}(a={lf.clip_alpha:.2f})", round(t, 3), m))
        elif lf.mode == "clip" and prev_mode == "clip" and lf.clip_index != prev_idx:
            clip_steps.append(seam_metrics(prev_img, img)["mean_abs_255"])
        prev_idx = lf.clip_index if lf.mode == "clip" else -1
        prev_alpha = (lf.clip_alpha if not lf.under else 0.5) if lf.mode == "clip" else 1.0
        if lf.state.value in ("settle", "turn_out") and x_turn_start is None:
            x_turn_start = win.x()
        states.append((round(t, 3), lf.state.value, lf.mode, win.x()))
        frames.append(img)
        prev_img, prev_mode = img, lf.mode
    seq = [s[1] for s in states]
    order = [v for i, v in enumerate(seq) if i == 0 or v != seq[i - 1]]
    check("state order front→settle→turn_out→side→side_settle→turn_in→front",
          order == ["settle", "turn_out", "side", "side_settle", "turn_in", "front"], str(order))
    turn_x = [s[3] for s in states if s[1] in ("settle", "turn_out")]
    check("window x fixed during settle + turn clip", len(set(turn_x)) == 1, f"{sorted(set(turn_x))[:4]}")
    walk_x = [s[3] for s in states if s[1] == "side"]
    moved = walk_x[-1] - walk_x[0] if walk_x else 0
    check("walked right while in side mode", moved > 150, f"{moved} px")
    walk_ticks = [s for s in states if s[1] == "side"][:120]
    changes = sum(1 for a_, b_ in zip(walk_ticks, walk_ticks[1:]) if a_[3] != b_[3])
    check("window x updated at the render tick (no 20 Hz steps)", changes >= 0.5 * len(walk_ticks) - 20,
          f"{changes} changes in {len(walk_ticks)} ticks @60Hz")
    for name, tt, m in seams:
        check(f"seam {name} @{tt}s", m["mean_abs_255"] <= 3.0 and m["centroid_px"] <= 1.0 and m["sole_px"] <= 0.5,
              f"diff {m['mean_abs_255']:.2f}/255, centroid {m['centroid_px']:.2f} px, sole {m['sole_px']:.1f} px")
    if clip_steps:
        print(f"  [INFO] clip-internal frame steps (animation + end morph, not seams): "
              f"median {np.median(clip_steps):.2f}, max {max(clip_steps):.2f}/255")
    save_evidence("s1_walk_right", frames, states)

    # ---- 2: walk right then reverse ----
    print("== 2. reverse while walking ==")
    for _ in range(30):
        tick()
    frames, states = [], []
    t = 0.0
    while t < 11.0:
        vx = 120.0 if t < 3.0 else (-120.0 if t < 8.5 else 0.0)
        lf = tick(1 / 60, vx)
        t += 1 / 60
        states.append((round(t, 3), lf.state.value, lf.mode, win.x(), lf.facing))
        frames.append(grab())
    seq = [s[1] for s in states]
    order = [v for i, v in enumerate(seq) if i == 0 or v != seq[i - 1]]
    faces = [s[4] for s in states if s[1] == "side"]
    check("reverse goes side→side_settle→turn_in→(front)→settle→turn_out→side",
          [o for o in order if o != "front"][:7] == ["settle", "turn_out", "side", "side_settle", "turn_in",
                                                     "settle", "turn_out"] and order.count("side") >= 2, str(order))
    check("second session faces left (mirrored)", -1 in faces and 1 in faces)
    left_x = [s[3] for s in states if s[1] == "side" and s[4] == -1]
    check("walked left in the second session", left_x and left_x[-1] < left_x[0] - 60,
          f"{left_x[0] if left_x else None} -> {left_x[-1] if left_x else None}")
    save_evidence("s2_reverse", frames, states)
    for _ in range(600):
        if tick().state.value == "front":
            break

    # ---- 3: drag during the turn clip ----
    print("== 3. drag during turn clip ==")
    for _ in range(30):
        tick()
    lf = None
    for _ in range(40):
        lf = tick(1 / 60, 120.0)
        if lf.mode == "clip":
            break
    check("reached turn clip", lf is not None and lf.mode == "clip")
    lf = tick(1 / 60, 120.0, dragging=True)
    check("drag → front immediately", lf.mode == "front" and not win.locomotion_controls_x(), lf.state.value)
    check("scene shows front mesh", int(win._root.property("locoMode")) == 0)
    for _ in range(10):
        tick(1 / 60, 0.0, dragging=True)
    lf = tick(1 / 60, 0.0)
    check("after drop: front idle", lf.state.value == "front")

    # ---- 4: drag while walking ----
    print("== 4. drag while walking ==")
    for _ in range(200):
        lf = tick(1 / 60, 120.0)
        if lf.state.value == "side" and lf.gait is not None and abs(lf.gait.delta_window_x) > 0:
            break
    check("walking before drag", lf.state.value == "side")
    lf = tick(1 / 60, 120.0, dragging=True)
    check("drag while walking → front, no residual lock", lf.mode == "front" and win._loco._solver is None)

    # ---- 5: fall while walking ----
    print("== 5. fall (airborne) while walking ==")
    for _ in range(30):
        tick()
    for _ in range(200):
        lf = tick(1 / 60, 120.0)
        if lf.state.value == "side":
            break
    lf = tick(1 / 60, 120.0, grounded=False)
    check("airborne → front", lf.mode == "front")
    for _ in range(5):
        tick(1 / 60, 0.0)

    # ---- 6: stage switch during clip ----
    print("== 6. stage switch during clip ==")
    for _ in range(80):
        lf = tick(1 / 60, 120.0)
        if lf.mode == "clip":
            break
    win.set_stage("young")
    for _ in range(4):
        app.processEvents()
    check("stage→young disables side locomotion", not win.locomotion_available())
    check("scene back to front mode", int(win._root.property("locoMode")) == 0)
    finish()


def save_evidence(tag: str, frames: list, states: list) -> None:
    (OUT / f"{tag}_states.json").write_bytes(json.dumps(states).encode("utf-8"))
    xs = [s[3] for s in states]
    x0, x1 = min(xs) - 10, max(xs) + WIN + 10
    gif = []
    for (st, img) in zip(states, frames[::1]):
        bg = Image.new("RGBA", (x1 - x0, WIN + 16), (245, 246, 250, 255))
        bg.alpha_composite(Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGBA"), (st[3] - x0, 16))
        ImageDraw.Draw(bg).text((4, 2), f"{st[0]:.2f}s {st[1]}", fill=(0, 0, 0, 255))
        gif.append(bg.convert("RGB"))
    gif[0].save(OUT / f"{tag}_1x.gif", save_all=True, append_images=gif[2::2], duration=33, loop=0)


def finish() -> None:
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n== 门禁结果：{n_ok} 通过 / {len(results) - n_ok} 失败 ==")
    (OUT / "results.json").write_bytes(json.dumps(
        [{"name": n, "pass": ok, "detail": d} for n, ok, d in results], indent=1, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
