"""Regression: neglected action frames use the muted palette on both backends.

Action frames (chew/eat_mouse/stretch/roll/fall/blink) exist once per stage in
colour. In the neglected branch they must render with the same muted palette as
the neglected static art and the side walk; healthy stays in colour.

rig backend  : scene property ``frameNeglected`` drives the mirrorNode layer.
frames backend: WindowBase desaturates the cached pixmap.
Run: python spikes/test_neglected_frame_palette.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

RESULTS: list[tuple[str, bool]] = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(f"[{'OK' if cond else 'FAIL'}] {name}")


def chroma_rgba(a: np.ndarray) -> float:
    m = (a[..., 3] > 127) & (a[..., :3].max(-1) - a[..., :3].min(-1) > 5)
    return float(np.ptp(a[..., :3].astype(float), axis=-1)[m].mean()) if m.any() else 0.0


def qimage_rgba(q) -> np.ndarray:
    from PySide6.QtGui import QImage
    q = q.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine() // 4, 4)
    return a[:, :q.width()].copy()


def rig_checks(stage: str) -> None:
    from render_rig_rest import RigRenderer
    from pet.asset_provider import AIArtProvider
    from pet.pet_state import Branch, PetState, Stage

    r = RigRenderer(stage=stage)
    w = r.win
    w.resize(256, 256)
    provider = AIArtProvider()
    w.set_sprite_provider(provider)
    chew = provider.frames_for(stage, "chew")
    check(f"rig/{stage}: chew frames exist", len(chew) >= 1)

    def play_and_grab(branch):
        w.stop_frames()
        w.on_state_change(PetState(stage=Stage(stage), branch=branch))
        r._pump()
        w.play_frames(list(chew), loop=True, interval_ms=10_000)
        r._pump(6)
        return bool(w._root.property("frameNeglected")), qimage_rgba(w._quick.grabFramebuffer())

    flag_h, im_h = play_and_grab(Branch.HEALTHY)
    check(f"rig/{stage}: healthy frames not muted", flag_h is False)
    flag_n, im_n = play_and_grab(Branch.NEGLECTED)
    check(f"rig/{stage}: neglected frames set frameNeglected", flag_n is True)
    ratio = chroma_rgba(im_n) / max(1e-6, chroma_rgba(im_h))
    check(f"rig/{stage}: neglected frames visibly muted (chroma ratio {ratio:.2f})", ratio < 0.6)
    w.stop_frames()
    r._pump()
    check(f"rig/{stage}: palette released after frames",
          not w._root.property("frameNeglected"))
    # branch change while frames play follows immediately (evolve/reset)
    w.play_frames(list(chew), loop=True, interval_ms=10_000)
    w.on_state_change(PetState(stage=Stage(stage), branch=Branch.HEALTHY))
    check(f"rig/{stage}: branch change mid-frames follows",
          not w._root.property("frameNeglected"))
    w.stop_frames()

    # v0.20.1：静止也走 rig——所有心情 × 双分支都显示蒙皮骨骼，neglected 灰调
    from pet.pet_state import Mood

    def idle_grab(branch, mood):
        w._conversation_mood = mood
        w.on_state_change(PetState(stage=Stage(stage), branch=branch))
        r._pump(6)
        return (w._root.property("activeFigure"), bool(w._root.property("skinnedMeshVisible")),
                bool(w._root.property("idleNeglected")), qimage_rgba(w._quick.grabFramebuffer()))

    fig_h, vis_h, neg_h, im_h = idle_grab(Branch.HEALTHY, Mood.HAPPY)
    check(f"rig/{stage}: healthy mood figure {fig_h} renders skinned rig", vis_h and not neg_h)
    fig_n, vis_n, neg_n, im_n = idle_grab(Branch.NEGLECTED, Mood.SAD)
    check(f"rig/{stage}: neglected mood figure {fig_n} renders skinned rig + idleNeglected",
          vis_n and neg_n)
    ratio = chroma_rgba(im_n) / max(1e-6, chroma_rgba(im_h))
    check(f"rig/{stage}: neglected idle rig visibly muted (chroma ratio {ratio:.2f})", ratio < 0.6)
    check(f"rig/{stage}: skinned_motion_active for mood figures", w.skinned_motion_active())
    w._conversation_mood = None


def frames_backend_checks(stage: str) -> None:
    from pet.asset_provider import AIArtProvider
    from pet.pet_state import Branch, PetState, Stage
    from pet.window import WindowBase

    provider = AIArtProvider()

    def run(branch):
        st = PetState(stage=Stage(stage), branch=branch)
        win = WindowBase(provider.get_static(st))
        win.set_sprite_provider(provider)
        win.on_state_change(st)
        win.play_frames(provider.frames_for(stage, "chew"), loop=True, interval_ms=10_000)
        during = qimage_rgba(win._label.pixmap().toImage())
        win.stop_frames()
        after = qimage_rgba(win._label.pixmap().toImage())
        return during, after

    h_during, _ = run(Branch.HEALTHY)
    n_during, n_after = run(Branch.NEGLECTED)
    ratio = chroma_rgba(n_during) / max(1e-6, chroma_rgba(h_during))
    check(f"frames/{stage}: neglected frames muted (chroma ratio {ratio:.2f})", ratio < 0.6)
    check(f"frames/{stage}: alpha untouched by muting",
          np.array_equal(n_during[..., 3] > 0, h_during[..., 3] > 0))
    # static neglected art is restored as-is (no double muting)
    st = PetState(stage=Stage(stage), branch=Branch.NEGLECTED)
    ref = WindowBase(provider.get_static(st))
    ref_im = qimage_rgba(ref._label.pixmap().toImage())
    check(f"frames/{stage}: static neglected art restored unmuted",
          abs(chroma_rgba(n_after) - chroma_rgba(ref_im)) < 1.0)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv)
    for stage in ("young", "final"):
        frames_backend_checks(stage)
    for stage in ("young", "adult", "final"):
        rig_checks(stage)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n== 门禁结果：{len(RESULTS) - len(failed)} 通过 / {len(failed)} 失败 ==")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
