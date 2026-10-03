"""Regression: all ADULT moods keep the rig, branch palette, and latest restore target.

Exercises real app frame selection and real Qt rendering; no normal app instance,
saved user state, network service, or desktop timer is started.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
OUT = ROOT / "spikes/_qa/adult_visual_fix_2026-09-30"


def main(stage="adult", size=256):
    global OUT
    if stage == "final":
        OUT = ROOT / "output/final_f7_integration_2026-10-03/moods"
    from render_rig_rest import RigRenderer
    from pet.rig import presenter
    from pet.asset_provider import AIArtProvider
    from pet.pet_state import Branch, Mood, PetState, PetStateStore, Stage
    from app import PetApp
    from PySide6.QtGui import QImage

    class Clock:
        t = 1000.0

        def perf_counter(self):
            return self.t

    clock = Clock()
    presenter.time = clock
    r = RigRenderer(stage=stage)
    w = r.win
    w.resize(size, size)
    r._pump()
    w._last_tick_s = clock.t
    provider = AIArtProvider()
    w.set_sprite_provider(provider)
    app = PetApp.__new__(PetApp)
    app.window, app.provider, app._part_walk = w, provider, False
    app.fsm, app._anim_key = SimpleNamespace(motion_mode="free"), None
    OUT.mkdir(parents=True, exist_ok=True)
    results, images = [], []

    def check(name, condition):
        results.append({"name": name, "pass": bool(condition)})
        print(f"[{'OK' if condition else 'FAIL'}] {name}")

    def tick(vx=0, airborne=False):
        w.set_motion_params(walking=bool(vx), walk_hz=1.2, airborne=airborne)
        w.set_locomotion_intent(vx)
        clock.t += 1 / 60
        w._motion_tick()
        r._pump(2)

    def grab():
        q = w._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine() // 4, 4)
        return Image.fromarray(a[:, :q.width()].copy())

    for branch in Branch:
        for mood in Mood:
            name = f"{branch.value}_{mood.value}"
            w.stop_frames()
            w.disable_side_locomotion()
            state = PetState(stage=Stage(stage), branch=branch)
            app.store, app._anim_key = PetStateStore(state), None
            w._conversation_mood = mood
            w.on_state_change(state)
            logical = w._sprite.path
            assert w.enable_side_locomotion(str(ROOT / f"assets/rig_{stage}_walk_v1"))
            w.move(200, 300)
            for i in range(360):
                tick(120)
                if i % 3 == 0:
                    app._frame_tick(None, "walk", "walk")
            check(f"{name}: visible side rig, no old walk frames",
                  w._root.property("skinnedMeshVisible") and w._loco_last.mode == "side"
                  and not w._frames and app._anim_key is None)
            check(f"{name}: logical mood retained", w._sprite.path == logical)
            check(f"{name}: branch palette during walk",
                  bool(w._root.property("locoNeglected")) == (branch == Branch.NEGLECTED))
            im = grab()
            images.append((name, im))
            a = np.asarray(im)
            mask = a[..., 3] > 127
            check(f"{name}: nonblank framebuffer", int(mask.sum()) > 5000)
            for _ in range(30):
                tick()
            check(f"{name}: carrier kept while braking/parking", w._loco_carrying)
            for _ in range(450):
                tick()
            check(f"{name}: restores original mood after whole session",
                  w._loco_last.mode == "front" and not w._loco_carrying
                  and w._root.property("activeFigure") == name and w._sprite.path == logical
                  and not w._root.property("locoNeglected"))

    # Compare rendered color, not just the palette property.
    def chroma(im):
        a = np.asarray(im)
        m = (a[..., 3] > 127) & (a[..., :3].max(-1) - a[..., :3].min(-1) > 5)
        return float(np.ptp(a[..., :3].astype(float), axis=-1)[m].mean())

    by_name = dict(images)
    ratio = chroma(by_name["neglected_neutral"]) / chroma(by_name["healthy_neutral"])
    check(f"neglected rendered palette is visibly muted (chroma ratio {ratio:.2f})", ratio < 0.6)

    state = PetState(stage=Stage(stage), branch=Branch.HEALTHY)
    w._conversation_mood = Mood.HAPPY
    w.on_state_change(state)
    for _ in range(250):
        tick(120)
    w.set_walk_figure(provider.side_walk_static(state))
    app._part_walk = True
    tick(120)
    app._frame_tick(None, "walk", "walk")
    check("paperdoll carrier cannot replace active side rig", w._root.property("skinnedMeshVisible")
          and not w._frames)
    app._part_walk = False
    w.set_conversation_mood(Mood.SAD)
    check("mood change during walking retains rig", w._root.property("skinnedMeshVisible"))
    w.locomotion_interrupt()
    check("interrupt restores latest mood", w._root.property("activeFigure") == "healthy_sad")
    # Interrupt may occur immediately after an intent, before the first active state.
    tick()
    w.set_locomotion_intent(120)
    w.locomotion_interrupt()
    check("pre-tick interrupt releases carrier", not w._loco_carrying)
    w.set_locomotion_intent(120)
    w.play_frames(provider.frames_for(stage, "fall")[:1])
    check("action frames interrupt locomotion", not w._loco_carrying and not w._loco.active)
    w.stop_frames()
    check("action restores latest mood", w._root.property("activeFigure") == "healthy_sad")
    w.enable_side_locomotion(str(ROOT / "assets/does_not_exist"))
    w.set_locomotion_intent(120)
    check("missing assets preserve fallback mood", not w._loco_carrying
          and w._root.property("activeFigure") == "healthy_sad")

    canvas = Image.new("RGB", (size * 5, (size+28) * 2), (245, 246, 250))
    for i, (name, im) in enumerate(images):
        b = Image.new("RGBA", im.size, (245, 246, 250, 255))
        b.alpha_composite(im)
        x, y = i % 5 * size, i // 5 * (size+28)
        canvas.paste(b.convert("RGB"), (x, y + 28))
        ImageDraw.Draw(canvas).text((x + 4, y + 8), name, fill="black")
    canvas.save(OUT / "mood_walk_after_256.png")
    (OUT / "mood_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    r.close()
    n = sum(x["pass"] for x in results)
    print(f"{n}/{len(results)} passed")
    assert n == len(results), f"{stage.upper()} mood locomotion regression"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("adult", "final"), default="adult")
    args = parser.parse_args()
    main(args.stage, 320 if args.stage == "final" else 256)
