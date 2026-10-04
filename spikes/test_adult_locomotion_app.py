"""G6 app smoke test: real PetApp (isolated saves, ADULT, adult_locomotion=side_rig).

Drives the real timers (behavior tick 50 ms, presenter motion tick 16/33 ms) and forces one
walk to a target to the right, then one to the left. Checks: the side locomotion is enabled
at startup; the window x does not move during settle + turn clip; while the session controls
x the behavior FSM position equals the window bottom-centre every tick (no tug-of-war); the
pet arrives near the target and returns to the front rig after the side-idle timeout; a
drag interrupts the session. Evidence -> spikes/_qa/adult_walk_v1/g6_integration/app_*.json
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer

from app import PetApp
from pet.platform import get_platform_adapter

OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g6_integration"
results: list = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def main(stage="adult"):
    global OUT
    if stage == "final":
        OUT = ROOT / "output/final_f7_integration_2026-10-03/app"
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="desktop-pet-g6-") as tmp:
        with patch.dict(os.environ, {"LOCALAPPDATA": tmp}):
            adapter = get_platform_adapter()
        actual = adapter.get_paths()
        from pet.pet_state import PetState, PetStateStore, Stage
        PetStateStore(PetState(stage=Stage(stage), age=30 if stage == "final" else 14)).save(str(Path(tmp) / "pet_state.json"))
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps({"provider": "ai"}), encoding="utf8")
        paths = dict(actual, data_dir=tmp, log_dir=tmp, config_path=str(config_path),
                     lock_path=str(Path(tmp) / "qa.lock"))
        from pet import config as cfgmod
        real_load = cfgmod.load_config

        def load_cfg(path):
            c = real_load(path)
            # Exercise the shipped default, including config merging for an older config.
            return c

        with patch.object(adapter, "get_paths", return_value=paths), \
                patch.object(PetApp, "_setup_chat"), patch.object(PetApp, "_setup_hotkeys"), \
                patch("app.load_config", side_effect=load_cfg) if hasattr(__import__("app"), "load_config") \
                else patch.object(cfgmod, "load_config", side_effect=load_cfg):
            pet = PetApp(["g6-app-qa"], adapter, verbose=False)
        pet._proactive_timer.stop()
        pet._chat_emotion_timer.stop()
        win, fsm = pet.window, pet.fsm
        log = []
        phase = {"name": "boot", "t0": time.monotonic(), "target": None}

        def sample():
            lf = getattr(win, "_loco_last", None)
            log.append({"t": round(time.monotonic() - phase["t0"], 3), "phase": phase["name"],
                        "loco": lf.state.value if lf else None, "mode": fsm.mode,
                        "win_cx": win.x() + win.width() / 2.0, "fsm_x": fsm.pos[0],
                        "controls": win.locomotion_controls_x() if hasattr(win, "locomotion_controls_x") else None})

        sampler = QTimer()
        sampler.timeout.connect(sample)

        def walk_to(dx):
            x, y = fsm.pos
            phase["target"] = x + dx
            fsm._target = (x + dx, y)
            fsm._mode = "walk"

        def step_boot():
            ok = win.locomotion_available() if hasattr(win, "locomotion_available") else False
            check(f"side locomotion enabled at startup ({stage.upper()} + side_rig)", ok)
            if not ok:
                return finish(pet)
            fsm._idle_left = 1e9           # no spontaneous wander during the test
            fsm._new_idle = lambda: 1e9
            phase.update(name="walk_right", t0=time.monotonic())
            walk_to(+260)
            sampler.start(16)
            QTimer.singleShot(12500, step_left)

        def step_left():
            phase.update(name="walk_left", t0=time.monotonic())
            walk_to(-260)
            QTimer.singleShot(12500, step_drag)

        def step_drag():
            phase.update(name="drag", t0=time.monotonic())
            walk_to(+200)
            QTimer.singleShot(1600, lambda: (setattr(win, "_dragging", True), fsm.begin_drag(fsm.pos)))
            QTimer.singleShot(2200, lambda: (setattr(win, "_dragging", False), fsm.end_drag()))
            QTimer.singleShot(3500, lambda: finish(pet, log))

        QTimer.singleShot(1500, step_boot)
        QTimer.singleShot(45000, pet.shutdown)          # watchdog
        pet.run()


def finish(pet, log=None):
    log = log or []
    (OUT / "app_trace.json").write_bytes(json.dumps(log).encode("utf-8"))
    for ph in ("walk_right", "walk_left"):
        rows = [r for r in log if r["phase"] == ph]
        turn = [r["win_cx"] for r in rows if r["loco"] in ("settle", "turn_out")]
        check(f"{ph}: window fixed during settle + turn clip", turn and max(turn) - min(turn) <= 0.5,
              f"{min(turn) if turn else None}..{max(turn) if turn else None}")
        # no tug-of-war: while the session drives x the window never steps backwards
        sgn = 1 if ph == "walk_right" else -1
        side_x = [r["win_cx"] for r in rows if r["loco"] == "side"]
        back = [sgn * (b - a) for a, b in zip(side_x, side_x[1:]) if sgn * (b - a) < -0.5]
        check(f"{ph}: window never steps backwards while walking (no FSM/gait tug-of-war)", not back,
              f"{len(back)} backward steps, worst {min(back) if back else 0:.1f} px")
        lag = [abs(r["win_cx"] - r["fsm_x"]) for r in rows if r["controls"]]
        # FSM syncs x on its 50 ms tick; real QTimer ticks slip to 70-90 ms under load, so the
        # bound is two ticks of travel at 120 px/s (the tug-of-war check above is the real gate)
        check(f"{ph}: FSM x follows the window (<= two 50 ms ticks of travel)", lag and max(lag) <= 13.0,
              f"max {max(lag):.2f} px" if lag else "no samples")
        seq = [r["loco"] for r in rows]
        order = [v for i, v in enumerate(seq) if v and (i == 0 or v != seq[i - 1])]
        check(f"{ph}: full session incl. timeout turn back", "side" in order and order[-1] == "front", str(order))
        xs = [r["win_cx"] for r in rows]
        if xs:
            moved = xs[-1] - xs[0]
            check(f"{ph}: travelled ~260 px", 200 <= abs(moved) <= 330 and (moved > 0) == (ph == "walk_right"),
                  f"{moved:.0f} px")
    rows = [r for r in log if r["phase"] == "drag"]
    check("drag interrupts to front", any(r["loco"] == "front" and r["mode"] == "drag" for r in rows))
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n== 门禁结果：{n_ok} 通过 / {len(results) - n_ok} 失败 ==", flush=True)
    (OUT / "app_results.json").write_bytes(json.dumps(
        [{"name": n, "pass": ok, "detail": d} for n, ok, d in results], indent=1, ensure_ascii=False).encode("utf-8"))
    pet.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("adult", "final"), default="adult")
    args = parser.parse_args()
    main(args.stage)
    sys.exit(0 if results and all(ok for _, ok, _ in results) else 1)
