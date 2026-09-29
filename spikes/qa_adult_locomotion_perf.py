"""G6 performance probe: presenter tick cost (CPU) and process memory with side locomotion.

Offscreen, real RigWindow: RSS before/after enabling side locomotion, then 60 s of scripted
walking (right/left/turns) at the 16 ms tick with the real render per tick (grabFramebuffer
is NOT called - only processEvents, like the app). Reports per-tick CPU time P50/P95/max and
RSS growth. Evidence -> spikes/_qa/adult_walk_v1/g6_integration/perf.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("QT_QUICK_BACKEND", "rhi")
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil


def main() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from pet.asset_provider import SpriteRef
    from pet.rig import presenter as pres
    from pet.rig.presenter import build_rig_window
    from pet.rig.spec import load_rig_spec
    from pet.window import WindowBase

    class Clock:
        t = 1000.0

        def perf_counter(self):
            return self.t

    proc = psutil.Process()
    app = QApplication.instance() or QApplication([])
    clock = Clock()
    pres.time = clock
    spec = load_rig_spec(str(ROOT / "assets" / "rig" / "adult"), "adult")
    win = build_rig_window(WindowBase, SpriteRef(spec.figures["healthy_neutral"], 256, 256), "adult")
    win._motion_timer.stop()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    win.move(300, 300)
    for _ in range(30):
        clock.t += 1 / 30
        win._motion_tick()
        app.processEvents()
    rss_front = proc.memory_info().rss
    assert win.enable_side_locomotion(str(ROOT / "assets" / "rig_adult_walk_v1"))
    for _ in range(30):
        clock.t += 1 / 60
        win._motion_tick()
        app.processEvents()
    rss_enabled = proc.memory_info().rss
    costs, rss_trace = [], []
    t = 0.0
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    while t < seconds:
        cyc = t % 15.0                      # 15 s cycle: right 4 s, idle, left 4 s, idle
        vx = 120.0 if cyc < 4 else (-120.0 if 7.5 <= cyc < 11.5 else 0.0)
        win.set_locomotion_intent(vx)
        clock.t += 1 / 60
        t += 1 / 60
        c0 = time.perf_counter_ns()
        win._motion_tick()
        app.processEvents()
        costs.append((time.perf_counter_ns() - c0) / 1e6)
        if int(t * 60) % 60 == 0:
            rss_trace.append(proc.memory_info().rss)
    costs = np.array(costs)
    rep = {
        "rss_front_mb": rss_front / 1e6, "rss_enabled_mb": rss_enabled / 1e6,
        "seconds": seconds, "rss_end_mb": rss_trace[-1] / 1e6,
        "rss_growth_second_half_mb": (rss_trace[-1] - rss_trace[len(rss_trace) // 2]) / 1e6,
        "rss_trace_mb_every_15s": [round(v / 1e6, 1) for v in rss_trace[::15]],
        "tick_ms_p50": float(np.percentile(costs, 50)), "tick_ms_p95": float(np.percentile(costs, 95)),
        "tick_ms_max": float(costs.max()),
        "note": "offscreen; tick = presenter logic + scene sync/render via processEvents (no grab)",
    }
    out = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g6_integration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "perf.json").write_bytes(json.dumps(rep, indent=2).encode("utf-8"))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
