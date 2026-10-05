"""v0.20.3 停步平滑回归（ADULT / FINAL 侧身会话，纯数值、无渲染）。

模拟 app 意图：匀速行走 → 按 _LOCO_INTENT_GAIN 接近目标减速 → 死区归零 → 收步静止。
锁定 v0.20.2 实测的三类"停步混乱"：
  S1 着地回跳：摆动脚落地一帧被拉回规划落点（旧版 117 canvas px）
  S2 原地踏步：刹车末段低速仍起新一步（水平不动、抬脚 ~75 px、两腿叠成一条）
  S3 收步猛冲：身体前移固定 0.25 s（19 px 时峰值 >110 px/s）
运行：python spikes/test_stop_park_smooth.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pet.rig import gait as gait_mod  # noqa: E402
from pet.rig.side_locomotion import SideLocomotion, TurnClip  # noqa: E402

RESULTS: list = []
STOP_STATES = ("walk_brake", "walk_stop", "walk_park", "idle_side")
SLOW_V = 20.0      # 判定"低速"的固定阈值（不读代码常量，防改常量即掩盖回归）


def check(name: str, ok: bool) -> None:
    RESULTS.append((name, bool(ok)))
    print(f"[{'OK' if ok else 'FAIL'}] {name}")


def simulate(stage: str, size: int, walk_speed: float = 120.0):
    pkg = os.path.join(ROOT, "assets", f"rig_{stage}_walk_v1")
    spec = json.load(open(os.path.join(pkg, "spec.json"), encoding="utf-8"))
    src_h = float(spec["skeleton"]["source_reference"]["image_size_px"][1])
    loco = SideLocomotion(spec, TurnClip(os.path.join(pkg, "clips", "turn_front_to_side_h256")),
                          TurnClip(os.path.join(pkg, "clips", "turn_side_to_front_h256")), size / src_h)
    dt, x, phase, walk_t, target = 1 / 60, 200.0, "walk", 0.0, None
    rows = []
    for _ in range(60 * 14):
        cx = x + size / 2
        if phase == "walk":
            vx = walk_speed
            walk_t += dt
            if walk_t > 3.0:
                phase, target = "approach", cx + 120.0
        elif phase == "approach":
            dx = target - cx
            if abs(dx) <= 4.0:
                vx, phase = 0.0, "stop"
            else:
                vx = min(200.0, walk_speed, 1.2 * abs(dx)) * (1 if dx > 0 else -1)
        else:
            vx = 0.0
        f = loco.update(dt, vx, x)
        if f.window_x is not None:
            x = f.window_x
        s = loco._solver
        if s is not None:
            rows.append(dict(
                gait=s.state.value, v=s._velocity, wx=s.window_x_float,
                feet={sd: (float(ft.ankle_now[0]), float(ft.ankle_now[1]), bool(ft.contact.is_locked))
                      for sd, ft in s._feet.items()}))
        if phase == "stop" and s is not None and s.state.value == "idle_side":
            break
    return rows


def run(stage: str, size: int) -> None:
    rows = simulate(stage, size)
    seg = [r for r in rows if r["gait"] in STOP_STATES]
    check(f"{stage}: 会话收步到侧身静止", bool(seg) and seg[-1]["gait"] == "idle_side")
    # S1：停步段任何一只脚的相邻帧位移（画布 px）——旧版着地回跳 117
    jump = max((abs(b["feet"][sd][0] - a["feet"][sd][0])
                for a, b in zip(seg, seg[1:]) for sd in ("l", "r")), default=0.0)
    check(f"{stage}: S1 停步段无脚部跳变（最大逐帧 {jump:.1f} canvas px < 30）", jump < 30.0)
    # S2：刹车/停步段身体慢于阈值后不应有脚离地摆动
    slow_swing = [r for r in seg if r["gait"] in ("walk_brake", "walk_stop")
                  and abs(r["v"]) < SLOW_V
                  and any(not lk for (_x, _y, lk) in r["feet"].values())]
    # 允许进入低速时已在空中的那一步落地（起始帧连续段），不允许低速期新起一步
    new_swings = 0
    prev_air = {sd: False for sd in ("l", "r")}
    for r in rows:                          # 全序列追踪离地起点（刹车前已在空中的一步不算）
        for sd, (_x, _y, lk) in r["feet"].items():
            air = not lk
            if air and not prev_air[sd] and r["gait"] in ("walk_brake", "walk_stop") \
                    and abs(r["v"]) < SLOW_V:
                new_swings += 1
            prev_air[sd] = air
    check(f"{stage}: S2 低速刹车不起原地一步（新起摆动 {new_swings}，低速离地帧 {len(slow_swing)}）",
          new_swings == 0)
    # S3：收步前移峰值速度
    park = [r for r in rows if r["gait"] == "walk_park"]
    vpk = max((abs(r["v"]) for r in park), default=0.0)
    check(f"{stage}: S3 收步前移峰值 {vpk:.0f} px/s ≤ {gait_mod.PARK_GLIDE_VMAX:.0f}",
          vpk <= gait_mod.PARK_GLIDE_VMAX + 0.5)
    # 静止时双脚落在静止站位（收步完成）
    s_last = seg[-1]["feet"]
    check(f"{stage}: 静止双脚着地", all(lk for (_x, _y, lk) in s_last.values()))


def main() -> int:
    for stage, size in (("adult", 256), ("final", 320)):
        run(stage, size)
    # 单元：_park_glide_s 按距离自适应
    g = gait_mod
    check("PARK_GLIDE_MIN_S 下限 0.25 s", g.PARK_GLIDE_MIN_S == 0.25)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n== 停步平滑：{len(RESULTS) - len(failed)} 通过 / {len(failed)} 失败 ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
