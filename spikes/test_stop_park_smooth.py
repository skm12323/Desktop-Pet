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
import math
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


def simulate(stage: str, size: int, walk_speed: float = 120.0, distance: float = 480.0):
    """app 同款游走意图（PetApp._loco_cruise_speed + 实测身体速度 50 ms 低通 + 同目标锁存）走向 distance 处的目标。

    返回逐帧步态行；附带 rows[-1]["meta"]：overshoot（窗口中心越过目标的 px，负 = 停在前方）。"""
    from app import PetApp
    pkg = os.path.join(ROOT, "assets", f"rig_{stage}_walk_v1")
    spec = json.load(open(os.path.join(pkg, "spec.json"), encoding="utf-8"))
    src_h = float(spec["skeleton"]["source_reference"]["image_size_px"][1])
    loco = SideLocomotion(spec, TurnClip(os.path.join(pkg, "clips", "turn_front_to_side_h256")),
                          TurnClip(os.path.join(pkg, "clips", "turn_side_to_front_h256")), size / src_h)
    dt, x = 1 / 60, 200.0
    target = x + size / 2 + distance
    vm, px, latched, stop_t = 0.0, x, False, None
    rows = []
    for i in range(60 * 30):
        cx = x + size / 2
        dx = target - cx
        vm += ((x - px) / dt - vm) * (1.0 - math.exp(-dt / 0.05))
        px = x
        if latched:
            vx = 0.0
        else:
            sp = PetApp._loco_cruise_speed(abs(dx), walk_speed, vm)
            if sp <= 0.0:
                latched, vx, stop_t = True, 0.0, i
            else:
                vx = sp if dx > 0 else -sp
        f = loco.update(dt, vx, x)
        if f.window_x is not None:
            x = f.window_x
        s = loco._solver
        if s is not None:
            rows.append(dict(
                i=i, intent=vx,
                gait=s.state.value, v=s._velocity, wx=s.window_x_float, dip=s._dip,
                kinds={sd: ft.contact.contact_type.name for sd, ft in s._feet.items()},
                feet={sd: (float(ft.ankle_now[0]), float(ft.ankle_now[1]), bool(ft.contact.is_locked))
                      for sd, ft in s._feet.items()}))
        if latched and s is not None and s.state.value == "idle_side":
            break
    rows[-1]["meta"] = dict(overshoot=(x + size / 2) - target, stop_i=stop_t)
    return rows


def stop_profile(rows):
    """减速开始（意图首次低于巡航）→ 静止：耗时与迈步数。"""
    cruise = max(abs(r["intent"]) for r in rows)
    i0 = next(k for k, r in enumerate(rows) if 0 < abs(r["intent"]) < cruise - 0.5 or r["intent"] == 0 and k > 60)
    steps, prev = 0, {sd: True for sd in ("l", "r")}
    for r in rows[i0:]:
        for sd, (_x, _y, lk) in r["feet"].items():
            if prev[sd] and not lk:
                steps += 1
            prev[sd] = lk
    return (len(rows) - i0) / 60.0, steps


def run(stage: str, size: int) -> None:
    rows = simulate(stage, size)
    seg = [r for r in rows if r["gait"] in STOP_STATES]
    check(f"{stage}: 会话收步到侧身静止", bool(seg) and seg[-1]["gait"] == "idle_side")
    # S1：停步段着地脚 / 落地那一帧的相邻帧位移（画布 px）——旧版着地回跳 117。
    # 摆动中的脚不计（v0.20.6 起从巡航速度直接刹车，正常摆动单帧可达 ~80，与行走同量级）
    jump = max((abs(b["feet"][sd][0] - a["feet"][sd][0])
                for a, b in zip(seg, seg[1:]) for sd in ("l", "r") if b["feet"][sd][2]),
               default=0.0)
    check(f"{stage}: S1 停步段着地/落地无跳变（最大逐帧 {jump:.1f} canvas px < 30）", jump < 30.0)
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
    # S4（v0.20.5）：收步不下蹲——下沉不超过进入收步时（旧版沿用行走伸展比，停下后蹲 ~9 px）
    if park:
        dip0 = park[0]["dip"]
        dmax = max(r["dip"] for r in park)
        check(f"{stage}: S4 收步不下蹲（下沉 {dip0:.1f} → 峰 {dmax:.1f} canvas px）", dmax <= dip0 + 0.5)
    # S5（v0.20.5）：低速刹车中平踩的脚不新进入踮脚（膝盖前顶后又不迈步）
    new_heel_rise = 0
    for a, b in zip(rows, rows[1:]):
        if b["gait"] in ("walk_brake", "walk_stop") and abs(b["v"]) < SLOW_V:
            for sd in ("l", "r"):
                if a["kinds"][sd] in ("FLAT_SOLE", "HEEL") and b["kinds"][sd] == "FOREFOOT":
                    new_heel_rise += 1
    check(f"{stage}: S5 低速刹车不新起踮脚（{new_heel_rise} 次）", new_heel_rise == 0)
    # 静止时双脚落在静止站位（收步完成）
    s_last = seg[-1]["feet"]
    check(f"{stage}: 静止双脚着地", all(lk for (_x, _y, lk) in s_last.values()))


def idle_wait(stage: str, size: int) -> float:
    """侧身静止 → SIDE_SETTLE（开始转回正面）的等待秒数。"""
    pkg = os.path.join(ROOT, "assets", f"rig_{stage}_walk_v1")
    spec = json.load(open(os.path.join(pkg, "spec.json"), encoding="utf-8"))
    src_h = float(spec["skeleton"]["source_reference"]["image_size_px"][1])
    loco = SideLocomotion(spec, TurnClip(os.path.join(pkg, "clips", "turn_front_to_side_h256")),
                          TurnClip(os.path.join(pkg, "clips", "turn_side_to_front_h256")), size / src_h)
    dt, x, t, t_idle = 1 / 60, 200.0, 0.0, None
    for i in range(60 * 20):
        t = i * dt
        f = loco.update(dt, 120.0 if t < 3.0 else 0.0, x)
        if f.window_x is not None:
            x = f.window_x
        s = loco._solver
        if t_idle is None and s is not None and s.state.value == "idle_side":
            t_idle = t
        if t_idle is not None and loco.state.value == "side_settle":
            return t - t_idle
    return float("inf")


def main() -> int:
    for stage, size in (("adult", 256), ("final", 320)):
        run(stage, size)
    # v0.20.6：游走停步直接——巡航到预测刹车点，停在目标附近、步数少
    for stage, size in (("adult", 256), ("final", 320)):
        for sp in (80.0, 120.0, 200.0):
            rows = simulate(stage, size, walk_speed=sp)
            dur, steps = stop_profile(rows)
            ov = rows[-1]["meta"]["overshoot"]
            check(f"{stage}@{sp:.0f}: 停步直接（减速→静止 {dur:.2f}s ≤ 1.2，迈步 {steps} ≤ 4，"
                  f"停位 {ov:+.1f}px ∈ [-6, +8]）", dur <= 1.2 and steps <= 4 and -6.0 <= ov <= 8.0)
    # v0.20.4：侧身站定后转回正面的等待 1.5 s（旧 4.0 s）
    for stage, size in (("adult", 256), ("final", 320)):
        w = idle_wait(stage, size)
        check(f"{stage}: 侧身站定 → 转回正面等待 {w:.2f}s ≈ 1.5s", abs(w - 1.5) < 0.1)
    # 单元：_park_glide_s 按距离自适应
    g = gait_mod
    check("PARK_GLIDE_MIN_S 下限 0.25 s", g.PARK_GLIDE_MIN_S == 0.25)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n== 停步平滑：{len(RESULTS) - len(failed)} 通过 / {len(failed)} 失败 ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
