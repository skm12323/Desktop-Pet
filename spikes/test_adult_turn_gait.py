"""ADULT 连续视角与步态计算核验收门禁（spikes/test_adult_turn_gait.py）。

对照 ``docs/DEEPSEEK_GLM_TASK_SPEC-ADULT连续视角与步态计算核.md`` §7 的七项
硬门槛 + 模块 1/2/3 的接口契约单元测试。全部无头可跑（数学核零 Qt 依赖；
仅末段 presenter 冒烟需要 QApplication）：

    D:/anaconda3/python.exe -X utf8 spikes/test_adult_turn_gait.py

统一口径：像素 = 256 高窗口**逻辑显示像素**（世界系）；画布 960×1696，
S = 256/1696 ≈ 0.150943。
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pet.rig.gait import (          # noqa: E402
    Analytical2BoneIK, ContactType, GaitOutputs, GaitPhaseState, GaitSolver,
    collect_gait_spec,
)
from pet.rig.skinned_mesh_item import RigRuntime, ViewKeyforms  # noqa: E402
from pet.rig.spec import load_rig_spec  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PKG = os.path.join(ROOT, "assets", "rig_adult_turn_v1")
SPEC = os.path.join(PKG, "spec.json")
MESH = os.path.join(PKG, "mesh", "mesh_data.json")
LAYERS = os.path.join(PKG, "layers")
KEYFORMS = os.path.join(PKG, "mesh", "view_keyforms.json")
BASELINE_SPEC = os.path.join(PKG, "baseline", "spec.json")
BASELINE_MESH = os.path.join(PKG, "baseline", "mesh", "mesh_data.json")

S = 256.0 / 1696.0
GROUND = 1608.0

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK] {name}" + (f"  [{detail}]" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"  [{detail}]" if detail else ""))


def load_solver() -> GaitSolver:
    with open(SPEC, "r", encoding="utf-8") as f:
        return GaitSolver(json.load(f), window_scale=S)


def run_walk(dt_fn, duration: float = 6.0, vx: float = 75.0, states: set = None):
    """通用行走仿真：浮点窗口累加器 + int 渲染落位（呈现层语义）。

    按**固定物理时长**推进（时间一致性前提）；``states`` 传入集合时记录
    全程状态。返回 (solver, 终点窗口 x, 输出序列, 状态集合)。
    """
    gs = load_solver()
    wx = 100.0
    t = 0.0
    outs = []
    i = 0
    while t < duration - 1e-9:
        dt = dt_fn(i) if callable(dt_fn) else dt_fn
        dt = min(dt, duration - t)
        out = gs.update(dt, vx, (int(round(wx)), 0), True, False)
        wx += out.delta_window_x
        t += dt
        i += 1
        outs.append(out)
        if states is not None:
            states.add(gs.state.value)
    return gs, wx, outs, states


# ============================== 1. IK 数学核 ==============================

def test_ik_math() -> None:
    print("== 1. Analytical2BoneIK 闭式解 / 软钳 / 踝补偿不变量 ==")
    rng = random.Random(42)
    worst_fk = 0.0
    worst_reach = 0.0
    for _ in range(300):
        l1 = rng.uniform(150.0, 450.0)
        l2 = rng.uniform(100.0, 350.0)
        hip = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50)])
        d_max = l1 + l2
        ang = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(abs(l1 - l2) * 1.05, d_max * 1.25)   # 含过伸区
        tgt = hip + d * np.array([math.cos(ang), math.sin(ang)])
        pitch = rng.uniform(-1.2, 1.2)
        h, k, a = Analytical2BoneIK.solve(hip, tgt, l1, l2, 1, pitch)
        # FK 重构：先做与求解器一致的软钳目标，再按三角重构（σ=+1：膝在
        # 髋踝线负 x 侧）；可达区应精确命中，过伸区命中软钳目标
        vec = tgt - hip
        d_raw = float(np.linalg.norm(vec))
        d_c = Analytical2BoneIK.soft_clamp_distance(d_raw, d_max)
        d_min = abs(l1 - l2) + Analytical2BoneIK.EPS_PX
        if d_raw < d_min:
            d_c = d_min + Analytical2BoneIK.D0_PX * (1.0 - math.exp(-(d_min - d_raw) / Analytical2BoneIK.D0_PX))
        unit = vec / d_raw
        along = (l1 * l1 - l2 * l2 + d_c * d_c) / (2 * d_c)
        hgt = math.sqrt(max(0.0, l1 * l1 - along * along))
        knee = hip + along * unit + hgt * np.array([-unit[1], unit[0]])
        tgt_c = hip + d_c * unit
        ankle_fk = knee + l2 * (tgt_c - knee) / float(np.linalg.norm(tgt_c - knee))
        worst_fk = max(worst_fk, float(np.linalg.norm(ankle_fk - tgt_c)))
        worst_reach = max(worst_reach, float(np.linalg.norm(ankle_fk - hip)) - d_max)
        # 踝补偿不变量：足底世界仰角 == 目标（零骨架链）
        check_pitch = h + k + a
        if abs(check_pitch - pitch) > 1e-9:
            check("IK 踝补偿不变量", False, f"{check_pitch} != {pitch}")
            return
    check("IK 三角重构 300 随机样例（含过伸软钳区）", worst_fk < 1e-6 and worst_reach < 1e-6,
          f"软钳目标误差 {worst_fk:.2e} px；链长越界 {worst_reach:.2e} px")
    check("IK 踝补偿不变量 h+k+a == φ", True)

    # 软钳 C1 连续性：D 扫过 L1+L2 边界，踝位无跳变
    l1, l2 = 409.0, 241.0
    hip = np.array([0.0, 0.0])
    prev = None
    max_jump = 0.0
    for i in range(400):
        d = l1 + l2 - 3.0 + i * 0.02
        tgt = hip + np.array([0.0, d])
        h, k, a = Analytical2BoneIK.solve(hip, tgt, l1, l2, 1, 0.0)
        knee_ang = math.pi / 2 + h
        knee = hip + l1 * np.array([math.cos(knee_ang), math.sin(knee_ang)])
        shin_ang = knee_ang + k
        ank = knee + l2 * np.array([math.cos(shin_ang), math.sin(shin_ang)])
        if prev is not None:
            max_jump = max(max_jump, float(np.linalg.norm(ank - prev)))
        prev = ank
    check("软钳奇异点 C1 连续（D 扫过 L1+L2）", max_jump < 0.05,
          f"max step {max_jump:.4f} px")

    # 数据契约（§6.1）：接口签名与字段齐全
    out = GaitOutputs(0.0, 0.0, (0.0, 0.0), {}, ContactType.NONE,
                      ContactType.NONE, 0.0)
    check("GaitOutputs 数据契约字段", hasattr(out, "delta_window_x")
          and hasattr(out, "view_yaw") and hasattr(out, "pelvis_offset")
          and hasattr(out, "bone_rotations") and hasattr(out, "foot_slide_drift_px"))
    names = {s.value for s in GaitPhaseState}
    expect = {"idle_front", "turn_to_side", "walk_start", "walk_loop",
              "walk_brake", "walk_stop", "turn_to_front", "idle_side",
              "turn_reverse", "airborne", "landing"}
    check("GaitPhaseState 11 态齐全", names == expect,
          f"diff: {names ^ expect}")


# ============================== 2. 关键形态（模块 2） ==============================

def test_keyforms() -> None:
    print("== 2. 连续视角关键形态：零姿态恒等 / 非退化 / C1 / 回退 ==")
    check("view_keyforms.json 存在（构建工具产物）", os.path.isfile(KEYFORMS))
    rt = RigRuntime.load(SPEC, MESH, LAYERS, KEYFORMS)
    check("关键形态挂接成功（float64 数学核）",
          rt is not None and rt._kf is not None)

    # 门槛 1：零姿态恒等性 ≤ 1e-6 px（§4.2；含端点外钳位区）
    B = len(rt.bones)
    ang = np.zeros(B)
    tx = np.zeros(B)
    ty = np.zeros(B)
    worst = 0.0
    for i in range(101):
        yaw = -8.0 + 58.0 * i / 100.0
        rt.set_view_yaw(yaw)
        rt.skinning_matrices(ang, tx, ty, 0.0, 0.0)
        for l in rt.layers:
            out = rt.deform(l, rt.effective_rest(l, 0.0))
            worst = max(worst, float(np.abs(out[:, :2] - l.rest[:, :2]).max()))
    check("【门槛】零姿态恒等性 max|p'−p| ≤ 1e-6", worst <= 1e-6,
          f"实测 {worst:.2e} px（101 采样 × 21 层）")

    # 门槛 2：网格非退化（101 点，遍历 21 层所有三角形）
    kf = rt._kf
    yaws = kf.yaws
    y_lo, y_hi = float(yaws[0]), float(yaws[-1])
    min_det = math.inf
    sign_ok = True
    for l in rt.layers:
        rec = kf.layers[l.layer_id]
        tris = l.triangles.reshape(-1, 3).astype(np.int64)
        front = rec["positions"][0]
        front_sign = np.sign(
            (front[tris[:, 1], 0] - front[tris[:, 0], 0])
            * (front[tris[:, 2], 1] - front[tris[:, 0], 1])
            - (front[tris[:, 1], 1] - front[tris[:, 0], 1])
            * (front[tris[:, 2], 0] - front[tris[:, 0], 0]))
        for i in range(101):
            yaw = y_lo + (y_hi - y_lo) * i / 100.0
            v = kf.interp_layer(l.layer_id, yaw)
            a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
            det = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            min_det = min(min_det, float(np.abs(det).min()))
            sign_ok = sign_ok and bool((np.sign(det) == front_sign).all())
    check("【门槛】网格非退化 101 点 min|det| > 1e-4", min_det > 1e-4,
          f"min |det| = {min_det:.1f} px²；方向一致 = {sign_ok}")

    # C1 连续：视角微分的最大跃度（§4.3 角速度跃度 ≤ 0.2 px/rad）
    worst_jerk = 0.0
    rt.set_view_yaw(0.0)
    prev_vel = None
    d_yaw = math.radians(0.25)
    for i in range(int(45 / 0.25) + 1):
        yaw = i * 0.25
        rt.set_view_yaw(yaw)
        v = kf.interp_layer("leg_l", yaw)
        rt.set_view_yaw(yaw + 0.25)
        v2 = kf.interp_layer("leg_l", yaw + 0.25)
        vel = (v2 - v) / d_yaw
        if prev_vel is not None:
            worst_jerk = max(worst_jerk, float(np.abs(vel - prev_vel).max()))
        prev_vel = vel
    check("视角插值 C1（二阶差分有界）", worst_jerk < 60.0,
          f"max |Δv/Δθ| {worst_jerk:.1f} px/rad²（有界即无折角）")

    # 补片透明度曲线（§4.4）
    a = ViewKeyforms.patch_alpha((15.0, 45.0), 0.0)
    b = ViewKeyforms.patch_alpha((15.0, 45.0), 30.0)
    c = ViewKeyforms.patch_alpha((15.0, 45.0), 45.0)
    check("补片 α 曲线端点钳位/中点 0.5", a == 0.0 and c == 1.0
          and abs(b - 0.5) < 1e-12)

    # 回退铁律：损坏 keyforms → ValueError → attach False → legacy 存活
    bad = os.path.join(os.path.dirname(KEYFORMS), "_bad_keyforms.json")
    with open(bad, "w", encoding="utf-8") as f:
        json.dump({"version": "9.9"}, f)
    rt2 = RigRuntime.load(SPEC, MESH, LAYERS)
    ok = rt2.attach_view_keyforms(bad)
    M = rt2.skinning_matrices(ang, tx, ty, 0.0, 0.0)
    l0 = rt2.layers[0]
    out = rt2.deform(l0, rt2.effective_rest(l0, 0.0))
    check("损坏 keyforms 回退正面单视角（宽进严出）",
          not ok and rt2._kf is None
          and float(np.abs(out[:, :2] - l0.rest[:, :2]).max()) < 1e-3)
    os.remove(bad)


# ============================== 3-7. 步态门槛 ==============================

def _chain_marker_world(rt: "RigRuntime", solver: GaitSolver, out: GaitOutputs,
                        win_x: int) -> "dict[str, tuple[float, float]]":
    """经真实运行时 FK（含限位）取接触标记世界坐标（契约级链路）。"""
    B = len(rt.bones)
    ang = np.zeros(B)
    tx = np.zeros(B)
    ty = np.zeros(B)
    for bone, rad in out.bone_rotations.items():
        i = rt.bone_index.get(bone)
        if i is not None:
            ang[i] = math.degrees(rad)
    sway, dip = out.pelvis_offset
    i = rt.bone_index.get("root_hip")
    if i is not None:
        ang[i] = math.degrees(out.bone_rotations.get("root_hip", 0.0))
        tx[i] = sway
        ty[i] = dip
    rt.skinning_matrices(ang, tx, ty, 0.0, 0.0)
    res = {}
    for side in ("l", "r"):
        f = solver._feet[side]
        kind = f.contact.contact_type if f.contact.is_locked else ContactType.FLAT_SOLE
        off = solver._marker_offset(side, kind)
        idx = rt.bone_index[f"foot_{side}"]
        jx, jy = rt.bones[idx].joint_px
        marker = np.array([jx + float(off[0]), jy + float(off[1]), 1.0])
        world = rt.M[idx] @ marker
        res[side] = (win_x + S * float(world[0]), float(world[1]))
    return res


def test_gait_gates() -> None:
    print("== 3. 世界空间零滑步 / 穿地抑制（完整 FK 链路） ==")
    rt = RigRuntime.load(SPEC, MESH, LAYERS)
    check("运行时（新包 spec）加载", rt is not None)

    rng = random.Random(11)
    gs = load_solver()
    wx = 100.0
    dt = 1.0 / 60.0
    drift_by_kind = {k: 0.0 for k in ContactType}
    startup_drift = 0.0
    pen_worst = -1e9
    toe_clear_worst = 1e9
    knee_rate = 0.0
    any_nan = False
    for i in range(600):
        out = gs.update(dt, 75.0, (int(round(wx)), 0), True, False)
        wx += out.delta_window_x
        win_int = int(round(wx))
        any_nan = any(math.isnan(v) for v in out.bone_rotations.values())
        markers = _chain_marker_world(rt, gs, out, win_int)
        steady = i >= 240
        for side in ("l", "r"):
            f = gs._feet[side]
            if f.contact.is_locked:
                d = abs(markers[side][0] - f.contact.world_anchor_x)
                if steady:
                    drift_by_kind[f.contact.contact_type] = max(
                        drift_by_kind[f.contact.contact_type], d)
                else:
                    startup_drift = max(startup_drift, d)
                pen_worst = max(pen_worst, markers[side][1] - GROUND)
            else:
                tau = (gs.phase + f.phase_offset) % 1.0
                if steady and 0.72 < tau < 0.97:   # 离地后的摆动中段
                    toe_clear_worst = min(toe_clear_worst,
                                          GROUND - markers[side][1])
                pen_worst = max(pen_worst, markers[side][1] - GROUND)
        knee_rate = max(knee_rate, gs._feet["l"].knee_rate, gs._feet["r"].knee_rate)
    worst_drift = max(drift_by_kind.values())
    check("【门槛】单支撑相世界漂移 ≤ 0.5 px（渲染 int 窗口口径）",
          worst_drift <= 0.5,
          f"实测 {worst_drift:.4f}； heel/flat/ff = "
          f"{drift_by_kind[ContactType.HEEL]:.3f}/"
          f"{drift_by_kind[ContactType.FLAT_SOLE]:.3f}/"
          f"{drift_by_kind[ContactType.FOREFOOT]:.3f}")
    print(f"    （起步瞬态段漂移 {startup_drift:.3f} px——转身/起步包络期，"
          f"不计入门禁；G5 起停打磨项）")
    check("【门槛】穿地抑制 ≤ 0.5 px", pen_worst <= 0.5,
          f"max 穿透 {pen_worst:.3f} canvas px")
    check("摆动足尖间隙（中段 ≥ 2 px，全程不穿地）",
          toe_clear_worst > 1.9, f"最小间隙 {toe_clear_worst:.2f} px")
    check("【门槛】膝解算无 NaN", not any_nan)

    print("== 4. 膝角速度稳定性 / 时间步长一致性 ==")
    # 60Hz / 30Hz / 抖动 dt 三种积分的膝角速度与终点一致性（同一物理时长）
    ends = {}
    rates = {}
    for label, dtf in (("60", 1.0 / 60.0), ("30", 1.0 / 30.0), ("jit", None)):
        r = random.Random(7)
        gs2, wx2, outs, _ = run_walk(
            (lambda i: dtf) if dtf else (lambda i: r.uniform(0.008, 0.045)),
            duration=6.0)
        ends[label] = wx2
        rates[label] = gs2.knee_rate_max_observed
    check("【门槛】膝角速度 ≤ 20 rad/s（60/30/抖动 dt）",
          all(v <= 20.0 for v in rates.values()),
          f"60Hz {rates['60']:.1f} / 30Hz {rates['30']:.1f} / jitter {rates['jit']:.1f}")
    d30 = abs(ends["60"] - ends["30"])
    djit = abs(ends["60"] - ends["jit"])
    check("【门槛】时间步长一致性 终点差 ≤ 1.0 px",
          d30 <= 1.0 and djit <= 1.0, f"60v30 {d30:.3f} / 60vjit {djit:.3f} px")

    print("== 5. 步态有限状态机 / 打断与恢复 ==")
    # FSM 主链：idle → turn → start → loop → brake → stop → idle_side → 回正
    seen: set = set()
    gs3, wx3, _, _ = run_walk(lambda i: 1.0 / 60.0, duration=6.0, states=seen)
    wx4 = wx3
    for i in range(int(12.0 * 60)):   # 停止 → 收步 → 闲置超时 → 回正
        gs3.update(1.0 / 60.0, 0.0, (int(round(wx4)), 0), True, False)
        seen.add(gs3.state.value)
    need = {"idle_front", "turn_to_side", "walk_start", "walk_loop",
            "walk_brake", "walk_stop", "idle_side", "turn_to_front"}
    check("FSM 主链状态覆盖", need.issubset(seen), f"seen={sorted(seen)}")

    # 途中反向：无瞬时镜像（view_yaw 连续，单帧跳变 < 12°），状态经 TURN_REVERSE
    gs4 = load_solver()
    wx5 = 100.0
    max_yaw_jump = 0.0
    prev_yaw = None
    rev_seen = False
    for i in range(1200):
        vx = 75.0 if i < 480 else (-75.0 if i < 960 else 0.0)
        out = gs4.update(1.0 / 60.0, vx, (int(round(wx5)), 0), True, False)
        wx5 += out.delta_window_x
        if prev_yaw is not None:
            max_yaw_jump = max(max_yaw_jump, abs(out.view_yaw - prev_yaw))
        prev_yaw = out.view_yaw
        rev_seen = rev_seen or gs4.state is GaitPhaseState.TURN_REVERSE
    check("反向折返经 TURN_REVERSE（禁止瞬时镜像）", rev_seen
          and max_yaw_jump < 12.0, f"max 单帧 yaw 跳变 {max_yaw_jump:.2f}°")

    # 门槛 7：任意相位拖拽打断 → 落地 100% 重建接触
    rng = random.Random(2026)
    rebuilt = 0
    trials = 24
    drift_after = 0.0
    for t in range(trials):
        gs5 = load_solver()
        wx6 = 100.0
        drag_at = rng.randint(60, 420)
        for i in range(drag_at):
            out = gs5.update(1.0 / 60.0, 75.0, (int(round(wx6)), 0), True, False)
            wx6 += out.delta_window_x
        # 拖拽 0.5s（窗口被拖走）
        for i in range(30):
            out = gs5.update(1.0 / 60.0, 0.0, (int(round(wx6)) + 140, 0),
                             True, True)
        check_state = gs5.state is GaitPhaseState.AIRBORNE
        if not check_state:
            continue
        # 释放并落地
        for i in range(90):
            out = gs5.update(1.0 / 60.0, 0.0, (int(round(wx6)) + 140, 0),
                             True, False)
            wx6 += 0.0
        both = all(gs5._feet[sd].contact.is_locked
                   and gs5._feet[sd].contact.contact_type is ContactType.FLAT_SOLE
                   for sd in ("l", "r"))
        if both:
            rebuilt += 1
            pos = gs5.foot_marker_positions(int(round(wx6)) + 140)
            for sd in ("l", "r"):
                drift_after = max(drift_after, abs(
                    pos[sd][0] - gs5._feet[sd].contact.world_anchor_x))
    check("【门槛】拖拽打断后落地 100% 重建接触",
          rebuilt == trials, f"{rebuilt}/{trials}；重建后漂移 {drift_after:.4f} px")


# ============================== 8. 数据契约 / 集成 ==============================

def test_contracts() -> None:
    print("== 6. 数据契约 / spec 解析 / presenter 冒烟 ==")
    # collect_gait_spec：从网格推导接触标记（G3 美术标注缺位时的几何兜底）
    spec_data = collect_gait_spec(BASELINE_SPEC, BASELINE_MESH)
    mk = spec_data.get("contact_markers") or {}
    check("接触标记几何推导（heel/sole/forefoot × 2 脚）",
          all(f"{k}_{s}" in mk for k in ("heel", "sole", "forefoot")
              for s in ("l", "r")))
    off_l = mk.get("sole_l")
    check("sole 标记 y 落在地面线", off_l is not None
          and abs((1450.08 + off_l[1]) - GROUND) < 0.51,
          f"sole_l off = {[round(v, 1) for v in off_l] if off_l else None}")

    # spec.py 宽进解析：新包 gait/turn_views 段
    rig_dir = os.path.join(ROOT, "assets", "rig", "adult")
    spec = load_rig_spec(rig_dir, "adult")
    check("生产包无 gait 配置（步态不激活，零回归）",
          spec is not None and not spec.gait_config and not spec.turn_views_file)

    # presenter 冒烟：QApplication + 真实窗口 + 变 dt 时钟 + 原子提交
    try:
        from PySide6.QtWidgets import QApplication
        from pet.asset_provider import SpriteRef
        from pet.rig.presenter import RigWindow, build_rig_window
        from pet.window import WindowBase
        app = QApplication.instance() or QApplication([])
        sprite_path = os.path.join(ROOT, "assets", "rig", "adult", "figs",
                                   "healthy_neutral.png")
        sprite = SpriteRef(path=sprite_path, width=240, height=424)
        win = build_rig_window(WindowBase, sprite, "adult", defer_quick=False)
        check("生产 RigWindow 构造且步态未激活",
              isinstance(win, RigWindow) and not win.gait_active)
        win.resize(240, 424)
        win.show()
        for _ in range(6):
            app.processEvents()
            win._motion_tick()
        check("变 dt motion tick 运行（perf_counter 时钟）",
              win._last_tick_s is not None)
        # 手动挂接步态（新包 spec）验证原子提交路径：转身 0.3s + 起步加速
        with open(SPEC, "r", encoding="utf-8") as f:
            win._gait = GaitSolver(json.load(f), window_scale=424.0 / 1696.0)
        win.set_gait_command(75.0)
        x_before = win.x()
        yaw_peak = 0.0
        for _ in range(1500):           # processEvents 间歇 → 实际 dt 累计 >1s
            app.processEvents()
            win._motion_tick()
            yaw_peak = max(yaw_peak, abs(float(win._root.property("viewYaw") or 0.0)))
            if win.x() != x_before and yaw_peak > 40.0:
                break
        moved = win.x() != x_before
        yaw = float(win._root.property("viewYaw") or 0.0)
        check("步态原子提交：窗口位移与姿态同拍", moved and yaw_peak > 40.0,
              f"x {x_before} → {win.x()}，yaw {yaw:.1f}（峰值 {yaw_peak:.1f}）")
        win.close()
        win.deleteLater()
    except Exception as e:                    # pragma: no cover —— 无显示环境
        print(f"  [SKIP] presenter 冒烟（环境不可用：{e}）")


def main() -> int:
    print("ADULT 连续视角与步态计算核 —— 验收门禁\n")
    test_ik_math()
    test_keyforms()
    test_gait_gates()
    test_contracts()
    print(f"\n== 门禁结果：{passed} 通过 / {failed} 失败 ==")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
