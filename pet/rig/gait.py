"""接触锁定与解析双骨 IK 步态引擎（ADULT 连续视角/横向行走，模块 1）。

设计规范：``docs/DEEPSEEK_GLM_TASK_SPEC-ADULT连续视角与步态计算核.md`` §2/§3/§6.1。

坐标系（§2）：骨骼/标记数学在 **Canvas Rig Space**（960×1696，x 右 +，y 下 +）
进行；接触锚点以 **Desktop World Space**（桌面逻辑像素）记录，经窗口位移
``X_win`` 与显示缩放 ``S = H_win / 1696``（256 高下 ≈ 0.150943）换算：

    world_x(canvas_x) = X_win + S * canvas_x        （水平向；x 原点平移由呈现层吸收）
    canvas_x(world_x) = (world_x - X_win) / S

零滑步（§2.2）不从"速度对消"实现，而是**每帧按真实窗口位置反解**接触点画布
目标——从构造上免疫窗口系统整数量化与掉帧：子步内锁定与反解使用同一窗口 x，
锚点世界坐标恒定；帧末目标用 ``wx + delta_window_x`` 重构，与调用方
``win.move(win.x() + out.delta_window_x)`` 的积分严格同值。

几何事实（baseline 实测，2026-09-25，本模块多处设计的依据）：

* 大腿/小腿 ≈ 409 / 241.5 canvas px，静止髋-踝距 649.6 = 伸展率 99.88%
  （近奇异）→ 行走必须动态骨盆下沉。规范 §3.2 的 ``A_dip·sin²(πΦ)`` 是形状
  近似；本实现按"支撑腿伸展率 ≤ r_target(0.985)"逐帧精确解算（vault 策略：
  侧视投影里支撑腿过身体时接近伸直，与静止站姿一致）。88–95% 伸展率导则与
  3.5–5px 抬脚/2px 足尖间隙在该原图比例下不可兼得（95% 恒定下沉会吞掉全部
  抬脚预算），取物理可解的一侧并由验收门槛（穿地/间隙/膝稳定）约束。
* 腿底网格顶点权重为 foot/shin 混合（≈0.46/0.46）→ 接触锁以**骨骼标记**
  （sole/heel/forefoot，随 foot 骨刚体变换）为契约；网格顶点级锁地属 G4
  重权重阶段（美术侧）。``foot_slide_drift_px`` 按限位后骨骼角前向解算真值。
* 基线 angle_clamp（thigh ±25° / knee [−35,10]° / foot ±10°）容不下 IK 步态
  解（heel-strike 需 thigh ≈ −34°、滚动需 foot ±35°、中摆需膝 ≈ −40°）→
  新资产包 ``assets/rig_adult_turn_v1/spec.json`` 放宽腿骨限位（基线冻结不
  动）；求解器读取任意限位并在规划期自适应（抬脚上限按膝限位推导），饱和时
  通过 QA 漂移通道上报而非静默漂移。

摆动相三段式（§3.3 的落地细化）：

1. **趾离地（toe-off）** τ∈[stance, stance+0.06)：前掌标记保持世界锚定，
   足仰角 35°→5°（脚跟继续抬起、踝位升高）——物理上真实步态的滚动延续，
   同时消除"摆动早期足尖穿地"（若直接从推蹬位起跳贝塞尔，前掌在 u<0.1 内
   位于地面之下 ~21 canvas px）。
2. **贝塞尔摆动**：三次贝塞尔（P2 与 P3 同高 → 软着陆 ``dY/du|₁=0``），
   仰角 5°→−15°（后跟着地角）。
3. 落地即进入后跟支撑相，锚定后跟世界坐标。

角约定（与 skinned_mesh_item.FK 一致）：y 向下，正角 = 屏幕顺时针；局部角 =
世界角差 − 静止角差。``Analytical2BoneIK.solve`` 返回"竖直向下静止位"的角
（无 rig 上下文的纯双骨数学）；``GaitSolver`` 换算到具体 rig 的骨骼增量
（弧度），呈现层转度后 ``setBonePose``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from pet.rig.motion import spring_from_frequency, spring_step

__all__ = [
    "GaitPhaseState", "ContactType", "FootContactLock", "GaitOutputs",
    "Analytical2BoneIK", "GaitSolver", "collect_gait_spec",
]

# ---- 任务规范常数（§3/§5/§7；可在 spec_data["gait"] 覆盖）----
SOFT_REACH_EPS_PX = 0.5          # 伸展安全容差 ε（§3.1）
SOFT_REACH_D0_PX = 6.0           # 软收缩渐近尺度 d0
HEEL_STRIKE_DEG = -15.0          # 后跟着地仰角（§3.3）
FOREFOOT_ROLL_DEG = 35.0         # 前掌蹬地仰角
TOE_OFF_END_DEG = 18.0           # 趾离地段结束仰角（防摆动早期穿地，见模块 docstring）
REF_DIP_RATE_PX_S = 120.0       # reference_curves：骨盆高度目标最大变化速率（画布 px/s）
REF_MAX_EXTENSION = 0.996       # reference_curves：支撑腿最大伸展率（伸直处 IK 奇异，脚跟一抬膝角突跳）
TOE_OFF_SPAN = 0.10              # 趾离地段占整周期比例（过短会使前掌长臂以 >20rad/s 甩动膝角）
STANCE_RATIO = 0.60              # 支撑相占比
DEFAULT_STRIDE_HZ = 1.25         # 步频初值（§3.4）
DEFAULT_WALK_SPEED_PX_S = 75.0   # 256 尺度世界速度
SWING_LIFT_PX = 4.2              # 256 尺度抬脚高度（规范 3.5–5.0 区间）
MAX_SUBSTEP_S = 0.005            # 相位/事件子步上限（§5.1 自适应分步）
MAX_DT_S = 0.25                  # 单帧 dt 钳制（防后台恢复巨帧）
TURN_DURATION_S = 0.30           # 转身时长（计划 0.25–0.35s）
START_RAMP_S = 0.45              # 起步包络
BRAKE_RAMP_S = 0.35              # 刹车斜坡
LANDING_CROUCH_S = 0.15          # 落地屈膝缓冲（§3.5）
IDLE_FRONT_TIMEOUT_S = 8.0       # IDLE_SIDE 闲置回正超时
STANCE_EXT_TARGET = 0.975        # 支撑腿伸展率上限（vault，见模块 docstring；
                                 # 0.975 相对 0.985 多出的下沉量是低通滞后期的软钳余量）
DIP_SMOOTH_TAU_S = 0.04          # 下沉低通（吸收 sqrt 拐点与急缓冲突）
SWAY_AMP_PX = 1.5                # 256 尺度横向重心摇摆幅
FOOT_TRACK_SEP_PX = 18.0         # 256 尺度双脚印迹中线间距（防前后脚投影重叠）
KNEE_RATE_LIMIT_RAD_S = 20.0     # §7 膝角速度门槛（QA 监测）
TELEPORT_JUMP_PX = 40.0          # 单拍窗口跳变阈值（无拖拽标志的瞬移 → 重锚）


class GaitPhaseState(Enum):
    IDLE_FRONT = "idle_front"
    TURN_TO_SIDE = "turn_to_side"
    WALK_START = "walk_start"
    WALK_LOOP = "walk_loop"
    WALK_BRAKE = "walk_brake"
    WALK_STOP = "walk_stop"
    WALK_PARK = "walk_park"
    TURN_TO_FRONT = "turn_to_front"
    IDLE_SIDE = "idle_side"
    TURN_REVERSE = "turn_reverse"
    AIRBORNE = "airborne"
    LANDING = "landing"


class ContactType(Enum):
    NONE = 0
    HEEL = 1
    FLAT_SOLE = 2
    FOREFOOT = 3


@dataclass
class FootContactLock:
    is_locked: bool = False
    contact_type: ContactType = ContactType.NONE
    world_anchor_x: float = 0.0
    world_anchor_y: float = 0.0
    initial_canvas_x: float = 0.0
    initial_canvas_y: float = 0.0


@dataclass
class GaitOutputs:
    delta_window_x: float                        # 驱动窗口的世界位移（逻辑 px）
    view_yaw: float                              # 连续视角（带符号，-45..+45 度）
    pelvis_offset: Tuple[float, float]           # 骨盆位移（画布 px，root_hip 平移）
    bone_rotations: Dict[str, float]             # 骨骼局部旋转增量（弧度）
    left_foot_contact: ContactType
    right_foot_contact: ContactType
    foot_slide_drift_px: float                   # 本帧接触点世界漂移（QA 监测）
    # 骨骼局部平移（画布 px）：行走时双腿整体后移（leg_shift_px），不改任何角度
    bone_offsets: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# ============================ 闭式 2-Bone IK（§3.1） ============================


class Analytical2BoneIK:
    """闭式解析平面双骨求解器。

    约定：y 向下；返回角为**相对竖直向下静止位**的弧度（顺时针正），调用方按
    rig 静止角换算骨骼增量：

    * ``hip_angle``：大腿世界角 − π/2（= γ + σ·α − π/2，§3.1 步骤 3）。
    * ``knee_angle``：小腿相对大腿折角（−σ·(π − β)，β 为膝内角；σ=+1 时膝
      折向髋→踝连线负 x 侧，即行进 +x 的"膝后折"）。
    * ``ankle_angle``：足底保持目标仰角所需脚踝补偿（§3.1 步骤 4，θ_pelvis=0）。

    过度拉伸软收缩（§3.1 步骤 1 的连续性修正）：规范原式
    ``D_c = M − d0·exp(−(D−M)/d0)`` 在 D=M 处有 d0 幅度位置跳变（违背其自身
    "平滑奇异点"目标），实现取 ``D_c = M − d0·(1 − exp(−(D−M)/d0))``：边界处
    与未钳制值恒等、一阶导连续、渐近收敛 M − d0，同样消除奇异点。
    """

    EPS_PX = SOFT_REACH_EPS_PX
    D0_PX = SOFT_REACH_D0_PX

    @staticmethod
    def soft_clamp_distance(d: float, d_max: float, eps: float = EPS_PX,
                            d0: float = D0_PX) -> float:
        """把距离 d 平滑钳到 (d_max − eps) 以内（C1 连续，见类 docstring）。"""
        m = d_max - eps
        if d <= m:
            return d
        return m - d0 * (1.0 - math.exp(-(d - m) / d0))

    @staticmethod
    def solve(
        hip: np.ndarray,
        target_ankle: np.ndarray,
        l1: float,
        l2: float,
        bend_direction: int = 1,
        target_foot_angle: float = 0.0,
        exact_within: float = 0.0,
    ) -> Tuple[float, float, float]:
        """返回 ``(hip_angle_rad, knee_angle_rad, ankle_angle_rad)``。

        exact_within：髋踝距 ≤ 该值时不做软收缩（静止站姿必须精确复原；侧身腿静止时已达
        全长 99.99%，落在软收缩带内会被解成膝盖多弯 4°）。"""
        hip = np.asarray(hip, np.float64)
        tgt = np.asarray(target_ankle, np.float64)
        dx = float(tgt[0] - hip[0])
        dy = float(tgt[1] - hip[1])
        d = math.hypot(dx, dy)
        d_min = abs(l1 - l2) + Analytical2BoneIK.EPS_PX
        L = l1 + l2
        if exact_within > 0.0 and exact_within > L - Analytical2BoneIK.EPS_PX:
            # 静止距离已落在软收缩带内：≤ 静止距离精确求解，之上从静止距离 C1 连续地渐近
            # 全长（旧软收缩从更低的值起算，交界处髋踝距突降 ~0.45 px → 近直腿膝角一帧跳 12°）
            e = min(exact_within, L - 1e-6)
            span = L - e
            d_c = d if d <= e else e + span * (1.0 - math.exp(-(d - e) / span))
        elif d <= exact_within:
            d_c = d
        else:
            d_c = Analytical2BoneIK.soft_clamp_distance(d, L)
        if d < d_min:                       # 过度折叠对称软钳（防膝翻折）
            d_c = d_min + Analytical2BoneIK.D0_PX * (
                1.0 - math.exp(-(d_min - d) / Analytical2BoneIK.D0_PX))
        if d_c <= 1e-9:
            gamma, alpha, bend = math.pi / 2.0, 0.0, 0.0
        else:
            cos_beta = (l1 * l1 + l2 * l2 - d_c * d_c) / (2.0 * l1 * l2)
            cos_alpha = (l1 * l1 + d_c * d_c - l2 * l2) / (2.0 * l1 * d_c)
            bend = math.pi - math.acos(min(1.0, max(-1.0, cos_beta)))
            alpha = math.acos(min(1.0, max(-1.0, cos_alpha)))
            gamma = math.atan2(dy, dx)
        sigma = 1.0 if bend_direction >= 0 else -1.0
        hip_angle = gamma + sigma * alpha - math.pi / 2.0
        knee_angle = -sigma * bend
        ankle_angle = target_foot_angle - (hip_angle + knee_angle)
        return hip_angle, knee_angle, ankle_angle


# ============================ 内部数据结构 ============================


@dataclass
class _LegModel:
    """单腿静态几何（画布 px）+ 限位（弧度）。"""

    side: str
    hip_bone: str
    knee_bone: str
    foot_bone: str
    hip_rest: np.ndarray
    knee_rest: np.ndarray
    ankle_rest: np.ndarray
    l1: float
    l2: float
    thigh_rest_angle: float                     # 世界静止角（rad，y 下 CW+）
    shin_rest_angle: float
    clamp_lo: Tuple[float, float, float]
    clamp_hi: Tuple[float, float, float]
    marker_heel: np.ndarray                     # 相对踝静止偏移（画布 px）
    marker_sole: np.ndarray
    marker_forefoot: np.ndarray

    @property
    def d_rest(self) -> float:
        return float(np.linalg.norm(self.ankle_rest - self.hip_rest))


@dataclass
class _FootState:
    """单脚运行态。摆动控制点 x 存世界系、y 存画布系（窗口移动时轨迹不漂）。"""

    contact: FootContactLock = field(default_factory=FootContactLock)
    phase_offset: float = 0.0
    track_center_canvas: float = 0.0
    ankle_now: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pitch_now: float = 0.0
    heel_entry_pitch: float = 0.0               # 后跟支撑进入仰角（连续性）
    swing_planned: bool = False
    swing_p0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    swing_p3: np.ndarray = field(default_factory=lambda: np.zeros(2))
    swing_lift: float = 0.0                     # 抬脚 bump 峰值（画布 px）
    swing_pitch0: float = 0.0                   # 摆动起始仰角（连续性）
    land_world_x: float = 0.0
    land_fix: float = 0.0                       # 刹车中途重规划的落点修正（世界 px，已计入 land_world_x）
    land_fix_b0: float = 0.0                    # 重规划时的轨迹进度（修正从此处渐入，位姿不跳）
    land_pitch: float = 0.0
    applied: Optional[Tuple[float, float, float]] = None
    swing_th0: Optional[float] = None           # 参考曲线摆动：起始大腿/膝局部角（rad）
    swing_kn0: float = 0.0
    swing_u0: float = 0.0                       # 规划时的摆动进度（起步可在摆动半途切入）
    swing_last_wx: Optional[float] = None       # 参考轨迹世界 x（单调前进约束）
    knee_prev: Optional[float] = None
    knee_rate: float = 0.0
    settle_active: bool = False            # 静止放平进行中
    settle_t: float = 0.0
    settle_pitch: float = 0.0


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _rotated(off: Tuple[float, float], ang: float) -> Tuple[float, float]:
    c, s = math.cos(ang), math.sin(ang)
    return (c * off[0] - s * off[1], s * off[0] + c * off[1])


def _unwrap(cur: float, prev: float) -> float:
    """相位回退修正（跨 1 环绕）：返回考虑环绕的上一相位。"""
    if prev > cur:
        prev -= 1.0
    return prev


# ============================ 规格收集（数据契约入口） ============================


def collect_gait_spec(spec_file: str,
                      mesh_file: Optional[str] = None) -> Dict:
    """从 baseline spec.json（+ 可选 mesh_data.json）构建 GaitSolver 输入。

    接触标记优先取 spec["contact_markers"]（G3 起美术标注），否则从腿层网格
    几何推导：鞋底线 = 腿层最大 y 行，heel/forefoot = 底线 x 两端，sole = 中点。
    """
    import json
    import os
    with open(spec_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    spec: Dict = dict(data)
    if mesh_file and os.path.isfile(mesh_file):
        with open(mesh_file, "r", encoding="utf-8") as f:
            mesh = json.load(f)
        markers = dict(spec.get("contact_markers") or {})
        layers = {l["id"]: l for l in mesh.get("layers", [])}
        for side in ("l", "r"):
            if any(k.endswith(f"_{side}") for k in markers):
                continue
            leg = layers.get(f"leg_{side}")
            ankle = _find_joint(data, f"foot_{side}")
            if leg is None or ankle is None:
                continue
            v = np.asarray(leg["vertices"], np.float64)
            y_max = float(v[:, 1].max())
            bottom = v[v[:, 1] > y_max - 3.0]
            heel_x = float(bottom[:, 0].min())
            fore_x = float(bottom[:, 0].max())
            sole_x = 0.5 * (heel_x + fore_x)
            markers[f"heel_{side}"] = [heel_x - ankle[0], y_max - ankle[1]]
            markers[f"sole_{side}"] = [sole_x - ankle[0], y_max - ankle[1]]
            markers[f"forefoot_{side}"] = [fore_x - ankle[0], y_max - ankle[1]]
        spec["contact_markers"] = markers
    return spec


def _find_joint(spec_data: Dict, bone: str) -> Optional[np.ndarray]:
    sk = spec_data.get("skeleton") or {}
    ref = (sk.get("source_reference") or {}).get("image_size_px") or [960, 1696]
    w, h = float(ref[0]), float(ref[1])
    for b in sk.get("bones", []):
        if b.get("bone_name") == bone:
            return np.array([float(b["joint_pos"][0]) * w,
                             float(b["joint_pos"][1]) * h])
    return None


# ============================ GaitSolver（§6.1 接口） ============================


class GaitSolver:
    """接触锁定 + 解析 IK 的双足步态求解器。

    用法（呈现层每渲染拍一次，dt 取 ``time.perf_counter()`` 差值）::

        solver = GaitSolver(spec_data, window_scale=256.0 / 1696.0)
        out = solver.update(dt, desired_vx, (win.x(), win.y()),
                            is_grounded=True, is_dragged=False)
        win.move(round(win.x() + out.delta_window_x), win.y())   # 与姿态同拍提交
    """

    def __init__(self, spec_data: dict, window_scale: float = 0.150943):
        self.scale = float(window_scale)
        self.state = GaitPhaseState.IDLE_FRONT
        self.view_yaw = 0.0
        self.phase = 0.0
        self.left_contact = FootContactLock()
        self.right_contact = FootContactLock()

        sk = spec_data.get("skeleton") or {}
        ref = (sk.get("source_reference") or {}).get("image_size_px") or [960, 1696]
        self.img_w, self.img_h = float(ref[0]), float(ref[1])
        self.ground_y = float(spec_data.get("ground_anchor_y_px",
                                            0.948 * self.img_h))
        joints: Dict[str, np.ndarray] = {}
        clamps: Dict[str, Tuple[float, float]] = {}
        for b in sk.get("bones", []):
            name = str(b.get("bone_name", ""))
            joints[name] = np.array([float(b["joint_pos"][0]) * self.img_w,
                                     float(b["joint_pos"][1]) * self.img_h])
            c = b.get("angle_clamp") or [-1e9, 1e9]
            lo, hi = float(c[0]), float(c[1])
            clamps[name] = (min(lo, hi), max(lo, hi))

        self.pelvis_rest = joints.get(
            "root_hip", np.array([self.img_w * 0.5, self.img_h * 0.46]))
        markers = spec_data.get("contact_markers") or {}

        def marker(side: str, kind: str) -> np.ndarray:
            v = markers.get(f"{kind}_{side}")
            if v:
                return np.array([float(v[0]), float(v[1])], np.float64)
            # 缺省：踝正下方至地面，heel/forefoot 各让 42 画布 px（≈6.3 逻辑 px）
            drop = self.ground_y - joints[f"foot_{side}"][1]
            width = 42.0
            if kind == "heel":
                return np.array([-width, drop])
            if kind == "forefoot":
                return np.array([width, drop])
            return np.array([0.0, drop])

        self.legs: Dict[str, _LegModel] = {}
        for side in ("l", "r"):
            hip, knee, ankle = (joints[f"upper_leg_{side}"],
                                joints[f"lower_leg_{side}"],
                                joints[f"foot_{side}"])
            th_cl = clamps.get(f"upper_leg_{side}", (-1e9, 1e9))
            kn_cl = clamps.get(f"lower_leg_{side}", (-1e9, 1e9))
            ft_cl = clamps.get(f"foot_{side}", (-1e9, 1e9))
            self.legs[side] = _LegModel(
                side=side,
                hip_bone=f"upper_leg_{side}",
                knee_bone=f"lower_leg_{side}",
                foot_bone=f"foot_{side}",
                hip_rest=hip, knee_rest=knee, ankle_rest=ankle,
                l1=float(np.linalg.norm(knee - hip)),
                l2=float(np.linalg.norm(ankle - knee)),
                thigh_rest_angle=float(math.atan2(knee[1] - hip[1], knee[0] - hip[0])),
                shin_rest_angle=float(math.atan2(ankle[1] - knee[1], ankle[0] - knee[0])),
                clamp_lo=(math.radians(th_cl[0]), math.radians(kn_cl[0]),
                          math.radians(ft_cl[0])),
                clamp_hi=(math.radians(th_cl[1]), math.radians(kn_cl[1]),
                          math.radians(ft_cl[1])),
                marker_heel=marker(side, "heel"),
                marker_sole=marker(side, "sole"),
                marker_forefoot=marker(side, "forefoot"),
            )

        g = dict(spec_data.get("gait") or {})
        # +1 bends the knee to screen-left; a right-facing human rig needs -1.
        # Keep the historical branch for existing front-view assets.
        self.knee_bend = -1 if float(g.get("knee_bend_direction", 1)) < 0 else 1
        self.park_feet = bool(g.get("park_feet", False))
        self.stance_extension = float(g.get("stance_extension", STANCE_EXT_TARGET))
        self.lean_degrees = float(g.get("lean_degrees", -1.8))
        # 侧身：裙摆前/后缘跟随最前/最后的大腿摆动（腿在裙下前后摆，裙子不能纹丝不动）
        self.skirt_follow = float(g.get("skirt_follow_gain", 0.0))
        self.skirt_limit = math.radians(float(g.get("skirt_follow_limit_deg", 180)))
        self.skirt_freq = float(g.get("skirt_freq_hz", 0))
        self.skirt_halflife = float(g.get("skirt_halflife_s", .1))
        self._skirt: Dict[str, list] = {}
        #   reference_curves  关节角按正常人步态参考曲线成形（docs/ADULT行走修复-2026-09-29.md §4）：
        #   摆动期 = 参考大腿/膝角的正向运动学轨迹（末段并入规划落点）；支撑期骨盆高度由
        #   领先腿的参考膝角（承重缓冲 → 近伸直）反求，而非"两腿取最差"（旧规则 = 全程半蹲）
        self.ref_curves = bool(g.get("reference_curves", False))
        #   leg_shift_px  行走时两条腿作为整体沿水平方向平移（canvas px，负 = 向后；随行走包络渐入）
        self.leg_shift = float(g.get("leg_shift_px", 0.0))
        #   far_leg_shift_px  远侧（后侧）腿 r 额外水平平移，拉开两腿间距（负 = 向后）
        self.far_leg_shift = float(g.get("far_leg_shift_px", 0.0))
        self.ref_swing_knee_peak = math.radians(float(g.get("ref_swing_knee_peak_deg", 58.0)))
        self.ref_load_knee = math.radians(float(g.get("ref_loading_knee_deg", 16.0)))
        self.ref_mid_knee = math.radians(float(g.get("ref_midstance_knee_deg", 4.0)))
        self._park_order: list[str] = []
        self._park_start: dict = {}
        self._park_step_s = 0.30          # 收脚一步的时长
        self._park_flatten_s = 0.18       # 支撑脚放平（绕当前接触点）
        self._park_glide_s = 0.25         # 身体前移到前脚上方（双脚锚定，不滑）
        self._park_glide_w = 0.0          # 前移总量（世界 px）
        self._park_end = 0.0
        self._park_wx_final = 0.0
        self._park_steps: dict = {}       # side -> (t_start, ankle_start)
        self._park_dur: dict = {}         # side -> step duration (short for small corrections)
        self.stance_ratio = float(g.get("stance_ratio", STANCE_RATIO))
        self.stride_hz = float(g.get("frequency_hz", DEFAULT_STRIDE_HZ))
        self._base_stride_hz = self.stride_hz
        self.adaptive_cadence = bool(g.get("adaptive_cadence", False))
        self.speed_target = float(g.get("speed_world_px_s", DEFAULT_WALK_SPEED_PX_S))
        self.swing_lift_world = float(g.get("swing_lift_world_px", SWING_LIFT_PX))
        self.track_sep_world = float(g.get("foot_track_sep_world_px", FOOT_TRACK_SEP_PX))
        self.idle_timeout = float(g.get("idle_front_timeout_s", IDLE_FRONT_TIMEOUT_S))
        self.turn_duration = float(g.get("turn_duration_s", TURN_DURATION_S))
        self.knee_rate_limit = float(g.get("knee_rate_limit_rad_s", KNEE_RATE_LIMIT_RAD_S))
        self.speed_cap = float(g.get("speed_cap_world_px_s", 220.0))
        # 侧身骨骼（assets/rig_adult_walk_v1，四分之三侧视）可选项；缺省 = 正面旧行为：
        #   per_side_ground  远侧脚站得更靠后、屏幕上更高 → 每只脚落回自己的静止鞋底线
        #   sway_world_px    横向重心摇摆（侧视下是纵深方向 → 0）
        #   arm_swing_deg    上臂按步态相位摆动的幅度（与同侧腿反相，×包络）；0 = 不输出手臂
        #   forearm_bend_deg 前臂在前摆时的附加屈曲
        self.sway_world = float(g.get("sway_world_px", SWAY_AMP_PX))
        self.arm_swing = math.radians(float(g.get("arm_swing_deg", 0.0)))
        self.forearm_bend = math.radians(float(g.get("forearm_bend_deg", 0.0)))
        #   forearm_base_deg 行走时肘部的常驻微屈（手臂不再僵直）
        #   arm_phase_lag    手臂相位滞后于腿（周期比例），摆动更松弛
        #   hand_follow      手腕随前臂反向跟随的比例
        #   track_offset_px  行走时落脚点整体后移（canvas px，负 = 向后）：行走下蹲 + 前屈膝
        #                    使大腿平均前倾约 10°，腿看起来在身体前方
        self.forearm_base = math.radians(float(g.get("forearm_base_deg", 0.0)))
        self.arm_phase_lag = float(g.get("arm_phase_lag", 0.0))
        self.hand_follow = float(g.get("hand_follow", 0.0))
        #   wrist_freq_hz / wrist_halflife_s  >0：手腕改为被动摆（二阶弹簧，motion.spring_from_frequency）：
        #                    手的世界角追随"前臂世界角 + 反向小补偿"，滞后即惯性甩动（人行走时腕部
        #                    放松，屈伸总幅约 10–15°）；wrist_limit_deg 限位（局部角，±）
        self.wrist_freq = float(g.get("wrist_freq_hz", 0.0))
        self.wrist_halflife = float(g.get("wrist_halflife_s", 0.07))
        self.wrist_limit = math.radians(float(g.get("wrist_limit_deg", 20.0)))
        self._wrist: Dict[str, list] = {}       # side → [手世界角, 角速度]
        #   far_arm_scale    远侧手臂（_r，身体后方）摆幅比例：四分之三视角下同样的屈肘前摆会显得前伸
        self.far_arm_scale = float(g.get("far_arm_scale", 1.0))
        self.track_offset = float(g.get("track_offset_px", 0.0))
        #   toe_off_end_deg  趾离地段结束仰角（缺省 18°：从 35° 回落）。侧身：回落会让踝下沉、
        #                    后腿在离地前一瞬被拉直（膝角一帧回弹）；真实步态离地时脚跟继续抬起
        self.toe_off_end_deg = float(g.get("toe_off_end_deg", TOE_OFF_END_DEG))
        #   torso_lean_deg   行走时上身（spine 骨，正 = 顺时针 = 向前）额外前倾，×包络——上身压在
        #                    腿上方，不再显得腿走在身体前面
        self.torso_lean = math.radians(float(g.get("torso_lean_deg", 0.0)))
        #   brake_linear_s   >0：刹车改为线性减速到 0（秒）；缺省指数衰减（BRAKE_RAMP_S
        #                    时间常数，120 px/s 需 ~1.4 s、三步才停，侧身行走显得拖沓）
        self.brake_linear = float(g.get("brake_linear_s", 0.0))
        #   dip_geometric    >0：骨盆下沉只按几何可达性（支撑腿伸展率 ≤ r_target），不乘步态
        #                    包络——起步/刹车/停步时包络与双脚间距不同步，乘包络会让骨盆
        #                    提前抬升、支撑腿够不着锚点（软收缩 → 脚滑 1–1.5 px）
        self.dip_geometric = float(g.get("dip_geometric", 0.0)) > 0.5
        #   track_from_rest  >0：脚印迹中心 = 各脚静止鞋底 x（侧视），而非双脚中线 ± 间距
        self.track_from_rest = float(g.get("track_from_rest", 0.0)) > 0.5
        self._brake_v0 = 0.0
        self._ground_side: Dict[str, float] = {}
        if g.get("per_side_ground", 0.0) > 0.5:
            for side, leg in self.legs.items():
                self._ground_side[side] = float(leg.ankle_rest[1] + leg.marker_sole[1])
        # 支撑子相位边界（§3.3 比例外推到可配 stance_ratio）
        self.heel_end = 0.25 * self.stance_ratio
        self.flat_end = 0.75 * self.stance_ratio
        # reference_curves: toe leaves at ~62% like a human; a 10% anchored toe drags the thigh back
        self.toe_off_span = 0.03 if self.ref_curves else TOE_OFF_SPAN
        self.toe_off_end = self.stance_ratio + self.toe_off_span

        # ---- 运行态 ----
        self._t = 0.0                     # 状态机局部时钟
        self._velocity = 0.0
        self._velocity_cmd = 0.0
        self._env = 0.0
        self._target_yaw = 0.0
        self._yaw_from = 0.0
        self._turn_dur = self.turn_duration
        self._post_turn_state = GaitPhaseState.IDLE_FRONT
        self._reverse_total = 0.0
        self._decel_dur = 0.0
        self._dip = 0.0
        self._dip_raw = 0.0
        self._sway = 0.0
        self._pelvis_rot = 0.0
        self._feet: Dict[str, _FootState] = {
            "l": _FootState(phase_offset=0.0), "r": _FootState(phase_offset=0.5)}
        self._idle_timer = 0.0
        self._wx = 0.0                    # 当前子步窗口世界 x（所有换算统一用它）
        self._window_initialized = False
        self._knee_rate_max = 0.0
        self._initialize_feet()

    # ---------------- 初始化：静止站姿双脚锚定 ----------------

    def _initialize_feet(self) -> None:
        for side, foot in self._feet.items():
            leg = self.legs[side]
            foot.track_center_canvas = self._track_center(side)
            foot.ankle_now = leg.ankle_rest.copy()
            foot.pitch_now = 0.0
            foot.applied = (0.0, 0.0, 0.0)
            self._lock_contact(side, ContactType.FLAT_SOLE)

    def _track_center(self, side: str) -> float:
        """脚印迹中心（画布 x）：绕双脚步态中线对称布置。

        原地站姿的脚距（基线 ~16 逻辑 px）不叠加印迹间距——以双脚步态中线
        （两脚 sole 静止中点）为基准放 ±sep/2，保证两腿 IK 负载对称
        （以踝静止位为基准会把原有站位宽度算双份，单腿 dx 超 170 canvas px，
        顶死膝/脚限位）。
        """
        l_sole = float(self.legs["l"].ankle_rest[0] + self.legs["l"].marker_sole[0])
        r_sole = float(self.legs["r"].ankle_rest[0] + self.legs["r"].marker_sole[0])
        if self.track_from_rest:
            # 侧视：每只脚围绕自己的静止站位摆动（四分之三视角下远侧脚在画面上本就
            # 靠前约 100 canvas px）→ 停步时双脚回到静止站姿，衔接转身片段首帧
            return l_sole if side == "l" else r_sole
        mid = 0.5 * (l_sole + r_sole)
        half = self.track_sep_world / self.scale * 0.5
        return mid - half if side == "l" else mid + half

    # ---------------- 世界↔画布换算（统一经 _wx） ----------------

    def _c2w(self, canvas_x: float) -> float:
        return self._wx + self.scale * canvas_x

    def _w2c(self, world_x: float) -> float:
        return (world_x - self._wx) / self.scale

    def _marker_offset(self, side: str, kind: ContactType) -> np.ndarray:
        leg = self.legs[side]
        return {ContactType.HEEL: leg.marker_heel,
                ContactType.FLAT_SOLE: leg.marker_sole,
                ContactType.FOREFOOT: leg.marker_forefoot,
                ContactType.NONE: leg.marker_sole}[kind]

    def _marker_canvas(self, side: str, kind: ContactType,
                       ankle: np.ndarray, pitch: float) -> np.ndarray:
        off = self._marker_offset(side, kind)
        rx, ry = _rotated((float(off[0]), float(off[1])), pitch)
        return ankle + np.array([rx, ry])

    def _lock_contact(self, side: str, kind: ContactType) -> None:
        """以当前规划位姿捕获标记世界坐标并锁定（支点转移按计划记录）。"""
        foot = self._feet[side]
        m = self._marker_canvas(side, kind, foot.ankle_now, foot.pitch_now)
        foot.contact.is_locked = True
        foot.contact.contact_type = kind
        foot.contact.initial_canvas_x = float(m[0])
        foot.contact.initial_canvas_y = float(m[1])
        foot.contact.world_anchor_x = self._c2w(float(m[0]))
        foot.contact.world_anchor_y = float(m[1])   # 垂直向窗口不动，画布 y 即锚 y

    def _unlock(self, side: str) -> None:
        c = self._feet[side].contact
        c.is_locked = False
        c.contact_type = ContactType.NONE

    # ---------------- 支撑/摆动目标解算（§3.3） ----------------

    def _stance_target(self, side: str, tau: float) -> Tuple[np.ndarray, float]:
        """支撑相：锚定世界坐标按当前窗口反解 → (踝画布目标, 足仰角)。"""
        foot = self._feet[side]
        c = foot.contact
        kind = c.contact_type
        if kind is ContactType.HEEL:
            u = min(1.0, tau / max(self.heel_end, 1e-6))
            pitch = foot.heel_entry_pitch * 0.5 * (1.0 + math.cos(math.pi * u))
        elif kind is ContactType.FOREFOOT:
            u = min(1.0, (tau - self.flat_end)
                    / max(self.stance_ratio - self.flat_end, 1e-6))
            pitch = math.radians(FOREFOOT_ROLL_DEG) * _smoothstep(u)
        else:
            pitch = 0.0
        off = self._marker_offset(side, kind)
        rx, ry = _rotated((float(off[0]), float(off[1])), pitch)
        anchor_x = self._w2c(c.world_anchor_x)
        return np.array([anchor_x - rx, c.world_anchor_y - ry]), pitch

    def _plan_swing(self, side: str, velocity: float) -> None:
        """摆动规划：落点（印迹中心 + 前扫半程，世界系）+ 贝塞尔控制点。

        抬脚高度按膝限位自适应上限（模块 docstring"几何事实"）。
        """
        foot = self._feet[side]
        leg = self.legs[side]
        direction = 1.0 if velocity >= 0 else -1.0
        swing_time = max(1.0 - self.toe_off_end, 0.0) / max(self.stride_hz, 1e-6)
        landing_speed = velocity
        travel = velocity * swing_time
        if self.adaptive_cadence and self.state in (GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP):
            # Starting from v≈0, v*T predicts a step behind the accelerating body.
            ramp = START_RAMP_S if self.state is GaitPhaseState.WALK_START else 0.3
            decay = math.exp(-swing_time / ramp)
            landing_speed = self._velocity_cmd + (velocity - self._velocity_cmd) * decay
            travel = self._velocity_cmd * swing_time + (velocity - self._velocity_cmd) * ramp * (1 - decay)
        elif self.state is GaitPhaseState.WALK_BRAKE:
            # 刹车中身体在减速：按 v·T 预测会让落点超出身体到达处（1.2 Hz 摆动更长，
            # 摆动腿够不着 → 伸展下限一帧压低骨盆 20 px、支撑膝跳 21°）。按实际减速曲线积分。
            if self.brake_linear > 0.0:
                rem = max(self.brake_linear - self._t, 1e-6)
                s = min(swing_time, rem)
                travel = velocity * (s - s * s / (2.0 * rem))
                landing_speed = velocity * max(0.0, 1.0 - swing_time / rem)
            else:
                decay = math.exp(-swing_time / BRAKE_RAMP_S)
                travel = velocity * BRAKE_RAMP_S * (1.0 - decay)
                landing_speed = velocity * decay
        sweep_half_canvas = abs(landing_speed) * self.stance_ratio \
            / self.stride_hz * 0.5 / self.scale
        # 落点以 heel 为 y 基准（后跟着地 = 后跟触地线，§3.3），扫掠对称性以
        # sole 印迹计：ankle = heel_land − R(p)·off_heel 两轴同源，保证着地
        # 锚点恰在地面上（若以 sole 反推踝位，−15° 仰角下 heel 标记低于地面
        # ~11 canvas px，整段支撑相锚点穿地）。
        off_heel = leg.marker_heel
        hx, hy = _rotated((float(off_heel[0]), float(off_heel[1])),
                          math.radians(HEEL_STRIKE_DEG))
        heel_land_x = (foot.track_center_canvas + direction * sweep_half_canvas
                       + float(off_heel[0]) - float(leg.marker_sole[0]))
        if self.track_offset and self.state in (GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP,
                                                GaitPhaseState.WALK_BRAKE, GaitPhaseState.TURN_REVERSE):
            # 行走时落脚点整体后移（随速度渐入；停步最后一步不后移，收步只向前）
            # 高速（跟随 200 px/s）时步幅本就长，后移会让后腿离地前接近伸直 → 按 120/v 缩小
            sp = abs(landing_speed)
            heel_land_x += direction * self.track_offset * min(1.0, sp / 60.0) * min(1.0, 120.0 / max(sp, 1e-6))
        if self.park_feet and self.state in (GaitPhaseState.WALK_BRAKE, GaitPhaseState.WALK_STOP)                 and abs(landing_speed) < 30.0:
            # 停步最后一步：身体随后会前移到仍领先站位的那只脚上方（_plan_park），这一步
            # 直接落到"前移之后的站位"——否则落在自己站位上正好与另一只脚前后重叠（叠鞋）
            other = "r" if side == "l" else "l"
            ahead = float(self._feet[other].ankle_now[0] - self.legs[other].ankle_rest[0])
            heel_land_x += max(0.0, ahead)
        foot.land_pitch = math.radians(HEEL_STRIKE_DEG)
        # 落点须领先摆动期间的窗口位移：规划时刻的"印迹中心 + 半扫"是相对**当前**
        # 身体的，着地时身体已前移 v·T_swing（120 px/s 下 ≈265 canvas px）——不补偿
        # 则每步落在身体后方，支撑相后扫深度翻倍、大腿顶死限位、骨盆过度下沉。
        foot.land_world_x = self._c2w(heel_land_x - hx) + travel   # 踝世界 x
        foot.land_fix, foot.land_fix_b0 = 0.0, 0.0
        # 抬脚上限：膝限位允许的最小髋-踝距（θ_k = −b − shin_rel_rest ≥ clamp_lo）
        shin_rel_rest = leg.shin_rest_angle - leg.thigh_rest_angle
        b_budget = max((-leg.clamp_lo[1] - shin_rel_rest) if self.knee_bend > 0
                       else (leg.clamp_hi[1] + shin_rel_rest), 0.30)
        d_min = math.sqrt(leg.l1 ** 2 + leg.l2 ** 2
                          + 2.0 * leg.l1 * leg.l2 * math.cos(b_budget))
        rise_budget = max(leg.d_rest - d_min - self._dip - 6.0, 3.0)
        foot.swing_lift = min(self.swing_lift_world / self.scale, rise_budget)
        # 端点（x 世界系 / y 画布系）：P0 = 当前踝位，P3 = 落地踝位
        foot.swing_p0 = np.array([self._c2w(float(foot.ankle_now[0])),
                                  float(foot.ankle_now[1])])
        # 后跟标记可高于鞋底标记（侧身鞋：后跟接触点比全掌支点高 7–14 canvas px）：
        # 着地高度按静止鞋底几何放置，全掌放平后 sole 恰在地面线上而非穿地
        heel_above_sole = float(off_heel[1]) - float(leg.marker_sole[1])
        foot.swing_p3 = np.array([foot.land_world_x,
                                  self._ground_side.get(side, self.ground_y)
                                  + heel_above_sole - hy])
        foot.swing_pitch0 = foot.pitch_now
        if foot.applied is not None:
            foot.swing_th0, foot.swing_kn0 = foot.applied[0], foot.applied[1]
            tau = (self.phase + foot.phase_offset) % 1.0
            foot.swing_u0 = min(0.95, max(0.0, (tau - self.toe_off_end) / max(1.0 - self.toe_off_end, 1e-6)))
            foot.swing_last_wx = None
        else:
            foot.swing_th0 = None
        foot.swing_planned = True

    def _swing_ankle(self, side: str, u: float) -> np.ndarray:
        """摆动踝位（u∈[0,1]；x 世界系存储 → 按当前窗口反解画布）。

        轨迹 = 端点零速的 C1 传输（smoothstep 插值 P0→P3）+ ``sin²(πu)·lift``
        抬脚 bump——两端速度为零：与趾离地段锚定零速/落地软着陆速度连续，
        消除接合点膝角速度突跳（单段三次贝塞尔无法同时满足零端点速度与
        内部抬升，见 _plan_swing docstring）。
        """
        foot = self._feet[side]
        s = min(1.0, max(0.0, u))
        blend = s * s * (3.0 - 2.0 * s)
        lift = (math.sin(math.pi * s) ** 2) * foot.swing_lift
        xw = foot.swing_p0[0] + blend * (self._land_x(foot, blend) - foot.swing_p0[0])
        y = (foot.swing_p0[1] + blend * (foot.swing_p3[1] - foot.swing_p0[1])
             - lift)
        planned = np.array([self._w2c(xw), y])
        # only swings that start near toe-off follow the reference (a swing entered halfway
        # has no time left for the flex-extend)
        if not self.ref_curves or foot.swing_th0 is None or s >= 1.0 or foot.swing_u0 > 0.05:
            return planned
        return self._ref_swing_ankle(side, s, planned)

    # ---------------- 参考曲线（正常人步态，§4 of docs/ADULT行走修复） ----------------

    def _leg_fk(self, side: str, ang_th: float, ang_kn: float) -> np.ndarray:
        """局部角 → 踝画布位置（与 _foot_marker_actual_world 同一几何）。"""
        leg = self.legs[side]
        hip = self._hip_canvas(side)
        th = leg.thigh_rest_angle + self._pelvis_rot + ang_th
        sh = leg.shin_rest_angle + self._pelvis_rot + ang_th + ang_kn
        return hip + leg.l1 * np.array([math.cos(th), math.sin(th)])             + leg.l2 * np.array([math.cos(sh), math.sin(sh)])

    def _ref_swing_ankle(self, side: str, s: float, planned: np.ndarray) -> np.ndarray:
        """摆动：大腿单调前摆（≈62% 处略过冲再回收）、膝在 25% 处达峰后伸直；
        以正向运动学得踝目标（IK 原样解回这组角），70% 起平滑并入规划落点。"""
        foot = self._feet[side]
        leg = self.legs[side]
        th1, kn1 = self._ik_angles(side, self._swing_end_planned(side))
        # progress from the moment this swing was planned (walk start enters mid-swing)
        s = (s - foot.swing_u0) / max(1.0 - foot.swing_u0, 1e-6)
        # 大腿：smoothstep 到 1.12 倍行程（≈62%），再回收到 1.0
        if s < 0.62:
            hshape = 1.12 * _smoothstep(s / 0.62)
        else:
            hshape = 1.12 - 0.12 * _smoothstep((s - 0.62) / 0.38)
        th = foot.swing_th0 + (th1 - foot.swing_th0) * hshape
        # 膝：以"弯曲量"表示（相对伸直），起点→峰（25%）→落地值（85%）
        rel = leg.shin_rest_angle - leg.thigh_rest_angle
        b0, b1 = foot.swing_kn0 + rel, kn1 + rel
        # flex speed cap: the rise to the peak takes 25% of the swing; at this cadence a swing
        # starting from a straight knee (walk start) would exceed 20 rad/s (smoothstep peak 1.5x)
        t_rise = 0.25 * (1.0 - foot.swing_u0) * (1.0 - self.toe_off_end) / max(self.stride_hz, 1e-6)
        bpk = max(min(self.ref_swing_knee_peak, b0 + 14.0 * t_rise / 1.5), b0)
        if s < 0.25:
            b = b0 + (bpk - b0) * _smoothstep(s / 0.25)
        else:
            b = bpk + (b1 - bpk) * _smoothstep((s - 0.25) / 0.6)
        ref = self._leg_fk(side, th, b - rel)
        w = _smoothstep((s - 0.7) / 0.3)
        out = ref * (1.0 - w) + planned * w
        # the swinging foot never moves backward over the ground (slow walks; the stop step
        # re-plans a shorter landing mid-swing)
        d = 1.0 if self._velocity >= 0 else -1.0
        wx = self._c2w(float(out[0]))
        if foot.swing_last_wx is not None and d * (wx - foot.swing_last_wx) < 0.0:
            wx = foot.swing_last_wx
        foot.swing_last_wx = wx
        return np.array([self._w2c(wx), float(out[1])])

    def _swing_end_planned(self, side: str) -> np.ndarray:
        foot = self._feet[side]
        tau = (self.phase + foot.phase_offset) % 1.0
        u = (tau - self.toe_off_end) / max(1.0 - self.toe_off_end, 1e-6) if tau >= self.stance_ratio else 1.0
        return np.array([self._w2c(self._land_x(foot, _smoothstep(u))), float(foot.swing_p3[1])])

    @staticmethod
    def _land_x(foot: "_FootState", blend: float) -> float:
        """落点世界 x；刹车重规划的修正随轨迹进度从 0 渐入到全量（重规划那一帧位姿连续）。"""
        if not foot.land_fix:
            return foot.land_world_x
        k = min(1.0, max(0.0, (blend - foot.land_fix_b0) / max(1.0 - foot.land_fix_b0, 1e-6)))
        return foot.land_world_x - foot.land_fix * (1.0 - k)

    def _brake_replan(self) -> None:
        """进入刹车：正在摆动的脚按减速曲线重算剩余摆动的身体位移与落地速度。

        摆动在行走中规划，按匀速 v·T 预测身体位移；刹车后身体走不到那么远，
        1.2 Hz（摆动更长）下摆动腿够不着落点 → 伸展下限一帧压低骨盆、支撑膝跳变。
        """
        v = self._velocity
        swing_time = max(1.0 - self.toe_off_end, 0.0) / max(self.stride_hz, 1e-6)
        for side, foot in self._feet.items():
            if not foot.swing_planned or foot.contact.is_locked:
                continue
            tau = (self.phase + foot.phase_offset) % 1.0
            if tau < self.stance_ratio:
                continue
            u = min(1.0, max(0.0, (tau - self.toe_off_end) / max(1.0 - self.toe_off_end, 1e-6)))
            t_rem = (1.0 - u) * swing_time
            if self.brake_linear > 0.0:
                s = min(t_rem, self.brake_linear)
                travel = v * (s - s * s / (2.0 * self.brake_linear))
                v_land = v * max(0.0, 1.0 - t_rem / self.brake_linear)
            else:
                decay = math.exp(-t_rem / BRAKE_RAMP_S)
                travel = v * BRAKE_RAMP_S * (1.0 - decay)
                v_land = v * decay
            # 落点 = 印迹中心 + 半扫（∝ 落地速度）+ 身体位移；两项都按减速后的值重算
            sweep_fix = (v_land - v) * self.stance_ratio / self.stride_hz * 0.5
            fix = (travel - v * t_rem) + sweep_fix
            foot.land_fix_b0 = _smoothstep(u)
            foot.land_world_x += fix
            foot.land_fix = fix

    def _ik_angles(self, side: str, ankle_t: np.ndarray) -> Tuple[float, float]:
        leg = self.legs[side]
        h, k, _a = Analytical2BoneIK.solve(self._hip_canvas(side), ankle_t, leg.l1, leg.l2,
                                           self.knee_bend, 0.0, exact_within=leg.d_rest + 1e-6)
        return (h - (leg.thigh_rest_angle - math.pi / 2.0) - self._pelvis_rot,
                k - (leg.shin_rest_angle - leg.thigh_rest_angle))

    def _ref_stance_dip(self) -> Optional[float]:
        """领先支撑腿按参考膝角（着地≈直 → 承重缓冲峰 → 中期近伸直）反求骨盆高度；
        其余支撑腿仍须够得着（按 0.995 伸展率兜底）。"""
        req = {}
        for side, foot in self._feet.items():
            tau = (self.phase + foot.phase_offset) % 1.0
            if tau >= self.stance_ratio or not foot.contact.is_locked:
                continue
            leg = self.legs[side]
            sp = tau / max(self.stance_ratio, 1e-6)
            if sp < 0.45:
                bend = self.ref_mid_knee + (self.ref_load_knee - self.ref_mid_knee) * math.sin(math.pi * sp / 0.45)
            else:
                bend = self.ref_mid_knee
            d_req = math.sqrt(leg.l1 ** 2 + leg.l2 ** 2 + 2.0 * leg.l1 * leg.l2 * math.cos(bend))
            ank = foot.ankle_now
            hip = self._hip_canvas(side)
            dx = float(ank[0] - hip[0])
            inner = d_req * d_req - dx * dx
            if inner > 0.0:
                req[side] = (sp, (float(ank[1]) - math.sqrt(inner)) - (float(hip[1]) - self._dip))
        if not req:
            return None, 0.0
        lead = min(req, key=lambda sd: req[sd][0])
        dip = req[lead][1]
        if len(req) == 2:
            # a freshly landed leg takes over the pelvis height over its first 15% of stance
            other = next(sd for sd in req if sd != lead)
            w = _smoothstep(req[lead][0] / 0.15)
            dip = dip * w + req[other][1] * (1.0 - w)
        floor = 0.0                                   # every stance leg: extension <= 0.995
        for side, foot in self._feet.items():
            tau = (self.phase + foot.phase_offset) % 1.0
            if tau >= self.toe_off_end:
                # stance legs always; a swing leg in its last 20% too, so the pelvis is already
                # low enough when it lands (else the floor snaps the pelvis down at contact)
                u = (tau - self.toe_off_end) / max(1.0 - self.toe_off_end, 1e-6)
                if u < 0.8:
                    continue
            lg = self.legs[side]
            h = self._hip_canvas(side)
            ddx = float(foot.ankle_now[0] - h[0])
            r = REF_MAX_EXTENSION * (lg.l1 + lg.l2)
            if r * r - ddx * ddx > 0.0:
                need = (float(foot.ankle_now[1]) - math.sqrt(r * r - ddx * ddx)) - (float(h[1]) - self._dip)
                floor = max(floor, need)
        return max(0.0, dip), max(0.0, floor)

    def _swing_pitch(self, side: str, u: float) -> float:
        """摆动仰角：从规划时仰角（趾离地末）→ 后跟着地角。"""
        foot = self._feet[side]
        return foot.swing_pitch0 + (foot.land_pitch - foot.swing_pitch0) \
            * _smoothstep(min(1.0, max(0.0, u)))

    def _toe_off_pitch(self, u_to: float) -> float:
        push = math.radians(FOREFOOT_ROLL_DEG)
        end = math.radians(self.toe_off_end_deg)
        return push + (end - push) * 0.5 * (1.0 - math.cos(math.pi * min(1.0, u_to)))

    # ---------------- 骨盆（§3.2 几何精确化） ----------------

    def _pelvis_dip(self, foot_targets: Dict[str, np.ndarray]) -> float:
        """支撑腿伸展率 ≤ r_target 的精确下沉（两腿当前 dx 取 max）。

        dip = max_leg( D_rest − sqrt((r·Ltot)² − dx²) )⁺。
        """
        worst = 0.0
        for side, ankle_t in foot_targets.items():
            leg = self.legs[side]
            hip = self._hip_canvas(side)
            dx = float(ankle_t[0] - hip[0])
            extension = self.stance_extension
            if self.park_feet:
                # Ease back to the bind stance after the feet have landed. A permanent
                # crouch in idle forced the presenter to pull both shoes sideways later.
                if self.state is GaitPhaseState.WALK_PARK:
                    weight = 1.0 - _smoothstep((self._t - (self._park_end - 0.2)) / 0.2)
                else:
                    weight = min(1.0, self._env * 4.0) if self.state in (
                        GaitPhaseState.IDLE_FRONT, GaitPhaseState.IDLE_SIDE,
                        GaitPhaseState.TURN_TO_SIDE, GaitPhaseState.TURN_TO_FRONT) else 1.0
                rest_extension = leg.d_rest / (leg.l1 + leg.l2)
                extension = rest_extension + (extension - rest_extension) * weight
            r_d = extension * (leg.l1 + leg.l2)
            inner = r_d * r_d - dx * dx
            if inner <= 0.0:
                worst = max(worst, self._dip + 40.0)
                continue
            vertical = (float(ankle_t[1] - hip[1]) + self._dip
                        if self.park_feet else leg.d_rest)
            worst = max(worst, vertical - math.sqrt(inner))
        return max(0.0, worst)

    def _hip_canvas(self, side: str) -> np.ndarray:
        """当前骨盆位姿下的髋关节画布位置。

        与运行时 FK 次序严格一致：``J_hip = J_root + t + R(θ)·(J_hip_rest −
        J_root)``（先旋转静止偏移、再平移）。若把平移算进旋转向量（旧式
        ``R(θ)·(h+t−p)+p``），θ×t 交叉项与运行时相差 ~1 canvas px，足部
        世界漂移净增 ~0.2 逻辑 px。
        """
        leg = self.legs[side]
        hip = leg.hip_rest.copy()
        if self._pelvis_rot:
            rx, ry = _rotated((float(hip[0] - self.pelvis_rest[0]),
                               float(hip[1] - self.pelvis_rest[1])), self._pelvis_rot)
            hip = self.pelvis_rest + np.array([rx, ry])
        return hip + np.array([self._sway, self._dip])

    # ---------------- IK 求解 ----------------

    def _solve_leg(self, side: str, ankle_t: np.ndarray, pitch: float
                   ) -> Tuple[float, float, float, bool]:
        """IK → rig 骨骼局部角（弧度）。返回 (θ大腿, θ膝, θ脚, 限位饱和)。"""
        leg = self.legs[side]
        hip = self._hip_canvas(side)
        h, k, _a = Analytical2BoneIK.solve(
            hip, ankle_t, leg.l1, leg.l2, self.knee_bend, pitch, exact_within=leg.d_rest + 1e-6)
        thigh_rest_off = leg.thigh_rest_angle - math.pi / 2.0
        shin_rel_rest = leg.shin_rest_angle - leg.thigh_rest_angle
        # 世界大腿角 = thigh_rest + pelvis_rot + ang_th（骨盆旋转计入局部角）
        ang_th = h - thigh_rest_off - self._pelvis_rot
        ang_kn = k - shin_rel_rest
        ang_ft = pitch - (self._pelvis_rot + ang_th + ang_kn)
        sat = any(a < lo - 1e-9 or a > hi + 1e-9
                  for a, lo, hi in zip((ang_th, ang_kn, ang_ft),
                                       leg.clamp_lo, leg.clamp_hi))
        ang_th = min(max(ang_th, leg.clamp_lo[0]), leg.clamp_hi[0])
        ang_kn = min(max(ang_kn, leg.clamp_lo[1]), leg.clamp_hi[1])
        ang_ft = min(max(ang_ft, leg.clamp_lo[2]), leg.clamp_hi[2])
        return ang_th, ang_kn, ang_ft, sat

    def _foot_marker_actual_world(self, side: str) -> float:
        """限位后骨骼角前向解算接触标记世界 x（QA 真值）。

        静止位无旋转（rest 仅贡献平移）：骨方向 = 静止向量被局部角旋转，
        即 thigh 几何角 = thigh_rest + pelvis + th；而**足底世界仰角只累积
        局部角** = pelvis + th + kn + ft（零姿态时恒 0 = 平贴）。
        """
        foot = self._feet[side]
        leg = self.legs[side]
        th, kn, ft = foot.applied or (0.0, 0.0, 0.0)
        hip = self._hip_canvas(side)
        # 几何向合成（与运行时 FK 逐项一致）：骨向量 = 静止向量被局部角
        # 累积旋转——大腿向角 = thigh_rest + pelvis + th，
        # 小腿向角 = shin_rest + pelvis + th + kn（局部角相对静止位，
        # 不可与几何向角直加，否则静止偏角重复计入）。
        thigh_ang = leg.thigh_rest_angle + self._pelvis_rot + th
        shin_ang = leg.shin_rest_angle + self._pelvis_rot + th + kn
        knee = hip + leg.l1 * np.array([math.cos(thigh_ang),
                                        math.sin(thigh_ang)])
        ankle = knee + leg.l2 * np.array([math.cos(shin_ang),
                                          math.sin(shin_ang)])
        off = self._marker_offset(side, foot.contact.contact_type)
        pitch = self._pelvis_rot + th + kn + ft
        rx, _ry = _rotated((float(off[0]), float(off[1])), pitch)
        return self._c2w(float(ankle[0]) + rx)

    # ---------------- 状态机（§3.5） ----------------

    def _transition(self, dt: float, desired_v: float, is_grounded: bool,
                    is_dragged: bool) -> None:
        s = self.state
        if is_dragged:
            if s is not GaitPhaseState.AIRBORNE:
                self.state = GaitPhaseState.AIRBORNE
                self._unlock("l")
                self._unlock("r")
                self._velocity = 0.0
                self._env = 0.0
            return
        if s is GaitPhaseState.AIRBORNE:
            if is_grounded:
                self.state = GaitPhaseState.LANDING
                self._t = 0.0
                self._enter_landing()
            return

        if s is GaitPhaseState.IDLE_FRONT:
            self._idle_timer = 0.0
            if abs(desired_v) > 1.0:
                self._begin_turn(math.copysign(45.0, desired_v),
                                 GaitPhaseState.WALK_START)
        elif s is GaitPhaseState.TURN_TO_SIDE:
            if self._yaw_reached():
                self.state = self._post_turn_state
                self._t = 0.0
                self._velocity = 0.0
                if self.state is GaitPhaseState.WALK_START:
                    self._begin_walking(desired_v)
            elif abs(desired_v) <= 1.0:
                self._begin_turn(0.0, GaitPhaseState.IDLE_FRONT)
        elif s is GaitPhaseState.WALK_START:
            if abs(desired_v) <= 1.0:
                self.state = GaitPhaseState.WALK_BRAKE
                self._t = 0.0
                self._brake_replan()
            elif self._reversal_requested(desired_v):
                self._begin_reverse(desired_v)
            elif self._env >= 0.97 and abs(self._velocity - self._velocity_cmd) < 2.0:
                self.state = GaitPhaseState.WALK_LOOP
                self._t = 0.0
        elif s is GaitPhaseState.WALK_LOOP:
            if abs(desired_v) <= 1.0:
                self.state = GaitPhaseState.WALK_BRAKE
                self._t = 0.0
                self._brake_replan()
            elif self._reversal_requested(desired_v):
                self._begin_reverse(desired_v)
        elif s is GaitPhaseState.WALK_BRAKE:
            if self._reversal_requested(desired_v):
                self._begin_reverse(desired_v)
            elif abs(desired_v) > 1.0 and abs(self._velocity) < 25.0:
                # 刹车中同向再启动：回起步（保持当前 yaw，不掉头）
                self.state = GaitPhaseState.WALK_START
                self._t = 0.0
            elif abs(self._velocity) < 2.0:
                self.state = GaitPhaseState.WALK_STOP
                self._t = 0.0
        elif s is GaitPhaseState.WALK_STOP:
            if self.park_feet and not self._any_foot_swinging():
                # 摆动脚一落地就收步——不再等包络衰减：旧逻辑在等待期间以包络速率推进
                # 相位，支撑脚滚到前掌/后跟后停住（前脚踮脚、后脚翘脚尖 0.3–0.6 s）
                self.state = GaitPhaseState.WALK_PARK
                self._t = 0.0
                self._idle_timer = 0.0
                self._plan_park()
            elif self._env <= 0.02 and not self._any_foot_swinging():
                self.state = GaitPhaseState.IDLE_SIDE
                self._t = 0.0
                self._idle_timer = 0.0
        elif s is GaitPhaseState.WALK_PARK:
            if self._t >= self._park_end:
                # 交给静止放平时直接沿用收好的站姿（不再按冻结相位重解，否则远侧脚
                # 会闪回一帧后跟着地角）
                for side, foot in self._feet.items():
                    foot.pitch_now = 0.0
                    self._lock_contact(side, ContactType.FLAT_SOLE)
                    foot.settle_active = True
                    foot.settle_t = 1.0
                    foot.settle_pitch = 0.0
                self.state = GaitPhaseState.IDLE_SIDE
                self._t = 0.0
                self._idle_timer = 0.0
        elif s is GaitPhaseState.IDLE_SIDE:
            self._idle_timer += dt
            if abs(desired_v) > 1.0:
                self._begin_turn(math.copysign(45.0, desired_v),
                                 GaitPhaseState.WALK_START)
            elif self._idle_timer >= self.idle_timeout:
                self._begin_turn(0.0, GaitPhaseState.IDLE_FRONT)
        elif s is GaitPhaseState.TURN_TO_FRONT:
            if self._yaw_reached():
                self.state = GaitPhaseState.IDLE_FRONT
                self._t = 0.0
        elif s is GaitPhaseState.TURN_REVERSE:
            if self._t >= self._reverse_total:
                self.state = GaitPhaseState.WALK_START
                self._t = 0.0
                self._velocity = 0.0
        elif s is GaitPhaseState.LANDING:
            if self._t >= LANDING_CROUCH_S:
                if abs(self.view_yaw) < 1.0:
                    self.state = GaitPhaseState.IDLE_FRONT
                else:
                    self._begin_turn(0.0, GaitPhaseState.IDLE_FRONT)
                self._t = 0.0

    def _any_foot_swinging(self) -> bool:
        for foot in self._feet.values():
            tau = (self.phase + foot.phase_offset) % 1.0
            if tau >= self.stance_ratio:
                return True
        return False

    def _enter_landing(self) -> None:
        """落地复位（§3.5.3）：双脚在当前窗口系下重锚地面（印迹中心落位）。"""
        for side, foot in self._feet.items():
            leg = self.legs[side]
            foot.ankle_now = np.array([
                self._track_center(side) - float(leg.marker_sole[0]),
                float(leg.ankle_rest[1])])
            foot.pitch_now = 0.0
            foot.swing_planned = False
            self._lock_contact(side, ContactType.FLAT_SOLE)

    def _begin_turn(self, target_yaw: float, post_state: GaitPhaseState,
                    duration: Optional[float] = None) -> None:
        self.state = (GaitPhaseState.TURN_TO_FRONT if abs(target_yaw) < 1.0
                      else GaitPhaseState.TURN_TO_SIDE)
        self._target_yaw = target_yaw
        self._yaw_from = self.view_yaw
        self._turn_dur = duration or self.turn_duration
        self._t = 0.0
        self._post_turn_state = post_state

    def _begin_walking(self, desired_v: float) -> None:
        """起步相位初始化（从静止双脚进入行走）。

        首摆脚 = 行进反侧（落后脚先迈）；把全局相位置于"首摆脚刚进入贝塞尔
        摆动、支撑脚处于支撑中段"。若不重置相位，静止锚定的支撑脚会从站位
        扫过整个支撑相（~0.48s · v/S ≈ 238 canvas px），后扫深度超设计
        2 倍，顶死骨盆下沉与膝/脚限位。
        """
        direction = 1.0 if desired_v >= 0 else -1.0
        if self.adaptive_cadence:
            reference_speed = self.speed_target * self.scale / (256.0 / 1696.0)
            self.stride_hz = self._base_stride_hz * max(1.0, abs(desired_v) / max(reference_speed, 1.0))
        swing = "l" if direction > 0 else "r"
        u_start = self.toe_off_end - self._feet[swing].phase_offset
        self.phase = u_start % 1.0

    def _begin_reverse(self, desired_v: float) -> None:
        """途中反向：速度平滑过零 + yaw 穿 0 掉头（禁止瞬时镜像，§3.5.1）。"""
        self.state = GaitPhaseState.TURN_REVERSE
        self._t = 0.0
        self._decel_dur = BRAKE_RAMP_S * 0.7
        self._yaw_from = self.view_yaw
        self._target_yaw = math.copysign(45.0, desired_v)
        self._turn_dur = 0.5
        self._reverse_total = self._decel_dur + self._turn_dur

    def _reversal_requested(self, desired_v: float) -> bool:
        """反向判定：新指令方向与**当前运动方向**相反（与自身 cmd 比较恒等，
        是永不触发的死分支）。"""
        return (abs(desired_v) > 1.0 and abs(self._velocity) > 1.0
                and math.copysign(1.0, desired_v) != math.copysign(1.0, self._velocity))

    def _yaw_reached(self) -> bool:
        return abs(self.view_yaw - self._target_yaw) < 0.5

    def _clamp_speed(self, v: float) -> float:
        return math.copysign(min(abs(v), self.speed_cap), v) if v else 0.0

    # ---------------- 主入口（§6.1 签名） ----------------

    def update(
        self,
        dt: float,
        desired_velocity_x: float,
        current_window_pos: Tuple[int, int],
        is_grounded: bool = True,
        is_dragged: bool = False,
    ) -> GaitOutputs:
        """主步态求解更新入口（呈现层每渲染拍调用一次）。

        dt 取单调时钟实测值；内部以 ≤5ms 子步推进相位并在子相位边界精确换锚
        （§5.1 自适应分步）。窗口位移按子步梯形积分（v 前后均值），30/60Hz 与
        掉帧轨迹一致（§7 时间步长一致性）。
        """
        dt = min(max(float(dt), 0.0), MAX_DT_S)
        wx_int = float(current_window_pos[0])
        if not self._window_initialized:
            self._wx = wx_int                  # 浮点累加器自窗口真实位播种
            self._window_initialized = True
            for side in self._feet:          # 锚点重定位到真实窗口系
                self._lock_contact(side, ContactType.FLAT_SOLE)
        elif abs(wx_int - round(self._wx)) > TELEPORT_JUMP_PX and \
                self.state is not GaitPhaseState.AIRBORNE:
            # 无拖拽标志的瞬移（跨屏/程序化移动）：按落地复位重建接触
            self._wx = wx_int
            self._enter_landing()
            self._t = 0.0
        self._velocity_cmd = self._clamp_speed(float(desired_velocity_x))
        base = self._wx

        delta_win = 0.0
        remaining = dt
        while remaining > 1e-9:
            step = min(MAX_SUBSTEP_S, remaining)
            remaining -= step
            v_before = self._velocity
            self._advance(step, base + delta_win, is_grounded, is_dragged)
            delta_win += 0.5 * (v_before + self._velocity) * step
        # 帧末统一以最终窗口 x 重解目标再输出。内部维护**浮点窗口累加器**
        # （与调用方累加器同值：同种子同增量）——锚点世界坐标按构造恒定，
        # 渲染口径漂移只剩窗口整数落位的 ±0.5 px 舍入（接触预算内）。切勿
        # 按整数量化内部累加（每帧 −0.25 的系统速度损失会让双脚逐步后坠）。
        self._wx = base + delta_win
        self._update_foot_targets(dt)
        out = self._emit(dt if dt > 1e-9 else 1e-9)
        out.delta_window_x = delta_win
        return out

    def _advance(self, dt: float, window_x: float,
                 is_grounded: bool, is_dragged: bool) -> None:
        """单子步：状态机 → 速度/包络/视角 → 相位 → 双脚计划 → 骨盆。"""
        self._wx = window_x
        self._transition(dt, self._velocity_cmd, is_grounded, is_dragged)
        s = self.state
        self._t += dt

        # ---- 速度 ----
        if s in (GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP,
                 GaitPhaseState.TURN_REVERSE):
            target = self._velocity_cmd
            ramp = START_RAMP_S if s is GaitPhaseState.WALK_START else 0.3
            self._velocity += (target - self._velocity) * min(1.0, dt / max(ramp, 1e-3))
            if abs(self._velocity - target) < 0.5:
                self._velocity = target
        elif s is GaitPhaseState.WALK_BRAKE:
            if self.brake_linear > 0.0:
                if self._t <= dt + 1e-9:            # 刚进入刹车：记录初速
                    self._brake_v0 = self._velocity
                self._velocity = self._brake_v0 * max(0.0, 1.0 - self._t / self.brake_linear)
            else:
                self._velocity += (0.0 - self._velocity) * min(1.0, dt / BRAKE_RAMP_S)
        elif s is GaitPhaseState.WALK_PARK and self._park_glide_w > 0.0 \
                and self._t < self._park_glide_s:
            # 身体前移到前脚上方：速度 = 前移量 × smoothstep 导数（双脚锚定世界坐标 → 不滑）
            u = min(1.0, self._t / self._park_glide_s)
            self._velocity = self._park_glide_w * 6.0 * u * (1.0 - u) / self._park_glide_s
        else:
            self._velocity = 0.0

        # ---- 步态包络 ----
        if s is GaitPhaseState.WALK_START:
            env_target = min(1.0, self._t / START_RAMP_S)
        elif s in (GaitPhaseState.WALK_LOOP, GaitPhaseState.TURN_REVERSE):
            env_target = 1.0
        elif s is GaitPhaseState.WALK_BRAKE:
            env_target = min(1.0, abs(self._velocity) / 25.0)
        elif s is GaitPhaseState.WALK_STOP:
            env_target = 0.3 if self._any_foot_swinging() else 0.0
        else:
            env_target = 0.0
        self._env += (env_target - self._env) * min(1.0, dt / 0.12)

        # ---- 视角（转身 0.25–0.35s smoothstep；反向在减速后穿 0 掉头） ----
        if s in (GaitPhaseState.TURN_TO_SIDE, GaitPhaseState.TURN_TO_FRONT):
            u = _smoothstep(min(1.0, self._t / self._turn_dur))
            self.view_yaw = self._yaw_from + (self._target_yaw - self._yaw_from) * u
        elif s is GaitPhaseState.TURN_REVERSE and self._t >= self._decel_dur:
            u = _smoothstep(min(1.0, (self._t - self._decel_dur) / self._turn_dur))
            self.view_yaw = self._yaw_from + (self._target_yaw - self._yaw_from) * u

        # Bound stride length at follow speeds rather than stretching the legs.
        # Do not change cadence while braking: finish the already planned swing.
        if self.adaptive_cadence and s in (GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP):
            reference_speed = self.speed_target * self.scale / (256.0 / 1696.0)
            target_hz = self._base_stride_hz * max(1.0, abs(self._velocity_cmd) / max(reference_speed, 1.0))
            self.stride_hz += (target_hz - self.stride_hz) * min(1.0, dt / 0.12)

        # ---- 相位（§3.4：Φ += f·dt mod 1；停止相等摆动脚收步后再停） ----
        if s in (GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP,
                 GaitPhaseState.WALK_BRAKE, GaitPhaseState.TURN_REVERSE):
            self.phase = (self.phase + self.stride_hz * dt) % 1.0
        elif s is GaitPhaseState.WALK_STOP:
            rate = 1.0 if self._any_foot_swinging() else max(0.0, self._env)
            self.phase = (self.phase + self.stride_hz * dt * rate) % 1.0

        # ---- 双脚：相位事件（换锚 / 摆动规划）→ 目标解算 ----
        # 非行走状态（idle/turn/landing）相位冻结，事件循环必须跳过——否则
        # 会按冻结相位反复换锚（如 τ=0.5 的前掌），与静止放平的 sole 基准
        # 错配产生假漂移。
        walking = s in (GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP,
                        GaitPhaseState.WALK_BRAKE, GaitPhaseState.TURN_REVERSE,
                        GaitPhaseState.WALK_STOP)
        for side, foot in self._feet.items():
            if s is GaitPhaseState.AIRBORNE or not walking:
                continue                    # 悬空保持末姿态 / 静止走放平
            tau = (self.phase + foot.phase_offset) % 1.0
            prev_tau = _unwrap(tau, (tau - self.stride_hz * dt) % 1.0)

            if tau < self.stance_ratio:
                kind = self._sub_phase_kind(tau)
                if self._needs_anchor(prev_tau, tau, foot, kind):
                    if prev_tau < 0.0:
                        # 相位环绕 = 新一步着地
                        if foot.swing_planned:
                            # 贝塞尔末端位姿 → 锚定后跟（仰角 = 着地角）
                            foot.ankle_now = self._swing_ankle(side, 1.0)
                            foot.pitch_now = foot.land_pitch
                            foot.swing_planned = False
                            foot.heel_entry_pitch = foot.land_pitch
                            self._lock_contact(side, ContactType.HEEL)
                        elif foot.contact.contact_type is ContactType.FLAT_SOLE:
                            # 起步首支撑（静止全掌进入）：跳过后跟子相，全掌保持
                            foot.heel_entry_pitch = foot.pitch_now
                            self._lock_contact(side, ContactType.FLAT_SOLE)
                        else:
                            foot.heel_entry_pitch = foot.pitch_now
                            self._lock_contact(side, kind)
                    else:
                        # 子相位边界转移（heel→flat→forefoot）：先按旧锚点+当前
                        # 子步窗口重解位姿（ankle_now 是上一帧刷新值，含一帧
                        # 窗口位移的陈旧量，直接捕获会让锚点漂移一个 Δframe），
                        # 再以当前位姿重锚新标记（§3.3 支点转移零迁移）。
                        self._refresh_stance_from_anchor(side)
                        foot.heel_entry_pitch = foot.pitch_now
                        self._lock_contact(side, kind)
            elif tau < self.toe_off_end:
                if foot.contact.contact_type is not ContactType.FOREFOOT:
                    self._lock_contact(side, ContactType.FOREFOOT)
            else:
                if not foot.swing_planned:
                    # 贝塞尔入口规划（P0 = 趾离地末位姿，pitch≈18°）；
                    # 也覆盖任意中途进入（起步/状态恢复）
                    self._plan_swing(side, self._velocity)
                if foot.contact.is_locked:
                    self._unlock(side)
        self._update_foot_targets()

        # ---- 骨盆：精确下沉（低通吸收 sqrt 拐点）+ 摇摆 + 微前倾 ----
        if s is GaitPhaseState.AIRBORNE:
            self._dip_raw = 0.0
        else:
            # 下沉须同时照顾摆动腿：趾离地瞬间摆动脚仍远在身后，若只按支撑腿算，
            # 骨盆在数十毫秒内抬升 ~15 canvas px，摆动腿被拉直、膝角在一帧内
            # 跳向反折限位（60Hz 下膝角速度 > 20 rad/s）
            base = self._pelvis_dip({sd: f.ankle_now for sd, f in self._feet.items()})
            if self.ref_curves and s in (GaitPhaseState.WALK_LOOP, GaitPhaseState.WALK_START,
                                         GaitPhaseState.WALK_BRAKE, GaitPhaseState.WALK_STOP):
                ref, ref_floor = self._ref_stance_dip()
                if ref is not None:
                    # 起步时从旧下沉平滑过渡
                    w = min(1.0, self._env)
                    base = base * (1.0 - w) + ref * w
            if s is GaitPhaseState.LANDING:
                crouch = math.sin(math.pi * min(1.0, self._t / LANDING_CROUCH_S))
                ltot = max(self.legs["l"].l1 + self.legs["l"].l2,
                           self.legs["r"].l1 + self.legs["r"].l2)
                base += 0.042 * ltot * crouch
                base *= max(self._env, 0.35)
            elif not self.dip_geometric:
                base *= self._env
            if self.ref_curves and s in (GaitPhaseState.WALK_LOOP, GaitPhaseState.WALK_START,
                                         GaitPhaseState.WALK_BRAKE, GaitPhaseState.WALK_STOP):
                # reference dip changes at stance handovers: bound its rate (knee <= 20 rad/s) -
                # but never let a stance leg reach full extension (the IK is singular there)
                lim = REF_DIP_RATE_PX_S * dt
                base = min(max(base, self._dip_raw - lim), self._dip_raw + lim)
                base = max(base, ref_floor)
            self._dip_raw = base
        k = 1.0 - math.exp(-dt / DIP_SMOOTH_TAU_S)
        self._dip += (self._dip_raw - self._dip) * k
        if self.ref_curves and s in (GaitPhaseState.WALK_LOOP, GaitPhaseState.WALK_START,
                                     GaitPhaseState.WALK_BRAKE, GaitPhaseState.WALK_STOP):
            # hard floor after the low-pass: a lagging dip let the trailing leg reach full
            # extension, where the IK is singular and the heel lift then flicked the knee 16 deg
            self._dip = max(self._dip, self._ref_stance_dip()[1])
        # 横向摇摆（§3.2）：重心压向支撑脚
        sup_l = self._feet["l"].contact.is_locked
        sup_r = self._feet["r"].contact.is_locked
        sway_target = 0.0
        if sup_l != sup_r:
            sway_target = -(self.sway_world / self.scale) * self._env \
                if sup_l else (self.sway_world / self.scale) * self._env
        self._sway += (sway_target - self._sway) * min(1.0, dt / 0.15)
        # 前倾（速度比例，微小）
        # In y-down canvas space, positive rotation moves the torso above the
        # pelvis toward +x. The right-facing side rig must lean forward, not back.
        lean = math.radians(self.lean_degrees) * min(1.0, abs(self._velocity) / 75.0)
        if self._velocity < 0:
            lean = -lean
        self._pelvis_rot += (lean - self._pelvis_rot) * min(1.0, dt / 0.25)

    @staticmethod
    def _needs_anchor(prev_tau: float, tau: float, foot: "_FootState",
                      kind: ContactType) -> bool:
        """是否需要（重）锚定：跨环绕着地 / 子相位边界转移 / 未锁定。"""
        if not foot.contact.is_locked:
            return True
        if prev_tau < 0.0:                  # 相位环绕 = 新一步着地
            return True
        return foot.contact.contact_type is not kind

    def _sub_phase_kind(self, tau: float) -> ContactType:
        if tau < self.heel_end:
            return ContactType.HEEL
        if tau < self.flat_end:
            return ContactType.FLAT_SOLE
        return ContactType.FOREFOOT

    def _update_foot_targets(self, dt: float = 0.0) -> None:
        """按当前相位与 ``self._wx`` 重解双脚目标（帧末刷新/子步内供下沉计算）。

        支撑/趾离地目标由锚定世界坐标反解，落点世界恒定 → 调用方以同帧
        ``X_win + delta_window_x`` 重构时接触标记世界坐标严格不变。
        非行走状态进入"静止放平"：sole 标记世界位置保持不变，仰角缓释为 0
        （从提踵/着地角收平，杜绝静止翘脚尖，且放平过程零滑移）。
        """
        if self.state is GaitPhaseState.WALK_PARK:
            self._park_targets()
            return
        walking = self.state in (
            GaitPhaseState.WALK_START, GaitPhaseState.WALK_LOOP,
            GaitPhaseState.WALK_BRAKE, GaitPhaseState.TURN_REVERSE,
            GaitPhaseState.WALK_STOP)
        for side, foot in self._feet.items():
            if self.state is GaitPhaseState.AIRBORNE:
                continue
            if not walking:
                self._settle_foot(side, foot, dt)
                continue
            foot.settle_active = False
            tau = (self.phase + foot.phase_offset) % 1.0
            if tau < self.stance_ratio:
                foot.ankle_now, foot.pitch_now = self._stance_target(side, tau)
            elif tau < self.toe_off_end:
                u_to = (tau - self.stance_ratio) / self.toe_off_span
                foot.pitch_now = self._toe_off_pitch(u_to)
                off = self.legs[side].marker_forefoot
                rx, ry = _rotated((float(off[0]), float(off[1])), foot.pitch_now)
                anchor_x = self._w2c(foot.contact.world_anchor_x)
                foot.ankle_now = np.array(
                    [anchor_x - rx, foot.contact.world_anchor_y - ry])
            else:
                u = (tau - self.toe_off_end) / max(1.0 - self.toe_off_end, 1e-6)
                foot.ankle_now = self._swing_ankle(side, u)
                foot.pitch_now = self._swing_pitch(side, u)

    def _planted_ankle(self, side: str, pitch: float) -> np.ndarray:
        """支撑脚：绕当前接触标记（后跟/全掌/前掌）旋转到给定仰角，接触点世界坐标不变。"""
        foot = self._feet[side]
        off = self._marker_offset(side, foot.contact.contact_type)
        rx, ry = _rotated((float(off[0]), float(off[1])), pitch)
        return np.array([self._w2c(foot.contact.world_anchor_x) - rx,
                         foot.contact.world_anchor_y - ry])

    def _plan_park(self) -> None:
        """停步收脚规划（只向前）。

        1. 双脚放平：绕各自当前接触点把仰角缓回 0（后跟着地的脚绕后跟、踮脚的脚绕前掌）。
        2. 身体前移：若有脚落在静止站位之前，身体（窗口）平滑前移到该脚上方，使所有脚都
           不在站位之前——双脚锚定世界坐标，前移过程不滑脚。
        3. 仍落后于站位的脚依次（最靠后的先）向前迈一小步落到站位；不向后迈、不交叉拖脚。
        """
        self._park_start = {}
        offsets = {}
        for side, foot in self._feet.items():
            if not foot.contact.is_locked:
                self._lock_contact(side, ContactType.FLAT_SOLE)
            self._park_start[side] = foot.pitch_now
            flat = self._planted_ankle(side, 0.0)
            leg = self.legs[side]
            offsets[side] = float(flat[0] - leg.ankle_rest[0])
        glide_c = max(0.0, max(offsets.values()))
        self._park_glide_w = glide_c * self.scale
        self._park_wx_final = self._wx + self._park_glide_w     # 前移结束后的窗口世界 x
        after = {sd: o - glide_c for sd, o in offsets.items()}
        # 离站位 > 1 canvas px 的脚都迈一步（小偏差 = 快而低的调整小步），全程不滑脚
        order = sorted((sd for sd, o in after.items() if o < -1.0), key=lambda sd: after[sd])
        # 第一步几乎立即迈出（与身体前移/放平并行，迈步脚在空中不受影响）——等前移结束
        # 再迈会让两只鞋前后重叠停顿约 0.5 s（四分之三视角下看起来像叠在一起）
        t0 = 0.06
        self._park_steps, self._park_dur = {}, {}
        for sd in order:
            size = min(1.0, abs(after[sd]) / 120.0)
            self._park_dur[sd] = 0.22 + (self._park_step_s - 0.22) * size
            self._park_steps[sd] = (t0, None)
            t0 += self._park_dur[sd]
        self._park_order = order
        self._park_end = max(t0, self._park_glide_s, self._park_flatten_s) + 0.15

    def _park_targets(self) -> None:
        """按状态时钟求双脚目标（帧末重复刷新不会重复推进）。"""
        k = _smoothstep(min(1.0, self._t / self._park_flatten_s))
        for side, foot in self._feet.items():
            foot.swing_planned = False
            foot.settle_active = False
            step = self._park_steps.get(side)
            if step is None or self._t < step[0]:
                foot.pitch_now = self._park_start.get(side, 0.0) * (1.0 - k)
                foot.ankle_now = self._planted_ankle(side, foot.pitch_now)
                continue
            ts, start = step
            if start is None:                       # 起步瞬间：从脚当前真实位姿起步（可能仍在放平中）
                p0 = self._park_start.get(side, 0.0) * (1.0 - k)
                a0 = self._planted_ankle(side, p0)
                start = (self._c2w(float(a0[0])), float(a0[1]), p0)
                self._park_steps[side] = (ts, start)
                self._unlock(side)
            dur = self._park_dur.get(side, self._park_step_s)
            u = min(1.0, (self._t - ts) / dur)
            end = self.legs[side].ankle_rest
            # 世界系插值到"身体前移结束后的站位"：与身体前移并行时也只向前移动
            end_w = self._park_wx_final + self.scale * float(end[0])
            size = min(1.0, abs(end_w - start[0]) / (120.0 * self.scale))
            blend = _smoothstep(u)
            foot.ankle_now = np.array([self._w2c(start[0] + (end_w - start[0]) * blend),
                                       start[1] + (float(end[1]) - start[1]) * blend])
            foot.ankle_now[1] -= ((1.0 + 2.0 * size) / self.scale) * math.sin(math.pi * u) ** 2
            # 仰角：从起步时的仰角连续过渡，途中脚尖微抬，落地放平
            foot.pitch_now = start[2] * (1.0 - blend) + math.radians(-6.0 * size) * math.sin(math.pi * u)
            if u >= 1.0 and not foot.contact.is_locked:
                foot.ankle_now = np.array([self._w2c(end_w), float(end[1])])
                foot.pitch_now = 0.0
                self._lock_contact(side, ContactType.FLAT_SOLE)

    def _refresh_stance_from_anchor(self, side: str) -> None:
        """按当前锚点与当前子步窗口重解单脚支撑位姿（换锚前调用，防陈旧）。"""
        foot = self._feet[side]
        if foot.contact.is_locked:
            tau = (self.phase + foot.phase_offset) % 1.0
            foot.ankle_now, foot.pitch_now = self._stance_target(side, tau)

    def _settle_foot(self, side: str, foot: _FootState, dt: float) -> None:
        """静止放平：进入静止时以当前 sole 标记位姿重锚，随后仰角 → 0。"""
        if not foot.settle_active:
            foot.settle_active = True
            foot.settle_t = 0.0
            self._refresh_stance_from_anchor(side)
            foot.settle_pitch = foot.pitch_now
            m = self._marker_canvas(side, ContactType.FLAT_SOLE,
                                    foot.ankle_now, foot.pitch_now)
            foot.contact.is_locked = True
            foot.contact.contact_type = ContactType.FLAT_SOLE
            foot.contact.world_anchor_x = self._c2w(float(m[0]))
            foot.contact.world_anchor_y = float(m[1])
            foot.heel_entry_pitch = 0.0
        foot.settle_t += dt
        k = _smoothstep(min(1.0, foot.settle_t / 0.18))
        pitch = foot.settle_pitch * (1.0 - k)
        off = self.legs[side].marker_sole
        rx, ry = _rotated((float(off[0]), float(off[1])), pitch)
        ax = self._w2c(foot.contact.world_anchor_x)
        foot.ankle_now = np.array([ax - rx, foot.contact.world_anchor_y - ry])
        foot.pitch_now = pitch

    def _emit(self, substep_dt: float) -> GaitOutputs:
        """IK 求解 → GaitOutputs（QA 漂移真值 + 膝角速度监测）。"""
        angles: Dict[str, float] = {}
        sat_any = False
        drift = 0.0
        for side, foot in self._feet.items():
            th, kn, ft, sat = self._solve_leg(side, foot.ankle_now, foot.pitch_now)
            foot.applied = (th, kn, ft)
            sat_any = sat_any or sat
            leg = self.legs[side]
            angles[leg.hip_bone] = th
            angles[leg.knee_bone] = kn
            angles[leg.foot_bone] = ft
            if foot.knee_prev is not None and substep_dt > 1e-9:
                foot.knee_rate = abs(kn - foot.knee_prev) / substep_dt
                self._knee_rate_max = max(self._knee_rate_max, foot.knee_rate)
            foot.knee_prev = kn
            if foot.contact.is_locked:
                actual = self._foot_marker_actual_world(side)
                drift = max(drift, abs(actual - foot.contact.world_anchor_x))
        angles["root_hip"] = self._pelvis_rot
        if self.torso_lean:
            angles["spine"] = self.torso_lean * self._env
        if self.skirt_follow:
            ths = [angles[self.legs[sd].hip_bone] for sd in ("l", "r")]
            for bone, th in (("skirt_hem_r", min(ths)), ("skirt_hem_l", max(ths))):
                target = max(-self.skirt_limit, min(self.skirt_limit, self.skirt_follow * th))
                if self.skirt_freq > 0:
                    st = self._skirt.setdefault(bone, [0.0, 0.0])
                    ks, kd = spring_from_frequency(self.skirt_freq, self.skirt_halflife)
                    st[0], st[1] = spring_step(st[0], st[1], target, ks, kd, substep_dt * 1000)
                    angles[bone] = max(-self.skirt_limit, min(self.skirt_limit, st[0]))
                else:
                    angles[bone] = target
        if self.arm_swing:
            # 侧视手臂与同侧腿反相：同侧脚跟着地（τ=0，腿最前）时上臂最后（正角 = 顺时针
            # = 向后），支撑末/趾离地附近最前；按相位而非大腿角——远侧腿静止站位偏后，
            # 用大腿角会让远侧手臂恒定前伸。前臂只在前摆时屈曲。
            for side, foot in self._feet.items():
                c = math.cos(2.0 * math.pi * ((self.phase + foot.phase_offset - self.arm_phase_lag) % 1.0))
                k = self.far_arm_scale if side == "r" else 1.0
                angles[f"upper_arm_{side}"] = self.arm_swing * self._env * c * k
                fore = -(self.forearm_base + k * self.forearm_bend * 0.5 * (1.0 - c)) * self._env
                angles[f"forearm_{side}"] = fore
                if self.wrist_freq > 0.0:
                    fw = angles[f"upper_arm_{side}"] + fore
                    target = fw - self.hand_follow * fore
                    st = self._wrist.get(side)
                    if st is None:
                        st = self._wrist[side] = [target, 0.0]
                    k_s, k_d = spring_from_frequency(self.wrist_freq, self.wrist_halflife)
                    st[0], st[1] = spring_step(st[0], st[1], target, k_s, k_d, substep_dt * 1000.0)
                    local = max(-self.wrist_limit, min(self.wrist_limit, st[0] - fw))
                    st[0] = fw + local
                    angles[f"hand_{side}"] = local
                elif self.hand_follow:
                    angles[f"hand_{side}"] = -self.hand_follow * fore
        if sat_any:
            drift = max(drift, 0.05)        # 限位饱和：QA 通道抬升（可观测）
        return GaitOutputs(
            delta_window_x=0.0,             # update() 聚合覆写
            view_yaw=self.view_yaw,
            pelvis_offset=(float(self._sway), float(self._dip)),
            bone_rotations=angles,
            left_foot_contact=self._feet["l"].contact.contact_type,
            right_foot_contact=self._feet["r"].contact.contact_type,
            foot_slide_drift_px=drift,
            bone_offsets=({leg.hip_bone: ((self.leg_shift + (self.far_leg_shift if side == "r" else 0.0))
                                          * self._env, 0.0) for side, leg in self.legs.items()}
                          if self.leg_shift or self.far_leg_shift else {}),
        )

    # ---------------- 观测（QA/测试） ----------------

    @property
    def knee_rate_max_observed(self) -> float:
        return self._knee_rate_max

    @property
    def velocity(self) -> float:
        return self._velocity

    @property
    def window_x_float(self) -> float:
        """内部浮点窗口累加器（呈现层据此 int 落位，见 update docstring）。"""
        return self._wx

    @property
    def envelope(self) -> float:
        return self._env

    @property
    def pelvis_pose(self) -> Tuple[float, float, float]:
        """(sway_x, dip_y, rot) 画布 px / 弧度。"""
        return (float(self._sway), float(self._dip), float(self._pelvis_rot))

    def foot_marker_positions(self, window_x: float) -> Dict[str, Tuple[float, float]]:
        """当前双脚活动接触标记世界坐标——测试/门禁观测用（不改内部状态）。"""
        out: Dict[str, Tuple[float, float]] = {}
        for side, foot in self._feet.items():
            kind = foot.contact.contact_type if foot.contact.is_locked \
                else ContactType.FLAT_SOLE
            m = self._marker_canvas(side, kind, foot.ankle_now, foot.pitch_now)
            out[side] = (float(window_x) + self.scale * float(m[0]), float(m[1]))
        return out
