"""语义契约 v1 —— 2D rig 与 3D renderer 唯一允许共享的东西（three_d D04/S2.1）。

设计铁律（见 three_d/wiki/设计-语义契约草案.md）：
* 纯 dataclass + 校验，**禁止 import 任何 Qt/引擎符号**——本文件必须能在
  无显示、无 PySide6 的环境被单测覆盖（tests/test_scene_contract.py）。
* 契约携带**高层语义**，不携带引擎表示：姿态是动作语义（action_id/相位/
  速度/朝向），不是骨骼四元数；骨骼展开是各 renderer 适配器的私事。
* 慢变量（光照/天气）与快变量（姿态/表情）分通道——更新频率、订阅方式不同。
* 词表来源：action_id 对齐 BehaviorFSM ActionType + 桌宠品类最小清单
  （调研-建模页 §六，VPet/Desktop Mate 佐证）；表情对齐 VRM 1.0 expressions
  标准预设（调研-建模页 §五）。

版本：v1（对应 wiki 契约草案 v0 的冻结版）。字段只增不改语义；破坏性变更
必须升 SCHEMA_VERSION 并在 wiki log 记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

SCHEMA_VERSION = 1

# ---- 动作词表（PoseSemantics.action_id 合法值） ------------------------------
# idle 变体以 "idle" 前缀扩展（idle_sit/idle_sleep…），renderer 未实现时回落 idle。
ACTION_IDLE = "idle"
ACTION_WALK = "walk"
ACTION_TURN = "turn"
ACTION_DRAG = "drag"        # 提起/悬晃/放下三段由 phase 区分
ACTION_PAT = "pat"          # 摸头反应
ACTION_CLICK = "click"      # 点击反应
ACTION_IDS: frozenset[str] = frozenset(
    {ACTION_IDLE, ACTION_WALK, ACTION_TURN, ACTION_DRAG, ACTION_PAT, ACTION_CLICK}
)

# ---- 表情预设词表（对齐 VRM 1.0 expressions 标准预设） ------------------------
EXP_HAPPY = "happy"
EXP_ANGRY = "angry"
EXP_SAD = "sad"
EXP_RELAXED = "relaxed"
EXP_SURPRISED = "surprised"
EXP_NEUTRAL = "neutral"
EXPRESSION_PRESETS: frozenset[str] = frozenset(
    {EXP_HAPPY, EXP_ANGRY, EXP_SAD, EXP_RELAXED, EXP_SURPRISED, EXP_NEUTRAL}
)
# VRM 另有 blink/lookUp 等功能预设——它们由本契约的独立字段驱动
# （blink_progress / gaze_x / gaze_y），不进 emotion_label 词表。


def _clamp(name: str, value: float, lo: float, hi: float) -> float:
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value!r} 越界，合法区间 [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class LightWeatherState:
    """慢变量通道：光照/天气（秒级采样 + 平滑，生产者 sun.py / wind.py / 天气源）。

    字段是**语义量**，不是渲染参数——renderer 自行决定 toon 阶梯数、湿身高光等。
    """

    sun_azimuth_deg: float = 0.0     # 太阳方位角（sun.py 已有），0-360
    sun_elevation_deg: float = 0.0   # 太阳高度角，-90-90（负=夜间，renderer 处理昼夜）
    sun_intensity: float = 1.0       # 0-1，云量/昼夜调制后的强度
    color_temp_k: float = 6500.0     # 色温，1500-12000（晨昏偏暖）
    cloud_cover: float = 0.0         # 0-1
    rain_rate: float = 0.0           # 0-1（前景雨丝走 2D 叠加层，3D 只消费此参数）
    wind_speed: float = 0.0          # m/s（wind.py 已有），≥0
    wetness: float = 0.0             # 0-1 材质参数（雨淋湿身），L3 光照级才消费

    def __post_init__(self) -> None:
        _clamp("sun_azimuth_deg", self.sun_azimuth_deg, 0.0, 360.0)
        _clamp("sun_elevation_deg", self.sun_elevation_deg, -90.0, 90.0)
        _clamp("sun_intensity", self.sun_intensity, 0.0, 1.0)
        _clamp("color_temp_k", self.color_temp_k, 1500.0, 12000.0)
        _clamp("cloud_cover", self.cloud_cover, 0.0, 1.0)
        _clamp("rain_rate", self.rain_rate, 0.0, 1.0)
        if self.wind_speed < 0.0:
            raise ValueError(f"wind_speed={self.wind_speed!r} 不能为负")
        _clamp("wetness", self.wetness, 0.0, 1.0)


@dataclass(frozen=True)
class PoseSemantics:
    """快变量通道：姿态语义（逐帧，生产者 MotionEngine——唯一）。

    phase 是步频/动作循环的锚点（0-1 环形）；speed_px_s 供步幅-位移同步
    （2D 侧身线踩过的坑，3D 同样要防脚底打滑）。
    """

    action_id: str = ACTION_IDLE
    phase: float = 0.0               # 0-1 环形
    speed_px_s: float = 0.0          # ≥0
    view_yaw_deg: float = 0.0        # 连续朝向（现有 viewYaw 语义平移），-180-180
    squash: float = 0.0              # 挤压拉伸系数，-1-1（现有 squashAt 语义）
    ground_shift_px: float = 0.0     # 贴地偏移（现有 skinnedGroundShift 语义）

    def __post_init__(self) -> None:
        if self.action_id not in ACTION_IDS:
            raise ValueError(
                f"action_id={self.action_id!r} 不在词表 {sorted(ACTION_IDS)}"
            )
        _clamp("phase", self.phase, 0.0, 1.0)
        if self.speed_px_s < 0.0:
            raise ValueError(f"speed_px_s={self.speed_px_s!r} 不能为负")
        _clamp("view_yaw_deg", self.view_yaw_deg, -180.0, 180.0)
        _clamp("squash", self.squash, -1.0, 1.0)

    def with_phase(self, phase: float) -> "PoseSemantics":
        """逐帧推进相位的便捷方法（frozen dataclass 的 replace 封装）。"""
        return replace(self, phase=phase)


@dataclass(frozen=True)
class ExpressionState:
    """快变量通道：表情/视线（生产者 chat_emotion + 程序化眨眼定时）。

    emotion_label=None 即中性（chat_emotion 的 5 分钟回落语义直接映射）；
    blink/gaze 由程序化层驱动，与情绪叠加（D10：表情平行于动作与情绪）。
    """

    emotion_label: str | None = None      # None=中性；否则必须在 EXPRESSION_PRESETS
    blink_progress: float = 0.0           # 0-1（0=睁眼，1=全闭）
    gaze_x: float = 0.0                   # -1..1（现有 lookAtX 语义）
    gaze_y: float = 0.0                   # -1..1

    def __post_init__(self) -> None:
        if self.emotion_label is not None and self.emotion_label not in EXPRESSION_PRESETS:
            raise ValueError(
                f"emotion_label={self.emotion_label!r} 不在词表 {sorted(EXPRESSION_PRESETS)}"
            )
        _clamp("blink_progress", self.blink_progress, 0.0, 1.0)
        _clamp("gaze_x", self.gaze_x, -1.0, 1.0)
        _clamp("gaze_y", self.gaze_y, -1.0, 1.0)


@dataclass(frozen=True)
class SceneState:
    """聚合快照：一次投递给 renderer（EngineBridge 分发，见 S2.9）。

    schema_version 随实例走，renderer 据此拒绝不认识的版本（降级而非猜）。
    """

    light: LightWeatherState = field(default_factory=LightWeatherState)
    pose: PoseSemantics = field(default_factory=PoseSemantics)
    expression: ExpressionState = field(default_factory=ExpressionState)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={self.schema_version} 与本契约 v{SCHEMA_VERSION} 不符，"
                "renderer 应拒绝并触发降级"
            )


@runtime_checkable
class SceneRenderer(Protocol):
    """2D presenter 与 3D renderer 都实现此协议（EngineBridge 按 flag 选实现）。

    apply() 不抛错承诺由调用方兜底实现（D04 铁律：任一环失败静默降级），
    协议本身不声明异常——is_ready()=False 是 renderer 表达"我不行了"的正道。
    """

    def apply(self, state: SceneState) -> None: ...

    def is_ready(self) -> bool: ...


# ---- 便捷构造 ---------------------------------------------------------------

def neutral_state() -> SceneState:
    """全中性快照（启动首帧 / 降级回落时的安全值）。"""
    return SceneState(
        light=LightWeatherState(),
        pose=PoseSemantics(action_id=ACTION_IDLE),
        expression=ExpressionState(),
    )
