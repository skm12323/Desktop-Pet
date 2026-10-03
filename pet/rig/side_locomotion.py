"""ADULT / FINAL 侧身行走编排：正面骨骼 ⇄ 转身片段 ⇄ 侧身骨骼（gait.py）。

方案：docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §2 / §6。本模块只做纯逻辑
（无 Qt），呈现层每拍调用 ``update`` 并按返回的 ``LocoFrame`` 摆放画面：

    FRONT ──行走意图──▶ SETTLE（正面姿态 200 ms 归位到静止）──▶ TURN_OUT（正→侧片段）
      ▲                                                                    │
      └ TURN_IN（侧→正片段）◀ SIDE_SETTLE（侧身归位 200 ms）◀ 待机超时/反向 ◀ SIDE（GaitSolver）◀┘

  片段两端 = 骨骼静止渲染；回到骨骼后次级运动（呼吸/弹簧/眨眼）300 ms 渐入。

- 片段首尾帧与两套骨骼的静止渲染逐像素一致（tools/process_turn_clip.py 的端点形变），
  切换处直接硬切，无需淡化。
- 向左行走 = 侧身骨骼与片段整体镜像；步态求解器在镜像坐标系运行（窗口 x 取反喂入、
  位移取反输出），锁脚数学不变。
- 拖拽 / 离地 = 立即回正面（``interrupt``）。
- 会话期间窗口 x 由本模块驱动（``controls_x``）：行为层只给意图，从窗口回读位置。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .gait import GaitOutputs, GaitPhaseState, GaitSolver


class LocoState(Enum):
    FRONT = "front"
    SETTLE = "settle"
    TURN_OUT = "turn_out"
    SIDE = "side"
    SIDE_SETTLE = "side_settle"
    TURN_IN = "turn_in"


@dataclass
class ClipFrame:
    path: str
    canvas_rect: tuple            # (x, y, w, h) in rig-canvas px


class TurnClip:
    """clip.json（tools/process_turn_clip.py）→ 逐帧画布矩形与文件路径。"""

    def __init__(self, clip_dir: str):
        with open(os.path.join(clip_dir, "clip.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.fps = float(data["fps"])
        sp = data["space"]
        s = float(sp["scale_per_canvas_px"])
        ox, oy = (float(v) for v in sp["origin_canvas"])
        self.frames: list[ClipFrame] = []
        for fr in data["frames"]:
            (u, v), (w, h) = fr["offset_px"], fr["size_px"]
            self.frames.append(ClipFrame(
                path=os.path.normpath(os.path.join(clip_dir, fr["file"])),
                canvas_rect=(ox + u / s, oy + v / s, w / s, h / s)))
        if len(self.frames) < 2:
            raise ValueError(f"clip {clip_dir} has < 2 frames")

    @property
    def duration(self) -> float:
        return (len(self.frames) - 1) / self.fps

    def index_at(self, t: float) -> int:
        return max(0, min(len(self.frames) - 1, int(math.floor(t * self.fps + 1e-6))))


@dataclass
class LocoFrame:
    state: LocoState
    mode: str                                  # "front" | "clip" | "side"
    facing: int = 1                            # 会话方向（+1 右 / −1 左）
    settle: float = 0.0                        # 0..1 当前骨骼姿态向静止的混合权重（1 = 静止渲染）
    gait_scale: float = 1.0                    # 侧身收尾时步态骨骼角/骨盆向 0 缩放
    clip: Optional[TurnClip] = None
    clip_index: int = 0
    clip_alpha: float = 1.0                    # 片段两端交叉淡化（其下为静止姿态的骨骼）
    under: str = ""                            # 片段下方垫的骨骼："front" | "side" | ""
    under_alpha: float = 1.0                   # 垫底骨骼不透明度（片段已不透明时再淡出，消边缘软硬差）
    gait: Optional[GaitOutputs] = None
    window_x: Optional[float] = None           # 会话驱动的窗口 x（浮点，调用方取整落位）
    controls_x: bool = False
    events: list = field(default_factory=list)


class SideLocomotion:
    def __init__(self, side_spec: dict, clip_out: TurnClip, clip_in: TurnClip,
                 window_scale: float, settle_s: float = 0.2, side_idle_timeout_s: float = 4.0,
                 fade_in_s: float = 0.3, crossfade_s: float = 0.1):
        self._spec = side_spec
        self.clip_out = clip_out
        self.clip_in = clip_in
        self.scale = float(window_scale)
        self.settle_s = float(settle_s)
        self.side_idle_timeout_s = float(side_idle_timeout_s)
        self.fade_in_s = float(fade_in_s)          # 片段结束后次级运动渐入（§6）
        self.crossfade_s = float(crossfade_s)      # 片段两端交叉淡化 ≈ 3 帧 @30fps
        timing = side_spec.get("locomotion") or {}
        self.clip_rate = max(1.0, min(1.5, float(timing.get("clip_rate", 1.0))))
        self.reverse_clip_rate = max(1.0, min(2.0, float(timing.get("reverse_clip_rate", 1.0))))
        self.reverse_settle_s = max(.05, min(self.settle_s, float(timing.get("reverse_settle_s", self.settle_s))))
        self._reverse_fast = False
        self._front_fade = 0.0                      # 正面刚从片段返回：剩余渐入时间
        self._last_gait: Optional[GaitOutputs] = None
        self.state = LocoState.FRONT
        self.dir = 1
        self._t = 0.0
        self._solver: Optional[GaitSolver] = None
        self._pending_dir = 0          # 反向：转回正面后要去的方向
        self._win_x = 0.0              # 会话内窗口 x 浮点累加器
        self._idle_t = 0.0
        self._side_t = 0.0

    # ---------------- 外部接口 ----------------

    @property
    def active(self) -> bool:
        return self.state is not LocoState.FRONT

    def set_window_scale(self, scale: float) -> None:
        """窗口尺寸变化（256/384/512）：仅在正面态生效，会话中不换尺度。"""
        if self.state is LocoState.FRONT:
            self.scale = float(scale)

    def interrupt(self) -> LocoFrame:
        """拖拽 / 离地：立即回正面，丢弃步态会话。"""
        was = self.state
        self._enter(LocoState.FRONT)
        self._solver = None
        self._pending_dir = 0
        self._reverse_fast = False
        return LocoFrame(LocoState.FRONT, "front", self.dir,
                         events=[f"interrupt:{was.value}"] if was is not LocoState.FRONT else [])

    def update(self, dt: float, desired_vx: float, window_x: float,
               grounded: bool = True, dragged: bool = False) -> LocoFrame:
        dt = min(max(float(dt), 0.0), 0.25)
        if dragged or not grounded:
            return self.interrupt()
        want = 0 if abs(desired_vx) <= 1.0 else (1 if desired_vx > 0 else -1)
        events: list = []
        s = self.state
        rate = self.reverse_clip_rate if self._reverse_fast else self.clip_rate
        self._t += dt * (rate if s in (LocoState.TURN_IN, LocoState.TURN_OUT) else 1.0)
        if self._pending_dir and (not want or want == self.dir):
            self._pending_dir = 0              # cancellation / latest intent takes precedence
            if s is LocoState.SIDE:
                self._reverse_fast = False

        if s is LocoState.FRONT:
            fade = 0.0
            if self._front_fade > 0.0:
                self._front_fade = max(0.0, self._front_fade - dt)
                u = self._front_fade / max(self.fade_in_s, 1e-6)
                fade = u * u * (3 - 2 * u)
            if want:
                self.dir = want
                self._win_x = float(window_x)
                self._enter(LocoState.SETTLE)
                self._t = self.settle_s * self._smooth_inv(fade)   # 从当前混合度继续归位
                self._front_fade = 0.0
                events.append("settle")
                return self._front_frame(settle=fade, events=events, controls=True)
            return self._front_frame(fade, events)

        if s is LocoState.SETTLE:
            if not want:                                   # 意图取消：直接回正面
                self._enter(LocoState.FRONT)
                return self._front_frame(0.0, ["cancel"])
            if want != self.dir:
                self.dir = want
            w = min(1.0, self._t / max(self.settle_s, 1e-6))
            if self._t >= self.settle_s:
                self._enter(LocoState.TURN_OUT)
                events.append("turn_out")
                return self._clip_frame(self.clip_out, 0, events)
            return self._front_frame(w * w * (3 - 2 * w), events, controls=True)

        if s is LocoState.TURN_OUT:
            if self._t >= self.clip_out.duration:
                self._start_side()
                self._reverse_fast = False
                events.append("side")
                return self._side_step(0.0, desired_vx, events)
            return self._clip_frame(self.clip_out, self.clip_out.index_at(self._t), events)

        if s is LocoState.SIDE:
            if want and want != self.dir:
                self._pending_dir = want                   # 反向：先停步，再转回正面
                self._reverse_fast = True
            return self._side_step(dt, desired_vx, events)

        if s is LocoState.SIDE_SETTLE:
            duration = self.reverse_settle_s if self._reverse_fast else self.settle_s
            w = min(1.0, self._t / max(duration, 1e-6))
            w = w * w * (3 - 2 * w)
            if self._t >= duration:
                self._enter(LocoState.TURN_IN)
                events.append("turn_in")
                return self._clip_frame(self.clip_in, 0, events)
            return LocoFrame(self.state, "side", self.dir, settle=w, gait=self._last_gait,
                             gait_scale=1.0 - w, window_x=self._win_x, controls_x=True,
                             events=events)

        # TURN_IN
        if self._t >= self.clip_in.duration:
            self._enter(LocoState.FRONT)
            self._solver = None
            self._front_fade = self.fade_in_s
            events.append("front")
            nxt = want                                    # never execute a cancelled stale reversal
            self._pending_dir = 0
            if nxt:                                        # 反向 / 立刻再走：直接下一次会话
                self.dir = nxt
                self._enter(LocoState.SETTLE)
                self._t = self.settle_s                    # 仍是静止姿态：下一拍直接进片段
                self._front_fade = 0.0
                events.append("settle")
                return self._front_frame(1.0, events, controls=True)
            self._reverse_fast = False
            return self._front_frame(1.0, events)
        return self._clip_frame(self.clip_in, self.clip_in.index_at(self._t), events)

    # ---------------- 内部 ----------------

    def _enter(self, st: LocoState) -> None:
        self.state = st
        self._t = 0.0

    def _front_frame(self, settle: float, events: list, controls: bool = False) -> LocoFrame:
        return LocoFrame(self.state, "front", self.dir, settle=settle,
                         window_x=self._win_x if controls else None,
                         controls_x=controls, events=events)

    def _clip_frame(self, clip: TurnClip, idx: int, events: list) -> LocoFrame:
        """片段帧 + 两端交叉淡化：开头在出发骨骼（静止）上淡入，结尾在到达骨骼（静止）上淡出。"""
        start, end = ("front", "side") if clip is self.clip_out else ("side", "front")
        f = max(self.crossfade_s, 1e-6)
        t = self._t if self.state in (LocoState.TURN_OUT, LocoState.TURN_IN) else 0.0
        d = clip.duration
        alpha, under, ua = 1.0, "", 1.0
        # 两段式：① 片段在不透明骨骼上淡入；② 骨骼在已不透明的片段下淡出（片段剪影边缘
        # 比骨骼软，一步撤掉骨骼会让整圈描边跳一下）。结尾对称倒放。
        if t < f:
            alpha, under = max(t / f, 0.0), start
        elif t < 2 * f:
            under, ua = start, max(1.0 - (t - f) / f, 0.0)
        elif t > d - f:
            alpha, under = max((d - t) / f, 0.0), end
        elif t > d - 2 * f:
            under, ua = end, min(1.0, (t - (d - 2 * f)) / f)
        return LocoFrame(self.state, "clip", self.dir, clip=clip, clip_index=idx,
                         clip_alpha=alpha, under=under, under_alpha=ua,
                         window_x=self._win_x, controls_x=True, events=events)

    def _start_side(self) -> None:
        self._enter(LocoState.SIDE)
        self._solver = GaitSolver(self._spec, window_scale=self.scale)
        self._idle_t = 0.0
        self._side_t = 0.0

    def _side_step(self, dt: float, desired_vx: float, events: list) -> LocoFrame:
        solver = self._solver
        assert solver is not None
        same_dir = desired_vx * self.dir > 1.0
        v = abs(desired_vx) if (same_dir and not self._pending_dir) else 0.0
        # 镜像坐标系：向左会话把窗口 x 取反喂入，位移取反输出
        out = solver.update(dt, v, (int(round(self.dir * self._win_x)), 0),
                            is_grounded=True, is_dragged=False)
        self._last_gait = out
        self._win_x = self.dir * solver.window_x_float
        idle = solver.state in (GaitPhaseState.IDLE_SIDE, GaitPhaseState.IDLE_FRONT)
        self._idle_t = self._idle_t + dt if (idle and v == 0.0) else 0.0
        if (self._pending_dir and idle) or self._idle_t >= self.side_idle_timeout_s:
            # 先归位到侧身静止姿态（转回片段首帧 = 侧身静止渲染），再放片段
            self._enter(LocoState.SIDE_SETTLE)
            events.append("side_settle:" + ("reverse" if self._pending_dir else "timeout"))
            return LocoFrame(self.state, "side", self.dir, settle=0.0, gait=out,
                             window_x=self._win_x, controls_x=True, events=events)
        self._side_t += dt
        u = max(0.0, 1.0 - self._side_t / max(self.fade_in_s, 1e-6))   # 片段后次级运动渐入
        return LocoFrame(self.state, "side", self.dir, settle=u * u * (3 - 2 * u), gait=out,
                         window_x=self._win_x, controls_x=True, events=events)

    @staticmethod
    def _smooth_inv(y: float) -> float:
        """smoothstep 的反函数（二分），y ∈ [0, 1]。"""
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if mid * mid * (3 - 2 * mid) < y:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
