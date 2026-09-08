"""Live2DWindow —— Cubism Native 呈现后端（v0.15），``WindowBase`` 同接口替换。

复用手势/菜单/拖放，把画面驱动换成 ``QOpenGLWidget`` + ``live2d-py``：

* 自动眨眼 / 呼吸 / 物理（模型自带）
* 养成 Mood → expression；交互 / 小动作 → motion group
* 视线跟随光标（``Drag``）；聊天流式口型（``ParamMouthOpenY``）
* 行走倾斜 / 朝向镜像 / 空中标志由 ``set_motion_params`` 喂 FSM 实况
* 缺库 / 缺模型 / GL 初始化失败 → 回退 ``QLabel`` 帧路径，不阻断启动

``glInit`` / ``LoadModelJson`` 必须在有效 OpenGL 上下文里调用，因此真正加载
发生在 canvas.initializeGL；构造期只校验路径与 mapping。
"""

from __future__ import annotations

import logging
import math
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QSurfaceFormat

from ..asset_provider import SpriteRef
from ..window import WindowBase
from .spec import Live2DMapping, load_live2d_mapping, resolve_model_path

log = logging.getLogger("pet")

_live2d_mod = None
_live2d_inited = False


def _import_live2d():
    global _live2d_mod
    if _live2d_mod is not None:
        return _live2d_mod
    try:
        import live2d.v3 as live2d  # type: ignore
    except Exception as e:
        log.warning("live2d-py 不可用（%s），回退帧动画", e)
        return None
    _live2d_mod = live2d
    return live2d


def _ensure_framework():
    """进程内只 init 一次 Cubism Framework。"""
    global _live2d_inited
    live2d = _import_live2d()
    if live2d is None:
        return None
    if not _live2d_inited:
        live2d.init()
        _live2d_inited = True
    return live2d


def default_live2d_cfg() -> dict:
    return {
        "model": "",
        "mapping": "",
        "scale": 1.0,
        "offset": [0.0, -0.15],
        "gaze": True,
        "lip_sync": True,
        "auto_blink": True,
        "auto_breath": True,
        "idle": True,
    }


def build_live2d_window(base_cls, sprite: SpriteRef,
                        cfg: dict | None = None) -> WindowBase:
    """装配入口：任一环不满足即返回 ``base_cls(sprite)``。"""
    live2d = _ensure_framework()
    if live2d is None:
        return base_cls(sprite)
    cfg = dict(default_live2d_cfg(), **(cfg or {}))
    model_path = resolve_model_path(cfg)
    if not model_path:
        return base_cls(sprite)
    mapping = load_live2d_mapping(model_path, cfg.get("mapping") or "")
    if mapping is None:
        log.info("live2d mapping 不可用，回退帧动画")
        return base_cls(sprite)
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401
    except Exception as e:
        log.warning("QOpenGLWidget 不可用（%s），回退帧动画", e)
        return base_cls(sprite)
    win = Live2DWindow(sprite, mapping, cfg)
    log.info("live2d 后端就绪：%s", mapping.display_name or mapping.model_path)
    return win


def _make_canvas_class(QOpenGLWidget, live2d):
    class Live2DCanvas(QOpenGLWidget):
        def __init__(self, host: "Live2DWindow"):
            super().__init__(host)
            self._host = host
            self._model = None
            self._gl_ok = False
            self._last_err = ""
            fmt = QSurfaceFormat()
            fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
            fmt.setVersion(2, 1)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
            fmt.setAlphaBufferSize(8)
            fmt.setStencilBufferSize(8)
            fmt.setDepthBufferSize(16)
            fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
            self.setFormat(fmt)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self.setAttribute(Qt.WA_AlwaysStackOnTop, True)
            self.setUpdateBehavior(
                QOpenGLWidget.UpdateBehavior.NoPartialUpdate)
            self.setAutoFillBackground(False)

        @property
        def model(self):
            return self._model

        @property
        def gl_ok(self) -> bool:
            return self._gl_ok and self._model is not None

        def initializeGL(self) -> None:
            try:
                self.makeCurrent()
                live2d.glInit()
                model = live2d.LAppModel()
                model.LoadModelJson(self._host._mapping.model_path)
                w, h = max(1, self.width()), max(1, self.height())
                model.Resize(w, h)
                cfg = self._host._l2d_cfg
                try:
                    model.SetAutoBlinkEnable(bool(cfg.get("auto_blink", True)))
                    model.SetAutoBreathEnable(bool(cfg.get("auto_breath", True)))
                except Exception:
                    pass
                ox, oy = self._host._offset
                try:
                    model.SetOffset(float(ox), float(oy))
                    model.SetScale(float(self._host._scale))
                except Exception:
                    pass
                self._model = model
                self._gl_ok = True
                self._host._on_gl_ready()
            except Exception as e:
                self._last_err = str(e)
                self._gl_ok = False
                log.warning("live2d initializeGL 失败，回退 QLabel：%s", e,
                            exc_info=True)
                QTimer.singleShot(0, self._host._on_gl_failed)

        def resizeGL(self, w: int, h: int) -> None:
            if self._model is not None:
                try:
                    self._model.Resize(max(1, w), max(1, h))
                except Exception:
                    pass

        def paintGL(self) -> None:
            if not self.gl_ok:
                live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
                return
            host = self._host
            live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
            model = self._model
            try:
                host._pre_draw(model)
                model.Update()
                host._post_update(model)
                model.Draw()
            except Exception:
                log.warning("live2d 绘制异常", exc_info=True)

        def disposeGL(self) -> None:
            if self._model is None:
                return
            try:
                self.makeCurrent()
                self._model.DestroyRenderer()
            except Exception:
                pass
            self._model = None
            self._gl_ok = False
            try:
                live2d.glRelease()
            except Exception:
                pass

    return Live2DCanvas


class Live2DWindow(WindowBase):
    """Cubism 驱动的呈现窗。GL 失败时行为等同基类。"""

    _canvas = None
    _model_ready = False
    _gl_pending = True

    def __init__(self, sprite: SpriteRef, mapping: Live2DMapping,
                 cfg: dict | None = None):
        super().__init__(sprite)
        self._mapping = mapping
        self._l2d_cfg = dict(default_live2d_cfg(), **(cfg or {}))
        self._scale = float(self._l2d_cfg.get("scale", 1.0) or 1.0)
        off = self._l2d_cfg.get("offset") or [0.0, -0.15]
        try:
            self._offset = (float(off[0]), float(off[1]))
        except (TypeError, ValueError, IndexError):
            self._offset = (0.0, -0.15)
        self._walk_phase = 0.0
        self._mouth_hold = 0.0
        self._lip = 0.0
        self._tilt = 0.0
        self._walking = False
        self._airborne = False
        self._walk_hz = 0.0
        self._current_exp = ""
        self._idle_wanted = bool(self._l2d_cfg.get("idle", True))
        self._draw_timer = QTimer(self)
        self._draw_timer.setInterval(16)
        self._draw_timer.timeout.connect(self._request_draw)
        self._init_canvas()

    def _init_canvas(self) -> None:
        live2d = _import_live2d()
        if live2d is None:
            self._gl_pending = False
            return
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
        except Exception as e:
            log.warning("QOpenGLWidget 导入失败：%s", e)
            self._gl_pending = False
            return
        Canvas = _make_canvas_class(QOpenGLWidget, live2d)
        c = Canvas(self)
        c.setGeometry(0, 0, self.width(), self.height())
        c.show()
        self._canvas = c
        self._label.hide()
        self._draw_timer.start()

    @property
    def live2d_active(self) -> bool:
        c = self._canvas
        return bool(c is not None and getattr(c, "gl_ok", False))

    @property
    def rig_active(self) -> bool:
        """app._tick 用 rig_active 决定是否喂 set_motion_params——一并接上。"""
        return self.live2d_active

    def _on_gl_ready(self) -> None:
        self._gl_pending = False
        self._model_ready = True
        self._label.hide()
        if self._canvas is not None:
            self._canvas.setVisible(True)
        self._apply_facing()
        state = getattr(self, "_last_state", None)
        if state is not None:
            self._apply_state_expression(state)
        else:
            self._set_expression_id(
                self._mapping.expression_for("neutral") or "")
        self._maybe_idle(force=True)
        log.info("live2d GL 就绪：expressions=%s motions=%s",
                 list(self._canvas.model.GetExpressionIds()
                      if self._canvas and self._canvas.model else []),
                 list((self._canvas.model.GetMotionGroups() or {}).keys()
                      if self._canvas and self._canvas.model else []))

    def _on_gl_failed(self) -> None:
        self._gl_pending = False
        self._model_ready = False
        self._draw_timer.stop()
        if self._canvas is not None:
            self._canvas.hide()
        self._label.show()
        super().set_sprite(self._sprite)

    def _request_draw(self) -> None:
        if self._canvas is not None and self.live2d_active:
            self._canvas.update()

    def _model(self):
        c = self._canvas
        return c.model if c is not None else None

    # ---- 每帧参数 ----
    def _pre_draw(self, model) -> None:
        cfg = self._l2d_cfg
        if cfg.get("gaze", True):
            gp = QCursor.pos()
            lp = self.mapFromGlobal(gp)
            try:
                model.Drag(float(lp.x()), float(lp.y()))
            except Exception:
                pass
        if self._idle_wanted:
            try:
                if model.IsMotionFinished():
                    self._maybe_idle()
            except Exception:
                pass

    def _post_update(self, model) -> None:
        params = self._mapping.parameters
        mouth_id = params.get("mouth") or "ParamMouthOpenY"
        body_x = params.get("body_x") or "ParamBodyAngleX"
        angle_z = params.get("angle_z") or "ParamAngleZ"
        # 口型：聊天流式 + 吃鼠标保持
        lip = max(float(self._lip), float(self._mouth_hold))
        if self._mouth_hold > 0:
            self._mouth_hold = max(0.0, self._mouth_hold - 0.02)
        if self._l2d_cfg.get("lip_sync", True) and lip > 0.01:
            try:
                model.SetParameterValue(mouth_id, min(1.0, lip), 1.0)
            except Exception:
                pass
        # 行走/速度倾斜
        try:
            model.AddParameterValue(body_x, self._tilt * 0.08)
            model.AddParameterValue(angle_z, self._tilt * 0.04)
        except Exception:
            pass
        if self._walking:
            dt = 0.016
            hz = self._walk_hz if self._walk_hz > 0 else 1.2
            self._walk_phase = (self._walk_phase + hz * dt) % 1.0
            sway = math.sin(self._walk_phase * 2 * math.pi) * 4.0
            try:
                model.AddParameterValue(angle_z, sway * 0.15)
                model.AddParameterValue(body_x, sway * 0.08)
            except Exception:
                pass
        if self._airborne:
            try:
                model.AddParameterValue(params.get("angle_x") or "ParamAngleX",
                                        8.0)
            except Exception:
                pass

    def _maybe_idle(self, force: bool = False) -> None:
        if not self._idle_wanted:
            return
        model = self._model()
        if model is None:
            return
        ref = self._mapping.motion_for("idle")
        group = ref.group if ref is not None else "Idle"
        prio = ref.priority if ref is not None else 1
        try:
            if force or model.IsMotionFinished():
                model.StartRandomMotion(group, prio)
        except Exception:
            pass

    def _start_motion(self, action: str) -> bool:
        model = self._model()
        if model is None:
            return False
        ref = self._mapping.motion_for(action)
        if ref is None:
            return False
        try:
            if ref.index is not None:
                model.StartMotion(ref.group, int(ref.index), ref.priority)
            else:
                model.StartRandomMotion(ref.group, ref.priority)
            return True
        except Exception:
            log.warning("live2d StartMotion(%s) 失败", action, exc_info=True)
            return False

    def _set_expression_id(self, exp_id: str) -> None:
        model = self._model()
        if model is None or not exp_id:
            return
        if exp_id == self._current_exp:
            return
        try:
            ids = list(model.GetExpressionIds() or [])
            if ids and exp_id not in ids:
                return
            model.SetExpression(exp_id)
            self._current_exp = exp_id
        except Exception:
            log.warning("live2d SetExpression(%s) 失败", exp_id, exc_info=True)

    def _apply_state_expression(self, state) -> None:
        from ..pet_state import Branch, Mood
        from ..asset_provider import _mood_from_state

        mood = getattr(self, "_conversation_mood", None)
        if mood is None:
            try:
                mood = _mood_from_state(state)
            except Exception:
                mood = Mood.NEUTRAL
        mood_key = mood.value if hasattr(mood, "value") else str(mood)
        neglected = getattr(state, "branch", None) == Branch.NEGLECTED
        exp = self._mapping.expression_for(mood_key, neglected=neglected)
        if exp:
            self._set_expression_id(exp)

    def _apply_facing(self) -> None:
        model = self._model()
        if model is None:
            return
        d = int(getattr(self, "_facing", 1) or 1)
        try:
            model.SetScale(self._scale)
            model.SetScaleX(self._scale * (1 if d >= 0 else -1))
        except Exception:
            pass

    # ---------------- WindowBase 同接口 ----------------
    def set_sprite(self, sprite: SpriteRef) -> None:
        self._sprite = sprite
        if self.live2d_active:
            if (sprite.width, sprite.height) != (self.width(), self.height()):
                self.resize(sprite.width, sprite.height)
                self._label.resize(sprite.width, sprite.height)
                if self._canvas is not None:
                    self._canvas.setGeometry(0, 0, sprite.width, sprite.height)
            return
        super().set_sprite(sprite)

    def on_state_change(self, state) -> None:
        if state is None:
            return
        self._last_state = state
        if self.live2d_active:
            self._apply_state_expression(state)
            sprite = None
            if self._provider is not None:
                sprite = self._provider.get_static(
                    state,
                    mood_override=getattr(self, "_conversation_mood", None))
            if sprite is not None:
                self._static_sprite = sprite
                if (sprite.width, sprite.height) != (self.width(), self.height()):
                    self.set_sprite(sprite)
            return
        super().on_state_change(state)

    def play_frames(self, frames: list, loop: bool = False,
                    interval_ms: int = 150) -> None:
        if not self.live2d_active:
            super().play_frames(frames, loop, interval_ms)
            return
        if not frames:
            self.stop_frames()
            return
        if not self._frames:
            self._static_sprite = self._sprite
        self._frames = list(frames)
        self._frame_idx = 0
        self._frame_loop = bool(loop)
        self._frame_timer.stop()
        first = os.path.basename(getattr(frames[0], "path", "") or "")
        if any(s in first for s in ("_chew_", "_eat_mouse_")):
            self._mouth_hold = 0.85
        elif "_fall_" in first:
            self._airborne = True
        elif any(s in first for s in ("_stretch_", "_roll")):
            self._start_motion("stretch" if "_stretch_" in first else "roll")
        elif "_blink" in first:
            return
        elif "_walk_" in first:
            return

    def stop_frames(self) -> None:
        if not self.live2d_active:
            super().stop_frames()
            return
        self._frames = []
        self._frame_timer.stop()
        self._mouth_hold = 0.0
        self._maybe_idle()

    def set_facing(self, d: int) -> None:
        if d not in (-1, 1) or d == getattr(self, "_facing", 1):
            return
        self._facing = d
        if self.live2d_active:
            self._apply_facing()
            return
        super().set_facing(d)

    def set_motion_params(self, tilt_deg: float = 0.0, walking: bool = False,
                          airborne: bool = False, walk_hz: float = 0.0) -> None:
        self._tilt = float(tilt_deg)
        self._walking = bool(walking)
        self._airborne = bool(airborne)
        self._walk_hz = float(walk_hz)

    def part_walk_active(self) -> bool:
        """挡住 frames 行走环——步态由参数/动作组驱动。"""
        return self.live2d_active

    def is_playing(self) -> bool:
        if not self.live2d_active:
            return super().is_playing()
        if self._frames:
            return True
        model = self._model()
        if model is None:
            return False
        try:
            return not model.IsMotionFinished()
        except Exception:
            return False

    def play_interaction(self, kind: str) -> None:
        """摸摸/喂食/洗澡/戳 → 对应 motion（缺映射则忽略）。"""
        if not self.live2d_active:
            return
        key = {"pet": "pat", "feed": "feed", "clean": "clean",
               "poke": "poke"}.get(kind, kind)
        self._start_motion(key)

    def set_lip_open(self, value: float) -> None:
        """聊天流式口型，0..1。"""
        self._lip = max(0.0, min(1.0, float(value)))

    def hit_at(self, x: float, y: float) -> str | None:
        """窗口坐标命中 HitArea 名（Head/Body…）。"""
        model = self._model()
        if model is None:
            return None
        for name in (self._mapping.hit_areas or {"Head": "pat", "Body": "poke"}):
            try:
                if model.HitTest(name, float(x), float(y)):
                    return name
            except Exception:
                continue
        return None

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._canvas is not None:
            self._canvas.setGeometry(0, 0, self.width(), self.height())

    def closeEvent(self, event):  # noqa: N802
        self.dispose_presentation()
        super().closeEvent(event)

    def dispose_presentation(self) -> None:
        self._draw_timer.stop()
        c = self._canvas
        if c is not None:
            try:
                c.disposeGL()
            except Exception:
                pass
            self._canvas = None
        self._model_ready = False
