"""数值飘字 HUD（v0.19.0 F1）——交互增量即时可见。

形态仿 BubbleWidget（无框 Tool 置顶层）。定位在宠物头顶右侧 76px，避开
头顶居中的气泡；正增量绿字上飘淡出、负增量红字下沉淡出，~1.2s 自动隐藏。
WA_TransparentForMouseEvents 点击穿透：不打断对宠物本体的交互。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QVariantAnimation
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_STYLE_POS = (
    "QLabel{background:rgba(45,45,48,200);color:#8ce99a;"
    "border-radius:10px;padding:4px 10px;font:bold 13px 'PingFang SC';}"
)
_STYLE_NEG = (
    "QLabel{background:rgba(45,45,48,200);color:#ff8787;"
    "border-radius:10px;padding:4px 10px;font:bold 13px 'PingFang SC';}"
)
# v0.19.1 F6/F7：拒绝/疲劳等中性态（不升不沉、白字原地淡出）
_STYLE_FLAT = (
    "QLabel{background:rgba(45,45,48,200);color:#f2f2f2;"
    "border-radius:10px;padding:4px 10px;font:bold 13px 'PingFang SC';}"
)

_RISE_PX = 46   # 正增量上飘行程
_SINK_PX = 34   # 负增量下沉行程


class FloatingTextWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)

        self._anchor: tuple | None = None  # (cx, pet_bottom_y, pet_height)
        self._rise = True
        self._flat = False
        self._t = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.valueChanged.connect(self._on_tick)
        self._anim.finished.connect(self.hide)

    def pop(self, text: str, delta: float = 0.0, tone: str | None = None,
            anchor: tuple | None = None, duration_ms: int = 1200) -> None:
        """弹出一条增量飘字。tone 显式指定（pos/neg/flat；v0.19.1 拒绝/疲劳
        用 flat），否则按 delta 推导：负数红字下沉，其余绿字上飘。

        连续触发（快速连点）重启动画——单实例复用，不叠死。
        """
        if tone is None:
            tone = "pos" if delta >= 0 else "neg"
        self._rise = tone != "neg"
        self._flat = tone == "flat"
        self._label.setText(text)
        self._label.setStyleSheet(
            {"pos": _STYLE_POS, "neg": _STYLE_NEG}.get(tone, _STYLE_FLAT))
        self.adjustSize()
        self._anchor = anchor
        self._apply_offset(0.0)
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(max(120, duration_ms))
        self._anim.start()

    def follow(self, anchor: tuple) -> None:
        """宠物移动时跟随（app._on_pet_moved 与气泡同通道）；未显示忽略。"""
        if not self.isVisible():
            return
        self._anchor = anchor
        self._apply_offset(self._t)

    def _on_tick(self, v) -> None:
        self._apply_offset(float(v))

    def _apply_offset(self, t: float) -> None:
        self._t = t
        base = self._base_pos()
        if base is None:
            return
        x, y = base
        # flat（拒绝/疲劳）不上飘也不下沉，原地淡出
        travel = 0 if self._flat else (-_RISE_PX if self._rise else _SINK_PX)
        self.move(x, int(y + travel * t))
        # 前 15% 行程全显，之后线性淡出
        self.setWindowOpacity(
            1.0 if t <= 0.15 else max(0.0, 1.0 - (t - 0.15) / 0.85))

    def _base_pos(self) -> tuple[int, int] | None:
        """宠物头顶右侧 76px（避开头顶居中的气泡）；贴右缘翻到左侧，
        贴顶钳在屏内。无锚点回退主屏右上角。多屏按宠物所在屏（同 B1）。"""
        if self._anchor is not None:
            cx, pet_y, pet_h = self._anchor
            screen = QGuiApplication.screenAt(
                QPoint(int(cx), int(pet_y))) or QGuiApplication.primaryScreen()
        else:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        g = screen.availableGeometry()
        w, h = self.width(), self.height()
        if self._anchor is not None:
            x = int(cx + 76)
            if x + w > g.x() + g.width() - 4:
                x = int(cx - 76 - w)
            y = int(pet_y - pet_h - 12 - h)
        else:
            x = g.x() + g.width() - w - 16
            y = g.y() + 16
        return x, max(g.y() + 4, y)
