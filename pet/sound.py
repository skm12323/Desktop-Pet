"""交互音效管线（v0.19.0 F4）——默认关，全程静默降级。

config ``sound`` 段：``{"enabled": false, "volume": 0.6}``。资产约定
``assets/sounds/<name>.wav``（pet/feed/clean/poke，0.19.1 起加 reject）；
缺 QtMultimedia 模块、缺资产文件均静默跳过，永不外抛（与 render3d 同
哲学）。QSoundEffect 实例按名缓存（需保活引用，播完自动停）。
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QUrl

log = logging.getLogger("pet")


class SoundFX:
    def __init__(self, cfg: dict | None = None, assets_root: str | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.volume = float(cfg.get("volume", 0.6))
        self.root = assets_root or os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "assets", "sounds"))
        self._cache: dict[str, object] = {}
        self._fx_cls = None
        self._error_status = None
        if not self.enabled:
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
            self._fx_cls = QSoundEffect
            self._error_status = QSoundEffect.Status.Error
        except Exception:
            log.warning("音效不可用（QtMultimedia 缺失/初始化失败），静音运行")

    def play(self, name: str) -> None:
        """播放 assets/sounds/<name>.wav；任何缺件静默返回。"""
        if not (self.enabled and self._fx_cls is not None):
            return
        fx = self._cache.get(name)
        if fx is None:
            path = os.path.join(self.root, f"{name}.wav")
            if not os.path.isfile(path):
                return
            fx = self._fx_cls()
            fx.setSource(QUrl.fromLocalFile(path))
            fx.setVolume(self.volume)
            self._cache[name] = fx
        if fx.status() == self._error_status:
            return
        fx.play()
