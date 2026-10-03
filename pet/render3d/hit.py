"""部位级命中（M3 后启用）——设计占位。

D11：现状交互=整窗语义（单击=摸/双击=喂/右键=菜单），3D parity 零成本沿用；
本模块在将来需要"摸头"类部位交互时启用：由骨骼位置推出头/身两个胶囊体做
命中测试（不用网格求交，不用 2D 图层）。接口预留：hit_test(x, y) -> part|None。
"""

from __future__ import annotations

from typing import Optional


def hit_test(x: float, y: float) -> Optional[str]:
    """v0 占位：恒 None（部位级交互未启用，整窗语义由 window 层负责）。"""
    return None
