"""Test runtime stage switching between Young and Adult 2D skinned meshes. (spikes/test_stage_switch.py)"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from pet.asset_provider import SpriteRef
from pet.rig.presenter import RigWindow, build_rig_window
from pet.window import WindowBase

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


def main():
    global passed, failed
    app = QApplication.instance() or QApplication([])

    print("== 1. 启动 Young 幼年体 RigWindow ==")
    sprite_young = SpriteRef(
        path=os.path.abspath("assets/rig/young/figs/healthy_neutral.png"),
        width=192, height=192
    )
    win = build_rig_window(WindowBase, sprite_young, "young", defer_quick=False)
    check("win.rig_active 活性", win.rig_active)
    check("当前阶段为 young", win._spec.stage == "young")
    check("幼年体 sourceFacing 为 -1", win._root.property("sourceFacing") == -1)
    check("幼年体蒙皮层数为 20", len(win._skinned_item._rt.layers) == 20)

    win.resize(400, 400)
    win.show()
    for _ in range(5):
        app.processEvents()
        win._motion_tick()
        win._quick.repaint()

    print("== 2. 运行时动态换档切换到 Adult 青年体 ==")
    sprite_adult = SpriteRef(
        path=os.path.abspath("assets/rig/adult/figs/healthy_neutral.png"),
        width=240, height=424
    )
    win.set_stage("adult")
    win.set_sprite(sprite_adult)
    check("切换后当前阶段为 adult", win._spec.stage == "adult")
    check("青年体 sourceFacing 为 1", win._root.property("sourceFacing") == 1)
    check("青年体蒙皮层数为 21", len(win._skinned_item._rt.layers) == 21)

    for _ in range(5):
        app.processEvents()
        win._motion_tick()
        win._quick.repaint()

    img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
    check("换档到 adult 后离屏抓图非空", not img.isNull())

    print("== 3. 运行时平滑回退换档到 Young 幼年体 ==")
    win.set_stage("young")
    win.set_sprite(sprite_young)
    check("切回后当前阶段为 young", win._spec.stage == "young")
    check("切回幼年体 sourceFacing 为 -1", win._root.property("sourceFacing") == -1)
    check("切回幼年体蒙皮层数为 20", len(win._skinned_item._rt.layers) == 20)

    for _ in range(5):
        app.processEvents()
        win._motion_tick()
        win._quick.repaint()

    win.close()
    win.deleteLater()

    print(f"\n== 换档测试结果：{passed} 通过 / {failed} 失败 ==")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
