"""Capture the production ADULT rest pose at its 256x256 window size."""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("QT_QUICK_BACKEND", "rhi")
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from pet.asset_provider import SpriteRef
from pet.rig.motion import MotionInputs
from pet.rig.presenter import build_rig_window
from pet.rig.spec import load_rig_spec
from pet.window import WindowBase

OUT = ROOT / "spikes/_qa/adult_rest_fix"


def render(widget, app, frame):
    widget._push_frame(frame)
    app.processEvents()
    img = widget._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
    assert not img.isNull()
    rgba = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return Image.fromarray(rgba[:, :img.width()].copy(), "RGBA").resize((widget.width(), widget.height()), Image.Resampling.LANCZOS)


def composited(image, bg):
    result = Image.new("RGBA", image.size, (*bg, 255))
    result.alpha_composite(image)
    return result.convert("RGB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    spec = load_rig_spec(str(ROOT / "assets/rig/adult"), "adult")
    win = build_rig_window(WindowBase, SpriteRef(spec.figures["healthy_neutral"], 256, 256), "adult")
    assert win.rig_active and win._skinned_item is not None and win._skinned_item._rt is not None
    win._motion_timer.stop()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()
    inputs = MotionInputs(walking=False, grounded=True)
    snapshots = []
    for i in range(91):
        frame = win._engine.step(inputs, 33.0)
        frame.blink_progress = 0.0
        if i in (0, 30, 90):
            snapshots.append((i, frame, render(win, app, frame)))
    for index, frame, image in snapshots:
        assert abs(frame.body_y) < 1e-7 and abs(frame.body_angle) < 1e-7
        assert frame.bone_angles["upper_arm_l"] > 0 and frame.bone_angles["forearm_l"] < 0
        assert frame.bone_angles["upper_arm_r"] < 0 and frame.bone_angles["forearm_r"] > 0
        pixels = np.array(image)[:, :, 3]
        assert np.flatnonzero((pixels[:, 90:166] > 20).any(axis=1))[-1] >= 253
        image.save(OUT / f"idle_{index:03d}.png")

    image = snapshots[0][2]
    alpha = np.array(image)[:, :, 3]
    sole_rows = np.flatnonzero((alpha[:, 90:166] > 20).any(axis=1))
    assert len(sole_rows) and int(sole_rows[-1]) >= 253, f"shoes not grounded: {sole_rows[-1]}"
    assert abs(float(win._root.property("skinnedGroundShift")) - 256 * (1696 - 1608) / 1696) < 0.05

    before_path = ROOT / "spikes/_qa/adult_review/actual_256.png"
    before = Image.open(before_path).convert("RGB") if before_path.exists() else None
    after = composited(image, (240, 243, 248))
    after.save(OUT / "after_256.png")
    dark = composited(image, (40, 44, 55))
    dark.save(OUT / "after_dark_256.png")
    if before:
        sheet = Image.new("RGB", (512, 282), (240, 243, 248))
        sheet.paste(before, (0, 26))
        sheet.paste(after, (256, 26))
        d = ImageDraw.Draw(sheet)
        d.text((10, 8), "Before", fill=(30, 40, 55))
        d.text((266, 8), "After", fill=(30, 40, 55))
        sheet.save(OUT / "before_after_256.png")

    win.set_facing(-1)
    mirrored = render(win, app, frame)
    composited(mirrored, (240, 243, 248)).save(OUT / "after_facing_left_256.png")
    win.set_facing(1)
    win.resize(480, 848)
    app.processEvents()
    larger = render(win, app, frame)
    composited(larger, (240, 243, 248)).save(OUT / "after_480x848.png")
    shift = float(win._root.property("skinnedGroundShift"))
    assert abs(shift - 848 * (1696 - 1608) / 1696) < 0.05
    young = load_rig_spec(str(ROOT / "assets/rig/young"), "young")
    win.set_stage("young")
    win.set_sprite(SpriteRef(young.figures["healthy_neutral"], 192, 192))
    app.processEvents()
    assert abs(float(win._root.property("skinnedGroundShift"))) < 1e-7
    win.set_stage("adult")
    win.set_sprite(SpriteRef(spec.figures["healthy_neutral"], 256, 256))
    app.processEvents()
    assert abs(float(win._root.property("skinnedGroundShift"))
               - win.height() * (1696 - 1608) / 1696) < 0.05
    win.close()
    print(f"ground shift at 480x848={shift:.3f}, 256-foot bottom row={sole_rows[-1]}, output={OUT}")


if __name__ == "__main__":
    main()
