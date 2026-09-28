"""Record Direct3D11/OpenGL hardware skinned mesh animation for Adult rig. (spikes/qa_skinned_adult_visual.py)"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from pet.asset_provider import SpriteRef
from pet.rig.presenter import RigWindow, build_rig_window
from pet.rig.spec import load_rig_spec
from pet.rig.motion import MotionInputs
from pet.window import WindowBase


def main():
    out_dir = ROOT / "assets" / "rig_adult"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "adult_motion.gif"

    app = QApplication.instance() or QApplication([])

    spec = load_rig_spec(str(ROOT / "assets" / "rig" / "adult"), "adult")
    sprite = SpriteRef(path=spec.figures["healthy_neutral"], width=480, height=848)
    win = build_rig_window(WindowBase, sprite, "adult", defer_quick=False)
    win._motion_timer.stop()
    win.resize(480, 848)
    win.setWindowTitle("Desktop Pet Adult Visual QA")
    win.show()

    animation = []
    total_frames = 50

    print(f"Recording {total_frames} frames of Adult 2D Skinned Mesh motion...")

    for index in range(total_frames):
        app.processEvents()

        # Motion sequence:
        # 0..15: Idle breathing, looking right towards cursor
        # 16..30: Gaze tracking, looking left towards cursor
        # 31..42: Gaze tracking, looking down-center
        # 43..49: Settled gaze, natural blinking and gentle physics sway
        is_walking = False
        walk_hz = 0.0

        if index < 15:
            cursor = (750.0, 280.0)
        elif index < 30:
            cursor = (150.0, 300.0)
        elif index < 42:
            cursor = (480.0, 550.0)
        else:
            cursor = (480.0, 400.0)

        inputs = MotionInputs(
            walking=is_walking,
            walk_hz=walk_hz,
            facing=1,
            cursor_pos=cursor,
            pet_rect=(0.0, 0.0, 480.0, 848.0),
        )

        frame = win._engine.step(inputs, 66.0)
        win._push_frame(frame)
        win._quick.repaint()
        app.processEvents()

        img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        rgba = Image.frombytes("RGBA", (img.width(), img.height()), bytes(img.constBits()))

        # Composite over a clean soft background
        bg = Image.new("RGBA", rgba.size, (244, 246, 250, 255))
        bg.alpha_composite(rgba)
        # Resize to compact 360x636 for fluid preview
        animation.append(bg.convert("RGB").resize((360, 636), Image.Resampling.LANCZOS))

        if (index + 1) % 10 == 0:
            print(f"  Captured frame {index + 1}/{total_frames}...")

    win.close()
    win.deleteLater()

    print(f"Saving animated GIF to {gif_path}...")
    animation[0].save(
        gif_path,
        save_all=True,
        append_images=animation[1:],
        duration=66,
        loop=0,
        disposal=2
    )
    print(f"[DONE] Adult animated GIF saved: {gif_path}")


if __name__ == "__main__":
    main()
