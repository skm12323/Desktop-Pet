"""Render candidate ADULT elbow poses against the current production mesh."""
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


def grab(win, app, frame):
    win._push_frame(frame)
    app.processEvents()
    img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
    rgba = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    pil = Image.fromarray(rgba[:, :img.width()].copy(), "RGBA")
    return pil.resize((win.width(), win.height()), Image.Resampling.LANCZOS)


def main():
    out = ROOT / "spikes/_qa/adult_rest_fix"
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    spec = load_rig_spec(str(ROOT / "assets/rig/adult"), "adult")
    win = build_rig_window(WindowBase, SpriteRef(spec.figures["healthy_neutral"], 480, 848), "adult")
    win._motion_timer.stop()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()
    frame = win._engine.step(MotionInputs(walking=False, grounded=True), 33.0)
    frame.blink_progress = 0
    variants = [
        ("current", (-9, 4, 8, -3)),
        ("V1", (0, -8, 0, 8)),
        ("V2", (4, -12, -4, 12)),
        ("V3", (6, -15, -6, 15)),
        ("V4", (2, -14, -2, 14)),
        ("V5", (-3, -9, 3, 9)),
    ]
    sheet = Image.new("RGB", (6 * 280, 370), (240, 243, 248))
    draw = ImageDraw.Draw(sheet)
    pair = Image.new("RGB", (960, 878), (240, 243, 248))
    pair_draw = ImageDraw.Draw(pair)
    for i, (name, angles) in enumerate(variants):
        for bone, angle in zip(("upper_arm_l", "forearm_l", "upper_arm_r", "forearm_r"), angles):
            frame.bone_angles[bone] = float(angle)
        image = grab(win, app, frame)
        crop = image.crop((0, 175, 480, 715)).resize((280, 315), Image.Resampling.LANCZOS)
        bg = Image.new("RGBA", crop.size, (240, 243, 248, 255))
        bg.alpha_composite(crop)
        sheet.paste(bg.convert("RGB"), (i * 280, 38))
        draw.text((i * 280 + 8, 10), f"{name}: {angles}", fill=(30, 40, 55))
        if name in ("current", "V2"):
            full = Image.new("RGBA", image.size, (240, 243, 248, 255))
            full.alpha_composite(image)
            x = 0 if name == "current" else 480
            pair.paste(full.convert("RGB"), (x, 30))
            pair_draw.text((x + 10, 8), "Previous" if name == "current" else "Outward elbow", fill=(30, 40, 55))
    sheet.save(out / "elbow_variants.jpg", quality=95)
    pair.save(out / "elbow_before_after_480.png")
    win.close()
    print(out / "elbow_variants.jpg")


if __name__ == "__main__":
    main()
