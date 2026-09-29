"""Render a skinned rig's rest pose through the real Qt path (G0 / G2 endpoint source).

``relaxed`` = spec ``rest_pose_angles`` applied, every other bone 0, eyes open,
gaze centred, no body transform (no breath / bob / tilt / squash) — the pose the
runtime settles to before a turn clip starts. ``zero`` = pure bind pose (the
A-pose for ADULT).

Default output is canvas-exact: the window equals the source canvas, fit = 1 and
the ADULT ground shift is disabled, so PNG pixel (x, y) is source-canvas pixel
(x, y) and the soles stay on ``ground_anchor_y_px``.

``--video-916`` also writes the video-model endpoint: canvas padded at the top to
9:16, scaled to 1080x1920 and composited on chroma green #00B140, plus a JSON
sidecar with the canvas<->video mapping used later to register clip frames.

Usage (repo root):
  D:\\anaconda3\\python.exe -X utf8 tools/render_rig_rest.py --stage adult \\
      --out assets/rig_adult_walk_v1/references/front_rest.png \\
      --video-916 assets/rig_adult_walk_v1/references/front_rest_green_916.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("QT_QUICK_BACKEND", "rhi")
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")   # device pixel ratio 1
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

KEY_RGB = (0, 177, 64)          # chroma green #00B140
VIDEO_SIZE = (1080, 1920)       # 9:16 portrait


class RigRenderer:
    """One QApplication + one offscreen RigWindow, reused for many renders."""

    def __init__(self, stage: str = "adult", rig_dir: str = "", figure: str = "healthy_neutral"):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from pet.asset_provider import SpriteRef
        from pet.rig.presenter import build_rig_window
        from pet.rig.spec import load_rig_spec
        from pet.window import WindowBase

        self.app = QApplication.instance() or QApplication([])
        rig_root = str(Path(rig_dir).parent) if rig_dir else ""
        rig_path = rig_dir or str(ROOT / "assets" / "rig" / stage)
        self.spec = load_rig_spec(rig_path, stage)
        if self.spec is None or figure not in self.spec.figures:
            raise SystemExit(f"rig spec / figure {figure!r} not found under {rig_path}")
        self.win = build_rig_window(WindowBase, SpriteRef(self.spec.figures[figure], 256, 256),
                                    stage, rig_root=rig_root)
        if not getattr(self.win, "rig_active", False):
            raise SystemExit("Qt Quick rig backend not active (needs a hardware RHI backend)")
        self.win._motion_timer.stop()                 # we drive every frame explicitly
        self.win.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.win.show()
        self._pump()
        if not self.win._root.property("skinnedMeshVisible"):
            raise SystemExit("skinned mesh not visible for this figure")
        self.rt = self.win._skinned_item._rt
        self.bone_names = [b.name for b in self.rt.bones]
        self.canvas = (int(round(self.rt.img_w)), int(round(self.rt.img_h)))
        self.ground_y = float(self.spec.ground_anchor_y_px or 0.0)

    def _pump(self, n: int = 4) -> None:
        for _ in range(n):
            self.app.processEvents()

    def rest_frame(self, pose: str = "relaxed"):
        from pet.rig.motion import MotionFrame
        angles = {n: 0.0 for n in self.bone_names}
        if pose == "relaxed":
            for name, deg in self.spec.rest_pose_angles.items():
                if name in angles:
                    angles[name] = float(deg)
        elif pose != "zero":
            raise ValueError(f"unknown pose {pose!r}")
        return MotionFrame(bone_angles=angles, blink_progress=0.0, look_at=(0.0, 0.0))

    def render(self, frame, size: tuple[int, int] | None = None,
               ground_shift: bool = False) -> Image.Image:
        """Render one MotionFrame. size=None -> canvas-exact (fit 1, no ground shift)."""
        from PySide6.QtGui import QImage
        w, h = size or self.canvas
        if (self.win.width(), self.win.height()) != (w, h):
            self.win.resize(w, h)
            self._pump()
        root = self.win._root
        root.setProperty("skinnedGroundYPx", self.ground_y if ground_shift else 0.0)
        self.win._push_frame(frame)
        # QML animates bodyAngle with a 90 ms Behavior; renders here are static poses
        root.setProperty("bodyAngle", 0.0)
        root.setProperty("bodyY", float(frame.body_y))
        self._pump()
        img = self.win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        if img.isNull():
            raise RuntimeError("Qt framebuffer unavailable; requires a hardware RHI backend")
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(
            img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()].copy()
        pil = Image.fromarray(arr, "RGBA")
        if pil.size != (w, h):                        # HiDPI fallback
            pil = pil.resize((w, h), Image.Resampling.LANCZOS)
        if np.count_nonzero(np.asarray(pil)[:, :, 3] > 20) < 1000:
            raise RuntimeError("blank render")
        return pil

    def close(self) -> None:
        self.win.close()
        self._pump()


def video_endpoint(canvas_rgba: Image.Image, ground_y: float, margin_x: int = 160,
                   floor_margin: int = 100, margin_right: int | None = 40) -> tuple[Image.Image, dict]:
    """Canvas RGBA -> 1080x1920 9:16 frame on chroma green + mapping metadata.

    The frame covers a canvas-space region wider than the rig canvas (margin_x on each
    side) so a swinging tail never leaves the frame; the soles sit floor_margin canvas
    px above the frame bottom. Mapping: video_xy = (canvas_xy - frame_origin) * s.
    """
    cw, ch = canvas_rgba.size
    vw, vh = VIDEO_SIZE
    mr = margin_x if margin_right is None else margin_right   # margin_x = left (or both)
    fw = cw + margin_x + mr
    fh = int(round(fw * vh / vw))
    x0 = -margin_x
    y0 = int(round(ground_y + floor_margin - fh))
    s = vw / fw
    region = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    region.paste(canvas_rgba, (-x0, -y0))
    scaled = region.resize((vw, vh), Image.Resampling.LANCZOS)
    frame = Image.new("RGBA", (vw, vh), (*KEY_RGB, 255))
    frame.alpha_composite(scaled)
    meta = {
        "canvas_size": [cw, ch],
        "video_size": [vw, vh],
        "frame_origin_canvas": [x0, y0],
        "frame_size_canvas": [fw, fh],
        "scale_video_per_canvas": s,
        "note": "video_xy = (canvas_xy - frame_origin_canvas) * scale_video_per_canvas",
        "key_rgb": list(KEY_RGB),
        "ground_y_canvas": ground_y,
        "ground_y_video": (ground_y - y0) * s,
    }
    return frame.convert("RGB"), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="adult", choices=["young", "adult", "final"])
    ap.add_argument("--rig-dir", default="", help="rig dir with manifest.json (default assets/rig/{stage})")
    ap.add_argument("--figure", default="healthy_neutral")
    ap.add_argument("--pose", default="relaxed", choices=["relaxed", "zero"])
    ap.add_argument("--out", required=True, help="canvas-exact RGBA PNG")
    ap.add_argument("--video-916", default="", help="also write 1080x1920 green endpoint PNG (+ .json)")
    args = ap.parse_args()

    r = RigRenderer(args.stage, args.rig_dir, args.figure)
    img = r.render(r.rest_frame(args.pose))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"[OK] {args.pose} rest -> {out} {img.size}")
    if args.video_916:
        frame, meta = video_endpoint(img, r.ground_y)
        vp = Path(args.video_916)
        vp.parent.mkdir(parents=True, exist_ok=True)
        frame.save(vp)
        meta.update({"source_png": out.as_posix(), "pose": args.pose, "figure": args.figure})
        vp.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[OK] video endpoint -> {vp} {frame.size}")
    r.close()


if __name__ == "__main__":
    main()
