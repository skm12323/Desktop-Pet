"""G0 finding: rig textures have no mipmaps -> the 256 px rig is aliased, while a
properly filtered clip frame is smooth, so rig<->clip seams would show a texture
quality jump. A/B (QA-only monkeypatch, production code untouched):

  A = production (createTextureFromImage without mipmaps, Linear filtering)
  B = mipmapped textures + linear mipmap filtering (trilinear)

Both vs the canvas-exact rest render mapped into the 256 window with a filtered
downscale (what a prepared clip frame looks like).

Run: D:\\anaconda3\\python.exe -X utf8 spikes/qa_g0_mipmap_ab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "spikes"))

import numpy as np
from PIL import Image

OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g0_baseline"


def enable_mipmaps() -> None:
    from PySide6.QtQuick import QQuickWindow, QSGTexture, QSGTextureMaterial
    orig_create = QQuickWindow.createTextureFromImage
    orig_filter = QSGTextureMaterial.setFiltering

    def create(self, img, options=None):
        return orig_create(self, img, QQuickWindow.CreateTextureOption.TextureHasMipmaps)

    def set_filtering(self, f):
        orig_filter(self, f)
        self.setMipmapFiltering(QSGTexture.Filtering.Linear)

    QQuickWindow.createTextureFromImage = create
    QSGTextureMaterial.setFiltering = set_filtering


def render_pair(mip: bool):
    if mip:
        enable_mipmaps()
    from render_rig_rest import RigRenderer
    r = RigRenderer("adult")
    rest = r.rest_frame("relaxed")
    canvas = np.asarray(r.render(rest), np.float32)
    small = np.asarray(r.render(rest, (256, 256), ground_shift=True), np.float32)
    info = (r.canvas, r.ground_y)
    # shimmer probe: 1 s of live idle at 30 fps; the face only moves sub-pixel
    # (breath rotation <= 0.5 deg), so frame-to-frame change there is aliasing noise
    from pet.rig.motion import MotionEngine, MotionInputs
    eng = MotionEngine(r.spec)
    inp = MotionInputs(walking=False, grounded=True, facing=1, source_facing=1)
    frames = []
    for _ in range(31):
        f = eng.step(inp, 1000 / 30)
        f.blink_progress = 0.0                      # keep eyelids out of the probe
        frames.append(np.asarray(r.render(f, (256, 256), ground_shift=True), np.float32))
    face = (slice(40, 100), slice(100, 160))
    steps = [np.abs(frames[i + 1][face][..., :3] - frames[i][face][..., :3]).mean()
             for i in range(30)]
    info = (*info, float(np.mean(steps)))
    r.close()
    return canvas, small, info


def main() -> None:
    import subprocess
    if len(sys.argv) > 1 and sys.argv[1] in ("A", "B"):
        canvas, small, (cv, gy, shimmer) = render_pair(sys.argv[1] == "B")
        np.save(OUT / f"_ab_{sys.argv[1]}_canvas.npy", canvas)
        np.save(OUT / f"_ab_{sys.argv[1]}_small.npy", small)
        (OUT / "_ab_meta.json").write_text(json.dumps({"canvas": cv, "ground_y": gy}), encoding="utf-8")
        (OUT / f"_ab_{sys.argv[1]}_shimmer.json").write_text(json.dumps(shimmer), encoding="utf-8")
        return
    # run A and B in separate processes (the monkeypatch must precede scene creation)
    for tag in ("A", "B"):
        subprocess.run([sys.executable, "-X", "utf8", __file__, tag], check=True)
    from qa_g0_baseline import map_canvas_to_window, premul_diff
    meta = json.loads((OUT / "_ab_meta.json").read_text(encoding="utf-8"))
    cw, ch = meta["canvas"]
    fit = min(256 / cw, 256 / ch)
    offx, offy = (256 - cw * fit) / 2, (256 - ch * fit) / 2
    shift = 256 - (offy + meta["ground_y"] * fit)
    canvas = np.load(OUT / "_ab_A_canvas.npy")
    mapped = map_canvas_to_window(canvas, fit, offx, offy + shift, (256, 256))
    res = {}
    tiles = []
    for tag in ("A", "B"):
        small = np.load(OUT / f"_ab_{tag}_small.npy")
        m, p = premul_diff(mapped, small)
        shimmer = json.loads((OUT / f"_ab_{tag}_shimmer.json").read_text(encoding="utf-8"))
        res[tag] = {"mean_abs_diff_255_vs_filtered_clip_frame": round(m, 3), "p99": round(p, 3),
                    "idle_face_frame_to_frame_change_255": round(shimmer, 3)}
        tiles.append(small)
    tiles.append(mapped)
    for f in OUT.glob("_ab_*"):
        f.unlink()
    labels = ["A_production_no_mip", "B_mipmapped", "filtered_clip_frame"]
    sheet = Image.new("RGB", (3 * 420, 440), (255, 255, 255))
    for k, t in enumerate(tiles):
        bg = Image.new("RGBA", (256, 256), (240, 242, 246, 255))
        bg.alpha_composite(Image.fromarray(np.clip(t, 0, 255).astype(np.uint8), "RGBA"))
        sheet.paste(bg.convert("RGB").crop((95, 30, 165, 100)).resize((420, 420), Image.NEAREST), (k * 420, 20))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for k, lab in enumerate(labels):
        d.text((k * 420 + 6, 4), lab, fill=(0, 0, 0))
    sheet.save(OUT / "mipmap_ab_face_zoom.png")
    full = Image.new("RGB", (3 * 256, 256), (240, 242, 246))
    for k, t in enumerate(tiles):
        bg = Image.new("RGBA", (256, 256), (240, 242, 246, 255))
        bg.alpha_composite(Image.fromarray(np.clip(t, 0, 255).astype(np.uint8), "RGBA"))
        full.paste(bg.convert("RGB"), (k * 256, 0))
    full.save(OUT / "mipmap_ab_256.png")
    (OUT / "mipmap_ab.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
