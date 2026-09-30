"""Export a high-resolution ADULT idle -> turn -> walk -> turn -> idle reference.

The presenter renders directly at 1024 px, while locomotion keeps the shipped
256-px world scale. High-resolution turn clips are selected by the presenter.
Only this isolated preview instance is driven; production settings/assets are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
FPS = 30
DT = 1 / 60
RENDER = 1024
SIZE = (768, 1152)
BG = (239, 242, 247, 255)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/adult_reference_hd_2026-09-30")
    args = ap.parse_args()
    out = ROOT / args.out
    work = ROOT / ".scratch/adult_reference_hd_2026-09-30/frames"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    from render_rig_rest import RigRenderer
    from pet.rig.motion import MotionInputs
    from PySide6.QtGui import QImage

    r = RigRenderer()
    w = r.win
    w.resize(RENDER, RENDER)
    r._pump()
    assert w.enable_side_locomotion(str(ROOT / "assets/rig_adult_walk_v1"))
    w._loco.scale = 256 / 1696  # preserve the actual desktop gait; render size is independent
    w.move(200, 300)
    inputs = MotionInputs(grounded=True, facing=1, source_facing=1,
                          wind_gain=1.0, cursor_pos=None, walk_hz=1.2)
    w._motion_inputs = inputs
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/simhei.ttf")
    title_font = ImageFont.truetype(str(font_path), 27)
    small_font = ImageFont.truetype(str(font_path), 20)
    paths, records, changes = [], [], []
    last = None
    returned_at = None
    i = 0

    while i < 28 * 60:
        t = (i + 1) * DT
        vx = 120.0 if 2 <= t < 8 else 0.0
        inputs.walking = bool(vx)
        inputs.pet_rect = (float(w.x()), 300.0, 256.0, 256.0)
        w.set_locomotion_intent(vx)
        frame = w._engine.step(inputs, 1000 * DT)
        lf = w._loco.update(DT, vx, float(w.x()), grounded=True, dragged=False)
        if lf.mode == "front":
            w._push_frame(w._settle_frame(frame, lf.settle))
        w._apply_loco(lf, frame)
        # No wall-clock QML rotation easing in this explicitly stepped reference.
        w._root.setProperty("bodyAngle", 0.0)
        r._pump(2)
        gait = w._loco._solver.state.value if lf.gait else ""
        key = (lf.state.value, gait)
        if key != last:
            changes.append({"t": round(t, 4), "loco": key[0], "gait": gait})
            last = key
            print(f"[capture] {t:.2f}s {key}", flush=True)
        if t > 8 and lf.state.value == "front" and returned_at is None:
            returned_at = t

        if i % 2 == 1:
            if lf.mode == "clip" or lf.state.value in ("settle", "side_settle"):
                label = "转身"
            elif lf.mode == "side":
                label = "行走" if vx else "收脚 / 等待转回"
            else:
                label = "静止"
            q = w._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
            assert not q.isNull(), "Qt framebuffer unavailable"
            a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine() // 4, 4)
            im = Image.fromarray(a[:, :q.width()].copy()).resize((RENDER, RENDER), Image.Resampling.LANCZOS)
            # Camera follows the pet. Crop only unused lateral padding; keep head, hands, tail, shoes.
            hero = im.crop((128, 0, 896, RENDER))
            assert np.count_nonzero(np.asarray(hero)[..., 3] > 20) > 50000, "blank capture"
            canvas = Image.new("RGBA", SIZE, BG)
            canvas.alpha_composite(hero, (0, 64))
            draw = ImageDraw.Draw(canvas)
            draw.text((26, 17), "ADULT  ·  动作参考", font=title_font, fill=(50, 63, 86, 255))
            text_width = draw.textlength(label, font=small_font)
            draw.text((SIZE[0] - text_width - 27, 24), label, font=small_font, fill=(75, 89, 112, 255))
            draw.line((38, 1088, SIZE[0] - 38, 1088), fill=(211, 218, 229, 255), width=1)
            draw.text((26, 1113), f"{t:05.2f}s   /   30 fps", font=small_font, fill=(100, 111, 128, 255))
            path = work / f"{len(paths):04d}.png"
            canvas.convert("RGB").save(path)
            paths.append(path)
            records.append({"t": round(t, 6), "mode": lf.mode, "state": lf.state.value,
                            "gait": gait, "window_x_256": w.x(), "label": label})
        i += 1
        if returned_at is not None and t >= returned_at + 2:
            break
    r.close()
    assert returned_at is not None, "Sequence did not return to front idle"
    assert any(x["gait"] == "walk_loop" for x in records), "No sustained walking"
    assert records[-1]["state"] == "front", "Final idle missing"

    # One palette shared by every frame avoids frame-to-frame quantization flicker.
    selected = sorted(set([0, len(paths) - 1] + list(range(0, len(paths), 24))))
    mosaic = Image.new("RGB", (384 * 6, 576 * ((len(selected) + 5) // 6)), BG[:3])
    for j, k in enumerate(selected):
        with Image.open(paths[k]) as im:
            mosaic.paste(im.resize((384, 576), Image.Resampling.LANCZOS), ((j % 6) * 384, (j // 6) * 576))
    palette = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for j, path in enumerate(paths):
        with Image.open(path) as im:
            frames.append(im.quantize(palette=palette, dither=Image.Dither.NONE))
        if (j + 1) % 90 == 0:
            print(f"[encode] palette frames {j + 1}/{len(paths)}", flush=True)
    # GIF uses centiseconds: 30/30/40 ms preserves exactly 30 fps over each three frames.
    durations = [30, 30, 40] * (len(frames) // 3) + [30, 30, 40][:len(frames) % 3]
    gif_path = out / "ADULT_静止-转身-行走-转身-静止_高清.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, disposal=1, optimize=True, palette=palette.getpalette())
    with Image.open(gif_path) as gif:
        actual_ms = 0
        for j in range(gif.n_frames):
            gif.seek(j)
            actual_ms += gif.info.get("duration", 0)
        assert gif.size == SIZE and actual_ms == sum(durations)
        actual_frames = gif.n_frames

    times = [1, 2.85, 6.5, returned_at - .65, returned_at + 1]
    thumbs = Image.new("RGB", (384 * 5, 576), BG[:3])
    for j, tt in enumerate(times):
        k = min(range(len(records)), key=lambda k: abs(records[k]["t"] - tt))
        with Image.open(paths[k]) as im:
            thumbs.paste(im.resize((384, 576), Image.Resampling.LANCZOS), (384 * j, 0))
    thumbs.save(out / "五阶段关键帧.png")
    meta = {"output": gif_path.name, "size": SIZE, "rig_render_height": RENDER,
            "gait_world_scale": 256 / 1696, "speed_px_s_at_256": 120,
            "fps": FPS, "duration_s": actual_ms / 1000, "frames": actual_frames,
            "file_bytes": gif_path.stat().st_size,
            "turn_assets": "highest existing 512-high production frame packages",
            "camera": "follows the character; horizontal window displacement is recorded, not shown",
            "palette": "one shared 256-color palette, no dithering; optimized unchanged pixels",
            "transitions": changes, "samples": records}
    (out / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("samples", "transitions")}, ensure_ascii=False, indent=2))
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
