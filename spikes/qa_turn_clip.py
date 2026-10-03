"""G2/G3 automatic checks for one generated turn clip (docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §5.2).

Pipeline per clip:
  decode -> scale to the endpoint mapping's video size -> colour-difference key on
  #00B140 + despill -> register back onto the 960x1696 rig canvas (inverse of the
  endpoint mapping) -> one global per-channel colour correction fitted on frame 0
  vs the first reference (allowed by the plan) -> metrics at the 256 px display.

This is the bake-off screen; the production keying (BiRefNet / MatAnyone 2 seeded with
the known endpoint alpha) lives in tools/process_turn_clip.py.

Usage:
  D:\\anaconda3\\python.exe -X utf8 spikes/qa_turn_clip.py CLIP.mp4 \\
      --endpoints assets/rig_adult_walk_v1/clips/endpoints/front_to_side \\
      --first-ref assets/rig_adult_walk_v1/references/front_rest.png \\
      --last-ref  assets/rig_adult_walk_v1/references/side_key.png \\
      --out spikes/_qa/adult_walk_v1/g2_bakeoff/<model>_<n>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def decode(path: str) -> tuple[list[np.ndarray], float]:
    """Decode all frames with the system ffmpeg (no PyAV / imageio-ffmpeg needed)."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return decode_qt(path)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    w, h = int(probe["width"]), int(probe["height"])
    fps = float(Fraction(probe.get("avg_frame_rate", "24/1") or "24/1"))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return [arr[i] for i in range(n)], fps


def decode_qt(path: str) -> tuple[list[np.ndarray], float]:
    """Use Qt's bundled video decoder when no system ffmpeg is available."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer, QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
    app = QCoreApplication.instance() or QCoreApplication([])
    player, sink, loop = QMediaPlayer(), QVideoSink(), QEventLoop()
    frames, timestamps, errors = [], [], []
    def receive(frame):
        if not frame.isValid():
            return
        stamp = frame.startTime()
        if timestamps and stamp == timestamps[-1]:
            return
        im = frame.toImage().convertToFormat(QImage.Format_RGBA8888)
        if im.isNull():
            errors.append("video frame cannot be mapped")
            loop.quit()
            return
        rgba = np.frombuffer(im.constBits(), np.uint8).reshape(im.height(),im.bytesPerLine()//4,4)
        frames.append(rgba[:,:im.width(),:3].copy())
        timestamps.append(stamp)
    def status(value):
        if value == QMediaPlayer.MediaStatus.EndOfMedia:
            loop.quit()
    def failure(*_):
        errors.append(player.errorString())
        loop.quit()
    sink.videoFrameChanged.connect(receive)
    player.mediaStatusChanged.connect(status)
    player.errorOccurred.connect(failure)
    player.setVideoSink(sink)
    player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(lambda: (errors.append("video decode timed out"),loop.quit()))
    timeout.start(30000)
    player.play()
    loop.exec()
    player.stop()
    timeout.stop()
    if errors or len(frames) < 2:
        raise RuntimeError("Qt video decoding failed: " + "; ".join(errors))
    spacing = np.diff(timestamps)
    period = float(np.median(spacing))
    if period <= 0 or np.max(np.abs(spacing-period)) > period*.1:
        raise RuntimeError("Qt video decoder skipped a frame; cannot build a deterministic clip")
    return frames, 1e6/period

KEY = np.array([0, 177, 64], np.float32)
CANVAS = (960, 1696)
DISP_H = 256
FLASH_DEV = 4.0          # luma levels (0-255) away from the 9-frame running median


def key_frame(rgb: np.ndarray, t0: float = 14.0, t1: float = 70.0) -> np.ndarray:
    """RGB float -> straight RGBA float; colour-difference key with despill."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess = g - np.maximum(r, b)
    a = 1.0 - np.clip((excess - t0) / (t1 - t0), 0.0, 1.0)
    fg = (rgb - (1.0 - a[..., None]) * KEY) / np.maximum(a[..., None], 1e-3)
    fg[..., 1] = np.minimum(fg[..., 1], np.maximum(fg[..., 0], fg[..., 2]))
    fg = np.where(a[..., None] > 0.02, np.clip(fg, 0, 255), 0.0)
    return np.dstack([fg, a * 255.0])


def resize_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pm = rgba.copy()
    pm[..., :3] *= pm[..., 3:4] / 255.0
    ch = [np.asarray(Image.fromarray(pm[..., i].astype(np.float32)).resize(size, Image.Resampling.LANCZOS))
          for i in range(4)]
    al = np.clip(ch[3], 0, 255)
    rgb = np.stack(ch[:3], -1) / np.maximum(al[..., None] / 255.0, 1e-3)
    return np.dstack([np.clip(rgb, 0, 255), al])


def to_canvas(rgba_video: np.ndarray, m: dict) -> np.ndarray:
    """Video RGBA -> extended canvas (full rig-canvas height, frame width), x origin = frame origin."""
    fw, fh = m["frame_size_canvas"]
    x0, y0 = m["frame_origin_canvas"]
    region = resize_rgba(rgba_video, (fw, fh))
    return region[-y0:-y0 + CANVAS[1]]


def pad_ref(ref: np.ndarray, m: dict) -> np.ndarray:
    """Rig-canvas reference -> the same extended canvas as to_canvas()."""
    fw = m["frame_size_canvas"][0]
    x0 = m["frame_origin_canvas"][0]
    out = np.zeros((CANVAS[1], fw, 4), np.float32)
    out[:, -x0:-x0 + ref.shape[1]] = ref
    return out


def display(rgba_canvas: np.ndarray) -> np.ndarray:
    fit = DISP_H / CANVAS[1]
    return resize_rgba(rgba_canvas, (round(rgba_canvas.shape[1] * fit), DISP_H))


def premul_stats(x: np.ndarray, y: np.ndarray) -> dict:
    sil = (x[..., 3] > 8) | (y[..., 3] > 8)
    px = x[..., :3] * x[..., 3:4] / 255.0
    py = y[..., :3] * y[..., 3:4] / 255.0
    d = np.abs(px - py).mean(-1)
    h, w = d.shape
    hb, wb = h // 8 * 8, w // 8 * 8
    blocks = d[:hb, :wb].reshape(hb // 8, 8, wb // 8, 8).mean((1, 3))
    bm = sil[:hb, :wb].reshape(hb // 8, 8, wb // 8, 8).any((1, 3))
    ax, ay = x[..., 3] / 255.0, y[..., 3] / 255.0
    return {"mean_abs_255": float(d[sil].mean()),
            "block8_p95_255": float(np.percentile(blocks[bm], 95)),
            "soft_iou": float(np.minimum(ax, ay).sum() / max(np.maximum(ax, ay).sum(), 1e-6))}


def silhouette_metrics(rgba_canvas: np.ndarray) -> dict:
    al = rgba_canvas[..., 3] > 127
    rows = np.where(al.any(1))[0]
    if not len(rows):
        return {"sole": None, "top": None, "height": None}
    widths = al.sum(1)
    top = np.where(widths >= 40)[0]
    top = int(top.min()) if len(top) else int(rows.min())
    return {"sole": int(rows.max()), "top": top, "height": int(rows.max()) - top}


def colour_fit(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = (src[..., 3] > 230) & (ref[..., 3] > 230)
    gains, offs = np.ones(3, np.float32), np.zeros(3, np.float32)
    if m.sum() > 500:
        for c in range(3):
            A = np.stack([src[..., c][m], np.ones(m.sum())], 1)
            sol, *_ = np.linalg.lstsq(A, ref[..., c][m], rcond=None)
            gains[c], offs[c] = sol
    return gains, offs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--endpoints", required=True, help="dir with mapping.json from prepare_video_endpoints.py")
    ap.add_argument("--first-ref", required=True)
    ap.add_argument("--last-ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((Path(a.endpoints) / "mapping.json").read_text(encoding="utf-8"))["first"]
    vw, vh = mapping["video_size"]
    frames, fps = decode(a.clip)
    H, W = frames[0].shape[:2]
    if abs(W / H - vw / vh) > 0.01:
        raise SystemExit(f"clip aspect {W}x{H} differs from endpoint mapping {vw}x{vh}")

    canvas_frames = []
    edge_frames = []
    for i, f in enumerate(frames):
        rgb = np.asarray(Image.fromarray(f[..., :3]).resize((vw, vh), Image.Resampling.LANCZOS), np.float32)
        keyed = key_frame(rgb)
        al = keyed[..., 3] > 127
        if al[:, :4].any() or al[:, -4:].any() or al[:4].any():
            edge_frames.append(i)
        canvas_frames.append(to_canvas(keyed, mapping))

    first_ref = pad_ref(np.asarray(Image.open(a.first_ref).convert("RGBA"), np.float32), mapping)
    last_ref = pad_ref(np.asarray(Image.open(a.last_ref).convert("RGBA"), np.float32), mapping)
    raw_luma_frames = [cf.copy() for cf in canvas_frames]
    gains, offs = colour_fit(canvas_frames[0], first_ref)
    for cf in canvas_frames:
        cf[..., :3] = np.clip(cf[..., :3] * gains + offs, 0, 255)

    disp = [display(cf) for cf in canvas_frames]
    d_first, d_last = display(first_ref), display(last_ref)
    first = premul_stats(disp[0], d_first)
    last = premul_stats(disp[-1], d_last)
    ref_h = silhouette_metrics(first_ref)["height"]
    ref_sole = silhouette_metrics(first_ref)["sole"]
    sil = [silhouette_metrics(cf) for cf in canvas_frames]
    heights = np.array([s["height"] or 0 for s in sil], np.float32)
    soles = np.array([s["sole"] or 0 for s in sil], np.float32)
    canvas_per_disp = CANVAS[1] / DISP_H
    iou_step = []
    for i in range(len(disp) - 1):
        x, y = disp[i][..., 3] / 255.0, disp[i + 1][..., 3] / 255.0
        iou_step.append(1.0 - float(np.minimum(x, y).sum() / max(np.maximum(x, y).sum(), 1e-6)))
    iou_step = np.array(iou_step)
    below = np.array([(cf[int(ref_sole) + 14:, :, 3] > 25).mean() for cf in canvas_frames])
    vis = [d[..., 3] > 51 for d in disp]
    green = np.concatenate([(d[..., 1] - np.maximum(d[..., 0], d[..., 2]))[v] for d, v in zip(disp, vis)])
    flick = []
    for seg in (disp[:3], disp[-3:]):
        for i in range(len(seg) - 1):
            x, y = seg[i][..., 3] / 255.0, seg[i + 1][..., 3] / 255.0
            s_ = (x > 0.5) | (y > 0.5)
            flick.append(float((np.abs(x - y)[s_] > 0.2).mean()))

    # flash: character brightness jumping away from its local trend (models often flash
    # when forced onto a pinned last frame); compared before the global colour fit
    luma = np.array([float((cf[..., :3] @ [0.299, 0.587, 0.114])[cf[..., 3] > 200].mean())
                     for cf in raw_luma_frames])
    from scipy.ndimage import median_filter
    dev = np.abs(luma - median_filter(luma, size=9, mode="nearest"))
    flash_frames = [int(i) for i in np.where(dev > FLASH_DEV)[0]]

    g2 = {
        "first_frame_match": {**first, "pass": first["mean_abs_255"] <= 4 and first["block8_p95_255"] <= 10
                              and first["soft_iou"] >= 0.97},
        "last_frame_match": {**last, "pass": last["mean_abs_255"] <= 4 and last["block8_p95_255"] <= 10
                             and last["soft_iou"] >= 0.97},
        "scale_stability": {"height_ratio_min": float(heights.min() / ref_h),
                            "height_ratio_max": float(heights.max() / ref_h),
                            "pass": bool(np.all(np.abs(heights / ref_h - 1) <= 0.02))},
        "grounding": {"sole_dev_disp_px_max": float(np.abs(soles - ref_sole).max() / canvas_per_disp),
                      "pass": bool(np.abs(soles - ref_sole).max() / canvas_per_disp <= 1.5)},
        "temporal_continuity": {"max_step": float(iou_step.max()), "median_step": float(np.median(iou_step)),
                                "pass": bool(iou_step.max() <= 3 * max(np.median(iou_step), 1e-3))},
        "no_floor_or_shadow": {"max_frac_alpha_below_ground": float(below.max()),
                               "pass": bool(below.max() <= 1e-3)},
        "stays_in_frame": {"frames_touching_left_right_top_edge": edge_frames,
                           "pass": not edge_frames},
        "no_flash": {"flash_frames": flash_frames, "max_luma_dev": round(float(dev.max()), 2),
                     "threshold": FLASH_DEV, "pass": not flash_frames},
    }
    g3 = {
        "residual_green_p99_255": float(np.percentile(green, 99)),
        "residual_green_pass": bool(np.percentile(green, 99) <= 6),
        "endpoint_alpha_flicker_max": float(max(flick) if flick else 0.0),
        "endpoint_alpha_flicker_pass": bool((max(flick) if flick else 0.0) <= 0.003),
    }
    report = {"clip": a.clip, "frames": len(frames), "fps": fps, "source_size": [W, H],
              "colour_fit": {"gains": gains.tolist(), "offsets": offs.tolist()},
              "g2": g2, "g3_screen": g3,
              "g2_numeric_pass": all(v["pass"] for v in g2.values())}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # evidence: contact sheet (light + dark), 1x and 1/4x GIFs at 256
    n = len(disp)
    idx = np.linspace(0, n - 1, min(12, n)).round().astype(int)
    tw = disp[0].shape[1]
    sheet = Image.new("RGB", (tw * len(idx), DISP_H * 2 + 16), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        for row, bgc in enumerate(((245, 246, 250), (34, 40, 49))):
            bg = Image.new("RGBA", (tw, DISP_H), (*bgc, 255))
            bg.alpha_composite(Image.fromarray(disp[i].clip(0, 255).astype(np.uint8), "RGBA"))
            sheet.paste(bg.convert("RGB"), (k * tw, 16 + row * DISP_H))
        dr.text((k * tw + 2, 2), str(i), fill=(0, 0, 0))
    sheet.save(out / "contact_sheet_256.png")
    gif = []
    for d in disp:
        bg = Image.new("RGBA", (tw, DISP_H), (245, 246, 250, 255))
        bg.alpha_composite(Image.fromarray(d.clip(0, 255).astype(np.uint8), "RGBA"))
        gif.append(bg.convert("RGB"))
    dur = int(round(1000 / fps))
    gif[0].save(out / "clip_256_1x.gif", save_all=True, append_images=gif[1:], duration=dur, loop=0)
    gif[0].save(out / "clip_256_quarter.gif", save_all=True, append_images=gif[1:], duration=dur * 4, loop=0)
    print(json.dumps({"g2": {k: v["pass"] for k, v in g2.items()}, "g3_screen": g3,
                      "frames": len(frames), "fps": fps}, indent=2))


if __name__ == "__main__":
    main()
