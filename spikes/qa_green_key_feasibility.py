"""G0 diagnosis (QA only, not product code): can a green-screen video of THIS
character be keyed cleanly after H.264 4:2:0 compression, as a 1080p portrait
clip from a video model would be delivered?

Ground truth = the ADULT rig's own rendered alpha (A-pose zero render at source
size). We composite it onto chroma green inside a 1080x1920 frame, encode with
libx264 yuv420p at two quality levels, decode, key with a plain
color-difference keyer + despill, and compare alpha at source scale and at the
256 px display height. A generated clip will be worse than this (uneven green,
AI edge noise, motion blur), so read the numbers as a lower bound on error,
i.e. "is the palette itself keyable".

Run: D:\\anaconda3\\python.exe -X utf8 spikes/qa_green_key_feasibility.py
"""
from __future__ import annotations

import colorsys
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g0_diagnosis"
KEY = np.array([0, 177, 64], np.float32)          # broadcast chroma green #00B140
FW, FH, CH = 1080, 1920, 1650                       # frame size, character height


def keyer(rgb: np.ndarray, t0: float = 12.0, t1: float = 70.0) -> tuple[np.ndarray, np.ndarray]:
    """Color-difference key: alpha from G - max(R, B); despill + unmix with KEY."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess = g - np.maximum(r, b)
    alpha = 1.0 - np.clip((excess - t0) / (t1 - t0), 0.0, 1.0)
    a = alpha[..., None]
    fg = (rgb - (1.0 - a) * KEY) / np.maximum(a, 1e-3)          # unmix the key colour
    fg[..., 1] = np.minimum(fg[..., 1], np.maximum(fg[..., 0], fg[..., 2]))  # despill
    return np.clip(fg, 0, 255), alpha


def premul_down(rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Premultiplied resample (avoids dark/green halos on downscale)."""
    pm = np.dstack([rgb * alpha[..., None], alpha * 255.0]).astype(np.float32)
    chans = [np.asarray(Image.fromarray(pm[..., i]).resize(size, Image.Resampling.LANCZOS)) for i in range(4)]
    pa = np.clip(chans[3] / 255.0, 0, 1)
    col = np.stack(chans[:3], -1) / np.maximum(pa[..., None], 1e-3)
    return np.clip(col, 0, 255), pa


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = Image.open(ROOT / "spikes/_qa/adult_review/rest_source_size.png").convert("RGBA")
    bbox = src.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    char = src.crop(bbox)
    ch = min(CH, int((FW - 40) * char.height / char.width))   # A-pose + tail is wide
    k = ch / char.height
    char = char.resize((round(char.width * k), ch), Image.Resampling.LANCZOS)
    ox, oy = (FW - char.width) // 2, FH - ch - 60
    gt_a = np.zeros((FH, FW), np.float32)
    gt_rgb = np.zeros((FH, FW, 3), np.float32)
    ca = np.asarray(char, np.float32)
    gt_a[oy:oy + ch, ox:ox + char.width] = ca[..., 3] / 255.0
    gt_rgb[oy:oy + ch, ox:ox + char.width] = ca[..., :3]
    comp = gt_rgb * gt_a[..., None] + KEY * (1.0 - gt_a[..., None])

    # palette conflict: opaque character pixels whose hue sits near the key hue
    op = ca[..., 3] > 250
    px = ca[op][:, :3] / 255.0
    sample = px[:: max(1, len(px) // 200000)]
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in sample])
    key_h = colorsys.rgb_to_hsv(*(KEY / 255.0))[0]
    dh = np.abs(((hsv[:, 0] - key_h) + 0.5) % 1.0 - 0.5) * 360.0
    near_key = float(np.mean((dh < 40) & (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.15)))
    excess_opaque = (px[:, 1] - np.maximum(px[:, 0], px[:, 2])) * 255.0

    results = {"palette": {"frac_opaque_pixels_within_40deg_of_key_hue": near_key,
                           "opaque_pixels_G_minus_maxRB_p99": float(np.percentile(excess_opaque, 99)),
                           "opaque_pixels_G_minus_maxRB_max": float(excess_opaque.max())}}
    disp_h = 256
    disp = (round(FW * disp_h / (ch + 60)), round(FH * disp_h / (ch + 60)))  # character height ~= 256 px
    gt_rgb_d, gt_a_d = premul_down(gt_rgb, gt_a, disp)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "f.png"
        Image.fromarray(comp.round().astype(np.uint8)).save(png)
        for crf in (18, 28):
            mp4 = Path(td) / f"c{crf}.mp4"
            dec = Path(td) / f"d{crf}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(png), "-t", "1", "-r", "24",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(mp4)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(mp4), "-frames:v", "1",
                            str(dec)], check=True)
            rgb = np.asarray(Image.open(dec).convert("RGB"), np.float32)
            fg, a = keyer(rgb)
            band = (gt_a > 0.02) & (gt_a < 0.98) | (np.abs(a - gt_a) > 0.02)
            fg_d, a_d = premul_down(fg, a, disp)
            err_d = np.abs(a_d - gt_a_d)
            edge_d = (gt_a_d > 0.02) & (gt_a_d < 0.98)
            # residual green cast on visible pixels at display scale
            vis = a_d > 0.2
            green_cast = (fg_d[..., 1] - np.maximum(fg_d[..., 0], fg_d[..., 2]))[vis]
            col_err = np.abs(fg_d - gt_rgb_d)[(a_d > 0.5) & (gt_a_d > 0.5)].mean()
            results[f"crf{crf}"] = {
                "source_scale_alpha_MAE_in_edge_band": float(np.abs(a - gt_a)[band].mean()),
                "source_scale_px_alpha_err_gt_0.25": int(np.sum(np.abs(a - gt_a) > 0.25)),
                "display256_alpha_MAE_edge": float(err_d[edge_d].mean()),
                "display256_px_alpha_err_gt_0.1": int(np.sum(err_d > 0.1)),
                "display256_opaque_px": int(np.sum(gt_a_d > 0.5)),
                "display256_green_cast_p99": float(np.percentile(green_cast, 99)),
                "display256_color_MAE_opaque": float(col_err),
            }
            # visual: keyed result at display scale over dark and light, 3x crop of hair/frills
            for name, bgc in (("dark", (34, 38, 48)), ("light", (245, 246, 250))):
                canvas = np.ones((*a_d.shape, 3), np.float32) * np.array(bgc, np.float32)
                over = fg_d * a_d[..., None] + canvas * (1 - a_d[..., None])
                Image.fromarray(over.round().astype(np.uint8)).resize(
                    (disp[0] * 3, disp[1] * 3), Image.Resampling.NEAREST).save(OUT / f"green_key_crf{crf}_{name}_x3.png")
    (OUT / "green_key_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
