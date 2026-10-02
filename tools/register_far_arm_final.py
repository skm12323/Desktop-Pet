r"""Register the gpt-image-2.5 whole far arm (FINAL side rig) onto the side key art canvas.

Similarity transform (scale, rotation, shift) maximising overlap with the arm's visible pixels in
the art, minus a penalty for arm pixels where the art is transparent (the arm must stay inside the
art's silhouette wherever it is visible there) - the same idea as the ADULT far-arm registration
(true silhouette + spill penalty). Writes prep/peel/arm_r_gpt_canvas.png (+ .json record), which
tools/build_final_side.py uses as the whole far-arm layer.

Usage: D:naconda3\python.exe -X utf8 tools/register_far_arm_final.py [--raw far_arm_gpt_02.png]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "assets" / "rig_final_walk_v1"
W, H = 1024, 1824


def key_green(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    g = raw[..., 1] - np.maximum(raw[..., 0], raw[..., 2])
    a = np.clip((60 - g) * 255 / 40, 0, 255).astype(np.uint8)
    rgb = raw.copy()
    sp = g > 0
    rgb[sp, 1] = np.maximum(raw[sp, 0], raw[sp, 2])
    arr = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), a])
    m = a > 127
    lab, n = ndimage.label(m)
    if n > 1:
        s = ndimage.sum(m, lab, range(1, n + 1))
        arr[lab != 1 + int(np.argmax(s))] = 0
    return arr


def transform(arr: np.ndarray, s: float, rot: float, c_src, c_dst) -> np.ndarray:
    """RGBA src -> canvas: scale s, rotate rot (deg) about c_src, then move c_src to c_dst."""
    th = np.radians(rot)
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Ai = np.linalg.inv(A)
    # PIL affine maps output -> input: x_in = Ai (x_out - c_dst) + c_src
    off = np.array(c_src) - Ai @ np.array(c_dst)
    im = Image.fromarray(arr, "RGBA").transform((W, H), Image.AFFINE,
                                                (Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1]),
                                                resample=Image.Resampling.BICUBIC)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="far_arm_gpt_02.png")
    a = ap.parse_args()
    raw_path = PKG / "references" / "gpt_raw" / a.raw
    arr = key_green(raw_path)
    key = np.asarray(Image.open(PKG / "references" / "side_key.png").convert("RGBA"))
    lab = np.asarray(Image.open(PKG / "prep" / "partition_labels.png")).astype(int)
    ids = {v: int(k) for k, v in json.loads((PKG / "prep" / "partition_ids.json").read_text(encoding="utf-8")).items()}
    own = lab == ids["arm_r"]
    empty = key[..., 3] < 20
    ys, xs = np.where(arr[..., 3] > 127)
    c_src = (float(xs.mean()), float(ys.mean()))
    oy, ox = np.where(own)
    q = 4                                  # coarse search on a 1/4 grid
    own_q = own[::q, ::q]
    empty_q = empty[::q, ::q]
    best = (-1e9, None)

    # score on a 1/q copy: transform the alpha only, in the reduced coordinate frame
    small = arr[::q, ::q].copy()

    def tf_small(s, rot, dx, dy):
        th = np.radians(rot)
        A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Ai = np.linalg.inv(A)
        cs, cd = np.array(c_src) / q, np.array([dx, dy]) / q
        off = cs - Ai @ cd
        im = Image.fromarray(small[..., 3]).transform((W // q, H // q), Image.AFFINE,
                                                      (Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1]),
                                                      resample=Image.Resampling.NEAREST)
        return np.asarray(im) > 127

    def score(s, rot, dx, dy):
        t = tf_small(s, rot, dx, dy)
        inter = (t & own_q).sum()
        cover = inter / max(own_q.sum(), 1)
        spill = (t & empty_q).sum() / max(t.sum(), 1)
        return cover - 2.0 * spill, cover, spill
    # initial centre: the visible arm's centre shifted up toward the hidden shoulder
    cx0, cy0 = float(ox.mean()), float(oy.mean()) - 40
    for s in np.arange(0.55, 0.86, 0.03):
        for rot in (-8, -4, 0, 4, 8):
            for dx in range(-60, 61, 12):
                for dy in range(-90, 91, 12):
                    v = score(s, rot, cx0 + dx, cy0 + dy)
                    if v[0] > best[0]:
                        best = (v[0], (s, rot, cx0 + dx, cy0 + dy), v)
    s, rot, cx, cy = best[1]
    for s2 in np.arange(s - 0.03, s + 0.031, 0.01):     # refine
        for r2 in np.arange(rot - 3, rot + 3.1, 1.0):
            for dx in range(-8, 9, 2):
                for dy in range(-8, 9, 2):
                    v = score(s2, r2, cx + dx, cy + dy)
                    if v[0] > best[0]:
                        best = (v[0], (s2, r2, cx + dx, cy + dy), v)
    s, rot, cx, cy = best[1]
    out = transform(arr, s, rot, c_src, (cx, cy))
    Image.fromarray(out, "RGBA").save(PKG / "prep" / "peel" / "arm_r_gpt_canvas.png")
    rec = {"note": "far arm redrawn whole by gpt-image-2.5 (run by the user 2026-10-02); green keyed; "
                   "similarity-registered: max visible-arm coverage - 2 x spill outside the art silhouette",
           "source": {"path": raw_path.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()},
           "scale": round(float(s), 4), "rot_deg": round(float(rot), 2),
           "src_centre": [round(c, 1) for c in c_src], "dst_centre": [round(float(cx), 1), round(float(cy), 1)],
           "visible_coverage": round(float(best[2][1]), 3), "spill_frac": round(float(best[2][2]), 4)}
    (PKG / "prep" / "peel" / "arm_r_gpt_canvas.json").write_bytes(json.dumps(rec, indent=1).encode("utf-8"))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
