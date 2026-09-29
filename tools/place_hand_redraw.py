"""Place a Qwen hand redraw (green background, 3x crop of the arm layer) back onto the layer canvas.

Side walk (user review 2026-09-29): the far hand shows its palm, the near hand its back; the art
drew both edge-on. Qwen redrew each hand from a 3x crop of the arm as the rig uses it (far arm = GPT redraw) on #00B140 (sleeve/cuff kept).
This keys the green, registers the redraw on the unchanged cuff (integer search), downsamples 3x
and writes a full-canvas RGBA holding only rows >= cut_y (mid white frill, where the seam is
invisible). build_side_spec.limb_source pastes those rows over the limb.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PKG = Path(__file__).resolve().parents[1] / "assets/rig_adult_walk_v1"
# limb: (redraw, crop box used for the 3x input, cut row in canvas px)
HANDS = {"arm_l": ("references/qwen_raw/near_s1.png", (170, 700, 470, 1000), 838),
         "arm_r": ("references/qwen_raw/far_s2.png", (560, 720, 760, 990), 832)}
S = 3
GREEN = np.array([0, 177, 64], np.float64)


def key(rgb: np.ndarray) -> np.ndarray:
    """Green-screen matte: alpha from the green excess, colour despilled."""
    f = rgb.astype(np.float64)
    excess = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    a = np.clip(1.0 - (excess - 25.0) / 60.0, 0.0, 1.0)
    f[..., 1] = np.minimum(f[..., 1], np.maximum(f[..., 0], f[..., 2]) + 8)   # despill
    return np.dstack([f, a * 255]).clip(0, 255).astype(np.uint8)


def shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate without wrap-around (np.roll wrapped sleeve rows into the hand area)."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def place(limb: str) -> Path:
    src, (x0, y0, x1, y1), cut = HANDS[limb]
    sys.path.insert(0, str(Path(__file__).parent))
    from build_side_spec import limb_source          # the arm as the rig uses it, before the hand paste
    art = limb_source(limb)
    gen = key(np.asarray(Image.open(PKG / src).convert("RGB")))
    small = np.asarray(Image.fromarray(gen).resize(((x1 - x0), (y1 - y0)), Image.LANCZOS)).astype(np.float64)
    ref = art[y0:y1, x0:x1].astype(np.float64)
    band = slice(cut - y0 - 60, cut - y0 - 10)                  # cuff rows above the cut
    best = None
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            sh = shift(small, dx, dy)
            e = np.abs(sh[band, :, 3] - ref[band, :, 3]).mean() + np.abs(sh[band, :, :3] - ref[band, :, :3]).mean()
            if best is None or e < best[0]:
                best = (e, dx, dy)
    _, dx, dy = best
    out = np.zeros_like(art)
    sh = shift(small, dx, dy).clip(0, 255).astype(np.uint8)
    out[y0:y1, x0:x1] = sh
    out[:cut] = 0
    path = PKG / "prep/peel" / f"{limb}_hand_qwen_s1_canvas.png"
    Image.fromarray(out).save(path)
    path.with_suffix(".json").write_bytes(json.dumps(
        {"source": src, "crop": [x0, y0, x1, y1], "scale": S, "cut_y": cut, "shift": [dx, dy],
         "cuff_error": round(best[0], 2)}, indent=1).encode())
    print(limb, "shift", dx, dy, "err", round(best[0], 2))
    return path


if __name__ == "__main__":
    for limb in sys.argv[1:] or HANDS:
        place(limb)
