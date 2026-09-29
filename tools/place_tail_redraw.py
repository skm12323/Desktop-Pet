"""Place the Qwen whole-tail redraw (green background, 2x crop of the key art) onto the canvas.

Side walk (user review 2026-09-30): the tail's hidden parts (fin lobe tip under the near hand and
skirt, tail base under the hem) were procedural fills that looked glued to the skirt edge. Qwen
redrew the whole tail alone on #00B140 from a 2x crop of the key art (box below). This keys the
green, registers the redraw on the art's visible tail pixels (scale + integer shift search:
coverage of the visible tail + colour error) and writes a full-canvas RGBA used as the complete
tail layer (build_side_spec.TAIL_REDRAW). The procedural tail fill is no longer used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from place_hand_redraw import key as green_key   # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "assets/rig_adult_walk_v1"
SRC = "references/qwen_raw/tail_whole_s1.png"
BOX = (0, 800, 480, 1320)                      # crop of side_key.png given to Qwen at 2x
OUT = "prep/peel/tail_qwen_s1_canvas.png"


def paste(canvas_shape, img: np.ndarray, ox: int, oy: int) -> np.ndarray:
    out = np.zeros(canvas_shape, img.dtype)
    H, W = canvas_shape[:2]
    h, w = img.shape[:2]
    xa, ya, xb, yb = max(ox, 0), max(oy, 0), min(ox + w, W), min(oy + h, H)
    if xb > xa and yb > ya:
        out[ya:yb, xa:xb] = img[ya - oy:yb - oy, xa - ox:xb - ox]
    return out


def main() -> None:
    lab = np.asarray(Image.open(PKG / "prep/partition_labels.png"))
    ids = {int(k): v for k, v in json.loads((PKG / "prep/partition_ids.json").read_text(encoding="utf-8")).items()}
    own = lab == next(i for i, n in ids.items() if n == "tail")
    art = np.asarray(Image.open(PKG / "references/side_key.png").convert("RGBA")).astype(np.float64)
    g = green_key(np.asarray(Image.open(PKG / SRC).convert("RGB")))
    x0, y0, x1, y1 = BOX
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best = None
    for sc in (0.49, 0.495, 0.5, 0.505, 0.51):
        w, h = round(g.shape[1] * sc), round(g.shape[0] * sc)
        sm = np.asarray(Image.fromarray(g).resize((w, h), Image.LANCZOS)).astype(np.float64)
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                ox, oy = round(cx - w / 2) + dx, round(cy - h / 2) + dy
                can = paste(art.shape, sm, ox, oy)
                a = can[..., 3] > 127
                miss = (own & ~a).sum() / own.sum()
                col = np.abs(can[own & a, :3] - art[own & a, :3]).mean()
                e = 100 * miss + col
                if best is None or e < best[0]:
                    best = (e, sc, ox, oy, miss, col)
    e, sc, ox, oy, miss, col = best
    w, h = round(g.shape[1] * sc), round(g.shape[0] * sc)
    sm = np.asarray(Image.fromarray(g).resize((w, h), Image.LANCZOS))
    can = paste(art.shape[:2] + (4,), sm, ox, oy)
    Image.fromarray(can).save(PKG / OUT)
    rec = {"source": SRC, "crop": list(BOX), "scale": sc, "offset": [ox, oy],
           "visible_tail_missed": round(float(miss), 4), "colour_err_255": round(float(col), 2)}
    (PKG / OUT).with_suffix(".json").write_bytes(json.dumps(rec, indent=1).encode())
    print(rec)


if __name__ == "__main__":
    main()
