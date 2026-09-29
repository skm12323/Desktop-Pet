"""G1: align a matted side-view candidate to the rig canvas and run the G1 numeric checks.

Alignment (uniform scale + translation only, never a non-uniform stretch):
  * scale so head-top -> sole height equals the front rest render's;
  * visible soles (alpha > 127) on the same row as the front rest render's (1604;
    the spec's ground_anchor_y_px 1608 is the lowest mesh vertex incl. the soft edge);
  * shoe-centre x equal to the front's, i.e. she turns in place.
"Head top" is the first row at least 40 canvas px wide, which skips the thin ahoge.

Checks (docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §5.2 G1): canvas and
real alpha, near-foot ground line (the far foot may stand higher: 3/4 ground-plane
perspective), shoe-centre offset vs the front (a small step during the turn),
applied scale (framing drift), per-colour-family mean colour ΔE2000 vs the front. The visual checklist
(identity, accessory sides, anatomy, rig-readiness, style) stays a human sign-off.

Usage:
  D:\\anaconda3\\python.exe -X utf8 tools/align_side_key.py CAND_RGBA.png OUT.png \\
      [--front assets/rig_adult_walk_v1/references/front_rest.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2hsv, rgb2lab

CANVAS = (960, 1696)
GROUND_Y = 1608
MARGIN = 8                 # keep the silhouette this far from the canvas edge
MAX_DEPTH_OFFSET = 45      # far foot may stand this much higher (3/4 ground-plane perspective)
MAX_STEP = 60              # shoe-centre offset vs front allowed as a small step in the turn


def metrics(rgba: np.ndarray, width_ref: float) -> dict:
    al = rgba[..., 3] > 127
    rows = np.where(al.any(1))[0]
    sole = int(rows.max())
    widths = al.sum(1)
    min_w = 40.0 * width_ref / CANVAS[0]
    top = int(np.where(widths >= min_w)[0].min())
    band = al[sole - max(6, int(0.03 * (sole - top))): sole + 1]
    xs = np.where(band.any(0))[0]
    return {"sole": sole, "top": top, "height": sole - top, "feet_cx": float(xs.mean())}


def feet_levels(rgba: np.ndarray) -> list[int]:
    """Lowest opaque row of the two shoes (split the bottom band at its widest gap)."""
    al = rgba[..., 3] > 127
    sole = int(np.where(al.any(1))[0].max())
    band = al[sole - 40: sole + 1]
    cols = np.where(band.any(0))[0]
    gaps = np.diff(cols)
    if len(gaps) and gaps.max() > 3:
        cut = cols[int(np.argmax(gaps))]
        groups = [cols[cols <= cut], cols[cols > cut]]
    else:
        mid = int(np.median(cols))
        groups = [cols[cols <= mid], cols[cols > mid]]
    levels = []
    for g in groups:
        sub = al[:, g.min(): g.max() + 1]
        levels.append(int(np.where(sub.any(1))[0].max()))
    return levels


FAMILIES = {   # hue in degrees, s/v in [0,1]
    "hair_mid_blue": lambda h, s, v: (h > 195) & (h < 235) & (s > 0.45) & (v > 0.45) & (v < 0.85),
    "dress_navy": lambda h, s, v: (h > 210) & (h < 260) & (s > 0.3) & (v < 0.35),
    "apron_white": lambda h, s, v: (s < 0.08) & (v > 0.9),
    "skin": lambda h, s, v: ((h < 35) | (h > 340)) & (s > 0.06) & (s < 0.35) & (v > 0.8),
    "light_blue_tips": lambda h, s, v: (h > 185) & (h < 215) & (s > 0.3) & (v > 0.85),
}


def family_means(rgba: np.ndarray) -> dict:
    op = rgba[..., 3] > 250
    rgb = rgba[..., :3][op].astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb[None])[0]
    h, s, v = hsv[:, 0] * 360.0, hsv[:, 1], hsv[:, 2]
    out = {}
    for name, sel in FAMILIES.items():
        m = sel(h, s, v)
        if m.sum() >= 200:
            out[name] = (rgb2lab(rgb[m][None])[0].mean(0), int(m.sum()))
    return out


def align(cand: Image.Image, front: np.ndarray) -> tuple[Image.Image, dict]:
    c = np.asarray(cand.convert("RGBA"), np.float32)
    fm = metrics(front, CANVAS[0])
    cm = metrics(c, cand.width)
    scale = fm["height"] / cm["height"]
    # premultiplied resize
    pm = c.copy()
    pm[..., :3] *= pm[..., 3:4] / 255.0
    new_size = (max(1, round(cand.width * scale)), max(1, round(cand.height * scale)))
    chans = [np.asarray(Image.fromarray(pm[..., i]).resize(new_size, Image.Resampling.LANCZOS)) for i in range(4)]
    alpha = np.clip(chans[3], 0, 255)
    rgb = np.stack(chans[:3], -1) / np.maximum(alpha[..., None] / 255.0, 1e-3)
    scaled = np.dstack([np.clip(rgb, 0, 255), alpha]).round().astype(np.uint8)
    sm = metrics(scaled.astype(np.float32), CANVAS[0])
    dy = int(round(fm["sole"] - sm["sole"]))
    # turn in place: put the shoes where the front's are, but never push the silhouette
    # (e.g. a trailing tail) off the canvas; the residual is a small step during the turn
    xs = np.where((scaled[..., 3] > 25).any(0))[0]
    lo, hi = MARGIN - int(xs.min()), CANVAS[0] - MARGIN - 1 - int(xs.max())
    want = int(round(fm["feet_cx"] - sm["feet_cx"]))
    dx = min(max(want, lo), hi) if lo <= hi else want
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    canvas.paste(Image.fromarray(scaled, "RGBA"), (dx, dy))     # canvas is empty: plain paste is exact
    info = {"scale_applied": round(scale, 4), "offset_px": [dx, dy],
            "feet_offset_vs_front_canvas_px": round(sm["feet_cx"] + dx - fm["feet_cx"], 1),
            "front": fm, "candidate_before": cm}
    return canvas, info


def check(side: np.ndarray, front: np.ndarray) -> dict:
    al = side[..., 3]
    real_alpha = bool((al == 0).mean() > 0.3 and (al == 255).mean() > 0.05)
    clipped = bool(al[0].any() or al[-1].any() or al[:, 0].any() or al[:, -1].any())
    sm = metrics(side, CANVAS[0])
    fm = metrics(front, CANVAS[0])
    levels = feet_levels(side)
    fam_s, fam_f = family_means(side), family_means(front)
    de = {}
    for k in FAMILIES:
        if k in fam_s and k in fam_f:
            de[k] = round(float(deltaE_ciede2000(fam_s[k][0][None], fam_f[k][0][None])[0]), 2)
        else:
            de[k] = None
    res = {
        "canvas": {"size": list(side.shape[1::-1]), "real_alpha": real_alpha,
                   "touches_canvas_edge": clipped,
                   "pass": side.shape[1::-1] == CANVAS and real_alpha and not clipped},
        "ground_line": {"sole_y": sm["sole"], "front_sole_y": fm["sole"], "feet_levels": levels,
                        "far_foot_depth_offset_px": abs(levels[0] - levels[1]),
                        "pass": abs(sm["sole"] - fm["sole"]) <= 3
                        and abs(levels[0] - levels[1]) <= MAX_DEPTH_OFFSET},
        "turn_in_place": {"feet_offset_vs_front_px": round(sm["feet_cx"] - fm["feet_cx"], 1),
                          "pass": abs(sm["feet_cx"] - fm["feet_cx"]) <= MAX_STEP},
        "height_vs_front": {"side": sm["height"], "front": fm["height"],
                            "ratio": round(sm["height"] / fm["height"], 4),
                            "pass": abs(sm["height"] / fm["height"] - 1) <= 0.03},
        "colour_de2000": {"per_family": de, "threshold": 6.0,
                          "pass": all(v is not None and v <= 6.0 for v in de.values())},
    }
    return res


def sheet(side: Image.Image, front: Image.Image, out: Path) -> None:
    from PIL import ImageDraw
    w, h = 480, 848
    s = Image.new("RGB", (w * 2, h), (240, 242, 246))
    for k, im in enumerate((front, side)):
        bg = Image.new("RGBA", im.size, (240, 242, 246, 255))
        bg.alpha_composite(im.convert("RGBA"))
        s.paste(bg.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), (k * w, 0))
    d = ImageDraw.Draw(s)
    gy = GROUND_Y * h / CANVAS[1]
    d.line([(0, gy), (2 * w, gy)], fill=(220, 60, 60), width=1)
    s.save(out, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("out")
    ap.add_argument("--front", default="assets/rig_adult_walk_v1/references/front_rest.png")
    a = ap.parse_args()
    front_im = Image.open(a.front).convert("RGBA")
    front = np.asarray(front_im, np.float32)
    aligned, info = align(Image.open(a.candidate), front)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    aligned.save(out)
    res = check(np.asarray(aligned, np.float32), front)
    report = {"candidate": a.candidate, "aligned": out.as_posix(), "alignment": info, "g1_numeric": res,
              "g1_numeric_pass": all(v["pass"] for v in res.values())}
    out.with_suffix(".g1.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    sheet(aligned, front_im, out.with_suffix(".sheet.jpg"))
    print(json.dumps({"alignment": info, "g1": res, "pass": report["g1_numeric_pass"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
