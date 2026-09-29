"""G4 v1 procedural completions for the side rig (hidden regions exposed by walking).

Writes assets/rig_adult_walk_v1/completions/<layer>.png (canvas RGBA). build_side_layers.py
only uses completion pixels that are hidden in the rest pose (under a higher layer), so the
rest composite stays pixel-exact; these pixels only show when parts move.

Methods:
  extrude  - thighs: extrapolate the fitted left/right leg edges upward and fill with the
             averaged across-leg colour profile (legs rotate about the hip, so the thigh must
             continue up under the skirt)
  image    - a generated, registered RGBA of the complete part (Qwen redraw + BiRefNet matte)
  copy     - a scaled, anchored copy of another layer's own pixels (far shoe heel <- near shoe heel)
  diffuse  - nearest-own-colour fill + smoothing inside a region (hair / torso / skirt / tail /
             far arm under the near arm, far shoe heel under the near shoe, ...)
Region = dilate(own layer, grow px) & union(listed occluder layers). Replaceable per layer by a
Qwen inpaint later (same file name).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PLAN = {
    "leg_l": {"method": "extrude", "fit": [1168, 1270], "band": [1168, 1180], "stop_y": 860},
    "leg_r": {"method": "extrude", "fit": [1182, 1270], "band": [1182, 1194], "stop_y": 860,
              # 远侧鞋后跟（被近侧鞋头挡住）：拷贝近侧鞋自己的后跟（同款鞋），缩放 0.95、
              # 后跟底角对齐 (400,1604) → (504,1565)；旧的邻色填充是一块黑方片，脚一倾斜就露出来
              "then": {"method": "copy", "src": "leg_l", "src_box": [392, 1468, 470, 1608],
                       "anchor_src": [400, 1604], "anchor_dst": [504, 1565], "scale": 0.95}},
    "tail": {"method": "diffuse", "grow": 70, "under": ["skirt", "arm_l"], "box": [0, 700, 395, 1260]},
    "hair_back": {"method": "diffuse", "grow": 70, "under": ["arm_l", "shoulder_frill_l", "torso"]},
    "torso": {"method": "diffuse", "grow": 60, "under": ["arm_l", "shoulder_frill_l", "apron", "hair_side_r"]},
    "skirt": {"method": "diffuse", "grow": 70, "under": ["arm_l", "arm_r", "apron", "torso"]},
    # far arm: Qwen redrew the whole arm alone on white (prep/peel/arm_r_extract_s1.png), BiRefNet
    # matte, registered to the visible far-arm pixels (best shift 0,-2 px; 93 % overlap)
    "arm_r": {"method": "image", "path": "prep/peel/arm_r_extract_s1_rgba.png", "offset": [528, 430]},
    "apron": {"method": "diffuse", "grow": 20, "under": ["arm_r", "arm_l"]},
}


# Qwen "peels": image with an occluding part removed (tools/qwen_edit.py, reference-first edit
# of the key-art crop). Pixels under the occluder are handed to the layers behind it by a
# nearest-neighbour classifier on (position, colour) trained on the labelled ring around it.
PEELS = [
    {"id": "arm_l", "crop": [160, 400, 480, 976], "image": "prep/peel/arm_l_edit_s1.png",
     "occluder": ["arm_l"], "targets": ["hair_back", "torso", "apron", "skirt"]},
    # 围裙后面的裙子/衣身：裙摆随大腿摆动时会从围裙边缘下露出（旧邻色填充 = 黑色涂抹）
    {"id": "apron", "crop": [384, 643, 736, 1091], "image": "prep/peel/apron_edit_s1.png",
     "occluder": ["apron"], "targets": ["torso", "skirt"]},
]


def peel_assignments(root: Path, key: np.ndarray, labels: np.ndarray, idx: dict) -> dict:
    from scipy.spatial import cKDTree
    out: dict = {}
    for pl in PEELS:
        x0, y0, x1, y1 = pl["crop"]
        img = np.asarray(Image.open(root / pl["image"]).convert("RGB").resize(
            (x1 - x0, y1 - y0), Image.Resampling.LANCZOS), np.float32)
        lab = labels[y0:y1, x0:x1]
        occ = np.isin(lab, [idx[o] for o in pl["occluder"]])
        region = ndimage.binary_dilation(occ, iterations=3)
        # background (label 0) is a class too: the peel shows white paper where the arm hung
        # outside the body, and those pixels must stay transparent
        ring = ndimage.binary_dilation(region, iterations=40) & ~region &             np.isin(lab, [0] + [idx[t] for t in pl["targets"]])
        ry, rx = np.where(ring)
        feat = lambda yy, xx: np.column_stack([xx / 25.0, yy / 25.0, img[yy, xx] / 18.0])
        tree = cKDTree(feat(ry, rx))
        qy, qx = np.where(region)
        _, nn = tree.query(feat(qy, qx), k=5)
        votes = lab[ry[nn], rx[nn]]
        choice = np.array([np.bincount(v).argmax() for v in votes])
        # majority smoothing of the class map (removes speckle / jagged class borders)
        classes = np.unique(choice)
        cmap = np.full(region.shape, -1, np.int32)
        cmap[qy, qx] = choice
        for _ in range(3):
            score = np.stack([ndimage.uniform_filter((cmap == c).astype(np.float32), 7) for c in classes])
            best = classes[score.argmax(0)]
            cmap[region] = best[region]
        # colour decides the body/background border (the peel shows it exactly): white paper
        # labelled skirt/hair -> background; coloured pixels labelled background -> nearest body class
        white = (img.min(-1) > 190) & (img.max(-1) - img.min(-1) < 30)
        dark_cls = [idx[n] for n in ("skirt", "hair_back") if n in pl["targets"]]
        cmap[region & white & np.isin(cmap, dark_cls)] = 0
        body = region & (cmap > 0)
        fix = region & (cmap == 0) & ~white
        if fix.any() and body.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~body, return_indices=True)
            cmap[fix] = cmap[iy[fix], ix[fix]]
        choice = cmap[qy, qx]
        for tname in pl["targets"]:
            sel = choice == idx[tname]
            if sel.any():
                m = np.zeros(labels.shape, bool)
                m[qy[sel] + y0, qx[sel] + x0] = True
                # drop 1-2 px stripes (old arm outline strokes split between body/background)
                own_t = labels == idx[tname]
                m = ndimage.binary_opening(m | own_t, iterations=2) & m
                col = np.zeros(key.shape[:2] + (3,), np.uint8)
                col[qy[sel] + y0, qx[sel] + x0] = np.clip(img[qy[sel], qx[sel]] + 0.5, 0, 255).astype(np.uint8)
                col[~m] = 0
                prev = out.get(tname)
                out[tname] = (m, col) if prev is None else (prev[0] | m, np.where(m[..., None], col, prev[1]))
    return out


def peel_regions(labels: np.ndarray, idx: dict) -> np.ndarray:
    """Union of all peeled areas: procedural fills must stay out of them."""
    m = np.zeros(labels.shape, bool)
    for pl in PEELS:
        x0, y0, x1, y1 = pl["crop"]
        occ = np.isin(labels[y0:y1, x0:x1], [idx[o] for o in pl["occluder"]])
        m[y0:y1, x0:x1] |= ndimage.binary_dilation(occ, iterations=3)
    return m


def diffuse_fill(rgba: np.ndarray, own: np.ndarray, region: np.ndarray, iters: int = 3) -> np.ndarray:
    out = rgba.copy()
    if not region.any():
        return out
    _, (iy, ix) = ndimage.distance_transform_edt(~own, return_indices=True)
    out[region, :3] = rgba[iy[region], ix[region], :3]
    out[region, 3] = 255
    f = out[..., :3].astype(np.float32)
    for _ in range(iters):
        blur = np.stack([ndimage.uniform_filter(f[..., c], 9) for c in range(3)], -1)
        f[region] = blur[region]
    out[..., :3] = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return out


def extrude(rgba: np.ndarray, own: np.ndarray, p: dict) -> np.ndarray:
    """Continue the thigh upward: fit straight lines to its left/right silhouette edges over
    `fit` rows, extrapolate both to stop_y, and fill each row by resampling the averaged
    colour profile of the `band` rows across the (normalised) width - outlines included."""
    out = rgba.copy()
    fy0, fy1 = p["fit"]
    rows = [y for y in range(fy0, fy1) if own[y].any()]
    lx = np.array([np.where(own[y])[0].min() for y in rows], np.float32)
    rx = np.array([np.where(own[y])[0].max() for y in rows], np.float32)
    ry = np.array(rows, np.float32)
    kl, bl = np.polyfit(ry, lx, 1)
    kr, br = np.polyfit(ry, rx, 1)
    y0, y1 = p["band"]
    prof = []
    for y in range(y0, y1):
        xs = np.where(own[y])[0]
        seg = rgba[y, xs.min():xs.max() + 1, :3].astype(np.float32)
        u = np.linspace(0, 1, 64)
        prof.append(np.stack([np.interp(u, np.linspace(0, 1, len(seg)), seg[:, c]) for c in range(3)], -1))
    prof = np.mean(prof, 0)                         # (64, 3)
    top = min(np.where(own.any(1))[0].min(), y0)
    for y in range(y0, p["stop_y"] - 1, -1):
        xl, xr = kl * y + bl, kr * y + br
        xa, xb = int(np.floor(xl)), int(np.ceil(xr))
        for x in range(xa, xb + 1):
            if own[y, x] and y >= top:
                continue
            u = np.clip((x - xl) / max(xr - xl, 1.0), 0, 1)
            col = np.array([np.interp(u, np.linspace(0, 1, 64), prof[:, c]) for c in range(3)])
            out[y, x, :3] = np.clip(col + 0.5, 0, 255).astype(np.uint8)
            out[y, x, 3] = 255
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="assets/rig_adult_walk_v1")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    root = Path(a.root)
    key = np.asarray(Image.open(root / "references" / "side_key.png").convert("RGBA"))
    labels = np.asarray(Image.open(root / "prep" / "partition_labels.png"))
    ids = json.loads((root / "prep" / "partition_ids.json").read_text(encoding="utf-8"))
    idx = {v: int(k) for k, v in ids.items()}
    (root / "completions").mkdir(exist_ok=True)
    peels = peel_assignments(root, key, labels, idx)
    peeled = peel_regions(labels, idx)
    # build_side_layers must not underlap into peeled areas: there the peel already decided
    # which layer (or background) is behind the occluder
    Image.fromarray((peeled * 255).astype(np.uint8), "L").save(root / "prep" / "peeled_mask.png")
    for lid, p in PLAN.items():
        if a.only and lid not in a.only.split(","):
            continue
        own = labels == idx[lid]
        rgba = np.zeros_like(key)
        rgba[own] = key[own]
        if lid in peels:                          # Qwen peel first, procedural fill for the rest
            m, col = peels[lid]
            m = m & ~own
            rgba[m, :3] = col[m]
            rgba[m, 3] = 255
        steps = [p] + ([p["then"]] if "then" in p else [])
        for st in steps:
            cur = rgba[..., 3] > 0
            if st["method"] == "extrude":
                rgba = extrude(rgba, cur, st)
            elif st["method"] == "copy":
                src_own = labels == idx[st["src"]]
                x0, y0, x1, y1 = st["src_box"]
                crop = key[y0:y1, x0:x1].copy()
                crop[~src_own[y0:y1, x0:x1]] = 0
                sc = float(st.get("scale", 1.0))
                im = Image.fromarray(crop, "RGBA").resize(
                    (max(1, round((x1 - x0) * sc)), max(1, round((y1 - y0) * sc))), Image.Resampling.LANCZOS)
                ax, ay = st["anchor_src"]
                bx, by = st["anchor_dst"]
                ox, oy = round(bx - (ax - x0) * sc), round(by - (ay - y0) * sc)
                patch = np.zeros_like(rgba)
                ph, pw = im.size[1], im.size[0]
                patch[oy:oy + ph, ox:ox + pw] = np.asarray(im)
                use = (patch[..., 3] > 127) & ~cur
                rgba[use, :3] = patch[use, :3]
                rgba[use, 3] = 255
            elif st["method"] == "image":
                src = np.asarray(Image.open(root / st["path"]).convert("RGBA"))
                ox, oy = st["offset"]
                h, w = src.shape[:2]
                patch = np.zeros_like(rgba)
                patch[oy:oy + h, ox:ox + w] = src
                use = (patch[..., 3] > 127) & ~cur
                rgba[use, :3] = patch[use, :3]
                rgba[use, 3] = 255
            else:
                under = np.isin(labels, [idx[u] for u in st["under"]])
                region = ndimage.binary_dilation(cur, iterations=st["grow"]) & under & ~cur & ~peeled
                if "box" in st:
                    x0, y0, x1, y1 = st["box"]
                    inbox = np.zeros_like(region)
                    inbox[y0:y1, x0:x1] = True
                    region &= inbox
                if "poly" in st:
                    from PIL import ImageDraw
                    pm = Image.new("L", (region.shape[1], region.shape[0]), 0)
                    ImageDraw.Draw(pm).polygon([tuple(v) for v in st["poly"]], fill=255)
                    region &= np.asarray(pm) > 127
                rgba = diffuse_fill(rgba, cur, region)
        Image.fromarray(rgba, "RGBA").save(root / "completions" / f"{lid}.png")
        print(f"[OK] {lid}: +{int((rgba[..., 3] > 0).sum() - own.sum())} px")


if __name__ == "__main__":
    main()
