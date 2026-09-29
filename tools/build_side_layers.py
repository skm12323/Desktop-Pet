"""G4: partition the approved side key art into pixel-exact rig layers.

Every opaque pixel of the key art belongs to exactly one layer, so the rest-pose composite
reproduces the art. Layers are then extended a few px underneath the layers drawn above them
("underlap") so bilinear/mipmap filtering and small motions never open seams; larger hidden
regions (thighs under the skirt, torso under the arm, ...) come from later completion passes
stored in <out>/completions/<layer>.png (RGBA, canvas-registered, only used where hidden).

Inputs: SAM masks (tools/sam3_segment.py) + the layer plan below.
Outputs: <out>/layers_full/<id>.png (canvas-size RGBA; build_side_spec.py writes the trimmed
runtime textures to <out>/layers/), <out>/prep/partition.png, <out>/prep/partition.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# z order low -> high. "from": which SAM masks (or derived regions) feed the layer.
LAYERS = [
    {"id": "tail", "z": 0},
    {"id": "hair_back", "z": 5},
    {"id": "ear_fin_r", "z": 8},
    {"id": "arm_r", "z": 10},
    {"id": "leg_r", "z": 15},
    {"id": "leg_l", "z": 18},
    {"id": "skirt", "z": 20},
    {"id": "torso", "z": 25},
    {"id": "apron", "z": 30},
    {"id": "hair_side_r", "z": 35},
    {"id": "arm_l", "z": 40},
    {"id": "shoulder_frill_l", "z": 45},
    {"id": "head_base", "z": 50},
    {"id": "ear_fin_l", "z": 55},
    {"id": "ahoge", "z": 60},
    {"id": "headdress", "z": 65},
]
# overlap resolution: earlier wins
PRIORITY = ["ahoge", "ear_fin_r", "ear_fin_l", "face", "headdress", "shoulder_frill_l", "arm_l",
            "arm_r", "apron", "torso", "leg_l", "leg_r", "tail", "skirt", "hair"]
MASK_TO_LAYER = {"face": "head_base"}


def split_hair(hair: np.ndarray, far_side: dict) -> tuple[np.ndarray, np.ndarray]:
    """Far-side strands hanging in front of the far shoulder -> hair_side_r; rest -> hair_back."""
    h, w = hair.shape
    yy, xx = np.mgrid[0:h, 0:w]
    region = (xx >= far_side["x_min"]) & (yy >= far_side["y_min"])
    side = hair & region
    lab, n = ndimage.label(side)
    if n:
        sizes = ndimage.sum(side, lab, range(1, n + 1))
        side = np.isin(lab, 1 + np.where(sizes >= 200)[0])
    return hair & ~side, side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--key", default="assets/rig_adult_walk_v1/references/side_key.png")
    ap.add_argument("--masks", default="assets/rig_adult_walk_v1/prep", help="dir with mask_sources.json")
    ap.add_argument("--out", default="assets/rig_adult_walk_v1")
    ap.add_argument("--underlap", type=int, default=6, help="px each layer extends under higher layers")
    ap.add_argument("--hair-side-x", type=int, default=556)
    ap.add_argument("--hair-side-y", type=int, default=400)
    ap.add_argument("--labels-only", action="store_true", help="write the partition label map and stop")
    a = ap.parse_args()

    key = np.asarray(Image.open(a.key).convert("RGBA"))
    alpha = key[..., 3]
    opaque = alpha > 0
    H, W = alpha.shape
    md = Path(a.masks)
    sources = json.loads((md / "mask_sources.json").read_text(encoding="utf-8"))
    masks = {}
    for name in PRIORITY:
        for src in sources.get(name, []):
            src, _, box = src.partition("@")
            m = np.asarray(Image.open(md / f"{src}.png").convert("L")) > 127
            if box:
                x0, y0, x1, y1 = (int(v) for v in box.split(","))
                clip = np.zeros_like(m)
                clip[y0:y1, x0:x1] = True
                m = m & clip
            masks[name] = masks.get(name, np.zeros_like(m)) | m
    if "hair" in masks:
        back, side = split_hair(masks.pop("hair"), {"x_min": a.hair_side_x, "y_min": a.hair_side_y})
        masks["hair_back"], masks["hair_side_r"] = back, side
    order = [MASK_TO_LAYER.get(n, n) for n in PRIORITY if n != "hair"] + ["hair_side_r", "hair_back"]
    masks = {MASK_TO_LAYER.get(k, k): v for k, v in masks.items()}

    label = np.full((H, W), -1, np.int32)
    ids = [l["id"] for l in LAYERS]
    for name in order:
        if name in masks:
            free = (label < 0) & masks[name] & opaque
            label[free] = ids.index(name)
    unassigned = opaque & (label < 0)
    if unassigned.any():
        _, (iy, ix) = ndimage.distance_transform_edt(label < 0, return_indices=True)
        label[unassigned] = label[iy[unassigned], ix[unassigned]]
    # islands join their surrounding layer: < 40 px everywhere; for rigid parts (limbs, tail,
    # ears...) every piece not connected to the main body (< 5 % of it) - a stray outline
    # fragment would fly off as soon as the part moves
    multi = {"hair_back", "hair_side_r", "skirt", "torso", "apron", "headdress"}
    for k in range(len(ids)):
        lab, n = ndimage.label(label == k, structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            limit = 40 if ids[k] in multi else max(40, 0.05 * sizes.max())
            for comp in np.where(sizes < limit)[0] + 1:
                m = lab == comp
                ring = ndimage.binary_dilation(m, iterations=2) & ~m & (label >= 0)
                if ring.any():
                    label[m] = np.bincount(label[ring]).argmax()

    (Path(a.out) / "prep").mkdir(parents=True, exist_ok=True)
    Image.fromarray((label + 1).astype(np.uint8), "L").save(Path(a.out) / "prep" / "partition_labels.png")
    (Path(a.out) / "prep" / "partition_ids.json").write_bytes(
        json.dumps({"0": "transparent", **{str(i + 1): l for i, l in enumerate(ids)}}, indent=1).encode("utf-8"))
    if a.labels_only:
        print("[OK] labels written")
        return

    comp_dir = Path(a.out) / "completions"
    pm = Path(a.out) / "prep" / "peeled_mask.png"
    peeled = np.asarray(Image.open(pm)) > 127 if pm.exists() else np.zeros((H, W), bool)
    lay_dir = Path(a.out) / "layers_full"      # canvas-size; build_side_spec.py trims into layers/
    lay_dir.mkdir(parents=True, exist_ok=True)
    z_of = {l["id"]: l["z"] for l in LAYERS}
    stats = {}
    for k, lid in enumerate(ids):
        own = label == k
        rgba = np.zeros_like(key)
        rgba[own] = key[own]
        above = np.isin(label, [ids.index(o) for o in ids if z_of[o] > z_of[lid]])
        # completion pass (only where hidden by higher layers)
        cp = comp_dir / f"{lid}.png"
        if cp.exists():
            c = np.asarray(Image.open(cp).convert("RGBA"))
            use = (c[..., 3] > 0) & ~own & above & (alpha == 255)
            rgba[use] = c[use]
            # 补全像素里与主体不相连的小碎片（< 30 px，如拷贝后跟的描边残点）会随肢体飞出——丢掉
            cc, n = ndimage.label(rgba[..., 3] > 0, structure=np.ones((3, 3)))
            if n > 1:
                sizes = ndimage.sum(np.ones_like(cc), cc, range(1, n + 1))
                for comp in np.where(sizes < 30)[0] + 1:
                    m = (cc == comp) & ~own
                    rgba[m] = 0
        # underlap: nearest own colour, full alpha of the covering pixel, only under higher layers
        cur = rgba[..., 3] > 0
        if a.underlap > 0 and cur.any():
            grow = ndimage.binary_dilation(cur, iterations=a.underlap) & ~cur & above & (alpha == 255) & ~peeled
            _, (iy, ix) = ndimage.distance_transform_edt(~cur, return_indices=True)
            rgba[grow, :3] = rgba[iy[grow], ix[grow], :3]
            rgba[grow, 3] = 255
        Image.fromarray(rgba, "RGBA").save(lay_dir / f"{lid}.png")
        stats[lid] = {"z": z_of[lid], "own_px": int(own.sum()), "total_px": int((rgba[..., 3] > 0).sum()),
                      "completion": cp.exists()}

    # verification: z-ordered composite == key art
    comp = np.zeros((H, W, 4), np.float32)
    for lid in sorted(ids, key=lambda i: z_of[i]):
        l = np.asarray(Image.open(lay_dir / f"{lid}.png"), np.float32) / 255.0
        a_ = l[..., 3:4]
        comp[..., :3] = l[..., :3] * a_ + comp[..., :3] * (1 - a_)
        comp[..., 3:4] = a_ + comp[..., 3:4] * (1 - a_)
    ref = key.astype(np.float32) / 255.0
    pm_c = comp[..., :3]                      # accumulated premultiplied colour
    pm_r = ref[..., :3] * ref[..., 3:4]
    err = np.abs(pm_c - pm_r).max(-1) * 255
    rep = {"key": a.key, "key_sha256": hashlib.sha256(Path(a.key).read_bytes()).hexdigest(),
           "underlap_px": a.underlap, "layers": stats,
           "composite_max_err_255": float(err.max()), "composite_mean_err_255": float(err[opaque].mean()),
           "unassigned_filled_px": int(unassigned.sum())}
    (Path(a.out) / "prep").mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "prep" / "partition.json").write_bytes(json.dumps(rep, indent=2).encode("utf-8"))

    rng = np.random.default_rng(3)
    pal = rng.integers(40, 255, (len(ids), 3))
    vis = np.full((H, W, 3), 255, np.uint8)
    vis[label >= 0] = pal[label[label >= 0]]
    edges = ndimage.morphological_gradient(label, size=3) != 0
    vis[edges & opaque] = 0
    Image.fromarray(vis).save(Path(a.out) / "prep" / "partition.png")
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=2))
    for lid, s in stats.items():
        print(f"  {lid:18s} z{s['z']:3d} own {s['own_px']:7d} total {s['total_px']:7d}")


if __name__ == "__main__":
    main()
