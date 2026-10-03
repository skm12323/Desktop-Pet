"""G4: partition the approved side key art into pixel-exact rig layers.

Every key-art pixel belongs to exactly one layer. Near-opaque occluders (including internal
alpha 252-254 left by matting) admit hidden completion pixels; the small rest-composite delta
is measured below. Layers are then extended underneath the layers drawn above them
("underlap") so bilinear/mipmap filtering and small motions never open seams; larger hidden
regions (thighs under the skirt, torso under the arm, ...) come from later completion passes
stored in <out>/completions/<layer>.png (RGBA, canvas-registered, only used where hidden).

Inputs: SAM masks (tools/sam3_segment.py) + the layer plan below (ADULT side rig), or another
rig's plan via --plan JSON: {"layers": [{"id", "z"}...], "priority": [...], "mask_to_layer": {...},
"multi": [...], "hair_split": {"layer": "hair_side_r", "x_min": .., "y_min": ..} | null}.
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


def repair_near_cuff_labels(key: np.ndarray, label: np.ndarray, ids: list[str]) -> int:
    """The approved 960x1696 art has a lace fragment SAM attached to the skirt.

    Keep these source pixels with the near arm: otherwise the frill stays on the
    dress while the wrist moves away. Limit to the registered cuff and adjacent
    arm pixels, so white skirt hems and waist-bow pixels remain in their layers.
    """
    if key.shape[:2] != (1696, 960):
        return 0
    yy, xx = np.mgrid[:key.shape[0], :key.shape[1]]
    rgb = key[..., :3].astype(np.int16)
    lace = (rgb.min(-1) > 185) & (np.ptp(rgb, axis=-1) < 50)
    roi = (xx >= 250) & (xx <= 335) & (yy >= 815) & (yy <= 875)
    arm = ids.index("arm_l")
    skirt = ids.index("skirt")
    near = ndimage.binary_dilation(label == arm, iterations=6)
    misplaced = (label == skirt) & lace & roi & near
    label[misplaced] = arm
    return int(misplaced.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--key", default="assets/rig_adult_walk_v1/references/side_key.png")
    ap.add_argument("--masks", default="assets/rig_adult_walk_v1/prep", help="dir with mask_sources.json")
    ap.add_argument("--out", default="assets/rig_adult_walk_v1")
    ap.add_argument("--underlap", type=int, default=6, help="px each layer extends under higher layers")
    ap.add_argument("--completion-alpha-min", type=int, default=240,
                    help="near-opaque key pixels count as occluders (matting often yields alpha 252-254 inside sleeves)")
    ap.add_argument("--hair-side-x", type=int, default=556)
    ap.add_argument("--hair-side-y", type=int, default=400)
    ap.add_argument("--labels-only", action="store_true", help="write the partition label map and stop")
    ap.add_argument("--plan", default="", help="layer plan JSON (default: the ADULT side plan in this file)")
    a = ap.parse_args()

    layers, priority, mask_to_layer = LAYERS, PRIORITY, MASK_TO_LAYER
    multi = {"hair_back", "hair_side_r", "skirt", "torso", "apron", "headdress"}
    hair_split = {"layer": "hair_side_r", "x_min": a.hair_side_x, "y_min": a.hair_side_y}
    if a.plan:
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        layers, priority = plan["layers"], plan["priority"]
        mask_to_layer = plan.get("mask_to_layer", {})
        multi = set(plan.get("multi", []))
        hair_split = plan.get("hair_split")

    key = np.asarray(Image.open(a.key).convert("RGBA"))
    alpha = key[..., 3]
    occluder_alpha = alpha >= max(128, min(255, a.completion_alpha_min))
    opaque = alpha > 0
    H, W = alpha.shape
    md = Path(a.masks)
    sources = json.loads((md / "mask_sources.json").read_text(encoding="utf-8"))
    masks = {}
    for name in priority:
        for src in sources.get(name, []):
            src, _, box = src.partition("@")
            m = np.asarray(Image.open(md / f"{src}.png").convert("L")) > 127
            if box:
                x0, y0, x1, y1 = (int(v) for v in box.split(","))
                clip = np.zeros_like(m)
                clip[y0:y1, x0:x1] = True
                m = m & clip
            masks[name] = masks.get(name, np.zeros_like(m)) | m
    if hair_split and "hair" in masks:
        back, side = split_hair(masks.pop("hair"), hair_split)
        masks["hair_back"], masks[hair_split["layer"]] = back, side
        order = [mask_to_layer.get(n, n) for n in priority if n != "hair"] + [hair_split["layer"], "hair_back"]
    else:
        order = [mask_to_layer.get(n, n) for n in priority]
    masks = {mask_to_layer.get(k, k): v for k, v in masks.items()}

    label = np.full((H, W), -1, np.int32)
    ids = [l["id"] for l in layers]
    for name in order:
        if name in masks:
            free = (label < 0) & masks[name] & opaque
            label[free] = ids.index(name)
    unassigned = opaque & (label < 0)
    if unassigned.any():
        _, (iy, ix) = ndimage.distance_transform_edt(label < 0, return_indices=True)
        label[unassigned] = label[iy[unassigned], ix[unassigned]]
    plan_cfg = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else {}
    hc = plan_cfg.get("hair_material_cleanup")
    if hc:
        rgb = key[..., :3].astype(np.int16)
        lum = rgb @ np.array([.299,.587,.114])
        hair_seed = (rgb[..., 2]-rgb[..., 0] > 45) & (lum > 95) & opaque
        distance = ndimage.distance_transform_edt(~hair_seed)
        bad = np.isin(label, [ids.index(n) for n in hc["layers"]])
        bad &= (distance > hc.get("rim_px", 4)) & (np.arange(H)[:, None] > hc["start_y"])
        navy = bad & (lum < 140)
        above_waist = np.arange(H)[:, None] < hc["waist_y"]
        label[navy & above_waist] = ids.index("torso")
        label[navy & ~above_waist] = ids.index("skirt")
        # Cloth fragments enclosed by SAM's broad curl mask belong to the
        # nearby solid cloth/cuff, rather than move with the back hair.
        white = bad & (lum > 185)
        candidates = opaque & ~np.isin(label, [ids.index(n) for n in hc["layers"]])
        _, (iy, ix) = ndimage.distance_transform_edt(~candidates, return_indices=True)
        label[white] = label[iy[white], ix[white]]
    if plan_cfg.get("antialias_owner_nearest"):
        # Matte fringes inherit the material of the nearby solid pixel. SAM
        # often leaves these fine curl outlines to a broad skirt catch-all.
        solid = (key[..., 3] >= 200) & (label >= 0)
        _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
        fringe = opaque & ~solid
        label[fringe] = label[iy[fringe], ix[fringe]]
    # islands join their surrounding layer: < 40 px everywhere; for rigid parts (limbs, tail,
    # ears...) every piece not connected to the main body (< 5 % of it) - a stray outline
    # fragment would fly off as soon as the part moves
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

    cuff_fixed = 0 if a.plan else repair_near_cuff_labels(key, label, ids)
    # plan "edge_to_occluder": a lower layer's rim along a higher layer is the occluder's outline /
    # anti-aliasing in the art; left on the lower layer it slides along as a dark seam when that
    # layer moves. Hand those pixels to the occluder (the lower layer's completion covers them).
    edge_moved = 0
    for rule in (json.loads(Path(a.plan).read_text(encoding="utf-8")).get("edge_to_occluder", []) if a.plan else []):
        lo, hi = ids.index(rule["layer"]), ids.index(rule["occluder"])
        near = ndimage.binary_dilation(label == hi, iterations=int(rule["px"])) & (label == lo)
        if rule.get("box"):
            x0, y0, x1, y1 = rule["box"]
            roi = np.zeros((H, W), bool)
            roi[y0:y1, x0:x1] = True
            near &= roi
        label[near] = hi
        edge_moved += int(near.sum())
    (Path(a.out) / "prep").mkdir(parents=True, exist_ok=True)
    Image.fromarray((label + 1).astype(np.uint8), "L").save(Path(a.out) / "prep" / "partition_labels.png")
    (Path(a.out) / "prep" / "partition_ids.json").write_bytes(
        json.dumps({"0": "transparent", **{str(i + 1): l for i, l in enumerate(ids)}}, indent=1).encode("utf-8"))
    if a.labels_only:
        print(f"[OK] labels written; cuff reassigned: {cuff_fixed} px")
        return

    comp_dir = Path(a.out) / "completions"
    pm = Path(a.out) / "prep" / "peeled_mask.png"
    peeled = np.asarray(Image.open(pm)) > 127 if pm.exists() else np.zeros((H, W), bool)
    lay_dir = Path(a.out) / "layers_full"      # canvas-size; build_side_spec.py trims into layers/
    lay_dir.mkdir(parents=True, exist_ok=True)
    z_of = {l["id"]: l["z"] for l in layers}
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
            use = (c[..., 3] > 0) & ~own & above & occluder_alpha
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
            grow = ndimage.binary_dilation(cur, iterations=a.underlap) & ~cur & above & occluder_alpha & ~peeled
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
    rep["near_cuff_reassigned_px"] = cuff_fixed
    rep["edge_to_occluder_px"] = edge_moved
    rep["completion_alpha_min"] = max(128, min(255, a.completion_alpha_min))
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
