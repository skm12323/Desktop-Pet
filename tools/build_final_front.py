"""FINAL F2: front skinned rig (static pose: breath / sway / blink / look-at) from the F1 key art.

Pipeline (one stage per call, or --stage all):
  masks        SAM masks (prep/sam_masks*, tools/sam3_segment.py) + geometric regions
               -> prep/derived/*.png, prep/mask_sources.json, prep/layer_plan.json
  labels       tools/build_side_layers.py --plan ... --labels-only (pixel-exact partition)
  completions  procedural fills where a layer is hidden at rest (diffuse, as side_completions.py)
  layers       tools/build_side_layers.py --plan ... (layers_full/, rest composite check)
  eyes         iris -> pupil_l/r (cut from head_base, sclera filled underneath);
               upper lash -> eyelid_l/r (copied, head_base keeps it - same as the side rig)
  spec         47-bone skeleton on the FINAL canvas + layers + face mechanics -> spec.json,
               trimmed runtime textures -> layers/
  mesh         tools/mesh_generator.py -> mesh/mesh_data.json

Outputs land in assets/rig_final/, which pet/rig/spec.py probes for the "final" stage
(spec.json + mesh/mesh_data.json + layers/).

Usage: D:\\anaconda3\\python.exe -X utf8 tools/build_final_front.py --stage all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "assets" / "rig_final"
KEY = PKG / "references" / "front_key.png"
W, H = 1024, 1824

# ---- partition plan (z low -> high). Layer ids follow the ADULT front rig where they map 1:1.
LAYERS = [
    {"id": "tail", "z": 0},
    {"id": "hair_back_l", "z": 30},
    {"id": "hair_back_r", "z": 40},
    {"id": "leg_l", "z": 50},
    {"id": "leg_r", "z": 55},
    {"id": "skirt", "z": 60},
    {"id": "torso", "z": 65},
    {"id": "apron", "z": 70},
    {"id": "arm_l", "z": 75},
    {"id": "arm_r", "z": 76},
    {"id": "hair_front", "z": 78},
    {"id": "head_base", "z": 100},
    {"id": "ear_fin_l", "z": 130},
    {"id": "ear_fin_r", "z": 135},
    {"id": "bangs", "z": 140},
    {"id": "ahoge_headdress", "z": 150},
]
PRIORITY = ["ahoge_headdress", "ear_fin_l", "ear_fin_r", "bangs", "head_base", "arm_l", "arm_r",
            "apron", "leg_l", "leg_r", "tail", "hair_front", "hair_back_l", "hair_back_r", "torso", "skirt"]
MULTI = ["hair_back_l", "hair_back_r", "hair_front", "bangs", "skirt", "torso", "apron", "ahoge_headdress"]
# geometric regions (canvas px) used to split SAM's single hair mask and to catch dress pixels
# SAM's text prompts missed (outer navy dress, bare shoulders)
HEAD_BOX = (405, 110, 640, 330)        # hair inside = bangs + crown (moves with the head)
FRONT_LOCK_BOX = (390, 330, 650, 620)  # hair inside = locks hanging in front of the chest
HAIR_SPLIT_X = 512
TORSO_BOX = (370, 320, 660, 620)
SKIRT_BOX = (130, 560, 890, 1690)
# white cuff frills SAM's sleeve masks miss (they would stay on the skirt when the arm moves)
CUFF_BOX = {"l": (240, 770, 340, 850), "r": (690, 770, 790, 850)}
# whole hands: SAM's "hand" masks miss fingers, which then stay on the skirt as a ghost hand
HAND_BOX = {"l": (228, 815, 348, 965), "r": (676, 815, 800, 965)}
# everything below the petticoat hem: stockings + shoes (SAM leg masks start at the shoe strap)
LEG_BOX = (430, 1572, 600, 1770)
# tail body continuing left under the petticoat (dark pixels below the white frill)
TAIL_LOW_BOX = (610, 1545, 800, 1620)
# apron frill edge: SAM's apron mask is loose there; white/light-grey cloth inside this box is apron
APRON_BOX = (340, 600, 700, 1120)
TAIL_RIM_BOX = (800, 1200, 1010, 1545)
# Qwen "remove both arms" edit of this canvas crop (single-ref, unmasked; only the pixels the arms
# hide at rest are used). Each hidden pixel goes to the nearest of the target layers.
ARM_PEEL = {"raw": "prep/peel/arms_removed_s1.png", "offset": (128, 432),
            "occluders": ["arm_l", "arm_r"], "targets": ["hair_back_l", "hair_back_r", "skirt", "torso", "apron"]}
# whole-tail Qwen redraw (green bg, prep/peel/tail_redraw_s1.png) on this canvas crop at 2x
TAIL_PEEL = {"raw": "prep/peel/tail_redraw_s1.png", "crop": (560, 1150, 1024, 1640),
             "out": "prep/peel/tail_redraw_canvas.png"}
SOURCES = {
    "ahoge_headdress": ["sam_masks_text/t_ahoge", "sam_masks_text/t_headdress", "sam_masks_text/t_hairbow"],
    "ear_fin_l": ["sam_masks/fin_l", "sam_masks_text/t_fin_l"],
    "ear_fin_r": ["sam_masks/fin_r", "sam_masks_text/t_fin_r"],
    "bangs": ["derived/bangs"],
    "head_base": ["sam_masks_text/t_face", "sam_masks/eye_l", "sam_masks/eye_r"],
    "arm_l": ["sam_masks_text/t_sleeve_l", "sam_masks_text/t_hand_l", "derived/cuff_l", "derived/hand_l"],
    "arm_r": ["sam_masks/sleeve_r", "sam_masks_text/t_sleeve_r", "sam_masks_text/t_hand_r", "derived/cuff_r", "derived/hand_r"],
    "apron": ["sam_masks_text/t_apron", "derived/apron_white"],
    "leg_l": ["sam_masks/leg_l", "derived/leg_l"],
    "leg_r": ["sam_masks/leg_r", "derived/leg_r"],
    "tail": ["sam_masks/tail", "sam_masks_text/t_tail", "derived/tail_low", "derived/tail_rim"],
    "hair_front": ["derived/hair_front"],
    "hair_back_l": ["derived/hair_back_l"],
    "hair_back_r": ["derived/hair_back_r"],
    "torso": ["sam_masks_text/t_torso", "sam_masks_text/t_collar", "sam_masks_text/t_bowtie",
              "sam_masks_text/t_skin", "derived/torso_box"],
    "skirt": ["sam_masks_text/t_skirt", "derived/skirt_box"],
}
# procedural completions: fill where hidden by the listed higher layers, within `grow` px of the
# layer's own pixels (only hidden pixels are kept by build_side_layers, so the rest composite
# stays exact; these only show when parts sway/breathe)
COMPLETIONS = {
    "tail": {"grow": 120, "under": ["skirt"], "image": TAIL_PEEL["out"]},
    "hair_back_l": {"grow": 40, "interior_src": True, "under": ["torso", "arm_l", "skirt", "head_base", "ear_fin_l", "hair_front"]},
    "hair_back_r": {"grow": 40, "interior_src": True, "under": ["torso", "arm_r", "skirt", "head_base", "ear_fin_r", "hair_front"]},
    "leg_l": {"grow": 50, "under": ["skirt"]},
    "leg_r": {"grow": 50, "under": ["skirt"]},
    "skirt": {"grow": 50, "interior_src": True, "src_max_lum": 110, "aline_hull": (980, 1450), "under": ["arm_l", "arm_r", "apron", "torso"]},
    "torso": {"grow": 40, "interior_src": True, "under": ["arm_l", "arm_r", "hair_front", "apron", "head_base", "bangs"]},
    "apron": {"grow": 16, "under": ["arm_l", "arm_r", "hair_front"]},
    "head_base": {"grow": 40, "under": ["bangs", "ear_fin_l", "ear_fin_r", "ahoge_headdress"], "skin_only": True},
}

# ---- eyes (canvas px, measured on front_key.png)
EYE_BOX = {"l": (436, 226, 500, 274), "r": (528, 218, 598, 268)}
# blink zone [cx, top, bottom, radius] (mesh_generator): lash top -> lower lid
BLINK_ZONES = {"l": [470, 238, 270, 31], "r": [563, 228, 263, 30]}
BLINK_BLEND_PX = 10       # the eyes are 33 px apart: the default 25 px side bands overlap and double the squash

# ---- skeleton: ADULT's 47 bones (names drive motion.py), joints placed on the FINAL art (px)
BONES_PX = [
    ("root_hip", None, (512, 720), (-4, 4), False),
    ("spine", "root_hip", (512, 590), (-3, 3), False),
    ("chest", "spine", (512, 470), (-4, 4), False),
    ("neck", "chest", (512, 365), (-5, 5), False),
    ("head", "neck", (512, 330), (-10, 10), False),
    # the tail leaves the petticoat horizontally, curls up at x~880 and ends in the fluke: the
    # hidden root barely moves (it would poke out under the frill), the sway lives in the curl
    ("tail_01", "root_hip", (700, 1580), (-1.5, 1.5), True),
    ("tail_02", "tail_01", (880, 1530), (-5, 5), True),
    ("tail_03", "tail_02", (895, 1420), (-8, 8), True),
    ("tail_fluke", "tail_03", (905, 1330), (-12, 12), True),
    ("hair_back_l_01", "head", (420, 220), (-6, 6), True),
    ("hair_back_l_02", "hair_back_l_01", (300, 500), (-9, 9), True),
    ("hair_back_l_03", "hair_back_l_02", (230, 780), (-12, 12), True),
    ("hair_back_r_01", "head", (604, 220), (-6, 6), True),
    ("hair_back_r_02", "hair_back_r_01", (724, 500), (-9, 9), True),
    ("hair_back_r_03", "hair_back_r_02", (794, 780), (-12, 12), True),
    # only the shins show below the ankle-length skirt: the leg chain pivots just above the hem,
    # so the front-walk gait (14 deg thigh swing) shuffles the shoes instead of throwing them sideways
    ("upper_leg_l", "root_hip", (472, 1480), (-4, 4), True),
    ("lower_leg_l", "upper_leg_l", (472, 1640), (-6, 3), True),
    ("foot_l", "lower_leg_l", (472, 1720), (-4, 4), False),
    ("upper_leg_r", "root_hip", (554, 1480), (-4, 4), True),
    ("lower_leg_r", "upper_leg_r", (554, 1640), (-6, 3), True),
    ("foot_r", "lower_leg_r", (554, 1720), (-4, 4), False),
    ("ear_fin_l", "head", (432, 292), (-12, 12), False),
    ("ear_fin_r", "head", (594, 286), (-12, 12), False),
    ("skirt_root", "root_hip", (512, 640), (-4, 4), False),
    ("skirt_hem_l", "skirt_root", (300, 1480), (-6, 6), False),
    ("skirt_hem_r", "skirt_root", (724, 1480), (-6, 6), False),
    ("apron_root", "spine", (512, 610), (-4, 4), False),
    ("apron_tip", "apron_root", (512, 1050), (-8, 8), False),
    ("upper_arm_l", "chest", (398, 482), (-20, 20), True),
    ("forearm_l", "upper_arm_l", (330, 660), (-25, 15), True),
    ("hand_l", "forearm_l", (300, 840), (-15, 15), False),
    ("upper_arm_r", "chest", (626, 482), (-20, 20), True),
    ("forearm_r", "upper_arm_r", (696, 660), (-25, 15), True),
    ("hand_r", "forearm_r", (726, 840), (-15, 15), False),
    ("pupil_l", "head", (0, 0), (0, 0), False),        # set from the iris centre
    ("pupil_r", "head", (0, 0), (0, 0), False),
    ("eyelid_l", "head", (468, 240), (0, 0), False),
    ("eyelid_r", "head", (563, 230), (0, 0), False),
    ("bangs_01", "head", (512, 150), (-6, 6), True),
    ("bangs_02", "bangs_01", (512, 200), (-8, 8), True),
    ("bangs_03", "bangs_02", (512, 250), (-10, 10), True),
    ("hair_side_l_01", "head", (432, 330), (-8, 8), True),
    ("hair_side_l_02", "hair_side_l_01", (440, 460), (-12, 12), True),
    ("hair_side_r_01", "head", (594, 330), (-8, 8), True),
    ("hair_side_r_02", "hair_side_r_01", (588, 460), (-12, 12), True),
    ("ahoge_01", "head", (440, 110), (-15, 15), True),
    ("ahoge_02", "ahoge_01", (405, 40), (-25, 25), True),
]
SPEC_LAYERS = {
    "tail": {"bind_bone": "tail_01", "influence_bones": ["tail_01", "tail_02", "tail_03", "tail_fluke"],
             "rigid_above": [{"bone": "tail_fluke", "y": 1330, "blend": 30}]},
    "hair_back_l": {"bind_bone": "hair_back_l_01",
                    "influence_bones": ["head", "hair_back_l_01", "hair_back_l_02", "hair_back_l_03"]},
    "hair_back_r": {"bind_bone": "hair_back_r_01",
                    "influence_bones": ["head", "hair_back_r_01", "hair_back_r_02", "hair_back_r_03"]},
    "leg_l": {"bind_bone": "lower_leg_l", "influence_bones": ["upper_leg_l", "lower_leg_l", "foot_l"]},
    "leg_r": {"bind_bone": "lower_leg_r", "influence_bones": ["upper_leg_r", "lower_leg_r", "foot_r"]},
    "skirt": {"bind_bone": "skirt_root",
              "influence_bones": ["root_hip", "skirt_root", "skirt_hem_l", "skirt_hem_r"]},
    "torso": {"bind_bone": "spine", "influence_bones": ["spine", "chest", "neck"]},
    "apron": {"bind_bone": "apron_root", "influence_bones": ["spine", "apron_root", "apron_tip"]},
    "arm_l": {"bind_bone": "upper_arm_l", "influence_bones": ["chest", "upper_arm_l", "forearm_l", "hand_l"]},
    "arm_r": {"bind_bone": "upper_arm_r", "influence_bones": ["chest", "upper_arm_r", "forearm_r", "hand_r"]},
    "hair_front": {"bind_bone": "hair_side_l_01",
                   "influence_bones": ["head", "hair_side_l_01", "hair_side_l_02", "hair_side_r_01", "hair_side_r_02"]},
    "head_base": {"bind_bone": "head", "influence_bones": ["neck", "head"]},
    "pupil_l": {"bind_bone": "pupil_l", "influence_bones": ["pupil_l"]},
    "pupil_r": {"bind_bone": "pupil_r", "influence_bones": ["pupil_r"]},
    "eyelid_l": {"bind_bone": "eyelid_l", "influence_bones": ["head", "eyelid_l"]},
    "eyelid_r": {"bind_bone": "eyelid_r", "influence_bones": ["head", "eyelid_r"]},
    "ear_fin_l": {"bind_bone": "ear_fin_l", "influence_bones": ["head", "ear_fin_l"]},
    "ear_fin_r": {"bind_bone": "ear_fin_r", "influence_bones": ["head", "ear_fin_r"]},
    "bangs": {"bind_bone": "bangs_01",
              "influence_bones": ["head", "bangs_01", "bangs_02", "bangs_03", "hair_side_l_01", "hair_side_r_01"]},
    "ahoge_headdress": {"bind_bone": "head", "influence_bones": ["head"]},
}
EYE_LAYERS_Z = {"pupil_l": 110, "pupil_r": 115, "eyelid_l": 120, "eyelid_r": 125}
GROUND_Y = 1764


def _box_mask(box) -> np.ndarray:
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def _load_mask(rel: str) -> np.ndarray:
    return np.asarray(Image.open(PKG / "prep" / f"{rel}.png").convert("L")) > 127


def _save_mask(rel: str, m: np.ndarray) -> None:
    p = PKG / "prep" / f"{rel}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m * 255).astype(np.uint8), "L").save(p)


def _write_json(path: Path, obj) -> None:
    path.write_bytes(json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def stage_masks() -> None:
    hair = _load_mask("sam_masks_text/t_hair")
    head = _box_mask(HEAD_BOX)
    front = _box_mask(FRONT_LOCK_BOX)
    bangs = (_load_mask("sam_masks_text/t_bangs") | (hair & head))
    hair_front = hair & front & ~bangs
    back = hair & ~bangs & ~hair_front
    xx = np.arange(W)[None, :]
    _save_mask("derived/bangs", bangs)
    _save_mask("derived/hair_front", hair_front)
    _save_mask("derived/hair_back_l", back & (xx < HAIR_SPLIT_X))
    _save_mask("derived/hair_back_r", back & (xx >= HAIR_SPLIT_X))
    key = np.asarray(Image.open(KEY).convert("RGBA"))
    lum = key[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    for side, box in CUFF_BOX.items():
        _save_mask(f"derived/cuff_{side}", _box_mask(box) & (lum > 200) & (key[..., 3] > 0))
        c = key[..., :3].astype(np.int16)
        skin = _box_mask(HAND_BOX[side]) & (c[..., 0] > c[..., 2] + 12) & (lum > 140) & (key[..., 3] > 0)
        skin = ndimage.binary_fill_holes(ndimage.binary_closing(skin, iterations=2))
        _save_mask(f"derived/hand_{side}", ndimage.binary_dilation(skin, iterations=2) & (key[..., 3] > 0)
                   & _box_mask(HAND_BOX[side]))
    sat = key[..., :3].max(-1).astype(np.int16) - key[..., :3].min(-1)
    white = _box_mask(APRON_BOX) & (lum > 165) & (sat < 40) & (key[..., 3] > 0)
    lab, _ = ndimage.label(white | _load_mask("sam_masks_text/t_apron"), structure=np.ones((3, 3)))
    main = np.unique(lab[_load_mask("sam_masks_text/t_apron")])
    _save_mask("derived/apron_white", white & np.isin(lab, main[main > 0]))   # skip skirt pinstripes
    # tail under the frill is a lighter blue-grey: saturated or dark, unlike the white/grey frill
    _save_mask("derived/tail_low", _box_mask(TAIL_LOW_BOX) & ((lum < 150) | (sat >= 30)) & (key[..., 3] > 0))
    # the tail's anti-aliased rim around the curl hole: SAM leaves it to the skirt, and a thin
    # orphan arc then stays put while the tail sways
    tail_sam = _load_mask("sam_masks/tail")
    _save_mask("derived/tail_rim", ndimage.binary_dilation(tail_sam, iterations=3) & (key[..., 3] > 0)
               & _box_mask(TAIL_RIM_BOX))
    legs = _box_mask(LEG_BOX) & (key[..., 3] > 0)
    _save_mask("derived/leg_l", legs & (xx < HAIR_SPLIT_X))
    _save_mask("derived/leg_r", legs & (xx >= HAIR_SPLIT_X))
    _save_mask("derived/torso_box", _box_mask(TORSO_BOX))
    _save_mask("derived/skirt_box", _box_mask(SKIRT_BOX))
    _write_json(PKG / "prep" / "mask_sources.json",
                {"_note": "FINAL front: union of SAM 3.1 masks per layer (sam_masks = point prompts, "
                          "sam_masks_text = concept prompts, derived = tools/build_final_front.py regions)",
                 **SOURCES})
    _write_json(PKG / "prep" / "layer_plan.json",
                {"layers": LAYERS, "priority": PRIORITY, "mask_to_layer": {}, "multi": MULTI, "hair_split": None,
                 "edge_to_occluder": [{"layer": "tail", "occluder": "skirt", "px": 3}]})
    print("[OK] derived masks + plan")


def _run_layers(labels_only: bool) -> None:
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "tools" / "build_side_layers.py"),
           "--key", str(KEY), "--masks", str(PKG / "prep"), "--out", str(PKG),
           "--plan", str(PKG / "prep" / "layer_plan.json")]
    if labels_only:
        cmd.append("--labels-only")
    subprocess.run(cmd, check=True)


def _diffuse_fill(rgba: np.ndarray, own: np.ndarray, region: np.ndarray, iters: int = 3) -> np.ndarray:
    out = rgba.copy()
    if not region.any():
        return out
    _, (iy, ix) = ndimage.distance_transform_edt(~own, return_indices=True)    # own = colour sources
    out[region, :3] = rgba[iy[region], ix[region], :3]
    out[region, 3] = 255
    f = out[..., :3].astype(np.float32)
    for _ in range(iters):
        blur = np.stack([ndimage.uniform_filter(f[..., c], 9) for c in range(3)], -1)
        f[region] = blur[region]
    out[..., :3] = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return out


def arm_peel(labels: np.ndarray, ids: list[str]) -> dict:
    path = PKG / ARM_PEEL["raw"]
    if not path.exists():
        return {}
    img = np.asarray(Image.open(path).convert("RGB"))
    ox, oy = ARM_PEEL["offset"]
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    inside = np.zeros((H, W), bool)
    inside[oy:oy + img.shape[0], ox:ox + img.shape[1]] = True
    occ = np.isin(labels, [ids.index(o) for o in ARM_PEEL["occluders"]]) & inside
    # classify the edit's pixels by material instead of by nearest layer: the edit narrows the
    # dress, so behind the forearms it paints hair (or background) where the skirt layer would be
    c = canvas.astype(np.int16)
    lum = c @ np.array([299, 587, 114]) / 1000
    sat = c.max(-1) - c.min(-1)
    hair = ((c[..., 2] - c[..., 0] > 45) & (lum > 85)) | (c[..., 2] - c[..., 0] > 55)   # navy cloth: b-r ~35
    navy = (lum < 95) & ~hair
    white = (lum > 195) & (sat < 30)
    paper = (lum > 228) & (sat < 16)

    def near(m: np.ndarray, px: int) -> np.ndarray:
        return ndimage.binary_dilation(m, iterations=px)
    yy, xx = np.mgrid[0:H, 0:W]
    waist = 600
    pick = {
        "hair_back_l": hair & (xx < HAIR_SPLIT_X),
        "hair_back_r": hair & (xx >= HAIR_SPLIT_X),
        "skirt": navy & (yy >= waist),
        # the edit's background is off-white: white only counts as cloth next to that cloth
        "torso": (navy | (white & near(labels == ids.index("torso"), 12))) & (yy < waist),
        "apron": white & ~paper & near(labels == ids.index("apron"), 12) & (yy >= waist),
    }
    out = {}
    for t, cls in pick.items():
        m = occ & cls
        out[t] = (m, canvas)
        print(f"[OK] arm peel -> {t}: {int(m.sum())} px")
    return out


def aline_hull(own: np.ndarray, labels: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """Inside of an A-line dress: straight left/right silhouette lines fitted to the rows where the
    dress edge meets the background (not an arm/hand). Hidden fills stop there - the hands hang
    partly over the background, and filling under them widened the dress into a hand-shaped blot."""
    y0, y1 = rows
    ls, rs = [], []
    for y in range(y0, y1):
        xs = np.where(own[y])[0]
        if len(xs) < 2:
            continue
        xl, xr = xs.min(), xs.max()
        if xl > 0 and labels[y, xl - 1] < 0:
            ls.append((y, xl))
        if xr < W - 1 and labels[y, xr + 1] < 0:
            rs.append((y, xr))
    m = np.zeros((H, W), bool)
    (al, bl), (ar, br) = (np.polyfit([v[0] for v in e], [v[1] for v in e], 1) for e in (ls, rs))
    yy, xx = np.mgrid[0:H, 0:W]
    m = (xx >= al * yy + bl - 2) & (xx <= ar * yy + br + 2)
    m[y1:] = True          # fitted on the clean lower rows, applied (extrapolated) above them
    return m


def stage_completions() -> None:
    key = np.asarray(Image.open(KEY).convert("RGBA"))
    labels = np.asarray(Image.open(PKG / "prep" / "partition_labels.png")).astype(np.int32) - 1
    ids = [l["id"] for l in LAYERS]
    (PKG / "completions").mkdir(exist_ok=True)
    peel = arm_peel(labels, ids)
    for lid, p in COMPLETIONS.items():
        own = labels == ids.index(lid)
        rgba = np.zeros_like(key)
        rgba[own] = key[own]
        if lid in peel:
            m, col = peel[lid]
            rgba[m, :3] = col[m]
            rgba[m, 3] = 255
            own = own | m
        under = np.isin(labels, [ids.index(u) for u in p["under"]])
        region = ndimage.binary_dilation(own, iterations=p["grow"]) & under & ~own
        if p.get("aline_hull"):
            region &= aline_hull(labels == ids.index(lid), labels, p["aline_hull"])
        src = ndimage.binary_erosion(own, iterations=3) if p.get("interior_src") else own
        if p.get("src_max_lum"):
            # skirt: fill from the navy cloth only - white cuff rims SAM left on the skirt and
            # anti-aliased edges otherwise paint a pale ghost of the hand behind it
            lum_k = key[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            src = src & (lum_k < p["src_max_lum"]) & (key[..., 3] == 255)
        if not src.any():
            src = own
        if p.get("skin_only"):
            # blink stretches the skin above the lash: what the bangs hide there must be skin,
            # not the nearest lash / sclera / hair pixels (they smeared grey-black bands)
            c = key[..., :3].astype(np.int16)
            lum = c @ np.array([299, 587, 114]) / 1000
            src = own & (lum > 190) & (c[..., 0] > c[..., 2] + 8)
        if p.get("image"):
            # a registered whole-part redraw: its pixels only, nothing procedural
            img = np.asarray(Image.open(PKG / p["image"]).convert("RGBA"))
            use = region & (img[..., 3] > 127)
            rgba[use, :3] = img[use, :3]
            rgba[use, 3] = 255
            region = use
        else:
            rgba = _diffuse_fill(rgba, src, region)
        Image.fromarray(rgba, "RGBA").save(PKG / "completions" / f"{lid}.png")
        print(f"[OK] completion {lid}: +{int(region.sum())} px")


def stage_peel() -> None:
    """Key the green whole-tail redraw, scale it back to the canvas crop and register it to the
    tail's own visible pixels (integer shift search, IoU)."""
    raw = np.asarray(Image.open(PKG / TAIL_PEEL["raw"]).convert("RGB")).astype(np.int16)
    g = raw[..., 1] - np.maximum(raw[..., 0], raw[..., 2])
    alpha = np.clip((60 - g) * 255 / 40, 0, 255).astype(np.uint8)     # green dominance -> transparent
    rgb = raw.copy()
    spill = g > 0
    rgb[spill, 1] = np.maximum(raw[spill, 0], raw[spill, 2])          # despill edges
    x0, y0, x1, y1 = TAIL_PEEL["crop"]
    im = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha]), "RGBA")
    im = im.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    a = np.asarray(im)[..., 3] > 127
    lab, n = ndimage.label(a)
    if n > 1:
        sizes = ndimage.sum(a, lab, range(1, n + 1))
        a = lab == (1 + int(np.argmax(sizes)))
    labels = np.asarray(Image.open(PKG / "prep" / "partition_labels.png")).astype(np.int32) - 1
    own = (labels == [l["id"] for l in LAYERS].index("tail"))[y0:y1, x0:x1]
    best = (-1.0, 0, 0)
    for dy in range(-16, 17):
        for dx in range(-16, 17):
            sh = np.roll(np.roll(a, dy, 0), dx, 1)
            iou = (sh & own).sum() / max(1, own.sum())
            if iou > best[0]:
                best = (float(iou), dx, dy)
    iou, dx, dy = best
    arr = np.asarray(im).copy()
    arr[~a] = 0
    canvas = np.zeros((H, W, 4), np.uint8)
    canvas[y0:y1, x0:x1] = np.roll(np.roll(arr, dy, 0), dx, 1)
    Image.fromarray(canvas, "RGBA").save(PKG / TAIL_PEEL["out"])
    _write_json(PKG / "prep" / "peel" / "tail_redraw_canvas.json",
                {"raw": TAIL_PEEL["raw"], "crop": TAIL_PEEL["crop"], "shift_px": [dx, dy],
                 "own_coverage": round(iou, 4)})
    print(f"[OK] tail redraw registered: shift ({dx},{dy}), covers {iou:.1%} of the visible tail")


def stage_eyes() -> dict:
    """Split iris/lash out of head_base. Returns per-eye geometry for the spec."""
    full = PKG / "layers_full"
    head = np.asarray(Image.open(full / "head_base.png").convert("RGBA")).copy()
    lum = head[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    geo = {}
    for side, (x0, y0, x1, y1) in EYE_BOX.items():
        box = _box_mask((x0, y0, x1, y1))
        iris = _load_mask(f"sam_masks/eye_{side}") & box
        iris = ndimage.binary_fill_holes(ndimage.binary_closing(iris, iterations=2))
        iris = iris & box & (head[..., 3] > 0)
        ys, xs = np.where(iris)
        cx, cy = float(xs.mean()), float(ys.mean())
        rx, ry = (xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2
        # pupil layer = iris pixels
        pup = np.zeros_like(head)
        pup[iris] = head[iris]
        Image.fromarray(pup, "RGBA").save(full / f"pupil_{side}.png")
        # upper lash: dark pixels above the iris centre line + the outer-corner wing
        dark = box & (lum < 110) & (head[..., 3] == 255) & ~iris
        lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
        top_rows = dark & (np.arange(H)[:, None] < cy)
        keep = np.unique(lab[top_rows])
        lash = np.isin(lab, keep[keep > 0])
        lash = ndimage.binary_dilation(lash, iterations=1) & box & (lum < 170) & (head[..., 3] == 255) & ~iris
        lid = np.zeros_like(head)
        lid[lash] = head[lash]
        Image.fromarray(lid, "RGBA").save(full / f"eyelid_{side}.png")
        # sclera under the iris (gaze moves the iris, blink squashes it): nearest light eye-white
        sclera_src = box & ~iris & ~lash & (lum > 205) & (head[..., 3] == 255)
        hole = iris                     # not dilated: that painted white over the lid lines
        _, (iy, ix) = ndimage.distance_transform_edt(~sclera_src, return_indices=True)
        fill = hole & ~sclera_src
        head[fill, :3] = head[iy[fill], ix[fill], :3]
        head[fill, 3] = 255
        f = head[..., :3].astype(np.float32)
        for _ in range(4):
            blur = np.stack([ndimage.uniform_filter(f[..., c], 5) for c in range(3)], -1)
            f[fill] = blur[fill]
        head[..., :3] = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        geo[side] = {"center": [round(cx, 1), round(cy, 1)], "radius": [round(rx, 1), round(ry, 1)],
                     "iris_px": int(iris.sum()), "lash_px": int(lash.sum())}
        print(f"[OK] eye {side}: iris {int(iris.sum())} px centre ({cx:.1f},{cy:.1f}), lash {int(lash.sum())} px")
    Image.fromarray(head, "RGBA").save(full / "head_base.png")
    _write_json(PKG / "prep" / "eyes.json", geo)
    return geo


def trim_layers(ids: list[str], margin: int = 2) -> dict:
    (PKG / "layers").mkdir(exist_ok=True)
    offs = {}
    for lid in ids:
        im = Image.open(PKG / "layers_full" / f"{lid}.png").convert("RGBA")
        ys, xs = np.where(np.asarray(im)[..., 3] > 0)
        x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
        x1, y1 = min(W, xs.max() + 1 + margin), min(H, ys.max() + 1 + margin)
        im.crop((x0, y0, x1, y1)).save(PKG / "layers" / f"{lid}.png", optimize=True)
        offs[lid] = [int(x0), int(y0)]
    return offs


def stage_spec() -> None:
    geo = json.loads((PKG / "prep" / "eyes.json").read_text(encoding="utf-8"))
    adult = json.loads((ROOT / "assets" / "rig_adult" / "spec.json").read_text(encoding="utf-8"))
    bones = []
    for name, parent, (x, y), clamp, chain in BONES_PX:
        if name.startswith("pupil_"):
            x, y = geo[name[-1]]["center"]
        bones.append({"bone_name": name, "parent": parent, "joint_pos": [round(x / W, 4), round(y / H, 4)],
                      "angle_clamp": list(clamp), "is_chain": chain})
    z_of = {l["id"]: l["z"] for l in LAYERS} | EYE_LAYERS_Z
    order = sorted(z_of, key=z_of.get)
    offs = trim_layers(order)
    layers = []
    for lid in order:
        e = {"id": lid, **SPEC_LAYERS[lid], "z_order": z_of[lid], "trim_offset_px": offs[lid]}
        if lid in ("head_base",):
            e["blink_zones"] = [BLINK_ZONES["l"], BLINK_ZONES["r"]]
            e["blink_blend_px"] = BLINK_BLEND_PX
        if lid.startswith(("pupil_", "eyelid_")):
            e["blink_zones"] = [BLINK_ZONES[lid[-1]]]
        if lid.startswith("pupil_"):
            g = geo[lid[-1]]
            # the iris is sampled through its own rest outline: it never draws over lids or skin
            e["gaze_ellipse"] = [g["center"][0], g["center"][1], g["radius"][0], g["radius"][1]]
        if lid in MULTI or lid in ("tail", "arm_l", "arm_r", "leg_l", "leg_r"):
            e["min_component_px"] = 40
        layers.append(e)
    fm = json.loads(json.dumps(adult["face_mechanics"]))
    fm["look_at"]["axis_limits_px"] = [4, 3]      # FINAL irises are ~35x23 px (ADULT 6x5 for ~45 px)
    fm["look_at"]["max_displacement_radius_px"] = 4
    fm["look_at"]["eyes"] = [{"bone": f"pupil_{s}", "rest_center": [round(geo[s]["center"][0] / W, 4),
                                                                    round(geo[s]["center"][1] / H, 4)]}
                             for s in ("l", "r")]
    spec = {
        "source_facing": 1,
        "texture_mipmaps": True,
        "ground_anchor_y_px": GROUND_Y,
        "rest_pose_angles": {},
        "skeleton": {"bones": bones, "source_reference": {"image_size_px": [W, H],
                                                          "key_art": "references/front_key.png"}},
        "face_mechanics": fm,
        "physics_presets": adult["physics_presets"],
        "layers": layers,
    }
    _write_json(PKG / "spec.json", spec)
    print(f"[OK] spec.json: {len(bones)} bones, {len(layers)} layers")


def stage_mesh() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from mesh_generator import generate_all_meshes
    generate_all_meshes(str(PKG / "spec.json"), str(PKG / "layers"), str(PKG / "mesh" / "mesh_data.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["masks", "labels", "peel", "completions", "layers", "eyes", "spec", "mesh", "all"])
    a = ap.parse_args()
    stages = ["masks", "labels", "peel", "completions", "layers", "eyes", "spec", "mesh"] if a.stage == "all" else [a.stage]
    for st in stages:
        {"masks": stage_masks, "labels": lambda: _run_layers(True), "peel": stage_peel,
         "completions": stage_completions,
         "layers": lambda: _run_layers(False), "eyes": stage_eyes, "spec": stage_spec,
         "mesh": stage_mesh}[st]()


if __name__ == "__main__":
    main()
