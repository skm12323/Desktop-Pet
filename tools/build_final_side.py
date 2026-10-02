"""FINAL F4: side-view skinned rig from the approved F3 key art (assets/rig_final_walk_v1).

Same recipe as the ADULT side rig (build_side_layers / side_completions / build_side_spec), with
FINAL's lessons from F2 built in: material-classified peels behind the near arm, whole redraws for
the far arm and the tail (green bg, local Qwen), interior-only diffuse fills, and the long skirt on
the four waist-hinged panel bones agreed after the prototype (docs/FINAL侧身原画F3与长裙原型-2026-10-02.md,
variant B; mesh weights mode "panels").

Stages (one per call, or --stage all):
  masks        SAM masks (prep/sam_masks = proto pass, prep/sam_masks_f4 = F4 pass) + geometric
               regions -> prep/derived/*.png, prep/mask_sources.json, prep/layer_plan.json
  labels       build_side_layers.py --plan ... --labels-only
  peel         register the Qwen redraws / edits (prep/peel/*) on the canvas
  completions  hidden-pixel fills per layer -> completions/
  layers       build_side_layers.py --plan ... -> layers_full/
  rig          whole far-arm redraw, hinged arm split, eyelids, trim, spec.json, mesh, manifest

Naming as on the ADULT side rig: _l = her right = near side (facing right), _r = far side.

Usage: D:\\anaconda3\\python.exe -X utf8 tools/build_final_side.py --stage all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "assets" / "rig_final_walk_v1"
KEY = PKG / "references" / "side_key.png"
W, H = 1024, 1824
GROUND_Y = 1761

HEM_Y = 1586           # lowest petticoat ruffle above the ankles
WAIST_Y = 650
HEAD_BOX = (300, 40, 720, 362)           # hair inside moves with the head (crown, bangs)
FAR_HAIR = {"x_min": 598, "y_min": 362}  # far-side hair hanging beside the far shoulder
TORSO_BOX = (410, 330, 612, 660)       # right edge stops before the far upper arm
SKIRT_BOX = (110, 600, 900, 1640)
NEAR_CUFF_BOX = (285, 820, 395, 885)
FAR_CUFF_BOX = (612, 820, 700, 885)
FRILL_BOX = (370, 380, 470, 560)
LEG_BOX = (400, 1560, 700, 1790)
LEG_SPLIT_X = 522

LAYERS = [
    {"id": "tail", "z": 0}, {"id": "hair_back", "z": 5}, {"id": "ear_fin_r", "z": 8},
    {"id": "hair_side_r", "z": 9}, {"id": "arm_r", "z": 10}, {"id": "leg_r", "z": 15},
    {"id": "leg_l", "z": 18}, {"id": "skirt", "z": 20}, {"id": "torso", "z": 25},
    {"id": "apron", "z": 30}, {"id": "arm_l", "z": 40}, {"id": "shoulder_frill_l", "z": 45},
    {"id": "head_base", "z": 50}, {"id": "bangs", "z": 53}, {"id": "ear_fin_l", "z": 55}, {"id": "ahoge", "z": 60},
    {"id": "headdress", "z": 65},
]
PRIORITY = ["ahoge", "ear_fin_r", "ear_fin_l", "headdress", "bangs", "head_base", "shoulder_frill_l", "arm_l",
            "arm_r", "apron", "leg_l", "leg_r", "tail", "torso", "hair_side_r", "hair_back", "skirt"]
# tail: the fluke shows between the near hand's fingers as a separate island; merged into the hand it
# stuck to the hand as a blue block
MULTI = ["hair_back", "hair_side_r", "skirt", "torso", "apron", "headdress", "head_base", "bangs", "tail"]
S1, S2 = "sam_masks", "sam_masks_f4"
SOURCES = {
    "ahoge": [f"{S2}/t_ahoge"],
    "ear_fin_r": [f"{S2}/fin_r"],
    "ear_fin_l": [f"{S2}/fin_l"],
    "headdress": [f"{S2}/t_headdress"],
    # hair over the face is its own rigid layer above the eyelids: inside head_base the blink
    # squash bent the strands that cross the eyes
    "bangs": ["derived/head_hair"],
    "head_base": [f"{S2}/t_face", f"{S2}/eye_l", f"{S2}/eye_r"],
    "shoulder_frill_l": ["derived/frill_l"],
    "arm_l": ["derived/arm_l"],
    "arm_r": ["derived/far_arm", f"{S1}/t_hand", "derived/cuff_r"],
    "apron": [f"{S1}/t_apron"],
    "leg_l": ["derived/leg_l"],
    "leg_r": ["derived/leg_r"],
    "tail": [f"{S1}/tail", "derived/tail_rim"],
    "torso": [f"{S2}/t_bodice", f"{S2}/t_bow", "derived/torso_box"],
    "hair_side_r": ["derived/hair_side_r"],
    "hair_back": ["derived/hair_back"],
    "skirt": [f"{S1}/t_skirt", f"{S1}/t_ruffles", "derived/skirt_box"],
}

# Qwen redraws / edits (prep/peel); crop = canvas box of the input crop (output scaled back to it)
PEELS = {
    "far_arm": {"raw": "prep/peel/far_arm_redraw_s1.png", "crop": (520, 400, 800, 1040), "green": True,
                "layer": "arm_r"},
    # only the fluke of the whole-tail redraw is used (its body curls differently: IoU 0.58), registered
    # on the fluke alone; it supplies the right lobe hidden by the hair and the near hand
    "tail": {"raw": "prep/peel/tail_redraw_s1.png", "crop": (0, 840, 460, 1400), "green": True, "layer": "tail",
             "fit_box": (0, 840, 345, 1110)},
    "near_arm_bg": {"raw": "prep/peel/near_arm_removed_s1.png", "crop": (180, 384, 564, 1056), "green": False},
}
COMPLETIONS = {
    # the Qwen whole-tail redraw registers poorly (IoU 0.58: different curl) and left ghost outlines
    # and stray islands that flew off with the fluke - the base is extruded under the skirt instead
    "tail": {"grow": 24, "under": ["arm_l", "hair_back", "skirt"], "interior": True,
             "fluke_image": "prep/peel/tail_canvas.png",
             "extrude_right": {"box": (240, 1120, 420, 1360), "px": 110}},
    "hair_back": {"grow": 40, "under": ["torso", "arm_l", "arm_r", "skirt", "head_base", "ear_fin_l",
                                       "shoulder_frill_l", "apron"], "interior": True},
    "hair_side_r": {"grow": 30, "under": ["arm_r", "torso", "apron", "head_base"], "interior": True},
    "ear_fin_r": {"grow": 15, "under": ["head_base", "headdress"], "interior": True},
    "leg_l": {"legs": True}, "leg_r": {"legs": True},
    "skirt": {"grow": 50, "under": ["arm_l", "arm_r", "apron", "torso"], "interior": True,
              "max_lum": 110, "aline_rows": (1180, 1500)},
    "torso": {"grow": 40, "under": ["arm_l", "arm_r", "apron", "head_base", "hair_side_r",
                                    "shoulder_frill_l"], "interior": True},
    "apron": {"grow": 16, "under": ["arm_l", "arm_r"]},
    "head_base": {"grow": 45, "under": ["bangs", "ear_fin_l", "headdress", "ahoge"], "skin_only": True},
}

# name: (parent, [x, y], clamp, is_chain) - canvas px on side_key.png
BONES = {
    "root_hip": (None, [520, 830], [-6, 6], False),
    "spine": ("root_hip", [520, 690], [-5, 5], False),
    "chest": ("spine", [515, 520], [-6, 6], False),
    "neck": ("chest", [522, 400], [-6, 6], False),
    "head": ("neck", [525, 362], [-12, 12], False),
    "tail_01": ("root_hip", [330, 1235], [-8, 8], True),
    "tail_02": ("tail_01", [215, 1265], [-14, 14], True),
    "tail_03": ("tail_02", [105, 1165], [-20, 20], True),
    "tail_fluke": ("tail_03", [60, 1050], [-26, 26], True),
    "hair_back_l_01": ("head", [400, 360], [-8, 8], True),
    "hair_back_l_02": ("hair_back_l_01", [285, 600], [-12, 12], True),
    "hair_back_l_03": ("hair_back_l_02", [205, 820], [-16, 16], True),
    "hair_side_r_01": ("head", [600, 380], [-8, 8], True),
    "hair_side_r_02": ("hair_side_r_01", [690, 620], [-12, 12], True),
    "upper_leg_l": ("root_hip", [505, 870], [-50, 50], True),
    "lower_leg_l": ("upper_leg_l", [503, 1290], [-10, 85], True),     # ahead of the hip-ankle line
    "foot_l": ("lower_leg_l", [482, 1705], [-85, 85], False),
    "upper_leg_r": ("root_hip", [545, 865], [-50, 50], True),
    "lower_leg_r": ("upper_leg_r", [556, 1290], [-10, 85], True),
    "foot_r": ("lower_leg_r", [552, 1700], [-85, 85], False),
    "ear_fin_l": ("head", [405, 262], [-12, 12], False),
    "ear_fin_r": ("head", [592, 312], [-12, 12], False),
    "skirt_root": ("root_hip", [520, WAIST_Y], [-5, 5], False),
    # long-skirt panels (prototype variant B): hinge at the waist, x at the waist / at the hem
    "skirt_back": ("skirt_root", [400, WAIST_Y], [-12, 12], False),
    "skirt_mid_b": ("skirt_root", [480, WAIST_Y], [-12, 12], False),
    "skirt_mid_f": ("skirt_root", [560, WAIST_Y], [-12, 12], False),
    "skirt_front": ("skirt_root", [640, WAIST_Y], [-12, 12], False),
    "apron_root": ("spine", [585, 660], [-4, 4], False),
    "apron_tip": ("apron_root", [640, 1000], [-10, 10], False),
    "upper_arm_l": ("chest", [418, 482], [-30, 30], True),
    "forearm_l": ("upper_arm_l", [393, 662], [-45, 20], True),
    "hand_l": ("forearm_l", [330, 852], [-15, 15], False),
    "upper_arm_r": ("chest", [598, 490], [-30, 30], True),
    "forearm_r": ("upper_arm_r", [628, 664], [-45, 20], True),
    "hand_r": ("forearm_r", [652, 852], [-15, 15], False),
    "eyelid_l": ("head", [505, 258], [0, 0], False),
    "eyelid_r": ("head", [572, 256], [0, 0], False),
    "ahoge_01": ("head", [372, 98], [-15, 15], True),
    "ahoge_02": ("ahoge_01", [340, 62], [-25, 25], True),
}
PANELS_AT_HEM = {"skirt_back": 180, "skirt_mid_b": 440, "skirt_mid_f": 660, "skirt_front": 840}
HINGED = {
    "arm_l": [("arm_l_upper", "upper_arm_l", 2), ("arm_l_fore", "forearm_l", 1), ("arm_l_hand", "hand_l", 0)],
    "arm_r": [("arm_r_upper", "upper_arm_r", 2), ("arm_r_fore", "forearm_r", 1), ("arm_r_hand", "hand_r", 0)],
}
EYE_BOX = {"eyelid_l": (474, 244, 538, 266), "eyelid_r": (560, 248, 598, 266)}   # upper lash search
# [cx, top, bottom, r] measured on the art: top = lash top (a lower top stretched the thick lash
# rows above it into a dark blot), bottom = lower lid
BLINK = {"eyelid_l": [506, 246, 284, 30], "eyelid_r": [578, 251, 289, 12]}

SPEC_LAYERS = {
    # largest_component: stray tail-coloured specks near the hair tips got their own mesh pieces
    # and flew off with the fluke
    "tail": {"bind_bone": "tail_01", "influence_bones": ["tail_01", "tail_02", "tail_03", "tail_fluke"],
             "min_component_px": 150,
             # the whole fluke (both lobes, up to x 280) is one rigid shape - a chain-weighted right
             # lobe tore off as a block under the near hand
             "rigid_above": [{"bone": "tail_fluke", "y": 1110, "blend": 50, "x_max": 345}],
             "weights": {"mode": "chain", "blend_px": 40}, "grid_step": 20},
    "hair_back": {"bind_bone": "head", "influence_bones": ["head", "hair_back_l_01", "hair_back_l_02", "hair_back_l_03"],
                  "root_lock": {"bone": "head", "full_before_y": 380, "free_after_y": 520},
                  "weights": {"mode": "chain", "blend_px": [40, 50, 50]}, "min_component_px": 40},
    "ear_fin_r": {"bind_bone": "ear_fin_r", "influence_bones": ["ear_fin_r"]},
    "hair_side_r": {"bind_bone": "head", "influence_bones": ["head", "hair_side_r_01", "hair_side_r_02"],
                    "root_lock": {"bone": "head", "full_before_y": 400, "free_after_y": 500},
                    "weights": {"mode": "chain", "blend_px": [30, 40]}, "min_component_px": 50},
    "arm_r_hand": {"bind_bone": "hand_r", "influence_bones": ["hand_r"], "min_component_px": 150},
    "arm_r_fore": {"bind_bone": "forearm_r", "influence_bones": ["forearm_r"], "min_component_px": 150},
    "arm_r_upper": {"bind_bone": "upper_arm_r", "influence_bones": ["upper_arm_r"], "min_component_px": 150},
    "leg_r": {"bind_bone": "upper_leg_r", "influence_bones": ["upper_leg_r", "lower_leg_r", "foot_r"],
              "joint_curve": {"parent": "upper_leg_r", "child": "lower_leg_r", "tip": "foot_r",
                              "upper_px": 300, "lower_px": 90},
              "rigid_below": [{"bone": "foot_r", "y": 1688, "blend": 18}],
              "weights": {"mode": "chain", "blend_px": [22, 28],
                          "inner": {"lower_leg_r": {"side": 1, "blend_px": 110, "down_px": 60, "half_width_px": 80},
                                    "foot_r": {"side": -1, "blend_px": 50, "down_px": 24, "half_width_px": 36}}},
              "grid_step": 14, "min_component_px": 150},
    "leg_l": {"bind_bone": "upper_leg_l", "influence_bones": ["upper_leg_l", "lower_leg_l", "foot_l"],
              "joint_curve": {"parent": "upper_leg_l", "child": "lower_leg_l", "tip": "foot_l",
                              "upper_px": 300, "lower_px": 90},
              "rigid_below": [{"bone": "foot_l", "y": 1694, "blend": 18}],
              "weights": {"mode": "chain", "blend_px": [22, 28],
                          "inner": {"lower_leg_l": {"side": 1, "blend_px": 110, "down_px": 60, "half_width_px": 80},
                                    "foot_l": {"side": -1, "blend_px": 50, "down_px": 24, "half_width_px": 36}}},
              "grid_step": 14, "min_component_px": 150},
    "skirt": {"bind_bone": "skirt_root",
              "influence_bones": ["skirt_root", "skirt_back", "skirt_mid_b", "skirt_mid_f", "skirt_front"],
              # applied by _panel_weights() after mesh_generator (kept out of the shared mesher)
              "panel_weights": {"root": "skirt_root", "y_waist": WAIST_Y, "y_hem": HEM_Y,
                          "ramp_start_px": 40, "ramp_px": 200,
                          "panels": {n: [BONES[n][1][0], x] for n, x in PANELS_AT_HEM.items()}},
              "grid_step": 20, "min_component_px": 200},
    "torso": {"bind_bone": "chest", "influence_bones": ["spine", "chest", "neck"],
              "weights": {"mode": "chain", "blend_px": [40, 30]}, "min_component_px": 150},
    "apron": {"bind_bone": "apron_root", "influence_bones": ["apron_root", "apron_tip"],
              "root_lock": {"bone": "apron_root", "full_before_y": 700, "free_after_y": 860},
              "weights": {"mode": "chain", "blend_px": 80}, "min_component_px": 150},
    "arm_l_hand": {"bind_bone": "hand_l", "influence_bones": ["hand_l"], "min_component_px": 150},
    "arm_l_fore": {"bind_bone": "forearm_l", "influence_bones": ["forearm_l"], "min_component_px": 150},
    "arm_l_upper": {"bind_bone": "upper_arm_l", "influence_bones": ["upper_arm_l"], "min_component_px": 150},
    "shoulder_frill_l": {"bind_bone": "chest", "influence_bones": ["chest"]},
    "head_base": {"bind_bone": "head", "influence_bones": ["head"],
                  "blink_zones": [BLINK["eyelid_l"], BLINK["eyelid_r"]], "blink_blend_px": 6},   # the far eye sits 20 px from the cheek outline
    "eyelid_l": {"bind_bone": "eyelid_l", "influence_bones": ["head", "eyelid_l"], "blink_zones": [BLINK["eyelid_l"]]},
    "eyelid_r": {"bind_bone": "eyelid_r", "influence_bones": ["head", "eyelid_r"], "blink_zones": [BLINK["eyelid_r"]]},
    "ear_fin_l": {"bind_bone": "ear_fin_l", "influence_bones": ["ear_fin_l"]},
    "bangs": {"bind_bone": "head", "influence_bones": ["head"], "min_component_px": 40},
    "ahoge": {"bind_bone": "ahoge_01", "influence_bones": ["ahoge_01", "ahoge_02"],
              "weights": {"mode": "chain", "blend_px": 20}, "min_component_px": 30},
    "headdress": {"bind_bone": "head", "influence_bones": ["head"]},
}
Z_OVERRIDE = {}
GAIT_OVERRIDES = {"frequency_hz": 1.6,          # long skirt: small steps (stride = speed / cadence)
                  "skirt_follow_gain": 0.0}     # the panel bones replace the two hem bones (F6 drive)


# ------------------------------------------------------------------ helpers
def _box(b) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[b[1]:b[3], b[0]:b[2]] = True
    return m


def _mask(rel: str) -> np.ndarray:
    return np.asarray(Image.open(PKG / "prep" / f"{rel}.png").convert("L")) > 127


def _save_mask(rel: str, m: np.ndarray) -> None:
    p = PKG / "prep" / f"{rel}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m * 255).astype(np.uint8), "L").save(p)


def _json(path: Path, obj) -> None:
    path.write_bytes(json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def _key() -> np.ndarray:
    return np.asarray(Image.open(KEY).convert("RGBA"))


def _lum(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def _labels() -> tuple[np.ndarray, list[str]]:
    lab = np.asarray(Image.open(PKG / "prep" / "partition_labels.png")).astype(np.int32) - 1
    return lab, [l["id"] for l in LAYERS]


# ------------------------------------------------------------------ stages
def _far_arm_visible(key: np.ndarray) -> np.ndarray:
    """SAM's far-arm points grabbed the apron frill. The whole-arm redraw is placed on the art by
    colour agreement (shift search), and the arm's visible pixels are where the art matches it."""
    p = PEELS["far_arm"]
    x0, y0, x1, y1 = p["crop"]
    rgb, alpha = _key_green(np.asarray(Image.open(PKG / p["raw"]).convert("RGB")))
    im = np.asarray(Image.fromarray(np.dstack([rgb, alpha]), "RGBA").resize((x1 - x0, y1 - y0),
                                                                           Image.Resampling.LANCZOS)).astype(np.int16)
    a = im[..., 3] > 200
    k = key[y0:y1, x0:x1].astype(np.int16)
    best = (1e9, 0, 0)
    for dy in range(-30, 31, 2):
        for dx in range(-30, 31, 2):
            sa = np.roll(np.roll(a, dy, 0), dx, 1)
            si = np.roll(np.roll(im, dy, 0), dx, 1)
            m = sa & (k[..., 3] > 200)
            if m.sum() < 500:
                continue
            err = np.abs(si[m, :3] - k[m, :3]).mean()
            if err < best[0]:
                best = (float(err), dx, dy)
    _, dx, dy = best
    sa = np.roll(np.roll(a, dy, 0), dx, 1)
    si = np.roll(np.roll(im, dy, 0), dx, 1)
    # ownership follows the redraw's silhouette (the art's far arm strip must move with the arm, a
    # torso-owned strip stayed behind as a second sleeve): every opaque pixel inside it except the
    # apron in front and the hair-blue strands
    kk = k[..., :3]
    hair = (kk[..., 2] - kk[..., 0] > 45) & (kk[..., 2] > 140)
    apron = _mask(f"{S1}/t_apron")[y0:y1, x0:x1]
    close = sa & (k[..., 3] > 200) & ~hair & ~apron
    lab, n = ndimage.label(close, structure=np.ones((3, 3)))
    if n > 1:
        sizes = ndimage.sum(close, lab, range(1, n + 1))
        close = np.isin(lab, 1 + np.where(sizes >= 150)[0])
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = close
    print(f"[OK] far arm placed by colour: shift ({dx},{dy}), mean |dRGB| {best[0]:.1f}, visible {int(out.sum())} px")
    return out


def stage_masks() -> None:
    key = _key()
    op = key[..., 3] > 0
    lum = _lum(key)
    yy, xx = np.mgrid[0:H, 0:W]
    hair = _mask(f"{S1}/t_hair")
    head_hair = hair & _box(HEAD_BOX)
    side = hair & ~head_hair & (xx >= FAR_HAIR["x_min"]) & (yy >= FAR_HAIR["y_min"])
    lab, n = ndimage.label(side)
    if n:
        sizes = ndimage.sum(side, lab, range(1, n + 1))
        side = np.isin(lab, 1 + np.where(sizes >= 200)[0])
    # strands lying over the eyes (SAM files them under the face): hair-blue, not iris, touching the
    # bangs - otherwise the blink squash bends / wipes them
    c = key[..., :3].astype(np.int16)
    blue = (c[..., 2] - c[..., 0] > 12) & (_lum(key) > 100) & op      # thin light strands too
    iris = ndimage.binary_dilation(_mask(f"{S2}/eye_l") | _mask(f"{S2}/eye_r"), iterations=3)
    over_eye = blue & ~iris & _box((440, 200, 640, 310))
    lab, _ = ndimage.label(over_eye | head_hair, structure=np.ones((3, 3)))
    keep = np.unique(lab[head_hair])
    head_hair = head_hair | (over_eye & np.isin(lab, keep[keep > 0]))
    _save_mask("derived/head_hair", head_hair)
    _save_mask("derived/hair_side_r", side)
    _save_mask("derived/hair_back", hair & ~head_hair & ~side)
    _save_mask("derived/frill_l", _mask(f"{S2}/t_frill") & _box(FRILL_BOX))
    _save_mask("derived/cuff_l", _box(NEAR_CUFF_BOX) & (lum > 195) & op)   # (used by derived/arm_l below)
    # far cuff frill: white next to the far arm / hand only (the apron frill runs right beside it)
    far = _far_arm_visible(key) | _mask(f"{S1}/t_hand")
    _save_mask("derived/cuff_r", _box(FAR_CUFF_BOX) & (lum > 195) & op & ~_mask(f"{S1}/t_apron")
               & ndimage.binary_dilation(far, iterations=10))
    _save_mask("derived/far_arm", _far_arm_visible(key))
    _save_mask("derived/torso_box", _box(TORSO_BOX))
    # the catch-all skirt box must not swallow the tail's outline or the fluke lobe hidden under the
    # hair: those pixels stayed on the skirt as a wire outline / a navy block once the tail moved
    tail_zone = ndimage.binary_dilation(_mask(f"{S1}/tail"), iterations=8) | _box((0, 840, 330, 1120))
    # ...nor the dark gaps of the back hair left of the skirt (they floated outside the skirt edge):
    # the box stops at the skirt's back silhouette, a line fitted to SAM's skirt edge
    sk = _mask(f"{S1}/t_skirt")
    rows = [y for y in range(1150, 1500) if sk[y].any()]
    a_, b_ = np.polyfit(rows, [np.where(sk[y])[0].min() for y in rows], 1)
    back_edge = xx >= a_ * yy + b_ - 6
    _save_mask("derived/skirt_box", _box(SKIRT_BOX) & (yy >= WAIST_Y - 10) & ~tail_zone & back_edge)
    _save_mask("derived/tail_rim", ndimage.binary_dilation(_mask(f"{S1}/tail"), iterations=8)
               & ~_mask(f"{S1}/t_skirt") & ~hair & op)
    # near hand without the navy skirt seen between the fingers (it stuck to the hand as a block)
    c = key[..., :3].astype(np.int16)
    navy = (c[..., 2] - c[..., 0] > 25) & (lum < 120)
    arm_l = _mask(f"{S2}/near_arm") | _mask(f"{S1}/t_sleeve") | _mask(f"{S2}/near_hand") | _mask("derived/cuff_l")
    gap = navy & _box((230, 880, 410, 1070))
    _save_mask("derived/arm_l", arm_l & ~gap)
    # legs. The near shoe overlaps the far one: the near shoe and its outline (SAM mask + 4 px) are
    # the near leg's, whatever is nearer (nearest-seed handed the near toe's outline to the far
    # foot: a stray toe contour under the far shoe). White below the hem is stocking only inside
    # the ankle columns measured on clean stocking rows - elsewhere it is petticoat ruffle.
    below = op & (yy >= HEM_Y - 2) & _box(LEG_BOX)
    # near shoe + its outline + the sock showing inside its opening
    near_shoe = ndimage.binary_fill_holes(ndimage.binary_dilation(_mask(f"{S1}/shoe_near"), iterations=6))
    near_shoe = ndimage.binary_erosion(near_shoe, iterations=2) & below
    far_shoe = _mask(f"{S1}/shoe_far") & ~near_shoe
    white = below & (lum > 170) & (yy >= 1604)      # rows 1586-1603: petticoat ruffle, not stocking
    cols = {}
    for side, xr in (("l", (LEG_BOX[0], LEG_SPLIT_X)), ("r", (LEG_SPLIT_X, LEG_BOX[2]))):
        band = white[1625:1650, xr[0]:xr[1]].any(0)
        xs = np.where(band)[0] + xr[0]
        cols[side] = (xs.min() - 3, xs.max() + 4)
    stock_l = white & (xx >= cols["l"][0]) & (xx < cols["l"][1]) & ~far_shoe
    stock_r = white & (xx >= max(cols["r"][0], cols["l"][1])) & (xx < cols["r"][1]) & ~near_shoe
    rest = below & ~white & ~near_shoe & ~far_shoe       # outlines between: nearest of the two
    seed_l, seed_r = near_shoe | stock_l, far_shoe | stock_r
    dl, dr = ndimage.distance_transform_edt(~seed_l), ndimage.distance_transform_edt(~seed_r)
    leg_l = seed_l | (rest & (dl <= dr) & (dl < 6))
    leg_r = (seed_r | (rest & (dr < dl) & (dr < 6))) & ~leg_l
    _save_mask("derived/leg_l", leg_l)
    _save_mask("derived/leg_r", leg_r)
    print(f"[OK] stocking columns {cols}")
    _json(PKG / "prep" / "mask_sources.json",
          {"_note": "FINAL side: SAM 3.1 masks (sam_masks = prototype pass, sam_masks_f4 = F4 pass) + "
                    "tools/build_final_side.py derived regions", **SOURCES})
    _json(PKG / "prep" / "layer_plan.json",
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


def _key_green(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = raw.astype(np.int16)
    g = r[..., 1] - np.maximum(r[..., 0], r[..., 2])
    alpha = np.clip((60 - g) * 255 / 40, 0, 255).astype(np.uint8)
    rgb = r.copy()
    spill = g > 0
    rgb[spill, 1] = np.maximum(r[spill, 0], r[spill, 2])
    return np.clip(rgb, 0, 255).astype(np.uint8), alpha


def _register(rgba: np.ndarray, own: np.ndarray, search: int = 20) -> tuple[np.ndarray, tuple, float]:
    a = rgba[..., 3] > 127
    best = (-1.0, 0, 0)
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            sh = np.roll(np.roll(a, dy, 0), dx, 1)
            iou = (sh & own).sum() / max(1, (sh | own).sum())
            if iou > best[0]:
                best = (float(iou), dx, dy)
    iou, dx, dy = best
    for ddy in (-1, 0, 1):          # 1 px refinement
        for ddx in (-1, 0, 1):
            sh = np.roll(np.roll(a, dy + ddy, 0), dx + ddx, 1)
            v = (sh & own).sum() / max(1, (sh | own).sum())
            if v > iou:
                iou, best = v, (v, dx + ddx, dy + ddy)
    _, dx, dy = best
    return np.roll(np.roll(rgba, dy, 0), dx, 1), (dx, dy), float(iou)


def stage_peel() -> None:
    lab, ids = _labels()
    rep = {}
    for name, p in PEELS.items():
        path = PKG / p["raw"]
        if not path.exists():
            print(f"[skip] {name}: {p['raw']} missing")
            continue
        x0, y0, x1, y1 = p["crop"]
        raw = np.asarray(Image.open(path).convert("RGB"))
        if p["green"]:
            rgb, alpha = _key_green(raw)
            im = Image.fromarray(np.dstack([rgb, alpha]), "RGBA").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
            arr = np.asarray(im).copy()
            a = arr[..., 3] > 127
            cc, n = ndimage.label(a)
            if n > 1:
                sizes = ndimage.sum(a, cc, range(1, n + 1))
                arr[cc != 1 + int(np.argmax(sizes))] = 0
            canvas = np.zeros((H, W, 4), np.uint8)
            canvas[y0:y1, x0:x1] = arr
            own = lab == ids.index(p["layer"])
            if p.get("fit_box"):
                fb = _box(p["fit_box"])
                own = own & fb
                canvas[~fb] = 0
            canvas, shift, iou = _register(canvas, own)
            Image.fromarray(canvas, "RGBA").save(PKG / "prep" / "peel" / f"{p['layer']}_canvas.png")
            rep[name] = {"raw": p["raw"], "shift_px": list(shift), "iou_vs_visible": round(iou, 3)}
            print(f"[OK] {name}: shift {shift}, IoU vs visible {iou:.2f}")
        else:
            im = Image.open(path).convert("RGB").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[y0:y1, x0:x1] = np.asarray(im)
            Image.fromarray(canvas).save(PKG / "prep" / "peel" / f"{name}_canvas.png")
            rep[name] = {"raw": p["raw"], "crop": list(p["crop"])}
            print(f"[OK] {name}: placed")
    _json(PKG / "prep" / "peel" / "peels.json", rep)


def _diffuse(rgba: np.ndarray, src: np.ndarray, region: np.ndarray, iters: int = 3) -> np.ndarray:
    out = rgba.copy()
    if not region.any() or not src.any():
        return out
    _, (iy, ix) = ndimage.distance_transform_edt(~src, return_indices=True)
    out[region, :3] = rgba[iy[region], ix[region], :3]
    out[region, 3] = 255
    f = out[..., :3].astype(np.float32)
    for _ in range(iters):
        blur = np.stack([ndimage.uniform_filter(f[..., c], 9) for c in range(3)], -1)
        f[region] = blur[region]
    out[..., :3] = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    return out


def _near_arm_peel(lab: np.ndarray, ids: list[str]) -> dict:
    """Pixels the near arm hides at rest, classified by material in the arm-removed edit (F2 lesson:
    the edit reshapes the cloth, so nearest-layer assignment paints the wrong material)."""
    path = PKG / "prep" / "peel" / "near_arm_bg_canvas.png"
    if not path.exists():
        return {}
    c = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    lum = c @ np.array([299, 587, 114]) / 1000
    sat = c.max(-1) - c.min(-1)
    occ = np.isin(lab, [ids.index("arm_l"), ids.index("shoulder_frill_l")])
    x0, y0, x1, y1 = PEELS["near_arm_bg"]["crop"]
    occ &= _box((x0, y0, x1, y1))
    hair = (c[..., 2] - c[..., 0] > 45) & (lum > 95)      # the navy dress is blue too, but dark
    navy = (lum < 95) & ~hair
    white = (lum > 195) & (sat < 30)
    yy = np.mgrid[0:H, 0:W][0]

    def near(layer, px):
        return ndimage.binary_dilation(lab == ids.index(layer), iterations=px)
    # every hidden pixel takes the edit's colour (a blurred diffuse fill there read as a ghost of the
    # arm once it swung); material only decides the layer: hair -> back hair, white below the
    # waist band next to the apron -> apron, the rest (bodice, back bow, dress) -> torso / skirt
    cloth = ~hair & (c.sum(-1) < 720)                         # not the edit's white background
    apron_w = white & near("apron", 14) & (yy >= WAIST_Y)
    pick = {"hair_back": hair,
            "apron": apron_w,
            "torso": cloth & ~apron_w & (yy < WAIST_Y + 40),
            "skirt": cloth & ~apron_w & (yy >= WAIST_Y + 40)}
    out = {}
    for t, cls in pick.items():
        m = occ & cls
        out[t] = (m, c.astype(np.uint8))
        print(f"[OK] near-arm peel -> {t}: {int(m.sum())} px")
    return out


def _leg_columns(rgba: np.ndarray, side: str, hidden: np.ndarray) -> np.ndarray:
    """Hidden stocking column hip -> ankle (only ever seen if the hem lifts)."""
    hip, knee, ank = BONES[f"upper_leg_{side}"][1], BONES[f"lower_leg_{side}"][1], BONES[f"foot_{side}"][1]
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for a, b, w0, w1 in ((hip, knee, 76, 56), (knee, (ank[0], ank[1] - 6), 56, 38)):
        v = np.array(b, float) - np.array(a, float)
        nrm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        poly = [tuple(np.array(a) + nrm * w0 / 2), tuple(np.array(a) - nrm * w0 / 2),
                tuple(np.array(b) - nrm * w1 / 2), tuple(np.array(b) + nrm * w1 / 2)]
        d.polygon(poly, fill=(236, 236, 242, 255))      # no outline: it peeked out below the hem
    d.ellipse([knee[0] - 28, knee[1] - 28, knee[0] + 28, knee[1] + 28], fill=(236, 236, 242, 255))
    col = np.asarray(im)
    use = hidden & (col[..., 3] > 0) & (rgba[..., 3] == 0)
    out = rgba.copy()
    out[use] = col[use]
    return out


def _extrude_right(rgba: np.ndarray, own: np.ndarray, hidden: np.ndarray, cfg: dict):
    """Continue a part rightwards under the layer covering it: each row repeats the part's last
    visible pixels (outline included) for cfg px - the tail base under the skirt."""
    out, own2 = rgba.copy(), own.copy()
    x0, y0, x1, y1 = cfg["box"]
    for y in range(y0, y1):
        xs = np.where(own[y, x0:x1])[0]
        if not len(xs):
            continue
        xe = x0 + int(xs.max())
        if not hidden[y, min(xe + 1, W - 1)]:
            continue
        src = out[y, max(xe - 3, 0):xe + 1].copy()
        for k in range(1, cfg["px"]):
            x = xe + k
            if x >= W or not hidden[y, x]:
                break
            out[y, x] = src[min(len(src) - 1, k % len(src))] if k > 3 else src[-1]
            out[y, x, 3] = 255
            own2[y, x] = True
    return out, own2


def _aline_hull(own: np.ndarray, lab: np.ndarray, z: dict, ids: list[str], rows: tuple) -> np.ndarray:
    """Inside of the A-line skirt: straight back/front silhouette lines fitted on rows where the
    skirt edge meets the background or a lower layer (hands hang over the background: fills under
    them must not widen the skirt into a hand-shaped blot - F2 lesson)."""
    lower = {i for i, n in enumerate(ids) if z[n] < z[ids[lab[own][0]]]} if own.any() else set()
    ls, rs = [], []
    for y in range(*rows):
        xs = np.where(own[y])[0]
        if len(xs) < 2:
            continue
        xl, xr = xs.min(), xs.max()
        if xl > 0 and (lab[y, xl - 1] < 0 or lab[y, xl - 1] in lower):
            ls.append((y, xl))
        if xr < W - 1 and (lab[y, xr + 1] < 0 or lab[y, xr + 1] in lower):
            rs.append((y, xr))
    (al, bl), (ar, br) = (np.polyfit([v[0] for v in e], [v[1] for v in e], 1) for e in (ls, rs))
    yy, xx = np.mgrid[0:H, 0:W]
    m = (xx >= al * yy + bl - 3) & (xx <= ar * yy + br + 3)
    m[rows[1]:] = True
    return m


def stage_completions() -> None:
    key = _key()
    lab, ids = _labels()
    z = {l["id"]: l["z"] for l in LAYERS}
    peel = _near_arm_peel(lab, ids)
    (PKG / "completions").mkdir(exist_ok=True)
    lum_k = _lum(key)
    for lid, p in COMPLETIONS.items():
        own = lab == ids.index(lid)
        rgba = np.zeros_like(key)
        rgba[own] = key[own]
        higher = np.isin(lab, [ids.index(o) for o in ids if z[o] > z[lid]])
        if p.get("legs"):
            rgba = _leg_columns(rgba, lid[-1], higher)
            Image.fromarray(rgba, "RGBA").save(PKG / "completions" / f"{lid}.png")
            print(f"[OK] completion {lid}: stocking column")
            continue
        if lid in peel:
            m, col = peel[lid]
            m = m & ~own
            rgba[m, :3] = col[m]
            rgba[m, 3] = 255
            own = own | m
        under = np.isin(lab, [ids.index(u) for u in p["under"]])
        region = ndimage.binary_dilation(own, iterations=p["grow"]) & under & ~own
        if p.get("aline_rows"):
            region &= _aline_hull(lab == ids.index(lid), lab, z, ids, p["aline_rows"])
        if p.get("fluke_image") and (PKG / p["fluke_image"]).exists():
            img = np.asarray(Image.open(PKG / p["fluke_image"]).convert("RGBA"))
            use = (img[..., 3] > 127) & higher & ~own
            rgba[use, :3] = img[use, :3]
            rgba[use, 3] = 255
            own = own | use
            # inside the fluke box the redraw alone decides the hidden shape (no procedural fill:
            # it grew a dark angular block where the right lobe hides under the hair)
            region &= ~_box(PEELS["tail"]["fit_box"])
        if p.get("extrude_right"):
            rgba, own = _extrude_right(rgba, own, higher, p["extrude_right"])
        if p.get("image"):
            img_path = PKG / p["image"]
            if img_path.exists():
                img = np.asarray(Image.open(img_path).convert("RGBA"))
                use = (img[..., 3] > 127) & higher & ~own
                rgba[use, :3] = img[use, :3]
                rgba[use, 3] = 255
                region &= ~use
                own = own | use
        src = ndimage.binary_erosion(own, iterations=3) if p.get("interior") else own
        if p.get("skin_only"):
            # the blink stretches the skin around the eye: what the bangs hide must be skin
            c = key[..., :3].astype(np.int16)
            src = src & (lum_k > 190) & (c[..., 0] > c[..., 2] + 8)
        if p.get("max_lum"):
            src &= (lum_k < p["max_lum"]) | ~(lab == ids.index(lid))
        if not src.any():
            src = own
        if not p.get("image"):
            rgba = _diffuse(rgba, src, region)
        Image.fromarray(rgba, "RGBA").save(PKG / "completions" / f"{lid}.png")
        print(f"[OK] completion {lid}: +{int((rgba[..., 3] > 0).sum() - (lab == ids.index(lid)).sum())} px")


# ------------------------------------------------------------------ rig
def _far_arm_whole() -> None:
    """Far arm layer = the whole Qwen redraw (visible sliver included: splicing the art's sliver to a
    separately drawn arm doubled the cuff on ADULT), clipped where it would show at rest."""
    path = PKG / "prep" / "peel" / "arm_r_canvas.png"
    if not path.exists():
        return
    lab, ids = _labels()
    z = {l["id"]: l["z"] for l in LAYERS}
    own = lab == ids.index("arm_r")
    higher = np.isin(lab, [ids.index(o) for o in ids if z[o] > z["arm_r"]])
    red = np.asarray(Image.open(path).convert("RGBA"))
    cur = np.asarray(Image.open(PKG / "layers_full" / "arm_r.png").convert("RGBA")).copy()
    key = _key()
    use = (red[..., 3] > 127) & (higher | own)
    cur[use] = red[use]                   # own pixels outside the redraw (fingertips) stay from the art
    vis = use & own                       # rest pose: the art's own colours where it is visible
    cur[vis] = key[vis]
    near_own = ndimage.binary_dilation(own, iterations=8)
    cur[higher & ~own & ~use & ~near_own] = 0
    Image.fromarray(cur, "RGBA").save(PKG / "layers_full" / "arm_r.png")


def _split_hinged() -> dict:
    sys.path.insert(0, str(ROOT / "tools"))
    from split_hinged_limb import split
    zoff = {}
    for limb, chain in HINGED.items():
        rgba = np.asarray(Image.open(PKG / "layers_full" / f"{limb}.png").convert("RGBA"))
        pieces = split(rgba, [np.array(BONES[b][1], float) for _, b, _ in chain])
        for (pid, _, dz), arr in zip(chain, pieces):
            Image.fromarray(arr, "RGBA").save(PKG / "layers_full" / f"{pid}.png")
            zoff[pid] = (limb, dz)
    return zoff


def _eyelids() -> None:
    head = np.asarray(Image.open(PKG / "layers_full" / "head_base.png").convert("RGBA"))
    lum = _lum(head)
    for lid, (x0, y0, x1, y1) in EYE_BOX.items():
        m = np.zeros(lum.shape, bool)
        m[y0:y1, x0:x1] = (lum[y0:y1, x0:x1] < 110) & (head[y0:y1, x0:x1, 3] == 255)
        # only the lash line hugging the eye: bang strands cross the eye box and closed into a
        # dark blot over the eye
        eye = _mask(f"{S2}/eye_{lid[-1]}")
        m &= ndimage.binary_dilation(eye, iterations=7) & (np.mgrid[0:H, 0:W][0] <= np.where(eye.any(1))[0].mean())
        m &= ~eye                     # the iris top is not lash (it made a thick dark blot when closed)
        out = np.zeros_like(head)
        out[m] = head[m]
        Image.fromarray(out, "RGBA").save(PKG / "layers_full" / f"{lid}.png")
        print(f"[OK] {lid}: {int(m.sum())} lash px")


def _trim(ids: list[str], margin: int = 2) -> dict:
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


def _contact_points() -> dict:
    out = {}
    for side, shoe in (("l", "shoe_near"), ("r", "shoe_far")):
        m = _mask(f"{S1}/{shoe}")
        rows = np.where(m.any(1))[0]
        sole = int(rows.max())
        band = m[sole - 6: sole + 1]
        xs = np.where(band.any(0))[0]
        ank = BONES[f"foot_{side}"][1]
        for k, x in (("heel", xs.min()), ("sole", 0.5 * (xs.min() + xs.max())), ("forefoot", xs.max())):
            out[f"{k}_{side}"] = [round(float(x) - ank[0], 1), float(sole - ank[1])]
    return out


def _panel_weights(cfg: dict, pts: np.ndarray) -> tuple[list, list]:
    """Long skirt (prototype variant B): waist band on the root; below it each vertex blends the
    two neighbouring panels, whose lines run from their waist x to their hem x."""
    names = list(cfg["panels"])
    xw = np.array([cfg["panels"][n][0] for n in names], float)
    xh = np.array([cfg["panels"][n][1] for n in names], float)
    out_b, out_w = [], []
    for x, y in pts:
        t = float(np.clip((y - cfg["y_waist"]) / (cfg["y_hem"] - cfg["y_waist"]), 0, 1))
        xs = xw + (xh - xw) * t
        hem = float(np.clip((y - (cfg["y_waist"] + cfg["ramp_start_px"])) / cfg["ramp_px"], 0, 1))
        hem = hem * hem * (3 - 2 * hem)
        k = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
        u = float(np.clip((x - xs[k - 1]) / max(xs[k] - xs[k - 1], 1.0), 0, 1))
        u = u * u * (3 - 2 * u)
        w = {cfg["root"]: 1 - hem, names[k - 1]: hem * (1 - u)}
        w[names[k]] = w.get(names[k], 0) + hem * u
        w = {b: v for b, v in w.items() if v > 1e-4}
        tot = sum(w.values())
        out_b.append(list(w))
        out_w.append([round(v / tot, 6) for v in w.values()])
    return out_b, out_w


def stage_rig(fresh_layers: bool = True) -> None:
    if fresh_layers:            # _far_arm_whole / _split_hinged edit layers_full in place: start clean
        _run_layers(False)
    _far_arm_whole()
    zoff = _split_hinged()
    _eyelids()
    part = json.loads((PKG / "prep" / "partition.json").read_text(encoding="utf-8"))["layers"]
    z = {lid: s["z"] for lid, s in part.items()}
    z.update({"eyelid_l": 51, "eyelid_r": 52})
    z.update({pid: z[limb] + dz for pid, (limb, dz) in zoff.items()})
    z.update(Z_OVERRIDE)
    ids = [lid for lid in SPEC_LAYERS]
    offs = _trim(ids)
    bones = [{"bone_name": n, "parent": p, "joint_pos": [round(x / W, 5), round(y / H, 5)],
              "angle_clamp": c, "is_chain": ch} for n, (p, (x, y), c, ch) in BONES.items()]
    layers = []
    for lid, l in SPEC_LAYERS.items():
        e = {"id": lid, "description": f"FINAL side rig layer {lid} (pixel-exact cut of side_key.png)",
             **l, "z_order": z[lid], "per_component": True, "trim_offset_px": offs[lid]}
        layers.append(e)
    adult_side = json.loads((ROOT / "assets" / "rig_adult_walk_v1" / "spec.json").read_text(encoding="utf-8"))
    gait = dict(adult_side["gait"])
    gait.update(GAIT_OVERRIDES)
    springs = {k: v for k, v in adult_side["physics_presets"]["spring_damper"].items() if k in BONES}
    spec = {
        "source_facing": 1, "texture_mipmaps": True, "ground_anchor_y_px": GROUND_Y + 2,
        "rest_pose_angles": {}, "view": "side_right_three_quarter",
        "skeleton": {"bones": bones, "source_reference": {"file": "references/side_key.png", "image_size_px": [W, H]}},
        "face_mechanics": {"blink": adult_side["face_mechanics"]["blink"]},
        "physics_presets": {"spring_damper": springs,
                            "spring_groups": [{"id": k, "bones": [k], "natural_frequency_hz": v["freq"],
                                               "damping_ratio": v["zeta"]} for k, v in springs.items()]},
        "contact_markers": _contact_points(),
        "gait": gait,
        "skirt_panels": {"bones": list(PANELS_AT_HEM), "note": "F6: driven from the shins (prototype variant B)"},
        "locomotion": dict(adult_side.get("locomotion", {})),
        "layers": sorted(layers, key=lambda e: e["z_order"]),
    }
    _json(PKG / "spec.json", spec)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / "mesh_generator.py"),
                    "--spec", str(PKG / "spec.json"), "--layers", str(PKG / "layers"),
                    "--out", str(PKG / "mesh" / "mesh_data.json")], check=True)
    mesh_path = PKG / "mesh" / "mesh_data.json"
    mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
    for ml in mesh["layers"]:
        cfg = SPEC_LAYERS.get(ml["id"], {}).get("panel_weights")
        if cfg:
            ml["weight_bones"], ml["weight_values"] = _panel_weights(cfg, np.asarray(ml["vertices"], float))
    mesh_path.write_bytes(json.dumps(mesh).encode("utf-8"))
    man = PKG / "final"              # build_rig_window loads <rig_root>/<stage>/manifest.json
    man.mkdir(exist_ok=True)
    _json(man / "manifest.json", {"spec": 1,
                                  "figures": {"healthy_neutral": "../../rig/final/figs/healthy_neutral.png"},
                                  "skinned": {"spec_file": "../spec.json", "mesh_file": "../mesh/mesh_data.json",
                                              "layers_dir": "../layers"}})
    print("[OK] spec, mesh, manifest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["masks", "labels", "peel", "completions", "layers", "rig", "all"])
    a = ap.parse_args()
    order = ["masks", "labels", "peel", "completions", "rig"]       # rig re-runs layers itself
    for st in (order if a.stage == "all" else [a.stage]):
        {"masks": stage_masks, "labels": lambda: _run_layers(True), "peel": stage_peel,
         "completions": stage_completions, "layers": lambda: _run_layers(False), "rig": stage_rig}[st]()


if __name__ == "__main__":
    main()
