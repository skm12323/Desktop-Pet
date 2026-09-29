"""G4: side-view ADULT rig spec + eyelid layers + mesh + manifest (assets/rig_adult_walk_v1).

Joint positions are canvas px on references/side_key.png (960x1696). Bone names follow the
front rig so MotionEngine's name-driven idle (breath, springs, blink) works unchanged;
_l = character's right = near side when facing right, _r = far side.

Run after tools/build_side_layers.py:
  D:\\anaconda3\\python.exe -X utf8 tools/build_side_spec.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "assets" / "rig_adult_walk_v1"
W, H = 960, 1696

# name: (parent, [x, y], clamp_deg, is_chain)
BONES = {
    "root_hip": (None, [530, 780], [-6, 6], False),
    "spine": ("root_hip", [525, 690], [-5, 5], False),
    "chest": ("spine", [520, 560], [-6, 6], False),
    "neck": ("chest", [512, 452], [-6, 6], False),
    "head": ("neck", [510, 410], [-12, 12], False),
    "tail_01": ("root_hip", [380, 1150], [-15, 15], True),
    "tail_02": ("tail_01", [260, 1225], [-20, 20], True),
    "tail_03": ("tail_02", [130, 1170], [-24, 24], True),
    "tail_fluke": ("tail_03", [75, 1040], [-30, 30], True),
    "hair_back_l_01": ("head", [330, 430], [-8, 8], True),
    "hair_back_l_02": ("hair_back_l_01", [250, 580], [-12, 12], True),
    "hair_back_l_03": ("hair_back_l_02", [190, 700], [-16, 16], True),
    "hair_side_r_01": ("head", [575, 430], [-8, 8], True),
    "hair_side_r_02": ("hair_side_r_01", [625, 600], [-12, 12], True),
    "upper_leg_l": ("root_hip", [490, 830], [-50, 50], True),
    "lower_leg_l": ("upper_leg_l", [464, 1225], [-10, 85], True),   # 须在髋–踝连线（y1225 处 x≈459）之前，否则静止即被解成反向膝
    "foot_l": ("lower_leg_l", [440, 1462], [-85, 85], False),
    "upper_leg_r": ("root_hip", [565, 820], [-50, 50], True),
    "lower_leg_r": ("upper_leg_r", [558, 1215], [-10, 85], True),   # 髋–踝连线 y1215 处 x≈552，膝在其前
    "foot_r": ("lower_leg_r", [544, 1455], [-85, 85], False),
    "ear_fin_l": ("head", [380, 352], [-12, 12], False),
    "ear_fin_r": ("head", [568, 372], [-12, 12], False),
    "skirt_root": ("root_hip", [520, 760], [-5, 5], False),
    "skirt_hem_l": ("skirt_root", [380, 900], [-20, 20], False),   # 摆动支点上移：裙后缘整体随腿摆
    "skirt_hem_r": ("skirt_root", [660, 900], [-20, 20], False),   # 摆动支点上移：裙前缘整体随腿摆
    "apron_root": ("spine", [560, 690], [-4, 4], False),
    "apron_tip": ("apron_root", [600, 1000], [-10, 10], False),
    "upper_arm_l": ("chest", [392, 485], [-30, 30], True),
    "forearm_l": ("upper_arm_l", [366, 656], [-45, 20], True),     # elbow centred in the sleeve (was 345: 10 px from the edge)
    "hand_l": ("forearm_l", [292, 828], [-15, 15], False),
    "upper_arm_r": ("chest", [555, 505], [-30, 30], True),        # shoulder inside the far arm (was 598: 16 px outside it)
    "forearm_r": ("upper_arm_r", [604, 671], [-45, 20], True),     # elbow centred (was 632: on the outline)
    "hand_r": ("forearm_r", [668, 832], [-15, 15], False),
    "eyelid_l": ("head", [478, 300], [0, 0], False),
    "eyelid_r": ("head", [556, 304], [0, 0], False),
    "ahoge_01": ("head", [445, 118], [-15, 15], True),
    "ahoge_02": ("ahoge_01", [390, 60], [-25, 25], True),
}

# hinged limbs split into rigid pieces: source layer -> [(piece id, bone, z offset)] parent first
HINGED = {
    "arm_l": [("arm_l_upper", "upper_arm_l", 2), ("arm_l_fore", "forearm_l", 1), ("arm_l_hand", "hand_l", 0)],
    "arm_r": [("arm_r_upper", "upper_arm_r", 2), ("arm_r_fore", "forearm_r", 1), ("arm_r_hand", "hand_r", 0)],
}

BLINK = {"eyelid_l": [478, 300, 341, 32], "eyelid_r": [556, 304, 343, 16]}   # [cx, top, bottom, radius]
EYE_BOX = {"eyelid_l": [444, 294, 514, 324], "eyelid_r": [538, 298, 578, 320]}

LAYERS = [
    {"id": "tail", "bind_bone": "tail_01", "influence_bones": ["tail_01", "tail_02", "tail_03", "tail_fluke"],
     "weights": {"mode": "chain", "blend_px": 40}, "grid_step": 20, "min_component_px": 200},
    {"id": "hair_back", "bind_bone": "head",
     "influence_bones": ["head", "hair_back_l_01", "hair_back_l_02", "hair_back_l_03"],
     "root_lock": {"bone": "head", "full_before_y": 430, "free_after_y": 560},
     "weights": {"mode": "chain", "blend_px": [40, 50, 50]}, "min_component_px": 40},
    {"id": "ear_fin_r", "bind_bone": "ear_fin_r", "influence_bones": ["ear_fin_r"]},
    # arm_r: rigid pieces with disc caps (plan D2; tools/split_hinged_limb.py) - a single mesh bent at the
    # elbow either folds or stretches the outer contour ~2x (plan D1)
    {"id": "arm_r_hand", "bind_bone": "hand_r", "influence_bones": ["hand_r"], "min_component_px": 150},
    {"id": "arm_r_fore", "bind_bone": "forearm_r", "influence_bones": ["forearm_r"], "min_component_px": 150},
    {"id": "arm_r_upper", "bind_bone": "upper_arm_r", "influence_bones": ["upper_arm_r"], "min_component_px": 150},
    # 腿不再钉骨盆（旧 root_lock 1100→1400 把膝盖 y≈1220 包在过渡带里，小腿中段出现假关节、
    # 反向弯折）：整条可见腿只由腿骨驱动，腿根在裙下的前后摆动由裙摆跟随大腿（gait
    # skirt_follow_gain）配合
    {"id": "leg_r", "bind_bone": "upper_leg_r", "influence_bones": ["upper_leg_r", "lower_leg_r", "foot_r"],
     "rigid_below": [{"bone": "foot_r", "y": 1478, "blend": 22}],
     # knee: crisp kneecap (22 px), wide crease on the back of the knee (no fold up to 90 deg)
     "weights": {"mode": "chain", "blend_px": [22, 28],
                 "inner": {"lower_leg_r": {"side": 1, "blend_px": 110, "down_px": 60, "half_width_px": 80},
                           "foot_r": {"side": -1, "blend_px": 60, "down_px": 28, "half_width_px": 40}}},
     "grid_step": 14, "min_component_px": 150},
    {"id": "leg_l", "bind_bone": "upper_leg_l", "influence_bones": ["upper_leg_l", "lower_leg_l", "foot_l"],
     "rigid_below": [{"bone": "foot_l", "y": 1485, "blend": 22}],
     # knee: crisp kneecap (22 px), wide crease on the back of the knee (no fold up to 90 deg)
     "weights": {"mode": "chain", "blend_px": [22, 28],
                 "inner": {"lower_leg_l": {"side": 1, "blend_px": 110, "down_px": 60, "half_width_px": 80},
                           "foot_l": {"side": -1, "blend_px": 60, "down_px": 28, "half_width_px": 40}}},
     "grid_step": 14, "min_component_px": 150},
    {"id": "skirt", "bind_bone": "skirt_root",
     "influence_bones": ["root_hip", "skirt_root", "skirt_hem_l", "skirt_hem_r"],
     "root_lock": {"bone": "root_hip", "full_before_y": 800, "free_after_y": 960},
     "weights": {"mode": "skirt", "root": "skirt_root", "hem_l": "skirt_hem_l", "hem_r": "skirt_hem_r",
                 "y0": 880, "y1": 1080, "cx": 520, "half_w": 90}, "grid_step": 20, "min_component_px": 200},
    {"id": "torso", "bind_bone": "chest", "influence_bones": ["spine", "chest", "neck"],
     "weights": {"mode": "chain", "blend_px": [40, 30]}, "min_component_px": 150},
    {"id": "apron", "bind_bone": "apron_root", "influence_bones": ["apron_root", "apron_tip"],
     "root_lock": {"bone": "apron_root", "full_before_y": 760, "free_after_y": 900},
     "weights": {"mode": "chain", "blend_px": 80}, "min_component_px": 150},
    {"id": "hair_side_r", "bind_bone": "head", "influence_bones": ["head", "hair_side_r_01", "hair_side_r_02"],
     "root_lock": {"bone": "head", "full_before_y": 440, "free_after_y": 520},
     "weights": {"mode": "chain", "blend_px": [30, 40]}, "min_component_px": 50},
    # arm_l: rigid pieces with disc caps (plan D2; tools/split_hinged_limb.py) - a single mesh bent at the
    # elbow either folds or stretches the outer contour ~2x (plan D1)
    {"id": "arm_l_hand", "bind_bone": "hand_l", "influence_bones": ["hand_l"], "min_component_px": 150},
    {"id": "arm_l_fore", "bind_bone": "forearm_l", "influence_bones": ["forearm_l"], "min_component_px": 150},
    {"id": "arm_l_upper", "bind_bone": "upper_arm_l", "influence_bones": ["upper_arm_l"], "min_component_px": 150},
    {"id": "shoulder_frill_l", "bind_bone": "chest", "influence_bones": ["chest"]},
    {"id": "head_base", "bind_bone": "head", "influence_bones": ["head"],
     "blink_zones": [BLINK["eyelid_l"], BLINK["eyelid_r"]]},
    {"id": "eyelid_l", "bind_bone": "eyelid_l", "influence_bones": ["head", "eyelid_l"],
     "blink_zones": [BLINK["eyelid_l"]]},
    {"id": "eyelid_r", "bind_bone": "eyelid_r", "influence_bones": ["head", "eyelid_r"],
     "blink_zones": [BLINK["eyelid_r"]]},
    {"id": "ear_fin_l", "bind_bone": "ear_fin_l", "influence_bones": ["ear_fin_l"]},
    {"id": "ahoge", "bind_bone": "ahoge_01", "influence_bones": ["ahoge_01", "ahoge_02"],
     "weights": {"mode": "chain", "blend_px": 20}, "min_component_px": 30},
    {"id": "headdress", "bind_bone": "head", "influence_bones": ["head"]},
]

# shoe-bottom contact points (canvas px on the key art); gait.py wants offsets from the ankle
CONTACT_POINTS = {"heel_l": [415, 1596], "sole_l": [475, 1603], "forefoot_l": [535, 1600],
                  "heel_r": [510, 1552], "sole_r": [575, 1566], "forefoot_r": [635, 1562]}
GAIT = {"frequency_hz": 1.6, "speed_world_px_s": 120.0, "stance_ratio": 0.6,
        "knee_bend_direction": -1,
        "park_feet": True, "stance_extension": 0.985,
        "adaptive_cadence": True,
        "lean_degrees": 1.8, "skirt_follow_gain": 0.8,
        "swing_lift_world_px": 4.0, "foot_track_sep_world_px": 0.0, "turn_duration_s": 0.02,
        "per_side_ground": 1.0, "sway_world_px": 0.0, "arm_swing_deg": 9.0, "forearm_bend_deg": 26.0,
        "forearm_base_deg": 8.0, "arm_phase_lag": 0.06, "hand_follow": 0.35, "far_arm_scale": 0.55, "track_offset_px": -30.0, "toe_off_end_deg": 35.0, "torso_lean_deg": 3.0,
        "brake_linear_s": 0.4, "dip_geometric": 1.0, "track_from_rest": 1.0, "speed_cap_world_px_s": 200.0}


def write_eyelids() -> None:
    """Upper lash lines copied from head_base (kept there too) - the blink cover layers."""
    head = np.asarray(Image.open(PKG / "layers_full" / "head_base.png").convert("RGBA"))
    lum = head[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    for lid, (x0, y0, x1, y1) in EYE_BOX.items():
        m = np.zeros(lum.shape, bool)
        m[y0:y1, x0:x1] = (lum[y0:y1, x0:x1] < 110) & (head[y0:y1, x0:x1, 3] == 255)
        out = np.zeros_like(head)
        out[m] = head[m]
        Image.fromarray(out, "RGBA").save(PKG / "layers_full" / f"{lid}.png")
        print(f"[OK] {lid}: {int(m.sum())} lash px")


def trim_layers(margin: int = 2) -> dict:
    """layers_full/<id>.png (canvas) -> layers/<id>.png cropped to alpha bbox + margin.
    Vertices stay in canvas px; mesh_generator maps texture px back with trim_offset_px."""
    (PKG / "layers").mkdir(exist_ok=True)
    offs, raw_full, raw_trim = {}, 0, 0
    for l in LAYERS:
        im = Image.open(PKG / "layers_full" / f"{l['id']}.png").convert("RGBA")
        a = np.asarray(im)[..., 3]
        ys, xs = np.where(a > 0)
        x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
        x1, y1 = min(W, xs.max() + 1 + margin), min(H, ys.max() + 1 + margin)
        im.crop((x0, y0, x1, y1)).save(PKG / "layers" / f"{l['id']}.png", optimize=True)
        offs[l["id"]] = [int(x0), int(y0)]
        raw_full += W * H * 4
        raw_trim += (x1 - x0) * (y1 - y0) * 4
    print(f"[OK] trimmed textures: {raw_full / 1e6:.1f} MB -> {raw_trim / 1e6:.1f} MB raw RGBA")
    return offs


# Whole-limb redraws (registered RGBA, canvas offset) used for everything the limb hides behind
# higher layers at rest. The far arm is mostly occluded by torso / side hair / apron; the old
# procedural completion left it a thin sliver plus dark smears, which showed once the arm swung
# as a rigid piece (user review 2026-09-29).
LIMB_REDRAW = {"arm_r": ("prep/peel/arm_r_gpt_s1_canvas.png", (0, 0))}   # gpt-image-2.5 whole arm, see .json


def limb_source(limb: str) -> np.ndarray:
    """Visible pixels from the art + redraw pixels where higher layers cover the limb at rest."""
    rgba = np.asarray(Image.open(PKG / "layers_full" / f"{limb}.png").convert("RGBA")).copy()
    if limb not in LIMB_REDRAW:
        return rgba
    path, (ox, oy) = LIMB_REDRAW[limb]
    red = np.asarray(Image.open(PKG / path).convert("RGBA"))
    lab = np.asarray(Image.open(PKG / "prep" / "partition_labels.png"))
    ids = {int(k): v for k, v in json.loads((PKG / "prep" / "partition_ids.json").read_text(encoding="utf-8")).items()}
    zs = {k: v["z"] for k, v in json.loads((PKG / "prep" / "partition.json").read_text(encoding="utf-8"))["layers"].items()}
    higher = np.isin(lab, [i for i, n in ids.items() if n in zs and zs[n] > zs[limb]])
    own = lab == next(i for i, n in ids.items() if n == limb)
    h, w = red.shape[:2]
    canvas = np.zeros_like(rgba)
    canvas[oy:oy + h, ox:ox + w] = red
    # whole redraw (visible sliver included): mixing the art's visible sliver with a separately
    # drawn arm doubled the cuff (bands ~10 px apart) as soon as the arm moved (review 2026-09-29)
    use = (canvas[..., 3] > 127) & (higher | own)
    rgba[own & ~use] = 0
    rgba[use] = canvas[use]
    from scipy import ndimage
    near_own = ndimage.binary_dilation(own, iterations=8)   # keep the anti-seam underlap
    drop = higher & ~own & ~use & ~near_own                 # old procedural fill hidden behind the body
    rgba[drop] = 0
    return rgba


# far side hair hangs BEHIND the far arm (user review 2026-09-29: the swinging far arm passed under
# it); it stays above the ear fin, the torso still covers the far arm
Z_OVERRIDE = {"hair_side_r": 9}


def uncover_far_hair() -> None:
    """Torso completion pixels under the far side hair would now cover it: drop them."""
    lab = np.asarray(Image.open(PKG / "prep" / "partition_labels.png"))
    ids = {int(k): v for k, v in json.loads((PKG / "prep" / "partition_ids.json").read_text(encoding="utf-8")).items()}
    hair = lab == next(i for i, n in ids.items() if n == "hair_side_r")
    path = PKG / "layers_full" / "torso.png"
    t = np.asarray(Image.open(path).convert("RGBA")).copy()
    t[hair] = 0
    Image.fromarray(t, "RGBA").save(path)


def split_hinged() -> dict:
    """layers_full/<limb>.png -> layers_full/<piece>.png; returns piece id -> z offset."""
    from split_hinged_limb import split
    zoff = {}
    for limb, chain in HINGED.items():
        rgba = limb_source(limb)
        pieces = split(rgba, [np.array(BONES[b][1], float) for _, b, _ in chain])
        for (pid, _, dz), arr in zip(chain, pieces):
            Image.fromarray(arr, "RGBA").save(PKG / "layers_full" / f"{pid}.png")
            zoff[pid] = (limb, dz)
    return zoff


def main() -> None:
    zoff = split_hinged()
    uncover_far_hair()
    write_eyelids()
    trims = trim_layers()
    front = json.loads((ROOT / "assets" / "rig_adult" / "spec.json").read_text(encoding="utf-8"))
    part = json.loads((PKG / "prep" / "partition.json").read_text(encoding="utf-8"))["layers"]
    z = {lid: s["z"] for lid, s in part.items()}
    z.update({"eyelid_l": 51, "eyelid_r": 52})
    z.update({pid: z[limb] + dz for pid, (limb, dz) in zoff.items()})
    z.update(Z_OVERRIDE)
    bones = [{"bone_name": n, "parent": p, "joint_pos": [round(x / W, 5), round(y / H, 5)],
              "angle_clamp": c, "is_chain": ch} for n, (p, (x, y), c, ch) in BONES.items()]
    layers = []
    for l in LAYERS:
        e = {"id": l["id"], "description": f"side rig layer {l['id']} (pixel-exact cut of side_key.png)",
             "bind_bone": l["bind_bone"], "influence_bones": l["influence_bones"], "z_order": z[l["id"]],
             "requires_inpaint": False, "bbox_hint": [0, 0, 1, 1]}
        for k in ("root_lock", "rigid_below", "blink_zones", "weights", "grid_step", "min_component_px"):
            if k in l:
                e[k] = l[k]
        e["per_component"] = True          # no mesh welding between separate pieces (plan D5)
        e["trim_offset_px"] = trims[l["id"]]
        layers.append(e)
    springs = {k: v for k, v in front["physics_presets"]["spring_damper"].items() if k in BONES}
    for k in ("hair_side_r_01", "hair_side_r_02"):
        springs.setdefault(k, {"freq": 1.3 if k.endswith("1") else 1.1, "zeta": 0.6})
    spec = {
        "source_facing": 1,
        "texture_mipmaps": True,
        "ground_anchor_y_px": 1608,
        "rest_pose_angles": {},
        "view": "side_right_three_quarter",
        "skeleton": {"bones": bones,
                     "source_reference": {"file": "references/side_key.png", "image_size_px": [W, H]}},
        "face_mechanics": {"blink": front["face_mechanics"]["blink"]},
        "physics_presets": {"spring_damper": springs,
                            "spring_groups": [{"id": k, "bones": [k], "natural_frequency_hz": v["freq"],
                                               "damping_ratio": v["zeta"]} for k, v in springs.items()]},
        "contact_markers": {k: [v[0] - BONES[f"foot_{k[-1]}"][1][0], v[1] - BONES[f"foot_{k[-1]}"][1][1]]
                            for k, v in CONTACT_POINTS.items()},
        "gait": GAIT,
        "layers": sorted(layers, key=lambda e: e["z_order"]),
    }
    (PKG / "spec.json").write_bytes(json.dumps(spec, indent=2, ensure_ascii=False).encode("utf-8"))
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools" / "mesh_generator.py"),
                    "--spec", str(PKG / "spec.json"), "--layers", str(PKG / "layers"),
                    "--out", str(PKG / "mesh" / "mesh_data.json")], check=True)
    man_dir = PKG / "adult"          # build_rig_window loads <rig_root>/<stage>/manifest.json
    man_dir.mkdir(exist_ok=True)
    manifest = {"spec": 1,
                "figures": {"healthy_neutral": "../../rig/adult/figs/healthy_neutral.png"},
                "skinned": {"spec_file": "../spec.json", "mesh_file": "../mesh/mesh_data.json",
                            "layers_dir": "../layers"}}
    (man_dir / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
    print("[OK] spec, mesh, manifest")


if __name__ == "__main__":
    main()
