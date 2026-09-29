"""G4 gates for the side-view ADULT rig (docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §5.2 G4).

  rest recovery   Qt rest render vs side_key.png (canvas and 256 display)
  holes / spill   key opaque but render transparent (and vice versa)
  extreme poses   walking extremes (thigh +-30, knee 0..60, foot +-30, arm +-20): new enclosed
                  transparent holes (display px^2) and flipped triangles (from RigRuntime LBS)
  weights         row sums; shoe vertices foot weight >= 0.8; skirt leg weight <= 0.3
  z order         near leg/arm above far leg/arm
Evidence -> spikes/_qa/adult_walk_v1/g4_side_rig/ (metrics.json, pose sheets on light/dark).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PKG = ROOT / "assets" / "rig_adult_walk_v1"
OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g4_side_rig"
DISP = 256 / 1696

POSES = {
    "near_leg_fwd": {"upper_leg_l": -25, "lower_leg_l": 30, "foot_l": -5, "upper_leg_r": 20,
                     "lower_leg_r": 35, "foot_r": -55, "upper_arm_l": 20, "upper_arm_r": -20},
    "near_leg_back": {"upper_leg_l": 20, "lower_leg_l": 35, "foot_l": -55, "upper_leg_r": -25,
                      "lower_leg_r": 30, "foot_r": -5, "upper_arm_l": -20, "upper_arm_r": 20},
    "passing": {"upper_leg_l": -25, "lower_leg_l": 60, "foot_l": -35, "upper_leg_r": 5,
                "lower_leg_r": 10, "foot_r": -15, "upper_arm_l": 5, "upper_arm_r": -5},
}


def premul(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    y[..., :3] *= y[..., 3:4] / 255.0
    return y


def to_disp(x: np.ndarray, size=(145, 256)) -> np.ndarray:
    pm = premul(x)
    return np.stack([np.asarray(Image.fromarray(pm[..., i]).resize(size, Image.Resampling.LANCZOS))
                     for i in range(4)], -1)


def enclosed_holes(alpha: np.ndarray) -> np.ndarray:
    solid = alpha > 127
    return ndimage.binary_fill_holes(solid) & ~solid


def main() -> None:
    from render_rig_rest import RigRenderer
    from pet.rig.skinned_mesh_item import RigRuntime

    OUT.mkdir(parents=True, exist_ok=True)
    key = np.asarray(Image.open(PKG / "references" / "side_key.png").convert("RGBA"))
    r = RigRenderer("adult", str(PKG / "adult"))
    rest_frame = r.rest_frame("zero")
    rest = np.asarray(r.render(rest_frame))
    Image.fromarray(rest).save(PKG / "references" / "side_rest.png")

    sil = (rest[..., 3] > 8) | (key[..., 3] > 8)
    d = np.abs(premul(rest)[..., :3] - premul(key)[..., :3]).mean(-1)
    dr, dk = to_disp(rest), to_disp(key)
    s256 = (dr[..., 3] > 8) | (dk[..., 3] > 8)
    d256 = np.abs(dr[..., :3] - dk[..., :3]).mean(-1)
    tot = int((key[..., 3] > 127).sum())
    metrics: dict = {"rest_recovery": {
        "canvas_mean_255": float(d[sil].mean()), "canvas_p99_255": float(np.percentile(d[sil], 99)),
        "disp256_mean_255": float(d256[s256].mean()), "disp256_p99_255": float(np.percentile(d256[s256], 99)),
        "holes_frac": float(((key[..., 3] > 127) & (rest[..., 3] <= 127)).sum() / tot),
        "spill_frac": float(((rest[..., 3] > 127) & (key[..., 3] <= 127)).sum() / tot)}}
    rr = metrics["rest_recovery"]
    rr["pass"] = bool(rr["disp256_mean_255"] <= 6 and rr["disp256_p99_255"] <= 50
                      and rr["holes_frac"] <= 1e-3 and rr["spill_frac"] <= 1e-3)

    # blink closed
    from pet.rig.motion import MotionFrame
    blink_f = MotionFrame(bone_angles=dict(rest_frame.bone_angles), blink_progress=1.0, look_at=(0.0, 0.0))
    renders = {"rest": rest, "blink": np.asarray(r.render(blink_f))}

    # extreme poses: holes + flipped triangles
    rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh" / "mesh_data.json"), str(PKG / "layers"))
    base_holes = enclosed_holes(rest[..., 3])
    nb = len(rt.bones)
    pose_rep = {}
    for name, ang in POSES.items():
        angles = dict(rest_frame.bone_angles)
        angles.update({k: float(v) for k, v in ang.items()})
        img = np.asarray(r.render(MotionFrame(bone_angles=angles, blink_progress=0.0, look_at=(0.0, 0.0))))
        renders[name] = img
        new_holes = enclosed_holes(img[..., 3]) & ~ndimage.binary_dilation(base_holes, iterations=3)
        lab, n = ndimage.label(new_holes)
        sizes = ndimage.sum(new_holes, lab, range(1, n + 1)) if n else np.zeros(0)
        pa = np.zeros(nb, np.float32)
        for b, v in angles.items():
            if b in rt.bone_index:
                pa[rt.bone_index[b]] = v
        rt.skinning_matrices(pa, np.zeros(nb, np.float32), np.zeros(nb, np.float32), 0.0, 0.0)
        flips = {}
        for layer in rt.layers:
            v = rt.deform(layer, layer.rest).copy()
            t = layer.triangles.reshape(-1, 3)

            def area(p):
                a, b, c = p[t[:, 0], :2], p[t[:, 1], :2], p[t[:, 2], :2]
                return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            a0, a1 = area(layer.rest), area(v)
            f = int(((np.sign(a0) != np.sign(a1)) & (np.abs(a0) > 1e-6)).sum())
            if f:
                flips[layer.layer_id] = f
        pose_rep[name] = {"new_hole_max_disp_px2": float((sizes.max() if len(sizes) else 0) * DISP * DISP),
                          "new_hole_total_disp_px2": float(new_holes.sum() * DISP * DISP),
                          "flipped_triangles": flips}
        pose_rep[name]["pass"] = bool(pose_rep[name]["new_hole_max_disp_px2"] <= 4 and not flips)
    metrics["extreme_poses"] = pose_rep

    # weights
    mesh = json.loads((PKG / "mesh" / "mesh_data.json").read_text(encoding="utf-8"))
    wrep = {"row_sum_max_err": 0.0, "shoe_min_foot_w": {}, "skirt_max_leg_w": 0.0}
    for ml in mesh["layers"]:
        for bones, ws in zip(ml["weight_bones"], ml["weight_values"]):
            wrep["row_sum_max_err"] = max(wrep["row_sum_max_err"], abs(sum(ws) - 1))
        if ml["id"] in ("leg_l", "leg_r"):
            side = ml["id"][-1]
            ys = [v[1] for v in ml["vertices"]]
            fw = [dict(zip(b, w)).get(f"foot_{side}", 0.0)
                  for b, w, y in zip(ml["weight_bones"], ml["weight_values"], ys) if y >= 1490]
            wrep["shoe_min_foot_w"][ml["id"]] = float(min(fw)) if fw else None
        if ml["id"] == "skirt":
            for b, w in zip(ml["weight_bones"], ml["weight_values"]):
                lw = sum(x for bb, x in zip(b, w) if "leg" in bb)
                wrep["skirt_max_leg_w"] = max(wrep["skirt_max_leg_w"], lw)
    wrep["pass"] = bool(wrep["row_sum_max_err"] <= 1e-3 and wrep["skirt_max_leg_w"] <= 0.3
                        and all(v is not None and v >= 0.8 for v in wrep["shoe_min_foot_w"].values()))
    metrics["weights"] = wrep
    z = {l["id"]: l["z_order"] for l in json.loads((PKG / "spec.json").read_text(encoding="utf-8"))["layers"]}
    metrics["z_order"] = {"near_leg_over_far": z["leg_l"] > z["leg_r"], "near_arm_over_far": z["arm_l"] > z["arm_r"]}
    r.close()

    (OUT / "metrics.json").write_bytes(json.dumps(metrics, indent=2).encode("utf-8"))
    # sheets: 512 tall on light + dark, and the 256 view
    names = list(renders)
    for h, tag in ((512, "512"), (256, "256")):
        w = int(round(960 * h / 1696))
        sheet = Image.new("RGB", (w * len(names), h * 2 + 14), "white")
        dr_ = ImageDraw.Draw(sheet)
        for k, n in enumerate(names):
            im = Image.fromarray(renders[n], "RGBA").resize((w, h), Image.Resampling.LANCZOS)
            for row, bgc in enumerate(((245, 246, 250), (34, 40, 49))):
                bg = Image.new("RGBA", (w, h), (*bgc, 255))
                bg.alpha_composite(im)
                sheet.paste(bg.convert("RGB"), (k * w, 14 + row * h))
            dr_.text((k * w + 2, 1), n, fill=(0, 0, 0))
        sheet.save(OUT / f"poses_{tag}.png")
    face = [Image.fromarray(renders[n], "RGBA").crop((380, 220, 660, 460)) for n in ("rest", "blink")]
    fs = Image.new("RGBA", (560, 240), (245, 246, 250, 255))
    fs.alpha_composite(face[0], (0, 0))
    fs.alpha_composite(face[1], (280, 0))
    fs.convert("RGB").save(OUT / "blink_face.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
