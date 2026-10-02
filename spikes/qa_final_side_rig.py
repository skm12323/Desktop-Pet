"""F4 gates for the FINAL side rig (assets/rig_final_walk_v1), after spikes/qa_side_rig.py (ADULT G4).

  rest recovery   Qt rest render vs side_key.png (canvas and 256-high display), holes / spill
  blink           closed-eye frame (sheet only)
  poses           walking extremes (thigh +-30, knee 0..60, foot), arm swing +-20 / elbow 40, skirt
                  panels +-8 (front/back in and out of phase), tail, head: new enclosed transparent
                  holes (display px^2) and flipped triangles (RigRuntime LBS)
  weights         row sums == 1; shoe vertices foot weight >= 0.8; skirt never weighted to legs
Evidence -> spikes/_qa/final_side_f4/ (metrics.json, pose sheet on light/dark).

Usage: QT_QPA_PLATFORM=windows QT_QUICK_BACKEND=rhi QSG_RHI_BACKEND=d3d11 <py312> -X utf8 spikes/qa_final_side_rig.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
PKG = ROOT / "assets" / "rig_final_walk_v1"
OUT = ROOT / "spikes" / "_qa" / "final_side_f4"
W, H = 1024, 1824
DISP = (round(256 * W / H), 256)

POSES = {
    "near_leg_fwd": {"upper_leg_l": -25, "lower_leg_l": 30, "foot_l": -5, "upper_leg_r": 20,
                     "lower_leg_r": 35, "foot_r": -55, "upper_arm_l": 20, "upper_arm_r": -20},
    "near_leg_back": {"upper_leg_l": 20, "lower_leg_l": 35, "foot_l": -55, "upper_leg_r": -25,
                      "lower_leg_r": 30, "foot_r": -5, "upper_arm_l": -20, "upper_arm_r": 20},
    "passing": {"upper_leg_l": -25, "lower_leg_l": 60, "foot_l": -35, "upper_leg_r": 5,
                "lower_leg_r": 10, "foot_r": -15, "upper_arm_l": 5, "upper_arm_r": -5},
    "elbows40": {"upper_arm_l": 15, "forearm_l": -40, "upper_arm_r": -15, "forearm_r": -40},
    # panel drive peaks ~2.5 deg in the 120 px/s walk (prototype variant B); 4 deg = 1.6x headroom
    "skirt_spread": {"skirt_back": 3, "skirt_mid_b": 4, "skirt_mid_f": -4, "skirt_front": -3},
    "skirt_sway_fwd": {"skirt_back": -2.5, "skirt_mid_b": -4, "skirt_mid_f": -4, "skirt_front": -2.5},
    "tail_head": {"tail_01": 8, "tail_02": 14, "tail_03": 20, "tail_fluke": 26, "head": 10,
                  "hair_back_l_01": 8, "hair_back_l_02": 12, "hair_side_r_01": -8},
    # idle / walk range: MotionEngine tail wave 3.5 deg x1.35..1.55 per link, hair springs ~2-4
    "dbg_tail": {"tail_01": 3.5, "tail_02": 4.7, "tail_03": 6.8, "tail_fluke": 10.6},
    "dbg_hair": {"hair_back_l_01": 3, "hair_back_l_02": 4, "hair_back_l_03": 5},
    "idle_sway": {"tail_01": 3.5, "tail_02": 4.7, "tail_03": 6.8, "tail_fluke": 10.6, "head": 2,
                  "hair_back_l_01": 3, "hair_back_l_02": 4, "hair_back_l_03": 5, "hair_side_r_01": -3,
                  "hair_side_r_02": -4, "ahoge_01": 4, "ear_fin_l": 3},
}


def premul(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    y[..., :3] *= y[..., 3:4] / 255.0
    return y


def disp(x: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(x).resize(DISP, Image.Resampling.LANCZOS))


def enclosed_holes(alpha: np.ndarray) -> np.ndarray:
    empty = alpha <= 20
    lab, _ = ndimage.label(empty)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return empty & ~np.isin(lab, list(border))


def main() -> None:
    from render_rig_rest import RigRenderer
    from pet.rig.motion import MotionFrame
    from pet.rig.skinned_mesh_item import RigRuntime

    OUT.mkdir(parents=True, exist_ok=True)
    key = np.asarray(Image.open(PKG / "references" / "side_key.png").convert("RGBA"))
    r = RigRenderer("final", str(PKG / "final"))
    rest_f = r.rest_frame("zero")
    rest = np.asarray(r.render(rest_f))
    Image.fromarray(rest).save(PKG / "references" / "side_rest.png")
    d = np.abs(premul(rest)[..., :3] - premul(key)[..., :3]).mean(-1)
    sil = (rest[..., 3] > 8) | (key[..., 3] > 8)
    dr, dk = disp(rest), disp(key)
    d256 = np.abs(premul(dr)[..., :3] - premul(dk)[..., :3]).mean(-1)
    s256 = (dr[..., 3] > 8) | (dk[..., 3] > 8)
    tot = int((key[..., 3] > 127).sum())
    holes = (key[..., 3] > 127) & (rest[..., 3] <= 127)
    spill = (rest[..., 3] > 127) & (key[..., 3] <= 127)
    rr = {"canvas_mean_255": float(d[sil].mean()), "canvas_p99_255": float(np.percentile(d[sil], 99)),
          "disp256_mean_255": float(d256[s256].mean()), "disp256_p99_255": float(np.percentile(d256[s256], 99)),
          "holes_frac": float(holes.sum() / tot), "spill_frac": float(spill.sum() / tot)}
    rr["pass"] = bool(rr["disp256_mean_255"] <= 6 and rr["disp256_p99_255"] <= 50
                      and rr["holes_frac"] <= 1e-3 and rr["spill_frac"] <= 1e-3)
    metrics: dict = {"rest_recovery": rr}
    if holes.any() or spill.any():
        v = np.full(key.shape[:2] + (3,), 255, np.uint8)
        v[key[..., 3] > 127] = 200
        v[holes] = (255, 0, 0)
        v[spill] = (0, 0, 255)
        Image.fromarray(v).save(OUT / "rest_holes_spill.png")

    renders = {"rest": rest}
    bf = MotionFrame(bone_angles=dict(rest_f.bone_angles), blink_progress=1.0, look_at=(0.0, 0.0))
    renders["blink"] = np.asarray(r.render(bf))

    rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh" / "mesh_data.json"), str(PKG / "layers"))
    nb = len(rt.bones)
    base_holes = enclosed_holes(rest[..., 3])
    pose_rep = {}
    for name, ang in POSES.items():
        angles = dict(rest_f.bone_angles)
        angles.update({k: float(v) for k, v in ang.items()})
        img = np.asarray(r.render(MotionFrame(bone_angles=angles, blink_progress=0.0, look_at=(0.0, 0.0))))
        renders[name] = img
        new = enclosed_holes(img[..., 3]) & ~ndimage.binary_dilation(base_holes, iterations=3)
        nd = np.asarray(Image.fromarray((new * 255).astype(np.uint8)).resize(DISP, Image.Resampling.BILINEAR)) > 64
        pa = np.zeros(nb, np.float32)
        for b, v in angles.items():
            if b in rt.bone_index:
                pa[rt.bone_index[b]] = v
        rt.skinning_matrices(pa, np.zeros(nb, np.float32), np.zeros(nb, np.float32), 0.0, 0.0)
        flips = {}
        for layer in rt.layers:
            v = rt.deform(layer, layer.rest)[:, :2].astype(np.float64)
            v0 = layer.rest[:, :2].astype(np.float64)
            t = layer.triangles.reshape(-1, 3)

            def area(p):
                return ((p[t[:, 1], 0] - p[t[:, 0], 0]) * (p[t[:, 2], 1] - p[t[:, 0], 1])
                        - (p[t[:, 2], 0] - p[t[:, 0], 0]) * (p[t[:, 1], 1] - p[t[:, 0], 1]))
            a0, a1 = area(v0), area(v)
            n = int(((np.abs(a0) > 1e-6) & (np.sign(a0) != np.sign(a1))).sum())
            if n:
                flips[layer.layer_id] = n
        pose_rep[name] = {"new_holes_disp_px2": int(nd.sum()), "flipped_triangles": flips}
    metrics["poses"] = pose_rep
    metrics["poses_pass"] = all(p["new_holes_disp_px2"] <= 2 and not p["flipped_triangles"] for p in pose_rep.values())
    metrics["idle_pass"] = pose_rep["idle_sway"]["new_holes_disp_px2"] <= 2 and not pose_rep["idle_sway"]["flipped_triangles"]

    # weights
    mesh = json.loads((PKG / "mesh" / "mesh_data.json").read_text(encoding="utf-8"))
    wrep = {"row_sum_max_err": 0.0}
    for ml in mesh["layers"]:
        for vals in ml["weight_values"]:
            wrep["row_sum_max_err"] = max(wrep["row_sum_max_err"], abs(sum(vals) - 1))
        if ml["id"] in ("leg_l", "leg_r"):
            foot = f"foot_{ml['id'][-1]}"
            v = np.asarray(ml["vertices"])
            shoe = v[:, 1] > 1715
            fw = [dict(zip(b, w)).get(foot, 0) for b, w, s in zip(ml["weight_bones"], ml["weight_values"], shoe) if s]
            wrep[f"{ml['id']}_shoe_foot_weight_min"] = round(float(min(fw)), 3) if fw else None
        if ml["id"] == "skirt":
            wrep["skirt_leg_weight_max"] = max(
                (w for b, ws in zip(ml["weight_bones"], ml["weight_values"]) for n, w in zip(b, ws) if "leg" in n),
                default=0.0)
    wrep["pass"] = bool(wrep["row_sum_max_err"] < 1e-3 and wrep["skirt_leg_weight_max"] <= 0.3
                        and all(wrep[k] is None or wrep[k] >= 0.8 for k in wrep if k.endswith("_min")))
    metrics["weights"] = wrep
    r.close()

    for bgc, tag in (((240, 242, 246), "light"), ((40, 42, 48), "dark")):
        tiles = []
        for name, img in renders.items():
            bg = Image.new("RGBA", (W, H), (*bgc, 255))
            bg.alpha_composite(Image.fromarray(img))
            t = bg.convert("RGB").resize((256, 456), Image.Resampling.LANCZOS)
            ImageDraw.Draw(t).text((4, 4), name, fill=(255, 0, 0))
            tiles.append(t)
        s = Image.new("RGB", (256 * len(tiles), 456))
        for k, t in enumerate(tiles):
            s.paste(t, (256 * k, 0))
        s.save(OUT / f"poses_{tag}.png")
    (OUT / "metrics.json").write_bytes(json.dumps(metrics, indent=1).encode("utf-8"))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
