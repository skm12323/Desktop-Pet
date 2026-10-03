"""F6 gait numbers for the FINAL side rig (no rendering): per cadence / speed
  shoe_travel   ankle x range relative to the pelvis (canvas px) - stride under the long skirt
  shoe_len      near shoe length (contact markers) for scale
  panels        skirt panel angle ranges (deg) driven by GaitSolver._drive_panels
  skirt sigma   worst triangle stretch / compression of the skirt mesh, folds
  drift         contact drift (foot lock, px)

  <py> -X utf8 spikes/qa_final_side_gait.py [--hz 1.6,2.0,2.4] [--speeds 120,200]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PKG = ROOT / "assets" / "rig_final_walk_v1"


def run(spec: dict, rt, hz: float, speed: float) -> dict:
    from pet.rig.gait import GaitSolver
    sp = copy.deepcopy(spec)
    sp["gait"]["frequency_hz"] = hz
    H = spec["skeleton"]["source_reference"]["image_size_px"][1]
    g = GaitSolver(sp, 256 / H)
    skirt = next(l for l in rt.layers if l.layer_id == "skirt")
    V0 = skirt.rest[:, :2].astype(np.float64)
    T = skirt.triangles.reshape(-1, 3).astype(np.int64)
    E = np.stack([V0[T[:, 1]] - V0[T[:, 0]], V0[T[:, 2]] - V0[T[:, 0]]], -1)
    ok = np.abs(np.linalg.det(E)) > 1e-6
    T, Einv = T[ok], np.linalg.inv(E[ok])
    nb = len(rt.bones)
    ax = {"l": [], "r": []}
    pan = {b: [] for b in spec["skirt_panels"]["bones"]}
    smax, smin, folds, drift = 1.0, 1.0, 0, 0.0
    n = int(90 + 3 * 60 / hz)
    for i in range(n):
        o = g.update(1 / 60, speed, (round(g.window_x_float), 0))
        if i < 90:
            continue
        drift = max(drift, o.foot_slide_drift_px)
        ang = {b: math.degrees(v) for b, v in o.bone_rotations.items()}
        for side in ("l", "r"):
            ax[side].append(float(g._feet[side].ankle_now[0]) - float(o.pelvis_offset[0]))
        for b in pan:
            pan[b].append(ang.get(b, 0.0))
        pa = np.array([ang.get(b.name, 0.0) for b in rt.bones], np.float32)
        tx = np.zeros(nb, np.float32)
        ty = np.zeros(nb, np.float32)
        tx[rt.bone_index["root_hip"]], ty[rt.bone_index["root_hip"]] = o.pelvis_offset
        rt.skinning_matrices(pa, tx, ty, 0.0, 0.0)
        X = rt.deform(skirt, skirt.rest)[:, :2].astype(np.float64)
        F = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]], -1) @ Einv
        sv = np.linalg.svd(F, compute_uv=False)
        smax, smin = max(smax, float(sv[:, 0].max())), min(smin, float(sv[:, 1].min()))
        folds += int((np.linalg.det(F) < 0).sum())
    return {"hz": hz, "speed": speed,
            "shoe_travel_px": {s: round(max(v) - min(v), 1) for s, v in ax.items()},
            "panels_deg": {b: [round(min(v), 2), round(max(v), 2)] for b, v in pan.items()},
            "skirt_sigma": [round(smin, 3), round(smax, 3)], "skirt_folds": folds,
            "contact_drift_px": round(drift, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", default="1.6,2.0,2.4")
    ap.add_argument("--speeds", default="120,200")
    a = ap.parse_args()
    from pet.rig.skinned_mesh_item import RigRuntime
    spec = json.loads((PKG / "spec.json").read_text(encoding="utf-8"))
    rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh" / "mesh_data.json"), str(PKG / "layers"))
    cm = spec["contact_markers"]
    shoe = round(cm["forefoot_l"][0] - cm["heel_l"][0], 1)
    out = {"shoe_len_px": shoe, "runs": [run(spec, rt, float(h), float(v))
                                          for v in a.speeds.split(",") for h in a.hz.split(",")]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
