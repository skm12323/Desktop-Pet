"""Deformation metrics for the ADULT side rig (docs/ADULT行走修复-2026-09-29.md, plan phase 0).

One run reports, per layer:
  * folds   - inverted triangles during the real gait (120 / 200 px/s, 256 window) and in a
              human-range stress pose set (knee 65, hip +30/-15, elbow 45)
  * sigma   - worst triangle stretch / compression (singular values of the deformation
              gradient) over gait + stress poses
  * welds   - triangles that bridge two separate alpha pieces of the same layer
Output: JSON to stdout and --out (default spikes/_qa/deform_metrics/<tag>.json).

  D:\\anaconda3\\python.exe -X utf8 spikes/qa_deform_metrics.py --tag baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PKG = ROOT / "assets" / "rig_adult_walk_v1"

STRESS = {
    "knee65_l": {"upper_leg_l": -25, "lower_leg_l": 65, "foot_l": -35},
    "knee65_r": {"upper_leg_r": -25, "lower_leg_r": 65, "foot_r": -35},
    "hip_fwd30": {"upper_leg_l": -30, "lower_leg_l": 10, "upper_leg_r": 15, "lower_leg_r": 30},
    "hip_back15": {"upper_leg_l": 15, "lower_leg_l": 40, "foot_l": -30, "upper_leg_r": -30},
    "elbow45": {"upper_arm_l": 20, "forearm_l": -45, "upper_arm_r": -20, "forearm_r": -45},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="current")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from pet.rig.gait import GaitSolver
    from pet.rig.skinned_mesh_item import RigRuntime
    spec = json.loads((PKG / "spec.json").read_text(encoding="utf-8"))
    rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh/mesh_data.json"), str(PKG / "layers"))
    pre = {}
    for L in rt.layers:
        V = L.rest[:, :2].astype(np.float64)
        T = L.triangles.reshape(-1, 3).astype(np.int64)
        E = np.stack([V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]], -1)
        ok = np.abs(np.linalg.det(E)) > 1e-6
        pre[L.layer_id] = (T[ok], np.linalg.inv(E[ok]))
    res = {L.layer_id: {"folds_gait": 0, "folds_stress": {}, "sigma_max": 1.0, "sigma_min": 1.0} for L in rt.layers}

    def measure(kind: str, name: str = "") -> None:
        for L in rt.layers:
            T, Einv = pre[L.layer_id]
            if not len(T):
                continue
            X = rt.deform(L, L.rest)[:, :2].astype(np.float64)
            F = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]], -1) @ Einv
            det = np.linalg.det(F)
            s = np.linalg.svd(F, compute_uv=False)
            r = res[L.layer_id]
            nf = int((det < 0).sum())
            if kind == "gait":
                r["folds_gait"] += int(nf > 0)
            elif nf:
                r["folds_stress"][name] = nf
            good = det > 0
            if good.any():
                r["sigma_max"] = max(r["sigma_max"], float(s[good, 0].max()))
                r["sigma_min"] = min(r["sigma_min"], float(s[good, 1].min()))

    for speed in (120, 200):
        g = GaitSolver(spec, 256 / 1696)
        for i in range(420):
            o = g.update(1 / 60, speed if i < 300 else 0, (round(g.window_x_float), 0))
            if i % 2:
                continue
            ang = np.array([np.degrees(o.bone_rotations.get(b.name, 0)) for b in rt.bones], np.float32)
            tx, ty = np.zeros_like(ang), np.zeros_like(ang)
            tx[rt.bone_index["root_hip"]], ty[rt.bone_index["root_hip"]] = o.pelvis_offset
            rt.skinning_matrices(ang, tx, ty, 0, 0)
            measure("gait")
    for name, pose in STRESS.items():
        ang = np.array([pose.get(b.name, 0) for b in rt.bones], np.float32)
        z = np.zeros_like(ang)
        rt.skinning_matrices(ang, z, z, 0, 0)
        measure("stress", name)

    # welds: a connected piece of MESH that covers k separate alpha pieces (dilated 2 px, the same
    # rule the mesher uses) welds k-1 of them together
    for L in rt.layers:
        alpha = np.asarray(Image.open(PKG / "layers_full" / f"{L.layer_id}.png"))[..., 3] > 0
        lab, _ = ndimage.label(ndimage.binary_dilation(alpha, iterations=2), structure=np.ones((3, 3)))
        lab = np.where(alpha, lab, 0)
        V = L.rest[:, :2]
        T = L.triangles.reshape(-1, 3).astype(np.int64)
        parent = list(range(len(V)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for t in T:
            for j in (1, 2):
                a_, b_ = find(int(t[0])), find(int(t[j]))
                if a_ != b_:
                    parent[a_] = b_
        pieces: dict[int, set] = {}
        h, w = lab.shape
        for t in T:
            x0, y0 = np.floor(V[t].min(0)).astype(int)
            x1, y1 = np.ceil(V[t].max(0)).astype(int)
            win = lab[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            got = set(np.unique(win[win > 0]).tolist())
            if got:
                pieces.setdefault(find(int(t[0])), set()).update(got)
        res[L.layer_id]["welds"] = int(sum(len(s) - 1 for s in pieces.values()))

    out = {"tag": a.tag, "layers": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                    for k, v in res.items()}}
    out["summary"] = {"folds_gait_frames": sum(v["folds_gait"] for v in res.values()),
                      "folds_stress": sum(sum(v["folds_stress"].values()) for v in res.values()),
                      "sigma_max": round(max(v["sigma_max"] for v in res.values()), 3),
                      "sigma_min": round(min(v["sigma_min"] for v in res.values()), 3),
                      "welds": sum(v["welds"] for v in res.values())}
    p = Path(a.out or ROOT / "spikes/_qa/deform_metrics" / f"{a.tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps(out, indent=1).encode("utf-8"))
    for k, v in out["layers"].items():
        if v["folds_gait"] or v["folds_stress"] or v["sigma_max"] > 1.25 or v["sigma_min"] < 0.8 or v["welds"]:
            print(f"{k:16s} {v}")
    print("SUMMARY", out["summary"])


if __name__ == "__main__":
    main()
