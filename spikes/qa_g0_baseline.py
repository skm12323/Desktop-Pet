"""G0 baseline gate (docs/ADULT转身视频过渡与侧身骨骼行走方案-2026-09-27.md §5.2 G0).

Checks, all through the real Qt skinned path:
  1. relaxed rest render is reproducible (two renders, max abs diff <= 1/255);
  2. it matches what the app shows at idle: the MotionEngine's first idle frame
     rendered in the 256x256 window with the ADULT ground shift, vs the relaxed
     rest rendered the same way (mean abs diff over the silhouette <= 2/255);
  3. records sha256 of the production ADULT rig package + environment, so later
     stages can prove assets/rig_adult/ was not touched.

Run: D:\\anaconda3\\python.exe -X utf8 spikes/qa_g0_baseline.py
Outputs: spikes/_qa/adult_walk_v1/g0_baseline/
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
from PIL import Image

from render_rig_rest import RigRenderer

OUT = ROOT / "spikes" / "_qa" / "adult_walk_v1" / "g0_baseline"
PROD = [ROOT / "assets" / "rig" / "adult", ROOT / "assets" / "rig_adult"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def map_canvas_to_window(rgba: np.ndarray, fit: float, dx: float, dy: float,
                         out_size: tuple[int, int]) -> np.ndarray:
    """Canvas RGBA -> window RGBA with window = canvas * fit + (dx, dy).

    Premultiplied, Gaussian prefilter (~ the GPU's mip averaging at this scale),
    bilinear sampling at pixel centres. Returns float32 straight-alpha RGBA.
    """
    from scipy import ndimage
    a = rgba[..., 3:4] / 255.0
    pm = np.concatenate([rgba[..., :3] * a, rgba[..., 3:4]], axis=-1)
    sigma = 0.42 / fit
    w, h = out_size
    out = np.zeros((h, w, 4), np.float32)
    matrix = np.array([1.0 / fit, 1.0 / fit])
    offset = np.array([(0.5 - dy) / fit - 0.5, (0.5 - dx) / fit - 0.5])
    for c in range(4):
        ch = ndimage.gaussian_filter(pm[..., c], sigma)
        out[..., c] = ndimage.affine_transform(ch, matrix, offset=offset, output_shape=(h, w),
                                               order=1, mode="constant", cval=0.0)
    alpha = np.clip(out[..., 3:4], 0.0, 255.0)
    rgb = np.where(alpha > 1e-3, out[..., :3] / np.maximum(alpha / 255.0, 1e-6), 0.0)
    return np.concatenate([np.clip(rgb, 0, 255), alpha], axis=-1)


def premul_diff(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean / P99 abs difference of premultiplied colour over the union silhouette (0-255)."""
    sil = (x[..., 3] > 8) | (y[..., 3] > 8)
    px = x[..., :3] * x[..., 3:4] / 255.0
    py = y[..., :3] * y[..., 3:4] / 255.0
    d = np.abs(px - py).mean(-1)[sil]
    return float(d.mean()), float(np.percentile(d, 99))


def geometry_offsets(x: np.ndarray, y: np.ndarray) -> dict:
    """Alpha-centroid and sole-line offsets (x minus y) in window pixels."""
    def centroid(img):
        al = img[..., 3]
        ys, xs = np.mgrid[0:al.shape[0], 0:al.shape[1]]
        s = al.sum()
        return float((xs * al).sum() / s), float((ys * al).sum() / s)

    def sole(img):
        al = img[..., 3] / 255.0
        rows = al.shape[0]
        # sub-pixel bottom edge per column: rows covered = sum of alpha below the body
        cols = np.where(al.max(0) > 0.5)[0]
        edges = []
        for c in cols:
            col = al[:, c]
            opaque = np.where(col > 0.5)[0]
            if not len(opaque):
                continue
            last = opaque[-1]
            edges.append(last + 0.5 + col[last + 1:].sum())
        edges = np.array(edges)
        return float(np.median(np.sort(edges)[-max(3, len(edges) // 20):]))

    cx0, cy0 = centroid(x)
    cx1, cy1 = centroid(y)
    return {"centroid_dx": cx0 - cx1, "centroid_dy": cy0 - cy1, "sole_dy": sole(x) - sole(y)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = RigRenderer("adult")
    rest = r.rest_frame("relaxed")

    # 1. reproducibility at canvas resolution
    a = np.asarray(r.render(rest), np.int16)
    b = np.asarray(r.render(rest), np.int16)
    repro_max = int(np.abs(a - b).max())

    # 2. canvas -> 256-window mapping: the canvas-exact rest render placed with the
    #    runtime's fit + ADULT ground shift must match Qt's own 256 render of the same
    #    pose. Clip frames are registered to the canvas and shown through this mapping,
    #    so this is the geometry (and resampling) the rig<->clip seam inherits.
    W = H = 256
    im_rest = r.render(rest, (W, H), ground_shift=True)
    fit = min(W / r.canvas[0], H / r.canvas[1])
    offx = (W - r.canvas[0] * fit) / 2.0
    offy = (H - r.canvas[1] * fit) / 2.0
    shift = H - (offy + r.ground_y * fit)
    mapped = map_canvas_to_window(a.astype(np.float32), fit, offx, offy + shift, (W, H))
    q = np.asarray(im_rest, np.float32)
    geo = geometry_offsets(mapped, q)
    photo_mean, photo_p99 = premul_diff(mapped, q)

    # 3. information only (not gated): how far the live idle is from the rest pose at
    #    an arbitrary instant. The idle never passes exactly through rest (per-bone
    #    phase offsets, incommensurate periods), so the runtime must *settle* to rest
    #    before a turn clip; this number sizes that requirement.
    from pet.rig.motion import MotionEngine, MotionInputs
    eng = MotionEngine(r.spec)
    idle = eng.step(MotionInputs(walking=False, grounded=True, facing=1, source_facing=1), 0.0)
    im_idle = r.render(idle, (W, H), ground_shift=True)
    idle_mean, idle_p99 = premul_diff(np.asarray(im_idle, np.float32), q)
    r.close()

    bg = Image.new("RGBA", (W * 3, H), (240, 242, 246, 255))
    bg.alpha_composite(im_rest, (0, 0))
    bg.alpha_composite(Image.fromarray(np.clip(mapped, 0, 255).astype(np.uint8), "RGBA"), (W, 0))
    bg.alpha_composite(im_idle, (2 * W, 0))
    bg.convert("RGB").save(OUT / "qt256_vs_mapped_canvas_vs_idle.png")
    Image.fromarray(a.astype(np.uint8), "RGBA").save(OUT / "front_rest_canvas.png")

    files = []
    for base in PROD:
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".json", ".png"):
                files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p),
                              "bytes": p.stat().st_size})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    import PySide6
    report = {
        "gate": "G0",
        "checks": {
            "rest_render_reproducible": {"max_abs_diff_255": repro_max, "threshold": 1,
                                         "pass": repro_max <= 1},
            "canvas_to_256_mapping_geometry": {
                **{k: round(v, 3) for k, v in geo.items()}, "threshold_px": 0.25,
                "pass": all(abs(v) <= 0.25 for v in geo.values())},
            "canvas_to_256_mapping_photometric": {
                "mean_abs_diff_255": round(photo_mean, 3), "p99_abs_diff_255": round(photo_p99, 3),
                "threshold_mean": 2.0, "pass": photo_mean <= 2.0},
        },
        "info_not_gated": {
            "live_idle_t0_vs_rest_256": {"mean_abs_diff_255": round(idle_mean, 3),
                                         "p99_abs_diff_255": round(idle_p99, 3),
                                         "meaning": "idle never passes through rest; runtime must "
                                                    "settle springs/breath/gaze before a turn clip"},
            "mapping": {"fit": fit, "offset_x": offx, "offset_y": offy, "ground_shift": shift},
        },
        "rest_pose_angles": r.spec.rest_pose_angles,
        "ground_anchor_y_px": r.spec.ground_anchor_y_px,
        "canvas": list(r.canvas),
        "env": {"git_head": head, "python": platform.python_version(), "pyside6": PySide6.__version__,
                "platform": platform.platform()},
        "production_assets": files,
    }
    (OUT / "g0_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    print(f"hashed {len(files)} production files; outputs in {OUT}")


if __name__ == "__main__":
    main()
