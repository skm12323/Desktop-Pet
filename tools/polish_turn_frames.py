"""Remove cyan matting fringes and gently sharpen current ADULT turn packages.

Only RGB changes: alpha, frame geometry, timing and the endpoint/morph frames
remain intact. The function is also used by process_turn_clip for future builds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def polish_rgba(rgba: np.ndarray, nominal_height: int = 512):
    out = rgba.copy()
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3]
    distance = ndimage.distance_transform_edt(alpha > 8)
    band = max(2.0, nominal_height / 256.0)
    cyan = (rgb[..., 1] > .82 * rgb[..., 2]) & (rgb[..., 2] > 70) & (rgb[..., 0] < .55 * rgb[..., 2])
    suspect = cyan & (alpha > 8) & (distance <= band + 1)
    reliable = (alpha >= 245) & (distance > band) & ~cyan
    repaired = np.zeros(alpha.shape, bool)
    if reliable.any():
        dist, (iy, ix) = ndimage.distance_transform_edt(~reliable, return_indices=True)
        anchor = rgb[iy, ix]
        blue_anchor = (anchor[..., 2] > 1.08 * anchor[..., 1]) & (anchor[..., 2] > 1.2 * anchor[..., 0])
        repaired = suspect & blue_anchor & (dist <= band * 4)
        rgb[repaired] = anchor[repaired]
    # Local-range-clamped unsharp mask improves line contrast without producing
    # new bright/dark halos, especially on white cloth next to transparency.
    blur = ndimage.gaussian_filter(rgb, sigma=(.65 * nominal_height / 512, .65 * nominal_height / 512, 0))
    detail = rgb - blur
    low = ndimage.minimum_filter(rgb, size=(3, 3, 1))
    high = ndimage.maximum_filter(rgb, size=(3, 3, 1))
    sharpen = np.clip(rgb + .4 * detail, low, high)
    interior = (alpha >= 245) & (distance > band) & (np.max(np.abs(detail), axis=-1) > 3)
    rgb[interior] = sharpen[interior]
    out[..., :3] = np.clip(rgb + .5, 0, 255).astype(np.uint8)
    assert np.array_equal(out[..., 3], rgba[..., 3])
    after = out[..., :3].astype(float)
    remain = suspect & (after[..., 1] > .82 * after[..., 2]) & (after[..., 0] < .55 * after[..., 2])
    return out, {"cyan_edge_before_px": int(suspect.sum()), "cyan_edge_after_px": int(remain.sum()),
                 "repaired_px": int(repaired.sum()), "sharpened_px": int(interior.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="assets/rig_adult_walk_v1/clips")
    ap.add_argument("--qa-out", default="spikes/_qa/adult_visual_round2_2026-09-30/turn_polish")
    args = ap.parse_args()
    root, qa = Path(args.root), Path(args.qa_out)
    qa.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in ("turn_front_to_side", "turn_side_to_front", "turn_front_to_side_h256", "turn_side_to_front_h256"):
        pkg = root / name
        meta = json.loads((pkg / "clip.json").read_text())
        if meta.get("postprocess", {}).get("version") == 1:
            print(f"[SKIP] {name}: already polished")
            continue
        keep = int(meta.get("morph_frames", 3)) + int(meta.get("hold_frames", 3)) + 1
        protected = {f["file"] for f in meta["frames"][:keep] + meta["frames"][-keep:]}
        height = int(meta["space"]["size_px"][1])
        picks = np.linspace(keep, len(meta["frames"]) - keep - 1, 5).round().astype(int)
        originals = {int(i): Image.open(pkg / meta["frames"][int(i)]["file"]).convert("RGBA") for i in picks}
        count = {"cyan_edge_before_px": 0, "cyan_edge_after_px": 0, "repaired_px": 0, "sharpened_px": 0}
        hashes = {}
        for filename in sorted({f["file"] for f in meta["frames"]} - protected):
            p = pkg / filename
            a = np.asarray(Image.open(p).convert("RGBA"))
            fixed, stat = polish_rgba(a, height)
            for key in count:
                count[key] += stat[key]
            Image.fromarray(fixed).save(p, optimize=True)
            hashes[filename] = hashlib.sha256(p.read_bytes()).hexdigest()
        canvas = Image.new("RGB", (220 * 5, 390 * 2), (32, 36, 45))
        for j, i in enumerate(picks):
            i = int(i)
            after = Image.open(pkg / meta["frames"][i]["file"]).convert("RGBA")
            for row, im in enumerate((originals[i], after)):
                im.thumbnail((220, 356), Image.Resampling.LANCZOS)
                bg = Image.new("RGBA", (220, 390), (32, 36, 45, 255))
                bg.alpha_composite(im, ((220-im.width)//2, 28))
                ImageDraw.Draw(bg).text((8, 7), f"{'before' if row == 0 else 'after'} frame {i}", fill="white")
                canvas.paste(bg.convert("RGB"), (j*220, row*390))
        canvas.save(qa / f"{name}_before_after.png")
        meta["postprocess"] = {"tool": "tools/polish_turn_frames.py", "version": 1,
                               "alpha_unchanged": True, "protected_endpoint_files": sorted(protected),
                               "changed_file_sha256": hashes, **count}
        meta["qa"]["disk_bytes"] = sum(p.stat().st_size for p in (pkg / "frames").glob("*.png"))
        (pkg / "clip.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        reports.append({"package": name, **count})
        print(json.dumps(reports[-1]), flush=True)
    (qa / "results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
