"""Constrain old FINAL paperdoll extensions to the core that hides them at rest.

Default is dry-run. --apply clips hidden extensions to the source silhouette,
restores native source colours and corrects compositing opacity. Canvas
rectangles and pivots stay the same. Extensions outside the core were visible
as rectangular colour streaks in neglected_neutral even at 0 deg.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=ROOT / "assets/rig/final")
    ap.add_argument("--figure", default="neglected_neutral")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    core = np.asarray(Image.open(args.package / manifest["figures"][args.figure]).convert("RGBA"))
    source = Image.open(ROOT / "assets/ai" / f"final_{args.figure}.png").convert("RGBA")
    source_rgba = np.asarray(source)
    extension_cover = np.zeros(core.shape[:2], bool)
    under = Image.new("RGBA", source.size)
    native_parts = []
    for part in manifest["parts"]:
        rows = int(part.get("split", {}).get("extended_up", 0))
        if part["source_figure"] != args.figure or not rows:
            continue
        path = args.package / part["file"]
        rgba = np.asarray(Image.open(path).convert("RGBA")).copy()
        rows = min(rows, rgba.shape[0])
        x0, y0, x1, _ = part["px_rect"]
        hidden = source_rgba[y0:y0 + rows, x0:x1, 3] >= 240
        if hidden.shape != rgba[:rows, :, 3].shape:
            raise ValueError(f"extension outside core canvas: {part['id']}")
        remove = ~hidden & (rgba[:rows, :, 3] > 0)
        count = int(remove.sum())
        rgba[:rows][remove] = 0
        extension_cover[y0:y0 + rows, x0:x1] |= hidden & (rgba[:rows, :, 3] > 0)
        if args.apply and count:
            Image.fromarray(rgba).save(path)
        print(f"{part['id']}: {count} visible extension pixels {'removed' if args.apply else 'to remove'}")

    # Feathered cuts multiplied source alpha on both sides of a seam, making
    # alpha compositing lose opacity. Solve the remaining core from the source
    # and the repaired underlay, rather than overlapping two feathered copies.
    for part in manifest["parts"]:
        if (part["source_figure"] == args.figure and part["z"] == "under_core"
                and part.get("kind") != "blink"):
            path = args.package / part["file"]
            rgba = np.asarray(Image.open(path).convert("RGBA")).copy()
            x0, y0, x1, y1 = part["px_rect"]
            rows = int(part.get("split", {}).get("extended_up", 0))
            src = source_rgba[y0:y1, x0:x1]
            # Native pixels come straight from the original artwork; only the
            # hidden generated extension keeps its completion colours.
            rgba[rows:, :, :3] = src[rows:, :, :3]
            rgba[rows:, :, 3] = np.minimum(rgba[rows:, :, 3], src[rows:, :, 3])
            rgba[rgba[:, :, 3] == 0] = 0
            native_parts.append((path, (x0, y0, x1, y1), rgba))
    # Keep overlapping components, but budget their combined alpha against
    # the source silhouette. AA edges must not become opaque after stacking.
    target = source_rgba[:, :, 3].astype(np.float32) / 255
    target[extension_cover] = np.minimum(target[extension_cover], .2)
    alphas = []
    for _, (x0, y0, x1, y1), rgba in native_parts:
        alpha = np.zeros(target.shape, np.float32)
        alpha[y0:y1, x0:x1] = rgba[:, :, 3].astype(np.float32) / 255
        alphas.append(alpha)
    def combined(scale):
        remaining = np.ones(target.shape, np.float32)
        for alpha in alphas:
            remaining *= 1 - alpha * scale
        return 1 - remaining
    excess = combined(1) > target + 1e-6
    low, high = np.zeros(target.shape, np.float32), np.ones(target.shape, np.float32)
    for _ in range(16):
        mid = (low + high) / 2
        above = combined(mid) > target
        high = np.where(above, mid, high)
        low = np.where(above, low, mid)
    scale = np.where(excess, low, 1)
    for path, (x0, y0, x1, y1), rgba in native_parts:
        rgba[:, :, 3] = np.clip(rgba[:, :, 3]*scale[y0:y1, x0:x1]+.5, 0, 255).astype(np.uint8)
        rgba[rgba[:, :, 3] == 0] = 0
        im = Image.fromarray(rgba)
        if args.apply:
            im.save(path)
        under.alpha_composite(im, (x0, y0))
    s, u = np.asarray(source, dtype=float) / 255, np.asarray(under, dtype=float) / 255
    sa, ua = s[:, :, 3:], u[:, :, 3:]
    ca = np.clip((sa - ua) / np.maximum(1 - ua, 1e-9), 0, 1)
    rgb = (s[:, :, :3] * sa - u[:, :, :3] * ua * (1 - ca)) / np.maximum(ca, 1e-9)
    corrected = np.clip(np.dstack([rgb, ca]) * 255 + .5, 0, 255).astype(np.uint8)
    corrected[corrected[:, :, 3] == 0] = 0
    if args.apply:
        Image.fromarray(corrected).save(args.package / manifest["figures"][args.figure])
    print("core cut opacity corrected" if args.apply else "core cut opacity to correct")


if __name__ == "__main__":
    main()
