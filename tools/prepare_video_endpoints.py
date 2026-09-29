"""G2: canvas-registered RGBA (960x1696) -> 1080x1920 chroma-green first/last frames.

Every endpoint uses the identical canvas<->video mapping (see render_rig_rest.video_endpoint),
so frames of the generated clip can be registered back onto the rig canvas.

Usage:
  D:\\anaconda3\\python.exe -X utf8 tools/prepare_video_endpoints.py \\
      --first assets/rig_adult_walk_v1/references/front_rest.png \\
      --last  assets/rig_adult_walk_v1/references/side_key.png \\
      --out-dir assets/rig_adult_walk_v1/clips/endpoints/front_to_side
Writes first.png / last.png (+ mapping.json) and a side-by-side preview.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from render_rig_rest import video_endpoint

GROUND_Y = 1608.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--margin-x", type=int, default=160, help="canvas px of frame margin left (and right unless --margin-right)")
    ap.add_argument("--margin-right", type=int, default=40, help="canvas px of frame margin right (tail swings left)")
    ap.add_argument("--floor-margin", type=int, default=100, help="canvas px between soles and frame bottom")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    frames = []
    for role, src in (("first", a.first), ("last", a.last)):
        if not src:
            continue
        im = Image.open(src).convert("RGBA")
        if im.size != (960, 1696):
            raise SystemExit(f"{src} is {im.size}; endpoints must be canvas-registered 960x1696 RGBA")
        frame, m = video_endpoint(im, GROUND_Y, margin_x=a.margin_x, floor_margin=a.floor_margin,
                                  margin_right=a.margin_right)
        frame.save(out / f"{role}.png")
        m.update({"source": Path(src).as_posix(),
                  "source_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest()})
        meta[role] = m
        frames.append(frame)
    (out / "mapping.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    prev = Image.new("RGB", (540 * len(frames), 960))
    for k, f in enumerate(frames):
        prev.paste(f.resize((540, 960), Image.Resampling.LANCZOS), (k * 540, 0))
    prev.save(out / "preview.jpg", quality=90)
    print(json.dumps({k: {"source": v["source"]} for k, v in meta.items()}, indent=2))


if __name__ == "__main__":
    main()
