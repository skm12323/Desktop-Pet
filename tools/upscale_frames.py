"""Anime super-resolution for generated clip frames (spandrel + Real-ESRGAN weights).

Frames are upscaled by the model's native factor (x4) and then resampled (Lanczos) to the
requested --scale, which gives a supersampled result. Run with ComfyUI's venv python
(torch + spandrel): D:\\AI\\ComfyUI\\venv\\Scripts\\python.exe -X utf8 tools/upscale_frames.py ...
Weights live in D:\\AI\\ComfyUI\\models\\upscale_models. One job at a time (GPU).

Usage:
  ... tools/upscale_frames.py --in frames_dir --out out_dir --model realesr-animevideov3 \
      --scale 2 [--frames 0,40,80]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

WEIGHTS = Path(r"D:\AI\ComfyUI\models\upscale_models")


def load(name: str):
    m = ModelLoader().load_from_file(str(WEIGHTS / f"{name}.pth"))
    assert isinstance(m, ImageModelDescriptor)
    return m.cuda().eval().half(), m.scale


@torch.no_grad()
def upscale(model, native: int, img: Image.Image, tile: int = 256, pad: int = 16) -> Image.Image:
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    x = x.cuda().half()
    _, _, h, w = x.shape
    out = torch.zeros(1, 3, h * native, w * native, device="cuda", dtype=torch.float16)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, h), min(x1 + pad, w)
            o = model(x[:, :, ya:yb, xa:xb])
            oy, ox = (y0 - ya) * native, (x0 - xa) * native
            out[:, :, y0 * native:y1 * native, x0 * native:x1 * native] = \
                o[:, :, oy:oy + (y1 - y0) * native, ox:ox + (x1 - x0) * native]
    arr = (out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="realesr-animevideov3")
    ap.add_argument("--scale", type=float, default=2.0, help="final factor vs the input (model native x4, then Lanczos)")
    ap.add_argument("--frames", default="", help="comma list of frame indices (default: all)")
    a = ap.parse_args()
    files = sorted(Path(a.src).glob("*.png"))
    if a.frames:
        files = [files[int(i)] for i in a.frames.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, native = load(a.model)
    t0 = time.time()
    for f in files:
        im = Image.open(f)
        big = upscale(model, native, im)
        size = (round(im.width * a.scale), round(im.height * a.scale))
        if big.size != size:
            big = big.resize(size, Image.LANCZOS)
        big.save(out / f.name)
    (out / "upscale.json").write_text(json.dumps({
        "tool": "tools/upscale_frames.py", "model": a.model, "native_scale": native, "final_scale": a.scale,
        "src": Path(a.src).as_posix(), "frames": len(files), "seconds": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    print(f"{len(files)} frames -> {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
