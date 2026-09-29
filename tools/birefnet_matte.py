"""Matte an illustration on a plain light background into RGBA via local ComfyUI + BiRefNet.

Uses the ComfyUI-BiRefNet-Hugo node with its bundled local weights (no download).
The soft mask is lightly sharpened and the white background is un-mixed from edge
pixels (colour decontamination), so edges do not keep a white halo.

Usage: D:\\anaconda3\\python.exe -X utf8 tools/birefnet_matte.py IN.png OUT.png [--bg 255,255,255]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = Path(r"D:\AI\ComfyUI\input")
COMFY_OUTPUT = Path(r"D:\AI\ComfyUI\output")
# HF-format weights shipped inside the node folder (the node's own default path is wrong)
LOCAL_WEIGHTS = r"D:\AI\ComfyUI\custom_nodes\ComfyUI-BiRefNet-Hugo\ComfyUI\models\BiRefNet"


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    body = urllib.request.urlopen(req, timeout=30).read()
    return json.loads(body) if body else {}


def birefnet_mask(src: Path, timeout_s: int = 300) -> np.ndarray:
    """Returns the BiRefNet mask as float32 [0, 1] at the source resolution."""
    tag = uuid.uuid4().hex[:8]
    name = f"claude_matte_{tag}.png"
    Image.open(src).convert("RGB").save(COMFY_INPUT / name)
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": name}},
        "2": {"class_type": "BiRefNet_Hugo",
              "inputs": {"image": ["1", 0], "model": "ZhengPeng7/BiRefNet", "load_local_model": True,
                         "local_model_path": LOCAL_WEIGHTS,
                         "background_color_name": "transparency", "device": "auto"}},
        "3": {"class_type": "MaskToImage", "inputs": {"mask": ["2", 1]}},
        "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"claude_matte/{tag}", "images": ["3", 0]}},
    }
    pid = _post("/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=10).read())
        except Exception:
            h = {}
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st)[:1500])
            for out in h[pid].get("outputs", {}).values():
                for img in out.get("images", []):
                    p = COMFY_OUTPUT / img.get("subfolder", "") / img["filename"]
                    return np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
        time.sleep(1)
    raise TimeoutError(pid)


def matte(src: Path, dst: Path, bg=(255, 255, 255)) -> dict:
    rgb = np.asarray(Image.open(src).convert("RGB"), np.float32)
    m = birefnet_mask(src)
    a = np.clip((m - 0.08) / 0.84, 0.0, 1.0)               # sharpen the soft transition a little
    bgc = np.array(bg, np.float32)
    fg = (rgb - (1.0 - a[..., None]) * bgc) / np.maximum(a[..., None], 1e-3)   # un-mix background
    fg = np.where(a[..., None] > 0.02, np.clip(fg, 0, 255), 0.0)
    out = np.dstack([fg, a * 255.0]).round().astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(dst)
    return {"src": src.as_posix(), "dst": dst.as_posix(), "opaque_px": int((a > 0.5).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bg", default="255,255,255", help="background colour to un-mix")
    a = ap.parse_args()
    print(json.dumps(matte(Path(a.src), Path(a.dst), tuple(int(v) for v in a.bg.split(","))), indent=2))


if __name__ == "__main__":
    main()
