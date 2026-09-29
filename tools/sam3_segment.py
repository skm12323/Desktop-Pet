"""Point-prompted part masks with ComfyUI's native SAM 3.1 (SAM3_Detect), one job per call.

Each part is segmented on its own crop (SAM runs at 1008x1008, so cropping to the part keeps
mask edges at ~1-2 source px). The RGBA source is flattened on a flat colour first.

Parts file (JSON list), canvas pixel coordinates:
  [{"id": "arm_l", "crop": [x0, y0, x1, y1], "pos": [[x, y], ...], "neg": [[x, y], ...]},
   {"id": "hair", "crop": [...], "text": "hair", "threshold": 0.4}, ...]   (text = concept prompt)

Writes <out>/<id>.png (L, 255 = part, full canvas) and <out>/sam3_record.json.

Usage:
  D:\\anaconda3\\python.exe -X utf8 tools/sam3_segment.py --image IMG.png --parts parts.json --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = Path(r"D:\AI\ComfyUI\input")
COMFY_OUTPUT = Path(r"D:\AI\ComfyUI\output")
CKPT = "sam3.1_multiplex_fp16.safetensors"


def _get(path: str) -> dict:
    return json.loads(urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=30).read())


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def run(image: Path, parts: list[dict], out: Path, bg: tuple[int, int, int] = (255, 255, 255),
        refine: int = 3) -> dict:
    q = _get("/queue")
    if q["queue_running"] or q["queue_pending"]:
        raise SystemExit("ComfyUI queue busy - one generation job at a time")
    src = Image.open(image).convert("RGBA")
    flat = Image.new("RGBA", src.size, (*bg, 255))
    flat.alpha_composite(src)
    flat = flat.convert("RGB")
    tag = uuid.uuid4().hex[:8]
    graph: dict = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}}}
    for k, p in enumerate(parts):
        x0, y0, x1, y1 = p["crop"]
        name = f"sam3_{tag}_{p['id']}.png"
        flat.crop((x0, y0, x1, y1)).save(COMFY_INPUT / name)
        pos = [{"x": int(x - x0), "y": int(y - y0)} for x, y in p.get("pos", [])]
        neg = [{"x": int(x - x0), "y": int(y - y0)} for x, y in p.get("neg", [])]
        graph[f"load{k}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        det = {"model": ["ckpt", 0], "image": [f"load{k}", 0],
               "threshold": float(p.get("threshold", 0.5)), "refine_iterations": refine,
               "individual_masks": False}
        if p.get("text"):     # concept prompt (detector path); points are ignored for this part
            graph[f"txt{k}"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": p["text"]}}
            det["conditioning"] = [f"txt{k}", 0]
        else:
            det["positive_coords"] = json.dumps(pos)
            if neg:
                det["negative_coords"] = json.dumps(neg)
        graph[f"det{k}"] = {"class_type": "SAM3_Detect", "inputs": det}
        graph[f"img{k}"] = {"class_type": "MaskToImage", "inputs": {"mask": [f"det{k}", 0]}}
        graph[f"save{k}"] = {"class_type": "SaveImage",
                             "inputs": {"images": [f"img{k}", 0], "filename_prefix": f"sam3_{tag}_{p['id']}"}}
    t0 = time.time()
    pid = _post("/prompt", {"prompt": graph, "client_id": tag})["prompt_id"]
    while True:
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise SystemExit(json.dumps(st.get("messages", []))[:3000])
            outputs = h[pid]["outputs"]
            break
        time.sleep(1)
    out.mkdir(parents=True, exist_ok=True)
    W, H = src.size
    for k, p in enumerate(parts):
        fn = outputs[f"save{k}"]["images"][0]
        m = Image.open(COMFY_OUTPUT / fn.get("subfolder", "") / fn["filename"]).convert("L")
        x0, y0, x1, y1 = p["crop"]
        if m.size != (x1 - x0, y1 - y0):
            m = m.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
        full = Image.new("L", (W, H), 0)
        full.paste(m, (x0, y0))
        full.save(out / f"{p['id']}.png")
    rec = {"tool": "tools/sam3_segment.py", "model": CKPT, "comfy_prompt_id": pid,
           "seconds": round(time.time() - t0, 1), "image": image.as_posix(),
           "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "flatten_bg": list(bg),
           "refine_iterations": refine, "parts": parts}
    (out / "sam3_record.json").write_bytes(json.dumps(rec, indent=2).encode("utf-8"))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated part ids to (re)run")
    ap.add_argument("--refine", type=int, default=3)
    a = ap.parse_args()
    parts = json.loads(Path(a.parts).read_text(encoding="utf-8"))
    if a.only:
        keep = set(a.only.split(","))
        parts = [p for p in parts if p["id"] in keep]
    rec = run(Path(a.image), parts, Path(a.out), refine=a.refine)
    print(f"[OK] {len(parts)} masks in {rec['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
