"""Reference-first Qwen-Image-2.1 edit via the local ComfyUI API (http://127.0.0.1:8188).

Images are passed in order as <image1>, <image2>, ... (up to 16). The output takes
the size of image 1 (TextEncodeQwenImage21 latent), so put the image being edited
/ the layout reference first and identity references after it. Every run writes a
JSON record next to the output: input paths + sha256, prompt, negative, seed,
steps, model files, ComfyUI prompt id.

Usage (repo root):
  D:\\anaconda3\\python.exe -X utf8 tools/qwen_edit.py \\
      --image pose.png --image front_rest.png \\
      --prompt "Redraw <image1> ... identity from <image2> ..." \\
      --out assets/rig_adult_walk_v1/references/qwen_raw/try1.png --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = Path(r"D:\AI\ComfyUI\input")
COMFY_OUTPUT = Path(r"D:\AI\ComfyUI\output")
MODELS = {"unet": "qwen_image_2.1_int8_convrot.safetensors",
          "clip": "qwen3vl_8b_w4a8.safetensors",
          "vae": "qwen_image_2.1_vae_bf16.safetensors"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_graph(image_names: list[str], prompt: str, negative: str, seed: int, steps: int,
                prefix: str, resolution: int, mask_name: str = "") -> dict:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "qwen_image",
                                                      "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": negative,
                         "resolution": resolution, "vae": ["3", 0]}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["4", 2], "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    for i, name in enumerate(image_names, start=1):
        node = str(100 + i)
        g[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["4"]["inputs"][f"images.image_{i}"] = [node, 0]
    if mask_name:
        # masked inpaint: sample only under the mask, starting from image 1's own latent
        g["20"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["101", 0], "vae": ["3", 0]}}
        g["21"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
        g["22"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["20", 0], "mask": ["21", 0]}}
        g["7"]["inputs"]["latent_image"] = ["22", 0]
    return g


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    body = urllib.request.urlopen(req, timeout=30).read()
    return json.loads(body) if body else {}


def _queue_and_wait(graph: dict, timeout_s: int) -> tuple[str, Path]:
    pid = _post("/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=10).read())
        except Exception:
            h = {}
        if pid in h:
            status = h[pid].get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {json.dumps(status)[:2000]}")
            for node_out in h[pid].get("outputs", {}).values():
                for img in node_out.get("images", []):
                    return pid, COMFY_OUTPUT / img.get("subfolder", "") / img["filename"]
        time.sleep(2)
    raise TimeoutError(f"no output for prompt {pid} after {timeout_s}s")


def run(images: list[Path], prompt: str, negative: str, out: Path, seed: int = 1,
        steps: int = 25, resolution: int = 0, timeout_s: int = 1500,
        flatten: bool = True, mask: Path | None = None) -> dict:
    """mask (white = repaint, image 1 size): masked inpaint; outside the mask the result is
    replaced by image 1's own pixels (feathered 2 px), so only the masked area changes."""
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    names = []
    for i, p in enumerate(images, start=1):
        name = f"claude_qe_{tag}_{i}.png"
        im = Image.open(p)
        if flatten and im.mode in ("RGBA", "LA", "P"):
            # transparent refs reach the VAE as raw RGBA and turn the background to noise
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            im = bg.convert("RGB")
        im.save(COMFY_INPUT / name)
        names.append(name)
    mask_name = ""
    if mask is not None:
        mask_name = f"claude_qe_{tag}_mask.png"
        Image.open(mask).convert("L").convert("RGB").save(COMFY_INPUT / mask_name)
    prefix = f"claude_qe/{out.stem}_{tag}"
    graph = build_graph(names, prompt, negative, seed, steps, prefix, resolution, mask_name)
    t0 = time.time()
    try:
        pid, produced = _queue_and_wait(graph, timeout_s)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        _post("/free", {"unload_models": True, "free_memory": True})   # drop cached models, retry once
        time.sleep(5)
        pid, produced = _queue_and_wait(graph, timeout_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, out)
    if mask is not None:
        from PIL import ImageFilter
        src = Image.open(COMFY_INPUT / names[0]).convert("RGB")
        gen = Image.open(out).convert("RGB").resize(src.size, Image.Resampling.LANCZOS)
        m = Image.open(mask).convert("L").filter(ImageFilter.GaussianBlur(1.0))
        Image.composite(gen, src, m).save(out)
    record = {
        "tool": "tools/qwen_edit.py", "models": MODELS, "comfy_prompt_id": pid,
        "seconds": round(time.time() - t0, 1), "seed": seed, "steps": steps, "resolution": resolution,
        "flatten_alpha_onto_white": flatten,
        "images": [{"slot": f"image{i}", "path": p.as_posix(), "sha256": sha256(p)}
                   for i, p in enumerate(images, start=1)],
        "mask": mask.as_posix() if mask is not None else None,
        "prompt": prompt, "negative": negative, "output": out.as_posix(), "output_sha256": sha256(out),
    }
    out.with_suffix(".json").write_bytes(json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", action="append", required=True, help="reference image, in order (repeat)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="blurry, lowres, deformed, extra limbs, text, watermark")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--resolution", type=int, default=0, help="0 = keep reference sizes (x32)")
    ap.add_argument("--keep-alpha", action="store_true", help="do not flatten transparent refs onto white")
    ap.add_argument("--mask", default="", help="inpaint mask for image 1 (white = repaint)")
    a = ap.parse_args()
    rec = run([Path(p) for p in a.image], a.prompt, a.negative, Path(a.out), a.seed, a.steps,
              a.resolution, flatten=not a.keep_alpha, mask=Path(a.mask) if a.mask else None)
    print(json.dumps({k: rec[k] for k in ("output", "seconds", "seed")}, indent=2))


if __name__ == "__main__":
    main()
