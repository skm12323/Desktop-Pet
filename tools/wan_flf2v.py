"""Local Wan 2.2 14B first+last-frame video via ComfyUI (official template, 4-step LightX2V path).

Graph = ComfyUI's bundled video_wan2_2_14B_flf2v template (fp8 high/low-noise experts,
umt5 text encoder, wan 2.1 VAE, ModelSamplingSD3 shift 5, 2+2 steps cfg 1) but frames are
saved as lossless PNGs (better for keying) and then encoded to a near-lossless MP4.
Writes a JSON record (inputs + sha256, prompt, seed, size, length) next to the MP4.
Run one job at a time (no parallel generation).

Usage:
  D:\\anaconda3\\python.exe -X utf8 tools/wan_flf2v.py --first F.png [--last L.png] \\
      --prompt "..." --out assets/rig_adult_walk_v1/clips/raw/wan/try1.mp4 --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = Path(r"D:\AI\ComfyUI\input")
COMFY_OUTPUT = Path(r"D:\AI\ComfyUI\output")
M = {"high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
     "low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
     "lora_high": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
     "lora_low": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
     "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "vae": "wan_2.1_vae.safetensors"}
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
       "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
       "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{COMFY_URL}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    body = urllib.request.urlopen(req, timeout=30).read()
    return json.loads(body) if body else {}


def graph(first: str, last: str | None, prompt: str, negative: str, w: int, h: int,
          length: int, seed: int, prefix: str) -> dict:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": M["high"], "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": M["low"], "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": M["lora_high"], "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": M["lora_low"], "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 5.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": 5.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": M["clip"], "type": "wan", "device": "default"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": M["vae"]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": negative}},
        "11": {"class_type": "LoadImage", "inputs": {"image": first}},
        "13": {"class_type": "WanFirstLastFrameToVideo",
               "inputs": {"positive": ["9", 0], "negative": ["10", 0], "vae": ["8", 0],
                          "width": w, "height": h, "length": length, "batch_size": 1,
                          "start_image": ["11", 0]}},
        "14": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["5", 0], "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "positive": ["13", 0],
                          "negative": ["13", 1], "latent_image": ["13", 2], "start_at_step": 0,
                          "end_at_step": 2, "return_with_leftover_noise": "enable"}},
        "15": {"class_type": "KSamplerAdvanced",
               "inputs": {"model": ["6", 0], "add_noise": "disable", "noise_seed": 0, "steps": 4, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "positive": ["13", 0],
                          "negative": ["13", 1], "latent_image": ["14", 0], "start_at_step": 2,
                          "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["8", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["16", 0]}},
    }
    if last:
        g["12"] = {"class_type": "LoadImage", "inputs": {"image": last}}
        g["13"]["inputs"]["end_image"] = ["12", 0]
    return g


def run(first: Path, last: Path | None, prompt: str, out: Path, seed: int = 1, w: int = 480,
        h: int = 848, length: int = 81, fps: int = 16, negative: str = NEG, timeout_s: int = 3600) -> dict:
    q = json.loads(urllib.request.urlopen(f"{COMFY_URL}/queue", timeout=10).read())
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit("ComfyUI queue is busy: run generations one at a time")
    tag = uuid.uuid4().hex[:8]
    names = {}
    for role, p in (("first", first), ("last", last)):
        if p:
            n = f"claude_wan_{tag}_{role}.png"
            Image.open(p).convert("RGB").save(COMFY_INPUT / n)
            names[role] = n
    prefix = f"claude_wan/{out.stem}_{tag}"
    pid = _post("/prompt", {"prompt": graph(names["first"], names.get("last"), prompt, negative,
                                               w, h, length, seed, prefix)})["prompt_id"]
    t0 = time.time()
    frames = None
    while time.time() - t0 < timeout_s:
        try:
            hist = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}", timeout=10).read())
        except Exception:
            hist = {}
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st)[:2000])
            imgs = [i for o in hist[pid].get("outputs", {}).values() for i in o.get("images", [])]
            if imgs:
                frames = [COMFY_OUTPUT / i.get("subfolder", "") / i["filename"] for i in imgs]
                break
        time.sleep(5)
    if not frames:
        raise TimeoutError(pid)
    out.parent.mkdir(parents=True, exist_ok=True)
    fdir = out.with_suffix("")
    fdir.mkdir(exist_ok=True)
    for k, f in enumerate(sorted(frames)):
        Image.open(f).save(fdir / f"{k:04d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(fdir / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "10", str(out)], check=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rec = {"tool": "tools/wan_flf2v.py", "models": M, "comfy_prompt_id": pid, "seconds": round(time.time() - t0, 1),
           "seed": seed, "size": [w, h], "length": length, "fps": fps, "steps": "2 high + 2 low (lightx2v 4-step)",
           "first": {"path": first.as_posix(), "sha256": sha(first)},
           "last": {"path": last.as_posix(), "sha256": sha(last)} if last else None,
           "prompt": prompt, "negative": negative, "frames_dir": fdir.as_posix(), "output": out.as_posix()}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", default="")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=848)
    ap.add_argument("--length", type=int, default=81)
    ap.add_argument("--negative-extra", default="", help="appended to the default Chinese negative prompt")
    a = ap.parse_args()
    r = run(Path(a.first), Path(a.last) if a.last else None, a.prompt, Path(a.out), a.seed, a.width, a.height, a.length,
            negative=NEG + ("，" + a.negative_extra if a.negative_extra else ""))
    print(json.dumps({k: r[k] for k in ("output", "seconds", "seed")}, indent=2))


if __name__ == "__main__":
    main()
