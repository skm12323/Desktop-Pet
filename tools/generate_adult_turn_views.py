"""
Dedicated tool for generating adult turning view reference candidates (G1).
Connects to ComfyUI (http://127.0.0.1:8188) and runs Qwen-Image-2.1 with
strict isolation to assets/rig_adult_turn_v1/candidates/{view}/.
Never touches production layer assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = r"D:\AI\ComfyUI\input"
COMFY_OUTPUT = r"D:\AI\ComfyUI\output"

ROOT = Path(__file__).resolve().parents[1]
TURN_DIR = ROOT / "assets" / "rig_adult_turn_v1"

PROMPT_TEMPLATES = {
    "right45": {
        "prompt": (
            "Using the exact same character from <image1>, maintain her identical facial features, blue eyes, "
            "dark blue hair, white frilled maid headdress, ear fins, navy blue maid dress with gold filigree and ribbons, "
            "white frilled bib apron with blue whale embroidery, white socks, blue shoes, and large blue whale tail. "
            "Turn the character in place to a three-quarters view facing right (yaw ~45 degrees). "
            "Maintain the exact same relaxed idle upright standing posture and gentle pleasant expression, no walking. "
            "Both feet must touch the ground evenly on the same horizontal plane. "
            "Keep the exact same character proportions, scale, camera height, and lighting as reference. "
            "Properly reveal newly visible body and clothing angles while keeping the whale tail visible on the right. "
            "Render full body from headdress to shoes with pure white background, no cropping of tail, feet, or hair."
        ),
        "negative": (
            "different character, walking, dynamic pose, crossed legs, bent knees, floating, blurry, lowres, "
            "deformed anatomy, missing limbs, duplicate tail, cropped feet, cropped head, dark background"
        ),
        "default_seeds": [20260925, 20260926, 20260927, 20260928]
    },
    "right20": {
        "prompt": (
            "Using the exact same character from <image1>, maintain her identical facial features, blue eyes, "
            "dark blue hair, white frilled maid headdress, ear fins, navy blue maid dress with gold filigree and ribbons, "
            "white frilled bib apron with blue whale embroidery, white socks, blue shoes, and large blue whale tail. "
            "Turn the character in place to a slight three-quarters view facing slightly right (yaw ~20 degrees), "
            "subtly intermediate between front view and 45 degree turn. "
            "Maintain the exact same relaxed idle upright standing posture and gentle pleasant expression, no walking. "
            "Both feet touch the ground evenly on the same horizontal plane. "
            "Keep the exact same character proportions, scale, camera height, and lighting as reference. "
            "Render full body from headdress to shoes with pure white background, no cropping of tail, feet, or hair."
        ),
        "negative": (
            "different character, walking, dynamic pose, crossed legs, bent knees, floating, blurry, lowres, "
            "deformed anatomy, missing limbs, duplicate tail, cropped feet, cropped head, dark background"
        ),
        "default_seeds": [20260925, 20260926, 20260927, 20260928]
    }
}


def check_comfyui() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def queue_qwen_job(image_filename: str, prompt_text: str, neg_prompt: str, prefix: str, seed: int) -> str:
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt_text,
                "negative_prompt": neg_prompt,
                "resolution": 0,
                "vae": ["3", 0],
                "images.image_1": ["5", 0]
            }
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": ["4", 2],
                "seed": seed,
                "steps": 25,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}}
    }
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=json.dumps({"prompt": prompt}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return res["prompt_id"]


def wait_for_job(prompt_id: str, timeout_s: int = 600) -> str:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            h = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=10).read())
            if prompt_id in h:
                outputs = h[prompt_id].get("outputs", {})
                for out in outputs.values():
                    for img in out.get("images", []):
                        return img["filename"]
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Job {prompt_id} timed out after {timeout_s}s")


def main():
    parser = argparse.ArgumentParser(description="Generate turning view reference candidates for Adult stage (G1).")
    parser.add_argument("--view", choices=["right45", "right20"], default="right45", help="View angle to generate (default: right45)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list (default: 4 curated seeds)")
    parser.add_argument("--input-ref", type=str, default=str(ROOT / "assets/reference/adult_ref.png"), help="Path to input reference")
    parser.add_argument("--steps", type=int, default=25, help="Sampling steps")
    args = parser.parse_args()

    if not check_comfyui():
        print(f"Error: ComfyUI server is not reachable at {COMFY_URL}. Start ComfyUI first.")
        sys.exit(1)

    ref_path = Path(args.input_ref).resolve()
    if not ref_path.is_file():
        print(f"Error: reference file {ref_path} not found.")
        sys.exit(1)

    view_cfg = PROMPT_TEMPLATES[args.view]
    seeds = args.seeds or view_cfg["default_seeds"]
    out_dir = TURN_DIR / "candidates" / args.view
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute ref hash
    ref_bytes = ref_path.read_bytes()
    ref_hash = hashlib.sha256(ref_bytes).hexdigest()

    # Copy input to ComfyUI input folder
    comfy_in_name = f"turn_input_adult_{args.view}_front.png"
    shutil_dest = Path(COMFY_INPUT) / comfy_in_name
    shutil_dest.write_bytes(ref_bytes)

    manifest_file = out_dir / "generation_manifest.json"
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "view": args.view,
        "ref_input": str(ref_path),
        "ref_sha256": ref_hash,
        "prompt": view_cfg["prompt"],
        "negative_prompt": view_cfg["negative"],
        "candidates": []
    }

    print(f"\n=======================================================")
    print(f"  Adult Turn Candidate Generator (G1) -> View: {args.view}")
    print(f"  Reference SHA-256: {ref_hash[:16]}...")
    print(f"  Seeds to run: {seeds}")
    print(f"  Output directory: {out_dir}")
    print(f"=======================================================\n")

    for i, seed in enumerate(seeds):
        prefix = f"adult_turn_{args.view}_s{seed}"
        print(f"[{i+1}/{len(seeds)}] Queuing seed {seed}...")
        prompt_id = queue_qwen_job(
            image_filename=comfy_in_name,
            prompt_text=view_cfg["prompt"],
            neg_prompt=view_cfg["negative"],
            prefix=prefix,
            seed=seed
        )
        print(f"  Job queued with ID {prompt_id}, generating...")
        gen_filename = wait_for_job(prompt_id)
        print(f"  [OK] Generated {gen_filename}")

        # Copy to candidates directory
        src_path = Path(COMFY_OUTPUT) / gen_filename
        dest_filename = f"candidate_s{seed}.png"
        dest_path = out_dir / dest_filename
        dest_path.write_bytes(src_path.read_bytes())

        manifest["candidates"].append({
            "seed": seed,
            "job_id": prompt_id,
            "raw_output": gen_filename,
            "candidate_file": dest_filename,
            "candidate_sha256": hashlib.sha256(dest_path.read_bytes()).hexdigest(),
            "status": "generated"
        })

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Generation complete. Manifest written to {manifest_file}")


if __name__ == "__main__":
    main()
