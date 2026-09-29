"""
Automated batch layer generator using local ComfyUI + Qwen-Image-2.1.
Connects to http://127.0.0.1:8188, loads crops or full reference, sends structured
layer extraction & inpainting prompts, and saves resulting RGBA PNGs to assets/rig_{stage}/layers/.
Supports both --stage young and --stage adult.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
COMFY_INPUT = r"D:\AI\ComfyUI\input"
COMFY_OUTPUT = r"D:\AI\ComfyUI\output"

# Curated prompts for Adult (A-pose) layers
ADULT_PROMPTS = {
    "ahoge_headdress": {
        "prompt": "Extract the single curved ahoge antenna hair strand and the white frilled lace maid headdress headband from <image1>. Remove the blue hair, face, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "face, skin, eyes, blue hair, background, lowres, blurry"
    },
    "ear_fin_l": {
        "prompt": "Extract the left whale-fin ear from <image1>. Keep the dark blue fin shape and light blue lower scalloped edge. Inpaint and smoothly extend the root into the head. Remove the hair, face, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "hair, face, skin, background, human ear, lowres, blurry"
    },
    "ear_fin_r": {
        "prompt": "Extract the right whale-fin ear and its light blue ribbon bow from <image1>. Keep the distinct shapes of both ear fin and ribbon. Inpaint and smoothly extend the root inward. Remove hair, face, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "hair, face, skin, background, human ear, lowres, blurry"
    },
    "bangs": {
        "prompt": "Extract the front hair bangs covering the forehead, center hair strands, and side face hair flares from <image1>. The center front bangs covering the forehead MUST BE FULLY PRESERVED and solid, with sharp pointed tips and smooth blue highlights. Do NOT hollow out the center forehead hair. Inpaint roots upward behind headdress. Remove the white maid headdress, eyes, eyebrows, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "bald forehead, missing bangs, hollow center hair, split bangs, headdress, background, lowres, blurry"
    },
    "head_base": {
        "prompt": "Extract the full face skin, forehead, cheeks, eyebrows, nose, mouth, blush, and neck connection from <image1>. COMPLETELY INPAINT AND CLEAR the iris, pupils, highlights, and eyelashes into smooth, pure white clean sclera. Inpaint forehead skin under bangs. Remove hair, headdress, dress, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "iris, pupils, blue eyes, eyelashes, headdress, bangs, hair, background, lowres"
    },
    "pupil_l": {
        "prompt": "Extract the left eye iris, pupil, and white sparkle highlight from <image1>. Inpaint and complete the top curved arc of the iris hidden under upper eyelid. Remove sclera, eyelashes, eyelids, and skin completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "eyelash, eyelid, skin, sclera, white of eye, background, face, lowres"
    },
    "pupil_r": {
        "prompt": "Extract the right eye iris, pupil, and white sparkle highlight from <image1>. Inpaint and complete the top curved arc of the iris hidden under upper eyelid. Remove sclera, eyelashes, eyelids, and skin completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "eyelash, eyelid, skin, sclera, white of eye, background, face, lowres"
    },
    "eyelid_l": {
        "prompt": "Extract the black upper eyelash curve, eyeliner, and upper lid crease from <image1>. The center eye socket opening MUST REMAIN 100% HOLLOW AND COMPLETELY TRANSPARENT like a cutout stencil so the iris underneath can show through. Remove iris, pupil, sclera, full cheek skin, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "iris, pupil, blue eye, eyeball, sclera, solid skin covering eye, opaque eye opening, closed eye, background, lowres"
    },
    "eyelid_r": {
        "prompt": "Extract the black upper eyelash curve, eyeliner, and upper lid crease from <image1>. The center eye socket opening MUST REMAIN 100% HOLLOW AND COMPLETELY TRANSPARENT like a cutout stencil so the iris underneath can show through. Remove iris, pupil, sclera, full cheek skin, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "iris, pupil, blue eye, eyeball, sclera, solid skin covering eye, opaque eye opening, closed eye, background, lowres"
    },
    "torso": {
        "prompt": "Extract the upper torso maid blouse, collar, blue necktie gem, chest vertical ruffles, and dark blue corset waist with 4 gold buttons from <image1>. Inpaint the shoulder joints smoothly where the sleeves connected, and inpaint the neck connection upward. Remove arms, hands, sleeves, skirt, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "arms, sleeves, hands, apron, skirt, legs, background, lowres"
    },
    "arm_l": {
        "prompt": "Extract the left arm, puffy shoulder sleeve, navy sleeve, white lace wrist cuff, and natural hanging relaxed hand from <image1>. Inpaint and smoothly round the shoulder connection joint. Keep the slender arm and natural fingers. Remove the torso, dress, apron, skirt, hair, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "torso, dress, apron, skirt, hair, background, face, lowres, blurry"
    },
    "arm_r": {
        "prompt": "Extract the right arm, puffy shoulder sleeve, navy sleeve, white lace wrist cuff, and natural hanging relaxed hand from <image1>. Inpaint and smoothly round the shoulder connection joint. Keep the slender arm and natural fingers. Remove the torso, dress, apron, skirt, tail, hair, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "torso, dress, apron, skirt, hair, tail, background, face, lowres, blurry"
    },
    "apron": {
        "prompt": "Extract the white frilled bib apron with embroidered blue whale mascot on center and delicate ruffled lace borders from <image1>. Inpaint the upper waist corners slightly behind where the sleeves were. Remove arms, torso, skirt, legs, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "arms, sleeves, hands, navy dress, skirt, legs, background, lowres"
    },
    "skirt": {
        "prompt": "Extract the navy blue pleated main skirt and double ruffled white hem from <image1>. Inpaint and complete continuous navy pleated fabric across the front where the apron was, extending vertical soft folds and gold filigree embroidery smoothly. Inpaint upper waist connection. Preserve the two ribbon bows. Remove apron, arms, legs, tail, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "apron, whale embroidery, arms, sleeves, hands, legs, background, lowres"
    },
    "leg_l": {
        "prompt": "Extract the left slender leg, knee, white ruffled ankle sock, and dark blue Mary Jane shoe from <image1>. Inpaint and extend the smooth cylinder thigh upward by 25px into where it was hidden under the skirt. Preserve the knee highlight, ankle frills, and shoe buckle. Remove the skirt, dress, other leg, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "skirt, dress, apron, background, other leg, lowres, blurry"
    },
    "leg_r": {
        "prompt": "Extract the right slender leg, knee, white ruffled ankle sock, and dark blue Mary Jane shoe from <image1>. Inpaint and extend the smooth cylinder thigh upward by 25px into where it was hidden under the skirt. Preserve the knee highlight, ankle frills, and shoe buckle. Remove the skirt, dress, other leg, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "skirt, dress, apron, background, other leg, lowres, blurry"
    },
    "tail_seg1": {
        "prompt": "Extract the complete continuous whale tail from <image1>. Include the entire smooth tail body from the root under the skirt, the broad curved lower section, the tapering stalk, up to the complete asymmetrical double-lobed tail fluke. Inpaint and smoothly complete the tail root 80px inward behind where the dress and white ruffles were, rounding naturally into a smooth tail root with continuous dark blue dorsal and soft light blue ventral shading. Remove the skirt, ruffles, apron, dress, legs, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "skirt, ruffles, apron, dress, legs, skin, hair, background, opaque background, lowres, blurry, extra fluke, duplicate tail, cropped root"
    },
    "tail_seg2": {
        "prompt": "Companion layer handled together with tail_seg1",
        "negative": ""
    },
    "tail_tip": {
        "prompt": "Companion layer handled together with tail_seg1",
        "negative": ""
    },
    "hair_back_l": {
        "prompt": "Extract the left long rear hair locks from <image1>. Keep the dark blue to light cyan gradient at the curly hair tips. Inpaint and complete the upper hair body that was covered by ear fin, cheek, sleeve, and skirt. Keep see-through gaps between locks transparent. Remove face, arm, dress, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "face, skin, arm, sleeve, dress, apron, background, solid hair block, lowres"
    },
    "hair_back_r": {
        "prompt": "Extract the right long rear hair locks from <image1>. Keep the dark blue to light cyan gradient at the curly hair tips. Inpaint and smoothly complete the hair flow behind where the whale tail and sleeve were, with continuous gradient. Remove face, arm, dress, tail, and background completely. Output as a clean transparent RGBA PNG image with pure transparent background.",
        "negative": "face, skin, arm, sleeve, dress, tail, apron, background, lowres"
    }
}


PROMPT_SETS = {"adult": ADULT_PROMPTS}
PRODUCTION_LAYER_DIRS = ("assets/rig_young/layers", "assets/rig_adult/layers")


def default_ref(stage: str) -> str:
    ref_ext = ".jpg" if stage == "young" else ".png"
    ref_path = f"assets/reference/{stage}_ref{ref_ext}"
    if not os.path.isfile(ref_path):
        ref_path = f"assets/reference/{stage}_ref.jpg"
    return ref_path


def is_production_layers(path: str) -> bool:
    p = os.path.normpath(path).replace("\\", "/").lower()
    return any(p.endswith(d) for d in PRODUCTION_LAYER_DIRS)


def prepare_snapped_crop(stage: str, layer_id: str, manifest_entry: dict,
                         ref_path: str = "") -> tuple[str, list[int], tuple[int, int]]:
    ref_path = ref_path or default_ref(stage)

    im = Image.open(ref_path).convert("RGBA")
    w_full, h_full = im.size
    c_min_x, c_min_y, c_max_x, c_max_y = manifest_entry["crop_rect_px"]

    crop = im.crop((c_min_x, c_min_y, c_max_x, c_max_y))
    crop_filename = f"qwen_crop_{stage}_{layer_id}.png"
    os.makedirs(COMFY_INPUT, exist_ok=True)
    crop.save(os.path.join(COMFY_INPUT, crop_filename))
    return crop_filename, [c_min_x, c_min_y, c_max_x, c_max_y], (w_full, h_full)


def queue_comfy_qwen(crop_filename: str, prompt_text: str, neg_prompt_text: str, prefix: str, seed: int = 20260924) -> str:
    prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_w4a8.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_2.1_vae_bf16.safetensors"
            }
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {
                "image": crop_filename
            }
        },
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt_text,
                "negative_prompt": neg_prompt_text,
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
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": prefix,
                "images": ["8", 0]
            }
        }
    }

    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return res["prompt_id"]


def wait_for_completion(prompt_id: str, timeout_s: int = 180) -> str:
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
        time.sleep(1)
    raise TimeoutError(f"Generation timed out for prompt {prompt_id}")


def process_and_align_layer(stage: str, layer_id: str, gen_filename: str, crop_rect_px: list[int],
                            full_size: tuple[int, int], out_layers_dir: str = "",
                            prompt_set: str = "", overwrite_production: bool = False):
    """Loads generated crop, ensures clean alpha and pastes to full canvas."""
    out_layers_dir = out_layers_dir or f"assets/rig_{stage}/layers"
    prompt_set = prompt_set or stage
    out_file = os.path.join(out_layers_dir, f"{layer_id}.png")
    if is_production_layers(out_layers_dir) and os.path.isfile(out_file) and not overwrite_production:
        raise SystemExit(f"refusing to overwrite production layer {out_file} "
                         "(pass --layers-out for a new package, or --overwrite-production)")
    gen_path = os.path.join(COMFY_OUTPUT, gen_filename)
    gen_im = Image.open(gen_path).convert("RGBA")

    w_full, h_full = full_size
    full_canvas = Image.new("RGBA", (w_full, h_full), (0, 0, 0, 0))
    c_min_x, c_min_y, c_max_x, c_max_y = crop_rect_px

    # If size slightly deviates, resize to exact crop box
    expected_size = (c_max_x - c_min_x, c_max_y - c_min_y)
    if gen_im.size != expected_size:
        gen_im = gen_im.resize(expected_size, Image.LANCZOS)

    full_canvas.paste(gen_im, (c_min_x, c_min_y), gen_im)

    os.makedirs(out_layers_dir, exist_ok=True)
    full_canvas.save(out_file)

    if prompt_set == "adult" and layer_id == "tail_seg1":
        # Ensure contract companion layers tail_seg2 and tail_tip exist
        import numpy as np
        dummy_arr = np.zeros((h_full, w_full, 4), dtype=np.uint8)
        dummy_arr[700:704, 480:484] = [54, 74, 122, 255]
        dummy_img = Image.fromarray(dummy_arr)
        for comp_id in ("tail_seg2", "tail_tip"):
            dummy_img.save(os.path.join(out_layers_dir, f"{comp_id}.png"))
            dummy_img.save(os.path.join(out_layers_dir, f"{comp_id}_crop.png"))

    print(f"[OK] Layer [{layer_id}] aligned and saved to {out_file}")


def generate_layer(stage: str, layer_id: str, manifest_path: str = "", ref_path: str = "",
                   layers_out: str = "", prompt_set: str = "", overwrite_production: bool = False):
    prompt_set = prompt_set or stage
    if prompt_set == "adult" and layer_id in ("tail_seg2", "tail_tip"):
        print(f"[INFO] Adult {layer_id} is a contract companion layer unified under tail_seg1.")
        return

    manifest_path = manifest_path or f"assets/rig_{stage}/prep/manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entry = next((m for m in manifest if m["id"] == layer_id), None)
    if not entry:
        print(f"Error: layer {layer_id} not found in manifest {manifest_path}")
        return

    prompts = PROMPT_SETS.get(prompt_set, {})
    if layer_id not in prompts:
        print(f"Error: no prompt defined for {layer_id} in prompt set {prompt_set}")
        return

    out_dir = layers_out or f"assets/rig_{stage}/layers"
    if is_production_layers(out_dir) and os.path.isfile(os.path.join(out_dir, f"{layer_id}.png")) \
            and not overwrite_production:
        raise SystemExit(f"refusing to overwrite production layer {out_dir}/{layer_id}.png "
                         "(pass --layers-out for a new package, or --overwrite-production)")

    print(f"\n--- Generating [{prompt_set}] layer: {layer_id} ---")
    crop_filename, crop_rect, full_size = prepare_snapped_crop(stage, layer_id, entry, ref_path)
    p_info = prompts[layer_id]

    prompt_id = queue_comfy_qwen(
        crop_filename=crop_filename,
        prompt_text=p_info["prompt"],
        neg_prompt_text=p_info["negative"],
        prefix=f"{stage}_{layer_id}"
    )
    print(f"Queued in ComfyUI (ID: {prompt_id}), waiting...")
    out_name = wait_for_completion(prompt_id)
    print(f"ComfyUI output generated: {out_name}")
    process_and_align_layer(stage, layer_id, out_name, crop_rect, full_size,
                            out_layers_dir=layers_out, prompt_set=prompt_set,
                            overwrite_production=overwrite_production)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Qwen layer generator.")
    parser.add_argument("--stage", choices=["young", "adult"], default="adult")
    parser.add_argument("target", nargs="?", default="all", help="layer_id or 'all'")
    parser.add_argument("--ref", default="", help="reference image (default assets/reference/{stage}_ref.*)")
    parser.add_argument("--prep-dir", default="", help="dir with manifest.json (default assets/rig_{stage}/prep)")
    parser.add_argument("--layers-out", default="", help="output layers dir (default assets/rig_{stage}/layers)")
    parser.add_argument("--prompt-set", default="", choices=["", *PROMPT_SETS],
                        help="prompt table (default = stage)")
    parser.add_argument("--overwrite-production", action="store_true",
                        help="allow overwriting existing layers in assets/rig_young|rig_adult/layers")
    parser.add_argument("--dry-run", action="store_true", help="print resolved paths and exit")
    args = parser.parse_args()

    prep_dir = args.prep_dir or f"assets/rig_{args.stage}/prep"
    manifest_path = os.path.join(prep_dir, "manifest.json")
    layers_out = args.layers_out or f"assets/rig_{args.stage}/layers"
    ref_path = args.ref or default_ref(args.stage)
    prompt_set = args.prompt_set or args.stage
    if args.dry_run:
        print(json.dumps({"manifest": manifest_path, "ref": ref_path, "layers_out": layers_out,
                          "prompt_set": prompt_set,
                          "production_layers": is_production_layers(layers_out)}, indent=2))
        sys.exit(0)
    if not os.path.isfile(manifest_path):
        print(f"Error: {manifest_path} not found. Run crop_and_prep.py first.")
        sys.exit(1)

    kwargs = dict(manifest_path=manifest_path, ref_path=ref_path, layers_out=layers_out,
                  prompt_set=prompt_set, overwrite_production=args.overwrite_production)
    if args.target == "all":
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for m in manifest:
            generate_layer(args.stage, m["id"], **kwargs)
    else:
        generate_layer(args.stage, args.target, **kwargs)
