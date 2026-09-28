"""
Pre-processing and crop preparation tool for Desktop Pet 2D Rigging.
Reads {stage}_rig_spec.json and {stage}_ref.png, extracts bounding box crops with context margins,
ensures 32-pixel multiple alignment for Qwen-Image-2.1 VAE,
and generates a visual debug overlay of all layers.
"""

import os
import json
import argparse
from PIL import Image, ImageDraw


def prep_layers(spec_path: str, ref_image_path: str, out_dir: str, stage: str = "young"):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    im = Image.open(ref_image_path).convert("RGBA")
    w, h = im.size

    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    final_layers_dir = os.path.join(out_dir, "..", "layers")
    os.makedirs(final_layers_dir, exist_ok=True)

    debug_img = im.copy()
    draw = ImageDraw.Draw(debug_img)

    manifest = []

    for layer in spec["layers"]:
        layer_id = layer["id"]
        bbox = layer["bbox_hint"]  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = bbox

        # Convert to pixel coordinates
        px_min = int(xmin * w)
        py_min = int(ymin * h)
        px_max = int(xmax * w)
        py_max = int(ymax * h)

        # Add 12% padding for context inpainting
        pad_x = int((px_max - px_min) * 0.12)
        pad_y = int((py_max - py_min) * 0.12)

        c_px_min = max(0, px_min - pad_x)
        c_py_min = max(0, py_min - pad_y)
        c_px_max = min(w, px_max + pad_x)
        c_py_max = min(h, py_max + pad_y)

        # 32-pixel multiple alignment for VAE
        crop_w = c_px_max - c_px_min
        crop_h = c_py_max - c_py_min
        rem_w = crop_w % 32
        rem_h = crop_h % 32
        if rem_w != 0:
            add_w = 32 - rem_w
            c_px_max = min(w, c_px_max + add_w)
            if (c_px_max - c_px_min) % 32 != 0:
                c_px_min = max(0, c_px_min - (32 - ((c_px_max - c_px_min) % 32)))
        if rem_h != 0:
            add_h = 32 - rem_h
            c_py_max = min(h, c_py_max + add_h)
            if (c_px_max - c_px_min) % 32 != 0:
                c_py_min = max(0, c_py_min - (32 - ((c_py_max - c_py_min) % 32)))

        # Final safety check: if still not divisible by 32 due to image boundaries, clamp width/height
        final_w = ((c_px_max - c_px_min) // 32) * 32
        final_h = ((c_py_max - c_py_min) // 32) * 32
        if final_w > 0:
            c_px_max = c_px_min + final_w
        if final_h > 0:
            c_py_max = c_py_min + final_h

        # Crop context
        crop = im.crop((c_px_min, c_py_min, c_px_max, c_py_max))
        crop_filename = f"{layer_id}_crop.png"
        crop_path = os.path.join(crops_dir, crop_filename)
        crop.save(crop_path)

        # Draw bbox on debug overlay
        draw.rectangle([px_min, py_min, px_max, py_max], outline="magenta", width=2)
        draw.rectangle([c_px_min, c_py_min, c_px_max, c_py_max], outline="cyan", width=1)
        draw.text((px_min + 4, py_min + 4), f"{layer['z_order']}: {layer_id}", fill="yellow")

        manifest.append({
            "id": layer_id,
            "z_order": layer["z_order"],
            "description": layer["description"],
            "bbox_norm": bbox,
            "crop_rect_px": [c_px_min, c_py_min, c_px_max, c_py_max],
            "crop_file": crop_filename,
            "requires_inpaint": layer["requires_inpaint"],
            "inpaint_targets": layer.get("inpaint_targets", []),
            "self_completion_guide": layer.get("self_completion_guide", ""),
            "inpaint_guide": layer.get("inpaint_guide", ""),
            "target_layer_png": f"assets/rig_{stage}/layers/{layer_id}.png"
        })

    debug_path = os.path.join(out_dir, "all_layers_bboxes.png")
    debug_img.save(debug_path)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(manifest)} layers in {out_dir}")
    print(f"Debug overlay saved to {debug_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and prep rig layers.")
    parser.add_argument("--stage", choices=["young", "adult", "final"], default="young")
    args = parser.parse_args()

    ref_ext = ".jpg" if args.stage == "young" else ".png"
    spec = f"assets/reference/{args.stage}_rig_spec.json"
    ref = f"assets/reference/{args.stage}_ref{ref_ext}"
    if not os.path.isfile(ref):
        ref = f"assets/reference/{args.stage}_ref.jpg"
    out = f"assets/rig_{args.stage}/prep"

    prep_layers(spec, ref, out, stage=args.stage)
