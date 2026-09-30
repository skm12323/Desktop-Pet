"""Replay captured 256-px poses to compare original and candidate side layers in Qt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=".scratch/adult_fix_v1/candidate")
    ap.add_argument("--before", default="assets/rig_adult_walk_v1")
    args = ap.parse_args()
    from render_rig_rest import RigRenderer
    from PySide6.QtGui import QImage

    out = ROOT / "spikes/_qa/adult_visual_fix_2026-09-30"
    out.mkdir(parents=True, exist_ok=True)
    trace = json.loads((ROOT / "spikes/_qa/adult_visual_review_2026-09-30/trace.json").read_text())
    r = RigRenderer()
    w = r.win
    w.resize(768, 768)
    r._pump()
    samples = [6, 6.2, 6.4, 6.7]
    images = []
    for tag, pkg in (("before", ROOT / args.before), ("after", ROOT / args.candidate)):
        assert w.enable_side_locomotion(str(pkg))
        w._root.setProperty("locoMode", 2)
        for t in samples:
            rec = min((x for x in trace if "angles" in x), key=lambda x: abs(x["t"] - t))
            item = w._side_item
            for bone in item._rt.bone_index:
                tx, ty = rec["pelvis"] if bone == "root_hip" else (0, 0)
                if bone.startswith("upper_leg_"):
                    tx = -36 if bone.endswith("l") else -50
                item.setBonePose(bone, rec["angles"].get(bone, 0), tx, ty)
            item.setBlink(0)
            item.setLookAt(0, 0)
            r._pump()
            q = w._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
            a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine() // 4, 4)
            im = Image.fromarray(a[:, :q.width()].copy()).resize((768, 768))
            im.save(out / f"layers_{tag}_{t:.2f}.png")
            images.append((tag, t, im))
        # Zero-pose rest composite should not change visibly.
        for bone in w._side_item._rt.bone_index:
            w._side_item.setBonePose(bone, 0, 0, 0)
        r._pump()
        q = w._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine() // 4, 4)
        Image.fromarray(a[:, :q.width()].copy()).save(out / f"side_rest_{tag}.png")
    canvas = Image.new("RGB", (380 * 4, 355 * 2), (32, 36, 45))
    for i, (tag, t, im) in enumerate(images):
        b = Image.new("RGBA", im.size, (32, 36, 45, 255))
        b.alpha_composite(im)
        x, y = (i % 4) * 380, (i // 4) * 355
        canvas.paste(b.convert("RGB").crop((220, 285, 600, 610)), (x, y + 30))
        ImageDraw.Draw(canvas).text((x + 8, y + 8), f"{tag} {t:.2f}s", fill="white")
    canvas.save(out / "bow_layers_before_after.png")
    before = np.asarray(Image.open(out / "side_rest_before.png"), float) / 255
    after = np.asarray(Image.open(out / "side_rest_after.png"), float) / 255
    mask = (before[..., 3] > .1) | (after[..., 3] > .1)
    err = np.abs(before[..., :3] * before[..., 3:] - after[..., :3] * after[..., 3:]).mean(-1)
    stats = {"premultiplied_mean_delta_255": float(err[mask].mean() * 255),
             "new_holes_px_at_768": int(((before[..., 3] > .9) & (after[..., 3] < .1)).sum())}
    (out / "layer_fix_metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    r.close()
    print(stats)


if __name__ == "__main__":
    main()
