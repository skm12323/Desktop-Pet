"""Reproduce the ADULT visual review without modifying rig assets.

Run: python -X utf8 spikes/qa_adult_review.py
Outputs fresh Qt renders and numerical evidence under spikes/_qa/adult_review.
The walk samples exercise the current production MotionEngine, not a new gait.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("QT_QUICK_BACKEND", "rhi")
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pet.asset_provider import SpriteRef
from pet.rig.motion import MotionEngine, MotionFrame, MotionInputs
from pet.rig.presenter import build_rig_window
from pet.rig.spec import load_rig_spec
from pet.window import WindowBase

OUT = ROOT / "spikes" / "_qa" / "adult_review"


def background(img, color=(240, 243, 248)):
    bg = Image.new("RGBA", img.size, (*color, 255))
    bg.alpha_composite(img)
    return bg.convert("RGB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    spec = load_rig_spec(str(ROOT / "assets/rig/adult"), "adult")
    win = build_rig_window(WindowBase, SpriteRef(spec.figures["healthy_neutral"], 480, 848), "adult")
    win._motion_timer.stop()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.resize(480, 848)
    win.show()
    for _ in range(4):
        app.processEvents()
    rt = win._skinned_item._rt
    names = [b.name for b in rt.bones]

    def capture(frame):
        win._push_frame(frame)
        # Freeze QML's angle Behavior so the screenshot represents the stated pose.
        win._root.setProperty("bodyAngle", 0.0)
        app.processEvents()
        img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
        assert not img.isNull(), "Qt framebuffer unavailable; requires a hardware RHI backend"
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
        pil = Image.fromarray(arr[:, :img.width()].copy(), "RGBA")
        assert np.count_nonzero(np.array(pil)[:, :, 3] > 20) > 5000, "Blank Qt render"
        return pil.resize((win.width(), win.height()), Image.Resampling.LANCZOS)

    rest = MotionFrame(bone_angles=dict.fromkeys(names, 0.0))
    static = capture(rest)
    static.save(OUT / "rest_rgba.png")
    background(static).save(OUT / "rest.png")
    win.resize(960, 1696)
    app.processEvents()
    high = capture(rest)
    high.save(OUT / "rest_source_size.png")
    win.resize(480, 848)
    app.processEvents()
    ref = Image.open(ROOT / "assets/reference/adult_ref-a-pose.png").convert("RGBA")
    ref.thumbnail((480, 848), Image.Resampling.LANCZOS)
    sheet = Image.new("RGB", (1440, 886), (240, 243, 248))
    sheet.paste(background(ref), ((480 - ref.width) // 2, 38))
    sheet.paste(background(static), (480, 38))
    sheet.paste(background(static, (40, 44, 55)), (960, 38))
    draw = ImageDraw.Draw(sheet)
    for x, label in [(12, "A-pose reference"), (492, "Current zero pose / light"), (972, "Current zero pose / dark")]:
        draw.text((x, 12), label, fill=(30, 40, 55))
    sheet.save(OUT / "static_comparison.jpg", quality=93)

    crops = [(120, 85, 330, 215), (105, 220, 360, 365), (280, 475, 477, 646), (160, 568, 298, 815)]
    details = Image.new("RGB", (1000, 480), (240, 243, 248))
    d = ImageDraw.Draw(details)
    for i, (box, label) in enumerate(zip(crops, ["Face", "Shoulders / waist", "Tail join", "Knees / shoes"])):
        crop = background(high).crop(tuple(v * 2 for v in box))
        crop.thumbnail((240, 430), Image.Resampling.LANCZOS)
        details.paste(crop, (i * 250 + (250 - crop.width) // 2, 40))
        d.text((i * 250 + 12, 12), label, fill=(30, 40, 55))
    details.save(OUT / "details.png")

    win.resize(256, 256)
    app.processEvents()
    small = capture(rest)
    background(small).save(OUT / "actual_256.png")
    win.resize(480, 848)
    app.processEvents()

    engine = MotionEngine(spec)
    inputs = MotionInputs(walking=True, walk_hz=1.2, wind_gain=0.0)
    for _ in range(120):
        engine.step(inputs, 1000 / 60)
    engine._gait_phase = 0.0
    animation = []
    samples = []
    records = []
    zero = np.zeros(len(names), np.float32)
    rt.skinning_matrices(zero, zero, zero, 0, 0)
    layers = {l.layer_id: l for l in rt.layers}
    rest_floor = {side: float(rt.deform(layers['leg_' + side], rt.effective_rest(layers['leg_' + side], 0))[:, 1].max()) for side in ('l', 'r')}
    for i in range(64):
        frame = engine.step(inputs, 1000 / 1.2 / 64)
        # Isolate the limb problem from whole-widget rotation and bob/breath.
        frame.body_angle = 0.0
        frame.body_y = 0.0
        frame.blink_progress = 0.0
        frame.look_at = (0, 0)
        frame.bone_angles.update({name: 0.0 for name in names if not any(t in name for t in ('leg_', 'foot_', 'arm_', 'hand_'))})
        img = capture(frame)
        animation.append(background(img.resize((300, 530), Image.Resampling.LANCZOS)))
        if i % 8 == 0:
            samples.append(background(img))
        angle = np.array([frame.bone_angles.get(n, 0.0) for n in names], np.float32)
        rt.skinning_matrices(angle, zero, zero, 0, 0)
        record = {"phase": engine.gait_phase}
        shoe_centers = {}
        for side in ('l', 'r'):
            layer = layers['leg_' + side]
            xy = rt.deform(layer, rt.effective_rest(layer, 0)).copy()
            record['floor_delta_' + side + '_display_px_at_256'] = (float(xy[:, 1].max()) - rest_floor[side]) * 256 / 1696
            shoe_centers[side] = float(xy[layer.rest[:, 1] >= 1490, 0].mean())
        record['shoe_center_gap_display_px_at_256'] = (shoe_centers['r'] - shoe_centers['l']) * 256 / 1696
        records.append(record)
    strip = Image.new("RGB", (4 * 240, 2 * 452), (240, 243, 248))
    for i, img in enumerate(samples):
        x, y = (i % 4) * 240, (i // 4) * 452
        strip.paste(img.resize((240, 424), Image.Resampling.LANCZOS), (x, y + 28))
        ImageDraw.Draw(strip).text((x + 10, y + 8), f"phase {records[i * 8]['phase']:.3f}", fill=(30, 40, 55))
    strip.save(OUT / "current_walk_keyframes.jpg", quality=93)
    # Deliberate slow-motion loop for seam inspection (30 ms per sample).
    animation[0].save(OUT / "current_walk_isolated.gif", save_all=True, append_images=animation[1:], duration=30, loop=0)

    mesh = json.loads(Path(spec.skinned_mesh).read_text(encoding="utf-8"))
    raw = json.loads(Path(spec.skinned_spec).read_text(encoding="utf-8"))
    binding = {}
    for layer in mesh['layers']:
        if layer['id'] in ('leg_l', 'leg_r', 'skirt', 'arm_l', 'arm_r'):
            verts = np.asarray(layer['vertices'])
            rows = list(zip(layer['weight_bones'], layer['weight_values']))
            shoe = verts[:, 1] >= 1490
            weights = [sum(w for b, w in zip(bs, ws) if b.startswith('upper_leg')) for bs, ws in rows]
            binding[layer['id']] = {'vertex_count': len(verts), 'bounds': [*verts.min(axis=0).tolist(), *verts.max(axis=0).tolist()]}
            if layer['id'].startswith('leg_') and shoe.any():
                binding[layer['id']]['mean_upper_leg_weight_on_shoe'] = float(np.mean(np.asarray(weights)[shoe]))
    metrics = {
        'bones': len(names), 'layers': len(rt.layers), 'vertices': sum(len(l['vertices']) for l in mesh['layers']),
        'triangles': sum(len(l['triangles']) // 3 for l in mesh['layers']),
        'visible_mesh': bool(win._root.property('skinnedMeshVisible')),
        'source_size': [rt.img_w, rt.img_h], 'rest_floor_source_px': rest_floor,
        'rest_bottom_gap_display_px_at_256': (1696 - max(rest_floor.values())) * 256 / 1696,
        'binding': binding, 'walk_sample_scope': 'limb-only current engine; body angle/bob/breath and secondary motion disabled',
        'walk_samples': records,
        'angle_limits': {b['bone_name']: b['angle_clamp'] for b in raw['skeleton']['bones'] if 'leg' in b['bone_name'] or 'foot' in b['bone_name']},
    }
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    win.close()
    print(json.dumps({k: v for k, v in metrics.items() if k != 'walk_samples'}, ensure_ascii=False, indent=2))
    for side in ('l', 'r'):
        values = [r['floor_delta_' + side + '_display_px_at_256'] for r in records]
        print(f'foot {side}: floor delta at 256 = {min(values):.2f} .. {max(values):.2f} px')
    gaps = [r['shoe_center_gap_display_px_at_256'] for r in records]
    print(f'shoe center signed gap at 256 = {min(gaps):.2f} .. {max(gaps):.2f} px')
    print(f'Outputs: {OUT}')


if __name__ == '__main__':
    main()
