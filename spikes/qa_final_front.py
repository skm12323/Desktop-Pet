"""FINAL F2 front rig QA through the real Qt skinned path (needs Python 3.12 + D3D11).

Writes spikes/_qa/final_front_f2/:
  rest_render.png      relaxed rest, canvas-exact; diff vs references/front_key.png
  blink_sheet.png      eye crops at blink 0 / .5 / 1 / reopen; closed-eye sclera count
  gaze_sheet.png       eye crops for look-at centre / left / right / up / down
  extremes_sheet.png   every bone group at its clamp limits (hidden completions exposed)
  walk_sheet.png       front-walk gait poses (arms/legs swing; what the arms hide at rest)
  idle.gif             MotionEngine idle (breath, sway, blink, gaze) rendered at 360x641, saved 240 px
  report.json          numbers

Usage: QT_QPA_PLATFORM=windows QT_QUICK_BACKEND=rhi QSG_RHI_BACKEND=d3d11 \
       <py312> -X utf8 spikes/qa_final_front.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
OUT = ROOT / "spikes" / "_qa" / "final_front_f2"
KEY = ROOT / "assets" / "rig_final" / "references" / "front_key.png"
EYES = (430, 215, 610, 285)            # both eyes, canvas px

GROUPS = {
    "head": ["head", "neck"],
    "arms": ["upper_arm_l", "forearm_l", "hand_l", "upper_arm_r", "forearm_r", "hand_r"],
    "hair": ["hair_back_l_01", "hair_back_l_02", "hair_back_l_03", "hair_back_r_01", "hair_back_r_02",
             "hair_back_r_03", "hair_side_l_01", "hair_side_l_02", "hair_side_r_01", "hair_side_r_02",
             "bangs_01", "bangs_02", "bangs_03", "ahoge_01", "ahoge_02"],
    "tail": ["tail_01", "tail_02", "tail_03", "tail_fluke"],
    "skirt": ["skirt_root", "skirt_hem_l", "skirt_hem_r", "apron_root", "apron_tip"],
    "body": ["root_hip", "spine", "chest"],
    "ears": ["ear_fin_l", "ear_fin_r"],
    "legs": ["upper_leg_l", "lower_leg_l", "foot_l", "upper_leg_r", "lower_leg_r", "foot_r"],
}


def premul_diff(a: Image.Image, b: Image.Image) -> np.ndarray:
    x = np.asarray(a.convert("RGBA"), np.float32)
    y = np.asarray(b.convert("RGBA"), np.float32)
    return np.abs(x[..., :3] * x[..., 3:] - y[..., :3] * y[..., 3:]).max(-1) / (255 * 255)


def on_bg(im: Image.Image, rgb=(244, 246, 250)) -> Image.Image:
    bg = Image.new("RGBA", im.size, (*rgb, 255))
    bg.alpha_composite(im.convert("RGBA"))
    return bg.convert("RGB")


def sclera_px(im: Image.Image) -> int:
    a = np.asarray(im.convert("RGBA"))[EYES[1]:EYES[3], EYES[0]:EYES[2]].astype(int)
    white = (a[..., :3].min(-1) > 225) & (np.ptp(a[..., :3], axis=-1) < 18) & (a[..., 3] > 200)
    return int(white.sum())


def iris_px(im: Image.Image) -> int:
    a = np.asarray(im.convert("RGBA"))[EYES[1]:EYES[3], EYES[0]:EYES[2]].astype(int)
    return int(((a[..., 2] > a[..., 0] + 25) & (a[..., 2] > 90) & (a[..., 3] > 200)).sum())


def sheet(tiles: list[tuple[str, Image.Image]], cols: int) -> Image.Image:
    tw = max(t.size[0] for _, t in tiles)
    th = max(t.size[1] for _, t in tiles) + 16
    s = Image.new("RGB", (tw * cols, th * ((len(tiles) + cols - 1) // cols)), (255, 255, 255))
    d = ImageDraw.Draw(s)
    for i, (name, t) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * th
        s.paste(t, (x, y + 16))
        d.text((x + 3, y + 2), name, fill=(0, 0, 0))
    return s


def main() -> None:
    from render_rig_rest import RigRenderer
    from pet.rig.motion import MotionInputs

    OUT.mkdir(parents=True, exist_ok=True)
    r = RigRenderer(stage="final")
    rep: dict = {}
    try:
        rest = r.render(r.rest_frame("relaxed"))
        rest.save(OUT / "rest_render.png")
        key = Image.open(KEY)
        d = premul_diff(rest, key)
        op = np.asarray(key)[..., 3] > 0
        rep["rest_vs_key"] = {"mean_255": float(d[op].mean() * 255), "p99_255": float(np.percentile(d[op], 99) * 255),
                              "frac_gt_24": float((d[op] * 255 > 24).mean())}

        # blink
        tiles, open_s, open_i = [], sclera_px(rest), iris_px(rest)
        rep["blink"] = {"open_sclera_px": open_s, "open_iris_px": open_i}
        for b in (0.0, 0.5, 1.0, 0.0):
            f = r.rest_frame("relaxed")
            f.blink_progress = b
            im = r.render(f)
            tiles.append((f"blink {b}", on_bg(im.crop(EYES)).resize(((EYES[2] - EYES[0]) * 3, (EYES[3] - EYES[1]) * 3),
                                                                     Image.Resampling.NEAREST)))
            rep["blink"][f"sclera_px_at_{b}"] = sclera_px(im)
            if b == 0.0 and len(tiles) == 4:
                rep["blink"]["reopen_identical"] = bool(np.array_equal(np.asarray(im), np.asarray(rest)))
        sheet(tiles, 2).save(OUT / "blink_sheet.png")

        # gaze
        tiles = []
        for name, la in [("centre", (0, 0)), ("left", (-1, 0)), ("right", (1, 0)), ("up", (0, -1)), ("down", (0, 1))]:
            f = r.rest_frame("relaxed")
            f.look_at = la
            im = r.render(f)
            tiles.append((name, on_bg(im.crop(EYES)).resize(((EYES[2] - EYES[0]) * 3, (EYES[3] - EYES[1]) * 3),
                                                            Image.Resampling.NEAREST)))
        sheet(tiles, 3).save(OUT / "gaze_sheet.png")

        # clamp extremes per group
        clamps = {b.name: b.clamp for b in r.rt.bones}
        tiles = []
        for g, names in GROUPS.items():
            for sgn in (-1, 1):
                f = r.rest_frame("relaxed")
                for n in names:
                    lo, hi = clamps[n]
                    f.bone_angles[n] = float(hi if sgn > 0 else lo)
                im = r.render(f)
                tiles.append((f"{g} {'max' if sgn > 0 else 'min'}", on_bg(im).resize((341, 608), Image.Resampling.LANCZOS)))
        sheet(tiles, 8).save(OUT / "extremes_sheet.png")

        # front-walk gait poses (motion.py section 6 at k_gait = 1, clamped by the spec)
        tiles = []
        for i in range(8):
            phi = 2 * np.pi * i / 8
            f = r.rest_frame("relaxed")
            f.bone_angles.update({
                "upper_leg_l": 14 * np.sin(phi), "lower_leg_l": -abs(18 * max(0.0, -np.cos(phi))),
                "upper_leg_r": 14 * np.sin(phi + np.pi), "lower_leg_r": -abs(18 * max(0.0, -np.cos(phi + np.pi))),
                "upper_arm_l": -10 * np.sin(phi), "forearm_l": -6 * max(0.0, np.cos(phi)),
                "upper_arm_r": -10 * np.sin(phi + np.pi), "forearm_r": -6 * max(0.0, np.cos(phi + np.pi))})
            im = r.render(f)
            tiles.append((f"walk {i}/8", on_bg(im.crop((180, 420, 860, 1790))).resize((340, 685), Image.Resampling.LANCZOS)))
        sheet(tiles, 8).save(OUT / "walk_sheet.png")

        # idle through the real motion engine
        win = r.win
        size = (360, 641)
        frames = []
        cursors = [(300.0, 120.0)] * 30 + [(40.0, 200.0)] * 30 + [(180.0, 500.0)] * 30
        for i, cur in enumerate(cursors):
            fr = win._engine.step(MotionInputs(walking=False, walk_hz=0.0, facing=1, cursor_pos=cur,
                                               pet_rect=(0.0, 0.0, float(size[0]), float(size[1]))), 66.0)
            im = r.render(fr, size=size, ground_shift=True)
            frames.append(on_bg(im))
        small = [f.resize((240, 427), Image.Resampling.LANCZOS) for f in frames[::2]]
        small[0].save(OUT / "idle.gif", save_all=True, append_images=small[1:], duration=132, loop=0)
        rep["idle_frames"] = len(frames)
    finally:
        r.close()
    (OUT / "report.json").write_bytes(json.dumps(rep, indent=1).encode("utf-8"))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
