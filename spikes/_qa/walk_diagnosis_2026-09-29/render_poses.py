"""Render captured walk poses (and human-range test poses) canvas-exact on the side rig."""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from render_rig_rest import RigRenderer
from pet.rig.motion import MotionFrame
OUT = ROOT / ".scratch/diag/poses"; OUT.mkdir(parents=True, exist_ok=True)
T = json.load(open(ROOT / ".scratch/diag/walk_256_120/trace.json", encoding="utf8"))
W = [r for r in T if r.get("gait") == "walk_loop" and r.get("pose")]
# one full cycle: 8 samples by phase
picks = []
for target in np.linspace(0, 1, 8, endpoint=False):
    picks.append(min(W, key=lambda r: min(abs(r["phase"] - target), 1 - abs(r["phase"] - target))))
r = RigRenderer(rig_dir=str(ROOT / "assets/rig_adult_walk_v1/adult"))
def frame(p):
    return MotionFrame(bone_angles=dict(p["a"]), bone_tx=dict(p["tx"]), bone_ty=dict(p["ty"]),
                       blink_progress=0.0, look_at=(0.0, 0.0))
imgs = []
for rec in picks:
    im = r.render(frame(rec["pose"]))
    im.save(OUT / f"walk_phase_{rec['phase']:.2f}.png"); imgs.append((rec["phase"], im))
# human-range test poses: scale the captured extreme frame's leg angles
ext = max(W, key=lambda r: r["pose"]["a"].get("upper_leg_l", 0) - r["pose"]["a"].get("upper_leg_r", 0))
tests = {}
for name, k in (("x1.5", 1.5), ("x2.0", 2.0)):
    p = json.loads(json.dumps(ext["pose"]))
    for b in ("upper_leg_l", "upper_leg_r", "lower_leg_l", "lower_leg_r"):
        p["a"][b] = p["a"].get(b, 0) * k
    tests[name] = r.render(frame(p)); tests[name].save(OUT / f"legs_{name}.png")
r.close()
def bg(im):
    b = Image.new("RGBA", im.size, (236, 238, 242, 255)); b.alpha_composite(im); return b.convert("RGB")
# strip: whole body
box = (60, 330, 900, 1680); s = 0.32
strip = Image.new("RGB", (8 * int((box[2]-box[0]) * s), int((box[3]-box[1]) * s) + 16), "white")
for i, (ph, im) in enumerate(imgs):
    c = bg(im).crop(box); c = c.resize((int(c.width * s), int(c.height * s)), Image.Resampling.LANCZOS)
    strip.paste(c, (i * c.width, 16)); ImageDraw.Draw(strip).text((i * c.width + 3, 2), f"phase {ph:.2f}", fill="black")
strip.save(OUT / "walk_cycle_strip.png")
# zoom: hips/knees region
zb = (250, 640, 760, 1330); zs = 0.6
z = Image.new("RGB", (4 * int((zb[2]-zb[0]) * zs), 2 * int((zb[3]-zb[1]) * zs) + 32), "white")
for i, (ph, im) in enumerate(imgs):
    c = bg(im).crop(zb); c = c.resize((int(c.width * zs), int(c.height * zs)), Image.Resampling.LANCZOS)
    x, y = (i % 4) * c.width, (i // 4) * (c.height + 16) + 16
    z.paste(c, (x, y)); ImageDraw.Draw(z).text((x + 3, y - 14), f"phase {ph:.2f}", fill="black")
z.save(OUT / "walk_hips_knees_zoom.png")
t = Image.new("RGB", (2 * int((box[2]-box[0]) * 0.45), int((box[3]-box[1]) * 0.45) + 16), "white")
for i, (n, im) in enumerate(tests.items()):
    c = bg(im).crop(box); c = c.resize((int(c.width * .45), int(c.height * .45)), Image.Resampling.LANCZOS)
    t.paste(c, (i * c.width, 16)); ImageDraw.Draw(t).text((i * c.width + 3, 2), f"leg angles {n} of the widest walk frame", fill="black")
t.save(OUT / "legs_amplified.png")
print("ok")
