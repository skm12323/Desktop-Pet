"""FINAL long-skirt walking prototype (docs/ADULT视觉流程复盘与FINAL复用指南-2026-10-01.md §3/§4).

Question: can ADULT's skirt rig (2 hem bones following the thighs, gain 0.4, +-8 deg, spring) carry
FINAL's ankle-length A-line skirt, or does it need a different rig? Built on the F3 side key art
(assets/rig_final_walk_v1/references/side_key.png) with a deliberately coarse layer split:
  tail | far leg | near leg | skirt (+petticoat) | body (everything else, rigid on the spine)
Legs below the hem are the art's ankles + shoes; above the hem they are hidden stocking columns.
Legs move with the real GaitSolver (ADULT side gait parameters, FINAL joint positions).

Skirt variants (same texture, same mesh, different bones / weights / drive):
  A  adult    - ADULT rig as is: skirt_hem_l (back) / skirt_hem_r (front) follow the thighs
  B  panels   - 4 panel bones across the skirt (back .. front), each a 2-bone chain (hip->knee
                height->hem), driven by the legs: the front panels are pushed by the forward
                knee / shin (contact, stays ahead of it), the back panels trail the rear leg with
                a spring lag; panel weights blend across x, so cloth bends instead of shearing

Outputs -> spikes/_qa/final_long_skirt_proto/:
  pkg/                          prototype rig package (spec, mesh, layers)
  walk_<variant>.gif            2 gait cycles at 120 px/s (canvas render, cropped, 1/3 scale)
  sheet_<variant>.png           8 phases side by side
  metrics.json                  per variant: skirt triangle stretch (sigma max/min over the walk),
                                hem swing amplitude, "legs under a frozen skirt" score (shoe travel
                                relative to the hem front edge), fold count

Usage (Python 3.12 + D3D11 for rendering; masks come from prep/sam_masks of the F3 package):
  QT_QPA_PLATFORM=windows QT_QUICK_BACKEND=rhi QSG_RHI_BACKEND=d3d11 \
    <py312> -X utf8 spikes/prototype_final_long_skirt.py
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
SRC = ROOT / "assets" / "rig_final_walk_v1"
OUT = ROOT / "spikes" / "_qa" / "final_long_skirt_proto"
PKG = OUT / "pkg"
W, H = 1024, 1824
KEY = SRC / "references" / "side_key.png"
ADULT_SIDE = ROOT / "assets" / "rig_adult_walk_v1" / "spec.json"

HEM_Y = 1586            # lowest petticoat ruffle row at the legs (measured on side_key.png)
WAIST_Y = 650
SKIRT_BOX = (110, 600, 900, 1640)
LEG_SPLIT_X = 522       # ankles: near (her right, screen-left "_l") | far ("_r")

# FINAL side joints (canvas px). Knee pivots sit ahead of the hip-ankle line (ADULT lesson 2.3.5).
J = {
    "root_hip": (520, 860), "spine": (520, 700), "chest": (530, 520),
    "upper_leg_l": (505, 870), "lower_leg_l": (503, 1290), "foot_l": (482, 1705),
    "upper_leg_r": (545, 865), "lower_leg_r": (556, 1290), "foot_r": (552, 1700),
    "skirt_root": (520, WAIST_Y),
    "skirt_hem_l": (330, 1000), "skirt_hem_r": (720, 1000),
}
# variant B: panel chains (root at the waist line, mid joint at knee height)
PANELS = {          # name: (x at waist, x at hem)  back -> mid -> front; single hinge at the waist
    "skirt_back": (400, 180), "skirt_mid_b": (480, 440), "skirt_mid_f": (560, 660), "skirt_front": (640, 840),
}
KNEE_Y = 1290
MARKERS = {"heel_l": [-40, 55], "sole_l": [18, 57], "forefoot_l": [44, 52],
           "heel_r": [-45, 37], "sole_r": [35, 40], "forefoot_r": [100, 30]}


def load_mask(name: str) -> np.ndarray:
    return np.asarray(Image.open(SRC / "prep" / "sam_masks" / f"{name}.png").convert("L")) > 127


def box(b) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[b[1]:b[3], b[0]:b[2]] = True
    return m


# ------------------------------------------------------------------ layers
def build_layers() -> dict:
    key = np.asarray(Image.open(KEY).convert("RGBA"))
    op = key[..., 3] > 0
    c = key[..., :3].astype(np.int16)
    lum = c @ np.array([299, 587, 114]) / 1000
    yy, xx = np.mgrid[0:H, 0:W]
    tail = load_mask("tail") & op
    below = op & (yy >= HEM_Y - 6) & box((400, 1560, 700, 1790))
    stock = below & (lum > 170) & ~load_mask("shoe_near") & ~load_mask("shoe_far")
    seed_l = load_mask("shoe_near") | (stock & (xx < LEG_SPLIT_X))
    seed_r = (load_mask("shoe_far") | (stock & (xx >= LEG_SPLIT_X))) & ~seed_l
    # every pixel below the hem belongs to a leg (outlines SAM left out stayed on the body as
    # ghost shoes): nearest seed wins
    dl = ndimage.distance_transform_edt(~seed_l)
    dr = ndimage.distance_transform_edt(~seed_r)
    leg_l = below & (dl <= dr)
    leg_r = below & (dr < dl)
    # the petticoat's lowest ruffles sit right above the ankles: keep them on the skirt
    leg_l &= yy >= HEM_Y - 2
    leg_r &= yy >= HEM_Y - 2
    upper = (load_mask("t_hair") | load_mask("t_sleeve") | load_mask("t_hand") | load_mask("t_apron")) & op
    skirt = (load_mask("t_skirt") | load_mask("t_ruffles") | box(SKIRT_BOX)) & op & ~upper & ~tail & ~leg_l & ~leg_r
    skirt &= yy >= WAIST_Y - 10
    body = op & ~tail & ~leg_l & ~leg_r & ~skirt
    parts = {"tail": tail, "leg_r": leg_r, "leg_l": leg_l, "skirt": skirt, "body": body}
    # islands < 60 px go to the surrounding part
    for name in ("leg_l", "leg_r", "skirt", "tail"):
        lab, n = ndimage.label(parts[name], structure=np.ones((3, 3)))
        if n > 1:
            sizes = ndimage.sum(parts[name], lab, range(1, n + 1))
            for k in np.where(sizes < 60)[0] + 1:
                m = lab == k
                parts[name] &= ~m
                parts["body"] |= m
    out = {}
    (PKG / "layers").mkdir(parents=True, exist_ok=True)
    for name, m in parts.items():
        rgba = np.zeros_like(key)
        rgba[m] = key[m]
        out[name] = rgba
    # hidden legs: a stocking column from the hip to the ankle (only shows if the hem lifts)
    for side in ("l", "r"):
        rgba = out[f"leg_{side}"]
        hip, knee, ank = J[f"upper_leg_{side}"], J[f"lower_leg_{side}"], J[f"foot_{side}"]
        im = Image.fromarray(rgba, "RGBA")
        d = ImageDraw.Draw(im)
        col = (236, 236, 242, 255)
        for (a, b, w0, w1) in ((hip, knee, 70, 52), (knee, (ank[0], ank[1] - 10), 52, 36)):
            ang = math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2
            ox, oy = math.cos(ang), math.sin(ang)
            poly = [(a[0] - ox * w0 / 2, a[1] - oy * w0 / 2), (a[0] + ox * w0 / 2, a[1] + oy * w0 / 2),
                    (b[0] + ox * w1 / 2, b[1] + oy * w1 / 2), (b[0] - ox * w1 / 2, b[1] - oy * w1 / 2)]
            d.polygon(poly, fill=col, outline=(90, 90, 110, 255))
        filled = np.asarray(im)
        hidden = (out["skirt"][..., 3] > 0) & (rgba[..., 3] == 0)
        rgba[hidden] = filled[hidden]
        # body on top of the legs' hip ends as well
        rgba[(out["body"][..., 3] > 0) & (rgba[..., 3] == 0) & (filled[..., 3] > 0)] = 0
    # skirt underlap below the body (apron, hands) so the waist never opens
    sk = out["skirt"]
    cur = sk[..., 3] > 0
    grow = ndimage.binary_dilation(cur, iterations=8) & ~cur & (out["body"][..., 3] > 0) & (yy > WAIST_Y)
    _, (iy, ix) = ndimage.distance_transform_edt(~cur, return_indices=True)
    sk[grow, :3] = sk[iy[grow], ix[grow], :3]
    sk[grow, 3] = 255
    for name, rgba in out.items():
        Image.fromarray(rgba, "RGBA").save(PKG / "layers" / f"{name}.png")
    return out


# ------------------------------------------------------------------ spec + mesh
def build_spec(variant: str) -> dict:
    adult = json.loads(ADULT_SIDE.read_text(encoding="utf-8"))
    bones = []
    for b in adult["skeleton"]["bones"]:
        b = dict(b)
        if b["bone_name"] in J:
            x, y = J[b["bone_name"]]
            b["joint_pos"] = [round(x / W, 5), round(y / H, 5)]
        else:      # keep the ADULT proportions for bones no prototype layer uses
            b["joint_pos"] = [b["joint_pos"][0] * 960 / W * 1.0, b["joint_pos"][1]]
        bones.append(b)
    if variant == "B":
        for name, (xw, xh) in PANELS.items():
            bones.append({"bone_name": name, "parent": "skirt_root", "joint_pos": [xw / W, WAIST_Y / H],
                          "angle_clamp": [-12, 12], "is_chain": False})
    gait = dict(adult["gait"])
    gait["frequency_hz"] = 1.6        # long skirt: small steps (speed / cadence = stride)
    if variant == "B":
        gait["skirt_follow_gain"] = 0.0        # panels are driven by the prototype instead
    layers = [
        {"id": "tail", "bind_bone": "root_hip", "influence_bones": ["root_hip"], "z_order": 0},
        {"id": "leg_r", "bind_bone": "upper_leg_r", "influence_bones": ["upper_leg_r", "lower_leg_r", "foot_r"],
         "z_order": 10, "weights": {"mode": "chain", "blend_px": [22, 28]},
         "rigid_below": [{"bone": "foot_r", "y": 1690, "blend": 18}], "min_component_px": 60},
        {"id": "leg_l", "bind_bone": "upper_leg_l", "influence_bones": ["upper_leg_l", "lower_leg_l", "foot_l"],
         "z_order": 15, "weights": {"mode": "chain", "blend_px": [22, 28]},
         "rigid_below": [{"bone": "foot_l", "y": 1695, "blend": 18}], "min_component_px": 60},
        {"id": "skirt", "bind_bone": "skirt_root", "z_order": 20, "grid_step": 24, "min_component_px": 200},
        {"id": "body", "bind_bone": "spine", "influence_bones": ["spine"], "z_order": 30},
    ]
    sk = layers[3]
    if variant == "A":
        sk["influence_bones"] = ["root_hip", "skirt_root", "skirt_hem_l", "skirt_hem_r"]
        sk["weights"] = {"mode": "skirt", "root": "skirt_root", "hem_l": "skirt_hem_l", "hem_r": "skirt_hem_r",
                         "y0": 820, "y1": 1150, "cx": 520, "half_w": 140}
    else:
        sk["influence_bones"] = ["skirt_root"] + list(PANELS)
    return {"source_facing": 1, "texture_mipmaps": True, "ground_anchor_y_px": 1761,
            "rest_pose_angles": {}, "view": adult.get("view", {}),
            "skeleton": {"bones": bones, "source_reference": {"image_size_px": [W, H]}},
            "face_mechanics": {}, "physics_presets": {}, "contact_markers": MARKERS,
            "gait": gait, "layers": layers}


def panel_weights(pts: np.ndarray) -> tuple[list, list]:
    """Variant B weights: waist band on the root, below it the cloth blends between the two
    neighbouring panels (panel lines run from their waist x to their hem x)."""
    names = list(PANELS)
    out_b, out_w = [], []
    for x, y in pts:
        t = float(np.clip((y - WAIST_Y) / (HEM_Y - WAIST_Y), 0, 1))
        xs = np.array([PANELS[n][0] + (PANELS[n][1] - PANELS[n][0]) * t for n in names])
        hem = float(np.clip((y - (WAIST_Y + 40)) / 200, 0, 1))
        hem = hem * hem * (3 - 2 * hem)
        k = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
        u = float(np.clip((x - xs[k - 1]) / max(xs[k] - xs[k - 1], 1), 0, 1))
        u = u * u * (3 - 2 * u)
        w = {"skirt_root": 1 - hem, names[k - 1]: hem * (1 - u)}
        w[names[k]] = w.get(names[k], 0) + hem * u
        w = {b_: v for b_, v in w.items() if v > 1e-4}
        tot = sum(w.values())
        out_b.append(list(w))
        out_w.append([round(v / tot, 6) for v in w.values()])
    return out_b, out_w


def build_package(variant: str) -> Path:
    from mesh_generator import generate_layer_mesh
    pkg = PKG / variant
    (pkg / "mesh").mkdir(parents=True, exist_ok=True)
    (pkg / "adult").mkdir(parents=True, exist_ok=True)
    if (pkg / "layers").exists():
        shutil.rmtree(pkg / "layers")
    shutil.copytree(PKG / "layers", pkg / "layers")
    spec = build_spec(variant)
    (pkg / "spec.json").write_bytes(json.dumps(spec, indent=1).encode("utf-8"))
    layers = []
    for l in spec["layers"]:
        m = generate_layer_mesh(l, spec["skeleton"], str(pkg / "layers" / f"{l['id']}.png"), (W, H))
        if variant == "B" and l["id"] == "skirt":
            m["weight_bones"], m["weight_values"] = panel_weights(np.asarray(m["vertices"]))
        layers.append(m)
    (pkg / "mesh" / "mesh_data.json").write_bytes(json.dumps(
        {"spec": 1, "image_size_px": [W, H], "layers": layers}).encode("utf-8"))
    fig = PKG / "fig.png"
    Image.open(KEY).save(fig)
    (pkg / "adult" / "manifest.json").write_bytes(json.dumps({
        "spec": 1, "figures": {"healthy_neutral": "../../fig.png"},
        "skinned": {"spec_file": "../spec.json", "mesh_file": "../mesh/mesh_data.json", "layers_dir": "../layers"},
    }, indent=1).encode("utf-8"))
    return pkg


# ------------------------------------------------------------------ variant B drive
class PanelDrive:
    """In this side view the A-line skirt is far wider than the stride: the legs never reach the
    front or back cloth, only the hem centre sits over the shins (shin dx +-130 px, the two legs
    in anti-phase, so one panel following "the legs" averages to ~0). So the hem centre is split:
      mid_f (front half over the legs) - follows the forward shin
      mid_b (back half)                - follows the rear shin
      front / back                     - follow their neighbour at reduced gain, softer spring
    gain < 1 caps the cloth stretch between mid_b and mid_f. All panels hinge at the waist:
    rotation keeps the cloth length."""

    def __init__(self, gain=0.45, side_gain=0.6, freq=(3.0, 1.8), halflife=(0.10, 0.18)):
        from pet.rig.motion import spring_from_frequency
        self.k_mid = spring_from_frequency(freq[0], halflife[0])
        self.k_side = spring_from_frequency(freq[1], halflife[1])
        self.gain, self.side_gain = gain, side_gain
        self.st = {n: [0.0, 0.0] for n in PANELS}
        self.rest = None

    @staticmethod
    def _spring(st, target, k, dt):
        st[1] += (k[0] * (target - st[0]) - k[1] * st[1]) * dt
        st[0] += st[1] * dt
        return st[0]

    def step(self, shin: dict, thigh_deg: dict, dt: float) -> dict:
        if self.rest is None:
            self.rest = {s_: p[0] for s_, p in shin.items()}
        dx = sorted(shin[s_][0] - self.rest[s_] for s_ in shin)
        L = HEM_Y - WAIST_Y
        to_ang = lambda d: -math.degrees(math.atan2(d, L))     # + dx (forward) -> negative angle
        ang = {"skirt_mid_f": self._spring(self.st["skirt_mid_f"], to_ang(self.gain * dx[1]), self.k_mid, dt),
               "skirt_mid_b": self._spring(self.st["skirt_mid_b"], to_ang(self.gain * dx[0]), self.k_mid, dt)}
        ang["skirt_front"] = self._spring(self.st["skirt_front"], self.side_gain * ang["skirt_mid_f"], self.k_side, dt)
        ang["skirt_back"] = self._spring(self.st["skirt_back"], self.side_gain * ang["skirt_mid_b"], self.k_side, dt)
        return ang


# ------------------------------------------------------------------ run
def knee_world(rt, angles_deg: dict, tx: dict, ty: dict) -> dict:
    nb = len(rt.bones)
    pa = np.zeros(nb, np.float32)
    px = np.zeros(nb, np.float32)
    py = np.zeros(nb, np.float32)
    for b, v in angles_deg.items():
        if b in rt.bone_index:
            pa[rt.bone_index[b]] = v
    for b, v in tx.items():
        if b in rt.bone_index:
            px[rt.bone_index[b]] = v
    for b, v in ty.items():
        if b in rt.bone_index:
            py[rt.bone_index[b]] = v
    M = rt.skinning_matrices(pa, px, py, 0.0, 0.0)
    out = {}
    for s in ("l", "r"):
        i = rt.bone_index[f"lower_leg_{s}"]
        k = np.array([*J[f"lower_leg_{s}"], 1.0])
        # 70 % down the shin, just above the hem: the shin pushes the lower skirt
        a = np.array([*J[f"foot_{s}"], 1.0])
        p = k + 0.7 * (a - k)
        m = M[i] if M is not None else np.eye(3)
        q = (m[:2, :3] @ p) if m.shape[-1] == 3 else p[:2]
        out[s] = (float(q[0]), float(q[1]))
    return out


def sigma_stats(rt, layer_id: str):
    L = next(l for l in rt.layers if l.layer_id == layer_id)
    V = L.rest[:, :2].astype(np.float64)
    T = L.triangles.reshape(-1, 3).astype(np.int64)
    E = np.stack([V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]], -1)
    ok = np.abs(np.linalg.det(E)) > 1e-6
    T, Einv = T[ok], np.linalg.inv(E[ok])
    X = rt.deform(L, L.rest)[:, :2].astype(np.float64)
    F = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]], -1) @ Einv
    s = np.linalg.svd(F, compute_uv=False)
    folds = int((np.linalg.det(F) < 0).sum())
    return float(s[:, 0].max()), float(s[:, 1].min()), folds, X


def run_variant(variant: str) -> dict:
    from pet.rig.gait import GaitSolver
    from pet.rig.motion import MotionFrame
    from pet.rig.skinned_mesh_item import RigRuntime
    from render_rig_rest import RigRenderer
    pkg = build_package(variant)
    spec = json.loads((pkg / "spec.json").read_text(encoding="utf-8"))
    rt = RigRuntime.load(str(pkg / "spec.json"), str(pkg / "mesh" / "mesh_data.json"), str(pkg / "layers"))
    g = GaitSolver(spec, 256 / H)
    drive = PanelDrive() if variant == "B" else None
    r = RigRenderer("adult", str(pkg / "adult"))  # build_rig_window reads <root>/<stage>
    frames, sig_hi, sig_lo, folds = [], 1.0, 1.0, 0
    hem_front, shoe_x = [], []
    skirt = next(l for l in rt.layers if l.layer_id == "skirt")
    rest_v = skirt.rest[:, :2]
    hem_idx = np.where((rest_v[:, 1] > HEM_Y - 40) & (rest_v[:, 0] > 420) & (rest_v[:, 0] < 700))[0]
    dt = 1 / 60
    n = int(2 / 1.6 * 60) + 90            # warm-up + two cycles at 1.6 Hz
    try:
        for i in range(n):
            o = g.update(dt, 120.0, (round(g.window_x_float), 0))
            ang = {b: math.degrees(v) for b, v in o.bone_rotations.items()}
            tx = {"root_hip": o.pelvis_offset[0]}
            ty = {"root_hip": o.pelvis_offset[1]}
            for b, (ox, oy) in o.bone_offsets.items():
                tx[b] = tx.get(b, 0) + ox
                ty[b] = ty.get(b, 0) + oy
            if drive is not None:
                ang.update(drive.step(knee_world(rt, ang, tx, ty),
                                      {"l": ang.get("upper_leg_l", 0.0), "r": ang.get("upper_leg_r", 0.0)}, dt))
            if i < 90 or i % 2:
                continue
            nb = len(rt.bones)
            pa = np.array([ang.get(b.name, 0.0) for b in rt.bones], np.float32)
            px = np.array([tx.get(b.name, 0.0) for b in rt.bones], np.float32)
            py = np.array([ty.get(b.name, 0.0) for b in rt.bones], np.float32)
            rt.skinning_matrices(pa, px, py, 0.0, 0.0)
            hi, lo, fo, X = sigma_stats(rt, "skirt")
            sig_hi, sig_lo, folds = max(sig_hi, hi), min(sig_lo, lo), folds + fo
            hem_front.append(float(X[hem_idx, 0].mean()))
            leg = next(l for l in rt.layers if l.layer_id == "leg_l")
            legv = rt.deform(leg, leg.rest)
            shoe_x.append(float(legv[leg.rest[:, 1] > HEM_Y, 0].mean()))
            fr = MotionFrame(bone_angles=dict(ang), bone_tx=dict(tx), bone_ty=dict(ty),
                             blink_progress=0.0, look_at=(0.0, 0.0))
            im = r.render(fr)
            bg = Image.new("RGBA", im.size, (240, 242, 246, 255))
            bg.alpha_composite(im)
            frames.append(bg.convert("RGB").crop((60, 560, 960, 1800)))
    finally:
        r.close()
    small = [f.resize((300, 413), Image.Resampling.LANCZOS) for f in frames]
    small[0].save(OUT / f"walk_{variant}.gif", save_all=True, append_images=small[1:], duration=33, loop=0)
    pick = np.linspace(0, len(frames) // 2 - 1, 8).astype(int)
    sheet = Image.new("RGB", (450 * 8, 620), (255, 255, 255))
    for k, j in enumerate(pick):
        sheet.paste(frames[j].resize((450, 620), Image.Resampling.LANCZOS), (450 * k, 0))
    sheet.save(OUT / f"sheet_{variant}.png")
    hf, sx = np.array(hem_front), np.array(shoe_x)
    return {"sigma_max": round(sig_hi, 3), "sigma_min": round(sig_lo, 3), "folds": folds,
            "hem_centre_swing_px": round(float(hf.max() - hf.min()), 1),
            "near_shoe_travel_px": round(float(sx.max() - sx.min()), 1),
            # how much of the shoe's travel the hem follows (0 = frozen skirt, legs slide under it)
            "hem_follow_ratio": round(float(np.corrcoef(hf, sx)[0, 1]) * float((hf.max() - hf.min()) / max(sx.max() - sx.min(), 1)), 3),
            "frames": len(frames)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_layers()
    rep = {v: run_variant(v) for v in ("A", "B")}
    (OUT / "metrics.json").write_bytes(json.dumps(rep, indent=1).encode("utf-8"))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
