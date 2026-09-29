"""Regression gates for the side rig: forward knees, planted support, rest recovery.

The old contact-only tests passed with backwards knees and feet left in mid-stride.
These checks include start acceleration, stop phase, follow speed and timing jitter.
"""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pet.rig.gait import GaitSolver, GaitPhaseState


class SideGaitTests(unittest.TestCase):
    def test_actual_walk_mesh_does_not_fold(self):
        from pet.rig.skinned_mesh_item import RigRuntime
        pkg = ROOT / "assets/rig_adult_walk_v1"
        spec = json.loads((pkg / "spec.json").read_text(encoding="utf-8"))
        rt = RigRuntime.load(str(pkg / "spec.json"), str(pkg / "mesh/mesh_data.json"), str(pkg / "layers"))
        legs = [layer for layer in rt.layers if layer.layer_id in ("leg_l", "leg_r")]
        skirt = next(layer for layer in rt.layers if layer.layer_id == "skirt")
        from PIL import Image
        opaque = {}                      # vertices whose own texture is painted (grid corners may be empty)
        for layer in legs:
            alpha = np.asarray(Image.open(pkg / "layers_full" / f"{layer.layer_id}.png"))[..., 3]
            xy = np.clip(np.rint(layer.rest[:, :2]).astype(int), 0, [alpha.shape[1] - 1, alpha.shape[0] - 1])
            opaque[layer.layer_id] = alpha[xy[:, 1], xy[:, 0]] > 127

        def areas(points, triangles):
            a, b, c = (points[triangles[:, i], :2] for i in range(3))
            return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

        for speed in (120, 200):
            g = GaitSolver(spec, 256 / 1696)
            for i in range(480):
                o = g.update(1 / 60, speed if i < 300 else 0, (round(g.window_x_float), 0))
                angles = np.array([np.degrees(o.bone_rotations.get(b.name, 0)) for b in rt.bones], np.float32)
                tx, ty = np.zeros_like(angles), np.zeros_like(angles)
                tx[rt.bone_index["root_hip"]], ty[rt.bone_index["root_hip"]] = o.pelvis_offset
                rt.skinning_matrices(angles, tx, ty, 0, 0)
                for layer in legs:
                    triangles = layer.triangles.reshape(-1, 3)
                    deformed = rt.deform(layer, layer.rest).copy()
                    products = areas(layer.rest, triangles) * areas(deformed, triangles)
                    self.assertTrue(np.all(products >= 0), (speed, i, layer.layer_id))
                    # The upper leg must stay tucked beneath the skirt as the thigh swings.
                    # 2026-09-28 round 2: legs are no longer pinned to the pelvis (the pin band
                    # 1100-1400 contained the knee and put a false joint in the calf); the leg
                    # tops swing with the thigh under the hem (skirt hem follows the thighs).
                    # Check: every leg-top vertex (y 1100-1150, hidden in the rest pose) stays
                    # covered by the deformed skirt mesh.
                    top = (layer.rest[:, 1] >= 1100) & (layer.rest[:, 1] <= 1150) & opaque[layer.layer_id]
                    sk = rt.deform(skirt, skirt.rest)[:, :2].astype(np.float64)
                    tri = skirt.triangles.reshape(-1, 3)
                    a, b, c = sk[tri[:, 0]], sk[tri[:, 1]], sk[tri[:, 2]]
                    for q in deformed[top, :2].astype(np.float64):
                        v0, v1, v2 = b - a, c - a, q - a
                        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
                        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
                        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
                        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
                        covered = np.any((u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6))
                        self.assertTrue(covered, (speed, i, layer.layer_id, q))

    def test_walk_and_park(self):
        spec = json.loads((ROOT / "assets/rig_adult_walk_v1/spec.json").read_text(encoding="utf-8"))
        for speed in (60, 120, 200):
            for hz, jitter in ((60, False), (30, False), (30, True)):
                for stop_at in (4.7, 5.1):
                    with self.subTest(speed=speed, hz=hz, jitter=jitter, stop_at=stop_at):
                        g = GaitSolver(spec, 256 / 1696)
                        rng = np.random.default_rng(7)
                        t, worst_drift, park_seen = 0.0, 0.0, False
                        idle_error, dips = [], []
                        while t < 9:
                            dt = (rng.uniform(.5, 1.5) if jitter else 1) / hz
                            t += dt
                            o = g.update(dt, speed if .5 < t < stop_at else 0,
                                         (round(g.window_x_float), 0))
                            worst_drift = max(worst_drift, o.foot_slide_drift_px)
                            if g.state is GaitPhaseState.WALK_LOOP:
                                self.assertGreater(g._pelvis_rot, 0, "right-facing torso must lean forward")
                                dips.append(g._dip * g.scale)
                                for side, foot in g._feet.items():
                                    leg = g.legs[side]
                                    th, kn, _ = foot.applied
                                    thigh = leg.thigh_rest_angle + g._pelvis_rot + th
                                    shin = leg.shin_rest_angle + g._pelvis_rot + th + kn
                                    # Right-facing knee flexion must be positive (shin behind thigh).
                                    self.assertGreater(shin - thigh, 0)
                            if g.state is GaitPhaseState.WALK_PARK:
                                park_seen = True
                                self.assertTrue(any(f.contact.is_locked for f in g._feet.values()))
                            if t > stop_at + 3 and g.state is GaitPhaseState.IDLE_SIDE:
                                idle_error.extend(float(np.linalg.norm(f.ankle_now - g.legs[s].ankle_rest))
                                                  for s, f in g._feet.items())
                        self.assertTrue(park_seen)
                        self.assertTrue(idle_error)
                        self.assertLess(max(idle_error) * g.scale, .1)
                        self.assertLess(worst_drift, .5)
                        self.assertLess(g.knee_rate_max_observed, 20)
                        self.assertLess(max(dips) - min(dips), 4)


if __name__ == "__main__":
    unittest.main()
