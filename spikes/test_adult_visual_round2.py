"""Meaningful regressions for the second ADULT visual repair."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "spikes")]
PKG = ROOT / "assets/rig_adult_walk_v1"
OUT = ROOT / "spikes/_qa/adult_visual_round2_2026-09-30"


class VisualRepair(unittest.TestCase):
    def setUp(self):
        from pet.rig.skinned_mesh_item import RigRuntime
        self.spec = json.loads((PKG / "spec.json").read_text())
        self.rt = RigRuntime.load(str(PKG / "spec.json"), str(PKG / "mesh/mesh_data.json"), str(PKG / "layers"))
        OUT.mkdir(parents=True, exist_ok=True)

    def pose(self, angles):
        a = np.array([angles.get(b.name, 0) for b in self.rt.bones], np.float32)
        z = np.zeros_like(a)
        self.rt.skinning_matrices(a, z, z, 0, 0)

    def test_curve_keeps_rest_and_rigid_motion(self):
        for side in "lr":
            leg = next(x for x in self.rt.layers if x.layer_id == "leg_" + side)
            self.assertIsNotNone(leg.joint_curve)
            for angle in (0, -20, 20):
                self.pose({"upper_leg_" + side: angle, "root_hip": 3})
                expected = leg.rest @ self.rt.M[self.rt.bone_index["upper_leg_" + side]].T
                actual = self.rt.deform(leg, leg.rest)
                self.assertLess(float(np.max(np.abs(actual[:, :2]-expected[:, :2]))), .001)

    def test_shoes_stay_exactly_on_original_fk(self):
        self.pose({"upper_leg_l": -25, "lower_leg_l": 65, "foot_l": -35,
                   "upper_leg_r": 15, "lower_leg_r": 60, "foot_r": -30})
        for side in "lr":
            leg = next(x for x in self.rt.layers if x.layer_id == "leg_" + side)
            foot = self.rt.bone_index["foot_" + side]
            col = int(np.where(leg.bone_idx == foot)[0][0])
            full = leg.weights[:, col] >= .99999
            expected = leg.rest @ self.rt.M[foot].T
            self.assertGreater(int(full.sum()), 30)
            self.assertLess(float(np.max(np.abs(self.rt.deform(leg, leg.rest)[full, :2]-expected[full, :2]))), .001)

    def test_fin_lobes_are_rigid(self):
        tail = next(x for x in self.rt.layers if x.layer_id == "tail")
        fork = (tail.rest[:, 1] < 1040) & (tail.rest[:, 0] < 300)
        self.assertGreater(int(fork.sum()), 10)
        self.pose({"tail_01": 8, "tail_02": 10, "tail_03": 12, "tail_fluke": 15})
        actual = self.rt.deform(tail, tail.rest)[fork, :2]
        expected = (tail.rest @ self.rt.M[self.rt.bone_index["tail_fluke"]].T)[fork, :2]
        self.assertLess(float(np.max(np.abs(actual-expected))), .001)

    def test_skirt_has_lag_and_bounded_swing(self):
        from pet.rig.gait import GaitSolver
        g = GaitSolver(self.spec, 256/1696)
        lag = []
        for i in range(420):
            o = g.update(1/60, 120 if i < 300 else 0, (round(g.window_x_float), 0))
            for name in ("skirt_hem_l", "skirt_hem_r"):
                self.assertLessEqual(abs(np.degrees(o.bone_rotations.get(name, 0))), 8.0001)
            thighs = [o.bone_rotations.get("upper_leg_"+s, 0) for s in "lr"]
            target = np.clip(.4 * min(thighs), -np.radians(8), np.radians(8))
            lag.append(abs(o.bone_rotations.get("skirt_hem_r", 0)-target))
        self.assertGreater(max(lag), .02)

    def simulate_reverse(self, spec, change=6.4, cancel=None):
        from pet.rig.side_locomotion import SideLocomotion, TurnClip
        loco = SideLocomotion(spec, TurnClip(str(PKG / "clips/turn_front_to_side_h256")),
                              TurnClip(str(PKG / "clips/turn_side_to_front_h256")), 256/1696)
        x, first, left, start = 200., None, False, None
        for i in range(840):
            t = (i+1)/60
            vx = 120 if t < change else -120
            if cancel is not None and t >= change + cancel:
                vx = 0
            prev = x
            f = loco.update(1/60, vx, x)
            if f.window_x is not None:
                x = f.window_x
            if start is None and x > 200.1:
                start = t
            left |= f.facing == -1
            if first is None and t >= change and f.facing == -1 and x < prev-.1:
                first = t-change
        return first, left, start

    def test_reverse_is_faster_without_skipping_park(self):
        old = deepcopy(self.spec)
        old.pop("locomotion", None)
        facts = []
        for change in (6.1, 6.4, 6.7):
            before, _, old_start = self.simulate_reverse(old, change)
            after, _, new_start = self.simulate_reverse(self.spec, change)
            self.assertIsNotNone(after)
            self.assertLess(after, before-1)
            self.assertLess(after, 2.8)
            self.assertLess(new_start, old_start-.2)
            facts.append({"request_at": change, "reverse_delay_before_s": before,
                          "reverse_delay_after_s": after, "start_before_s": old_start, "start_after_s": new_start})
        (OUT / "timing_before_after.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    def test_cancelled_reverse_does_not_walk_left(self):
        first, left, _ = self.simulate_reverse(self.spec, cancel=.05)
        self.assertIsNone(first)
        self.assertFalse(left)


if __name__ == "__main__":
    unittest.main()
