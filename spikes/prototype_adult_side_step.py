"""Numerical feasibility study for front-view ADULT side steps.

This does not render or change production animation. It verifies world-space
foot locks, non-crossing foot targets, and reachable two-bone IK at one modest
speed. Texture deformation and animation transitions still need visual QA.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'spikes/_qa/adult_review'


def smooth5(u):
    return u ** 3 * (10 + u * (-15 + 6 * u))


def smooth_max(a, b, epsilon=0.2):
    return (a + b + math.sqrt((a - b) ** 2 + epsilon ** 2)) / 2


def solve(hip, ankle, a, b, sign):
    delta = ankle - hip
    distance = np.linalg.norm(delta)
    assert abs(a - b) < distance < a + b
    unit = delta / distance
    along = (a * a - b * b + distance * distance) / (2 * distance)
    height = math.sqrt(max(0, a * a - along * along))
    knee = hip + unit * along + sign * np.array([-unit[1], unit[0]]) * height
    return knee


def main():
    raw = json.loads((ROOT / 'assets/rig_adult/spec.json').read_text(encoding='utf-8'))
    scale = 256 / 1696
    joints = {b['bone_name']: np.array(b['joint_pos']) * [960 * scale, 256] for b in raw['skeleton']['bones']}
    speed, frequency, stance, lift, minimum_gap = 45., 1.1, .6, 3.5, 18.
    cycle_distance = speed / frequency
    centers = minimum_gap + cycle_distance / 2
    half_travel = speed * stance / frequency / 2
    swing_duration = (1 - stance) / frequency
    pelvis_x = joints['root_hip'][0]
    foot_heights = {s: (1608 if s == 'l' else 1607) * scale - joints['foot_' + s][1] for s in ('l', 'r')}
    floor = 256.
    floor_shift = floor - 1608 * scale
    lengths = {s: (np.linalg.norm(joints['lower_leg_' + s] - joints['upper_leg_' + s]),
                   np.linalg.norm(joints['foot_' + s] - joints['lower_leg_' + s])) for s in ('l', 'r')}

    def foot(t, side):
        phase = (t * frequency + (.6 if side == 'r' else .1)) % 1
        center = centers / 2 * (1 if side == 'r' else -1)
        if phase < stance:
            x = center + half_travel - speed * phase / frequency
            h = 0.
        else:
            u = (phase - stance) / (1 - stance)
            tangent = -speed * swing_duration
            x = center - half_travel + tangent * u + (2 * half_travel - tangent) * smooth5(u)
            h = 64 * lift * u ** 3 * (1 - u) ** 3
        ankle = np.array([pelvis_x + speed * t + x, floor - foot_heights[side] - h])
        return phase, ankle, h

    summaries = []
    max_lock_speed = 0.
    max_length_error = 0.
    for t in np.linspace(0, 2 / frequency, 2400, endpoint=False):
        targets = {s: foot(t, s) for s in ('l', 'r')}
        lower = 1.
        for s in ('l', 'r'):
            hip = joints['upper_leg_' + s] + [speed * t, floor_shift]
            delta_x = targets[s][1][0] - hip[0]
            reach = sum(lengths[s]) * .997
            assert abs(delta_x) < reach
            required = targets[s][1][1] - math.sqrt(reach ** 2 - delta_x ** 2) - hip[1]
            lower = smooth_max(lower, required)
        row = {'pelvis_lower_px': lower, 'gap_px': targets['r'][1][0] - targets['l'][1][0]}
        for s in ('l', 'r'):
            phase, ankle, h = targets[s]
            hip = joints['upper_leg_' + s] + [speed * t, floor_shift + lower]
            a, b = lengths[s]
            knee = solve(hip, ankle, a, b, 1 if s == 'l' else -1)
            max_length_error = max(max_length_error, abs(np.linalg.norm(knee - hip) - a), abs(np.linalg.norm(ankle - knee) - b))
            first = math.atan2(*(knee - hip)[::-1])
            second = math.atan2(*(ankle - knee)[::-1])
            rest_first = math.atan2(*(joints['lower_leg_' + s] - joints['upper_leg_' + s])[::-1])
            rest_second = math.atan2(*(joints['foot_' + s] - joints['lower_leg_' + s])[::-1])
            row['thigh_' + s] = math.degrees(first - rest_first)
            row['knee_' + s] = math.degrees((second - first) - (rest_second - rest_first))
            row['ankle_counter_' + s] = -row['thigh_' + s] - row['knee_' + s]
            if .001 < phase < stance - .001:
                before = foot(t - 1e-5, s)[1]
                after = foot(t + 1e-5, s)[1]
                max_lock_speed = max(max_lock_speed, float(np.linalg.norm(after - before) / 2e-5))
                assert h == 0
        summaries.append(row)
    ranges = {key: [min(r[key] for r in summaries), max(r[key] for r in summaries)] for key in summaries[0]}
    assert ranges['gap_px'][0] >= minimum_gap - 1e-6
    assert max_lock_speed < 1e-5
    assert max_length_error < 1e-8
    result = {'scope': 'Foot targets and ideal two-bone IK only; no skin, perspective, start/stop or clamp validation',
              'speed_px_s_at_256': speed, 'cycles_per_second': frequency,
              'stance_ratio': stance, 'lift_px_at_256': lift, 'samples': len(summaries),
              'max_stance_foot_speed_px_s': max_lock_speed, 'max_bone_length_error_px': max_length_error,
              'ranges': ranges}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'side_step_feasibility.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
