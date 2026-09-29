"""Split a hinged limb layer into rigid pieces joined by disc caps (plan D2, docs/ADULT行走修复-2026-09-29.md).

A single mesh bent at a hinge must either fold on the inner side or stretch the outer contour by
~2x at human-range angles (plan D1: det F = 1 + theta*|grad w|*l). Rigid pieces cannot distort
(sigma = 1) or fold. Pixels are assigned to the segment of their nearest chain segment, split at
each joint across the bisector line (the same rule as mesh_generator.chain_weights, zero band).
Around every joint, the disc (pivot, radius = distance to the limb outline) belongs to BOTH
pieces, copied from the art: overlapping identical colours composite to the same colour at any
alpha, so the rest pose stays pixel-exact, and when the joint bends the parent's disc forms the
round elbow/knee between the two straight edges instead of a gap.

split(layer_rgba, joints, radii) -> list of RGBA arrays, one per chain bone (parent first).
"""
from __future__ import annotations

import numpy as np


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def outline_radius(alpha: np.ndarray, pivot: np.ndarray, child_dir: np.ndarray, cap: float = 60.0) -> float:
    """Distance from the pivot to the nearer limb outline, perpendicular to the child bone."""
    d = child_dir / np.linalg.norm(child_dir)
    n = np.array([-d[1], d[0]])
    h, w = alpha.shape
    best = cap
    for sgn in (1, -1):
        for t in range(1, int(cap) + 1):
            x, y = np.rint(pivot + sgn * t * n).astype(int)
            if not (0 <= x < w and 0 <= y < h) or not alpha[y, x]:
                best = min(best, float(t))
                break
    return best


def split(rgba: np.ndarray, joints: list[np.ndarray], radii: list[float] | None = None) -> list[np.ndarray]:
    """joints = rest joint positions of the chain bones (parent first); returns one RGBA per bone."""
    n = len(joints)
    J = [np.asarray(j, np.float64) for j in joints]
    alpha = rgba[..., 3] > 0
    ys, xs = np.nonzero(alpha)
    p = np.stack([xs, ys], 1).astype(np.float64)
    d = [(J[k + 1] - J[k]) / np.linalg.norm(J[k + 1] - J[k]) for k in range(n - 1)]
    normals = [None] + [(d[k - 1] + d[k]) / np.linalg.norm(d[k - 1] + d[k]) if k <= n - 2 else d[k - 1]
                        for k in range(1, n)]
    k = np.argmin(np.stack([_seg_dist(p, J[i], J[i + 1]) for i in range(n - 1)], 1), 1)
    owner = k.copy()
    past_end = np.einsum("ij,ij->i", p - np.stack([J[i + 1] for i in k]), np.stack([normals[i + 1] for i in k])) > 0
    owner[past_end] += 1
    before = (k > 0) & (np.einsum("ij,ij->i", p - np.stack([J[i] for i in k]),
                                  np.stack([normals[i] if i > 0 else np.zeros(2) for i in k])) < 0)
    owner[before & ~past_end] -= 1
    if radii is None:
        radii = [outline_radius(alpha, J[j], J[j + 1] - J[j] if j < n - 1 else J[j] - J[j - 1]) for j in range(1, n)]
    pieces = []
    for b in range(n):
        m = np.zeros(alpha.shape, bool)
        sel = owner == b
        m[ys[sel], xs[sel]] = True
        for j, r in zip(range(1, n), radii):          # disc caps at this bone's joints
            if j in (b, b + 1):
                disc = (xs - J[j][0]) ** 2 + (ys - J[j][1]) ** 2 <= r * r
                m[ys[disc], xs[disc]] = True
        out = np.zeros_like(rgba)
        out[m] = rgba[m]
        pieces.append(out)
    return pieces
