"""
Alpha-covering Grid Mesh Generator for Desktop Pet Skeletal Skinning.
Generates gap-free triangle meshes and calculates bone weights for rig layers
based on young_rig_spec.json and transparent layer PNGs.
Outputs assets/rig_young/mesh/mesh_data.json matching SkinnedMeshItem contract.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from PIL import Image
from scipy import ndimage


def dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (N, 2) to segment ab (2,)."""
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-6:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(np.sum((p - a) * ab, axis=1) / ab_len_sq, 0.0, 1.0)
    projection = a + t[:, None] * ab
    return np.linalg.norm(p - projection, axis=1)


def _smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def chain_weights(pts: np.ndarray, bones: list[str], bones_dict: dict, img_size: tuple[int, int],
                  cfg: dict) -> tuple[list[list[str]], list[list[float]]]:
    """Rigid-segment weights for a parent->child bone chain (long limbs, tail, hair strands).

    The inverse-distance default spreads a joint's blend over a whole limb segment (measured on the
    ADULT side rig: 51% of leg vertices had no bone >= 0.9, the knee blend ran ~200 px), so bending
    turns limbs into rubber hoses. Here each vertex belongs to the bone of its nearest chain segment
    and only blends with the neighbouring bone inside +-blend_px of the joint, measured across the
    joint's bisector line. Using the nearest segment (not an infinite cut line alone) keeps curled
    chains (the tail) from being sliced by a far joint.

    cfg: {"mode": "chain", "blend_px": float | [per joint 1..n-1],
          "inner": {"<joint bone name>": {"side": +1|-1, "blend_px": wider, "half_width_px": w}}}

    "inner": a hinge (knee, elbow) bends one way only. Its inner (flexion) side creases - the calf
    back meets the thigh back - and a narrow band folds triangles there (ADULT side leg: the knee
    pivot sits at the front of the leg so IK keeps a forward knee, the knee back is ~87 px from it,
    and a 26 px band folds at 25 deg; dual-complex skinning did not help - it is a sweep, not a
    chord collapse). The inner side gets a wider band, ramped by the vertex's lateral offset from
    the chain axis (side = sign of cross(bone dir, vertex offset) on the inner side), so the outer
    contour (kneecap) stays crisp while the crease compresses (170 px: no fold up to 90 deg).
    """
    w_img, h_img = img_size
    n = len(bones)
    for a, b in zip(bones, bones[1:]):
        if bones_dict[b].get("parent") != a:
            raise ValueError(f"chain weights need a parent->child bone list: {bones}")
    J = np.array([[bones_dict[b]["joint_pos"][0] * w_img, bones_dict[b]["joint_pos"][1] * h_img] for b in bones])
    d = [(J[k + 1] - J[k]) / max(np.linalg.norm(J[k + 1] - J[k]), 1e-6) for k in range(n - 1)]
    normals = [None]
    for k in range(1, n):
        v = d[k - 1] + d[k] if k <= n - 2 else d[k - 1]
        normals.append(v / max(np.linalg.norm(v), 1e-6))
    bp = cfg.get("blend_px", 24.0)
    blend = [None] + (list(bp) if isinstance(bp, (list, tuple)) else [float(bp)] * (n - 1))
    inner = cfg.get("inner", {})

    def band(j: int, p: np.ndarray) -> tuple[float, float]:
        """(parent-side, child-side) blend extents at joint j for vertex p.

        On a hinge's inner side the band widens mostly into the PARENT segment (the crease is taken
        up by the back of the thigh / front of the upper arm) and only "down_px" into the child: a
        band reaching down the calf bends the shin (round-2 gate: shin straight within 8 deg),
        while a child side that stays fully rigid concentrates the sweep and folds earlier.
        """
        spec_i = inner.get(bones[j])
        if not spec_i:
            return blend[j], blend[j]
        axis = normals[j]
        lateral = float(axis[0] * (p - J[j])[1] - axis[1] * (p - J[j])[0]) * spec_i["side"]
        s = _smooth(lateral / float(spec_i.get("half_width_px", 40.0)))
        down = float(spec_i.get("down_px", blend[j]))
        return blend[j] + (float(spec_i["blend_px"]) - blend[j]) * s, blend[j] + (down - blend[j]) * s

    out_b, out_w = [], []
    for p in pts:
        k = int(np.argmin([dist_point_to_segment(p[None, :], J[i], J[i + 1])[0] for i in range(n - 1)]))
        w = {}
        d_end = float(np.dot(p - J[k + 1], normals[k + 1]))
        up, down = band(k + 1, p)
        if d_end > -up:
            t = _smooth((d_end + up) / (up + down))
            w = {bones[k]: 1 - t, bones[k + 1]: t}
        elif k > 0 and float(np.dot(p - J[k], normals[k])) < band(k, p)[1]:
            up, down = band(k, p)
            t = _smooth((float(np.dot(p - J[k], normals[k])) + up) / (up + down))
            w = {bones[k - 1]: 1 - t, bones[k]: t}
        else:
            w = {bones[k]: 1.0}
        w = {b: round(v, 6) for b, v in w.items() if v > 1e-4}
        s = sum(w.values())
        out_b.append(list(w))
        out_w.append([round(v / s, 6) for v in w.values()])
    return out_b, out_w


def skirt_weights(pts: np.ndarray, cfg: dict) -> tuple[list[list[str]], list[list[float]]]:
    """Skirt: waist band on the skirt root, lower skirt split front/back onto the two hem bones.

    cfg: {"mode": "skirt", "root": bone, "hem_l": bone, "hem_r": bone, "y0": px, "y1": px,
          "cx": px, "half_w": px}; hem share ramps 0 -> 1 from y0 to y1, the front/back split ramps
    across cx +- half_w (the inverse-distance default left 76% of skirt vertices blended across all
    four bones, so a hem swing dragged the waist).
    """
    out_b, out_w = [], []
    for x, y in pts:
        hem = _smooth((y - cfg["y0"]) / (cfg["y1"] - cfg["y0"]))
        right = _smooth((x - (cfg["cx"] - cfg["half_w"])) / (2 * cfg["half_w"]))
        w = {cfg["root"]: 1 - hem, cfg["hem_l"]: hem * (1 - right), cfg["hem_r"]: hem * right}
        w = {b: v for b, v in w.items() if v > 1e-4}
        s = sum(w.values())
        out_b.append(list(w))
        out_w.append([round(v / s, 6) for v in w.values()])
    return out_b, out_w


def generate_layer_mesh(
    layer_spec: dict,
    skeleton_spec: dict,
    png_path: str,
    img_size: tuple[int, int] = (1280, 1284),
    grid_step: int = 28
) -> dict | None:
    layer_id = layer_spec["id"]
    if not os.path.exists(png_path):
        print(f"Warning: {png_path} not found, skipping layer {layer_id}")
        return None

    # Mesh cells cover the alpha support, including antialiased edges. Sampling
    # only opaque contour pixels and dropping long triangles cuts holes in art.
    im = Image.open(png_path).convert("RGBA")
    tw, th = im.size
    w_img, h_img = img_size
    # P1（纹理 trim）：裁透明边后 (tw,th) 是裁后尺寸，trim_offset_px 是裁剪区
    # 在源图坐标的左上角。顶点/骨骼活在源图像素空间（scale=1.0 + 平移），
    # uv 用纹理自身归一化；瞳层 uv 需减去 trim_offset 再除以裁后尺寸。
    trim_off = layer_spec.get("trim_offset_px", [0, 0])
    trim_off_x, trim_off_y = float(trim_off[0]), float(trim_off[1])
    alpha = np.asarray(im)[:, :, 3]
    mask = alpha > 0
    if layer_spec.get("largest_component"):
        labels, count = ndimage.label(alpha > 15)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = ndimage.binary_dilation(labels == sizes.argmax(), iterations=2)
    min_px = layer_spec.get("min_component_px")
    if min_px:
        # Stray fragments (mis-assigned specks from the partition) must not get their own mesh
        # cells: a cell bound to this layer's bones drags the speck along when the limb moves.
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = sizes >= min_px
        keep[0] = False
        dropped = int(((~keep[labels]) & mask).sum())
        mask = keep[labels]
        if dropped:
            print(f"  [{layer_id}] dropped {int((sizes[1:] < min_px).sum())} fragments < {min_px} px "
                  f"({dropped} px) from the mesh support")
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    target = layer_spec.get("target_bbox_px")
    if target:
        # Register generated content through geometry, preserving the RGBA file.
        sy, sx = np.where(alpha > 15)
        bx0, bx1, by0, by1 = sx.min(), sx.max() + 1, sy.min(), sy.max() + 1
        scale_x = (target[2] - target[0]) / (bx1 - bx0)
        scale_y = (target[3] - target[1]) / (by1 - by0)
        off_x, off_y = target[0] - bx0 * scale_x, target[1] - by0 * scale_y
    else:
        # 裁后纹理像素 → 源图像素 = +trim_offset；scale 恒 1.0。
        scale_x, scale_y = 1.0, 1.0
        off_x, off_y = trim_off_x, trim_off_y
    off_x += layer_spec.get("offset_px", [0, 0])[0]
    off_y += layer_spec.get("offset_px", [0, 0])[1]
    step = layer_spec.get("grid_step", grid_step)
    gx = sorted(set([max(0, x0 - 2), min(tw, x1 + 2)] + list(range(x0, x1, max(2, int(step / scale_x))))))
    gy = sorted(set([max(0, y0 - 2), min(th, y1 + 2)] + list(range(y0, y1, max(2, int(step / scale_y))))))
    # Add exact eye-opening boundaries so full closure cannot leave white slivers.
    for zone in layer_spec.get("blink_zones", []):
        xs_zone = [zone[0] - zone[3], zone[0], zone[0] + zone[3]]
        if "blink_blend_px" in layer_spec:     # exact outer edge of the narrower side band
            xs_zone += [zone[0] - zone[3] - layer_spec["blink_blend_px"], zone[0] + zone[3] + layer_spec["blink_blend_px"]]
        for value in xs_zone:
            tx = (value - off_x) / scale_x
            if min(gx) < tx < max(gx):
                gx.append(tx)
        for value in [zone[1] - 35, zone[1], zone[2], zone[2] + 35]:
            ty = (value - off_y) / scale_y
            if min(gy) < ty < max(gy):
                gy.append(ty)
    gx, gy = sorted(set(gx)), sorted(set(gy))
    points, lookup, triangles = [], {}, []
    # per_component: grid cells shared by two separate pieces (gap < one cell) used to weld them
    # through common vertices (measured: 92 bridging triangles on the ADULT side rig). Each piece
    # (mask dilated 2 px, 8-connected) now gets its own vertices; a cell touching two pieces is
    # emitted once per piece.
    if layer_spec.get("per_component"):
        comp_lab, _ = ndimage.label(ndimage.binary_dilation(mask, iterations=2), structure=np.ones((3, 3), bool))
        comp_lab = np.where(mask, comp_lab, 0)
    else:
        comp_lab = None

    def vertex(x, y, c=0):
        key = (c, x, y)
        if key not in lookup:
            lookup[key] = len(points)
            points.append((x, y))
        return lookup[key]
    for ya, yb in zip(gy, gy[1:]):
        for xa, xb in zip(gx, gx[1:]):
            cell = (slice(int(ya), int(np.ceil(yb))), slice(int(xa), int(np.ceil(xb))))
            if not mask[cell].any():
                continue
            comps = [0] if comp_lab is None else [int(c) for c in np.unique(comp_lab[cell]) if c]
            for c in comps:
                ids = [vertex(xa, ya, c), vertex(xb, ya, c), vertex(xb, yb, c), vertex(xa, yb, c)]
                triangles.extend([ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]])
    vertices = [[round(x * scale_x + off_x, 4), round(y * scale_y + off_y, 4)] for x, y in points]
    uvs = [[x / tw, y / th] for x, y in points]
    pts_arr = np.array(vertices)
    influence_bones = layer_spec.get("influence_bones", [layer_spec["bind_bone"]])
    bones_dict = {b["bone_name"]: b for b in skeleton_spec["bones"]}

    if layer_spec.get("gaze_polygon"):
        # Sample the iris inside the eye opening, rather than clipping it to
        # the iris's original outline as soon as the look direction changes.
        outline = layer_spec["gaze_polygon"]
        cx, cy = np.asarray(outline, float).mean(axis=0)
        count = len(outline)
        vertices = [[float(cx), float(cy)]] + [[float(x), float(y)] for x, y in outline]
        uvs = [[(x - trim_off_x) / tw, (y - trim_off_y) / th] for x, y in vertices]
        triangles = [idx for j in range(count) for idx in (0, 1 + j, 1 + (j + 1) % count)]
        pts_arr = np.asarray(vertices)
    elif layer_spec.get("gaze_ellipse"):
        # The pupil is sampled through the fixed sclera silhouette. Moving UVs
        # looks around without drawing iris pixels over the surrounding skin.
        cx, cy, rx, ry = layer_spec["gaze_ellipse"]
        count = 48
        points = [[cx, cy]] + [[cx + rx * np.cos(t), cy + ry * np.sin(t)]
                              for t in np.linspace(0, 2 * np.pi, count, endpoint=False)]
        vertices = [[round(x, 4), round(y, 4)] for x, y in points]
        uvs = [[(x - trim_off_x) / tw, (y - trim_off_y) / th] for x, y in points]
        triangles = [idx for j in range(count) for idx in (0, 1 + j, 1 + (j + 1) % count)]
        pts_arr = np.asarray(vertices)

    # Calculate bone weights for each vertex
    weight_bones: list[list[str]] = []
    weight_values: list[list[float]] = []

    wmode = (layer_spec.get("weights") or {}).get("mode")
    if len(influence_bones) == 1:
        single_bone = influence_bones[0]
        weight_bones = [[single_bone] for _ in range(len(vertices))]
        weight_values = [[1.0] for _ in range(len(vertices))]
    elif wmode == "chain":
        weight_bones, weight_values = chain_weights(pts_arr, influence_bones, bones_dict, (w_img, h_img),
                                                    layer_spec["weights"])
    elif wmode == "skirt":
        weight_bones, weight_values = skirt_weights(pts_arr, layer_spec["weights"])
    else:
        # Multi-bone weighting
        # Collect bone joint positions and segments
        bone_anchors = []
        for bname in influence_bones:
            if bname in bones_dict:
                b_info = bones_dict[bname]
                j_pos = np.array([b_info["joint_pos"][0] * w_img, b_info["joint_pos"][1] * h_img])
                # Check for child in influence set to form a segment
                child = next((b for b in skeleton_spec["bones"] if b.get("parent") == bname and b["bone_name"] in influence_bones), None)
                if child:
                    c_pos = np.array([child["joint_pos"][0] * w_img, child["joint_pos"][1] * h_img])
                    bone_anchors.append((bname, j_pos, c_pos))
                else:
                    bone_anchors.append((bname, j_pos, None))
            else:
                bone_anchors.append((bname, np.array([w_img / 2, h_img / 2]), None))

        # Calculate distances for all vertices
        for pt in pts_arr:
            dists = []
            for bname, j_pos, c_pos in bone_anchors:
                if c_pos is not None:
                    d = dist_point_to_segment(pt[None, :], j_pos, c_pos)[0]
                else:
                    d = np.linalg.norm(pt - j_pos)
                dists.append(d)

            dists = np.array(dists)
            # Softmax / inverse power distance with smooth radius
            # Power = 2.0, epsilon = 25px
            weights = 1.0 / (np.maximum(dists, 1.0) + 25.0) ** 2.2
            # Normalize
            weights = weights / np.sum(weights)

            # Top 4 bones maximum for performance
            top_indices = np.argsort(weights)[::-1][:4]
            top_bones = [influence_bones[i] for i in top_indices if weights[i] > 0.02]
            top_weights = [float(weights[i]) for i in top_indices if weights[i] > 0.02]
            
            # Re-normalize top weights
            sum_w = sum(top_weights)
            if sum_w > 0:
                top_weights = [round(w / sum_w, 4) for w in top_weights]
            else:
                top_bones = [influence_bones[0]]
                top_weights = [1.0]

            weight_bones.append(top_bones)
            weight_values.append(top_weights)

    lock = layer_spec.get("root_lock")
    if lock:
        root = lock["bone"]
        t = np.clip((pts_arr[:, 1] - lock["full_before_y"]) /
                    (lock["free_after_y"] - lock["full_before_y"]), 0, 1)
        t = t * t * (3 - 2 * t)
        for index, blend in enumerate(t):
            weights = {b: w * blend for b, w in zip(weight_bones[index], weight_values[index])}
            weights[root] = weights.get(root, 0) + 1 - blend
            weight_bones[index] = list(weights)
            weight_values[index] = [round(v, 6) for v in weights.values()]

    # rigid_below: vertices below y belong to one bone (shoe -> foot, hand -> hand), with a
    # smoothstep blend band above it (optional; side rig G4 weight gate: shoe foot weight >= 0.8)
    for rb in layer_spec.get("rigid_below", []):
        t = np.clip((pts_arr[:, 1] - (rb["y"] - rb.get("blend", 20))) / max(rb.get("blend", 20), 1e-6), 0, 1)
        t = t * t * (3 - 2 * t)
        for index, blend in enumerate(t):
            if blend <= 0:
                continue
            weights = {b: w * (1 - blend) for b, w in zip(weight_bones[index], weight_values[index])}
            weights[rb["bone"]] = weights.get(rb["bone"], 0) + blend
            weights = {b: w for b, w in weights.items() if w > 1e-6}
            weight_bones[index] = list(weights)
            weight_values[index] = [round(v, 6) for v in weights.values()]

    # A forked fin is one rigid shape, even when its two lobes are far away
    # from the skeleton's centreline. Blend into the narrow peduncle below it.
    for ra in layer_spec.get("rigid_above", []):
        t = np.clip((ra["y"] + ra.get("blend", 20) - pts_arr[:, 1]) /
                    max(ra.get("blend", 20), 1e-6), 0, 1)
        t = t * t * (3 - 2 * t)
        if "x_max" in ra:
            t[pts_arr[:, 0] > ra["x_max"]] = 0
        for index, blend in enumerate(t):
            if blend <= 0:
                continue
            weights = {b: w * (1-blend) for b, w in zip(weight_bones[index], weight_values[index])}
            weights[ra["bone"]] = weights.get(ra["bone"], 0) + blend
            weights = {b: w for b, w in weights.items() if w > 1e-6}
            weight_bones[index] = list(weights)
            weight_values[index] = [round(v, 6) for v in weights.values()]

    blink_delta = np.zeros_like(pts_arr)
    for cx, top, bottom, radius in layer_spec.get("blink_zones", []):
        x, y = pts_arr[:, 0], pts_arr[:, 1]
        closure = top + (bottom - top) * 0.78
        vb = float(layer_spec.get("blink_vertical_blend_px", 16))
        mapped = np.interp(y, [top - vb, top, bottom, bottom + vb],
                           [top - vb, closure, closure, bottom + vb])
        mapped = np.where((y < top - vb) | (y > bottom + vb), y, mapped)
        # A slight downward arc and retained stroke thickness read as a closed eye.
        curve = 6 * np.clip(1 - ((x - cx) / radius) ** 2, 0, 1)
        ramp = np.interp(y, [top - vb, top, bottom, bottom + vb], [0, 1, 1, 0])
        mapped += curve * ramp
        if layer_id.startswith("eyelid"):
            mapped = closure + curve + (y - top) * 0.25
            blend = np.ones_like(x)
        else:
            bb = float(layer_spec.get("blink_blend_px", 25))   # skin band beside the eye that follows the lid
            blend = np.clip((radius + bb - abs(x - cx)) / bb, 0, 1)
        blink_delta[:, 1] += (mapped - y) * blend
    edge_fade = float(layer_spec.get("blink_edge_fade_px", 0))
    if edge_fade > 0:
        # Skin bordering a cut-out (e.g. bangs) keeps its silhouette. Moving
        # the contour with the eyelid stretched the cut into a vertical seam.
        distance = ndimage.distance_transform_edt(mask)
        tx = (pts_arr[:, 0] - off_x) / scale_x
        ty = (pts_arr[:, 1] - off_y) / scale_y
        edge_dist = ndimage.map_coordinates(distance, [ty, tx], order=1, mode="constant", cval=0, prefilter=False)
        t = np.clip(edge_dist / edge_fade, 0, 1)
        blink_delta *= (t*t*(3-2*t))[:, None]
    blink_delta = np.round(blink_delta, 4).tolist()

    return {
        "id": layer_id,
        "texture": layer_spec.get("texture", f"{layer_id}.png"),
        "trim_offset_px": [trim_off_x, trim_off_y],   # 便于回算；运行时不消费
        "blink_delta": blink_delta if layer_spec.get("blink_zones") else None,
        "blink_reveal_pivot_y": layer_spec.get("blink_reveal_pivot_y"),
        "gaze_uv": bool(layer_spec.get("gaze_ellipse") or layer_spec.get("gaze_polygon")),
        "texture_size_px": [tw, th],
        "z_order": layer_spec["z_order"],
        "vertices": vertices,
        "uvs": uvs,
        "triangles": triangles,
        "weight_bones": weight_bones,
        "weight_values": weight_values
    }


def generate_all_meshes(spec_path: str, layers_dir: str, out_mesh_path: str, grid_step: int = 28):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    img_w, img_h = spec["skeleton"]["source_reference"]["image_size_px"]
    layers_spec = sorted(spec["layers"], key=lambda x: x["z_order"])

    out_layers = []
    total_verts = 0
    total_tris = 0

    print(f"Generating mesh data for {len(layers_spec)} layers from {spec_path}...")

    for l in layers_spec:
        lid = l["id"]
        png_path = os.path.join(layers_dir, l.get("texture", f"{lid}.png"))
        mesh_layer = generate_layer_mesh(l, spec["skeleton"], png_path, (img_w, img_h), grid_step=grid_step)
        if mesh_layer:
            out_layers.append(mesh_layer)
            nv = len(mesh_layer["vertices"])
            nt = len(mesh_layer["triangles"]) // 3
            total_verts += nv
            total_tris += nt
            print(f"  [OK] {lid:14s}: {nv:4d} vertices, {nt:4d} triangles, bones: {l['influence_bones']}")

    mesh_data = {
        "spec": 1,
        "image_size_px": [img_w, img_h],
        "layers": out_layers
    }

    os.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)
    with open(out_mesh_path, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, indent=2)

    print(f"\n[DONE] Generated {len(out_layers)} layers, total {total_verts} vertices, {total_tris} triangles.")
    print(f"Mesh data saved to: {out_mesh_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 2D skinned mesh data from rig spec and layer PNGs")
    parser.add_argument("--stage", choices=["young", "adult"], default="young", help="Stage name (default: young)")
    parser.add_argument("--spec", default=None, help="Path to spec.json")
    parser.add_argument("--layers", default=None, help="Path to layers directory")
    parser.add_argument("--out", default=None, help="Output path for mesh_data.json")
    parser.add_argument("--grid-step", type=int, default=28, help="Grid step in pixels (default: 28)")
    args = parser.parse_args()

    spec = args.spec or f"assets/rig_{args.stage}/spec.json"
    layers = args.layers or f"assets/rig_{args.stage}/layers"
    out = args.out or f"assets/rig_{args.stage}/mesh/mesh_data.json"

    generate_all_meshes(spec, layers, out, grid_step=args.grid_step)
