"""ADULT 连续视角关键形态构建工具（模块 2，G2/G3 配准核）。

从**共用拓扑**（baseline mesh 为规范网格）出发，为每个关键视角生成顶点位置、
骨骼动态绑定支点与 Hermite 切线，产出 ``mesh/view_keyforms.json``：
其解析/校验/回退与写入同阶段交付（``pet/rig/skinned_mesh_item.ViewKeyforms``）。

两种模式：

* ``--mode procedural``（默认）：参数化视角压缩 warp——**数值验收夹具，非美术
  产物**。生成水平前缩 + 远侧收缩 + 轻微纵向起伏的确定性形变，供 101 点采样、
  零姿态恒等、C1 连续等数学门禁在真实资产（G1/G2 美术定稿）就绪前先行验证。
  输出 JSON 的 ``generated_by.provenance = "procedural-fixture"`` 明确标记。
* ``--mode landmarks``：TPS 薄板样条配准（正则化 λ）——输入
  ``annotations/{view}_landmarks.json``（同名语义标记 front↔view 对应表），
  把规范网格与骨骼关节配准到该视角的真实形变。标记缺失/退化时**报错退出**
  （计划约定"不假定自动配准一次成功"，宁可失败也不静默外推）。

硬校验（任一失败即退出码 1，不写输出）：

1. 共享拓扑：各视角顶点数/三角索引与规范网格一致（独立生成的网格不得通过）；
2. 网格非退化：yaw 区间 101 点采样，每三角形有向面积 |det| > 1e-4 且方向
   与正面一致（无翻面/自相交候选）；
3. Hermite 端点恒等：插值在关键视角处精确返回输入位置（|err| == 0）。

用法（仓库根目录）::

    D:/anaconda3/python.exe -X utf8 tools/build_adult_view_keyforms.py \
        --package assets/rig_adult_turn_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AREA_EPS = 1e-4          # §4.1 三角形有向面积门槛（px²）
SAMPLES = 101            # §4.1 视角区间采样点数


# ---------------- 过程式视角 warp（数值夹具） ----------------

def procedural_warp(points: np.ndarray, yaw_deg: float, canvas_w: float,
                    canvas_h: float, axis_x: float) -> np.ndarray:
    """水平前缩 + 远侧收缩 + 纵向微起伏的确定性视角形变（保持定向）。

    x' = axis + (x − axis)·(k − a·sigmoid((axis − x)/w))，k/a/w 随 yaw 单调；
    单调性下界 ≈ 0.5 > 0（见模块 docstring 校验 2 的 sweep 实测）。
    """
    t = min(max(yaw_deg / 45.0, -1.0), 1.0)
    k = 1.0 - 0.24 * abs(t)              # 整体前缩
    a = 0.15 * abs(t)                    # 远侧额外收缩幅度
    w = 160.0                            # 远/近侧过渡宽度（canvas px）
    dx = points[:, 0] - axis_x
    s = 1.0 / (1.0 + np.exp(np.clip(-dx / w, -60.0, 60.0)))
    x_new = axis_x + dx * (k - a * s)
    y_new = points[:, 1] + 3.0 * t * np.sin(math.pi * points[:, 1] / canvas_h)
    return np.column_stack((x_new, y_new))


# ---------------- TPS 薄板样条配准（landmarks 模式） ----------------

def tps_fit(src: np.ndarray, dst: np.ndarray, lam: float = 1e-3):
    """拟合 2D→2D TPS（每分量一个样条）。返回 eval 函数。"""
    n = len(src)
    if n < 3:
        raise SystemExit(f"标记数 {n} < 3，TPS 不可解（拒绝静默外推）")
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = d2 * np.log(d2 + 1e-20)
    P = np.column_stack((np.ones(n), src))
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K + lam * np.eye(n)
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.zeros((n + 3, 2))
    Y[:n] = dst
    try:
        W = np.linalg.solve(L, Y)
    except np.linalg.LinAlgError as e:
        raise SystemExit(f"TPS 线性系统退化：{e}")
    wx, wy = W[:n, 0], W[:n, 1]
    ax, ay = W[n:, 0], W[n:, 1]

    def eval_pts(pts: np.ndarray) -> np.ndarray:
        d2p = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        Kp = d2p * np.log(d2p + 1e-20)
        x = Kp @ wx + ax[0] + ax[1] * pts[:, 0] + ax[2] * pts[:, 1]
        y = Kp @ wy + ay[0] + ay[1] * pts[:, 0] + ay[2] * pts[:, 1]
        return np.column_stack((x, y))

    return eval_pts


def load_landmarks(path: str) -> dict:
    """annotations landmark 文件：{"pairs": [{"name", "front": [x,y], "view": [x,y]}]}。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pairs = raw.get("pairs") if isinstance(raw, dict) else raw
    if not isinstance(pairs, list) or not pairs:
        raise SystemExit(f"landmark 文件无 pairs：{path}")
    return {p["name"]: (tuple(p["front"]), tuple(p["view"])) for p in pairs}


# ---------------- Hermite 切线（§4.3） ----------------

def catmull_rom_tangents(keys: list, yaws: list) -> list:
    """关键视角序列的 Hermite 切线：内部中心差分，端点单侧。"""
    tans = []
    for i in range(len(keys)):
        if i > 0 and i < len(keys) - 1:
            m = (keys[i + 1] - keys[i - 1]) / (yaws[i + 1] - yaws[i - 1])
        elif i == 0:
            m = (keys[1] - keys[0]) / (yaws[1] - yaws[0])
        else:
            m = (keys[-1] - keys[-2]) / (yaws[-1] - yaws[-2])
        tans.append(m)
    return tans


def hermite_eval(p0, m0, p1, m1, mu: float):
    """§4.3 三次 Hermite 基（节点处精确返回端点，供共享校验 3）。

    ``m0/m1`` 为对 μ 的切线（= dP/dyaw × Δyaw，由 catmull_rom_tangents 换算）。
    """
    mu2, mu3 = mu * mu, mu * mu * mu
    return ((2 * mu3 - 3 * mu2 + 1) * p0 + (mu3 - 2 * mu2 + mu) * m0
            + (-2 * mu3 + 3 * mu2) * p1 + (mu3 - mu2) * m1)


def interp_at_yaw(keys: list, tans: list, yaws: list, yaw: float):
    """按 yaw 取 Hermite 插值（端点钳位到端视角，切线随 Δyaw 换算到 μ）。"""
    j = min(max(int(np.searchsorted(yaws, yaw) - 1), 0), len(yaws) - 2)
    d = yaws[j + 1] - yaws[j]
    mu = min(max((yaw - yaws[j]) / d, 0.0), 1.0)
    return hermite_eval(keys[j], tans[j] * d, keys[j + 1], tans[j + 1] * d, mu)


def orient_dets(v: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


# ---------------- 主流程 ----------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--package", default=os.path.join(ROOT, "assets", "rig_adult_turn_v1"))
    ap.add_argument("--mode", choices=("procedural", "landmarks"), default="procedural")
    ap.add_argument("--views", default="front,right20,right45",
                    help="逗号分隔视角名（front 为规范网格）")
    ap.add_argument("--yaws", default="", help="逗号分隔 yaw 度数（缺省按既有 JSON）")
    ap.add_argument("--annotations-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkg = args.package
    spec_path = os.path.join(pkg, "spec.json")
    baseline_mesh = os.path.join(pkg, "baseline", "mesh", "mesh_data.json")
    if not os.path.isfile(spec_path):
        spec_path = os.path.join(pkg, "baseline", "spec.json")
    mesh_path = os.path.join(pkg, "mesh", "mesh_data.json")
    if not os.path.isfile(mesh_path):
        mesh_path = baseline_mesh
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(mesh_path, "r", encoding="utf-8") as f:
        mesh = json.load(f)
    W, H = (float(v) for v in mesh["image_size_px"])
    ref = (spec.get("skeleton") or {}).get("source_reference") or {}
    W = float((ref.get("image_size_px") or [W, H])[0])
    H = float((ref.get("image_size_px") or [W, H])[1])

    view_names = [v.strip() for v in args.views.split(",") if v.strip()]
    if "front" not in view_names:
        raise SystemExit("views 必须包含 front（规范网格）")

    # 关节与网格（规范 = front）
    bones = spec["skeleton"]["bones"]
    joints = {b["bone_name"]: np.array([b["joint_pos"][0] * W, b["joint_pos"][1] * H])
              for b in bones}
    spine_x = 0.5 * (joints["root_hip"][0] + joints["neck"][0])   # 转身轴近似

    layers = mesh["layers"]
    view_yaws: dict[str, float] = {"front": 0.0, "right20": 20.0, "right45": 45.0}
    if args.yaws:
        for name, y in zip(view_names, (float(v) for v in args.yaws.split(","))):
            view_yaws[name] = y

    # ---- 每视角形变（网格 + 关节共用同一映射，保证绑定一致） ----
    pos_by_view: dict[str, dict[str, np.ndarray]] = {}
    joints_by_view: dict[str, dict[str, np.ndarray]] = {}
    for vn in view_names:
        yaw = view_yaws.get(vn)
        if yaw is None:
            raise SystemExit(f"视角 {vn} 缺 yaw 定义")
        if vn == "front":
            pos_by_view[vn] = {l["id"]: np.asarray(l["vertices"], np.float64)
                               for l in layers}
            joints_by_view[vn] = {k: v.copy() for k, v in joints.items()}
            continue
        if args.mode == "landmarks":
            ann = args.annotations_dir or os.path.join(pkg, "annotations")
            lm_path = os.path.join(ann, f"{vn}_landmarks.json")
            if not os.path.isfile(lm_path):
                raise SystemExit(f"landmarks 模式缺标记文件：{lm_path}")
            pairs = load_landmarks(lm_path)
            src = np.array([p[0] for p in pairs.values()], np.float64)
            dst = np.array([p[1] for p in pairs.values()], np.float64)
            warp = tps_fit(src, dst)
        else:
            def warp(pts, _y=yaw):
                return procedural_warp(pts, _y, W, H, spine_x)
        pos_by_view[vn] = {l["id"]: warp(np.asarray(l["vertices"], np.float64))
                           for l in layers}
        joints_by_view[vn] = {k: warp(v[None, :])[0] for k, v in joints.items()}

    # ---- 校验 1：共享拓扑（front 位置也须与规范网格逐位一致） ----
    for vn in view_names:
        for l in layers:
            arr = pos_by_view[vn][l["id"]]
            if arr.shape != (len(l["vertices"]), 2) or not np.isfinite(arr).all():
                raise SystemExit(f"校验1失败：{vn}/{l['id']} 顶点形状/有限性")
        if vn == "front":
            for l in layers:
                if not np.array_equal(pos_by_view["front"][l["id"]],
                                      np.asarray(l["vertices"], np.float64)):
                    raise SystemExit(f"校验1失败：front/{l['id']} 与规范网格不一致")

    # ---- Hermite 切线（按 yaw 升序的视角序列，per-layer / per-joint） ----
    ordered = sorted(view_names, key=lambda n: view_yaws[n])
    o_yaws = [view_yaws[n] for n in ordered]
    tangents: dict[str, dict[str, np.ndarray]] = {}
    jtangents: dict[str, dict[str, np.ndarray]] = {}
    for li, l in enumerate(layers):
        keys = [pos_by_view[vn][l["id"]] for vn in ordered]
        tangents[l["id"]] = catmull_rom_tangents(keys, o_yaws)
    jkeys_all = {bname: [joints_by_view[vn][bname] for vn in ordered] for bname in joints}
    jtangents = {bn: catmull_rom_tangents(ks, o_yaws) for bn, ks in jkeys_all.items()}

    # ---- 校验 2：非退化 101 点 sweep（含切线插值中段） ----
    y_lo, y_hi = min(o_yaws), max(o_yaws)
    worst = None
    for l in layers:
        tris = np.asarray(l["triangles"], np.int64).reshape(-1, 3)
        keys = [pos_by_view[vn][l["id"]] for vn in ordered]
        tans = tangents[l["id"]]
        for i in range(SAMPLES):
            yaw = y_lo + (y_hi - y_lo) * i / (SAMPLES - 1)
            p = interp_at_yaw(keys, tans, o_yaws, yaw)
            dets = orient_dets(p, tris)
            if worst is None or dets.min() < worst[0]:
                worst = (float(dets.min()), l["id"], yaw)
            if (dets <= AREA_EPS).any():
                raise SystemExit(
                    f"校验2失败：{l['id']} yaw={yaw:.2f} 存在 |det| ≤ {AREA_EPS} 的三角形")
    # 方向一致性（与 front 同号）
    for l in layers:
        tris = np.asarray(l["triangles"], np.int64).reshape(-1, 3)
        front_sign = np.sign(orient_dets(np.asarray(l["vertices"], np.float64), tris))
        for vn in ordered[1:]:
            dets = orient_dets(pos_by_view[vn][l["id"]], tris)
            if not (np.sign(dets) == front_sign).all():
                raise SystemExit(f"校验2失败：{vn}/{l['id']} 存在与正面反向的三角形")

    # ---- 校验 3：Hermite 节点恒等（μ=0/1 处基函数精确归一到端点） ----
    for l in layers:
        keys = [pos_by_view[vn][l["id"]] for vn in ordered]
        tans = tangents[l["id"]]
        for i in range(len(ordered) - 1):
            d = o_yaws[i + 1] - o_yaws[i]
            p_end = hermite_eval(keys[i], tans[i] * d, keys[i + 1],
                                 tans[i + 1] * d, 1.0)
            if not np.array_equal(p_end, keys[i + 1]):
                raise SystemExit(f"校验3失败：{ordered[i + 1]}/{l['id']} 节点不恒等")

    # ---- 输出 ----
    out = {
        "version": "1.0",
        "generated_by": {
            "tool": "build_adult_view_keyforms.py",
            "mode": args.mode,
            "provenance": ("procedural-fixture：数值验收夹具，非美术定稿；"
                           "G2/G3 定稿后以 --mode landmarks 重建"),
            "source_spec": os.path.relpath(spec_path, ROOT),
            "source_mesh": os.path.relpath(mesh_path, ROOT),
        },
        "views": {vn: {"yaw_deg": view_yaws[vn]} for vn in view_names},
        "bones_dynamic_bind": {
            vn: {bn: {"pivot": [float(j[0]), float(j[1])],
                      "world_matrix": [1.0, 0.0, 0.0, 1.0,
                                       float(j[0]), float(j[1])]}
                 for bn, j in joints_by_view[vn].items()}
            for vn in view_names
        },
        "layers_keyforms": {
            l["id"]: {
                "vertex_count": len(l["vertices"]),
                "positions": {vn: pos_by_view[vn][l["id"]].tolist() for vn in view_names},
                "tangents": {vn: tangents[l["id"]][i].tolist()
                             for i, vn in enumerate(ordered)},
            } for l in layers
        },
        "occlusion_patches": [],
    }
    out_path = args.out or os.path.join(pkg, "mesh", "view_keyforms.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[OK] view_keyforms.json → {out_path}（{size_mb:.2f} MB, "
          f"{len(view_names)} 视角 / {len(layers)} 层）")
    print(f"[OK] 校验通过：共享拓扑 / {SAMPLES} 点非退化 "
          f"(min|det|={worst[0]:.1f}px² @ {worst[1]}) / 节点恒等")
    return 0


if __name__ == "__main__":
    sys.exit(main())
