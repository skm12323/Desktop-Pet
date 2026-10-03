#!/usr/bin/env python3
"""逆向还原工具 v0 —— 由 2D rig 正/侧双 spec 反推 3D 骨架基准坐标。

背景（three_d/wiki/调研-建模与资产管线.md D13「对位建模」）：
  现有 2D rig 的正面 spec（assets/rig_adult/spec.json，47 骨）与侧身 spec
  （assets/rig_adult_walk_v1/spec.json，37 骨）共用同一画布（960×1696）与
  同一地面锚点（ground_anchor_y_px=1608），且骨骼命名互相咬合——恰好构成
  「正侧对位建模」所需的双正交视图。本工具把两份 spec 的 joint_pos 归一化
  坐标合成为每个关节的 3D 基准坐标（px 单位，y 向上自地面，+z 为面朝方向），
  供 Blender 建模期作关节放置参照，及后续 VRM 骨骼命名的草案依据。

产出（three_d/assets_src/reverse/）：
  skeleton3d_adult.json  3D 骨架（含 VRM 命名草案/来源方法标注/Y 对齐误差）
  report.md              对齐质量报告 + Blender 使用说明
  preview.svg            三联预览：正面板 / 侧面板 / 3D 等轴投影

用法：python3 three_d/tools/reverse_skeleton.py [--stage adult]

仅依赖 stdlib + PySide6(QImage)（读图板/缩放，无需 QApplication）。
数据不修改源 spec；所有"估计"在输出里逐骨标注 method，不与实测混淆。
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import math
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "three_d", "assets_src", "reverse")

# ---- 输入清单（adult；young 后续按同构 spec 追加） --------------------------
STAGES = {
    "adult": {
        "front": {
            "spec": "assets/rig_adult/spec.json",
            "board": "assets/rig_adult_walk_v1/references/front_rest.png",
        },
        "side": {
            "spec": "assets/rig_adult_walk_v1/spec.json",
            "board": "assets/rig_adult_walk_v1/references/side_key.png",
        },
    }
}

# ---- VRM 1.0 humanoid 命名草案（核心 15 骨直映，其余归组） -------------------
VRM_HUMANOID = {
    "root_hip": "hips", "spine": "spine", "chest": "chest", "neck": "neck",
    "head": "head",
    "upper_arm_l": "leftUpperArm", "forearm_l": "leftLowerArm", "hand_l": "leftHand",
    "upper_arm_r": "rightUpperArm", "forearm_r": "rightLowerArm", "hand_r": "rightHand",
    "upper_leg_l": "leftUpperLeg", "lower_leg_l": "leftLowerLeg", "foot_l": "leftFoot",
    "upper_leg_r": "rightUpperLeg", "lower_leg_r": "rightLowerLeg", "foot_r": "rightFoot",
}
SPRING_GROUPS = {  # 自定义骨 → 运行时拟走 verlet 弹簧骨（调研-引擎页 B-2）
    "tail": "spring:tail", "skirt": "spring:skirt", "apron": "spring:apron",
    "hair_back": "spring:hair", "hair_side": "spring:hair", "bangs": "spring:hair",
    "ahoge": "spring:hair", "ear_fin": "custom:ear_fin",
}
FACE_DRIVEN = {"eyelid", "pupil"}  # 表情/视线通道驱动，不进骨骼（D10）


def _group(name: str) -> str:
    if name in VRM_HUMANOID:
        return "humanoid"
    if any(name.startswith(p) for p in FACE_DRIVEN):
        return "face"
    for prefix, grp in SPRING_GROUPS.items():
        if name.startswith(prefix):
            return grp
    return "custom"


def _vrm(name: str) -> str | None:
    return VRM_HUMANOID.get(name)


def load_spec(rel: str) -> dict:
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return json.load(f)


def bone_maps(spec: dict) -> tuple[dict, dict]:
    """返回 {name: (jx, jy 归一化)}, {name: parent}。"""
    joints, parents = {}, {}
    for b in spec["skeleton"]["bones"]:
        n = b["bone_name"]
        joints[n] = tuple(float(v) for v in b["joint_pos"])
        parents[n] = b["parent"]
    return joints, parents


def mirror_name(name: str) -> str | None:
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="adult", choices=STAGES)
    args = ap.parse_args()
    cfg = STAGES[args.stage]

    f_spec, s_spec = load_spec(cfg["front"]["spec"]), load_spec(cfg["side"]["spec"])
    f_j, f_p = bone_maps(f_spec)
    s_j, s_p = bone_maps(s_spec)
    W, H = (float(v) for v in f_spec["skeleton"]["source_reference"]["image_size_px"])
    sW, sH = (float(v) for v in s_spec["skeleton"]["source_reference"]["image_size_px"])
    assert (W, H) == (sW, sH), "正/侧画布规格不一致，需先做比例归一"
    ground = float(f_spec["ground_anchor_y_px"])
    assert ground == float(s_spec["ground_anchor_y_px"]), "地面锚点不一致"

    # 深度零轴：以 root_hip 的侧视 x 为体轴（+z=面朝方向；侧图中尾在 x 小侧，故尾 z<0 自洽）
    root_sx = s_j["root_hip"][0]
    root_fx = f_j["root_hip"][0]

    all_names = list(dict.fromkeys(list(f_j) + list(s_j)))
    bones, y_deltas = [], []
    for n in all_names:
        parent = f_p.get(n) or s_p.get(n)
        method = x = y = z = None
        delta = None
        if n in f_j and n in s_j:
            fx, fy = f_j[n]
            sx, sy = s_j[n]
            x = (fx - root_fx) * W
            y_f = (ground - fy * H)          # 正面给的高度
            y_s = (ground - sy * H)          # 侧身给的高度
            delta = y_f - y_s                # 对齐误差（正=正面显得更高）
            y = (y_f + y_s) / 2.0
            z = (sx - root_sx) * sW
            method = "measured"
        elif n in f_j:
            fx, fy = f_j[n]
            x = (fx - root_fx) * W
            y = ground - fy * H
            m = mirror_name(n)
            if m and m in s_j:               # 侧视只画了单侧（如 hair_side_r）
                z = (s_j[m][0] - root_sx) * sW
                method = "mirrored"
            elif n.startswith("pupil"):      # 瞳孔：用同侧眼睑的实测深度
                lid = "eyelid_l" if n.endswith("_l") else "eyelid_r"
                if lid in s_j:
                    z = (s_j[lid][0] - root_sx) * sW
                    method = "inherited(eyelid)"
            if method is None:               # 兜底：继承最近已测祖先的深度
                z = 0.0
                anc = parent
                while anc:
                    if anc in s_j:
                        z = (s_j[anc][0] - root_sx) * sW
                        method = f"inherited({anc})"
                        break
                    anc = f_p.get(anc) or s_p.get(anc)
        else:  # 仅侧视有（当前 adult 数据无此情形，防御保留）
            sx, sy = s_j[n]
            x, y = 0.0, ground - sy * H
            z = (sx - root_sx) * sW
            method = "side_only(x=0)"
        if delta is not None:
            y_deltas.append((abs(delta), delta, n))
        bones.append({
            "name": n, "parent": parent, "pos_px": [round(x, 1), round(y, 1), round(z, 1)],
            "vrm": _vrm(n), "group": _group(n), "method": method,
            **({"y_delta_px": round(delta, 1)} if delta is not None else {}),
        })

    # 单位换算提案：头顶（ahoge_02 顶点）到地面 ≈ 目标身高的映射
    top = max((b["pos_px"][1] for b in bones), default=0.0)
    px_per_m = top / 1.5  # 成年目标身高 1.5m（Blender 里再按真实立绘校）

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "meta": {
            "tool": "three_d/tools/reverse_skeleton.py",
            "date": datetime.date.today().isoformat(),
            "stage": args.stage,
            "sources": {"front": cfg["front"]["spec"], "side": cfg["side"]["spec"],
                        "boards": [cfg["front"]["board"], cfg["side"]["board"]]},
            "canvas_px": [W, H], "ground_anchor_y_px": ground,
            "units": "px；x 右正、y 自地面向上、z 面朝方向正（尾侧为负）",
            "px_per_meter_proposal": round(px_per_m, 1),
            "notes": "坐标为关节基准草案（D13 对位建模参照）；VRM 绑定需重摆 T-pose；"
                     "method 含 mirrored/inherited 者为估计值，勿当实测用。",
        },
        "bones": bones,
    }
    json_path = os.path.join(OUT_DIR, f"skeleton3d_{args.stage}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 对齐质量报告 ----
    y_deltas.sort(reverse=True)
    m_hat = [d for d in y_deltas if d[2].startswith("hair")]
    core = [d for d in y_deltas if not d[2].startswith("hair")]
    lines = [
        f"# 逆向还原报告 — {args.stage}（{datetime.date.today()}）", "",
        f"- 正面骨 {len(f_j)} / 侧身骨 {len(s_j)} / 合成 {len(bones)}",
        f"- Y 对齐误差（|正面高度−侧身高度|，px）：最大 {y_deltas[0][0]:.0f}（{y_deltas[0][2]}），"
        f"均值 {sum(d[0] for d in y_deltas)/len(y_deltas):.0f}",
        f"- 核心骨（除 hair_*）最大误差：{core[0][0]:.0f}px（{core[0][2]}）",
        "- 误差含义：两套立绘的关节标注不完全一致，属预期；Blender 对位时以正面为准、"
        "侧视微调。hair_* 链误差偏大不阻塞建模（弹簧骨区，形态自由度高）。",
        "",
        "## 误差 Top10", "", "| 骨骼 | Δpx |", "|---|---|",
    ]
    lines += [f"| {n} | {d:+.0f} |" for _, d, n in y_deltas[:10]]
    lines += ["", "## 方法统计", ""]
    stat: dict[str, int] = {}
    for b in bones:
        stat[b["method"]] = stat.get(b["method"], 0) + 1
    lines += [f"- {k}: {v}" for k, v in sorted(stat.items(), key=lambda kv: -kv[1])]
    lines += [
        "", "## Blender 使用说明", "",
        f"1. 正面板（front_rest.png）作为 XY 图板、侧面板（side_key.png）作为 ZY 图板导入；",
        f"2. 关节坐标（skeleton3d_{args.stage}.json，px）按 `px_per_meter_proposal` 缩放后"
        "可作 empty/骨架放置参照（注意本工具 y 向上，Blender Z 向上，导入时 y↔z 互换）；",
        "3. 当前姿态为自然站立（A-pose 倾向），VRM 绑定需重摆标准 T-pose；",
        "4. eyelid/pupil 不建骨骼（表情 morph / lookAt 通道，D10）；tail/skirt/apron/hair/ahoge"
        " 建模为弹簧骨候选组（运行时 verlet，参数后置 sidecar JSON）。",
    ]
    report_path = os.path.join(OUT_DIR, f"report_{args.stage}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 三联预览 SVG（正面/侧面/3D 等轴） ----
    svg = build_preview(cfg, f_j, s_j, bones, W, H, ground, root_fx, root_sx)
    svg_path = os.path.join(OUT_DIR, f"preview_{args.stage}.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"[reverse_skeleton] {len(bones)} bones -> {json_path}")
    print(f"[reverse_skeleton] report -> {report_path}")
    print(f"[reverse_skeleton] preview -> {svg_path}")
    if y_deltas:
        print(f"[reverse_skeleton] Y 对齐：max {y_deltas[0][0]:.0f}px ({y_deltas[0][2]}), "
              f"mean {sum(d[0] for d in y_deltas)/len(y_deltas):.0f}px")


def _b64_img(rel: str, scale: float = 0.5) -> str:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage
    img = QImage(os.path.join(REPO, rel))
    img = img.scaled(int(img.width() * scale), int(img.height() * scale))
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.ReadWrite)
    img.save(buf, "PNG")
    return base64.b64encode(bytes(data)).decode()


def build_preview(cfg, f_j, s_j, bones, W, H, ground, root_fx, root_sx) -> str:
    S = 0.5                                    # 图板缩放
    bw, bh = int(W * S), int(H * S)
    pad = 24
    iso_x = pad * 3 + bw * 2                   # 第三面板起点
    total_w = iso_x + bw + pad * 2
    total_h = bh + pad * 2 + 30
    pos = {b["name"]: b["pos_px"] for b in bones}
    par = {b["name"]: b["parent"] for b in bones}

    def fx(n):  # 正面板屏幕坐标
        return (f_j[n][0] * W * S, f_j[n][1] * H * S)

    def sx(n):  # 侧面板屏幕坐标
        return (s_j[n][0] * W * S, s_j[n][1] * H * S)

    a = math.radians(32)                       # 3D 等轴投影
    def iso(p):
        x, y, z = p
        u = (x * math.cos(a) + z * math.sin(a)) * S * 0.9
        v = (y + (x * math.sin(a) - z * math.cos(a)) * 0.35) * S * 0.9
        return (iso_x + bw / 2 + u, pad + bh - 40 - v)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
           f'font-family="monospace" font-size="11">']
    try:
        fb64, sb64 = _b64_img(cfg["front"]["board"]), _b64_img(cfg["side"]["board"])
        out.append(f'<image x="{pad}" y="{pad}" width="{bw}" height="{bh}" '
                   f'href="data:image/png;base64,{fb64}"/>')
        out.append(f'<image x="{pad*2+bw}" y="{pad}" width="{bw}" height="{bh}" '
                   f'href="data:image/png;base64,{sb64}"/>')
    except Exception as e:                     # 图板缺了也出骨架（调试友好）
        out.append(f'<text x="{pad}" y="{pad}">board load failed: {e}</text>')
    out.append(f'<text x="{pad}" y="{total_h-8}">FRONT (joints)</text>')
    out.append(f'<text x="{pad*2+bw}" y="{total_h-8}">SIDE (joints)</text>')
    out.append(f'<text x="{iso_x}" y="{total_h-8}">3D ISO DRAFT</text>')

    human = [b for b in bones if b["group"] == "humanoid"]
    hn = {b["name"] for b in human}
    # 骨连线（父→子），humanoid 高亮、其余淡色
    for b in bones:
        p = b["parent"]
        if not p or p not in pos:
            continue
        for panel, pt in (("f", fx), ("s", sx)):
            try:
                x1, y1 = pt(p); x2, y2 = pt(b["name"])
            except KeyError:
                continue
            col = "#d33" if b["name"] in hn else "#888"
            wd = 1.6 if b["name"] in hn else 0.8
            ox = 0 if panel == "f" else pad + bw
            out.append(f'<line x1="{x1+ox:.1f}" y1="{y1+pad:.1f}" x2="{x2+ox:.1f}" '
                       f'y2="{y2+pad:.1f}" stroke="{col}" stroke-width="{wd}" opacity="0.75"/>')
        u1, v1 = iso(pos[p]); u2, v2 = iso(pos[b["name"]])
        col = "#d33" if b["name"] in hn else "#8af"
        wd = 1.6 if b["name"] in hn else 0.8
        out.append(f'<line x1="{u1:.1f}" y1="{v1:.1f}" x2="{u2:.1f}" y2="{v2:.1f}" '
                   f'stroke="{col}" stroke-width="{wd}"/>')
    # 关节点
    for b in bones:
        if b["name"] in f_j:
            x, y = fx(b["name"])
            out.append(f'<circle cx="{x+pad:.1f}" cy="{y+pad:.1f}" r="3" fill="#ffd80c" '
                       f'stroke="#333" stroke-width="0.6"/>')
        if b["name"] in s_j:
            x, y = sx(b["name"])
            out.append(f'<circle cx="{x+pad+bw+pad:.1f}" cy="{y+pad:.1f}" r="3" fill="#ffd80c" '
                       f'stroke="#333" stroke-width="0.6"/>')
        u, v = iso(pos[b["name"]])
        col = "#d33" if b["name"] in hn else "#8af"
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="2.4" fill="{col}"/>')
    for b in human:                            # 3D 面板只标 humanoid 骨名，防糊
        u, v = iso(pos[b["name"]])
        out.append(f'<text x="{u+4:.1f}" y="{v+3:.1f}" fill="#333" font-size="10">'
                   f'{b["vrm"]}</text>')
    out.append("</svg>")
    return "".join(out)


if __name__ == "__main__":
    main()
