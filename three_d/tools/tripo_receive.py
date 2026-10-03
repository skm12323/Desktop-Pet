#!/usr/bin/env python3
"""Tripo 草模回收 —— inbox/ 里的 GLB 与逆向骨架对齐，出缩放/偏移报告。

浏览器或 API 导出的 GLB 丢进 three_d/assets_src/tripo/inbox/ 后运行本脚本：
  1. 解析 GLB（stdlib：header + JSON chunk，POSITION accessor 的 min/max
     是 glTF 规范强制字段，不解码顶点即得包围盒）；
  2. 与 skeleton3d_adult.json 对齐：算 GLB 高度 → 目标米制高度
     （骨架 px_per_meter_proposal）→ Blender 导入缩放建议 + 居中/落地偏移；
  3. 写 receive_report.md。

用法：python3 three_d/tools/tripo_receive.py [--skeleton skeleton3d_adult.json]
"""

from __future__ import annotations

import argparse
import json
import os
import struct

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRIPO = os.path.join(REPO, "three_d", "assets_src", "tripo")
INBOX = os.path.join(TRIPO, "inbox")
GLB_MAGIC = 0x46546C67  # 'glTF'
CHUNK_JSON = 0x4E4F534A  # 'JSON'


def glb_bbox(path: str) -> tuple[list[float], list[float]]:
    """读 GLB/glTF 的 POSITION accessor min/max（规范必填），返回 (mn, mx)。"""
    with open(path, "rb") as f:
        head = f.read(12)
        magic, version, length = struct.unpack("<III", head)
        if magic != GLB_MAGIC:
            raise ValueError(f"{path}: 不是 GLB（.gltf JSON 请先转 GLB）")
        doc = None
        offset = 12
        while offset < length:
            clen, ctype = struct.unpack("<II", f.read(8))
            data = f.read(clen)
            if ctype == CHUNK_JSON:
                doc = json.loads(data.decode(errors="replace"))
            offset += 8 + clen + (clen & 3)  # chunk 4 字节对齐 padding
    assert doc, "GLB 缺 JSON chunk"
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            a_mn, a_mx = acc.get("min"), acc.get("max")
            if not a_mn or not a_mx:
                raise ValueError(f"{path}: POSITION accessor 缺 min/max（非法 glTF）")
            mn = [min(a, b) for a, b in zip(mn, a_mn)]
            mx = [max(a, b) for a, b in zip(mx, a_mx)]
    return mn, mx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", default="skeleton3d_adult.json")
    args = ap.parse_args()
    skel_path = os.path.join(TRIPO, "..", "reverse", args.skeleton)
    with open(skel_path, encoding="utf-8") as f:
        skel = json.load(f)
    px_per_m = skel["meta"]["px_per_meter_proposal"]
    ys = [b["pos_px"][1] for b in skel["bones"]]
    target_h_m = max(ys) / px_per_m  # 含呆毛顶点，Blender 内按头型复核

    if not os.path.isdir(INBOX) or not any(
            f.endswith(".glb") for f in os.listdir(INBOX)):
        print(f"[receive] inbox 空（{INBOX}）——先从 Tripo 导出 GLB 丢进来")
        return

    lines = [f"# Tripo 草模回收报告（{target_h_m:.2f}m 目标身高，含呆毛顶点）", ""]
    for f in sorted(os.listdir(INBOX)):
        if not f.endswith(".glb"):
            continue
        path = os.path.join(INBOX, f)
        try:
            mn, mx = glb_bbox(path)
        except Exception as e:
            lines += [f"## {f}", f"- 解析失败：{e}", ""]
            continue
        d = [mx[i] - mn[i] for i in range(3)]
        # 竖直轴自适应：站立角色高度≫宽/深，取 dy/dz 中大者为身高轴
        # （TripoSR 导出为 Z-up，glTF 惯例 Y-up，两种都兼容）
        up = 1 if d[1] >= d[2] else 2
        up_name = "Y" if up == 1 else "Z"
        height = d[up]
        horiz = [i for i in range(3) if i != up]
        scale = target_h_m / height if height > 0 else 0.0
        c = [(mn[i] + mx[i]) / 2 for i in horiz]
        lines += [
            f"## {f}",
            f"- 包围盒：{d[0]:.3f} × {d[1]:.3f} × {d[2]:.3f}（glTF 单位），"
            f"竖直轴判定为 **{up_name}**（TripoSR 出 Z-up，Blender 导入时注意轴向转换）",
            f"- Blender 导入缩放：**{scale:.4f}**（使身高={target_h_m:.2f}m）",
            f"- 居中：两水平轴（{'XYZ'[horiz[0]]}/{'XYZ'[horiz[1]]}）各平移 "
            f"{c[0]:+.3f}、{c[1]:+.3f} 取负（对齐骨架中轴）",
            f"- 落地：{up_name} 轴平移 {mn[up]:+.3f} 取负（脚底=0，与骨架地面锚点同语义）",
            f"- 对照：骨架踝高 {min(ys) / px_per_m:.2f}m…头顶 {target_h_m:.2f}m；"
            "比例观感异常时先怀疑草模大头（图生 3D 通病，调研 C-1）",
            "",
        ]
    out = os.path.join(TRIPO, "receive_report.md")
    with open(out, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[receive] -> {out}")


if __name__ == "__main__":
    main()
