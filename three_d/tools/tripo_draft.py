#!/usr/bin/env python3
"""Tripo 图生 3D 草模 —— 输入准备 + API 调用（产线 C 加速器，调研-建模页）。

用法（两种通道）：
  A. 浏览器通道（推荐先试效果）：
     1. python3 three_d/tools/tripo_draft.py --prepare   # 生成输入图
     2. 浏览器开 tripo3d.ai → 上传 input/adult_front_white.png（或 _alpha）
        → 生成 → （可选）自动绑骨 → 导出 GLB
     3. 把 GLB 丢进 three_d/assets_src/tripo/inbox/
     4. python3 three_d/tools/tripo_receive.py           # 对齐骨架出报告
  B. API 通道（本仓库沙箱网络到 tripo3d.ai 不通，需在**你自己的终端**跑）：
     export TRIPO_API_KEY=...   # platform.tripo3d.ai 注册获取
     python3 three_d/tools/tripo_draft.py --run

⚠️ API 形态未经在线核实（2026-10-03 本机与 tripo3d.ai 的 TLS 均断，
文档站打不开）。BASE/端点/字段按公开 v2 API 的通行形态写成常量，
首次运行若 404/字段不符，按报错里的响应体对照官方文档改头部常量即可。
隐私提醒（调研-建模页 C-2）：上传即把角色立绘交给第三方，注册时顺手
过一遍当期条款。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.path.join(REPO, "three_d", "assets_src", "tripo")
INPUT_DIR = os.path.join(ROOT, "input")
INBOX_DIR = os.path.join(ROOT, "inbox")

SOURCES = {
    "front": "assets/rig_adult_walk_v1/references/front_rest.png",
    "side": "assets/rig_adult_walk_v1/references/side_key.png",
}

# ---- API 常量（UNVERIFIED：见文件头说明） ------------------------------------
API_BASE = "https://api.tripo3d.ai"
EP_TASK = "/v2/openapi/task"           # 创建任务（POST）/ 查询（GET {id}）
EP_UPLOAD = "/v2/openapi/upload/sts"   # 取上传凭证（各版本命名不一）
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 600


def prepare() -> None:
    """生成 Tripo 输入图：透明版 + 白底方形版（正/侧各一套）。"""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter, QColor

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(INBOX_DIR, exist_ok=True)
    for key, rel in SOURCES.items():
        img = QImage(os.path.join(REPO, rel))
        assert not img.isNull(), f"读图失败: {rel}"
        alpha_path = os.path.join(INPUT_DIR, f"adult_{key}_alpha.png")
        img.save(alpha_path)
        # 白底正方形（部分管线对纯白背景的分割更稳）
        side = max(img.width(), img.height())
        canvas = QImage(side, side, QImage.Format_ARGB32)
        canvas.fill(QColor(255, 255, 255))
        p = QPainter(canvas)
        p.drawImage((side - img.width()) // 2, (side - img.height()) // 2, img)
        p.end()
        white_path = os.path.join(INPUT_DIR, f"adult_{key}_white.png")
        canvas.save(white_path)
        print(f"[prepare] {key}: {alpha_path}\n[prepare]        {white_path}")
    print(f"[prepare] inbox 就绪：{INBOX_DIR}（浏览器导出的 GLB 丢这里）")


def _req(url: str, key: str, data: dict | None = None) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method="POST" if body else "GET",
                               headers={"Content-Type": "application/json",
                                        "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read().decode())


def run_api(key: str) -> None:
    print("⚠️ API 形态未经在线核实，报错时把响应体贴到文档对照改头部常量。")
    # 1) 输入图需要公网 URL：v2 流程为先取上传凭证再 PUT 文件。
    #    （本地沙箱到 tripo3d.ai 不通，此脚本预期在用户终端运行）
    img_path = os.path.join(INPUT_DIR, "adult_front_white.png")
    if not os.path.isfile(img_path):
        prepare()
    up = _req(API_BASE + EP_UPLOAD, key, {"format": "png"})
    print("[upload-sts]", json.dumps(up, ensure_ascii=False)[:500])
    print("→ 把返回的上传地址 PUT 成人 PNG 后，再走 create task（下同）：")
    # 2) 创建图生模型任务（字段名以官方文档为准）
    create = _req(API_BASE + EP_TASK, key, {
        "type": "image_to_model",
        "file": {"type": "png", "url": "<上一步上传后的公网 URL>"},
    })
    print("[create]", json.dumps(create, ensure_ascii=False)[:500])
    task_id = (create.get("data") or {}).get("task_id")
    if not task_id:
        sys.exit("未拿到 task_id：对照文档检查响应字段")
    # 3) 轮询
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        st = _req(f"{API_BASE}{EP_TASK}/{task_id}", key)
        status = (st.get("data") or {}).get("status")
        print(f"[poll] {status}")
        if status in ("success", "failed", "banned"):
            print(json.dumps(st, ensure_ascii=False)[:2000])
            return
        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="生成输入图（无需 key）")
    ap.add_argument("--run", action="store_true", help="走 API（需 TRIPO_API_KEY）")
    args = ap.parse_args()
    if args.prepare or not args.run:
        prepare()
    if args.run:
        key = os.environ.get("TRIPO_API_KEY")
        if not key:
            sys.exit("缺 TRIPO_API_KEY（platform.tripo3d.ai 注册获取）")
        run_api(key)


if __name__ == "__main__":
    main()
