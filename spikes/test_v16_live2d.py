"""v0.15 Live2D 展示后端自动化验证（offscreen，不要求 GPU/真窗）。

覆盖：
  T1  resolve_model_path：默认 Haru 存在；缺文件 → None
  T2  mapping：合法 mapping 解析 expressions/motions；非法 JSON → None
  T3  expression_for：mood + neglected 回落
  T4  motion_for：有组 / 显式 null
  T5  装配：缺模型回退基类；有模型返回 Live2DWindow（GL 失败仍是同实例+label）
  T6  WindowBase 同接口：part_walk_active / set_lip_open / play_interaction 不崩
  T7  config schema 接受 presentation=live2d
运行：
  python -X utf8 spikes/test_v16_live2d.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication  # noqa: E402

from pet.asset_provider import SpriteRef  # noqa: E402
from pet.config import load_config  # noqa: E402
from pet.live2d.spec import (  # noqa: E402
    load_live2d_mapping,
    resolve_model_path,
)
from pet.window import WindowBase  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ✅ " if cond else "  ❌ ") + name)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARU = os.path.join(REPO, "assets", "live2d", "haru", "Haru.model3.json")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    # ---- T1 路径 ----
    p = resolve_model_path({})
    check("T1a 默认 Haru 存在", p is not None and os.path.isfile(p))
    check("T1b 显式相对路径",
          resolve_model_path({"model": "assets/live2d/haru/Haru.model3.json"})
          == os.path.normpath(HARU))
    check("T1c 缺文件 → None",
          resolve_model_path({"model": "assets/live2d/nope/nope.model3.json"})
          is None)

    # ---- T2 mapping ----
    m = load_live2d_mapping(HARU)
    check("T2a 默认 mapping 非 None", m is not None)
    check("T2b display_name=Haru", m is not None and m.display_name == "Haru")
    check("T2c happy→F05", m is not None and m.expressions.get("happy") == "F05")
    tmp = tempfile.mkdtemp()
    bad = os.path.join(tmp, "mapping.json")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("{ not json")
    check("T2d 非法 mapping → None",
          load_live2d_mapping(HARU, bad) is None)
    tmp2 = tempfile.mkdtemp()
    empty_model = os.path.join(tmp2, "Empty.model3.json")
    with open(empty_model, "w", encoding="utf-8") as f:
        json.dump({"Version": 3, "FileReferences": {}}, f)
    m2 = load_live2d_mapping(empty_model)  # 无 mapping.json → 空表仍可用
    check("T2e 无 mapping.json 仍返回对象", m2 is not None)

    # ---- T3 expression_for ----
    check("T3a happy", m.expression_for("happy") == "F05")
    check("T3b neglected 优先 F02",
          m.expression_for("happy", neglected=True) == "F02")
    check("T3c 未知 mood 回落 neutral F01",
          m.expression_for("???") == "F01")

    # ---- T4 motions ----
    pat = m.motion_for("pat")
    check("T4a pat 组 TapBody",
          pat is not None and pat.group == "TapBody" and pat.priority == 3)
    check("T4b blink 显式 null", m.motion_for("blink") is None)
    check("T4c 未知动作 None", m.motion_for("dance") is None)

    # ---- T5 装配 ----
    from pet.live2d.presenter import Live2DWindow, build_live2d_window

    sprite = SpriteRef(path="🐱", width=192, height=192)
    miss = build_live2d_window(
        WindowBase, sprite, {"model": "assets/live2d/missing.model3.json"})
    check("T5a 缺模型回退 WindowBase",
          type(miss) is WindowBase and not getattr(miss, "live2d_active", False))
    win = build_live2d_window(WindowBase, sprite, {
        "model": "assets/live2d/haru/Haru.model3.json",
        "idle": False,
        "gaze": False,
    })
    check("T5b 有模型返回 Live2DWindow", isinstance(win, Live2DWindow))
    # offscreen 下 GL 可能失败——失败不得阻断，且 set_sprite 仍可用
    try:
        win.set_sprite(sprite)
        ok = True
    except Exception:
        ok = False
    check("T5c set_sprite 不崩", ok)

    # ---- T6 同接口 ----
    try:
        win.set_motion_params(tilt_deg=3.0, walking=True, walk_hz=1.2)
        win.set_facing(-1)
        win.set_lip_open(0.6)
        win.play_interaction("pet")
        win.play_frames([])
        win.dispose_presentation()
        iface_ok = True
    except Exception:
        iface_ok = False
        import traceback
        traceback.print_exc()
    check("T6 运动/口型/交互/释放不崩", iface_ok)

    # ---- T7 config ----
    cfg_dir = tempfile.mkdtemp()
    cfg_path = os.path.join(cfg_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "config_version": 1,
            "presentation": "live2d",
            "live2d": {"model": "assets/live2d/haru/Haru.model3.json",
                       "gaze": False},
        }, f)
    loaded = load_config(cfg_path)
    check("T7a presentation=live2d 过校验",
          loaded.get("presentation") == "live2d")
    check("T7b live2d.gaze 合并保留",
          loaded.get("live2d", {}).get("gaze") is False)

    print(f"\n通过 {len(PASS)}  失败 {len(FAIL)}")
    for n in FAIL:
        print("  FAIL", n)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
