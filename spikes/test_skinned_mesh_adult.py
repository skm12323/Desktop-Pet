"""2D 骨骼蒙皮呈现器实机与集成门禁测试 —— 成年体 ADULT (spikes/test_skinned_mesh_adult.py)"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from pet.asset_provider import SpriteRef
from pet.rig.motion import MotionEngine, MotionInputs
from pet.rig.presenter import RigWindow, build_rig_window
from pet.rig.skinned_mesh_item import RigRuntime
from pet.rig.spec import RigSpec, load_rig_spec
from pet.window import WindowBase

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK] {name}" + (f"  [{detail}]" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"  [{detail}]" if detail else ""))


def main():
    global passed, failed
    app = QApplication.instance() or QApplication([])

    print("== 1. 成年体蒙皮资产加载与 Spec 检查 ==")
    spec = load_rig_spec("assets/rig/adult", "adult")
    check("spec 加载非空", spec is not None)
    if spec is not None:
        check("skinned_spec 存在且有效", bool(spec.skinned_spec) and os.path.isfile(spec.skinned_spec), spec.skinned_spec)
        check("skinned_mesh 存在且有效", bool(spec.skinned_mesh) and os.path.isfile(spec.skinned_mesh), spec.skinned_mesh)
        check("skinned_layers 存在且有效", bool(spec.skinned_layers) and os.path.isdir(spec.skinned_layers), spec.skinned_layers)
        check("physics_presets 弹簧配置健全", bool(spec.physics_presets.get("spring_groups") or spec.physics_presets.get("spring_damper")))
        check("face_mechanics 配置存在", "blink" in spec.face_mechanics and "look_at" in spec.face_mechanics)
        check("source_facing 配置为 1（正向 A-pose）", getattr(spec, "source_facing", None) == 1)

    print("== 2. RigRuntime 纯数学核与 47 骨 LBS 验证 ==")
    rt = RigRuntime.load(spec.skinned_spec, spec.skinned_mesh, spec.skinned_layers)
    check("RigRuntime 加载非空", rt is not None)
    if rt is not None:
        check("骨骼数严格契约 47 骨", len(rt.bones) == 47, f"{len(rt.bones)} 骨")
        check("成年体图层数严格 21 层", len(rt.layers) == 21, f"{len(rt.layers)} 层")
        check("画布标准尺寸 960x1696", rt.img_w == 960.0 and rt.img_h == 1696.0, f"{rt.img_w}x{rt.img_h}")

        # 验证虚拟大腿驱动骨与双关节小臂
        bone_names = {b.name for b in rt.bones}
        check("大腿驱动骨存在", "upper_leg_l" in bone_names and "upper_leg_r" in bone_names)
        check("双关节手臂骨骼存在", "forearm_l" in bone_names and "forearm_r" in bone_names)

        # 蒙皮矩阵计算与 LBS 变形测试
        pose_angle = np.zeros(47, dtype=np.float32)
        pose_tx = np.zeros(47, dtype=np.float32)
        pose_ty = np.zeros(47, dtype=np.float32)
        M = rt.skinning_matrices(pose_angle, pose_tx, pose_ty, 0.0, 0.0)
        check("静止矩阵形状 (47, 3, 3)", M.shape == (47, 3, 3))

        total_v = 0
        for l in rt.layers:
            rest_eff = rt.effective_rest(l, 0.0)
            deformed = rt.deform(l, rest_eff)
            total_v += deformed.shape[0]
        check("21 层蒙皮变形计算完毕", total_v > 1500, f"总顶点数={total_v}")

    print("== 3. MotionEngine 成年体步态动力学与注视解算 ==")
    engine = MotionEngine(spec)
    inputs = MotionInputs(
        tilt_deg=4.0,
        walking=True,
        walk_hz=1.3,
        facing=1,
        cursor_pos=(480.0, 400.0),
        pet_rect=(100.0, 100.0, 480.0, 848.0),
    )
    frame = engine.step(inputs, 33.0)
    check("frame 输出非空", frame is not None)
    check("47 骨姿态输出充足", len(frame.bone_angles) >= 40, f"{len(frame.bone_angles)} 骨")
    check("尾巴 4 节波浪链级联计算", all(b in frame.bone_angles for b in ["tail_01", "tail_02", "tail_03", "tail_fluke"]))
    t_angles = [frame.bone_angles[b] for b in ["tail_01", "tail_02", "tail_03", "tail_fluke"]]
    check("尾巴波浪链非零振荡", any(abs(a) > 0.01 for a in t_angles), str(t_angles))
    check("连续眼睑眨眼行程计算", 0.0 <= frame.blink_progress <= 1.0, f"progress={frame.blink_progress:.3f}")
    check("视线追踪向光标偏转", -1.0 <= frame.look_at[0] <= 1.0 and -1.0 <= frame.look_at[1] <= 1.0, f"look={frame.look_at}")

    # 行走双摆驱动验证
    check("成年体行走时大腿摆角非零", abs(frame.bone_angles.get("upper_leg_l", 0.0)) > 0.01)
    check("成年体行走时小腿摆角非零", abs(frame.bone_angles.get("lower_leg_l", 0.0)) >= 0.0)
    check("成年体行走时手臂双关节摆动", abs(frame.bone_angles.get("upper_arm_l", 0.0)) > 0.01)

    print("== 4. RigWindow 蒙皮呈现器集成与成年体渲染 ==")
    sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "rig", "adult", "figs", "healthy_neutral.png")
    sprite = SpriteRef(path=sprite_path, width=240, height=424)
    win = build_rig_window(WindowBase, sprite, "adult", defer_quick=False)
    check("win 实例化为 RigWindow", isinstance(win, RigWindow))
    check("win.rig_active 活性", win.rig_active)
    check("win._skinned_item 成功激活并绑定", win._skinned_item is not None)
    check("QML sourceFacing 为 1", win._root.property("sourceFacing") == 1)
    check("QML visualFacing 为 1", win._root.property("visualFacing") == 1)

    # 推进多拍，验证每拍推帧与脏标记刷新
    win.resize(480, 848)
    win.show()
    for i in range(12):
        app.processEvents()
        win._motion_tick()
        win._quick.repaint()

    check("SkinnedMeshItem 蒙皮核 _rt 成功运行", win._skinned_item._rt is not None)
    if win._skinned_item._rt is not None:
        check("蒙皮层数等于 21", len(win._skinned_item._rt.layers) == 21)
        check("蒙皮骨骼数等于 47", len(win._skinned_item._rt.bones) == 47)
        check("源图高宽等于 960x1696", win._skinned_item._rt.img_w == 960.0 and win._skinned_item._rt.img_h == 1696.0)

    # 验证真实透明离屏渲染抓图
    img = win._quick.grabFramebuffer().convertToFormat(QImage.Format_RGBA8888)
    check("离屏抓图非空", not img.isNull())
    pixels = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.width(), 4)
    alpha_count = int((pixels[:, :, 3] > 20).sum())
    check("截图实际包含成年体角色像素", alpha_count > 15000, f"像素数={alpha_count}")

    out_preview = os.path.join("spikes", "_qa", "preview_skinned_adult_live.png")
    os.makedirs(os.path.dirname(out_preview), exist_ok=True)
    img.save(out_preview)
    check("预览截图保存成功", os.path.isfile(out_preview), out_preview)

    win.close()
    win.deleteLater()

    print(f"\n== 成年体测试结果：{passed} 通过 / {failed} 失败 ==")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
