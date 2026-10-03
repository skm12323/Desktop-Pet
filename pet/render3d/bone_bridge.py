"""骨骼批量桥（Qt 依赖）——Python→QML 逐帧骨骼变换通道。

设计（调研-引擎页 A-6）：**每帧 1 次 Signal 携带紧凑 payload**，QML 侧 JS 分发到
Joint——避免数百次/帧的 Python→C++ 逐属性穿越（估 0.5–2ms/帧）。payload 形态 =
扁平 list[float]（n 骨 × 7：px,py,pz,qx,qy,qz,qw），QML 侧按下标切片。

pose_to_joint_payload：由契约 PoseSemantics + 逆向骨架基准（sidecar
skeleton3d.json）合成每帧关节位姿。v0 变换 = 中轴 yaw 旋转 + phase 驱动的
四肢摆动近似（真 LBS/clip 驱动待 S1.4 蒙皮模型；本函数负责通道与坐标系打通）。
"""

from __future__ import annotations

import math

from pet.scene_contract import PoseSemantics

STRIDE = 7  # px,py,pz,qx,qy,qz,qw


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """绕 y 轴的四元数（契约 view_yaw → 场景朝向；场景 +z=面朝）。"""
    h = math.radians(yaw_deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def pose_to_joint_payload(pose: PoseSemantics, skeleton: dict) -> list[float]:
    """skeleton = skeleton3d_adult.json 全量 dict（meta + bones）。

    输出扁平 list：每骨 7 float（位置已绕中轴旋 yaw；四元数=根 yaw）。
    骨骼缺 sidecar 时返回空 list（QML 侧跳过）。
    """
    bones = (skeleton or {}).get("bones") or []
    if not bones:
        return []
    ppm = float((skeleton.get("meta") or {}).get("px_per_meter_proposal", 1000.0))
    yaw = math.radians(pose.view_yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # phase 驱动的摆动近似：走路时四肢按相位正弦摆（±12°），idle 呼吸 ±2°
    swing = 0.0
    if pose.action_id == "walk":
        swing = 12.0 * math.sin(pose.phase * 2.0 * math.pi)
    q = _yaw_quat(pose.view_yaw_deg)
    out: list[float] = []
    for b in bones:
        name = b.get("bone_name") or b.get("name") or ""
        x, y, z = (v / ppm for v in b["pos_px"])   # y 向上自地面
        # 中轴 yaw 旋转（x-z 平面）
        rx = x * cy + z * sy
        rz = -x * sy + z * cy
        # 四肢摆动近似：行走相位调制 y（上下交替）——v0 占位，S1.4 换真 IK/LBS
        ry = y + (swing / 360.0) * 0.1 if ("leg" in name or "arm" in name) else y
        out += [round(rx, 4), round(ry, 4), round(rz, 4), *q]
    return out


def payload_slice(payload: list[float], index: int) -> tuple[float, ...]:
    """QML 侧取第 index 骨的 7 元组（越界返回全零）。"""
    base = index * STRIDE
    if base + STRIDE > len(payload):
        return (0.0,) * STRIDE
    return tuple(payload[base:base + STRIDE])
