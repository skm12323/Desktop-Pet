"""rig 呈现层（v0.20.0 起唯一展示后端）—— 分层绑骨 + 2D 骨骼蒙皮 + 侧身行走。

``pet.rig.presenter.RigWindow`` 继承 ``WindowBase`` 复用全部手势/菜单/拖放，
画面驱动为 Qt Quick 场景：正面蒙皮网格（assets/rig_{stage}）、ADULT/FINAL
转身片段 + 侧身骨骼行走（assets/rig_{stage}_walk_v1）、mood 立绘交叉淡化与
部件弹簧（assets/rig/{stage}/manifest.json）、动作帧（assets/frames）。

模块仅含平台中立代码；QML 场景在 ``rig_scene.qml``。任一资产/环境缺失时
build_rig_window 返回平台基类窗（静态立绘 + 动作帧）。
"""

from .spec import RigSpec, load_rig_spec

__all__ = ["RigSpec", "load_rig_spec"]
