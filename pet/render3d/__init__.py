"""pet/render3d —— 3D 渲染实验线运行时（three_d D04/S2 线）。

包结构（设计-子模块与接口.md）::

    bootstrap → contract_adapter → {anim, morph, spring, lighting, bone_bridge}
                                    → scene_host → assets

铁律（D04）：本包与 pet/rig/ 零 import；唯一共享物 = pet.scene_contract。
任何异常不得外抛给调用方（EngineBridge/降级矩阵消费 is_ready）。
纯逻辑模块（assets/morph/spring/lighting）零 Qt 依赖，可在无显示环境单测。
"""

__version__ = "0.1.0"
