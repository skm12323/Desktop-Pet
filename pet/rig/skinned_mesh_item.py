"""骨架蒙皮网格渲染件（Linear Blend Skinning on Qt Quick Scene Graph）。

把图层与骨架（``assets/rig_young/spec.json``）的蒙皮规范落到
``QQuickItem`` 自定义场景图节点上，替代刚体旋转切片（paper-doll）的僵硬观感：
驱动方每帧推 ``setBonePose`` / ``setBlink`` / ``setLookAt``，本件做 FK + LBS
后把变形顶点直写场景图顶点缓冲。与 motion.py 分工同构：本件是**哑渲染器**
——不做补间/平滑（spec ``face_mechanics.look_at.smoothing_time_ms`` 等时序
归驱动方），只把输入姿态确定性栅格化。

性能契约（渲染热路径逐条对照）：

* **无逐顶点 Python 循环**：蒙皮形变 = 每层一次
  ``np.einsum('vk,kij,vj->vi', w, M_sel, p)`` 全向量化
  （v=顶点、k=影响骨、ij=3×3 仿射，即 ``v'_k = Σ_b w_{k,b}·(M_b·[v_k,1])``）；
  FK 是 47 骨的**骨级**循环（非顶点级）。
* **零拷贝顶点写入**：``int(geom.vertexData())`` 取 C++ 顶点缓冲地址，
  ``(ctypes.c_float * (4·V)).from_address`` + ``np.ctypeslib.as_array`` 建
  NumPy 视图直读直写；``arr[:, 0:2] = deformed_xy`` 后
  ``geom.markVertexDataDirty()``。
* **复用帧缓冲**：FK 矩阵栈、LBS 输出、眨眼形变与视图变换复用 NumPy
  缓冲；骨矩阵的高级索引仍产生小型临时数组。
* **纹理显式生命周期**：``window().createTextureFromImage(QImage)`` 创建的
  QSGTexture 不由场景图托管、须调用者显式删；登记进纹理表，在渲染线程
  ``_release_textures`` 显式释放，场景图失效时只丢引用（上下文拆除已释放）。

硬件后端使用持久 QSGGeometryNode + QSGTextureMaterial，逐帧更新顶点。
软件后端不支持此自定义材质，由 ready=False 通知呈现器回退分层位图。
Python Flag 不支持 int() 并不表示 Qt markDirty 失效；不可据此叠挂旧树。

坐标空间（单一真相源）：

* 骨与顶点全部活在**源参考图像素空间**（左上原点、y 向下；spec 的
  ``joint_pos`` 归一化坐标 × ``image_size_px`` 入境）。
* 静止姿态 = 全部单位旋转（spec 只给关节位置），``T_rest(b)`` 为纯平移，
  ``M_b = T_b · T_rest(b)^{-1}`` 在零姿态下恒等于 I。
* 局部矩阵 ``L_b = inv(T_rest(parent)) · T(J_b) · T(t_b) · R(θ_b)``：
  关节位置按**世界静止坐标**解释（spine.joint=[0.469,0.681] 若当父系局部
  偏移会落到画面外），父骨旋转时子关节绕父关节摆动——经典 FK 链行为。
* 渲染前按 KeepAspectRatio + 居中（与 rig_scene.qml 的 fitScale/offX/offY
  同构）把源图像素映射到 item 坐标。

meshDataFile 格式（json，UTF-8；资产工具产出的中间格式）::

    {
      "spec": 1,
      "image_size_px": [1280, 1284],        # 顶点坐标的像素空间基准
      "layers": [
        { "id": "tail_seg2",                # 对应 spec.json layers[].id
          "texture": "tail_seg2.png",       # 相对 layersDir；缺省 f"{id}.png"
          "z_order": 10,                    # 缺省取 spec 同名层，再缺省按文件序
          "vertices": [[x, y], ...],        # 静止顶点（源图像素，y 向下）
          "uvs": [[u, v], ...],             # 归一化纹理坐标（v=0 = 图顶）
          "triangles": [i0, j0, k0, ...],   # 平铺三角索引（uint16 界）
          "weight_bones": [["tail_02", "tail_03", "tail_fluke"], ...],
          "weight_values": [[0.5, 0.35, 0.15], ...]   # 每顶点，自动归一化
        }
      ]
    }

面部专用通道（数值优先读 spec ``face_mechanics``，缺省用任务规范常数兜底）：

* look-at：``lookAtX/lookAtY`` 归一化偏移 → pupil_l/pupil_r 骨平移，椭圆限幅
  ``(dx/10)² + (dy/7)² ≤ 1``（10/7 = spec ``look_at.axis_limits_px``，源图
  像素）。专用通道**覆盖**对瞳骨的 ``setBonePose`` 平移（瞳骨 clamp=[0,0]
  本就只走平移）。
* blink：``blink_delta`` 是每顶点闭眼位移，眼白、瞳孔、睫毛共用闭合曲线；
  先在静止空间变形，再做 LBS。旧网格无此字段时保留旧版眼睑挤压。

降级铁律（宽进严出）：spec/mesh 任一整体损坏 → 空渲染 + 一次性告警；单层
损坏（顶点/UV/索引/权重/纹理任一不合格）→ 弃层不弃场；权重引用未知骨 →
剔除该影响并归一化；渲染期任何异常按帧吞掉并留痕。绝不向宿主窗口抛异常。

QML 用法（``import`` 本模块即完成 ``PetRig 1.0`` 类型注册）::

    import QtQuick
    import PetRig 1.0

    SkinnedMeshItem {
        anchors.fill: parent
        specFile: "assets/reference/young_rig_spec.json"
        meshDataFile: "assets/rig_young/mesh/mesh_data.json"
        layersDir: "assets/rig_young/layers"
    }
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtQuick import (
    QQuickItem,
    QQuickWindow,
    QSGGeometry,
    QSGGeometryNode,
    QSGMaterial,
    QSGNode,
    QSGTexture,
    QSGTextureMaterial,
)

# QSGGeometry stores a reference: keep the Python AttributeSet alive.
_TEXTURED_ATTRIBUTES = QSGGeometry.defaultAttributes_TexturedPoint2D()

log = logging.getLogger("pet")

__all__ = ["SkinnedMeshItem", "RigRuntime", "register_qml_type"]

# PySide6 版本差异：6.x 中后期枚举 scoped 化（QQuickItem.Flag.ItemHasContents），
# 早期仅暴露非 scoped 别名——两手抓，构造期即定死
try:
    _ITEM_HAS_CONTENTS = QQuickItem.Flag.ItemHasContents
except AttributeError:                     # pragma: no cover —— 旧版绑定
    _ITEM_HAS_CONTENTS = QQuickItem.ItemHasContents   # type: ignore[attr-defined]


# ============================ 纯数学核（零 Qt 对象依赖） ============================
#
# RigRuntime 只持有 NumPy 数组：骨骼层级、静止/逆绑定矩阵、逐层蒙皮缓冲。
# 可脱离窗口直接单测（FK 恒等性、旋转传播、限幅、挤压都在这一层验证）。


def _mat_trans(tx: float, ty: float) -> np.ndarray:
    """平移 3×3 仿射（float64；静止位形纯平移，动态绑定视角重算共用）。"""
    m = np.zeros((3, 3), np.float64)
    m[0, 0] = m[1, 1] = m[2, 2] = 1.0
    m[0, 2], m[1, 2] = tx, ty
    return m


def _inv_affine3(m: np.ndarray) -> np.ndarray:
    """3×3 仿射封闭求逆（float64）；退化（det≈0）回退单位阵——防御。"""
    a, b, c = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    d, e, f = float(m[1, 0]), float(m[1, 1]), float(m[1, 2])
    det = a * e - b * d
    if not math.isfinite(det) or abs(det) < 1e-12:
        return np.eye(3, dtype=np.float64)
    out = np.empty((3, 3), np.float64)
    out[0, 0], out[0, 1], out[0, 2] = e / det, -b / det, (b * f - c * e) / det
    out[1, 0], out[1, 1], out[1, 2] = -d / det, a / det, (c * d - a * f) / det
    out[2, 0], out[2, 1], out[2, 2] = 0.0, 0.0, 1.0
    return out


def _build_bind_matrices(bones: list, parent_idx: np.ndarray,
                         joints_xy: np.ndarray,
                         order: "list[int] | None" = None) -> tuple:
    """静止绑定三件套（``w_rest`` 隐含）：按世界静止关节建局部/逆绑定矩阵。

    与 load 期同式（关节位置按全图世界坐标解释，静止姿态无旋转）；
    ``set_view_yaw`` 的动态绑定重算共用此函数（§4.2 Dynamic Bind Poses）。
    ``order`` 为拓扑序（父先于子）——缺省按数组序（仅当骨骼已父先子后时
    正确；非根骨的父矩阵未就绪会静默产出错误绑定）。
    """
    B = len(bones)
    a_local = np.zeros((B, 3, 3), np.float64)
    inv_w_rest = np.zeros((B, 3, 3), np.float64)
    w_rest = np.zeros((B, 3, 3), np.float64)
    for i in (order if order is not None else range(B)):
        t = _mat_trans(float(joints_xy[i, 0]), float(joints_xy[i, 1]))
        p = int(parent_idx[i])
        w_rest[i] = t
        a_local[i] = t if p < 0 else _inv_affine3(w_rest[p]) @ t
        inv_w_rest[i] = _inv_affine3(w_rest[i])
    return a_local, inv_w_rest


# ============================ 视角关键形态（§4/§6.2 数据契约） ============================


class ViewKeyforms:
    """``view_keyforms.json`` 解析与插值核（纯数学，无 Qt 依赖）。

    * 严格校验（任一失败抛 ValueError → 调用方整体回退正面单视角模式）：
      版本号、视角 yaw 升序且唯一、逐层顶点数一致、positions/tangents 形状
      有限、骨骼 pivot 覆盖。
    * Hermite 插值（§4.3）：切线由构建工具按 Catmull-Rom 提供（per yaw），
      运行时换算到 μ 单位；端视角外钳位。
    * 零姿态恒等性（§4.2）为**结构性质**：任意 yaw 的绑定矩阵由同一组插值
      关节重建，零姿态下 M = W·inv(W) = I（float64 舍入 ~1e-12 px）。
    * 补片透明度（§4.4）：余弦加权平滑阶跃，C1 且端点钳 0/1。
    """

    def __init__(self, raw: dict):
        if not isinstance(raw, dict) or str(raw.get("version")) != "1.0":
            raise ValueError("view_keyforms 缺 version=1.0")
        views = raw.get("views")
        if not isinstance(views, dict) or len(views) < 2:
            raise ValueError("view_keyforms 需要至少 2 个视角")
        names = sorted(views, key=lambda n: float(views[n]["yaw_deg"]))
        yaws = [float(views[n]["yaw_deg"]) for n in names]
        if any(b <= a for a, b in zip(yaws, yaws[1:])):
            raise ValueError(f"视角 yaw 必须严格升序：{list(zip(names, yaws))}")
        self.view_names = names
        self.yaws = np.asarray(yaws, np.float64)
        k = len(names)
        idx = {n: i for i, n in enumerate(names)}

        binds = raw.get("bones_dynamic_bind") or {}
        self.joints: dict[str, np.ndarray] = {}      # bone -> (K,2)
        for n in names:
            for bone, rec in (binds.get(n) or {}).items():
                piv = rec.get("pivot")
                if not isinstance(piv, (list, tuple)) or len(piv) != 2:
                    raise ValueError(f"{n}/{bone} pivot 非法")
                arr = self.joints.setdefault(bone, np.full((k, 2), np.nan))
                arr[idx[n]] = (float(piv[0]), float(piv[1]))
        for bone, arr in self.joints.items():
            if not np.isfinite(arr).all():
                raise ValueError(f"骨骼 {bone} 缺部分视角 pivot")
        # 关节切线：与构建工具同式的 Catmull-Rom（per yaw，内部中心差分）
        self.joint_tans: dict[str, np.ndarray] = {
            bone: self._catmull_rom_per_yaw(arr, self.yaws)
            for bone, arr in self.joints.items()}

        layers = raw.get("layers_keyforms")
        if not isinstance(layers, dict) or not layers:
            raise ValueError("view_keyforms 缺 layers_keyforms")
        self.layers: dict[str, dict] = {}
        for lid, rec in layers.items():
            pos = rec.get("positions") or {}
            tan = rec.get("tangents") or {}
            vcount = int(rec.get("vertex_count", -1))
            pos_arr = np.full((k, max(vcount, 0), 2), np.nan, np.float64)
            tan_arr = np.zeros((k, max(vcount, 0), 2), np.float64)
            for n in names:
                p = pos.get(n)
                if not isinstance(p, list):
                    raise ValueError(f"{lid} 缺 {n} positions")
                a = np.asarray(p, np.float64)
                if a.shape != (vcount, 2):
                    raise ValueError(f"{lid}/{n} positions 形状 {a.shape} ≠ ({vcount},2)")
                pos_arr[idx[n]] = a
                t = tan.get(n)
                ta = np.asarray(t, np.float64) if t is not None else np.zeros_like(a)
                if ta.shape != (vcount, 2):
                    raise ValueError(f"{lid}/{n} tangents 形状非法")
                tan_arr[idx[n]] = ta
            if not (np.isfinite(pos_arr).all() and np.isfinite(tan_arr).all()):
                raise ValueError(f"{lid} positions/tangents 含非有限值")
            self.layers[lid] = {"positions": pos_arr, "tangents": tan_arr}

        self.patches: list[dict] = []
        for p in raw.get("occlusion_patches") or []:
            try:
                self.patches.append({
                    "patch_id": str(p["patch_id"]),
                    "base_layer": str(p.get("base_layer", p["patch_id"])),
                    "yaw_range": (float(p["visible_yaw_range"][0]),
                                  float(p["visible_yaw_range"][1])),
                    "alpha": None,
                })
            except (KeyError, TypeError, ValueError, IndexError) as e:
                log.warning("occlusion_patch 解析失败，弃片：%s", e)

    @staticmethod
    def _catmull_rom_per_yaw(keys: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """(K,…) 关键序列的 Catmull-Rom 切线（per yaw：内部中心差分，端点单侧）。"""
        k = keys.shape[0]
        tans = np.empty_like(keys)
        for i in range(k):
            if 0 < i < k - 1:
                tans[i] = (keys[i + 1] - keys[i - 1]) / (yaws[i + 1] - yaws[i - 1])
            elif i == 0:
                tans[i] = (keys[1] - keys[0]) / (yaws[1] - yaws[0])
            else:
                tans[i] = (keys[-1] - keys[-2]) / (yaws[-1] - yaws[-2])
        return tans

    def _bracket(self, yaw: float) -> tuple:
        y = min(max(float(yaw), float(self.yaws[0])), float(self.yaws[-1]))
        j = int(np.searchsorted(self.yaws, y))
        j = min(max(j - 1, 0), len(self.yaws) - 2)
        d = float(self.yaws[j + 1] - self.yaws[j])
        mu = 0.0 if d <= 0 else min(max((y - float(self.yaws[j])) / d, 0.0), 1.0)
        return j, d, mu

    def interp_layer(self, layer_id: str, yaw: float) -> np.ndarray:
        """(V,2) float64 Hermite 插值（§4.3 公式；切线按 Δyaw 换算到 μ）。"""
        rec = self.layers[layer_id]
        j, d, mu = self._bracket(yaw)
        p0 = rec["positions"][j]
        p1 = rec["positions"][j + 1]
        m0 = rec["tangents"][j] * d
        m1 = rec["tangents"][j + 1] * d
        mu2, mu3 = mu * mu, mu * mu * mu
        h00 = 2 * mu3 - 3 * mu2 + 1
        h10 = mu3 - 2 * mu2 + mu
        h01 = -2 * mu3 + 3 * mu2
        h11 = mu3 - mu2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def interp_joints(self, yaw: float) -> dict[str, np.ndarray]:
        j, d, mu = self._bracket(yaw)
        out = {}
        for bone, keys in self.joints.items():
            p0, p1 = keys[j], keys[j + 1]
            m0 = self.joint_tans[bone][j] * d
            m1 = self.joint_tans[bone][j + 1] * d
            mu2, mu3 = mu * mu, mu * mu * mu
            out[bone] = ((2 * mu3 - 3 * mu2 + 1) * p0 + (mu3 - 2 * mu2 + mu) * m0
                         + (-2 * mu3 + 3 * mu2) * p1 + (mu3 - mu2) * m1)
        return out

    @staticmethod
    def patch_alpha(yaw_range: tuple, yaw: float) -> float:
        """§4.4 余弦加权平滑阶跃（0/1 钳位，C1）。"""
        t0, t1 = yaw_range
        if t1 <= t0:
            return 1.0 if yaw >= t1 else 0.0
        if yaw <= t0:
            return 0.0
        if yaw >= t1:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * (yaw - t0) / (t1 - t0)))

    @classmethod
    def load(cls, path: str) -> "ViewKeyforms":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw)


@dataclass(frozen=True)
class _BoneDef:
    """一根骨的静态描述（joint_px：源图像素）。"""

    name: str
    parent: str | None
    joint_px: tuple[float, float]
    clamp: tuple[float, float]


@dataclass(frozen=True)
class _EyeBlinkCfg:
    """单眼眨眼几何（spec face_mechanics.blink，源图像素）。"""

    layer_id: str
    upper_piv_y: float
    lower_piv_y: float
    center_y: float
    upper_frac: float       # 上睑行程占比（spec 0.78）
    lower_frac: float       # 下睑行程占比（spec 0.22）


@dataclass(frozen=True)
class _LookAtCfg:
    """注视通道参数（spec face_mechanics.look_at）。"""

    axis_px: tuple[float, float]      # 椭圆半轴（源图像素），spec [10, 7]
    pupil_bones: tuple[str, ...]


@dataclass
class _LayerSkin:
    """一层蒙皮网格：静止数据 + 预分配帧缓冲（渲染热路径零分配的载体）。"""

    layer_id: str
    z_order: int
    texture_path: str
    bind_bone: str
    rest: np.ndarray                 # (V,3) f32 齐次静止坐标（源图像素）
    uv: np.ndarray                   # (V,2) f32
    triangles: np.ndarray            # (T,) uint16 平铺索引
    bone_idx: np.ndarray             # (K,) i32 → 全骨数组下标
    weights: np.ndarray              # (V,K) f32（行和=1）
    blink: _EyeBlinkCfg | None = None
    blink_delta: np.ndarray | None = None
    gaze_uv: bool = False
    texture_size: tuple[float, float] = (1.0, 1.0)
    # ---- 以下均为帧复用缓冲，加载期一次分配 ----
    scratch: np.ndarray | None = None     # (V,3) LBS 输出
    eff_rest: np.ndarray | None = None    # (V,3) 眼睑挤压后的有效静止坐标
    upper_mask: np.ndarray | None = None  # (V,) bool：y < closure_center
    lower_mask: np.ndarray | None = None
    _ybuf: np.ndarray | None = None       # (V,) 标量链中间量
    _blink_applied: float | None = None   # 缓存判定


class RigRuntime:
    """骨架 + 网格的静态数据与 FK/LBS 计算核（无 Qt 对象，可独立单测）。

    矩阵列向量约定：``p' = M @ [x, y, 1]ᵀ``；全局 ``T_b = T_parent @ L_b``；
    蒙皮 ``M_b = T_b · T_rest(b)⁻¹``。legacy 单视角路径全部数组 float32
    （像素空间幅值 ≤ ~2¹⁰，精度充裕）；**视角关键形态模式升级为 float64**
    （§4.2 零姿态恒等 1e-6 px 门槛在 ~1600 px 幅值下超出 float32 量化能力
    ~1e-4，见 attach_view_keyforms）。
    """

    texture_mipmaps: bool = False    # spec "texture_mipmaps"（load 时赋值）

    def __init__(self, bones: list[_BoneDef], parent_idx: np.ndarray,
                 a_local_rest: np.ndarray, inv_w_rest: np.ndarray,
                 layers: list[_LayerSkin], look: _LookAtCfg,
                 img_w: float, img_h: float) -> None:
        self.bones = bones                      # 拓扑序（父先于子）
        self.bone_index: dict[str, int] = {b.name: i for i, b in enumerate(bones)}
        self.parent_idx = parent_idx            # (B,) i32；根 = -1
        self._bone_order = _topo_order(bones)[0]
        self.layers = layers                    # z_order 升序
        self.look = look
        self.img_w = img_w
        self.img_h = img_h
        uv_bones = {layer.bind_bone for layer in layers if layer.gaze_uv}
        self.pupil_idx = np.array(
            [self.bone_index[n] for n in look.pupil_bones
             if n in self.bone_index and n not in uv_bones],
            dtype=np.int32)

        # ---- 视角关键形态态（attach_view_keyforms 激活）----
        self._kf: ViewKeyforms | None = None
        self.view_yaw: float | None = None      # 当前已应用视角（None=未激活）

        # ---- 静止矩阵（加载期一次算清）----
        self._a_local_rest = a_local_rest       # (B,3,3) L_b 的零姿态部分
        self.inv_w_rest = inv_w_rest            # (B,3,3) T_rest⁻¹
        self._clamp_lo = np.array([b.clamp[0] for b in bones], np.float32)
        self._clamp_hi = np.array([b.clamp[1] for b in bones], np.float32)

        # ---- 帧复用缓冲（热路径零分配）----
        n = len(bones)
        self._L = a_local_rest.copy()
        self._W = a_local_rest.copy()
        self.M = np.zeros_like(a_local_rest)    # 蒙皮矩阵（调用方读）
        self._rad = np.zeros(n, np.float32)
        self._cos = np.zeros(n, np.float32)
        self._sin = np.zeros(n, np.float32)
        self._tx = np.zeros(n, np.float32)
        self._ty = np.zeros(n, np.float32)
        max_v = max((l.rest.shape[0] for l in layers), default=1)
        self.buf_x = np.zeros(max_v, np.float32)
        self.buf_y = np.zeros(max_v, np.float32)

    # ---------------- 视角关键形态（§4 动态绑定） ----------------

    def attach_view_keyforms(self, path_or_kf) -> bool:
        """挂接 view_keyforms（失败 → 整体回退正面单视角，宽进严出）。

        激活后数学核升级 float64：绑定矩阵/静止顶点/蒙皮缓冲全部换 64 位，
        ``set_view_yaw`` 重算动态绑定与各层静止位置。GPU 顶点缓冲仍 float32
        （写入时自动降采，量化 ~1e-4 px 远低于其余门槛）。
        """
        try:
            kf = path_or_kf if isinstance(path_or_kf, ViewKeyforms) \
                else ViewKeyforms.load(str(path_or_kf))
            missing = [l.layer_id for l in self.layers
                       if l.layer_id not in kf.layers]
            if missing:
                raise ValueError(f"关键形态缺 {len(missing)} 层：{missing[:4]}")
        except Exception as e:                  # noqa: BLE001 —— 回退铁律
            log.warning("view keyforms 挂接失败，保持正面单视角：%s", e)
            self._kf = None
            self.view_yaw = None
            return False
        self._kf = kf
        self._a_local_rest = self._a_local_rest.astype(np.float64)
        self.inv_w_rest = self.inv_w_rest.astype(np.float64)
        self._L = self._L.astype(np.float64)
        self._W = self._W.astype(np.float64)
        self.M = self.M.astype(np.float64)
        for buf in (self._rad, self._cos, self._sin, self._tx, self._ty,
                    self.buf_x, self.buf_y):
            buf = None
        n = len(self.bones)
        self._rad = np.zeros(n, np.float64)
        self._cos = np.zeros(n, np.float64)
        self._sin = np.zeros(n, np.float64)
        self._tx = np.zeros(n, np.float64)
        self._ty = np.zeros(n, np.float64)
        max_v = max((l.rest.shape[0] for l in self.layers), default=1)
        self.buf_x = np.zeros(max_v, np.float64)
        self.buf_y = np.zeros(max_v, np.float64)
        for layer in self.layers:
            layer.rest = layer.rest.astype(np.float64)
            layer.scratch = layer.scratch.astype(np.float64)
            # 权重 float32 行和差 ~6e-8 × 坐标幅值 ~1600 ≈ 1e-4 px，超出
            # §4.2 零姿态恒等 1e-6 门槛——升 f64 并重归一（Σw=1 至 ~1e-16）
            w64 = layer.weights.astype(np.float64)
            rs = w64.sum(axis=1, keepdims=True)
            layer.weights = np.where(rs > 0.5, w64 / rs, w64)
            if layer.eff_rest is not None:
                layer.eff_rest = layer.eff_rest.astype(np.float64)
            if layer.blink_delta is not None:
                layer.blink_delta = layer.blink_delta.astype(np.float64)
            if layer._ybuf is not None:
                layer._ybuf = layer._ybuf.astype(np.float64)
            layer._blink_applied = None         # dtype 变更，失效眨眼缓存
        self.view_yaw = float(kf.yaws[0])
        self._apply_view(self.view_yaw)
        log.info("view keyforms 就绪：%d 视角 %s（float64 数学核）",
                 len(kf.yaws), np.round(kf.yaws, 1).tolist())
        return True

    def _apply_view(self, yaw: float) -> None:
        """按视角重算动态绑定（§4.2）与各层静止位置（§4.3 Hermite）。"""
        assert self._kf is not None
        joints_map = self._kf.interp_joints(yaw)
        joints_xy = np.empty((len(self.bones), 2), np.float64)
        for i, b in enumerate(self.bones):
            j = joints_map.get(b.name)
            joints_xy[i] = j if j is not None else b.joint_px   # 缺骨骼回退正面
        a_local, inv_w = _build_bind_matrices(self.bones, self.parent_idx,
                                              joints_xy, self._bone_order)
        self._a_local_rest[...] = a_local
        self.inv_w_rest[...] = inv_w
        for layer in self.layers:
            layer.rest[:, :2] = self._kf.interp_layer(layer.layer_id, yaw)
            # 眨眼缓存只按 blink 值失效——rest 随视角变化后必须一并失效，
            # 否则眼睑/瞳孔层返回陈旧位置（legacy 单视角无此问题）
            layer._blink_applied = None

    def set_view_yaw(self, yaw_deg: float) -> bool:
        """连续视角切换（端视角钳位）。返回是否实际重算（未变/未激活=False）。"""
        if self._kf is None:
            return False
        y = min(max(float(yaw_deg), float(self._kf.yaws[0])),
                float(self._kf.yaws[-1]))
        if self.view_yaw is not None and abs(y - self.view_yaw) < 1e-9:
            return False
        self.view_yaw = y
        self._apply_view(y)
        return True

    def patch_alphas(self, yaw: float | None = None) -> dict:
        """各补片当前透明度（§4.4 余弦阶跃；缺省用当前视角）。"""
        if not self._kf or not self._kf.patches:
            return {}
        y = self.view_yaw if yaw is None else yaw
        return {p["patch_id"]: ViewKeyforms.patch_alpha(p["yaw_range"], y)
                for p in self._kf.patches}

    # ---------------- FK ----------------

    def skinning_matrices(self, pose_angle: np.ndarray, pose_tx: np.ndarray,
                          pose_ty: np.ndarray, look_dx: float,
                          look_dy: float) -> np.ndarray:
        """FK 全链 → 蒙皮矩阵 ``M[b] = T_b · T_rest(b)⁻¹``（就地写 self.M）。

        ``pose_*`` 为 (B,) 全骨数组（缺省 0）；瞳骨平移被 look-at 专用通道
        覆盖。角度按 spec ``angle_clamp`` 钳制。除 47 骨级标量临时量外无分配。
        """
        np.clip(pose_angle, self._clamp_lo, self._clamp_hi, out=self._rad)
        np.radians(self._rad, out=self._rad)
        np.cos(self._rad, out=self._cos)
        np.sin(self._rad, out=self._sin)
        np.copyto(self._tx, pose_tx)
        np.copyto(self._ty, pose_ty)
        if self.pupil_idx.size:
            self._tx[self.pupil_idx] = look_dx
            self._ty[self.pupil_idx] = look_dy

        # L = A_local_rest @ T(t) @ R(θ)（列向量 p' = L@[p,1]）：
        # (A@R) 列 = A_col0·c + A_col1·s / −A_col0·s + A_col1·c；平移列 = A·[t,1]
        a, L = self._a_local_rest, self._L
        a0, a1, a2 = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        c, s = self._cos[:, None], self._sin[:, None]
        L[:, :, 0] = a0 * c + a1 * s
        L[:, :, 1] = a1 * c - a0 * s
        L[:, :, 2] = a0 * self._tx[:, None] + a1 * self._ty[:, None] + a2

        W = self._W
        for i in self._bone_order:
            p = int(self.parent_idx[i])
            if p < 0:
                W[i] = L[i]
            else:
                np.matmul(W[p], L[i], out=W[i])
        np.matmul(W, self.inv_w_rest, out=self.M)
        return self.M

    def look_offset(self, look_x: float, look_y: float) -> tuple[float, float]:
        """归一化注视偏移 → 瞳骨平移（源图像素），椭圆限幅 ``(dx/lx)²+(dy/ly)²≤1``。"""
        lx, ly = self.look.axis_px
        if lx <= 0 or ly <= 0:
            return 0.0, 0.0
        dx, dy = look_x * lx, look_y * ly
        rr = (dx / lx) ** 2 + (dy / ly) ** 2
        if rr > 1.0:
            k = 1.0 / math.sqrt(rr)
            dx, dy = dx * k, dy * k
        return dx, dy

    # ---------------- LBS ----------------

    def effective_rest(self, layer: _LayerSkin, blink: float) -> np.ndarray:
        """层的有效静止坐标：眼睑层施加眨眼挤压（带缓存），普通层直返 rest。"""
        if layer.blink_delta is not None:
            if layer._blink_applied != blink:
                np.multiply(layer.blink_delta, blink, out=layer.eff_rest)
                np.add(layer.rest, layer.eff_rest, out=layer.eff_rest)
                layer._blink_applied = blink
            return layer.eff_rest
        cfg = layer.blink
        if cfg is None:
            return layer.rest
        if layer._blink_applied == blink:      # 同值跳过（int 比较同一来源的浮点）
            return layer.eff_rest                              # type: ignore[return-value]
        eff, y, buf = layer.eff_rest, layer.rest[:, 1], layer._ybuf
        eff[:] = layer.rest
        # 上睑：朝 upper_scale_pivot 挤压 (upper_frac·blink)
        k = 1.0 - cfg.upper_frac * blink
        np.subtract(y, cfg.upper_piv_y, out=buf)
        np.multiply(buf, k, out=buf)
        np.add(buf, cfg.upper_piv_y, out=buf)
        np.copyto(eff[:, 1], buf, where=layer.upper_mask)
        # 下睑：朝 lower_scale_pivot 挤压 (lower_frac·blink)
        k = 1.0 - cfg.lower_frac * blink
        np.subtract(y, cfg.lower_piv_y, out=buf)
        np.multiply(buf, k, out=buf)
        np.add(buf, cfg.lower_piv_y, out=buf)
        np.copyto(eff[:, 1], buf, where=layer.lower_mask)
        layer._blink_applied = blink
        return eff

    def deform(self, layer: _LayerSkin, rest_eff: np.ndarray) -> np.ndarray:
        """单层 LBS 全向量化：``v' = Σ_k w·(M_{b_k} @ v)`` → layer.scratch。"""
        np.einsum("vk,kij,vj->vi", layer.weights, self.M[layer.bone_idx],
                  rest_eff, out=layer.scratch)                 # type: ignore[arg-type]
        return layer.scratch                                   # type: ignore[return-value]

    # ---------------- 加载（宽进严出） ----------------

    @classmethod
    def load(cls, spec_file: str, mesh_file: str,
             layers_dir: str, keyforms_file: str = "") -> RigRuntime | None:
        """解析 spec + mesh（+ 可选 view_keyforms）。

        整体不可用返回 None（调用方空渲染）；keyforms 挂接失败仅降级为
        正面单视角（attach_view_keyforms 宽进严出）。
        """
        try:
            with open(spec_file, "r", encoding="utf-8") as f:
                raw_spec = json.load(f)
            with open(mesh_file, "r", encoding="utf-8") as f:
                raw_mesh = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("蒙皮资产加载失败（%s / %s）：%s", spec_file, mesh_file, e)
            return None

        img_w, img_h = cls._resolve_image_size(raw_spec, raw_mesh)
        if img_w <= 0 or img_h <= 0:
            log.warning("蒙皮资产缺有效 image_size_px，放弃加载")
            return None

        bones = cls._parse_bones(raw_spec, img_w, img_h)
        if not bones:
            log.warning("spec 无可用骨骼，放弃加载：%s", spec_file)
            return None

        fm = raw_spec.get("face_mechanics") or {}
        blink_cfgs = cls._parse_blink_cfgs(fm, img_w, img_h)
        look_cfg = cls._parse_look_cfg(fm)

        order, parent_idx, severed = _topo_order(bones)
        if severed:
            log.warning("骨骼层级存在环/断链，已按断链为根处理（%d 根断链）",
                        severed)

        B = len(bones)
        w_rest = np.zeros((B, 3, 3), np.float32)
        a_local = np.zeros((B, 3, 3), np.float32)
        inv_w_rest = np.zeros((B, 3, 3), np.float32)
        for i in order:
            t = _mat_trans(*bones[i].joint_px)
            p = int(parent_idx[i])
            # 静止全局 = 纯平移到关节（静止姿态无旋转，不随父累积）；
            # 局部 = inv(父静止全局) @ T(J_i)——关节位置按全图世界坐标解释
            w_rest[i] = t
            a_local[i] = t if p < 0 else _inv_affine3(w_rest[p]) @ t
            inv_w_rest[i] = _inv_affine3(w_rest[i])

        spec_layers = cls._spec_layer_index(raw_spec)
        idx = {b.name: i for i, b in enumerate(bones)}
        seen_ids: set[str] = set()
        layers: list[_LayerSkin] = []
        mesh_layers = raw_mesh.get("layers")
        if not isinstance(mesh_layers, list) or not mesh_layers:
            log.warning("mesh 数据无 layers 数组，放弃加载：%s", mesh_file)
            return None
        for mi, ml in enumerate(mesh_layers):
            try:
                layer = cls._parse_layer(ml, mi, spec_layers, layers_dir,
                                         idx, blink_cfgs)
            except Exception as e:             # noqa: BLE001 —— 单层坏弃层不弃场
                log.warning("mesh 层 #%d 解析失败，弃层：%s", mi, e)
                continue
            if layer.layer_id in seen_ids:
                log.warning("mesh 层 %s 重复定义，保留首个", layer.layer_id)
                continue
            seen_ids.add(layer.layer_id)
            layers.append(layer)
        if not layers:
            log.warning("mesh 无任何可用层，放弃加载：%s", mesh_file)
            return None
        layers.sort(key=lambda l: l.z_order)

        rt = cls(bones, parent_idx, a_local, inv_w_rest, layers, look_cfg,
                 img_w, img_h)
        # 按阶段可选：源图缩到 ~15% 显示时无 mipmap 会锯齿/闪烁，且与视频片段帧
        # 观感不一致（G0 实测 12.1→3.7/255）。缺省关闭，YOUNG 保持原样。
        rt.texture_mipmaps = bool(raw_spec.get("texture_mipmaps", False))
        if keyforms_file:
            rt.attach_view_keyforms(keyforms_file)
        log.info("蒙皮核就绪：%d 骨 / %d 层 / 源图 %.0f×%.0f%s",
                 len(bones), len(layers), img_w, img_h,
                 "（含视角关键形态）" if rt._kf is not None else "")
        return rt

    # ---- 加载辅助：逐段防御，坏件降级 ----

    @staticmethod
    def _resolve_image_size(raw_spec: dict,
                            raw_mesh: dict) -> tuple[float, float]:
        for src in (raw_mesh.get("image_size_px"),
                    (raw_spec.get("skeleton") or {}).get(
                        "source_reference", {}).get("image_size_px")):
            try:
                w, h = float(src[0]), float(src[1])            # type: ignore[index]
                if math.isfinite(w) and math.isfinite(h) and w > 0 and h > 0:
                    return w, h
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        return 0.0, 0.0

    @staticmethod
    def _parse_bones(raw_spec: dict, img_w: float,
                     img_h: float) -> list[_BoneDef]:
        bones: list[_BoneDef] = []
        seen: set[str] = set()
        raw_bones = ((raw_spec.get("skeleton") or {}).get("bones") or [])
        for i, b in enumerate(raw_bones if isinstance(raw_bones, list) else []):
            try:
                if not isinstance(b, dict):
                    raise ValueError(f"非对象：{type(b).__name__}")
                name = str(b["bone_name"])
                if not name:
                    raise ValueError("空 bone_name")
                if name in seen:
                    raise ValueError("重名（保留首个）")
                jp = b["joint_pos"]
                jx, jy = float(jp[0]) * img_w, float(jp[1]) * img_h
                if not (math.isfinite(jx) and math.isfinite(jy)):
                    raise ValueError(f"joint_pos 非有限：{jp}")
                clamp = b.get("angle_clamp", [-1e9, 1e9])
                lo, hi = float(clamp[0]), float(clamp[1])
                if hi < lo:
                    lo, hi = hi, lo
                parent = b.get("parent")
                parent = str(parent) if parent else None
                bones.append(_BoneDef(name, parent, (jx, jy), (lo, hi)))
                seen.add(name)
            except (KeyError, TypeError, ValueError, IndexError) as e:
                log.warning("骨骼 #%d 解析失败，跳过：%s", i, e)
        return bones

    @staticmethod
    def _parse_blink_cfgs(fm: dict, img_w: float,
                          img_h: float) -> dict[str, _EyeBlinkCfg]:
        blink = fm.get("blink") or {}
        try:
            up_frac = float(blink.get("upper_lid_travel_fraction", 0.78))
            lo_frac = float(blink.get("lower_lid_travel_fraction", 0.22))
        except (TypeError, ValueError):
            up_frac, lo_frac = 0.78, 0.22
        cfgs: dict[str, _EyeBlinkCfg] = {}
        eyes = blink.get("eyes") or []
        for e in eyes if isinstance(eyes, list) else []:
            try:
                cfgs[str(e["eyelid"])] = _EyeBlinkCfg(
                    layer_id=str(e["eyelid"]),
                    upper_piv_y=float(e["upper_scale_pivot"][1]) * img_h,
                    lower_piv_y=float(e["lower_scale_pivot"][1]) * img_h,
                    center_y=float(e["closure_curve_center"][1]) * img_h,
                    upper_frac=up_frac,
                    lower_frac=lo_frac,
                )
            except (KeyError, TypeError, ValueError, IndexError) as ex:
                log.warning("blink 眼配置解析失败，跳过一只眼：%s", ex)
        return cfgs

    @staticmethod
    def _parse_look_cfg(fm: dict) -> _LookAtCfg:
        look = fm.get("look_at") or {}
        axis = look.get("axis_limits_px", [10.0, 7.0])
        try:
            lx, ly = float(axis[0]), float(axis[1])
            if not (math.isfinite(lx) and math.isfinite(ly)
                    and lx > 0 and ly > 0):
                raise ValueError(f"axis_limits_px 非法：{axis}")
        except (TypeError, ValueError, IndexError):
            lx, ly = 10.0, 7.0
        pupils: list[str] = []
        eyes = look.get("eyes") or []
        for e in eyes if isinstance(eyes, list) else []:
            try:
                bone = str(e["bone"])
                if bone:
                    pupils.append(bone)
            except (TypeError, ValueError, KeyError):
                continue
        if not pupils:
            pupils = ["pupil_l", "pupil_r"]
        return _LookAtCfg(axis_px=(lx, ly), pupil_bones=tuple(pupils))

    @staticmethod
    def _spec_layer_index(raw_spec: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        layers = raw_spec.get("layers") or []
        for l in layers if isinstance(layers, list) else []:
            if isinstance(l, dict) and l.get("id"):
                out[str(l["id"])] = l
        return out

    @classmethod
    def _parse_layer(cls, ml: dict, mesh_index: int, spec_layers: dict[str, dict],
                     layers_dir: str, bone_idx: dict[str, int],
                     blink_cfgs: dict[str, _EyeBlinkCfg]) -> _LayerSkin:
        """单层解析（抛异常 = 弃层）。加权/索引/纹理逐项校验后建 NumPy 缓冲。"""
        if not isinstance(ml, dict):
            raise ValueError(f"非对象：{type(ml).__name__}")
        layer_id = str(ml.get("id") or "")
        if not layer_id:
            raise ValueError("缺 id")

        verts = np.asarray(ml["vertices"], np.float64)
        uvs = np.asarray(ml["uvs"], np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError(f"vertices 形状非法：{verts.shape}")
        if uvs.shape != verts.shape:
            raise ValueError(f"uvs 形状 {uvs.shape} 与 vertices {verts.shape} 不符")
        if not (np.isfinite(verts).all() and np.isfinite(uvs).all()):
            raise ValueError("vertices/uvs 含非有限值")
        vcount = verts.shape[0]
        if vcount > 65535:
            raise ValueError(f"顶点数 {vcount} 超 uint16 索引界")

        tri = np.asarray(ml["triangles"], np.int64)
        if tri.ndim != 1 or tri.size % 3 != 0:
            raise ValueError(f"triangles 非平铺三角序列：shape={tri.shape}")
        if tri.size and (tri.min() < 0 or tri.max() >= vcount):
            raise ValueError("triangles 索引越界")
        triangles = tri.astype(np.uint16)

        # ---- 权重：未知骨剔除 → 负值截零 → 归一化；空则回退绑定骨刚体 ----
        spec_l = spec_layers.get(layer_id) or {}
        bind_name = str(ml.get("bind_bone") or spec_l.get("bind_bone") or "")
        wb_raw = ml.get("weight_bones")
        wv_raw = ml.get("weight_values")
        if not isinstance(wb_raw, list) or not isinstance(wv_raw, list) \
                or len(wb_raw) != vcount or len(wv_raw) != vcount:
            raise ValueError("weight_bones/weight_values 须为每顶点平行数组")
        rows: list[tuple[list[int], list[float]]] = []
        dropped_unknown: set[str] = set()
        for names, vals in zip(wb_raw, wv_raw):
            if not isinstance(names, list) or not isinstance(vals, list) \
                    or len(names) != len(vals) or not names:
                raise ValueError("权重行长度不齐或为空")
            sel: list[int] = []
            wsel: list[float] = []
            for nm, w in zip(names, vals):
                try:
                    w = float(w)
                except (TypeError, ValueError):
                    continue
                nm = str(nm)
                if nm not in bone_idx or not math.isfinite(w) or w <= 0.0:
                    if nm in bone_idx:
                        continue
                    dropped_unknown.add(nm)
                    continue
                sel.append(bone_idx[nm])
                wsel.append(w)
            total = math.fsum(wsel)
            if total <= 1e-9:
                sel, wsel = [], []
            else:
                wsel = [w / total for w in wsel]
            rows.append((sel, wsel))
        if dropped_unknown:
            log.warning("层 %s 权重引用未知骨 %s，已剔除并归一化",
                        layer_id, sorted(dropped_unknown))
        if not bind_name or bind_name not in bone_idx:
            bind_name = next(iter(bone_idx))     # 首骨兜底（根骨）
            log.warning("层 %s 绑定骨缺失/未知，回退 %s", layer_id, bind_name)
        # Columns identify bones, not per-vertex influence ranks. Different
        # vertices may list the same bones in a different order.
        selected = sorted({b for sel, _ in rows for b in sel}
                          | {bone_idx[bind_name]})
        bone_sel = np.asarray(selected, np.int32)
        columns = {b: i for i, b in enumerate(selected)}
        weights = np.zeros((vcount, len(selected)), np.float32)
        for r, (sel, wsel) in enumerate(rows):
            if sel:
                for b, weight in zip(sel, wsel):
                    weights[r, columns[b]] += weight
            else:
                weights[r, columns[bone_idx[bind_name]]] = 1.0

        texture = str(ml.get("texture") or f"{layer_id}.png")
        texture_path = os.path.normpath(os.path.join(layers_dir, texture))
        try:
            z_order = int(ml.get(
                "z_order", spec_l.get("z_order", mesh_index)))
        except (TypeError, ValueError):
            z_order = mesh_index

        blink = blink_cfgs.get(layer_id)
        texture_size = tuple(float(v) for v in ml.get("texture_size_px", [1, 1]))
        if len(texture_size) != 2 or not all(math.isfinite(v) and v > 0 for v in texture_size):
            raise ValueError("invalid texture_size_px")
        delta = ml.get("blink_delta")
        if delta is not None:
            delta = np.asarray(delta, np.float32)
            if delta.shape != (vcount, 2) or not np.isfinite(delta).all():
                raise ValueError("blink_delta must match vertices")
            delta = np.column_stack((delta, np.zeros(vcount, np.float32)))
        layer = _LayerSkin(
            layer_id=layer_id,
            z_order=z_order,
            texture_path=texture_path,
            bind_bone=bind_name,
            rest=np.concatenate(
                [verts, np.ones((vcount, 1))], axis=1).astype(np.float32),
            uv=uvs.astype(np.float32),
            triangles=triangles,
            bone_idx=bone_sel,
            weights=weights,
            blink=blink,
            blink_delta=delta,
            gaze_uv=bool(ml.get("gaze_uv", False)),
            texture_size=texture_size,
            scratch=np.zeros((vcount, 3), np.float32),
            eff_rest=np.zeros((vcount, 3), np.float32) if blink or delta is not None else None,
            upper_mask=(None if blink is None else
                        verts[:, 1] < blink.center_y),
            lower_mask=(None if blink is None else
                        verts[:, 1] >= blink.center_y),
            _ybuf=(None if blink is None else
                   np.zeros(vcount, np.float32)),
        )
        return layer


def _topo_order(bones: list[_BoneDef]) -> tuple[list[int], np.ndarray, int]:
    """Kahn 拓扑排序（父先于子）+ 断环：环上骨按索引序斩断父链成为根。

    返回 (order, parent_idx, severed_count)。未知父在此时已按 None 处理。
    """
    idx = {b.name: i for i, b in enumerate(bones)}
    parent = np.full(len(bones), -1, np.int32)
    children: list[list[int]] = [[] for _ in bones]
    indeg = np.zeros(len(bones), np.int32)
    for i, b in enumerate(bones):
        if b.parent is not None and b.parent in idx and idx[b.parent] != i:
            parent[i] = idx[b.parent]
            children[idx[b.parent]].append(i)
            indeg[i] += 1
        elif b.parent is not None and b.parent not in idx:
            log.warning("骨 %s 的父 %s 不存在，按根处理", b.name, b.parent)

    order: list[int] = [i for i in range(len(bones)) if indeg[i] == 0]
    head = 0
    while head < len(order):
        i = order[head]
        head += 1
        for c in children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                order.append(c)
    # 剩余 = 环：逐个斩断父链成为根（不追求最小反馈弧集，够用且可预测）
    severed = 0
    for i in range(len(bones)):
        if indeg[i] > 0:
            log.warning("骨 %s 处于层级环，斩断父链 %s 按根处理",
                        bones[i].name, bones[i].parent)
            parent[i] = -1
            severed += 1
            order.append(i)
    return order, parent, severed


# ============================ Qt 场景图层 ============================


@dataclass
class _LayerSG:
    """一层的场景图簿记：节点/几何/纹理 + 零拷贝顶点视图。

    保持 Python 包装器存活；节点拥有几何和材质，纹理由 item 的缓存持有。
    """

    node: QSGGeometryNode
    geometry: QSGGeometry
    material: QSGTextureMaterial | None
    texture: QSGTexture
    verts: np.ndarray             # (V,4) f32 —— ctypes 直指 C++ 顶点缓冲


class SkinnedMeshItem(QQuickItem):
    """高性能 2D 骨架蒙皮网格项（QML 注册名 ``PetRig 1.0 / SkinnedMeshItem``）。

    属性：``specFile`` / ``meshDataFile`` / ``layersDir``（三者齐备后惰性加载，
    变更即整体重载）；``lookAtX`` / ``lookAtY``（归一化注视偏移）；
    ``blinkProgress``（0=睁开 1=闭合）。

    槽：``setBonePose(boneName, angleDeg, tx=0, ty=0)``（局部旋转角按 spec
    ``angle_clamp`` 钳制；平移单位 = 源图像素）、``setBlink(progress)``、
    ``setLookAt(targetX, targetY)``。

    线程模型：属性/槽在 GUI 线程写，``updatePaintNode`` 在渲染线程的同步阶段
    读（GUI 阻塞期）——Qt 场景图自绘项的标准做法，无额外锁。
    """

    # ---- QML 属性通知 ----
    specFileChanged = Signal(str)
    meshDataFileChanged = Signal(str)
    layersDirChanged = Signal(str)
    lookAtXChanged = Signal(float)
    lookAtYChanged = Signal(float)
    blinkProgressChanged = Signal(float)
    viewYawChanged = Signal(float)
    readyChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.setFlag(_ITEM_HAS_CONTENTS, True)
        # ---- 输入路径（三者齐备后才尝试加载；QML url 属性会以 file:/// 传入）----
        self._spec_file: str = ""
        self._mesh_file: str = ""
        self._layers_dir: str = ""
        # ---- 姿态输入（GUI 线程）----
        self._look_x: float = 0.0
        self._look_y: float = 0.0
        self._blink: float = 0.0
        self._view_yaw: float = 0.0
        self._patch_hidden: set[str] = set()
        self._patch_step: dict[str, int] = {}
        self._patch_tex_cache: dict[tuple, QSGTexture] = {}
        self._pose_angle: dict[str, float] = {}
        self._pose_tx: dict[str, float] = {}
        self._pose_ty: dict[str, float] = {}
        self._pose_ang_buf: np.ndarray | None = None   # 加载后分配 (B,)
        self._pose_tx_buf: np.ndarray | None = None
        self._pose_ty_buf: np.ndarray | None = None
        self._warned_bones: set[str] = set()
        # ---- 加载态 ----
        self._rt: RigRuntime | None = None
        self._load_failed: bool = False
        self._ready = False
        self._assets_dirty = True
        # ---- 场景图态（仅渲染同步阶段触碰）----
        self._root: QSGNode | None = None
        self._layer_sgs: dict[str, _LayerSG] = {}
        self._tex_cache: dict[str, QSGTexture] = {}           # 自建 QSGTexture 登记表（渲染线程显式释放，非跨重建复用）
        # ---- 帧簿记 ----
        self._pose_dirty: bool = True
        self._view_key: tuple[float, float] = (0.0, 0.0)
        self._render_error_logged: bool = False
        try:
            self.windowChanged.connect(self._on_window_changed)
        except Exception:                            # pragma: no cover
            log.warning("SkinnedMeshItem 无法监听 windowChanged")

    ready = Property(bool, lambda self: self._ready, notify=readyChanged)

    def _set_ready(self, value: bool) -> None:
        if self._ready != value:
            self._ready = value
            self.readyChanged.emit()

    def prepare(self) -> bool:
        """Validate before hiding the fallback image; no scene graph work here."""
        try:
            return self._prepare()
        except Exception:
            log.warning("Invalid skinned asset bundle; using figure fallback", exc_info=True)
            self._rt = None
            self._load_failed = True
            self._set_ready(False)
            self._invalidate_pose()
            return False

    def _prepare(self) -> bool:
        self._rt = RigRuntime.load(self._spec_file, self._mesh_file, self._layers_dir)
        # 完整性预检只做 stat（零解码）：旧版 all(QImage(...).isNull()) 对 20 层
        # 各做一次完整 PNG 解码后丢弃，启动白白 churn ~124MB（真正建纹理的是
        # _build_layer_node）。坏图/非空坏文件在 _build_layer_node 的 img.isNull()
        # 处弃层不弃场；「全有或全无」由下方 spec layers 数量门兜底。
        complete = self._rt is not None and all(
            os.path.exists(layer.texture_path)
            and os.path.getsize(layer.texture_path) > 0
            for layer in self._rt.layers)
        if complete:
            with open(self._spec_file, encoding="utf-8") as f:
                complete = len(self._rt.layers) == len(json.load(f)["layers"])
        self._load_failed = not complete
        self._assets_dirty = True
        if complete:
            n = len(self._rt.bones)
            self._pose_ang_buf = np.zeros(n, np.float32)
            self._pose_tx_buf = np.zeros(n, np.float32)
            self._pose_ty_buf = np.zeros(n, np.float32)
        self._set_ready(complete)
        self._invalidate_pose()
        return complete

    # ---------------- QML 属性 ----------------

    def _get_spec_file(self) -> str:
        return self._spec_file

    def _set_spec_file(self, v: str) -> None:
        self._set_asset_path("_spec_file", self.specFileChanged, v)

    def _get_mesh_file(self) -> str:
        return self._mesh_file

    def _set_mesh_file(self, v: str) -> None:
        self._set_asset_path("_mesh_file", self.meshDataFileChanged, v)

    def _get_layers_dir(self) -> str:
        return self._layers_dir

    def _set_layers_dir(self, v: str) -> None:
        self._set_asset_path("_layers_dir", self.layersDirChanged, v)

    specFile = Property(str, _get_spec_file, _set_spec_file,
                        notify=specFileChanged)
    meshDataFile = Property(str, _get_mesh_file, _set_mesh_file,
                            notify=meshDataFileChanged)
    layersDir = Property(str, _get_layers_dir, _set_layers_dir,
                         notify=layersDirChanged)

    def _set_asset_path(self, attr: str, changed: Signal, v: str) -> None:
        v = _normalize_path(v)
        # getattr 缺省：QML 初始绑定可能在 __init__ 完成前写属性（PySide6
        # 元调用时序），类实例字段未就绪时按空路径处理不炸
        if v == getattr(self, attr, ""):
            return
        setattr(self, attr, v)
        changed.emit(v)
        # 资产路径变更 = 整体重载：丢弃数学核与场景图，静默等待三者重新齐备
        self._rt = None
        self._load_failed = False
        self._pose_ang_buf = None
        self._pose_tx_buf = None
        self._pose_ty_buf = None
        self._assets_dirty = True
        self._set_ready(False)
        self._invalidate_pose()

    def _get_look_x(self) -> float:
        return self._look_x

    def _set_look_x(self, v: float) -> None:
        self._set_look("x", v)

    def _get_look_y(self) -> float:
        return self._look_y

    def _set_look_y(self, v: float) -> None:
        self._set_look("y", v)

    lookAtX = Property(float, _get_look_x, _set_look_x, notify=lookAtXChanged)
    lookAtY = Property(float, _get_look_y, _set_look_y, notify=lookAtYChanged)

    def _get_blink(self) -> float:
        return self._blink

    def _set_blink(self, v: float) -> None:
        self._apply_blink(v)

    blinkProgress = Property(float, _get_blink, _set_blink,
                             notify=blinkProgressChanged)

    def _get_view_yaw(self) -> float:
        return self._view_yaw

    def _set_view_yaw(self, v: float) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v) or v == self._view_yaw:
            return
        self._view_yaw = v
        self.viewYawChanged.emit(v)
        rt = self._rt
        if rt is not None and rt.set_view_yaw(v):
            self._apply_patch_alphas()
            self._invalidate_pose()

    viewYaw = Property(float, _get_view_yaw, _set_view_yaw,
                       notify=viewYawChanged)

    def _apply_patch_alphas(self) -> None:
        """补片遮挡透明度（§4.4）：α=0 折叠隐藏；部分透明按 16 级量化换纹理。

        TexturedPoint2D 顶点格式无颜色/alpha 通道，逐帧顶点透明不可行——
        以量化步进的纹理 alpha 缩放缓存实现平滑阶跃（补片数量少、仅在
        视角变化期间换档），杜绝 Z-Order 瞬时闪现。
        """
        rt = self._rt
        if rt is None or not getattr(rt, "_kf", None) or not rt._kf.patches:
            return
        win = self.window()
        if win is None:
            return
        alphas = rt.patch_alphas(self._view_yaw)
        for patch in rt._kf.patches:
            lid = patch["base_layer"]
            sg = self._layer_sgs.get(lid)
            if sg is None:
                continue
            alpha = float(alphas.get(patch["patch_id"], 0.0))
            if alpha <= 0.03:
                self._patch_hidden.add(lid)
                continue
            self._patch_hidden.discard(lid)
            step = int(round(alpha * 16.0))
            if step == self._patch_step.get(lid, 16):
                continue
            self._patch_step[lid] = step
            tex = self._patch_texture(sg, step, win)
            sg.material.setTexture(tex)
            sg.node.markDirty(QSGNode.DirtyState.DirtyMaterial)

    def _mipmaps(self) -> bool:
        return bool(self._rt is not None and self._rt.texture_mipmaps)

    def _create_texture(self, win, img: QImage) -> QSGTexture:
        if self._mipmaps():
            return win.createTextureFromImage(
                img, QQuickWindow.CreateTextureOption.TextureHasMipmaps)
        return win.createTextureFromImage(img)

    def _patch_texture(self, sg: "_LayerSG", step: int, win) -> QSGTexture:
        key = (id(sg), step)
        tex = self._patch_tex_cache.get(key)
        if tex is not None:
            return tex
        src = QImage(sg.texture_path)
        # 纹理 alpha 整体缩放（step/16）；ARGB32 逐像素乘法
        img = src.convertToFormat(QImage.Format_ARGB32)
        if step < 16:
            w, h = img.width(), img.height()
            arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 1)[:h, :w * 4].copy()
            arr = arr.reshape(h, w, 4)
            # ARGB32 little-endian 内存序为 B,G,R,A
            arr[:, :, 3] = (arr[:, :, 3].astype(np.uint32) * step // 16).astype(np.uint8)
            out = QImage(arr.data, w, h, w * 4, QImage.Format_ARGB32)
            out = out.copy()          # arr 生命周期与 QImage 解耦
            img = out
        tex = self._create_texture(win, img)
        self._patch_tex_cache[key] = tex
        if len(self._patch_tex_cache) > 256:     # LRU 粗截断（补片×步数有界）
            self._patch_tex_cache.clear()
            self._patch_tex_cache[key] = tex
        return tex

    def _set_look(self, axis: str, v: float) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        v = max(-1.0, min(1.0, v))
        if axis == "x":
            if v == self._look_x:
                return
            self._look_x = v
            self.lookAtXChanged.emit(v)
        else:
            if v == self._look_y:
                return
            self._look_y = v
            self.lookAtYChanged.emit(v)
        self._invalidate_pose()

    def _apply_blink(self, progress: float) -> None:
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            return
        if not math.isfinite(progress):
            return
        progress = max(0.0, min(1.0, progress))
        if progress == self._blink:
            return
        self._blink = progress
        self.blinkProgressChanged.emit(progress)
        self._invalidate_pose()

    # ---------------- 槽 ----------------

    @Slot(str, float, float, float)
    def setBonePose(self, boneName: str, angleDeg: float, tx: float = 0.0,
                    ty: float = 0.0) -> None:
        """设置单骨局部姿态（度 + 源图像素平移）。未知骨忽略并一次性告警。"""
        vals = (angleDeg, tx, ty)
        try:
            vals = tuple(float(v) for v in vals)     # type: ignore[assignment]
        except (TypeError, ValueError):
            log.warning("setBonePose(%s) 参数非数值，忽略", boneName)
            return
        if not all(map(math.isfinite, vals)):
            log.warning("setBonePose(%s) 含非有限值，忽略：%s", boneName, vals)
            return
        rt = self._rt
        if rt is not None and boneName not in rt.bone_index:
            if boneName not in self._warned_bones:
                self._warned_bones.add(boneName)
                log.warning("setBonePose 未知骨骼 %s，忽略（后续静默）", boneName)
            return
        self._pose_angle[boneName] = vals[0]
        self._pose_tx[boneName] = vals[1]
        self._pose_ty[boneName] = vals[2]
        self._invalidate_pose()

    @Slot(float)
    def setBlink(self, progress: float) -> None:
        self._apply_blink(progress)

    @Slot(float, float)
    def setLookAt(self, targetX: float, targetY: float) -> None:
        self._set_look("x", targetX)
        self._set_look("y", targetY)

    # ---------------- 渲染 ----------------

    def updatePaintNode(self, oldNode: QSGNode | None,
                        nodeData: QQuickItem.UpdatePaintNodeData
                        ) -> QSGNode | None:
        if self._load_failed:
            return oldNode
        try:
            return self._update_paint_node(oldNode)
        except Exception:                            # noqa: BLE001 —— 渲染期兜底
            if not self._render_error_logged:
                self._render_error_logged = True
                log.exception("SkinnedMeshItem 渲染异常，本帧起静默跳帧")
            self._set_ready(False)
            return oldNode

    def _update_paint_node(self, oldNode: QSGNode | None) -> QSGNode | None:
        win = self.window()
        if win is None:
            return oldNode
        from PySide6.QtQuick import QSGRendererInterface
        if win.rendererInterface().graphicsApi() == QSGRendererInterface.GraphicsApi.Software:
            self._set_ready(False)
            return oldNode
        if self._rt is None and not self.prepare():
            return oldNode
        w, h = float(self.width()), float(self.height())
        if self._assets_dirty or oldNode is None:
            self._build_scene_graph(oldNode, win)
            self._assets_dirty = False
        if self._pose_dirty or (w, h) != self._view_key:
            self._render_frame(w, h)
        return self._root

    # ---- 场景图构建（oldNode None/失效时）----

    def _build_scene_graph(self, oldNode: QSGNode | None,
                           win: QQuickWindow) -> None:
        """Build on the render thread, retaining one ordinary grouping root."""
        rt = self._rt
        assert rt is not None
        if oldNode is not None:
            self._detach_children(oldNode)
        self._layer_sgs.clear()
        self._release_textures()       # 渲染线程显式释放旧纹理（勿只 clear 丢引用）
        self._root = oldNode if oldNode is not None else QSGNode()
        for layer in rt.layers:
            try:
                sg = self._build_layer_node(layer, win)
            except Exception as e:         # noqa: BLE001 —— 单层坏弃层不弃场
                log.warning("层 %s 场景图节点构建失败，弃层：%s",
                            layer.layer_id, e)
                continue
            self._layer_sgs[layer.layer_id] = sg
            self._root.appendChildNode(sg.node)
        self._set_ready(len(self._layer_sgs) == len(rt.layers))
        self._pose_dirty = True            # 重建后首帧必须全量写顶点
        self._apply_patch_alphas()

    def _build_layer_node(self, layer: _LayerSkin,
                          win: QQuickWindow) -> _LayerSG:
        # 每层每次重建都新建纹理并登记；_tex_cache 仅作所有权登记用于显式释放，
        # 不做跨重建复用（_build_scene_graph 每次先 _release_textures 清空）。
        # mipmap（spec texture_mipmaps，G0 决策7）：经 _create_texture 统一创建。
        img = QImage(layer.texture_path)
        if img.isNull():
            raise ValueError(f"纹理缺失/不可读：{layer.texture_path}")
        texture = self._create_texture(win, img)
        self._tex_cache[layer.layer_id] = texture
        material: QSGTextureMaterial | None = None
        vcount = layer.rest.shape[0]
        icount = int(layer.triangles.size)

        material = QSGTextureMaterial()
        material.setTexture(texture)
        material.setFiltering(QSGTexture.Filtering.Linear)
        if self._mipmaps():
            material.setMipmapFiltering(QSGTexture.Filtering.Linear)   # 三线性
        material.setFlag(QSGMaterial.Flag.Blending, True)
        node = QSGGeometryNode()
        node.setMaterial(material)
        node.setFlag(QSGNode.Flag.OwnsMaterial, True)
        geometry = QSGGeometry(
            _TEXTURED_ATTRIBUTES, vcount, icount)
        geometry.setVertexDataPattern(QSGGeometry.DataPattern.DynamicPattern)
        node.setGeometry(geometry)
        node.setFlag(QSGNode.Flag.OwnsGeometry, True)
        try:
            geometry.setDrawingMode(QSGGeometry.DrawTriangles)
        except AttributeError:              # pragma: no cover —— Qt < 6.6
            geometry.setDrawMode(QSGGeometry.DrawTriangles)  # type: ignore[attr-defined]

        # ---- 零拷贝视图：ctypes 直指 C++ 缓冲（UV/索引加载期写一次）----
        c_verts = (ctypes.c_float * (4 * vcount)).from_address(
            int(geometry.vertexData()))
        verts = np.ctypeslib.as_array(c_verts).reshape(vcount, 4)
        verts[:, 2:4] = layer.uv
        if icount:
            c_idx = (ctypes.c_ushort * icount).from_address(
                int(geometry.indexData()))
            c_idx[:] = layer.triangles.tolist()
            geometry.markIndexDataDirty()
        geometry.markVertexDataDirty()
        node.markDirty(QSGNode.DirtyState.DirtyGeometry)
        return _LayerSG(node=node, geometry=geometry, material=material,
                        texture=texture, verts=verts)

    # ---- 每帧变形（热路径：FK + LBS + 视图变换，全部预分配缓冲）----

    def _pose_geometry(self, w: float, h: float) -> tuple[float, float, float] | None:
        """姿态三件套：全骨数组回填 + look-at 限幅 + FK。返回视图参数
        (fit, off_x, off_y)；item 无有效尺寸返回 None。"""
        rt = self._rt
        assert rt is not None and self._pose_ang_buf is not None
        # 1) 手工姿态 → 全骨数组（≤47 项字典回填）
        ang, tx, ty = self._pose_ang_buf, self._pose_tx_buf, self._pose_ty_buf
        ang.fill(0.0)
        tx.fill(0.0)
        ty.fill(0.0)
        index = rt.bone_index
        for name, v in self._pose_angle.items():
            i = index.get(name)
            if i is not None:
                ang[i] = v
        for name, v in self._pose_tx.items():
            i = index.get(name)
            if i is not None:
                tx[i] = v
        for name, v in self._pose_ty.items():
            i = index.get(name)
            if i is not None:
                ty[i] = v

        # 2) look-at 专用通道：椭圆限幅后经 FK 覆盖瞳骨平移
        look_dx, look_dy = rt.look_offset(self._look_x, self._look_y)
        look_dx *= 1.0 - self._blink
        look_dy *= 1.0 - self._blink

        # 3) FK → M_b = T_b·T_rest⁻¹
        rt.skinning_matrices(ang, tx, ty, look_dx, look_dy)

        # 4) 视图变换（KeepAspectRatio + 居中，同 rig_scene.qml 语义）
        fit = min(w / rt.img_w, h / rt.img_h)
        self._view_key = (w, h)
        self._pose_dirty = False
        if not (fit > 0.0):
            return None                    # 0 尺寸 item：无可写像素，等待尺寸
        return fit, (w - rt.img_w * fit) * 0.5, (h - rt.img_h * fit) * 0.5

    def _deform_into(self, layer: _LayerSkin, sg: _LayerSG, fit: float,
                     off_x: float, off_y: float) -> None:
        """单层 LBS → 视图变换 → 顶点缓冲切片直写（就地、零分配）。"""
        rt = self._rt
        assert rt is not None
        if layer.layer_id in self._patch_hidden:
            sg.verts[:, :2] = sg.verts[0, :2]    # α=0 补片折叠隐藏
            sg.geometry.markVertexDataDirty()
            sg.node.markDirty(QSGNode.DirtyState.DirtyGeometry)
            return
        vcount = layer.rest.shape[0]
        scratch = rt.deform(layer, rt.effective_rest(layer, self._blink))
        bx, by = rt.buf_x[:vcount], rt.buf_y[:vcount]
        np.multiply(scratch[:, 0], fit, out=bx)
        np.add(bx, off_x, out=bx)
        np.multiply(scratch[:, 1], fit, out=by)
        np.add(by, off_y, out=by)
        sg.verts[:, 0] = bx                # arr[:, 0:2] = deformed_xy 的就地写法
        sg.verts[:, 1] = by
        if layer.gaze_uv:
            dx, dy = rt.look_offset(self._look_x, self._look_y)
            sg.verts[:, 2] = layer.uv[:, 0] - dx * (1 - self._blink) / layer.texture_size[0]
            sg.verts[:, 3] = layer.uv[:, 1] - dy * (1 - self._blink) / layer.texture_size[1]
            if self._blink >= 0.999:
                sg.verts[:, :2] = sg.verts[0, :2]  # no iris sliver on a closed eye
        sg.geometry.markVertexDataDirty()
        sg.node.markDirty(QSGNode.DirtyState.DirtyGeometry)

    def _render_frame(self, w: float, h: float) -> None:
        """主路径：持久节点树上的顶点就地刷新。"""
        rt = self._rt
        assert rt is not None
        view = self._pose_geometry(w, h)
        if view is None:
            return
        fit, off_x, off_y = view
        for layer in rt.layers:
            sg = self._layer_sgs.get(layer.layer_id)
            if sg is not None:
                self._deform_into(layer, sg, fit, off_x, off_y)

    # ---- 生命周期 / 失效 ----

    def _invalidate_pose(self) -> None:
        if not self._pose_dirty:
            self._pose_dirty = True
        self.update()

    def _on_window_changed(self, win: QQuickWindow | None) -> None:
        self._assets_dirty = True
        if win is not None:
            from PySide6.QtCore import Qt
            win.sceneGraphInvalidated.connect(
                self._on_sg_invalidated, Qt.ConnectionType.DirectConnection)
            self.update()

    def _on_sg_invalidated(self) -> None:
        # Qt owns the old node tree. Never mutate it from a GUI timer.
        self._layer_sgs.clear()
        # 场景图失效时渲染上下文(RHI)正被拆除，GPU 纹理随之释放；此处只丢
        # Python 引用即可，不可 shiboken6.delete 悬空的 QSGTexture。
        self._tex_cache.clear()
        self._patch_tex_cache.clear()
        self._patch_step.clear()
        self._root = None
        self._assets_dirty = True
        self._pose_dirty = True

    @staticmethod
    def _detach_children(root: QSGNode) -> None:
        import shiboken6
        while root.childCount() > 0:
            child = root.firstChild()
            root.removeChildNode(child)
            shiboken6.delete(child)  # reload runs only in updatePaintNode

    def _release_textures(self) -> None:
        """渲染线程上显式释放自建 QSGTexture（调用前提：节点树已 detach）。

        QQuickWindow.createTextureFromImage 返回的 QSGTexture 不由场景图托管，
        须调用者显式删除；若只 clear() 丢 Python 引用，析构会落到 GUI 线程 GC，
        跨线程释放底层 QRhiTexture 属未定义行为且可能残留 VRAM。本方法只在
        _build_scene_graph（updatePaintNode 渲染线程）调用；场景图失效时
        上下文已拆、GPU 资源已释放，故 _on_sg_invalidated 只 clear 不 delete。
        补片缓存（_patch_tex_cache，视角关键形态 16 级量化纹理）同为自建
        QSGTexture，随主缓存一并显式释放；_patch_step 随之归零。
        """
        import shiboken6
        for cache in (self._tex_cache, self._patch_tex_cache):
            for texture in cache.values():
                try:
                    shiboken6.delete(texture)
                except Exception:  # noqa: BLE001 —— 释放失败不阻断重建
                    log.warning("QSGTexture 释放失败（忽略）", exc_info=True)
            cache.clear()
        self._patch_step.clear()


def _normalize_path(v: object) -> str:
    """QML ``url`` 属性会以 ``file:///`` 形式注入字符串：转本地路径。"""
    if not isinstance(v, str) or not v:
        return ""
    if v.startswith("file:///") or v.startswith("file:"):
        local = QUrl(v).toLocalFile()
        if local:
            return local
    return v


_QML_REGISTERED = False


def register_qml_type(uri: str = "PetRig", major: int = 1,
                      minor: int = 0) -> bool:
    """把 ``SkinnedMeshItem`` 注册为 QML 类型（幂等；import 本模块即已调用）。"""
    global _QML_REGISTERED
    if _QML_REGISTERED:
        return True
    try:
        from PySide6.QtQml import qmlRegisterType
        qmlRegisterType(SkinnedMeshItem, uri, major, minor, "SkinnedMeshItem")
        _QML_REGISTERED = True
    except Exception as e:                # pragma: no cover —— 环境缺件
        log.warning("SkinnedMeshItem QML 注册失败（%s）", e)
    return _QML_REGISTERED


register_qml_type()
