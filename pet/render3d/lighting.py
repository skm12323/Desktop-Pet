"""光照分级 L0–L3（纯逻辑，零 Qt）——契约 LightWeatherState → 渲染 uniforms。

分级（D03）：
  L0 unlit      —— 无光照（剪影/占位）
  L1 toon 明暗  —— 平行光方向/色温/强度 + 环境色（≈免费，首发档）
  L2 + 自阴影   —— L1 基础上开 shadow map 标志（scene_host 据此切材质开关）
  L3 + 湿身/雨  —— L2 基础上 wetness 进材质（roughness/高光调整）

太阳方位→方向向量约定：场景 +x 右、+y 上、+z 面朝（与契约 pos_px 一致）；
高度角 0=地平线、90=头顶。色温→RGB 用经验近似（Tanner Helland 拟合，可单测）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pet.scene_contract import LightWeatherState

LEVEL_MIN, LEVEL_MAX = 0, 3


def kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    """色温(K) → 线性 RGB 0-1（Tanner Helland 拟合，1500-12000 有效域）。"""
    k = min(max(k, 1500.0), 12000.0) / 100.0
    if k <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(k) - 161.1195681661
    else:
        r = 329.698727446 * ((k - 60) ** -0.1332047592)
        g = 288.1221695283 * ((k - 60) ** -0.0755148492)
    if k >= 66:
        b = 255.0
    elif k <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(k - 10) - 305.0447927307
    f = lambda v: min(max(v / 255.0, 0.0), 1.0)
    return (f(r), f(g), f(b))


def sun_direction(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """太阳方位(0-360, 0=+z 北面朝光、90=+x)与高度角 → 指向太阳的单位向量。"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az))


@dataclass(frozen=True)
class LightUniforms:
    """一次计算好的渲染 uniform（scene3d.qml 直接消费的语义）。"""

    level: int
    dir_x: float
    dir_y: float
    dir_z: float
    color_r: float
    color_g: float
    color_b: float
    intensity: float          # L0 恒 0；云量调制
    ambient_r: float
    ambient_g: float
    ambient_b: float
    shadow_enabled: bool      # L≥2
    wetness: float            # L3 消费，其余恒 0

    def as_qml(self) -> dict:
        """扁平 dict（QML 属性注入用；tuple/QObject 不便跨界，全走标量）。"""
        return {
            "lightLevel": self.level,
            "lightDirX": round(self.dir_x, 4), "lightDirY": round(self.dir_y, 4),
            "lightDirZ": round(self.dir_z, 4),
            "lightColorR": round(self.color_r, 4), "lightColorG": round(self.color_g, 4),
            "lightColorB": round(self.color_b, 4),
            "lightIntensity": round(self.intensity, 4),
            "ambientR": round(self.ambient_r, 4), "ambientG": round(self.ambient_g, 4),
            "ambientB": round(self.ambient_b, 4),
            "shadowEnabled": self.shadow_enabled,
            "wetness": round(self.wetness, 4),
        }


def compute(state: LightWeatherState, level: int) -> LightUniforms:
    if not LEVEL_MIN <= level <= LEVEL_MAX:
        raise ValueError(f"level={level} 越界 [{LEVEL_MIN}, {LEVEL_MAX}]")
    if level == 0:                     # unlit：无任何直射（剪影/占位档）
        return LightUniforms(level=0, dir_x=0.0, dir_y=0.0, dir_z=0.0,
                             color_r=0.0, color_g=0.0, color_b=0.0,
                             intensity=0.0, ambient_r=0.0, ambient_g=0.0,
                             ambient_b=0.0, shadow_enabled=False, wetness=0.0)
    dx, dy, dz = sun_direction(state.sun_azimuth_deg, state.sun_elevation_deg)
    r, g, b = kelvin_to_rgb(state.color_temp_k)
    daylight = min(max(state.sun_elevation_deg / 90.0, 0.0), 1.0)
    intensity = state.sun_intensity * (1.0 - state.cloud_cover * 0.7) * daylight
    ambient = 0.25 + 0.45 * state.cloud_cover + 0.25 * (1.0 - daylight)   # 阴天/夜里环境占比升
    return LightUniforms(
        level=level,
        dir_x=round(dx, 4), dir_y=round(dy, 4), dir_z=round(dz, 4),
        color_r=r, color_g=g, color_b=b,
        intensity=round(intensity, 4),
        ambient_r=round(ambient * (0.9 + 0.1 * b), 4),
        ambient_g=round(ambient * (0.9 + 0.1 * g), 4),
        ambient_b=round(ambient, 4),
        shadow_enabled=level >= 2,
        wetness=state.wetness if level >= 3 else 0.0,
    )
