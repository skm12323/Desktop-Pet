"""表情通道（纯逻辑，零 Qt）——契约 ExpressionState → VRM 预设 morph 权重。

映射来源：sidecar `expression_map.json`（建模期产出，调研-建模页管线第 6 步）；
缺省时内置直同名映射（VRM 1.0 expressions 标准预设词表 = 契约词表）。
设计要点（D10）：表情平行于动作与情绪——blink/gaze 是独立字段，本模块把它们
一并解析为 morph 权重（blink → blink 预设；gaze → look* 预设按主导轴取一）。
"""

from __future__ import annotations

from pet.scene_contract import ExpressionState, EXPRESSION_PRESETS

# 内置默认：契约词表 → VRM 面部 morph 预设（直同名；blink/look 走功能预设）。
DEFAULT_MAP: dict[str, dict[str, float]] = {
    label: {label: 1.0} for label in sorted(EXPRESSION_PRESETS)
}
DEFAULT_MAP["neutral"] = {}   # 中性 = 全零权重


class ExpressionResolver:
    """把 ExpressionState 解析为 {morph 预设名: 权重 0-1}。"""

    def __init__(self, sidecar: dict | None = None):
        # sidecar 形如 {"happy": {"Fcl_HAP": 1.0}, ...}——键=契约词表，值=morph 权重表
        self._map = dict(DEFAULT_MAP)
        if sidecar:
            for label, weights in sidecar.items():
                if isinstance(weights, dict) and weights:
                    self._map[label] = {str(k): float(v) for k, v in weights.items()}

    def resolve(self, expr: ExpressionState) -> dict[str, float]:
        out: dict[str, float] = {}
        # 情绪预设（None=中性=全零）
        if expr.emotion_label is not None:
            out.update(self._map.get(expr.emotion_label, {}))
        # 眨眼：blink_progress 直接映射 blink 预设（VRM 眨眼为闭眼幅度）
        if expr.blink_progress > 0.0:
            out["blink"] = round(expr.blink_progress, 4)
        # 视线：主导轴取一（避免对角冲突），权重=偏移幅度
        ax, ay = abs(expr.gaze_x), abs(expr.gaze_y)
        if max(ax, ay) > 0.02:
            if ax >= ay:
                out["lookLeft" if expr.gaze_x < 0 else "lookRight"] = round(ax, 4)
            else:
                out["lookDown" if expr.gaze_y < 0 else "lookUp"] = round(ay, 4)
        return out
