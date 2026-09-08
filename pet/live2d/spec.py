"""Live2D 模型路径与 mood/动作映射 —— v0.15。

``mapping.json`` 把养成 Mood / 交互动作对到具体模型的 expression id
与 motion group。缺文件或非法 JSON → None（调用方回退 frames）。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import jsonschema

log = logging.getLogger("pet")

_MAPPING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "spec": {"const": 1},
        "model": {"type": "string", "minLength": 1},
        "display_name": {"type": "string"},
        "license": {"type": "string"},
        "expressions": {
            "type": "object",
            "additionalProperties": {"type": ["string", "null"]},
        },
        "motions": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "group": {"type": "string", "minLength": 1},
                            "index": {"type": "integer", "minimum": 0},
                            "priority": {"type": "integer", "minimum": 0,
                                         "maximum": 3},
                        },
                        "required": ["group"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "parameters": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "hit_areas": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["spec"],
    "additionalProperties": False,
}

_DEFAULT_EXPRESSIONS = {
    "happy": None,
    "neutral": None,
    "sad": None,
    "sleepy": None,
    "hungry": None,
    "neglected": None,
}


@dataclass
class MotionRef:
    group: str
    index: int | None = None
    priority: int = 2


@dataclass
class Live2DMapping:
    """一份模型旁的语义映射（路径已解析为绝对路径）。"""

    model_path: str
    display_name: str = ""
    expressions: dict[str, str | None] = field(default_factory=dict)
    motions: dict[str, MotionRef | None] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)
    hit_areas: dict[str, str] = field(default_factory=dict)

    def expression_for(self, mood: str, neglected: bool = False) -> str | None:
        """Mood 名 → expression id；落寞分支优先 ``neglected`` 再回落 mood。"""
        if neglected:
            n = self.expressions.get("neglected")
            if n:
                return n
        return self.expressions.get(mood) or self.expressions.get("neutral")

    def motion_for(self, action: str) -> MotionRef | None:
        return self.motions.get(action)


def default_live2d_root() -> str:
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "assets", "live2d"))


def resolve_model_path(cfg: dict | None = None, root: str = "") -> str | None:
    """从 config.live2d.model 或默认 Haru 路径解析 .model3.json。

    相对路径相对仓库根（``assets/live2d/...``）；绝对路径原样。
    文件不存在 → None。
    """
    cfg = cfg or {}
    raw = (cfg.get("model") or "").strip()
    if not root:
        root = default_live2d_root()
    repo = os.path.normpath(os.path.join(root, "..", ".."))
    if raw:
        path = raw if os.path.isabs(raw) else os.path.normpath(
            os.path.join(repo, raw))
    else:
        path = os.path.join(root, "haru", "Haru.model3.json")
    if os.path.isfile(path):
        return path
    log.warning("live2d 模型文件不存在：%s", path)
    return None


def load_live2d_mapping(model_path: str, mapping_path: str = "") -> Live2DMapping | None:
    """读模型旁 mapping.json；缺失时用空映射（仍可加载模型，只是无表情表）。

    mapping 非法 → None（宁可回退 frames，避免静默绑错动作）。
    """
    if not model_path or not os.path.isfile(model_path):
        return None
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if not mapping_path:
        mapping_path = os.path.join(model_dir, "mapping.json")
    elif not os.path.isabs(mapping_path):
        repo = os.path.normpath(os.path.join(default_live2d_root(), "..", ".."))
        mapping_path = os.path.normpath(os.path.join(repo, mapping_path))

    expressions = dict(_DEFAULT_EXPRESSIONS)
    motions: dict[str, MotionRef | None] = {}
    parameters: dict[str, str] = {}
    hit_areas: dict[str, str] = {}
    display_name = os.path.splitext(os.path.basename(model_path))[0]
    declared_model = None

    if os.path.isfile(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            jsonschema.Draft7Validator(_MAPPING_SCHEMA).validate(raw)
        except (OSError, ValueError, jsonschema.ValidationError) as e:
            log.warning("live2d mapping %s 非法，回退帧动画：%s",
                        mapping_path, e)
            return None
        display_name = raw.get("display_name") or display_name
        declared_model = raw.get("model")
        expressions.update(raw.get("expressions") or {})
        parameters.update(raw.get("parameters") or {})
        hit_areas.update(raw.get("hit_areas") or {})
        for key, val in (raw.get("motions") or {}).items():
            if val is None:
                motions[key] = None
            else:
                motions[key] = MotionRef(
                    group=val["group"],
                    index=val.get("index"),
                    priority=int(val.get("priority", 2)),
                )
        if declared_model:
            cand = os.path.normpath(os.path.join(model_dir, declared_model))
            if os.path.isfile(cand):
                model_path = cand

    return Live2DMapping(
        model_path=os.path.abspath(model_path),
        display_name=display_name,
        expressions=expressions,
        motions=motions,
        parameters=parameters,
        hit_areas=hit_areas,
    )
