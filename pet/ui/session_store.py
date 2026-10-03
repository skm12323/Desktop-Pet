"""聊天多会话存储 —— v0.17.1 细节打磨（多会话数据层 + JSON 持久化）。

``SessionStore``：会话（id/标题/时间/UI messages/DS history）+ active_id，
落盘 ``data_dir/chat_sessions.json``。原子写模式对齐 ``PetStateStore``
（tmp→fsync→.bak→replace）；读取损坏先 .json 后 .bak 兜底，都坏返回空库
（会话历史非关键档，不抛——大不了从头聊）。

v0.17.1 摘要解耦（行为变化）：UI messages 本地全量保留、重启可恢复；
滚动摘要只压缩喂 DS 的 ``history``（摘要头 turn 也在 history 里随会话
持久化），不再删除 UI 行——旧版 beginRemoveRows 删 UI 消息与"回看
历史"的诉求冲突。

序列化边界：messages 只存 role/content（``rich`` 渲染 HTML 恢复时由
``_md_to_html`` 重算，格式升级自动生效）；history 存全 ChatTurn 字段
（含 tool 轮与 tool_calls 原始 dict）。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("pet")

SCHEMA_VERSION = 1
_TITLE_MAX = 16


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_title(text: str) -> str:
    """会话标题：首条 user 消息前 16 字符（压空白）；空 → 新对话。"""
    t = " ".join((text or "").split())
    return t[:_TITLE_MAX] or "新对话"


def turns_to_dicts(turns: list) -> list[dict]:
    """ChatTurn → 可 JSON dict（四字段全存，含 tool 轮）。"""
    return [
        {
            "role": t.role,
            "content": t.content,
            "tool_call_id": getattr(t, "tool_call_id", ""),
            "tool_calls": getattr(t, "tool_calls", None) or [],
        }
        for t in turns
    ]


def dicts_to_turns(dicts: list) -> list:
    """dict → ChatTurn（延迟 import：与 chat_bridge 同风格，避免模块顶
    拉起 llm 的 provider 依赖链）。"""
    from ..llm import ChatTurn

    return [
        ChatTurn(
            d.get("role", "user"),
            d.get("content", ""),
            d.get("tool_call_id", ""),
            d.get("tool_calls") or [],
        )
        for d in dicts
    ]


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list = field(default_factory=list)  # [{"role","content","rich"}] UI 行
    history: list = field(default_factory=list)   # list[ChatTurn] 喂 DS

    def touch(self) -> None:
        self.updated_at = _now()


class SessionStore:
    """会话容器。``path=None`` 纯内存（测试/降级），否则 load+save 落盘。"""

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._order: list[str] = []            # 插入序（新会话在尾）
        self._sessions: dict[str, ChatSession] = {}
        self._active_id: str | None = None
        if path:
            self._load()

    # ---- 会话操作 ----
    def new_session(self) -> ChatSession:
        s = ChatSession(
            id=uuid.uuid4().hex[:12], title="新对话",
            created_at=_now(), updated_at=_now(),
        )
        self._sessions[s.id] = s
        self._order.append(s.id)
        self._active_id = s.id
        return s

    def switch(self, sid: str) -> ChatSession | None:
        s = self._sessions.get(sid)
        if s is not None:
            self._active_id = sid
        return s   # 未命中返回 None，active 不动（QML 层可提示）

    @property
    def active(self) -> ChatSession | None:
        return self._sessions.get(self._active_id) if self._active_id else None

    def get(self, sid: str) -> ChatSession | None:
        return self._sessions.get(sid)

    def list_sessions(self) -> list[ChatSession]:
        """最近更新在前（v0.17.2 顶栏下拉数据源）。"""
        return sorted(self._sessions.values(),
                      key=lambda s: s.updated_at, reverse=True)

    # ---- 持久化 ----
    def save(self) -> None:
        """原子写（tmp→fsync→.bak→replace）；失败保留内存态不抛。"""
        if not self._path:
            return
        data = {
            "version": SCHEMA_VERSION,
            "active_id": self._active_id,
            "sessions": [
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    # messages 只存 role/content（rich 重算，见模块 docstring）
                    "messages": [
                        {"role": m["role"], "content": m.get("content", "")}
                        for m in s.messages
                    ],
                    "history": turns_to_dicts(s.history),
                }
                for s in (self._sessions[i] for i in self._order)
            ],
        }
        tmp = self._path + ".tmp"
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(self._path):
                try:
                    shutil.copy2(self._path, self._path + ".bak")
                except OSError:
                    pass
            os.replace(tmp, self._path)
        except OSError:
            log.warning("[会话] 持久化失败（保留内存态）", exc_info=True)

    def _load(self) -> None:
        for candidate in (self._path, self._path + ".bak"):
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, encoding="utf-8") as f:
                    data = json.load(f)
                for d in data.get("sessions", []):
                    s = ChatSession(
                        id=d["id"],
                        title=d.get("title") or "新对话",
                        created_at=d.get("created_at", ""),
                        updated_at=d.get("updated_at", ""),
                        messages=[
                            {"role": m["role"], "content": m.get("content", "")}
                            for m in d.get("messages", [])
                        ],
                        history=dicts_to_turns(d.get("history", [])),
                    )
                    self._sessions[s.id] = s
                    self._order.append(s.id)
                self._active_id = data.get("active_id")
                if self._active_id not in self._sessions:
                    self._active_id = self._order[-1] if self._order else None
                log.info("[会话] 载入 %d 个会话", len(self._order))
                return
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("[会话] 读取失败(%s)，尝试下一来源", candidate,
                            exc_info=True)
        # 首次运行/全部损坏：空库（调用方 new_session 兜底）
        self._sessions.clear()
        self._order.clear()
        self._active_id = None
