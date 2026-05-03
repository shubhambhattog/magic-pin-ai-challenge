from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@dataclass
class ConversationMeta:
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    send_as: Optional[str] = None


class ContextStore:
    def __init__(self) -> None:
        self._contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}
        self._conversation_meta: Dict[str, ConversationMeta] = {}
        self._suppressed: set[str] = set()
        self._sent_bodies: Dict[str, set[str]] = {}
        self._auto_reply_counts: Dict[str, int] = {}
        self._ended_conversations: set[str] = set()

    def push_context(self, scope: str, context_id: str, version: int, payload: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[int]]:
        if scope not in VALID_SCOPES:
            return False, "invalid_scope", None
        key = (scope, context_id)
        current = self._contexts.get(key)
        if current and current.get("version", 0) > version:
            return False, "stale_version", current.get("version", 0)
        self._contexts[key] = {"version": version, "payload": payload}
        return True, None, None

    def get_context(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        entry = self._contexts.get((scope, context_id))
        if not entry:
            return None
        return entry.get("payload")

    def count_by_scope(self) -> Dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _context_id) in self._contexts.keys():
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def is_suppressed(self, key: Optional[str]) -> bool:
        if not key:
            return False
        return key in self._suppressed

    def add_suppression(self, key: Optional[str]) -> None:
        if key:
            self._suppressed.add(key)

    def record_sent(self, conversation_id: str, body: str) -> None:
        body = (body or "").strip()
        if not body:
            return
        self._sent_bodies.setdefault(conversation_id, set()).add(body)

    def is_repeat(self, conversation_id: str, body: str) -> bool:
        body = (body or "").strip()
        if not body:
            return False
        return body in self._sent_bodies.get(conversation_id, set())

    def append_turn(self, conversation_id: str, from_role: str, message: str, timestamp: Optional[str]) -> None:
        self._conversations.setdefault(conversation_id, []).append(
            {"from_role": from_role, "message": message, "timestamp": timestamp}
        )

    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self._conversations.get(conversation_id, [])

    def set_conversation_meta(self, conversation_id: str, meta: ConversationMeta) -> None:
        self._conversation_meta[conversation_id] = meta

    def get_conversation_meta(self, conversation_id: str) -> ConversationMeta:
        return self._conversation_meta.get(conversation_id, ConversationMeta())

    def bump_auto_reply_count(self, conversation_id: str) -> int:
        self._auto_reply_counts[conversation_id] = self._auto_reply_counts.get(conversation_id, 0) + 1
        return self._auto_reply_counts[conversation_id]

    def reset_auto_reply_count(self, conversation_id: str) -> None:
        if conversation_id in self._auto_reply_counts:
            del self._auto_reply_counts[conversation_id]

    def end_conversation(self, conversation_id: str) -> None:
        self._ended_conversations.add(conversation_id)

    def is_conversation_ended(self, conversation_id: str) -> bool:
        return conversation_id in self._ended_conversations
