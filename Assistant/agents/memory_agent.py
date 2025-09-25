from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from Assistant.core.memory import (
    delete_fact as _delete_fact,
    list_user_facts,
    save_inference,
)
from Assistant.core.tools import ToolBox


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


class MemoryAgent:
    """Agent encapsulating memory management tools."""

    def __init__(self, default_user_id: str = "user_default") -> None:
        self.default_user_id = default_user_id
        self.toolbox = ToolBox(namespace="memory")
        self._register_tools()
        self._primary_actions = {
            "list_facts": self.toolbox.qualify("list_facts"),
            "save_fact": self.toolbox.qualify("save_fact"),
            "delete_fact": self.toolbox.qualify("delete_fact"),
        }
        self._action_map = self._build_action_map()

    def _build_action_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for action, tool_name in self._primary_actions.items():
            mapping[action] = tool_name
        synonyms = {
            "list": "list_facts",
            "get": "list_facts",
            "facts": "list_facts",
            "remember": "save_fact",
            "save": "save_fact",
            "add": "save_fact",
            "remove": "delete_fact",
            "delete": "delete_fact",
            "forget": "delete_fact",
        }
        for alias, canonical in synonyms.items():
            mapping[alias] = self._primary_actions[canonical]
        return mapping

    def _resolve_action(self, action: str | None) -> str | None:
        key = (action or "list_facts").strip().lower()
        return self._action_map.get(key)

    def actions(self) -> list[str]:
        return sorted(self._primary_actions.keys())

    def capabilities(self) -> list[str]:
        return [schema["function"]["name"] for schema in self.toolbox.tools]

    def run(self, action: str | None, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        tool_name = self._resolve_action(action)
        if tool_name is None:
            return {"error": f"Unknown memory action: {action}"}
        try:
            return self.toolbox.invoke(tool_name, params or {})
        except TypeError as exc:
            return {"error": str(exc)}

    def _register_tools(self) -> None:
        @self.toolbox.tool(description="List stored user facts.")
        def list_facts(user_id: str | None = None) -> Dict[str, Any]:
            uid = self._resolve_user(user_id)
            return {"ok": True, "user_id": uid, "facts": list_user_facts(uid)}

        @self.toolbox.tool(description="Save or update a user fact.")
        def save_fact(
            fact: str,
            confidence: float = 0.5,
            sensitivity: str = "low",
            rationale: str | None = None,
            source: str | None = None,
            require_consent: bool = False,
            user_id: str | None = None,
        ) -> Dict[str, Any]:
            uid = self._resolve_user(user_id)
            return save_inference(
                uid,
                fact=fact,
                confidence=float(confidence),
                sensitivity=sensitivity,
                rationale=rationale,
                source=source,
                require_consent=_as_bool(require_consent),
            )

        @self.toolbox.tool(description="Remove a fact by exact text match.")
        def delete_fact(fact: str, user_id: str | None = None) -> Dict[str, Any]:
            uid = self._resolve_user(user_id)
            return _delete_fact(uid, fact)

    def _resolve_user(self, user_id: Optional[str]) -> str:
        return user_id or self.default_user_id

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.toolbox.invoke(tool_name, arguments)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _build_transcript(entries: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for e in entries:
        u = (e.get("user_input") or "").strip()
        a = (e.get("assistant_output") or "").strip()
        if u:
            parts.append(f"User: {u}")
        if a:
            parts.append(f"Assistant: {a}")
    return "\n".join(parts)


def _analysis_prompt(transcript: str) -> List[Dict[str, str]]:
    system = (
        "You are a memory analyst for an assistant. Read the full conversation "
        "transcript and extract a small list (0-8) of durable, non-sensitive, actionable user facts/preferences. "
        "Skip meta facts about AI or transient statements. Prefer preferences, routines, stable attributes. "
        "For each fact, estimate confidence in [0.2, 0.9] and sensitivity one of: low, medium, high. "
        "Output ONLY a JSON array of objects with keys: fact, confidence, sensitivity."
    )
    user = (
        "Conversation transcript follows. Return JSON only.\n\n" + transcript
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    """Best-effort to parse a JSON array from model output."""
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj  # type: ignore[return-value]
    except Exception:
        pass
    try:
        start = s.find("[")
        end = s.rfind("]")
        if start >= 0 and end > start:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, list):
                return obj  # type: ignore[return-value]
    except Exception:
        return None
    return None


def review_and_save_from_log(
    *,
    session_id: str,
    user_id: str,
    client: Any,
    log_dir: str = "interaction_logs",
) -> Dict[str, Any]:
    """Analyze a session log and save/update facts in the background."""
    path = os.path.join(log_dir, f"{session_id}.jsonl")
    entries = _read_jsonl(path)
    if not entries:
        return {"ok": False, "error": f"no log entries for session {session_id}"}

    transcript = _build_transcript(entries)
    messages = _analysis_prompt(transcript)

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            max_completion_tokens=800,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"ok": False, "error": f"model error: {e}"}

    items = _parse_json_list(text) or []
    saved = 0
    skipped = 0
    errors: List[str] = []
    for it in items:
        try:
            fact = str((it or {}).get("fact") or "").strip()
            if not fact:
                skipped += 1
                continue
            conf_raw = (it or {}).get("confidence")
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = 0.3
            sensitivity = str((it or {}).get("sensitivity") or "low").strip().lower()
            res = save_inference(
                user_id,
                fact=fact,
                confidence=confidence,
                sensitivity=sensitivity,
                rationale=None,
                source=f"log:{session_id}",
                require_consent=(sensitivity in {"medium", "high"}),
            )
            if res.get("ok") or res.get("pending"):
                saved += 1
            else:
                skipped += 1
        except Exception as e:
            errors.append(str(e))
            skipped += 1

    return {
        "ok": True,
        "analyzed": len(entries),
        "candidates": len(items),
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
    }


__all__ = [
    "MemoryAgent",
    "review_and_save_from_log",
]
