from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from Assistant.core.paths import FACTS_PATH


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _empty_store() -> Dict[str, Any]:
    return {"users": {}}


def _load_store() -> Dict[str, Any]:
    if not FACTS_PATH.exists():
        return _empty_store()
    try:
        with open(FACTS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return _empty_store()
        data.setdefault("users", {})
        return data
    except Exception:
        return _empty_store()


def _save_store(data: Dict[str, Any]) -> None:
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _user_bucket(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    users = data.setdefault("users", {})
    bucket = users.setdefault(user_id, {})
    bucket.setdefault("facts", [])
    return bucket


def list_user_facts(user_id: str) -> List[Dict[str, Any]]:
    data = _load_store()
    bucket = data.get("users", {}).get(user_id, {})
    facts = bucket.get("facts", [])
    if not isinstance(facts, list):
        return []
    return [dict(fact) for fact in facts if isinstance(fact, dict)]


def save_inference(
    user_id: str,
    *,
    fact: str,
    confidence: float,
    sensitivity: str,
    rationale: str | None,
    source: str | None,
    require_consent: bool = False,
) -> Dict[str, Any]:
    fact_text = (fact or "").strip()
    if not fact_text:
        return {"ok": False, "error": "fact text required"}

    data = _load_store()
    bucket = _user_bucket(data, user_id)
    facts: List[Dict[str, Any]] = bucket["facts"]
    normalized = fact_text.lower()
    now = _now_iso()

    for existing in facts:
        if not isinstance(existing, dict):
            continue
        if (existing.get("fact") or "").strip().lower() == normalized:
            existing["fact"] = fact_text
            existing["confidence"] = float(confidence)
            existing["sensitivity"] = sensitivity
            existing["rationale"] = rationale
            existing["last_updated"] = now
            existing.setdefault("first_seen", now)
            existing.setdefault("status", "active")
            existing["count"] = int(existing.get("count", 0)) + 1
            evidence = existing.setdefault("evidence", [])
            if source and source not in evidence:
                evidence.append(source)
            existing["pending"] = require_consent
            _save_store(data)
            return {"ok": True, "fact": existing, "pending": require_consent}

    entry = {
        "fact": fact_text,
        "confidence": float(confidence),
        "sensitivity": sensitivity,
        "status": "active",
        "rationale": rationale,
        "evidence": [source] if source else [],
        "first_seen": now,
        "last_updated": now,
        "count": 1,
        "pending": require_consent,
    }
    facts.append(entry)
    _save_store(data)
    return {"ok": True, "fact": entry, "pending": require_consent}


def delete_fact(user_id: str, fact_text: str) -> Dict[str, Any]:
    target = (fact_text or "").strip().lower()
    if not target:
        return {"ok": False, "error": "fact text required"}

    data = _load_store()
    bucket = _user_bucket(data, user_id)
    facts: List[Any] = bucket["facts"]
    kept: List[Any] = []
    removed = False
    for item in facts:
        if isinstance(item, dict):
            candidate = (item.get("fact") or "").strip().lower()
            if candidate == target:
                removed = True
                continue
        kept.append(item)
    if not removed:
        return {"ok": False, "error": "fact not found"}
    bucket["facts"] = kept
    _save_store(data)
    return {"ok": True}


def rename_user(old_user_id: str, new_user_id: str) -> None:
    if old_user_id == new_user_id:
        return
    data = _load_store()
    users = data.setdefault("users", {})
    if old_user_id not in users:
        return
    users[new_user_id] = users.pop(old_user_id)
    _save_store(data)


__all__ = [
    "list_user_facts",
    "save_inference",
    "delete_fact",
    "rename_user",
]
