from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from Assistant.services.calendar.service import CalendarService

_calendar_service: CalendarService | None
try:
    _calendar_service = CalendarService()
except Exception:
    _calendar_service = None

AVAILABLE: bool = _calendar_service is not None


def _event_to_dict(ev: Any) -> Dict[str, Any]:
    try:
        return {
            "id": getattr(ev, "id", None),
            "title": getattr(ev, "title", None),
            "start": getattr(ev, "start", None),
            "end": getattr(ev, "end", None),
            "timezone": getattr(ev, "timezone", None),
            "location": getattr(ev, "location", None),
            "notes": getattr(ev, "notes", None),
            "attendees": getattr(ev, "attendees", None),
        }
    except Exception:
        try:
            return dict(ev)  # type: ignore[arg-type]
        except Exception:
            return {"error": "unserializable_event"}


def _service() -> CalendarService:
    if _calendar_service is None:
        raise RuntimeError("calendar service not available")
    return _calendar_service


def add_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Add an event to the calendar."""
    try:
        ev = _service().add_event(event)
        return {"ok": True, "event": _event_to_dict(ev)}
    except Exception as exc:
        return {"error": str(exc)}


def _now_local() -> datetime:
    return datetime.now().astimezone()


def list_events(*, include_past: bool = False) -> Dict[str, Any]:
    try:
        evs = _service().list_events()
    except Exception as exc:
        return {"error": str(exc)}

    now = _now_local()
    output: List[Dict[str, Any]] = []
    for event in evs:
        try:
            start = datetime.fromisoformat((getattr(event, "start", "") or "").replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=now.tzinfo)
            else:
                start = start.astimezone(now.tzinfo)
        except Exception:
            if not include_past:
                continue
        else:
            if not include_past and start < now:
                continue
        output.append(_event_to_dict(event))

    try:
        output.sort(key=lambda item: item.get("start") or "")
    except Exception:
        pass
    return {"ok": True, "events": output}


def upcoming(*, limit: int = 10) -> Dict[str, Any]:
    try:
        events = _service().upcoming(now=_now_local(), limit=int(limit))
        return {"ok": True, "events": [_event_to_dict(ev) for ev in events]}
    except Exception as exc:
        return {"error": str(exc)}


def events_in_range(*, start: str, end: str) -> Dict[str, Any]:
    if not start or not end:
        return {"error": "start and end required (ISO)"}
    try:
        events = _service().range(start, end)
    except Exception as exc:
        return {"error": str(exc)}
    return {"ok": True, "events": [_event_to_dict(ev) for ev in events]}


def events_on_date(*, date: str) -> Dict[str, Any]:
    if not date:
        return {"error": "date (YYYY-MM-DD) required"}
    try:
        events = _service().on_date(date)
        return {"ok": True, "events": [_event_to_dict(ev) for ev in events]}
    except Exception as exc:
        return {"error": str(exc)}


def this_week() -> Dict[str, Any]:
    now = _now_local()
    weekday = now.weekday()
    days_until_sunday = 6 - weekday
    end_of_week = (now + timedelta(days=days_until_sunday)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    try:
        events = _service().range(now.isoformat(), end_of_week.isoformat())
        return {"ok": True, "events": [_event_to_dict(ev) for ev in events]}
    except Exception as exc:
        return {"error": str(exc)}


def update_event(
    *,
    id: Optional[str] = None,
    title: Optional[str] = None,
    start: Optional[str] = None,
    set_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not id and not title:
        return {"error": "Provide 'id' or ('title' and optional 'start')"}
    payload = set_fields or {}
    try:
        if id:
            event = _service().update_by_id(id, payload)
        else:
            event = _service().update_by_title_start(title or "", start, payload)
    except Exception as exc:
        return {"error": str(exc)}
    if not event:
        return {"error": "event not found"}
    return {"ok": True, "event": _event_to_dict(event)}


def delete_event(
    *,
    id: Optional[str] = None,
    title: Optional[str] = None,
    start: Optional[str] = None,
) -> Dict[str, Any]:
    if not id and not title:
        return {"error": "Provide 'id' or ('title' and optional 'start')"}
    try:
        if id:
            event = _service().delete_by_id(id)
        else:
            if not title:
                return {"error": "Provide 'id' or 'title'"}
            event = _service().delete_by_title_start(title, start)
    except Exception as exc:
        return {"error": str(exc)}
    if not event:
        return {"error": "event not found"}
    return {"ok": True, "event": _event_to_dict(event)}


__all__ = [
    "AVAILABLE",
    "add_event",
    "list_events",
    "upcoming",
    "events_in_range",
    "events_on_date",
    "this_week",
    "update_event",
    "delete_event",
]
