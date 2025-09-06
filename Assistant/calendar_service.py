import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import re


@dataclass
class CalendarEvent:
    id: str
    title: str
    start: str  # ISO 8601 string
    end: Optional[str] = None  # ISO 8601 string
    timezone: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    attendees: Optional[List[str]] = None

    @staticmethod
    def from_dict(d: dict) -> "CalendarEvent":
        # Coerce attendees to a list[str] if provided incorrectly
        raw_attendees = d.get("attendees")
        attendees: List[str]
        if isinstance(raw_attendees, list):
            attendees = [str(x).strip() for x in raw_attendees if str(x).strip()]
        elif isinstance(raw_attendees, str):
            attendees = [s.strip() for s in re.split(r"\s*(?:,|;| and )\s*", raw_attendees) if s.strip()]
        else:
            attendees = []

        # Extract attendees embedded in notes like "Attendees: Alice, Bob"
        notes = d.get("notes")
        if isinstance(notes, str) and notes.strip():
            lines = notes.splitlines()
            kept_lines: List[str] = []
            extracted: List[str] = []
            rx = re.compile(r"^\s*(attendees?|participants?)\s*:\s*(.+)$", re.IGNORECASE)
            for line in lines:
                m = rx.match(line)
                if m:
                    names_str = m.group(2).strip()
                    if names_str:
                        parts = [s.strip() for s in re.split(r"\s*(?:,|;| and )\s*", names_str) if s.strip()]
                        extracted.extend(parts)
                else:
                    kept_lines.append(line)
            if extracted:
                # Merge unique attendees preserving order
                seen = set()
                merged: List[str] = []
                for name in attendees + extracted:
                    key = name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(name)
                attendees = merged
                notes = "\n".join([l for l in kept_lines if l.strip()]) or None

        return CalendarEvent(
            id=d.get("id") or str(uuid.uuid4()),
            title=d.get("title", "Untitled"),
            start=d.get("start"),
            end=d.get("end"),
            timezone=d.get("timezone"),
            location=d.get("location"),
            notes=notes,
            attendees=attendees,
        )


class CalendarService:
    def __init__(self, storage_path: str = "Assistant/calendar_events.json") -> None:
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        self._events: List[CalendarEvent] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            self._events = []
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._events = [CalendarEvent.from_dict(e) for e in data]
            # If any normalization changed notes/attendees compared to raw file, persist back
            dirty = False
            for ev, raw in zip(self._events, data):
                raw_attendees = raw.get("attendees") or []
                if isinstance(raw_attendees, str):
                    raw_attendees = [s.strip() for s in re.split(r"\s*(?:,|;| and )\s*", raw_attendees) if s.strip()]
                # Compare case-insensitively to avoid trivial diffs
                def _norm_list(xs):
                    return [str(x).strip().lower() for x in (xs or []) if str(x).strip()]
                if (raw.get("notes") != ev.notes) or (_norm_list(raw_attendees) != _norm_list(ev.attendees)):
                    dirty = True
            if dirty:
                self._save()
        except Exception:
            self._events = []

    def _save(self) -> None:
        data = [asdict(e) for e in self._events]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_event(self, event_dict: dict) -> CalendarEvent:
        event = CalendarEvent.from_dict(event_dict)
        # Ensure notes/attendees are normalized before saving
        self._normalize_event_object(event)
        if not event.start:
            raise ValueError("Event 'start' (ISO 8601) is required")
        # Basic sanity: ensure ISO-like format; will not strictly validate TZ
        try:
            # Accept naive or offset-aware ISO strings
            datetime.fromisoformat(event.start.replace("Z", "+00:00"))
            if event.end:
                datetime.fromisoformat(event.end.replace("Z", "+00:00"))
        except Exception:
            # keep, but caller should ensure correctness; we don't block
            pass

        if not event.id:
            event.id = str(uuid.uuid4())
        self._events.append(event)
        self._save()
        return event

    def list_events(self) -> List[CalendarEvent]:
        return list(self._events)

    def upcoming(self, now: Optional[datetime] = None, limit: int = 10) -> List[CalendarEvent]:
        now = now or datetime.now()
        def start_dt(ev: CalendarEvent):
            try:
                return datetime.fromisoformat((ev.start or "").replace("Z", "+00:00"))
            except Exception:
                return datetime.max
        events = sorted(self._events, key=start_dt)
        return [e for e in events if start_dt(e) >= now][:limit]

    def range(self, start_iso: str, end_iso: str) -> List[CalendarEvent]:
        """Return events whose start is within [start_iso, end_iso].

        - Accepts naive or offset-aware ISO strings.
        - Normalizes all comparisons to the local timezone to avoid
          naive/aware comparison errors.
        """
        # Local tzinfo for normalization
        local_tz = datetime.now().astimezone().tzinfo

        def _to_local_aware(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=local_tz)
            return dt.astimezone(local_tz)

        try:
            start_dt_raw = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            end_dt_raw = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            start_dt = _to_local_aware(start_dt_raw)
            end_dt = _to_local_aware(end_dt_raw)
        except Exception:
            return []

        out: List[CalendarEvent] = []
        for e in self._events:
            try:
                s_raw = datetime.fromisoformat((e.start or "").replace("Z", "+00:00"))
                s = _to_local_aware(s_raw)
            except Exception:
                continue
            if start_dt <= s <= end_dt:
                out.append(e)

        # Sort consistently by actual datetime value in local tz
        def _sort_key(ev: CalendarEvent):
            try:
                dt_raw = datetime.fromisoformat((ev.start or "").replace("Z", "+00:00"))
                return _to_local_aware(dt_raw)
            except Exception:
                # Place unparseable events at the end
                return datetime.max.replace(tzinfo=local_tz)

        out.sort(key=_sort_key)
        return out

    def on_date(self, date_str: str) -> List[CalendarEvent]:
        """Return events that start on the given local date (YYYY-MM-DD)."""
        try:
            y, m, d = map(int, date_str.split("-"))
        except Exception:
            return []
        out: List[CalendarEvent] = []
        for e in self._events:
            try:
                s = datetime.fromisoformat((e.start or "").replace("Z", "+00:00"))
            except Exception:
                continue
            if s.year == y and s.month == m and s.day == d:
                out.append(e)
        out.sort(key=lambda ev: ev.start)
        return out

    def candidates_by_title(self, title_substring: str, date_only: Optional[str] = None) -> List[CalendarEvent]:
        """Find events whose title contains the substring (case-insensitive).

        If date_only (YYYY-MM-DD) is provided, filter to events starting on that date.
        Results are sorted by start time ascending.
        """
        ts = (title_substring or "").strip().lower()
        if not ts:
            return []
        def start_dt(ev: CalendarEvent):
            try:
                return datetime.fromisoformat((ev.start or "").replace("Z", "+00:00"))
            except Exception:
                return datetime.max
        results: List[CalendarEvent] = []
        for e in self._events:
            et = (e.title or "").strip().lower()
            if ts in et or et in ts:
                if date_only:
                    try:
                        s = datetime.fromisoformat((e.start or "").replace("Z", "+00:00"))
                        d_only = f"{s.year:04d}-{s.month:02d}-{s.day:02d}"
                        if d_only != date_only:
                            continue
                    except Exception:
                        continue
                results.append(e)
        results.sort(key=start_dt)
        return results

    # --- Find / Update / Delete helpers ---
    def get_by_id(self, event_id: str) -> Optional[CalendarEvent]:
        for e in self._events:
            if e.id == event_id:
                return e
        return None

    def find_by_title_start(self, title: str, start: Optional[str] = None) -> Optional[CalendarEvent]:
        title_l = (title or "").strip().lower()
        for e in self._events:
            if (e.title or "").strip().lower() == title_l:
                if start is None or (e.start or "") == start:
                    return e
        return None

    def delete_by_id(self, event_id: str) -> Optional[CalendarEvent]:
        for i, e in enumerate(self._events):
            if e.id == event_id:
                removed = self._events.pop(i)
                self._save()
                return removed
        return None

    def delete_by_title_start(self, title: str, start: Optional[str] = None) -> Optional[CalendarEvent]:
        ev = self.find_by_title_start(title, start)
        if not ev:
            return None
        self._events = [e for e in self._events if e.id != ev.id]
        self._save()
        return ev

    def update_by_id(self, event_id: str, set_fields: dict) -> Optional[CalendarEvent]:
        ev = self.get_by_id(event_id)
        if not ev:
            return None
        self._apply_fields(ev, set_fields)
        self._normalize_event_object(ev)
        self._save()
        return ev

    def update_by_title_start(self, title: str, start: Optional[str], set_fields: dict) -> Optional[CalendarEvent]:
        ev = self.find_by_title_start(title, start)
        if not ev:
            return None
        self._apply_fields(ev, set_fields)
        self._normalize_event_object(ev)
        self._save()
        return ev

    @staticmethod
    def _apply_fields(ev: CalendarEvent, fields: dict) -> None:
        for k in ["title", "start", "end", "timezone", "location", "notes", "attendees"]:
            if k in fields:
                setattr(ev, k, fields[k])

    @staticmethod
    def _normalize_event_object(ev: CalendarEvent) -> bool:
        """Normalize attendee info embedded in notes. Returns True if modified."""
        changed = False
        # Normalize attendees list to list[str]
        if isinstance(ev.attendees, list):
            attendees = [str(x).strip() for x in ev.attendees if str(x).strip()]
        elif isinstance(ev.attendees, str):
            attendees = [s.strip() for s in re.split(r"\s*(?:,|;| and )\s*", ev.attendees) if s.strip()]
            changed = True
        else:
            attendees = []
            if ev.attendees not in (None, []):
                changed = True

        notes = ev.notes
        extracted: List[str] = []
        if isinstance(notes, str) and notes.strip():
            lines = notes.splitlines()
            kept: List[str] = []
            rx = re.compile(r"^\s*(attendees?|participants?)\s*:\s*(.+)$", re.IGNORECASE)
            for line in lines:
                m = rx.match(line)
                if m:
                    names_str = m.group(2).strip()
                    if names_str:
                        parts = [s.strip() for s in re.split(r"\s*(?:,|;| and )\s*", names_str) if s.strip()]
                        extracted.extend(parts)
                else:
                    kept.append(line)
            if extracted:
                # Merge attendees with extracted; dedupe case-insensitively preserving order
                seen = set()
                merged: List[str] = []
                for name in attendees + extracted:
                    key = name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(name)
                if merged != attendees:
                    attendees = merged
                    changed = True
                new_notes = "\n".join([l for l in kept if l.strip()]) or None
                if new_notes != notes:
                    notes = new_notes
                    changed = True

        if changed:
            ev.attendees = attendees
            ev.notes = notes
        return changed
