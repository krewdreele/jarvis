import os
import json
import datetime
import re
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .calendar_service import CalendarService
except Exception:  # pragma: no cover
    from calendar_service import CalendarService


load_dotenv()


# Calendar tool tags (calendar model private contract)
# Accept both ":" and no-colon forms before JSON payloads
CAL_CREATE_TAG_RE = re.compile(r"^\s*(CALENDAR_EVENT|CALENDAR_CREATE)\s*:?\s*(\{.*\})\s*$", re.IGNORECASE)
CAL_DELETE_TAG_RE = re.compile(r"^\s*CALENDAR_DELETE\s*:?\s*(\{.*\})\s*$", re.IGNORECASE)
CAL_UPDATE_TAG_RE = re.compile(r"^\s*(CALENDAR_UPDATE|CALENDAR_EDIT)\s*:?\s*(\{.*\})\s*$", re.IGNORECASE)


def _current_datetime_prompt() -> str:
    now = datetime.datetime.now().astimezone()
    offset = now.strftime('%z')
    offset = offset[:3] + ":" + offset[3:] if len(offset) == 5 else offset
    now_iso = now.strftime(f"%Y-%m-%dT%H:%M:%S{offset}")
    tz_name = now.tzname() or "local"
    return (
        "Today's datetime is " + now_iso + f" ({tz_name}). Resolve all relative dates/times against this. "
        "Use future dates when ambiguous (e.g., if 'Friday' today has passed, use next Friday). "
        "Always output ISO 8601 with year, month, day, and time; include timezone offset when known."
    )


def _attempt_parse_json_from_buffer(buf: str):
    s = buf.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
    if s.endswith("```"):
        s = s.rstrip("`")
    try:
        return json.loads(s)
    except Exception:
        pass
    if '{' in s and '}' in s:
        start = s.find('{')
        end = s.rfind('}')
        if end > start:
            candidate = s[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None


def parse_calendar_tags(text: str):
    if not text:
        return text, []
    lines = text.splitlines()
    actions = []
    kept = []
    i = 0
    while i < len(lines):
        line = lines[i]
        matched = False
        for kind, rx, group_idx in (
            ("create", CAL_CREATE_TAG_RE, 2),
            ("delete", CAL_DELETE_TAG_RE, 1),
            ("update", CAL_UPDATE_TAG_RE, 2),
        ):
            m = rx.match(line)
            if m:
                matched = True
                payload_buf = m.group(group_idx)
                payload = _attempt_parse_json_from_buffer(payload_buf)
                j = i
                # Accumulate subsequent lines if needed (simple multi-line JSON)
                while payload is None and j + 1 < len(lines) and (j - i) < 20:
                    j += 1
                    payload_buf += "\n" + lines[j]
                    payload = _attempt_parse_json_from_buffer(payload_buf)
                if payload is not None:
                    actions.append((kind, payload))
                i = j + 1
                break
        # Fallback: tolerate bare CALENDAR_CREATE with an ISO-ish datetime token
        if not matched:
            m_simple = re.match(r"^\s*CALENDAR_CREATE\b\s*(.+)?$", line, flags=re.IGNORECASE)
            if m_simple:
                remainder = (m_simple.group(1) or "").strip()
                # Take first token as the datetime candidate
                token = remainder.split()[0] if remainder else ""
                if token:
                    # Accept 'Z' or offset; leave validation to service
                    actions.append(("create", {"start": token}))
                    matched = True
                    i += 1
            
        if not matched:
            kept.append(line)
            i += 1
    cleaned = "\n".join(kept).strip()
    return cleaned, actions


def _format_event_list_with_ids(items):
    lines = ["Candidates:"]
    for e in items:
        bits = [f"id={e.id}", e.title]
        if e.start:
            bits.append(f"on {_speakable_datetime(e.start)}")
        if e.location:
            bits.append(f"in {e.location}")
        lines.append(" - " + ", ".join(bits))
    return "\n".join(lines)


def _guess_title_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer quoted phrases
    m = re.search(r'"([^"]+)"', text)
    if not m:
        m = re.search(r"'([^']+)'", text)
    if m:
        val = (m.group(1) or "").strip()
        return val if val else None
    # Heuristic patterns like: "add a party to my calendar", "schedule dinner for ..."
    patterns = [
        r"\b(?:add|schedule|create|put|set)\s+(?:a|an|the\s+)?([^\n]+?)(?:\s+to\s+(?:my\s+)?calendar|\s+for\b)",
        r"\b(?:add|schedule|create|put|set)\s+(?:a|an|the\s+)?(.+)$",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            cand = (m.group(1) or "").strip()
            # Trim trailing common time/location prepositions if they slipped in
            cand = re.sub(r"\s+(?:on|at|for|tomorrow|today)\b.*$", "", cand, flags=re.IGNORECASE).strip()
            # Clip very long candidates
            if 1 <= len(cand) <= 80:
                return cand
    return None


def _append_note(existing: Optional[str], new_note: str) -> str:
    ex = (existing or "").strip()
    nn = (new_note or "").strip()
    if not ex:
        return nn
    # Deduplicate simple cases
    if nn.lower() in ex.lower():
        return ex
    # Strip trailing punctuation for smoother joins
    ex_clean = re.sub(r"[\s\.;,!]+$", "", ex)
    nn_clean = re.sub(r"^[\s\-:]+", "", nn)
    nn_clean = re.sub(r"[\s\.;,!]+$", "", nn_clean)
    # Prefer "and" for short notes, otherwise "; "
    if len(ex_clean) <= 60 and len(nn_clean) <= 60 and " and " not in ex_clean.lower():
        return f"{ex_clean} and {nn_clean}"
    return f"{ex_clean}; {nn_clean}"


class CalendarAgent:
    def __init__(self, service: CalendarService, client: Optional[OpenAI] = None, model: Optional[str] = None):
        self.service = service
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_CAL_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ---- Execution helpers ----
    def _get_current_month_events(self):
        try:
            now = datetime.datetime.now().astimezone()
            # First day of current month at 00:00:00
            start_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # First day of next month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            # End of current month is just before next_month
            end_dt = next_month - datetime.timedelta(microseconds=1)
            start_iso = start_dt.isoformat()
            end_iso = end_dt.isoformat()
            events = list(self.service.range(start_iso, end_iso))
        except Exception:
            events = []
        # Sort by start where possible
        from datetime import datetime as _dt
        local_tz = datetime.datetime.now().astimezone().tzinfo
        def _key(ev):
            try:
                d = _dt.fromisoformat((ev.start or "").replace("Z", "+00:00"))
                if d.tzinfo is None:
                    d = d.replace(tzinfo=local_tz)
                else:
                    d = d.astimezone(local_tz)
                return d.timestamp()
            except Exception:
                # Push unparseable to the end
                return float('inf')
        return sorted(events, key=_key)

    def _format_calendar_context(self, events) -> str:
        if not events:
            return "(No events this month)"
        # Title reflects current month context
        now = datetime.datetime.now().astimezone()
        lines = [f"Current month ({now.strftime('%B %Y')}) events:"]
        for e in events:
            bits = [f"id={e.id}", e.title]
            if e.start:
                bits.append(f"on {_speakable_datetime(e.start)}")
            if e.location:
                bits.append(f"in {e.location}")
            lines.append(" - " + ", ".join(bits))
        return "\n".join(lines)

    def _handle_delete(self, d: dict) -> Optional[str]:
        try:
            event = None
            if d.get("id"):
                event = self.service.delete_by_id(d.get("id"))
            else:
                title = d.get("title") or (d.get("match") or {}).get("title")
                start = d.get("start") or (d.get("match") or {}).get("start")
                date_only = d.get("date") or (d.get("match") or {}).get("date")
                if title:
                    event = self.service.delete_by_title_start(title, start)
                    if not event:
                        cands = self.service.candidates_by_title(title, date_only=date_only)
                        if len(cands) == 1:
                            event = self.service.delete_by_id(cands[0].id)
                        elif len(cands) > 1:
                            return "Ambiguous calendar delete. " + _format_event_list_with_ids(cands)
            if not event:
                return "No matching calendar event to delete."
            parts = [event.title]
            if event.start:
                parts.append(f"on {_speakable_datetime(event.start)}")
            if event.location:
                parts.append(f"in {event.location}")
            return "Deleted event: " + ", ".join(parts)
        except Exception:
            return ""

    def _handle_update(self, u: dict) -> Optional[str]:
        try:
            set_fields = u.get("set") or {k: v for k, v in u.items() if k in {"title", "start", "end", "timezone", "location", "notes", "attendees"}}
            if not set_fields:
                return ""

            # Resolve target event first (without mutating it yet)
            target = None
            if u.get("id"):
                target = self.service.get_by_id(u.get("id"))
            else:
                title = u.get("title") or (u.get("match") or {}).get("title")
                start = u.get("start") or (u.get("match") or {}).get("start")
                date_only = u.get("date") or (u.get("match") or {}).get("date")
                if title:
                    target = self.service.find_by_title_start(title, start)
                    if not target:
                        cands = self.service.candidates_by_title(title, date_only=date_only)
                        if len(cands) == 1:
                            target = cands[0]
                        elif len(cands) > 1:
                            return "Ambiguous calendar update. " + _format_event_list_with_ids(cands)

            if not target:
                return "No matching calendar event to update."

            # Handle notes append semantics
            new_notes = None
            if isinstance(set_fields, dict) and "notes" in set_fields:
                new_notes = set_fields.get("notes")
            # Allow explicit mode flags
            notes_mode = (u.get("notes_mode")
                          or (u.get("set") or {}).get("notes_mode")
                          or (set_fields.pop("notes_mode", None)))
            if new_notes:
                if (notes_mode == "append") or (notes_mode is None and (target.notes or "").strip()):
                    set_fields = dict(set_fields)
                    set_fields["notes"] = _append_note(target.notes, str(new_notes))

            # Persist update now that fields are finalized
            event = self.service.update_by_id(target.id, set_fields)
            if not event:
                return "No matching calendar event to update."
            parts = [event.title]
            if event.start:
                parts.append(f"on {_speakable_datetime(event.start)}")
            if event.location:
                parts.append(f"in {event.location}")
            return "Updated event: " + ", ".join(parts)
        except Exception:
            return ""

    # ---- Model call ----
    def _call(self, messages: List[dict]) -> str:
        data = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=512,
        )
        return (data.choices[0].message.content or "").strip()

    def handle(self, user_input: str) -> str:
        """Handle a single user calendar request using a calendar-specialized model with full calendar context."""
        events = self._get_current_month_events()
        sys_prompt = (
            "You are a specialized calendar agent."
            " You ONLY help with creating, reading, updating, or deleting calendar events.\n\n"
            + _current_datetime_prompt() + "\n\n"
            "You are provided all events for the current month below as context."
            " Do NOT use CALENDAR_QUERY. Instead, reason over the provided context and either:\n"
            "- answer the user concisely from the context, or\n"
            "- append exactly one CALENDAR_CREATE/UPDATE/DELETE tool line at the very end to make a change.\n"
            "Tool line format (strict):\n"
            "  CALENDAR_CREATE: {\"title\": \"...\", \"start\": \"YYYY-MM-DDTHH:MM[:SS][±HH:MM]\", \"location\": \"...\", \"notes\": \"...\"}\n"
            "  CALENDAR_UPDATE: {\"id\": \"...\" or \"match\": {\"title\": \"...\", \"date\": \"YYYY-MM-DD\"}, \"set\": {\"start\": \"...\", ...}}\n"
            "  CALENDAR_DELETE: {\"id\": \"...\"} or {\"title\": \"...\", \"start\": \"...\"}\n"
            "When the user asks to add a note to an existing event, include \"notes\" in set and also set \"notes_mode\": \"append\" so notes are appended, not replaced.\n"
            "Only one tool line. Do not output bare CALENDAR_* without JSON.\n"
            "Keep your natural-language reply to ONE short sentence before any tool line. If you include a CALENDAR_* tool line, keep the sentence brief and do not restate the exact details; the system will announce the change."
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "system", "content": self._format_calendar_context(events)},
            {"role": "user", "content": user_input},
        ]
        assistant_response = self._call(messages)
        cleaned, actions = parse_calendar_tags(assistant_response)
        natural_text = cleaned

        confirmations: List[str] = []
        for kind, payload in actions:
            if kind == "create":
                try:
                    payload = dict(payload or {})
                    if not payload.get("title"):
                        guess = _guess_title_from_text(user_input) or _guess_title_from_text(assistant_response)
                        if guess:
                            payload["title"] = guess
                    saved = self.service.add_event(payload)
                    parts = [saved.title]
                    if saved.start:
                        parts.append(f"on {_speakable_datetime(saved.start)}")
                    if saved.location:
                        parts.append(f"at {saved.location}")
                    confirmations.append(f"(Added to your calendar: {', '.join(parts)})")
                except Exception:
                    pass
            elif kind == "delete":
                info = self._handle_delete(payload)
                if info:
                    confirmations.append(info)
            elif kind == "update":
                info = self._handle_update(payload)
                if info:
                    confirmations.append(info)
        # Prefer a single, definitive confirmation over repeating the model's intent
        if confirmations:
            return "\n\n".join(confirmations)
        # If no tool line emitted, just return the concise answer
        return natural_text

# ---- Speakable date/time formatting ----
def _ordinal(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return str(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _speakable_datetime(iso_str: str) -> str:
    """Convert ISO-like datetime to a friendly, read-aloud string in local time.

    Examples:
    - "today at 3 pm"
    - "tomorrow at 9:30 am"
    - "Wednesday at 11 am"
    - "September 10th at 3:30 pm"
    - "September 10th, 2026 at 3 pm" (if different year)
    """
    if not iso_str:
        return ""
    try:
        raw = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return iso_str

    local_tz = datetime.datetime.now().astimezone().tzinfo
    if raw.tzinfo is None:
        dt = raw.replace(tzinfo=local_tz)
    else:
        dt = raw.astimezone(local_tz)

    now = datetime.datetime.now().astimezone()
    today = now.date()
    target = dt.date()
    delta_days = (target - today).days

    # Day phrase
    if delta_days == 0:
        day_phrase = "today"
    elif delta_days == 1:
        day_phrase = "tomorrow"
    elif delta_days == -1:
        day_phrase = "yesterday"
    elif 2 <= delta_days <= 6:
        day_phrase = dt.strftime("%A")
    elif -6 <= delta_days <= -2:
        day_phrase = "last " + dt.strftime("%A")
    else:
        if dt.year == now.year:
            day_phrase = f"{dt.strftime('%B')} {_ordinal(dt.day)}"
        else:
            day_phrase = f"{dt.strftime('%B')} {_ordinal(dt.day)}, {dt.year}"

    # Time phrase (12-hour, omit :00)
    hour = dt.hour
    minute = dt.minute
    ampm = "am" if hour < 12 else "pm"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    if minute == 0:
        time_phrase = f"{hour12} {ampm}"
    else:
        time_phrase = f"{hour12}:{minute:02d} {ampm}"

    return f"{day_phrase} at {time_phrase}"
