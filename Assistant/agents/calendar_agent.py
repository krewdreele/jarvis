from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from Assistant.core.tools import ToolBox
from Assistant.tools import calendar as calendar_tools


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _split_attendees(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    items: list[str] = []
    for part in raw.replace(";", ",").split(","):
        name = part.strip()
        if name:
            items.append(name)
    return items


def _token_arg_for_model(model: str, tokens: int) -> Dict[str, int]:
    m = (model or "").lower()
    if m.startswith("gpt-5"):
        return {"max_completion_tokens": tokens}
    return {"max_tokens": tokens}


def _normalize_tool_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if result is None:
        return {"ok": True, "result": None}
    if isinstance(result, (str, int, float, bool)):
        return {"ok": True, "result": result}
    try:
        return {"ok": True, "result": json.loads(json.dumps(result, default=str))}
    except Exception:
        return {"ok": True, "result": str(result)}


class CalendarAgent:
    """Autonomous calendar specialist that can call calendar tools via its own LLM."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.client = client or self._create_client()
        self.model = model or os.getenv("CALENDAR_AGENT_MODEL", "gpt-4o-mini")
        self.max_tool_rounds = max_tool_rounds
        self.toolbox = ToolBox(namespace="calendar")
        self._register_tools()

    @staticmethod
    def _create_client() -> Any | None:
        if OpenAI is None:  # pragma: no cover
            return None
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        try:
            if base_url:
                return OpenAI(base_url=base_url, api_key=api_key) if api_key else OpenAI(base_url=base_url)
            return OpenAI(api_key=api_key) if api_key else OpenAI()
        except Exception:  # pragma: no cover
            return None

    def capabilities(self) -> list[str]:
        return [schema["function"]["name"] for schema in self.toolbox.tools]

    def run(self, request: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "error": "calendar agent client not configured"}

        now_local = datetime.now().astimezone()
        messages: list[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are the calendar specialist. Read the user request and decide how to use the"
                    " available calendar tools. Only call tools when necessary to gather data or apply"
                    " changes. Provide a short, clear response summarizing the outcome."
                ),
            },
            {
                "role": "system",
                "content": f"Current datetime: {now_local.isoformat()} (tz: {now_local.tzname()})",
            },
        ]
        if payload:
            payload_str = json.dumps(payload, ensure_ascii=False)
            messages.append({
                "role": "system",
                "content": f"Additional JSON context: {payload_str}",
            })
        messages.append({"role": "user", "content": request})

        last_tool_output: Dict[str, Any] | None = None
        for _ in range(self.max_tool_rounds):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.toolbox.tools,
                    tool_choice="auto",
                    **_token_arg_for_model(self.model, 1500),
                )
            except Exception as exc:
                return {"ok": False, "error": f"calendar agent request failed: {exc}"}

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                })
                for call in tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except Exception:
                        args = {}
                    try:
                        raw_result = self.toolbox.invoke(name, args)
                    except Exception as exc:
                        raw_result = {"ok": False, "error": str(exc)}
                    tool_result = _normalize_tool_result(raw_result)
                    last_tool_output = tool_result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                continue

            final_text = (msg.content or "").strip()
            if final_text:
                return {"ok": True, "response": final_text, "tool_output": last_tool_output}
            if last_tool_output is not None:
                return {"ok": True, "response": json.dumps(last_tool_output), "tool_output": last_tool_output}
            break

        return {"ok": False, "error": "calendar agent could not complete the request"}

    def _register_tools(self) -> None:
        @self.toolbox.tool(description="Create a calendar event.")
        def add_event(
            title: str,
            start: str,
            end: str | None = None,
            timezone: str | None = None,
            location: str | None = None,
            notes: str | None = None,
            attendees: str | None = None,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "title": title,
                "start": start,
            }
            if end:
                payload["end"] = end
            if timezone:
                payload["timezone"] = timezone
            if location:
                payload["location"] = location
            if notes:
                payload["notes"] = notes
            attendee_list = _split_attendees(attendees)
            if attendee_list:
                payload["attendees"] = attendee_list
            return calendar_tools.add_event(payload)

        @self.toolbox.tool(description="List calendar events, optionally by range.")
        def list_events(
            include_past: bool = False,
            start: str | None = None,
            end: str | None = None,
        ) -> Dict[str, Any]:
            if start and end:
                return calendar_tools.events_in_range(start=start, end=end)
            return calendar_tools.list_events(include_past=_as_bool(include_past))

        @self.toolbox.tool(description="List upcoming events.")
        def upcoming(limit: int = 10) -> Dict[str, Any]:
            return calendar_tools.upcoming(limit=int(limit))

        @self.toolbox.tool(description="List events within an ISO datetime range.")
        def events_in_range(start: str, end: str) -> Dict[str, Any]:
            return calendar_tools.events_in_range(start=start, end=end)

        @self.toolbox.tool(description="List events occurring on a given YYYY-MM-DD date.")
        def events_on_date(date: str) -> Dict[str, Any]:
            return calendar_tools.events_on_date(date=date)

        @self.toolbox.tool(description="List remaining events in the current week.")
        def this_week() -> Dict[str, Any]:
            return calendar_tools.this_week()

        @self.toolbox.tool(description="Update an existing event.")
        def update_event(
            id: str | None = None,
            title: str | None = None,
            start: str | None = None,
            new_title: str | None = None,
            new_start: str | None = None,
            new_end: str | None = None,
            new_timezone: str | None = None,
            new_location: str | None = None,
            new_notes: str | None = None,
            new_attendees: str | None = None,
        ) -> Dict[str, Any]:
            set_fields: Dict[str, Any] = {}
            if new_title is not None:
                set_fields["title"] = new_title
            if new_start is not None:
                set_fields["start"] = new_start
            if new_end is not None:
                set_fields["end"] = new_end
            if new_timezone is not None:
                set_fields["timezone"] = new_timezone
            if new_location is not None:
                set_fields["location"] = new_location
            if new_notes is not None:
                set_fields["notes"] = new_notes
            attendees_list = _split_attendees(new_attendees)
            if attendees_list:
                set_fields["attendees"] = attendees_list
            return calendar_tools.update_event(
                id=id,
                title=title,
                start=start,
                set_fields=set_fields or None,
            )

        @self.toolbox.tool(description="Delete an event by id or title.")
        def delete_event(
            id: str | None = None,
            title: str | None = None,
            start: str | None = None,
        ) -> Dict[str, Any]:
            return calendar_tools.delete_event(id=id, title=title, start=start)

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.toolbox.invoke(tool_name, arguments)

    @property
    def available(self) -> bool:
        return calendar_tools.AVAILABLE


__all__ = [
    "CalendarAgent",
]
