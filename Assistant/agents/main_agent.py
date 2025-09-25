from __future__ import annotations

import json
from typing import Any, Dict

from Assistant.agents.calendar_agent import CalendarAgent
from Assistant.agents.memory_agent import MemoryAgent
from Assistant.core.tools import ToolBox


class MainAgent:
    """Top-level agent exposing calendar and memory agents as callable tools."""

    def __init__(self, calendar_agent: CalendarAgent, memory_agent: MemoryAgent) -> None:
        self.calendar_agent = calendar_agent
        self.memory_agent = memory_agent
        self.toolbox = ToolBox(namespace="agent")
        self._register_tools()

    @staticmethod
    def _parse_json(arguments_json: str | None) -> Dict[str, Any]:
        if not arguments_json:
            return {}
        try:
            data = json.loads(arguments_json)
            if not isinstance(data, dict):
                raise ValueError("Payload must decode to an object")
            return data
        except Exception as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

    def _register_tools(self) -> None:
        @self.toolbox.tool(description="Delegate a natural-language request to the calendar agent.")
        def calendar(request: str, payload_json: str | None = None) -> Dict[str, Any]:
            payload = self._parse_json(payload_json)
            return self.calendar_agent.run(request, payload or None)

        @self.toolbox.tool(description="Call the memory agent with a specific action.")
        def memory(action: str = "list_facts", payload_json: str | None = None) -> Dict[str, Any]:
            payload = self._parse_json(payload_json)
            return self.memory_agent.run(action, payload or None)

        @self.toolbox.tool(description="List available capabilities for each sub-agent.")
        def list_capabilities() -> Dict[str, Any]:
            return {
                "ok": True,
                "calendar": self.calendar_agent.capabilities(),
                "memory": getattr(self.memory_agent, "capabilities", self.memory_agent.toolbox.handlers().keys())
                if hasattr(self.memory_agent, "capabilities")
                else list(self.memory_agent.toolbox.handlers().keys()),
            }

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        func = self.toolbox.get_tool_function(tool_name)
        if func is None:
            raise ValueError(f"Unknown agent tool: {tool_name}")
        return func(**arguments)


__all__ = [
    "MainAgent",
]
