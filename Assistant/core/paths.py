from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CALENDAR_EVENTS_PATH = DATA_DIR / "calendar_events.json"
FACTS_PATH = DATA_DIR / "facts.json"

__all__ = [
    "DATA_DIR",
    "CALENDAR_EVENTS_PATH",
    "FACTS_PATH",
]
