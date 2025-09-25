from __future__ import annotations

import os
import argparse
import json
import uuid
from typing import List, Dict, Any, Tuple
import sys
from datetime import datetime

# Ensure project root is on sys.path when running as a script (python Assistant/main.py)
_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Optional dotenv to mirror main.py behavior
try:  # pragma: no cover
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "The 'openai' package is required. Install with: pip install openai"
    )

# New modularized tools and paths
# Graph tools removed
from Assistant.speech.stack import (
    init_speech_stack,
    transcribe_audio,
    speak,
    wait_for_wake_word,
    log_interaction,
)  # type: ignore
from Assistant.agents.calendar_agent import CalendarAgent  # type: ignore
from Assistant.agents.memory_agent import MemoryAgent, review_and_save_from_log  # type: ignore
from Assistant.agents.main_agent import MainAgent  # type: ignore


def _token_arg_for_model(model: str, tokens: int) -> Dict[str, int]:
    """Return the correct token parameter name for the given model.

    Some newer models (e.g., gpt-5 family) expect 'max_completion_tokens'
    instead of 'max_tokens'.
    """
    m = (model or "").lower()
    # Adjust as needed for families that require different parameter names
    if m.startswith("gpt-5"):
        return {"max_completion_tokens": tokens}
    return {"max_tokens": tokens}


def chat_once(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> str:
    """Call the chat.completions API once and return assistant text.

    Absolute basics: single request, no streaming, no tools.
    """
    # Use the appropriate token parameter per model
    token_arg = _token_arg_for_model(model, 512)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **token_arg,
    )
    return (resp.choices[0].message.content or "").strip()




def _normalize_tool_result(value: Any) -> Dict[str, Any]:
    """Coerce arbitrary tool return values into JSON-serializable dicts."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {"ok": True, "result": None}
    if isinstance(value, (str, int, float, bool)):
        return {"ok": True, "result": value}
    return {"ok": True, "result": value}

def run_with_tools(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    handlers: Dict[str, Any],
    tool_box: Any | None = None,
    max_tool_rounds: int = 5,
    debug_tools: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run a minimal tool loop using provided tool handlers.

    Returns assistant_text and updated messages.
    """
    last_tool_results: Dict[str, Any] = {}
    for _ in range(max_tool_rounds):
        token_arg = _token_arg_for_model(model, 1500)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **token_arg,
        )
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
                if debug_tools:
                    print(f"[tool-call] {name} args={args}")
                tool_result = None
                # Prefer dynamic toolbox lookup when available
                if tool_box is not None:
                    func = getattr(tool_box, "get_tool_function", lambda _n: None)(name)
                    if func is not None:
                        try:
                            tool_result = func(**args)
                        except Exception as e:
                            tool_result = {"error": str(e)}
                if tool_result is None:
                    handler = handlers.get(name)
                    if not handler:
                        tool_result = {"error": f"Unknown tool: {name}"}
                    else:
                        try:
                            tool_result = handler(args)
                        except Exception as e:
                            tool_result = {"error": str(e)}
                last_tool_results[name] = tool_result
                tool_result = _normalize_tool_result(tool_result)
                if debug_tools:
                    preview = json.dumps(tool_result)
                    if len(preview) > 400:
                        preview = preview[:400] + "..."
                    print(f"[tool-result] {name} -> {preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
            continue
        # No tool calls: return final assistant text or synthesize from last tool results
        final_text = (msg.content or "").strip()
        return (final_text, messages)
    # Fallback if exceeded rounds
    return ("Sorry, I couldn't complete the tool interaction.", messages)

def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis assistant")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name to use (default: env OPENAI_MODEL or gpt-4o-mini)",
    )
    # System prompts are kept minimal and embedded; no --system override.
    parser.add_argument(
        "--mode",
        choices=["text", "speech"],
        default="text",
        help="Interaction mode (default: text)",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("ALT_USER_ID", "user_default"),
        help="User id for scoping graph/tool operations.",
    )
    dbg = parser.add_mutually_exclusive_group()
    dbg.add_argument("--debug-tools", action="store_true", default=False, help="Print tool calls/results (default off)")
    parser.add_argument(
        "--memory-review",
        dest="memory_review",
        action="store_true",
        default=True,
        help="Run a post-session memory review to save facts (default).",
    )
    parser.add_argument(
        "--no-memory-review",
        dest="memory_review",
        action="store_false",
        help="Disable post-session memory review.",
    )
    args = parser.parse_args()

    if load_dotenv:
        try:
            load_dotenv()
        except Exception:
            pass
        try:
            here = os.path.dirname(__file__)
            env_path = os.path.join(here, "config/.env")
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)
        except Exception:
            pass

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key) if api_key else OpenAI(base_url=base_url)
    else:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # Build toolboxes for each agent and expose the main agent tools
    calendar_agent = CalendarAgent(client=client)
    memory_agent = MemoryAgent(default_user_id=args.user_id)
    main_agent = MainAgent(calendar_agent, memory_agent)
    tools = list(main_agent.toolbox.tools)
    handlers = main_agent.toolbox.handlers()
    combo_box = main_agent.toolbox
    def _base_messages() -> List[Dict[str, Any]]:
        now_local = datetime.now().astimezone()
        msgs: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": f"Current datetime: {now_local.isoformat()} (tz: {now_local.tzname()})",
            },
            {
                "role": "system",
                "content": (
                    "You are a concise, friendly assistant. Keep answers short unless the user asks for detail."
                    " Use tools when helpful. Never expose raw tool outputs; summarize results naturally."
                ),
            },
        ]
        try:
            tool_names = ", ".join([(t.get("function") or {}).get("name") or "" for t in tools if (t.get("function") or {}).get("name")])
            if tool_names:
                msgs.append({
                    "role": "system",
                    "content": f"Tools available: {tool_names}.",
                })
        except Exception:
            pass

        return msgs

    if args.mode == "speech":
        print(f"Model: {args.model}")
        if base_url:
            print(f"Base URL: {base_url}")
        init_speech_stack()
        session_id = str(uuid.uuid4())
        messages: List[Dict[str, Any]] = _base_messages()
        try:
            while True:
                wait_for_wake_word()
                print("Assistant: Yes?")
                speak("Yes?")
                user = transcribe_audio(timeout=5, phrase_time_limit=20)
                if not user:
                    continue
                if isinstance(user, str) and user.lower() in {"exit", "quit"}:
                    print("Assistant: Goodbye.")
                    speak("Goodbye.")
                    break
                while user:
                    print(f"You: {user}")
                    messages.append({"role": "user", "content": user})
                    try:
                        if tools:
                            assistant, messages = run_with_tools(
                                client=client,
                                model=args.model,
                                messages=messages,
                                tools=tools, handlers=handlers, tool_box=combo_box, debug_tools=args.debug_tools,
                            )
                        else:
                            assistant = chat_once(client, args.model, messages)
                    except Exception as e:
                        print(f"Error: {e}")
                        break
                    messages.append({"role": "assistant", "content": assistant})
                    print(f"Assistant: {assistant}\n")
                    speak(assistant)
                    log_interaction(session_id, user, assistant)
                    user = transcribe_audio(timeout=5, phrase_time_limit=20)
                    if not user:
                        print("No follow-up. Returning to wake mode.")
                        break
        except KeyboardInterrupt:
            pass
        # Post-session memory review
        try:
            if args.memory_review:
                summary = review_and_save_from_log(
                    session_id=session_id,
                    user_id=args.user_id,
                    client=client,
                )
                print(f"[memory] review summary: {summary}")
        except Exception as e:
            print(f"[memory] review error: {e}")
        return

    # Text mode
    messages: List[Dict[str, Any]] = _base_messages()
    session_id = str(uuid.uuid4())
    print(f"Model: {args.model}")
    if base_url:
        print(f"Base URL: {base_url}")
    print("Type 'exit' or press Ctrl+C to quit.\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            messages.append({"role": "user", "content": user})
            try:
                if tools:
                    assistant, messages = run_with_tools(
                        client=client,
                        model=args.model,
                        messages=messages,
                        tools=tools, handlers=handlers, tool_box=combo_box, debug_tools=args.debug_tools,
                    )
                else:
                    assistant = chat_once(client, args.model, messages)
            except Exception as e:
                print(f"Error: {e}")
                continue
            messages.append({"role": "assistant", "content": assistant})
            print(f"Assistant: {assistant}\n")
            try:
                log_interaction(session_id, user, assistant)
            except Exception:
                pass
    except KeyboardInterrupt:
        pass

    # Post-session memory review (text mode)
    try:
        if args.memory_review:
            summary = review_and_save_from_log(
                session_id=session_id,
                user_id=args.user_id,
                client=client,
            )
            print(f"[memory] review summary: {summary}")
    except Exception as e:
        print(f"[memory] review error: {e}")

if __name__ == "__main__":
    main()




