import os
import json
import datetime
import uuid
import time
import argparse
import re
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from typing import Optional, Tuple

try:
    from .calendar_service import CalendarService  # when run as a package
    from .calendar_agent import CalendarAgent
except Exception:  # pragma: no cover
    from calendar_service import CalendarService  # when run as a script
    from calendar_agent import CalendarAgent

# Audio/TTS/STT deps (initialized lazily for speech mode)
import pyttsx3
import whisper
import speech_recognition as sr
import sounddevice as sd
from openwakeword.model import Model

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()

# === Configuration ===
LOG_DIR = "interaction_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Calendar persistence
CALENDAR_PATH = os.path.join("Assistant", "calendar_events.json")
calendar_service = CalendarService(CALENDAR_PATH)
calendar_agent = CalendarAgent(calendar_service, client=client, model=os.getenv("OPENAI_CAL_MODEL", OPENAI_MODEL))


# === Helper: Convert to OpenAI message schema ===
def _to_openai_messages(chat_history):
    role_map = {"System": "system", "User": "user", "Chatbot": "assistant"}
    messages = []
    for entry in chat_history:
        role = role_map.get(entry.get("role"), str(entry.get("role", "user")).lower())
        content = entry.get("message", "") or ""
        messages.append({"role": role, "content": content})
    return messages


def call_openai_api(chat_history):
    try:
        if not OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        data = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=_to_openai_messages(chat_history),
            temperature=0.7,
            max_tokens=512,
        )
        return data.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call error: {e}")
        return "Sorry, I had trouble reaching the server."


# Note: Calendar tool parsing and execution moved into CalendarAgent.


def _current_datetime_prompt() -> str:
    now = datetime.datetime.now().astimezone()
    # Format offset as "+HH:MM"
    offset = now.strftime('%z')
    offset = offset[:3] + ":" + offset[3:] if len(offset) == 5 else offset
    now_iso = now.strftime(f"%Y-%m-%dT%H:%M:%S{offset}")
    tz_name = now.tzname() or "local"
    return (
        "Today's datetime is " + now_iso + f" ({tz_name}). Resolve all relative dates/times against this. "
        "Use future dates when ambiguous (e.g., if 'Friday' today has passed, use next Friday). "
        "Always output ISO 8601 with year, month, day, and time; include timezone offset when known."
    )


def _looks_like_calendar_intent(text: str) -> bool:
    """Legacy heuristic keyword check for calendar intent.

    Kept as a safe fallback if the LLM-based router is unavailable.
    """
    if not text:
        return False
    t = text.lower()
    keywords = [
        'calendar', 'schedule', 'meeting', 'event', 'appointment', 'remind', 'reminder',
        'add to', 'add a note', 'note to', 'reschedule', 'move', 'cancel', 'delete', 'update',
        'today', 'this week', 'tonight', 'next week'
    ]
    return any(k in t for k in keywords)


def _route_intent_with_llm(user_input: str, history: Optional[list] = None) -> str:
    """Ask the base assistant model to route the request.

    Returns 'CALENDAR' when the request should be handled by the calendar agent,
    otherwise returns 'GENERAL'. Uses deterministic settings and a strict output
    contract. Falls back to 'GENERAL' on any error.
    """
    try:
        router_messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent router. Decide if the user's latest request "
                    "requires calendar operations such as creating, listing, updating, "
                    "rescheduling, or deleting events; adding notes to events; asking "
                    "about someone's availability; or setting reminders. "
                    + "\n\nRespond with exactly one token: CALENDAR or GENERAL. "
                      "Do not include punctuation or explanations."
                ),
            },
        ]
        if history:
            # Include only the last user utterance as context if provided
            try:
                last_user = next((h for h in reversed(history) if h.get("role") == "User"), None)
                if last_user:
                    router_messages.append({"role": "user", "content": last_user.get("message", "")})
            except Exception:
                pass
        # Always include the current input explicitly
        router_messages.append({"role": "user", "content": user_input})

        result = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=router_messages,
            temperature=0,
            max_tokens=2,
        )
        decision = (result.choices[0].message.content or "").strip().upper()
        if "CAL" in decision:  # tolerate variants like "CALENDAR"
            return "CALENDAR"
        return "GENERAL"
    except Exception as e:
        # On any failure, be conservative and fall back to heuristic
        try:
            return "CALENDAR" if _looks_like_calendar_intent(user_input) else "GENERAL"
        except Exception:
            return "GENERAL"


def log_interaction(session_id, user_input, assistant_output):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_input": user_input,
        "assistant_output": assistant_output,
    }
    with open(os.path.join(LOG_DIR, f"{session_id}.jsonl"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")


"""
Speech mode initialization and utilities
"""

# Globals (lazy-initialized)
tts_engine = None
whisper_model = None
recognizer = None
mic = None
wake_model = None
stream = None

# Global Flags
wake_word_detected = False
wake_word_suppressed = False


def audio_callback(indata, frames, time_info, status):
    global wake_word_detected, wake_word_suppressed
    # Ignore wake-word inference while we are speaking to avoid false triggers
    if wake_word_suppressed:
        return
    audio_data = np.frombuffer(indata, dtype=np.int16)
    prediction = wake_model.predict(audio_data)
    if prediction.get("hey_jarvis", 0) > 0.5:
        wake_word_detected = True
        print("Wake word detected!")


def init_speech_stack():
    global tts_engine, whisper_model, recognizer, mic, wake_model, stream

    # TTS Setup
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 150)
    except Exception as e:
        print(f"TTS init failed: {e}")
        tts_engine = None

    # Whisper STT Setup
    whisper_model = whisper.load_model("base")
    recognizer = sr.Recognizer()
    # Tune STT to avoid premature cutoffs
    recognizer.pause_threshold = 1.0           # require ~1.0s silence to end phrase
    recognizer.non_speaking_duration = 0.6     # tolerate brief pauses while speaking
    recognizer.phrase_threshold = 0.3          # ignore very short spurts
    mic = sr.Microphone()

    # Wake Word Detection
    wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, dtype='int16')
    stream.start()

    # Calibrate ambient noise once at startup for faster subsequent listens
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        # Freeze threshold after calibration for stability
        recognizer.dynamic_energy_threshold = False
    except Exception as e:
        print(f"Ambient calibration skipped: {e}")


def transcribe_audio(timeout=5, phrase_time_limit=20):
    try:
        with mic as source:
            print(f"\u000f Listening for user input (timeout={timeout}s)...")
            # Do not re-calibrate every time; speeds up start-of-speech capture
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

        audio_data = audio.get_wav_data()
        with open("temp.wav", "wb") as f:
            f.write(audio_data)

        result = whisper_model.transcribe("temp.wav")
        return result["text"]
    except Exception as e:
        print(f"STT error or timeout: {e}")
        return None


def speak(text):
    global wake_word_suppressed, stream
    if not tts_engine:
        # Captions are printed by callers in speech mode; avoid duplicates here
        return
    # Suppress wakeword during TTS and temporarily stop stream to avoid buffering
    wake_word_suppressed = True
    try:
        try:
            if stream is not None:
                stream.stop()
        except Exception:
            pass

        tts_engine.say(text)
        tts_engine.runAndWait()  # blocks until TTS completes
    finally:
        # Small grace to let playback tail settle
        time.sleep(0.1)
        try:
            if stream is not None:
                stream.start()
        except Exception:
            pass
        wake_word_suppressed = False


def wait_for_wake_word():
    global wake_word_detected
    wake_word_detected = False
    print("Waiting for wake word...")
    while not wake_word_detected:
        time.sleep(0.1)


# === Main Loops ===
def speech_main():
    session_id = str(uuid.uuid4())
    chat_history = [{
        "role": "System",
        "message": (
            "You are JARVIS, a helpful, witty, british personal assistant. Be concise.\n\n"
            + _current_datetime_prompt()
        ),
    }]

    while True:
        wait_for_wake_word()
        print("Assistant: Yes?")
        speak("Yes?")
        user_input = transcribe_audio(timeout=5, phrase_time_limit=20)
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye.")
            speak("Goodbye.")
            break

        while user_input:
            print(f"You: {user_input}")
            chat_history.append({"role": "User", "message": user_input})

            route = _route_intent_with_llm(user_input, chat_history)
            if route == "CALENDAR":
                assistant_response = calendar_agent.handle(user_input)
            else:
                assistant_response = call_openai_api(chat_history)
            chat_history.append({"role": "Chatbot", "message": assistant_response})
            print(f"Assistant: {assistant_response}")
            speak(assistant_response)
            log_interaction(session_id, user_input, assistant_response)

            print("\u0007' Waiting briefly for follow-up (10s)...")
            user_input = transcribe_audio(timeout=5, phrase_time_limit=20)
            if not user_input:
                print("No follow-up. Returning to wake mode.")
                break


def text_main():
    session_id = str(uuid.uuid4())
    chat_history = [{
        "role": "System",
        "message": (
            "You are JARVIS, a helpful, witty, british personal assistant. Be concise.\n\n"
            + _current_datetime_prompt()
        ),
    }]

    print("Text mode active. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye.")
            break

        chat_history.append({"role": "User", "message": user_input})
        route = _route_intent_with_llm(user_input, chat_history)
        if route == "CALENDAR":
            assistant_response = calendar_agent.handle(user_input)
        else:
            assistant_response = call_openai_api(chat_history)
        chat_history.append({"role": "Chatbot", "message": assistant_response})
        print(f"Assistant: {assistant_response}")
        log_interaction(session_id, user_input, assistant_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jarvis assistant in speech or text mode")
    parser.add_argument("--mode", "-m", choices=["speech", "text"], default="speech", help="Interaction mode")
    args = parser.parse_args()

    if args.mode == "speech":
        init_speech_stack()
        speech_main()
    else:
        text_main()
