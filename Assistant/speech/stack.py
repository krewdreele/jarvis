from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore
try:  # pragma: no cover
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore
try:  # pragma: no cover
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore
try:  # pragma: no cover
    import speech_recognition as sr  # type: ignore
except Exception:
    sr = None  # type: ignore
try:  # pragma: no cover
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None  # type: ignore
try:  # pragma: no cover
    from openwakeword.model import Model  # type: ignore
except Exception:
    Model = None  # type: ignore


LOG_DIR = "interaction_logs"
os.makedirs(LOG_DIR, exist_ok=True)

tts_engine = None
whisper_model = None
recognizer = None
mic = None
wake_model = None
stream = None
wake_word_detected = False
wake_word_suppressed = False
_debug = bool(os.getenv("SPEECH_DEBUG"))


def log_interaction(session_id: str, user_input: str, assistant_output: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "assistant_output": assistant_output,
    }
    with open(os.path.join(LOG_DIR, f"{session_id}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _audio_callback(indata, frames, time_info, status):  # pragma: no cover
    global wake_word_detected, wake_word_suppressed
    if wake_word_suppressed:
        return
    try:
        if isinstance(indata, np.ndarray):
            samples = indata
            if samples.ndim == 2 and samples.shape[1] >= 1:
                samples = samples[:, 0]
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        prediction = wake_model.predict(samples)
        # Some model versions expose different keys; trigger on the highest score
        if isinstance(prediction, dict) and prediction:
            # Print top score in debug mode
            if _debug:
                try:
                    top_key = max(prediction, key=prediction.get)
                    print(f"[wake] top={top_key} score={prediction.get(top_key):.3f}")
                except Exception:
                    pass
            score = prediction.get("hey_jarvis") or max(prediction.values())
            if score and score > 0.5:
                wake_word_detected = True
                print("Wake word detected!")
    except Exception:
        pass


def init_speech_stack() -> None:
    global tts_engine, whisper_model, recognizer, mic, wake_model, stream
    if pyttsx3 is not None:
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 150)
        except Exception:
            tts_engine = None
    if whisper is not None:
        try:
            whisper_model = whisper.load_model("base")
        except Exception:
            whisper_model = None
    if sr is not None:
        try:
            recognizer = sr.Recognizer()
            recognizer.pause_threshold = 1.0
            recognizer.non_speaking_duration = 0.6
            recognizer.phrase_threshold = 0.3
            global mic
            mic = sr.Microphone()
        except Exception:
            pass
    if Model is not None and sd is not None:
        try:
            global wake_model, stream
            wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            stream = sd.InputStream(callback=_audio_callback, channels=1, samplerate=16000, dtype='int16')
            stream.start()
        except Exception:
            pass
    try:
        if recognizer is not None and mic is not None:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            recognizer.dynamic_energy_threshold = False
    except Exception:
        pass
    # Print a concise status summary so users know what's active
    try:
        status = {
            "TTS": "ON" if tts_engine is not None else "OFF",
            "STT": "ON" if (recognizer is not None and mic is not None and whisper_model is not None) else "OFF",
            "Wake": "ON" if (wake_model is not None and stream is not None) else "OFF",
        }
        print(f"Speech stack -> TTS:{status['TTS']} STT:{status['STT']} Wake:{status['Wake']}")
        if status["Wake"] == "OFF":
            print("Tip: wake word disabled; listening starts immediately.")
    except Exception:
        pass


def transcribe_audio(timeout: int = 5, phrase_time_limit: int = 20) -> Optional[str]:  # pragma: no cover
    try:
        if recognizer is None or mic is None or whisper_model is None:
            return ""
        with mic as source:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        data = audio.get_wav_data()
        tmp = os.path.join(os.getcwd(), "temp.wav")
        with open(tmp, "wb") as f:
            f.write(data)
        result = whisper_model.transcribe(tmp)
        return (result.get("text") or "").strip()
    except Exception as e:
        print(f"STT error or timeout: {e}")
        return None


def speak(text: str) -> None:  # pragma: no cover
    global wake_word_suppressed, stream
    if not tts_engine:
        return
    wake_word_suppressed = True
    try:
        try:
            if stream is not None:
                stream.stop()
        except Exception:
            pass
        tts_engine.say(text)
        tts_engine.runAndWait()
    finally:
        try:
            if stream is not None:
                stream.start()
        except Exception:
            pass
        wake_word_suppressed = False


def wait_for_wake_word(max_wait_seconds: float = 8.0) -> None:  # pragma: no cover
    """Wait for wake word, but fail open after a short timeout.

    If wake detection isn't available or doesn't trigger within max_wait_seconds,
    proceed to listening so the experience doesn't stall silently.
    Pass max_wait_seconds<=0 to wait indefinitely.
    """
    global wake_word_detected
    wake_word_detected = False
    # If no wake model/stream, don't block
    if wake_model is None or stream is None:
        return
    print("Waiting for wake word...")
    import time as _t
    start = _t.time()
    while not wake_word_detected:
        _t.sleep(0.05)
        if max_wait_seconds and (start + max_wait_seconds) <= _t.time():
            print("Wake timeout; starting to listen.")
            break
