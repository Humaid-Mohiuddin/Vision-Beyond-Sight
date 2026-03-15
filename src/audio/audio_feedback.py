"""
This module is designed to leverage multithreading for real-time audio guidance without
interruptions to the vision module.
"""


import pyttsx3
import threading
import time

_latest_text = None
_lock = threading.Lock()
_wakeup = threading.Event()
_stop = threading.Event()

def _audio_worker():
    cooldown = 2.0   # repeat every 2 seconds
    last_spoken_time = 0

    while not _stop.is_set():
        _wakeup.wait(timeout=0.1)   # small timeout so loop keeps checking

        with _lock:
            text = _latest_text

        if text is None:
            continue

        current_time = time.time()

        # Speak if cooldown expired
        if current_time - last_spoken_time >= cooldown:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

            last_spoken_time = current_time


_thread = threading.Thread(target=_audio_worker, daemon=True)

def start_audio():
    if not _thread.is_alive():
        _thread.start()

def speak_command(text: str):
    global _latest_text
    with _lock:
        _latest_text = text
    _wakeup.set()

def stop_audio():
    _stop.set()
    _wakeup.set()


