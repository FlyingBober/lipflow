"""Private microphone helper. Native audio hangs must not hold the dictation UI hostage.

Audio travels over an anonymous pipe, never a file or network connection. Closing
stdin stops capture. A watchdog exits even if PortAudio hangs in start/stop/close,
so the OS can reclaim the microphone. This also works if the parent app exits.
"""
from __future__ import annotations

import json
import os
import pickle
import queue
import sys
import threading
import time

RATE = 16000
STOP_TIMEOUT = 1.0


def _builtin(sd):
    """Prefer the built-in microphone over a Continuity iPhone."""
    for i, device in enumerate(sd.query_devices()):
        name = device["name"].lower()
        if device["max_input_channels"] > 0 and "iphone" not in name:
            if "macbook" in name or "built-in" in name:
                return i
    return None


def capture(device=None):
    stop, closed = threading.Event(), threading.Event()

    def watch_parent():
        sys.stdin.buffer.read(1)
        stop.set()
        if not closed.wait(STOP_TIMEOUT):
            os._exit(2)  # Do not run PortAudio's atexit handlers after a native deadlock.

    threading.Thread(target=watch_parent, daemon=True).start()
    messages = queue.SimpleQueue()
    stream = None
    latency = 0.0

    def callback(data, frames, timing, status):
        if not stop.is_set():
            timestamp = time.time() - frames / RATE - latency
            messages.put(("audio", timestamp, data[:, 0].copy()))

    def send(message):
        pickle.dump(message, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.flush()

    try:
        import sounddevice as sd
        if stop.is_set():
            return
        if device is None:
            device = _builtin(sd)
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=device,
                                blocksize=0, callback=callback)
        latency = float(stream.latency or 0.0)
        if stop.is_set():
            return
        stream.start()
        while not stop.is_set():
            try:
                send(messages.get(timeout=0.02))
            except queue.Empty:
                pass
    except BrokenPipeError:
        os._exit(0)  # The parent exited; there is nowhere to deliver audio.
    except Exception as exc:
        if sys.platform.startswith("linux") and ("PortAudio" in str(exc) or "sounddevice" in str(exc)):
            import shutil
            import subprocess
            import numpy as np
            cmd = None
            if shutil.which("parec"):
                cmd = ["parec", "--latency-msec=20", "--rate=16000", "--channels=1", "--format=float32le"]
            elif shutil.which("pw-record"):
                cmd = ["pw-record", "--rate=16000", "--channels=1", "--format=f32", "-"]
            elif shutil.which("arecord"):
                cmd = ["arecord", "-q", "-r", "16000", "-c", "1", "-f", "FLOAT_LE", "-t", "raw"]
            if cmd:
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
                    chunk_bytes = 2560
                    while not stop.is_set() and proc.poll() is None:
                        raw = proc.stdout.read(chunk_bytes)
                        if not raw:
                            break
                        now = time.time()
                        data = np.frombuffer(raw, dtype=np.float32)
                        t0 = now - len(data) / RATE
                        send(("audio", t0, data.copy()))
                    try:
                        proc.terminate()
                        proc.wait(timeout=0.2)
                    except Exception:
                        pass
                    return
                except BrokenPipeError:
                    os._exit(0)
                except Exception as e:
                    exc = e
        try:
            send(("error", str(exc)))
        except BrokenPipeError:
            os._exit(0)
    finally:
        # Bound cleanup even when startup failed before the parent requested stop.
        def deadline():
            if not closed.wait(STOP_TIMEOUT):
                os._exit(2)
        threading.Thread(target=deadline, daemon=True).start()
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            closed.set()
            # PortAudio also has an atexit termination path that can hang on macOS.
            os._exit(0)


if __name__ == "__main__":
    capture(json.loads(sys.argv[1]))
