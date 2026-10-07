"""Microphone for whisper mode: on only while you hold the key, timestamped on the same clock as
the camera frames, so the audio can be cut to exactly the recorded video.

Timing matters: on test clips, audio 0.4 s off the video doubled the word error rate.
Supports sounddevice (macOS/Windows) and native PipeWire/PulseAudio parec/pw-record (Linux)
when PortAudio library is not installed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time

import numpy as np

RATE = 16000
CAMERA_LATENCY = 0.06  # a frame is read ~2 frames after it was exposed


class Mic:
    def __init__(self, device=None):
        self.device = device if device is not None else self._builtin()
        self._stream = None
        self._proc = None
        self._proc_thread = None
        self._chunks: list[tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()
        self.error: str | None = None
        self._backend = "unknown"

    @staticmethod
    def _builtin():
        """The default microphone device index for sounddevice if available."""
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and "iphone" not in d["name"].lower():
                    if "macbook" in d["name"].lower() or "built-in" in d["name"].lower():
                        return i
        except Exception:
            pass
        return None  # system default

    def start(self):
        with self._lock:
            self._chunks = []
        if self._stream is not None or self._proc is not None:
            return

        # 1. Try sounddevice first (standard for macOS and Windows, or Linux with libportaudio)
        try:
            import sounddevice as sd
            # Test that PortAudio is actually dynamically loaded and usable
            sd.query_devices()
            self._stream = sd.InputStream(
                samplerate=RATE, channels=1, dtype="float32", device=self.device,
                blocksize=0, callback=self._cb,
            )
            self._stream.start()
            self._backend = "sounddevice"
            self.error = None
            return
        except (ImportError, OSError):
            # PortAudio not found or sounddevice unavailable
            pass

        # 2. On Linux, fall back to native PipeWire / PulseAudio recording utilities
        if sys.platform.startswith("linux"):
            cmd = None
            if shutil.which("parec"):
                # parec works with both PulseAudio and PipeWire (via pipewire-pulse)
                cmd = ["parec", "--latency-msec=20", "--rate=16000", "--channels=1", "--format=float32le"]
            elif shutil.which("pw-record"):
                cmd = ["pw-record", "--rate=16000", "--channels=1", "--format=f32", "-"]
            elif shutil.which("arecord"):
                cmd = ["arecord", "-q", "-r", "16000", "-c", "1", "-f", "FLOAT_LE", "-t", "raw"]

            if cmd:
                try:
                    self._proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        bufsize=0,
                    )
                    self._backend = cmd[0]
                    self.error = None
                    self._proc_thread = threading.Thread(target=self._read_proc, daemon=True)
                    self._proc_thread.start()
                    return
                except Exception as e:
                    self.error = f"Failed to start {cmd[0]}: {e}"
                    print(f"[lipflow] {self.error}")
                    return

        self.error = "No audio recording library (PortAudio) or command (parec/pw-record) found"
        print(f"[lipflow] microphone unavailable: {self.error}")

    def _cb(self, data, frames, t, status):
        latency = self._stream.latency if self._stream is not None else 0.0
        t0 = time.time() - frames / RATE - float(latency or 0.0)
        with self._lock:
            self._chunks.append((t0, data[:, 0].copy()))

    def _read_proc(self):
        """Read 40 ms (640 float32 samples = 2560 bytes) chunks from the recording process."""
        chunk_bytes = 2560
        while self._proc and self._proc.poll() is None:
            raw = self._proc.stdout.read(chunk_bytes)
            if not raw:
                break
            now = time.time()
            data = np.frombuffer(raw, dtype=np.float32)
            t0 = now - len(data) / RATE
            with self._lock:
                self._chunks.append((t0, data.copy()))

    def stop(self) -> list[tuple[float, np.ndarray]]:
        # Stop process if used
        if self._proc is not None:
            p, self._proc = self._proc, None
            try:
                p.terminate()
                p.wait(timeout=0.2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        # Stop stream if used
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return chunks


def segment(chunks: list[tuple[float, np.ndarray]], t_start: float, n_frames: int, fps: int = 25) -> "np.ndarray | None":
    """Audio for n_frames video frames starting at t_start (video clock), 640 samples per frame."""
    if not chunks:
        return None
    t_start -= CAMERA_LATENCY
    want = n_frames * RATE // fps
    out = np.zeros(want, np.float32)
    got = 0
    for t0, a in chunks:
        off = int(round((t0 - t_start) * RATE))  # where this chunk lands in the output
        lo, hi = max(off, 0), min(off + len(a), want)
        if hi > lo:
            out[lo:hi] = a[lo - off:hi - off]
            got += hi - lo
    return out if got > want * 0.5 else None
