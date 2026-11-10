"""Quiet-speech ASR using faster-whisper with webcam quality gating.

This is audio ASR gated by visual lip/face activity, not neural AV fusion.
Microphone is active only while the push-to-talk key is held.
"""
from __future__ import annotations

import math
import numpy as np

from .confidence import Hypothesis


class WhisperASR:
    def __init__(self, model: str = "large-v3-turbo", device: str = "cpu", language: str = "ru"):
        self.name = model
        self.device = device
        self.language = language
        self.model = None

    def load(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError(
                    f"For {self.language} whisper input run: uv sync --extra whisper"
                ) from e
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(self.name, device=self.device, compute_type=compute_type)
        return self.model

    def hypotheses(self, wave: np.ndarray | None) -> list[Hypothesis]:
        if wave is None or len(wave) < 9600 or not np.all(np.isfinite(wave)):
            return []
        if np.sqrt(np.mean(np.square(wave))) < 1e-5:
            return []  # silence must not produce hallucinated dictation

        self.load()
        # Quiet speech can be rejected by standard voiced-speech VAD.
        # Amplify bounded amount, then let the decoder's no-speech check reject silence.
        wave = np.asarray(wave, dtype=np.float32)
        wave = wave * min(32.0, 0.3 / (float(np.max(np.abs(wave))) + 1e-8))

        segments, _ = self.model.transcribe(
            wave,
            language=self.language,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        parts, score, n = [], 0.0, 0
        for seg in segments:
            if seg.no_speech_prob > 0.6 or not math.isfinite(seg.avg_logprob):
                continue
            parts.append(seg.text.strip())
            count = max(len(seg.tokens), 1)
            score += seg.avg_logprob * count
            n += count

        text = " ".join(parts).strip()
        if self.language == "zh" and text:
            try:
                from opencc import OpenCC
                text = OpenCC("t2s").convert(text)
            except ImportError:
                pass

        return [Hypothesis(text, score, n)] if text else []
