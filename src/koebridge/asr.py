"""Faster-Whisper ASR wrapper (int8, CPU/GPU)."""

from __future__ import annotations

import numpy as np


class ASR:
    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", compute: str = "int8"):
        from faster_whisper import WhisperModel

        try:
            self.m = WhisperModel(model, device=device, compute_type=compute)
        except Exception:  # noqa: BLE001 - fall back to CPU
            self.m = WhisperModel(model, device="cpu", compute_type="int8")

    def transcribe(self, wav: np.ndarray, sr: int, lang: str) -> dict:
        import time

        t0 = time.monotonic()
        if sr != 16000:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sr, 16000)
            wav = resample_poly(wav.astype(np.float32), 16000 // g, sr // g).astype(np.float32)
        segs, info = self.m.transcribe(wav, language=lang, beam_size=5)
        text = "".join(s.text for s in segs).strip()
        return {"text": text, "latency_s": time.monotonic() - t0,
                "detected_lang": getattr(info, "language", lang)}
