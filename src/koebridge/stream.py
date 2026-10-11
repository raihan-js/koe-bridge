"""VAD wrapper (silero, torch.hub) + fixed-size chunker with timestamps."""

from __future__ import annotations

from typing import Any

import numpy as np


class VAD:
    def __init__(self, thresh: float = 0.5):
        import torch

        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True)
        model.eval()
        self.m = model
        self.thresh = thresh

    def is_speech(self, chunk_16k: np.ndarray) -> bool:
        import torch

        x = torch.from_numpy(np.asarray(chunk_16k, dtype=np.float32))
        with torch.no_grad():
            p = float(self.m(x, 16000).item())
        return p >= self.thresh


class Chunker:
    """Split audio into fixed chunks; VAD marks speech; segments end on
    trailing silence (min_silence_s) or max length. Emits (start_s, end_s)."""

    def __init__(self, chunk_s: float = 0.5, min_silence_s: float = 0.4, max_seg_s: float = 20.0):
        self.chunk_s = chunk_s
        self.min_silence_s = min_silence_s
        self.max_seg_s = max_seg_s

    def segment(self, wav: np.ndarray, sr: int, vad: Any = None) -> list[tuple[float, float]]:
        n_chunk = int(self.chunk_s * sr)
        flags = []
        for i in range(0, len(wav), n_chunk):
            c = wav[i:i + n_chunk]
            flags.append(True if vad is None else bool(vad.is_speech(_to16k(c, sr))))
        segs: list[tuple[float, float]] = []
        start: int | None = None
        sil = 0
        for i, f in enumerate(flags):
            t = i * self.chunk_s
            if f:
                if start is None:
                    start = i
                sil = 0
                if (i + 1 - start) * self.chunk_s >= self.max_seg_s:
                    segs.append((start * self.chunk_s, (i + 1) * self.chunk_s))
                    start, sil = None, 0
            elif start is not None and (sil := sil + 1) * self.chunk_s >= self.min_silence_s:
                segs.append((start * self.chunk_s, i * self.chunk_s))
                start, sil = None, 0
        if start is not None:
            segs.append((start * self.chunk_s, len(flags) * self.chunk_s))
        return segs


def _to16k(c: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return np.asarray(c, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, 16000)
    return resample_poly(np.asarray(c, dtype=np.float32), 16000 // g, sr // g).astype(np.float32)
