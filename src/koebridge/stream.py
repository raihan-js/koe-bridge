"""VAD wrapper (silero, torch.hub) + fixed-size chunker with timestamps."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class VAD:
    def __init__(self, thresh: float = 0.5):
        import torch

        self.thresh = thresh
        loaded: Any = torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True)
        model, _ = loaded
        model.eval()
        with torch.no_grad():  # warmup: first forward differs (JIT/state init)
            model(torch.zeros(512), 16000)
            model.reset_states()
        self.m = model

    def reset(self) -> None:
        """Reset RNN hidden state. MUST be called between unrelated utterances —
        the silero model otherwise carries state across files (non-deterministic
        segmentation across runs)."""
        self.m.reset_states()

    def is_speech(self, chunk_16k: npt.NDArray[np.float64]) -> bool:
        import torch

        x = np.asarray(chunk_16k, dtype=np.float32).reshape(-1)
        # Silero takes fixed 512-sample frames at 16kHz; max over frames.
        best = 0.0
        for i in range(0, max(len(x) - 511, 1), 512):
            w = x[i:i + 512]
            if len(w) < 512:
                w = np.pad(w, (0, 512 - len(w)))
            with torch.no_grad():
                p: Any = self.m(torch.from_numpy(w), 16000)
                best = max(best, float(p.item()))
        return best >= self.thresh


class Chunker:
    """Split audio into fixed chunks; VAD marks speech; segments end on
    trailing silence (min_silence_s) or max length. Emits (start_s, end_s)."""

    def __init__(self, chunk_s: float = 0.5, min_silence_s: float = 0.4, max_seg_s: float = 20.0):
        self.chunk_s = chunk_s
        self.min_silence_s = min_silence_s
        self.max_seg_s = max_seg_s

    def segment(self, wav: npt.NDArray[np.float64], sr: int, vad: Any = None) -> list[tuple[float, float]]:
        """Split into speech segments. Resets stateful VAD (RNN state) per file."""
        if vad is not None and hasattr(vad, "reset"):
            vad.reset()
        n_chunk = int(self.chunk_s * sr)
        flags = []
        for i in range(0, len(wav), n_chunk):
            c = wav[i:i + n_chunk]
            flags.append(True if vad is None else bool(vad.is_speech(_to16k(c, sr))))
        segs: list[tuple[float, float]] = []
        start: int | None = None
        sil = 0
        for i, f in enumerate(flags):
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


def _to16k(c: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
    if sr == 16000:
        return np.asarray(c, dtype=np.float64)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, 16000)
    return np.asarray(resample_poly(np.asarray(c, dtype=np.float32), 16000 // g, sr // g), dtype=np.float64)
