"""Streaming pipeline: VAD segments -> ASR chunks -> incremental MT -> chunk TTS.

Lockstep simulation on the audio timeline (no sleeping): each stage records
emission times relative to audio time. First-audio lag = first translated
audio ready (audio-timeline seconds) minus end-of-speech. Honest about what
it is: model+chunking latency on the critical path, not wall-clock realtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


def run_streaming(wav: npt.NDArray[np.float64], sr: int, vad: Any,
                  asr_fn: Callable[[npt.NDArray[np.float64], int], dict[str, Any]],
                  mt_fn: Callable[[list[str]], dict[str, Any]],
                  tts_fn: Callable[[str], dict[str, Any]],
                  chunk_s: float = 0.5, max_seg_s: float = 20.0) -> dict[str, Any]:
    from koebridge.stream import Chunker

    segs = Chunker(chunk_s=chunk_s, max_seg_s=max_seg_s).segment(wav, sr, vad)
    events: list[dict[str, Any]] = []
    t_proc = 0.0
    mt_texts: list[str] = []
    for start, end in segs:
        seg = wav[int(start * sr):int(end * sr)]
        a = asr_fn(seg, sr)
        t_proc += float(a.get("latency_s", 0.0))
        mt_texts.append(a.get("text", ""))
        m = mt_fn(mt_texts)
        t_proc += float(m.get("latency_s", 0.0))
        cur = m.get("texts", [""])[-1] if m.get("texts") else ""
        t = tts_fn(cur)
        t_proc += float(t.get("latency_s", 0.0))
        events.append({"seg": (start, end), "asr": a.get("text", ""),
                       "mt": cur, "audio_ready_at": end + t_proc,
                       "end_of_speech": end})
    first_lag = events[0]["audio_ready_at"] - events[0]["end_of_speech"] if events else None
    return {"segments": segs, "events": events, "first_audio_lag_s": first_lag,
            "n_segments": len(segs)}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    lags = [r["first_audio_lag_s"] for r in runs if r["first_audio_lag_s"] is not None]
    return {"n": len(runs),
            "p50_lag_s": round(float(np.median(lags)), 2) if lags else None,
            "p95_lag_s": round(float(np.quantile(lags, 0.95)), 2) if lags else None}
