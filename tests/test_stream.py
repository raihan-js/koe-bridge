"""Streaming pipeline tests (fake stages; timeline math must hold)."""

import numpy as np
import pytest

from koebridge.pipeline import run_streaming, summarize
from koebridge.stream import Chunker


class FakeVAD:
    def __init__(self, speech):
        self.speech = speech
        self.i = 0

    def is_speech(self, chunk):
        v = self.speech[min(self.i, len(self.speech) - 1)]
        self.i += 1
        return v


def test_chunker_segments_on_silence():
    flags = [True] * 6 + [False] * 4 + [True] * 4
    ch = Chunker(chunk_s=0.5, min_silence_s=0.4)
    wav = np.zeros(14 * 8000)
    segs = ch.segment(wav, 16000, FakeVAD(flags))
    assert segs == [(0.0, 3.0), (5.0, 7.0)], segs


def test_pipeline_lag_math():
    wav = np.zeros(16000 * 4)
    vad = FakeVAD([True] * 8)
    asr = lambda seg, sr: {"text": "hi", "latency_s": 1.0}
    mt = lambda texts: {"texts": texts, "latency_s": 2.0}
    tts = lambda text: {"latency_s": 4.0}
    r = run_streaming(wav, 16000, vad, asr, mt, tts, chunk_s=0.5, max_seg_s=20.0)
    assert r["n_segments"] == 1
    # end of speech 4.0s, critical path 1+2+4=7 -> lag 7.0
    assert abs(r["first_audio_lag_s"] - 7.0) < 1e-9
    s = summarize([r, r])
    assert s["p50_lag_s"] == 7.0 and s["n"] == 2


def test_segment_resets_stateful_vad(tmp_state=None):
    """Regression: silero VAD carries RNN state; segment() must reset per file."""
    import shutil
    import sys
    from pathlib import Path

    if not Path("data/fleurs/audio/test").exists():
        pytest.skip("needs FLEURS audio")
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    from koebridge.stream import VAD
    sys.path.insert(0, "scripts")
    from run_m1 import load_wav, read_fleurs
    vad = VAD()
    r = read_fleurs("en_us", 1)[0]
    sr, y = load_wav(f"data/fleurs/audio/test/{r['file']}")
    a = Chunker().segment(y, sr, vad)
    b = Chunker().segment(y, sr, vad)
    assert a == b
