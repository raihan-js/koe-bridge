"""M4 demo: one FLEURS utterance end-to-end through the streaming pipeline.

Prints a timeline: VAD segments -> ASR text -> MT text -> TTS wav,
with cumulative lag at each stage. CPU + GPU (Qwen MT).
Usage: .venv/bin/python scripts/demo.py [ja_jp|en_us] [row_idx]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_m1 import read_fleurs, load_wav  # noqa: E402
from koebridge.stream import VAD, Chunker  # noqa: E402
from koebridge.pipeline import run_streaming  # noqa: E402


def main() -> None:
    lang = sys.argv[1] if len(sys.argv) > 1 else "ja_jp"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    row = read_fleurs(lang, 50)[idx]
    sr, y = load_wav(f"data/fleurs/audio/test/{row['file']}")
    print(f"file: {row['file']}  dur: {len(y) / sr:.1f}s")
    print(f"reference: {row['text'][:120]}")

    from koebridge.asr import ASR
    from koebridge.tts import SBV2JA, KokoroEN
    asr = ASR()
    # Canonical M1/M2 pipeline: ja->en via opus (>>en<<), en->ja via Qwen.
    if lang == "ja_jp":
        from koebridge.mt import MT
        mt_b = MT("data/mt/opus-mt-ja-en")
        tts = KokoroEN()
        mt_fn = lambda texts: mt_b.translate([f">>en<< {t}" for t in texts])  # noqa: E731
    else:
        from koebridge.llm_mt import LLMTranslator
        mt_b = LLMTranslator("data/mt/qwen2.5-1.5b-instruct", device=0)
        tts = SBV2JA()
        mt_fn = lambda texts: mt_b.translate(texts, "English", "Japanese")  # noqa: E731
    vad = VAD()
    alang = "ja" if lang == "ja_jp" else "en"
    t0 = time.monotonic()
    res = run_streaming(
        y, sr, vad,
        lambda w, s: asr.transcribe(w, s, alang),
        mt_fn,
        lambda t: tts.synth(t),
    )
    for e in res["events"]:
        s, en = e["seg"]
        print(f"[{s:5.1f}-{en:5.1f}s] ASR: {e['asr'][:90]}")
        print(f"           MT: {e['mt'][:90]}")
    print(f"first-audio lag: {res['first_audio_lag_s']:.2f}s")
    print(f"total wall: {time.monotonic() - t0:.1f}s, segments: {res['n_segments']}")


if __name__ == "__main__":
    main()
