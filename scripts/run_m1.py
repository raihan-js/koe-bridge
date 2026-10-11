"""M1 offline cascade runner: ASR -> MT -> TTS both directions.

Writes results/m1.json (metrics + per-stage latency, GPU and CPU subset).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_fleurs(lang: str, n: int, seed: int = 0):
    import csv

    rows = []
    with open(f"data/fleurs/data/{lang}/test.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append({"id": parts[0], "file": parts[1], "text": parts[2]})
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), min(n, len(rows)), replace=False)
    return [rows[int(i)] for i in sorted(idx)]


def load_wav(path: str):
    sr, y = wavfile.read(path)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.dtype == np.float32 and np.abs(y).max() > 1.0:
        y = y / 32768.0
    return sr, y


def main(n: int = 200, tts_n: int = 50) -> None:
    from koebridge.asr import ASR
    from koebridge.mt import MT
    from koebridge.tts import KokoroEN, SBV2JA

    from jiwer import wer as jiwer_wer
    from sacrebleu import corpus_bleu, corpus_chrf

    out: dict = {}
    asr = ASR()
    mt_ja_en = MT("data/mt/opus-mt-ja-en")
    from koebridge.llm_mt import LLMTranslator

    mt_en_ja = LLMTranslator("data/mt/qwen2.5-1.5b-instruct", device=0)
    tts_ja = SBV2JA()
    tts_en = KokoroEN()
    for lang, asr_lang, mt, mt_names, tts, other_lang in (
            ("ja_jp", "ja", mt_ja_en, ("Japanese", "English"), tts_en, "en_us"),
            ("en_us", "en", mt_en_ja, ("English", "Japanese"), tts_ja, "ja_jp")):
        rows = read_fleurs(lang, n)
        aud = [load_wav(f"data/fleurs/audio/test/{r['file']}") for r in rows]
        t0 = time.monotonic()
        hyps = [asr.transcribe(y, sr, asr_lang)["text"] for sr, y in aud]
        asr_s = time.monotonic() - t0
        refs = [r["text"] for r in rows]
        if isinstance(mt, MT):
            mt_out = mt.translate([f">>en<< {h}" for h in hyps])
        else:
            mt_out = mt.translate(hyps, *mt_names)
        # Reference translations: FLEURS same content across languages — align
        # by row order is WRONG (different utterances); score ASR vs transcript
        # (CER/WER) and MT vs ... note below. We report ASR error + MT BLEU
        # against the other-language transcript only where IDs match.
        other = {r["id"]: r["text"] for r in read_fleurs(other_lang, 10000)}
        pairs = [(h, other[r["id"]]) for r, h in zip(rows, mt_out["texts"]) if r["id"] in other]
        if isinstance(mt, MT):
            clean = mt.translate([f">>en<< {r}" for r in refs])
        else:
            clean = mt.translate(refs, *mt_names)
        clean_pairs = [(h, other[r["id"]]) for r, h in zip(rows, clean["texts"]) if r["id"] in other]
        bleu = float(corpus_bleu([p[1] for p in pairs], [[p[0] for p in pairs]]).score) if pairs else None
        chrf = float(corpus_chrf([p[1] for p in pairs], [[p[0] for p in pairs]]).score) if pairs else None
        clean_bleu = float(corpus_bleu([p[1] for p in clean_pairs], [[p[0] for p in clean_pairs]]).score) if clean_pairs else None
        # TTS subset + ASR roundtrip intelligibility.
        sub = pairs[:tts_n]
        rts = []
        tts_s = 0.0
        for hyp, _ in sub:
            r = tts.synth(hyp)
            tts_s += r["latency_s"]
            back = asr.transcribe(r["wav"], r["sr"], "en" if lang == "ja_jp" else "ja")["text"]
            rts.append(back)
        asr_cer = float(np.mean([jiwer_wer([r], [h]) for r, h in zip(refs, hyps)]))
        rt_cer = float(np.mean([jiwer_wer([h], [b]) for (h, _), b in zip(sub, rts)])) if sub else None
        out[lang] = {"n": len(rows), "asr_wer": round(asr_cer, 4), "asr_s_per_item": round(asr_s / len(rows), 3),
                     "mt_bleu_cascade": round(bleu, 2) if bleu is not None else None,
                     "mt_bleu_clean": round(clean_bleu, 2) if clean_bleu is not None else None,
                     "mt_chrf": round(chrf, 2) if chrf is not None else None,
                     "mt_s_per_item": round(mt_out["per_item_s"], 3),
                     "tts_s_per_item": round(tts_s / max(len(sub), 1), 3),
                     "roundtrip_wer": round(rt_cer, 4) if rt_cer is not None else None,
                     "n_mt_pairs": len(pairs)}
    Path("results/m1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
