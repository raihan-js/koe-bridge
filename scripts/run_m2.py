"""M2 streaming measurement: VAD segments -> ASR -> MT -> TTS, lockstep sim.

20 items per direction (FLEURS). Reports p50/p95 first-audio lag + per-stage
latencies + segment counts. Writes results/m2.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from run_m1 import load_wav, read_fleurs


def main(n: int = 20, langs: tuple[str, ...] = ("ja_jp", "en_us")) -> None:
    import json

    from koebridge.asr import ASR
    from koebridge.mt import MT
    from koebridge.pipeline import run_streaming, summarize
    from koebridge.stream import VAD
    from koebridge.tts import SBV2JA, KokoroEN

    asr = ASR()
    vad = VAD()
    out = {}
    cfgs = {"ja_jp": ("ja", "marian", KokoroEN()),
            "en_us": ("en", "qwen", SBV2JA())}
    mt_ja_en = MT("data/mt/opus-mt-ja-en")
    sys.path.insert(0, "src")
    from koebridge.llm_mt import LLMTranslator

    mt_en_ja = LLMTranslator("data/mt/qwen2.5-1.5b-instruct", device=0)
    for lang in langs:
        asr_lang, kind, tts = cfgs[lang]
        mt_obj = mt_ja_en if kind == "marian" else mt_en_ja
        runs = []
        for r in read_fleurs(lang, n):
            sr, y = load_wav(f"data/fleurs/audio/test/{r['file']}")
            asr_fn = lambda seg, s, _a=asr, _l=asr_lang: _a.transcribe(seg, s, _l)
            if kind == "marian":
                mt_fn = lambda texts, _m=mt_obj: _m.translate([f">>en<< {t}" for t in texts])
            else:
                mt_fn = lambda texts, _m=mt_obj: _m.translate(texts, "English", "Japanese")
            tts_fn = lambda text, _t=tts: _t.synth(text)
            runs.append(run_streaming(y, sr, vad, asr_fn, mt_fn, tts_fn))
        out[lang] = summarize(runs)
        out[lang]["runs"] = [{k: v for k, v in r.items() if k != "events"} for r in runs[:3]]
    prev = {}
    if Path("results/m2.json").exists():
        prev = json.loads(Path("results/m2.json").read_text())
    prev.update(out)
    Path("results/m2.json").write_text(json.dumps(prev, indent=2))
    print(json.dumps({k: {kk: v for kk, v in d.items() if kk != "runs"} for k, d in out.items()}, indent=2))


if __name__ == "__main__":
    import sys

    langs = tuple(sys.argv[1:]) or ("ja_jp", "en_us")
    main(langs=langs)
