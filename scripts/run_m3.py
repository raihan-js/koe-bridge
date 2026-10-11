"""M3: word-order wait-cost study.

Question: does EN->JA pay a bigger wait-k penalty than JA->EN?
Method: for each utterance, translate source prefixes (first 25/50/75/100% of
words) with the M1 MT backends, using REFERENCE transcripts (isolates MT from
ASR error). Metric: prefix-survival = |LCP(prefix_trans, full_trans)|
/ |prefix_trans| in words. Low survival = the early translation gets
rewritten = high wait-k penalty for a simultaneous system.
Writes results/m3.json. ~25 min GPU (Qwen arm).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_m1 import read_fleurs

from koebridge.llm_mt import LLMTranslator
from koebridge.mt import MT

FRACS = [0.25, 0.50, 0.75, 1.0]


def survival(prefix: str, full: str) -> float:
    a, b = prefix.split(), full.split()
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), 1)


def main() -> None:
    lang = sys.argv[1] if len(sys.argv) > 1 else "en_us"
    out = Path("results/m3.json")
    allres: dict = json.loads(out.read_text()) if out.exists() else {}
    rows = read_fleurs(lang, 100)
    # Canonical pipeline calling convention (must match run_m1/run_m2):
    # Qwen with full language names, opus with the >>en<< prefix.
    if lang == "en_us":
        backend = LLMTranslator("data/mt/qwen2.5-1.5b-instruct", device=0)

        def mt(texts: list[str]) -> list[str]:
            return backend.translate(texts, "English", "Japanese")["texts"]
    else:
        backend = MT("data/mt/opus-mt-ja-en")

        def mt(texts: list[str]) -> list[str]:
            return backend.translate([f">>en<< {t}" for t in texts])["texts"]
    agg: dict[str, list[float]] = {str(f): [] for f in FRACS}
    t0 = time.time()
    for i, r in enumerate(rows):
        # EN: word prefixes. JA (no spaces): character prefixes.
        units = r["text"].split() if lang == "en_us" else list(r["text"])
        full = mt([r["text"]])[0]
        for f in FRACS:
            k = max(1, int(len(units) * f))
            joiner = " " if lang == "en_us" else ""
            p = mt([joiner.join(units[:k])])[0]
            agg[str(f)].append(round(survival(p, full), 3))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/100 ({time.time() - t0:.0f}s)", flush=True)
    allres[lang] = {
        "n": len(rows),
        "source": "reference transcripts (no ASR error)",
        "mean_survival": {f: round(sum(v) / len(v), 3) for f, v in agg.items()},
    }
    out.write_text(json.dumps(allres, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(allres[lang], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
