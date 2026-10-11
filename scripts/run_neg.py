"""Negation-polarity study: does verb-final negation break prefix translation?

Japanese negation (ない/ません/...) comes at the END of the clause. A
simultaneous system translating prefixes may commit to AFFIRMATIVE, then flip
when ない arrives. Method: mine FLEURS test transcripts for negation-bearing
sentences (deduped), translate char-prefixes at 10% steps with the canonical
M1/M2 backends, classify output polarity with documented regexes, and report:
flip rate by prefix fraction, median first-correct-polarity fraction, and
full-translation vs source agreement (MT negation-drop rate).
Writes results/neg.json.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_m1 import read_fleurs

from koebridge.llm_mt import LLMTranslator
from koebridge.mt import MT

JA_NEG_SRC = re.compile(r"ない|ません|なかった|ませんでした|ずに|ざる|まい|なし")
# Lexical adjectives ending in ない that are NOT grammatical negation.
JA_FALSE_POS = re.compile(r"少ない|危ない|汚い|浅ない")
EN_NEG_SRC = re.compile(
    r"n't\b|\bnot\b|\bnever\b|\bneither\b|\bnor\b|\bnone\b|\bnothing\b"
    r"|\bwithout\b|\bnobody\b|\bnowhere\b|\bhardly\b|\bseldom\b")


def neg_sents(lang: str, cap: int = 40) -> list[str]:
    rx = JA_NEG_SRC if lang == "ja_jp" else EN_NEG_SRC
    seen: dict[str, None] = {}
    for r in read_fleurs(lang, 10000):
        if rx.search(r["text"]) and not (
                lang == "ja_jp" and JA_FALSE_POS.search(r["text"])):
            seen.setdefault(r["text"], None)
        if len(seen) >= cap:
            break
    return list(seen)
# Output classifiers (documented, validated against full translations below).
JA_NEG_OUT = re.compile(r"ない|ません|なかった|ませんでした|ざる|まい|なし|ぬ\b|ず\b")
EN_NEG_OUT = re.compile(
    r"n't\b|\bnot\b|\bnever\b|\bneither\b|\bnor\b|\bnone\b|\bnothing\b"
    r"|\bwithout\b|\bnobody\b|\bnowhere\b|\bhardly\b|\bno\b")

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main() -> None:
    lang = sys.argv[1] if len(sys.argv) > 1 else "ja_jp"
    out = Path("results/neg.json")
    allres: dict = json.loads(out.read_text()) if out.exists() else {}
    if lang == "ja_jp":  # canonical: opus + >>en<< prefix, EN output
        backend = MT("data/mt/opus-mt-ja-en")
        out_rx = EN_NEG_OUT

        def mt(texts: list[str]) -> list[str]:
            return backend.translate([f">>en<< {t}" for t in texts])["texts"]
    else:  # canonical: Qwen English->Japanese, JA output
        backend = LLMTranslator("data/mt/qwen2.5-1.5b-instruct", device=0)
        out_rx = JA_NEG_OUT

        def mt(texts: list[str]) -> list[str]:
            return backend.translate(texts, "English", "Japanese")["texts"]

    sents = neg_sents(lang)
    units_fn = (lambda t: t.split()) if lang == "en_us" else (lambda t: list(t))
    joiner = " " if lang == "en_us" else ""
    flips: dict[str, list[int]] = {str(f): [] for f in FRACS}
    first_correct: list[float | None] = []
    full_agree = 0
    aff_then_neg = 0
    t0 = time.time()
    for i, sent in enumerate(sents):
        units = units_fn(sent)
        full = mt([sent])[0]
        full_neg = bool(out_rx.search(full))
        full_agree += 1 if full_neg else 0  # source is negated by construction
        pols = []
        for f in FRACS:
            k = max(1, int(len(units) * f))
            p = mt([joiner.join(units[:k])])[0]
            pn = bool(out_rx.search(p))
            pols.append(pn)
            flips[str(f)].append(0 if pn == full_neg else 1)
        fc = next((f for f, pn in zip(FRACS, pols) if pn == full_neg), None)
        first_correct.append(fc)
        if pols[0] is False and full_neg:
            aff_then_neg += 1  # opened affirmative, ended negated
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sents)} ({time.time() - t0:.0f}s)", flush=True)
    fc_real = [f for f in first_correct if f is not None]
    allres[lang] = {
        "n": len(sents),
        "source": "FLEURS test transcripts containing negation (deduped)",
        "full_neg_rate": round(full_agree / len(sents), 3),
        "flip_rate": {f: round(sum(v) / len(v), 3) for f, v in flips.items()},
        "median_first_correct_frac": sorted(fc_real)[len(fc_real) // 2] if fc_real else None,
        "affirm_then_negate_rate": round(aff_then_neg / len(sents), 3),
    }
    out.write_text(json.dumps(allres, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(allres[lang], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
