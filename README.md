# Koe-Bridge — open streaming Japanese-English speech translator

Research question: Japanese puts the verb (and negation) last — how long must
a simultaneous translator wait, and what does waiting cost in lag vs quality?

All local, no API: Whisper-small (ASR) → opus-mt-ja-en / Qwen2.5-1.5B (MT) →
Kokoro / Style-Bert-VITS2 (TTS). Eval speech: FLEURS test (CC-BY-4.0).

Plan: `docs/PLAN.md`. Decisions: `docs/decisions.md` (append-only).

## Results

### M1 — cascade audit (n=200/direction, `results/m1.json` + per-row JSONL)

| dir | ASR err | MT BLEU cascade / clean | MT chrF | roundtrip |
|-----|---------|------------------------|---------|-----------|
| ja→en (opus) | CER 0.091 | 10.6 / 18.1 (13a) | 37.0 | WER 0.223 |
| en→ja (Qwen-1.5B) | WER 0.104 | 24.1 / 24.3 (char) | 20.9 | CER 0.817 |

Per-stage GPU cost: ASR ~0.35s, opus MT 1.39s, Qwen MT 0.38s, TTS ~0.2s.
ASR barely degrades MT (cascade ≈ clean). opus-mt-en-jap scored BLEU 0.0
(dead — fluent unrelated output) and was replaced by Qwen (NMT-vs-LLM arm).
Roundtrip en-side CER 0.82 is flagged, not explained (chain metric).

### M2 — streaming first-audio lag (n=20/direction, `results/m2.json`)

VAD segments → ASR → MT → TTS, lockstep sim on full utterances.

| dir | p50 lag | p95 lag |
|-----|---------|---------|
| ja→en | 1.86s | 3.32s |
| en→ja | 1.26s | 2.22s |

Bug found and fixed: silero VAD carries RNN state across files
(non-deterministic segmentation); now reset per file + regression test.

### M3 — word-order wait cost (n=100/direction, `results/m3.json`)

Prefix-survival = |LCP(prefix translation, full translation)| /
|prefix translation| on reference transcripts (no ASR error).
Low survival = a retranslation strategy flickers.

| dir (backend) | 25% | 50% | 75% |
|---------------|-----|-----|-----|
| en→ja (Qwen) | 0.007 | 0.014 | 0.020 |
| ja→en (opus, canonical) | 0.077 | 0.088 | 0.132 |
| ja→en (Qwen control) | 0.134 | 0.184 | 0.204 |

Same backend (Qwen), matched 50% prefixes: 0.184 ja→en vs 0.014 en→ja
(~13x). The SVO→SOV direction pays the wait penalty, as predicted.
Demo: `scripts/demo.py [ja_jp|en_us] [row]` (one utterance, printed timeline).

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# model weights + FLEURS audio go in data/ (research-only, never committed)
.venv/bin/python scripts/run_m1.py   # ~15 min
.venv/bin/python scripts/run_m2.py   # ~20 min
.venv/bin/python scripts/run_m3.py en_us && .venv/bin/python scripts/run_m3.py ja_jp
.venv/bin/python -m pytest tests/ -q
```

## Limits

- One 3s-chunk Whisper model, one VAD threshold, greedy decoding everywhere.
- M2 is a lockstep sim, not a live streaming server (no partial-audio policy).
- M3 uses reference transcripts; ASR noise would lower survival further.
- FLEURS read speech only; no disfluencies, no negation-focus evaluation
  (the original polarity question is still open).
- TTS voices are research-only (JSUT/JVS-derived); weights never published.
