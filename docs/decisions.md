# Koe-Bridge — decisions (append-only, newest first)

## 2026-10-11 — M1 cascade (results/m1.json, n=200/dir + audit rows)
- Metric bugs caught before publishing: jiwer WER on Japanese = 1.0 always
  (no spaces); sacrebleu 13a on Japanese ≈ 0. Use CER-ja / WER-en and
  char-BLEU for ja targets (documented in results).
- opus-mt-en-jap is DEAD: BLEU 0.0, fluent-unrelated outputs, encoder verified
  working, transformers 4.48 + beam + tags all tried. Replaced by local
  Qwen2.5-1.5B (en->ja char-BLEU 24.3 clean / 24.1 cascade) = NMT-vs-LLM arm.
  opus ja->en needs `>>en<<` prefix (else emits Japanese); kept.
- Qwen failure modes (kept raw, noted): 2/50 English echoes (lowercased
  input); `日本語:` label prefixes; occasional Chinese fragments on short
  inputs (stronger system prompt reduced but did not eliminate).
- Roundtrip is a CHAIN metric, not TTS quality: ja-side WER 0.223 (Kokoro),
  en-side CER 0.82 (echoes + SBV2/Whisper katakana-ization + Whisper loop
  hallucinations like ご視聴ありがとうございました on garbage input).
- CoVoST-2 dropped (CC-BY-NC); FLEURS IDs align across languages (321 shared).
- Venv pins (learned): numpy==1.26.4 (pyopenjtalk ABI), pyarrow==17.0.0 +
  datasets==3.6.0 (datasets 5 needs pyarrow 24), transformers 4.48.3
  (v5 dropped translation pipelines), sacremoses installed.
