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

## 2026-10-11 — M2/M3 + pipeline-convention fix
- Silero VAD (torch.hub) carries RNN hidden state across forward calls:
  same file segmented 0.87 then 0.98; files scored 0.0 speech kept in one
  process, 0.5+ in another. Fix: warmup forward at load + reset_states()
  per file (test_segment_resets_stateful_vad). M2 re-ran after fix.
- Pipeline calling convention is canonical: Qwen with full language names
  ("English"->"Japanese", in the system prompt), opus ja->en with `>>en<<`
  prefix. run_m3/demo initially used "en"/"ja" + no prefix; realigned and
  re-ran (numbers changed materially). Lesson: one helper for backend
  construction shared by M1/M2/M3/demo (deferred; conventions documented).
- M3 JA arm first ran with word-split prefixes: Japanese has no spaces, so
  every prefix == full sentence (survival 0.978, invalid). Fixed with
  character prefixes; bogus arm deleted and re-run.
