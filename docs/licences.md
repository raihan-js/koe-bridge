# Koe-Bridge — licence table (verified 2026-10-11, re-check at release)

| component | choice | licence | notes |
|---|---|---|---|
| ASR ja/en | faster-whisper large-v3-turbo (OpenAI weights) | MIT (code) + Whisper licence | local, int8 |
| MT ja→en | Helsinki-NLP/opus-mt-ja-en | Apache-2.0 ✓ | MarianMT |
| MT en→ja | Helsinki-NLP/opus-mt-en-jap | VERIFY at download | Helsinki naming uses `jap` |
| TTS JA | Style-Bert-VITS2 jvnv-F1-jp | code AGPL-3.0 (pip dep only); voice CC BY-SA 4.0 | reuse akusento recipe |
| TTS EN (+JA compare) | hexgrad/Kokoro-82M | Apache-2.0 ✓ | |
| VAD | silero-vad | MIT (verify) | |
| Eval speech | google/fleurs (ja_jp, en_us) | CC-BY-4.0 ✓ | subsamples |
| Eval MT | FLEURS transcripts (same) | CC-BY-4.0 ✓ | CoVoST-2 DROPPED (CC-BY-NC) |
| Metrics | sacrebleu, jiwer (both permissive) | MIT/Apache | COMET optional (2GB; decided M1) |
| NLLB-200 | EXCLUDED by default | CC-BY-NC | non-commercial only |

CoVoST-2 exclusion is a verify-at-build adaptation (licence, not preference).
Qwen/NLLB API options need no keys (all local).
