"""TTS backends: SBV2 jvnv (Japanese) + Kokoro (English, +JA compare)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SBV2JA:
    """Style-Bert-VITS2 jvnv-F1-jp (akusento recipe: fp32 workarounds)."""

    def __init__(self, assets: str | Path = "/home/raihan/Desktop/APPS/research-learn/akusento/data"):
        import torch
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models
        from style_bert_vits2.tts_model import TTSModel

        base = Path(assets)
        bert = base / "sbv2-bert" / "deberta-v2-large-japanese-char-wwm"
        bert_models.load_model(Languages.JP, str(bert))
        bert_models.load_tokenizer(Languages.JP, str(bert))
        for _m in getattr(bert_models, "__loaded_models").values():
            _m.float()
        mp = base / "sbv2" / "jvnv-F1-jp"
        self.m = TTSModel(model_path=mp / "jvnv-F1-jp_e160_s14000.safetensors",
                          config_path=mp / "config.json",
                          style_vec_path=np.load(mp / "style_vectors.npy"),
                          device="cuda" if torch.cuda.is_available() else "cpu")
        self.m.load()
        net = getattr(self.m, "_TTSModel__net_g", None)
        if net is not None:
            net.float()

    def synth(self, text: str) -> dict:
        import time

        t0 = time.monotonic()
        sr, wav = self.m.infer(text=text, line_split=False)
        return {"sr": sr, "wav": np.asarray(wav), "latency_s": time.monotonic() - t0}


class KokoroEN:
    """Kokoro-82M for English (Apache-2.0). Lazy: loads on first synth."""

    def __init__(self, voice: str = "af_heart"):
        self.voice = voice
        self.pipe = None

    def synth(self, text: str) -> dict:
        import time

        if self.pipe is None:
            from kokoro import KPipeline

            self.pipe = KPipeline(lang_code="a")
        t0 = time.monotonic()
        chunks = list(self.pipe(text, voice=self.voice))
        sr, parts = 24000, []
        for _, _, audio in chunks:
            parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return {"sr": sr, "wav": wav, "latency_s": time.monotonic() - t0}
