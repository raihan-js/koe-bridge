"""MarianMT translation wrapper (Helsinki OPUS models, CPU/GPU)."""

from __future__ import annotations

from typing import Any


class MT:
    def __init__(self, model_id: str, device: int = -1):
        from transformers import MarianMTModel, MarianTokenizer

        self.tok = MarianTokenizer.from_pretrained(model_id)
        self.m: Any = MarianMTModel.from_pretrained(model_id)
        if device >= 0:

            self.m = self.m.to(f"cuda:{device}").eval()
        self.device = device

    def translate(self, texts: list[str], batch: int = 16) -> dict[str, Any]:
        import time

        t0 = time.monotonic()
        outs = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True)
            if self.device >= 0:
                enc = {k: v.to(f"cuda:{self.device}") for k, v in enc.items()}
            import torch

            with torch.no_grad():
                gen = self.m.generate(**enc, max_new_tokens=256)
            outs += self.tok.batch_decode(gen, skip_special_tokens=True)
        return {"texts": outs, "latency_s": time.monotonic() - t0,
                "per_item_s": (time.monotonic() - t0) / max(len(texts), 1)}
