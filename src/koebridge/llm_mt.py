"""Decoder-only LLM translator (Qwen2.5-1.5B-Instruct, local, temp 0).

Used for en->ja after opus-mt-en-jap measured BLEU 0.0 (degenerate;
documented in docs/decisions.md), and as the LLM-vs-NMT comparison arm.
"""

from __future__ import annotations

from typing import Any


class LLMTranslator:
    def __init__(self, model_id: str, device: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        m: Any = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
            device_map=f"cuda:{device}" if device >= 0 else "cpu")
        m.eval()
        self.m = m
        self.device = device

    def translate(self, texts: list[str], src: str, tgt: str, batch: int = 4) -> dict[str, Any]:
        import time

        import torch

        t0 = time.monotonic()
        outs = []
        for i in range(0, len(texts), batch):
            batch_ps = []
            for x in texts[i:i + batch]:
                msgs = [{"role": "system", "content": f"You translate {src} to {tgt}. Output ONLY the translation in {tgt}, nothing else, no explanations, no other languages."},
                        {"role": "user", "content": x}]
                batch_ps.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = self.tok(batch_ps, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if self.device >= 0:
                enc = {k: v.to(f"cuda:{self.device}") for k, v in enc.items()}
            with torch.no_grad():
                gen = self.m.generate(**enc, max_new_tokens=256, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
            outs += [str(self.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)).strip()
                     for g in gen]
        dt = time.monotonic() - t0
        return {"texts": outs, "latency_s": dt, "per_item_s": dt / max(len(texts), 1)}
