"""Wrapper around a small local HuggingFace model.

Design notes:
- LAZY imports: transformers/torch are only imported when the model actually
  loads, so mock mode (and therefore the test harness) runs on stdlib alone.
- Exact token counts come from the tokenizer, not word counts — accounting
  must match what the provider actually bills.
- Quantization: on constrained hardware, prefer a
  PRE-QUANTIZED checkpoint (GPTQ/AWQ, or a -GGUF variant via llama.cpp) over
  runtime bitsandbytes, which is NVIDIA-only. Swapping checkpoints is just
  LOCAL_MODEL_NAME; nothing else here changes.
- MOCK mode (AGENT_MOCK=1) returns deterministic canned output so routing and
  accounting can be tested with zero downloads and zero network.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from config import ROUTE_LOCAL, settings
from schemas import Completion


class LocalModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.local_model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        # main.run_all calls generate() from a thread pool. One model
        # instance + compute-bound generation → serialize the real path;
        # threads doing remote calls never touch this lock.
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load weights. Call once at startup (or bake into the Docker image)
        so the first task doesn't pay the cold-start."""
        if settings.mock_mode or self.loaded:
            return
        import torch  # heavy import, deliberately deferred
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._pick_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # On GPU, "auto" keeps fp16/bf16 checkpoints at half memory. On CPU,
        # torch cannot run fp16 Linear layers ("addmm_impl_cpu_ not
        # implemented for 'Half'"), and most modern checkpoints ARE fp16/bf16
        # — so force fp32 there. A 1-3B model in fp32 is ~4-12 GB RAM: fine.
        dtype = torch.float32 if self._device == "cpu" else "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        )
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _pick_device() -> str:
        import torch

        # torch.cuda.is_available() is also True on AMD GPUs with ROCm builds
        # of torch, so this branch covers both vendors.
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def generate(self, prompt: str) -> Completion:
        started = time.time()

        if settings.mock_mode:
            text = f"[mock-local] concise answer to: {prompt[:60]}"
            return Completion(
                text=text,
                prompt_tokens=len(prompt.split()),  # fake but deterministic
                completion_tokens=len(text.split()),
                source=ROUTE_LOCAL,
                latency_s=time.time() - started,
            )

        with self._lock:
            return self._generate_locked(prompt, started)

    def _generate_locked(self, prompt: str, started: float) -> Completion:
        if not self.loaded:
            self.load()
        import torch

        # Instruct models need their chat template or output quality craters.
        # Render to a string first, then tokenize WITHOUT re-adding special
        # tokens (the template already contains them).
        if getattr(self._tokenizer, "chat_template", None):
            # Same concise-answer directive as the remote side: local tokens
            # cost no API spend, but a rambling answer is more likely to be
            # graded wrong (observed: "mixed" in reply to an options question).
            messages = []
            if settings.system_prompt:
                messages.append(
                    {"role": "system", "content": settings.system_prompt}
                )
            messages.append({"role": "user", "content": prompt})
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            add_special_tokens = False
        else:
            rendered = prompt
            add_special_tokens = True

        encoded = self._tokenizer(
            rendered, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        # Some tokenizers (Falcon family, older conversions) emit
        # token_type_ids, which model.generate rejects with a ValueError.
        encoded.pop("token_type_ids", None)
        encoded = encoded.to(self._device)

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        with torch.no_grad():
            # Greedy decoding: deterministic outputs → reproducible accuracy
            # measurements, and no sampling-induced flakiness while debugging.
            # output_scores + return_dict_in_generate: keep the per-step
            # logits so we can compute the model's own confidence in its
            # answer — the "draft-and-judge" signal. It reuses logits the
            # forward pass already produced, so it costs only memory.
            outputs = self._model.generate(
                **encoded,
                max_new_tokens=settings.local_max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = int(encoded["input_ids"].shape[-1])
        new_tokens = outputs.sequences[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Mean per-token probability of the generated answer (0..1).
        # compute_transition_scores aligns scores to the GENERATED tokens
        # only (never the prompt); normalize_logits=True is what turns raw
        # logits into actual log-probabilities. Caveat (accepted): the mean
        # flatters very short answers — a 3-token reply is "confident"
        # almost by construction. If the real task set exposes that, switch
        # to min-token-prob or fraction-below-a-floor; the plumbing is
        # identical.
        confidence = None
        if len(new_tokens) > 0:
            transition_scores = self._model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            confidence = float(transition_scores[0].exp().mean())

        return Completion(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(new_tokens.shape[-1]),
            source=ROUTE_LOCAL,
            latency_s=time.time() - started,
            confidence=round(confidence, 4) if confidence is not None else None,
        )
