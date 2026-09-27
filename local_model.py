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
from queue import Empty
from typing import Optional

from config import ROUTE_LOCAL, settings
from schemas import Completion


class LocalModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.local_model_name

        self._model = None
        self._tokenizer = None
        self._device = "cpu"

        # Only one real local generation at a time.
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def load(self) -> None:
        """
        Load model weights.

        Called lazily so starting the API does not automatically
        load a multi-GB model unless local generation is requested.
        """

        if settings.mock_mode or self.loaded:
            return

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._device = self._pick_device()

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        # CPU cannot reliably run fp16 Linear layers,
        # so use fp32 there.
        dtype = (
            torch.float32
            if self._device == "cpu"
            else "auto"
        )

        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
        )

        self._model.to(
            self._device
        )

        self._model.eval()

    @staticmethod
    def _pick_device() -> str:
        import torch

        # CUDA covers NVIDIA and ROCm-compatible AMD builds.
        if torch.cuda.is_available():
            return "cuda"

        mps = getattr(
            torch.backends,
            "mps",
            None,
        )

        if (
            mps is not None
            and mps.is_available()
        ):
            return "mps"

        return "cpu"

    # =========================================================
    # PROMPT PREPARATION
    # =========================================================

    def _prepare_input(self, prompt: str):
        """
        Apply the model's chat template and tokenize the prompt.

        Returns:
            encoded input
            prompt token length
            padding token id
        """

        if not self.loaded:
            self.load()

        if getattr(
            self._tokenizer,
            "chat_template",
            None,
        ):
            messages = []

            if settings.system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            settings.system_prompt
                        ),
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            rendered = (
                self._tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            add_special_tokens = False

        else:
            rendered = prompt
            add_special_tokens = True

        encoded = self._tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=(
                add_special_tokens
            ),
        )

        # Some tokenizers emit token_type_ids,
        # which some causal language models reject.
        encoded.pop(
            "token_type_ids",
            None,
        )

        encoded = encoded.to(
            self._device
        )

        prompt_len = int(
            encoded["input_ids"]
            .shape[-1]
        )

        pad_id = (
            self._tokenizer
            .pad_token_id
        )

        if pad_id is None:
            pad_id = (
                self._tokenizer
                .eos_token_id
            )

        return (
            encoded,
            prompt_len,
            pad_id,
        )

    # =========================================================
    # NORMAL NON-STREAMING GENERATION
    # =========================================================

    def generate(
        self,
        prompt: str,
    ) -> Completion:
        started = time.time()

        if settings.mock_mode:
            text = (
                "[mock-local] "
                "concise answer to: "
                f"{prompt[:60]}"
            )

            return Completion(
                text=text,

                prompt_tokens=len(
                    prompt.split()
                ),

                completion_tokens=len(
                    text.split()
                ),

                source=ROUTE_LOCAL,

                latency_s=(
                    time.time()
                    - started
                ),

                model_name=(
                    settings
                    .local_model_name
                ),

                provider="local",
            )

        with self._lock:
            return self._generate_locked(
                prompt,
                started,
            )

    def _generate_locked(
        self,
        prompt: str,
        started: float,
    ) -> Completion:

        import torch

        (
            encoded,
            prompt_len,
            pad_id,
        ) = self._prepare_input(
            prompt
        )

        with torch.no_grad():
            outputs = (
                self._model.generate(
                    **encoded,

                    max_new_tokens=(
                        settings
                        .local_max_new_tokens
                    ),

                    do_sample=False,

                    pad_token_id=(
                        pad_id
                    ),

                    output_scores=True,

                    return_dict_in_generate=True,
                )
            )

        new_tokens = (
            outputs.sequences[0][
                prompt_len:
            ]
        )

        text = (
            self._tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )

        (
            confidence,
            min_token_prob,
            low_token_frac,
        ) = self._calculate_confidence(
            outputs,
            new_tokens,
        )

        return Completion(
            text=text,

            prompt_tokens=(
                prompt_len
            ),

            completion_tokens=int(
                new_tokens.shape[-1]
            ),

            source=ROUTE_LOCAL,

            latency_s=(
                time.time()
                - started
            ),

            model_name=(
                settings
                .local_model_name
            ),

            provider="local",

            confidence=confidence,

            min_token_prob=(
                min_token_prob
            ),

            low_token_frac=(
                low_token_frac
            ),
        )

    # =========================================================
    # STREAMING GENERATION
    # =========================================================

    def stream_generate(
        self,
        prompt: str,
        stop_event=None,
    ):
        """
        Stream local model output.

        Yields:

            (chunk, None)

        while text is being generated.

        At completion:

            (None, Completion)

        stop_event should be a threading.Event. When it is
        set, HuggingFace generation stops at the next token.
        """

        started = time.time()

        # -----------------------------------------------------
        # Mock streaming
        # -----------------------------------------------------

        if settings.mock_mode:
            text = (
                "[mock-local] "
                "concise answer to: "
                f"{prompt[:60]}"
            )

            generated_parts = []

            for word in text.split():

                if (
                    stop_event is not None
                    and stop_event.is_set()
                ):
                    break

                chunk = (
                    word + " "
                )

                generated_parts.append(
                    chunk
                )

                yield (
                    chunk,
                    None,
                )

            final_text = "".join(
                generated_parts
            ).strip()

            completion = Completion(
                text=final_text,

                prompt_tokens=len(
                    prompt.split()
                ),

                completion_tokens=len(
                    final_text.split()
                ),

                source=ROUTE_LOCAL,

                latency_s=(
                    time.time()
                    - started
                ),

                model_name=(
                    settings
                    .local_model_name
                ),

                provider="local",
            )

            yield (
                None,
                completion,
            )

            return

        # -----------------------------------------------------
        # Real local streaming
        # -----------------------------------------------------

        with self._lock:

            import torch

            from transformers import (
                StoppingCriteria,
                StoppingCriteriaList,
                TextIteratorStreamer,
            )

            (
                encoded,
                prompt_len,
                pad_id,
            ) = self._prepare_input(
                prompt
            )

            # -------------------------------------------------
            # Stop button logic
            # -------------------------------------------------

            class StopOnEvent(
                StoppingCriteria
            ):
                def __init__(
                    self,
                    event,
                ):
                    self.event = event

                def __call__(
                    self,
                    input_ids,
                    scores,
                    **kwargs,
                ):
                    return bool(
                        self.event
                        is not None
                        and self.event.is_set()
                    )

            stopping_criteria = (
                StoppingCriteriaList(
                    [
                        StopOnEvent(
                            stop_event
                        )
                    ]
                )
            )

            # -------------------------------------------------
            # HuggingFace streamer
            # -------------------------------------------------

            streamer = (
                TextIteratorStreamer(
                    self._tokenizer,

                    skip_prompt=True,

                    skip_special_tokens=True,

                    # Avoid hanging forever if generation
                    # crashes in the worker thread.
                    timeout=1.0,
                )
            )

            result_holder = {}
            error_holder = {}

            # -------------------------------------------------
            # Run model.generate in background thread
            # -------------------------------------------------

            def run_generation():
                try:
                    with torch.no_grad():
                        outputs = (
                            self._model.generate(
                                **encoded,

                                max_new_tokens=(
                                    settings
                                    .local_max_new_tokens
                                ),

                                do_sample=False,

                                pad_token_id=(
                                    pad_id
                                ),

                                output_scores=True,

                                return_dict_in_generate=True,

                                streamer=(
                                    streamer
                                ),

                                stopping_criteria=(
                                    stopping_criteria
                                ),
                            )
                        )

                    result_holder[
                        "outputs"
                    ] = outputs

                except Exception as err:
                    error_holder[
                        "error"
                    ] = err

            worker = threading.Thread(
                target=run_generation,
                daemon=True,
            )

            worker.start()

            text_parts = []

            # -------------------------------------------------
            # Consume output as it is generated
            # -------------------------------------------------

            while True:
                try:
                    chunk = next(
                        streamer
                    )

                except StopIteration:
                    break

                except Empty:
                    # The streamer timed out while waiting.
                    # If the worker crashed, surface the error.
                    if error_holder:
                        raise error_holder[
                            "error"
                        ]

                    # If generation finished while the queue
                    # was temporarily empty, we're done.
                    if not worker.is_alive():
                        break

                    continue

                if chunk:
                    text_parts.append(
                        chunk
                    )

                    yield (
                        chunk,
                        None,
                    )

            worker.join()

            if error_holder:
                raise error_holder[
                    "error"
                ]

            outputs = (
                result_holder.get(
                    "outputs"
                )
            )

            if outputs is None:
                raise RuntimeError(
                    "Local model produced "
                    "no generation result"
                )

            # -------------------------------------------------
            # Build final Completion
            # -------------------------------------------------

            new_tokens = (
                outputs.sequences[0][
                    prompt_len:
                ]
            )

            text = "".join(
                text_parts
            ).strip()

            (
                confidence,
                min_token_prob,
                low_token_frac,
            ) = self._calculate_confidence(
                outputs,
                new_tokens,
            )

            completion = Completion(
                text=text,

                prompt_tokens=(
                    prompt_len
                ),

                completion_tokens=int(
                    new_tokens.shape[-1]
                ),

                source=ROUTE_LOCAL,

                latency_s=(
                    time.time()
                    - started
                ),

                model_name=(
                    settings
                    .local_model_name
                ),

                provider="local",

                confidence=confidence,

                min_token_prob=(
                    min_token_prob
                ),

                low_token_frac=(
                    low_token_frac
                ),
            )

            yield (
                None,
                completion,
            )

    # =========================================================
    # CONFIDENCE CALCULATION
    # =========================================================

    def _calculate_confidence(
        self,
        outputs,
        new_tokens,
    ):
        """
        Calculate confidence statistics from generated-token logits.

        Returns:
            mean token probability
            minimum token probability
            fraction of tokens below probability 0.5
        """

        confidence = None
        min_token_prob = None
        low_token_frac = None

        if len(new_tokens) == 0:
            return (
                None,
                None,
                None,
            )

        scores = getattr(
            outputs,
            "scores",
            None,
        )

        if not scores:
            return (
                None,
                None,
                None,
            )

        transition_scores = (
            self._model
            .compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
        )

        token_probs = (
            transition_scores[0]
            .exp()
        )

        if len(token_probs) == 0:
            return (
                None,
                None,
                None,
            )

        confidence = float(
            token_probs.mean()
        )

        min_token_prob = float(
            token_probs.min()
        )

        low_token_frac = float(
            (
                token_probs < 0.5
            )
            .float()
            .mean()
        )

        return (
            round(
                confidence,
                4,
            ),

            round(
                min_token_prob,
                4,
            ),

            round(
                low_token_frac,
                4,
            ),
        )