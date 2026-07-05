"""Shared data contracts passed between modules.

Kept in one tiny file so local_model / remote_client / main all agree on the
same shapes WITHOUT importing each other — that keeps each backend
independently swappable on kickoff day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Task:
    """One unit of work handed to the agent."""

    task_id: str
    prompt: str
    # Free-form bag for anything the organizers attach (category, expected
    # format, difficulty tag, ...). The router may use it; nothing else
    # should depend on it.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    """A model response from either backend, in one normalized shape."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    source: str  # ROUTE_LOCAL or ROUTE_REMOTE
    latency_s: float = 0.0
    # Model self-confidence: mean per-token probability of the generated
    # text (0..1), from the model's own logits. Only the LOCAL backend sets
    # it (remote APIs don't expose logprobs by default; mock mode has no
    # logits at all) — None means "no signal", never "zero confidence".
    confidence: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
