from dataclasses import dataclass
from typing import Optional


@dataclass
class Conversation:
    id: int
    title: str
    mode: str
    model: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str
    route: Optional[str] = None
    model_name: Optional[str] = None
    latency_s: Optional[float] = None
    estimated_cost_usd: Optional[float] = None