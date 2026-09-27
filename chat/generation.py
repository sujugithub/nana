import threading
from typing import Dict, Optional


_active_generations: Dict[int, threading.Event] = {}
_lock = threading.Lock()


def start_generation(conversation_id: int) -> threading.Event:
    """
    Register a new active generation for a conversation.

    If one is already running, stop it first.
    """

    with _lock:
        existing = _active_generations.get(conversation_id)

        if existing is not None:
            existing.set()

        stop_event = threading.Event()

        _active_generations[conversation_id] = stop_event

        return stop_event


def get_generation(
    conversation_id: int,
) -> Optional[threading.Event]:
    with _lock:
        return _active_generations.get(conversation_id)


def stop_generation(conversation_id: int) -> bool:
    """
    Signal the active generation to stop.

    Returns True if something was actually running.
    """

    with _lock:
        stop_event = _active_generations.get(conversation_id)

        if stop_event is None:
            return False

        stop_event.set()

        return True


def finish_generation(conversation_id: int) -> None:
    """
    Remove generation from the active registry.
    """

    with _lock:
        _active_generations.pop(
            conversation_id,
            None,
        )