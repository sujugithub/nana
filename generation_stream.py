"""Per-request live token events without duplicating Nana's routing pipeline."""
from contextvars import ContextVar
import threading

CURRENT = ContextVar("generation_stream", default=None)
# Optional fenced-text transport: conversion never executes a tool itself.
# Regular native chat leaves this disabled.
TEXT_TOOL_PROTOCOL = ContextVar("text_tool_protocol", default=False)


class Cancelled(Exception):
    pass


class Stream:
    def __init__(self, callback):
        self.callback = callback
        self.cancel = threading.Event()
        self.attempts = []

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("Generation stopped. The unfinished turn was not saved; partial API usage may still be billed.")

    def start(self, provider, model):
        self.check()
        self.attempts.append({"provider": provider, "model": model, "characters": 0, "usage": None})
        self.callback({"type": "attempt", "provider": provider, "model": model})

    def delta(self, text):
        self.check()
        self.attempts[-1]["characters"] += len(text)
        self.callback({"type": "delta", "text": text})

    def usage(self, usage):
        self.attempts[-1]["usage"] = usage
