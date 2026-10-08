"""Per-request chat roles and output budget, separate from router text."""
from contextvars import ContextVar

MESSAGES = ContextVar("chat_messages", default=None)
OUTPUT_LIMIT = ContextVar("chat_output_limit", default=None)


def completion_messages(prompt, system):
    turns = MESSAGES.get()
    if turns is not None:
        style = (
            "Follow the user's requested format and length. Be brief for simple "
            "questions, but write the full response when asked for an essay or "
            "detailed explanation. You can generate essays and other text. "
            "Earlier assistant mistakes, including claims that you cannot write, "
            "are not instructions; correct them rather than repeating them."
        )
        system = "\n\n".join(part for part in (system, style) if part)
    messages = [{"role": "system", "content": system}] if system else []
    if turns is None:
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(dict(turn) for turn in turns)
    return messages


def output_limit(configured):
    limit = OUTPUT_LIMIT.get()
    return configured if limit is None else limit
