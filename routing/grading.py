"""Deterministic answer grading for outcome collection — without guessing.

``grade_answer`` returns a structured ``GradeResult`` instead of a bare float:

    status     "graded" (score is trustworthy) or "needs_review" (a human
               must decide; score is a 0.0 placeholder and must not be used)
    score      0..1 correctness, meaningful only when status == "graded"
    flags      response conditions recorded SEPARATELY from correctness:
               empty, refusal, truncated, thinking_removed,
               unclosed_thinking, contradictory, ambiguous, no_final_answer
    extracted  the answer token/number the grader compared, if any
    method     how it was found (final_marker, leading_answer, sole_mention,
               sole_number, trailing_number, exact, contains, ...)

Rules (EVALUATION.md: graders must be unarguable):

- The raw response is never modified. Thinking blocks (<think>...</think>)
  and markdown emphasis are removed from a separate grading copy only.
- An explicit final answer ("Answer: B", "the answer is 42", "#### 42",
  \\boxed{42}) wins over anything else in the text.
- Several different explicit answers, or a leading answer that disagrees
  with the explicit one, is ``contradictory`` -> review.
- A multiple-choice response that also mentions other options ("A. Paris
  (not B)") is ``ambiguous`` -> review. No regex here tries to interpret
  negation.
- No recognisable answer -> review, unless the response is empty, a refusal,
  or truncated, which are graded 0.0 with that flag: the model produced no
  answer, which is a model outcome, not a grading uncertainty.
- Infrastructure failures (timeouts, HTTP errors) are not graded here; the
  collector records them separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

GRADED = "graded"
NEEDS_REVIEW = "needs_review"

EMPTY = "empty"
REFUSAL = "refusal"
TRUNCATED = "truncated"
THINKING_REMOVED = "thinking_removed"
UNCLOSED_THINKING = "unclosed_thinking"
CONTRADICTORY = "contradictory"
AMBIGUOUS = "ambiguous"
NO_FINAL_ANSWER = "no_final_answer"

# Provider finish reasons meaning "stopped at the output-token cap"
# (OpenAI-compatible "length"; Ollama done_reason "length").
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


@dataclass(frozen=True)
class GradeResult:
    status: str
    score: float
    flags: Tuple[str, ...] = ()
    extracted: Optional[str] = None
    method: str = ""

    @property
    def needs_review(self) -> bool:
        return self.status == NEEDS_REVIEW

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "flags": list(self.flags),
            "extracted": self.extracted,
            "method": self.method,
        }


# ── Grading copy ─────────────────────────────────────────────────────────

_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1\s*>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(?:think|thinking)>", re.I)
_THINK_CLOSE = re.compile(r"</(?:think|thinking)\s*>", re.I)
_EMPHASIS = re.compile(r"\*\*|__|`")


def grading_copy(raw: Optional[str]) -> Tuple[str, List[str]]:
    """Return (text to grade, flags). ``raw`` itself is left untouched."""
    text = raw or ""
    flags: List[str] = []
    copy = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(copy))
    if closes:  # template emitted only the closing tag: answer follows it
        copy = copy[closes[-1].end():]
    if copy != text:
        flags.append(THINKING_REMOVED)
    opening = _THINK_OPEN.search(copy)
    if opening:  # reasoning never finished; nothing after it is an answer
        copy = copy[:opening.start()]
        flags.append(UNCLOSED_THINKING)
    return _EMPHASIS.sub("", copy).strip(), flags


_REFUSAL = re.compile(
    r"^\W*(?:i\s*(?:['’]m|am)\s+sorry\b"
    r"|i\s*(?:can['’]t|cannot|can\s+not|won['’]t|will\s+not"
    r"|(?:['’]m|am)\s+unable\s+to)"
    r"\s+(?:help|answer|assist|provide|comply|do\s+that)\b"
    r"|as\s+an\s+ai\b)",
    re.I,
)


def _no_answer(flags: List[str], copy: str, truncated: bool) -> GradeResult:
    """No answer was extracted: a model outcome (0.0) or a review case."""
    if not copy:
        return GradeResult(GRADED, 0.0, tuple(flags + [EMPTY]), None, "empty")
    if _REFUSAL.search(copy):
        return GradeResult(GRADED, 0.0, tuple(flags + [REFUSAL]), None, "refusal")
    if truncated:
        return GradeResult(GRADED, 0.0, tuple(flags), None, "truncated_no_final")
    return GradeResult(
        NEEDS_REVIEW, 0.0, tuple(flags + [NO_FINAL_ANSWER]), None, "none"
    )


def _review(flags: List[str], flag: str, extracted: Optional[str]) -> GradeResult:
    return GradeResult(NEEDS_REVIEW, 0.0, tuple(flags + [flag]), extracted, "review")


# ── Multiple choice ──────────────────────────────────────────────────────

# Option letters. "I" is excluded: it is far more often the pronoun.
_LETTERS = "ABCDEFGH"
_TRUTH_SETS: Sequence[frozenset] = (
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
)
# Case-insensitive answer phrases; the captured option stays case-sensitive
# for letters so "the answer is a bit unclear" never reads as option A.
_CUE = (
    r"(?i:\b(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|final"
    r"|option|choice)\b\s*(?:is\s*:?|:|=|-)?\s*)"
)
_STANDALONE_END = r"(?=\s*(?:[.,;:!)\]\"'\n]|$))"
_LETTER_CUE = re.compile(_CUE + r"[(\[]?([A-H])(?![A-Za-z0-9])")
_LETTER_BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-H])\)?")
_LETTER_LEADING = re.compile(r"^[\s\"'\[(]*([A-H])(?:\s*$|[\]).:,]|\s+[-\u2013\u2014])")
_LETTER_MENTION = re.compile(r"(?<![A-Za-z0-9])([A-H])(?![A-Za-z0-9])")
_WORD_CUE = re.compile(
    _CUE + r"[(\[\"']?(true|false|yes|no)\b" + _STANDALONE_END, re.I
)
_WORD_BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*(true|false|yes|no)\b", re.I)
_WORD_LEADING = re.compile(
    r"^[\s\"'\[(]*(true|false|yes|no)\b" + _STANDALONE_END, re.I
)


def _letter_mentions(copy: str) -> List[str]:
    found = []
    for m in _LETTER_MENTION.finditer(copy):
        letter = m.group(1)
        before = copy[max(0, m.start() - 1):m.start()]
        after = copy[m.end():m.end() + 2]
        # "A bakery..." is an article, not option A — unless written "(A".
        if letter == "A" and before != "(" and re.match(r"\s[a-z]", after):
            continue
        found.append(letter)
    return found


def _grade_choice(reference: str, copy: str, flags: List[str],
                  truncated: bool) -> GradeResult:
    ref = reference.strip().strip("()").strip()
    if len(ref) == 1 and ref.upper() in _LETTERS:
        expected = ref.upper()
        cues = [m.group(1) for m in _LETTER_CUE.finditer(copy)]
        cues += [m.group(1) for m in _LETTER_BOXED.finditer(copy)]
        lead = _LETTER_LEADING.match(copy)
        leading = lead.group(1) if lead else None
        mentions = set(_letter_mentions(copy))
    else:
        expected = ref.lower()
        vocab = next((s for s in _TRUTH_SETS if expected in s), None)
        if vocab is None:
            # Unsupported reference shape: only an exact whole-answer match
            # is unarguable; anything else goes to a human.
            if copy.strip(" .").lower() == expected:
                return GradeResult(GRADED, 1.0, tuple(flags), expected, "exact")
            return _no_answer(flags, copy, truncated)
        cues = [m.group(1).lower() for m in _WORD_CUE.finditer(copy)]
        cues += [m.group(1).lower() for m in _WORD_BOXED.finditer(copy)]
        cues = [c for c in cues if c in vocab]
        lead = _WORD_LEADING.match(copy)
        leading = lead.group(1).lower() if lead and lead.group(1).lower() in vocab else None
        mentions = {
            w.lower() for w in re.findall(r"\b(true|false|yes|no)\b", copy, re.I)
            if w.lower() in vocab
        }

    if len(set(cues)) > 1:
        return _review(flags, CONTRADICTORY, None)
    if cues:
        value = cues[0]
        if leading is not None and leading != value:
            return _review(flags, CONTRADICTORY, None)
        method = "final_marker"
    elif leading is not None:
        if mentions - {leading}:
            return _review(flags, AMBIGUOUS, leading)
        value, method = leading, "leading_answer"
    elif len(mentions) == 1 and not truncated:
        value, method = next(iter(mentions)), "sole_mention"
    elif len(mentions) > 1:
        return _review(flags, AMBIGUOUS, None)
    else:
        return _no_answer(flags, copy, truncated)
    score = 1.0 if value == expected else 0.0
    return GradeResult(GRADED, score, tuple(flags), value, method)


# ── Numeric ──────────────────────────────────────────────────────────────

_NUM = (
    r"([-\u2212]?\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:\s*/\s*\d+(?:\.\d+)?)?%?)"
)
_NOT_EXPRESSION = r"(?!\s*[-+*/\u00d7=]\s*[\d(])"
_NUM_CUE = re.compile(
    r"(?i:\b(?:final\s+answer|answer|final)\b\s*(?:is\s*:?|:|=)?\s*"
    r"(?:approximately\s+|about\s+)?)" + _NUM + _NOT_EXPRESSION
)
_NUM_HASH = re.compile(r"####\s*" + _NUM)
_NUM_BOXED = re.compile(r"\\boxed\{\s*" + _NUM)
_NUM_ANY = re.compile(r"(?<![\w.])" + _NUM)
_NUM_SOLE = re.compile(r"^" + _NUM + r"\s*(?:[A-Za-z%]+)?\s*\.?$")
_NUM_TAIL = re.compile(r"^\s*(?:[A-Za-z%]+\s*){0,2}[.!]?\s*$")


def _to_fraction(token: str) -> Fraction:
    cleaned = (
        token.replace("\u2212", "-").replace("$", "").replace(",", "")
        .replace("%", "").replace(" ", "")
    )
    if "/" in cleaned:
        top, bottom = cleaned.split("/", 1)
        return Fraction(top) / Fraction(bottom)
    return Fraction(cleaned)


def _numbers_equal(a: Fraction, b: Fraction) -> bool:
    return abs(a - b) <= Fraction(1, 10**6) * max(1, abs(b))


def _grade_numeric(reference: str, copy: str, flags: List[str],
                   truncated: bool) -> GradeResult:
    expected = _to_fraction(reference.strip())  # ValueError: bad task file

    def parsed(tokens):
        out = []
        for token in tokens:
            try:
                out.append((token.strip(), _to_fraction(token)))
            except (ValueError, ZeroDivisionError):
                continue
        return out

    cues = parsed(
        [m.group(1) for m in _NUM_HASH.finditer(copy)]
        + [m.group(1) for m in _NUM_BOXED.finditer(copy)]
        + [m.group(1) for m in _NUM_CUE.finditer(copy)]
    )
    if cues:
        first = cues[0][1]
        if any(not _numbers_equal(v, first) for _, v in cues[1:]):
            return _review(flags, CONTRADICTORY, None)
        token, value, method = cues[0][0], first, "final_marker"
    else:
        sole = _NUM_SOLE.match(copy)
        found = None
        if sole:
            found = (parsed([sole.group(1)]) or [None])[0]
            method = "sole_number"
        elif not truncated:
            # The prompt protocol asks the model to FINISH with the answer:
            # accept a number only when at most two words (units) follow.
            matches = list(_NUM_ANY.finditer(copy))
            if matches and _NUM_TAIL.match(copy[matches[-1].end():]):
                found = (parsed([matches[-1].group(1)]) or [None])[0]
                method = "trailing_number"
        if found is None:
            return _no_answer(flags, copy, truncated)
        token, value = found
    score = 1.0 if _numbers_equal(value, expected) else 0.0
    return GradeResult(GRADED, score, tuple(flags), token, method)


# ── Entry point ──────────────────────────────────────────────────────────

def grade_answer(
    grader: str,
    reference: str,
    answer: Optional[str],
    finish_reason: Optional[str] = None,
) -> GradeResult:
    """Grade ``answer`` against ``reference`` with the named grader
    (exact | numeric | contains | choice). ``finish_reason`` is the
    provider's stop reason when known; "length" marks truncation."""
    if grader not in ("exact", "numeric", "contains", "choice"):
        raise ValueError(f"unknown grader {grader!r}")
    copy, flags = grading_copy(answer)
    truncated = (finish_reason or "").strip().lower() in TRUNCATION_FINISH_REASONS
    if truncated:
        flags.append(TRUNCATED)
    if not copy:
        return _no_answer(flags, copy, truncated)

    if grader == "numeric":
        return _grade_numeric(reference, copy, flags, truncated)
    if grader == "choice":
        return _grade_choice(reference, copy, flags, truncated)

    answer_norm = copy.lower()
    ref_norm = reference.strip().lower()
    if grader == "exact":
        hit = answer_norm == ref_norm
    else:  # contains: unchanged substring semantics
        hit = ref_norm in answer_norm
    if not hit and _REFUSAL.search(copy):
        flags.append(REFUSAL)
    return GradeResult(GRADED, 1.0 if hit else 0.0, tuple(flags),
                       copy if grader == "exact" else None, grader)
