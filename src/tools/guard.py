'Validate tool queries and redact answer leakage.'
from __future__ import annotations

import re

Q_MAX_CHARS = 300
Q_OVERLAP_NGRAM = 8
Q_OVERLAP_MAX = 0.5
Q_OVERLAP_MIN_QUERY_WORDS = 20
REDACTED = "[redacted]"

_BOXED_RE = re.compile(r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")

_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|option|choice|letter)\s*(?:is|:|=)\s*[^.\n]*",
    re.IGNORECASE,
)




_LONE_LETTER_RE = re.compile(r"^\s*\(?([A-Ja-j])\)?\.?\s*$", re.MULTILINE)


def validate_query(q: str, max_chars: int = Q_MAX_CHARS) -> tuple[bool, str]:
    'Structural check on an outgoing query. Over-cap is a rejection, not a truncation.'
    if not q or not q.strip():
        return False, "empty query"
    if len(q) > max_chars:
        return False, f"query exceeds {max_chars} characters ({len(q)})"
    return True, ""


def filter_leakage(text: str) -> tuple[str, int]:
    'Structural leak filter applied to every observation (train AND eval).'
    n = 0
    for pattern in (_BOXED_RE, _ANSWER_IS_RE, _LONE_LETTER_RE):
        text, k = pattern.subn(REDACTED, text)
        n += k
    return text, n


def redact_gold(text: str, gold: str, answer_format: str = "math") -> tuple[str, int]:
    'Training-only redaction: the gold answer never reaches the policy via an observation.'
    if not gold:
        return text, 0
    n = 0
    if answer_format == "mcq":
        letter = gold.strip().upper()[:1]
        pat = re.compile(rf"(?<![A-Za-z])\(?{letter}\)?(?![A-Za-z])")

        text, n = pat.subn(REDACTED, text)
    else:
        g = gold.strip()
        if g:
            pat = re.compile(rf"(?<![\w.]){re.escape(g)}(?!\w|\.\d)")
            text, n = pat.subn(REDACTED, text)
    return text, n


def _word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i: i + n]) for i in range(len(words) - n + 1)}


def q_overlap(q: str, problem: str, n: int = Q_OVERLAP_NGRAM) -> float:
    "Fraction of q's word n-grams that also occur in the problem text."
    q_words = re.findall(r"[a-z0-9]+", q.lower())
    if not q_words or not problem.strip():
        return 0.0
    eff_n = min(n, len(q_words))
    q_grams = _word_ngrams(q, eff_n)
    p_grams = _word_ngrams(problem, eff_n)
    if not q_grams:
        return 0.0
    return len(q_grams & p_grams) / len(q_grams)


def check_query_overlap(q: str, problem: str) -> tuple[bool, float]:
    'Return ``(reject, overlap)``; short concept queries are always allowed.'
    overlap = q_overlap(q, problem)
    query_word_count = len(re.findall(r"[a-z0-9]+", q.lower()))
    reject = query_word_count >= Q_OVERLAP_MIN_QUERY_WORDS and overlap > Q_OVERLAP_MAX
    return reject, overlap
