'Grading — math_verify for math, letter match for MCQ.'
from __future__ import annotations

import re

_BOXED_START = "\\boxed{"


def extract_boxed(text: str) -> str:
    'Contents of the last \\boxed{...}, with brace matching.'
    start = text.rfind(_BOXED_START)
    if start < 0:
        return ""
    i = start + len(_BOXED_START)
    depth = 1
    out = []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()




_LETTER_RE = re.compile(r"^\(?\s*([A-Ja-j])\s*\)?\.?$")




_ANSWER_LETTER_RE = re.compile(
    r"answer\s*(?:is|:)\s*(?:\(\s*([A-Ja-j])\s*\)|([A-Ja-j])(?![A-Za-z0-9]))",
    re.IGNORECASE,
)
_AMBIGUOUS_BARE = {"A", "I"}
_CLAUSE_END = re.compile(r"^\s*(?:[.,;:!?)\]]|$)")
_A_TO_Z_LETTER_RE = re.compile(r"^\(?\s*([A-Za-z])\s*\)?\.?$")
_A_TO_Z_ANSWER_LETTER_RE = re.compile(
    r"answer\s*(?:is|:)\s*(?:\(\s*([A-Za-z])\s*\)|([A-Za-z])(?![A-Za-z0-9]))",
    re.IGNORECASE,
)


def extract_letter(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        m = _LETTER_RE.match(boxed)
        if m:
            return m.group(1).upper()
        return ""
    found = ""
    for m in _ANSWER_LETTER_RE.finditer(text):
        letter = (m.group(1) or m.group(2)).upper()
        if m.group(2) and letter in _AMBIGUOUS_BARE and not _CLAUSE_END.match(text[m.end():]):
            continue
        found = letter
    return found


def grade_mcq(response_text: str, gold_letter: str) -> bool:
    return extract_letter(response_text) == gold_letter.strip().upper()


def extract_a_to_z_letter(text: str) -> str:
    'Extract a boxed/final A-Z choice for benchmarks with more than ten options.'
    boxed = extract_boxed(text)
    if boxed:
        match = _A_TO_Z_LETTER_RE.match(boxed)
        return match.group(1).upper() if match else ""
    found = ""
    for match in _A_TO_Z_ANSWER_LETTER_RE.finditer(text):
        letter = (match.group(1) or match.group(2)).upper()
        if (
            match.group(2)
            and letter in _AMBIGUOUS_BARE
            and not _CLAUSE_END.match(text[match.end() :])
        ):
            continue
        found = letter
    return found


def grade_mcq_a_to_z(response_text: str, gold_letter: str) -> bool:
    return extract_a_to_z_letter(response_text) == gold_letter.strip().upper()


def _norm_math(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    return s.rstrip(".")


def grade_math(response_text: str, gold: str) -> bool:
    answer = extract_boxed(response_text)
    if not answer:
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse(f"${gold}$"), parse(f"${answer}$")))
    except Exception:
        return _norm_math(answer) == _norm_math(gold)


def grade(answer_format: str, response_text: str, gold: str) -> bool:
    if answer_format == "mcq":
        return grade_mcq(response_text, gold)
    if answer_format == "mcq_a_to_z":
        return grade_mcq_a_to_z(response_text, gold)
    return grade_math(response_text, gold)
