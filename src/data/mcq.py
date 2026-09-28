'MCQ rendering convention — seeded shuffle, gold letter, permutation record.'
from __future__ import annotations

import random

LETTERS = "ABCDEFGHIJ"
BOXED_LETTER_INSTRUCTION = (
    "Answer with the letter of the correct choice. "
    "Put your final answer within \\boxed{}, e.g. \\boxed{A}."
)


def permute(n_choices: int, shuffle_seed: int, qid: str) -> list[int]:
    rng = random.Random(f"{shuffle_seed}:{qid}")
    perm = list(range(n_choices))
    rng.shuffle(perm)
    return perm


def render_mcq(
    stem: str,
    correct: str,
    incorrect: list[str],
    shuffle_seed: int,
    qid: str,
    letters: str = LETTERS,
) -> dict:
    'Returns {question, choices, gold, perm, stem}. `choices` is in displayed order.'
    original = [correct] + list(incorrect)
    if len(original) > len(letters):
        raise ValueError(f"{len(original)} choices exceed letter set {letters!r}")
    perm = permute(len(original), shuffle_seed, qid)
    displayed = [original[j] for j in perm]
    gold_pos = perm.index(0)
    gold_letter = letters[gold_pos]
    lines = [stem.strip(), ""]
    for i, choice in enumerate(displayed):
        lines.append(f"({letters[i]}) {choice.strip()}")
    lines += ["", BOXED_LETTER_INSTRUCTION]
    return {
        "question": "\n".join(lines),
        "choices": displayed,
        "gold": gold_letter,
        "perm": perm,
        "stem": stem.strip(),
    }
