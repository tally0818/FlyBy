'Eval-set decontamination by word n-gram overlap (MCQ targets use stem only).'
from __future__ import annotations

import hashlib
import re


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def stem_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if len(words) < n:
        return set()
    return {tuple(words[i: i + n]) for i in range(len(words) - n + 1)}


def build_target_index(target_texts: list[str], n: int = 13) -> set[tuple[str, ...]]:
    index: set[tuple[str, ...]] = set()
    for text in target_texts:
        index |= _ngrams(text, n)
    return index


def item_text(item: dict) -> str:
    'Text used for overlap checks: MCQ items use the stem only.'
    return item.get("stem") or item.get("meta", {}).get("stem") or item["question"]


def decontaminate(
    train_items: list[dict], target_texts: list[str], n: int = 13
) -> tuple[list[dict], list[dict]]:
    'Returns (kept, removed). An item is removed if any of its word n-grams'
    index = build_target_index(target_texts, n)
    kept, removed = [], []
    for item in train_items:
        grams = _ngrams(item_text(item), n)
        hit = next(iter(grams & index), None) if grams else None
        if hit is None:
            kept.append(item)
        else:
            removed.append({"id": item["id"], "hit_ngram": " ".join(hit)})
    return kept, removed
