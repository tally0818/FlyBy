'Shared token-budget helpers for tool inputs.'
from __future__ import annotations

from typing import Any


def resolve_own_token_budgets(
    response_length: int,
    max_own_tokens_total: int | None = None,
    train_max_own_tokens_total: int | None = None,
    val_max_own_tokens_total: int | None = None,
) -> tuple[int, int, int]:
    'Resolve the base, train, and validation cumulative decode budgets.'
    values = {
        "response_length": response_length,
        "max_own_tokens_total": max_own_tokens_total,
        "train_max_own_tokens_total": train_max_own_tokens_total,
        "val_max_own_tokens_total": val_max_own_tokens_total,
    }
    for name, value in values.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    base = response_length if max_own_tokens_total is None else max_own_tokens_total
    train = base if train_max_own_tokens_total is None else train_max_own_tokens_total
    val = base if val_max_own_tokens_total is None else val_max_own_tokens_total
    return int(base), int(train), int(val)


def generation_allowance(
    total_budget: int,
    generated_tokens: int,
    live_capacity: int,
    per_turn_cap: int,
) -> int:
    'Return the next decode allowance without refunding discarded tokens.'
    if total_budget < 0 or generated_tokens < 0 or per_turn_cap < 0:
        raise ValueError("token budgets and counts must be non-negative")
    cumulative_remaining = total_budget - generated_tokens
    return max(0, min(cumulative_remaining, live_capacity, per_turn_cap))


def truncate_tail_tokens(text: str, max_tokens: int, tokenizer: Any) -> str:
    'Keep at most the newest ``max_tokens`` according to ``tokenizer``.'
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=False)
