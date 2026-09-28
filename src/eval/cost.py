'Compute local inference and external API costs.'
from __future__ import annotations

import json
from pathlib import Path


def cost_local(local_input_tokens: int, own_tokens: int, p_gpu_hr: float,
               prefill_tok_s: float, decode_tok_s: float) -> float:
    if prefill_tok_s <= 0 or decode_tok_s <= 0:
        raise ValueError("prefill_tok_s and decode_tok_s must be > 0 "
                         "(run bench_throughput.sh first)")
    seconds = local_input_tokens / prefill_tok_s + own_tokens / decode_tok_s
    return p_gpu_hr * seconds / 3600.0


def cost_api(calls: list[dict], tiers: dict[int, dict]) -> float:
    'Return API cost for tool calls, pricing each by its depth tier.'
    if not tiers:
        raise ValueError("no depth tiers supplied; cannot price API calls")
    default = max(tiers)
    total = 0.0
    for call in calls:
        depth = call.get("depth")
        tier = tiers.get(default if depth is None else depth)
        if tier is None:
            raise KeyError(f"call recorded depth {depth!r} with no matching tier "
                           f"(configured: {sorted(tiers)})")
        total += call.get("in_toks", 0) * tier["p_in"] + call.get("out_toks", 0) * tier["p_out"]
    return total / 1e6


def load_throughput(path: str | Path) -> dict[str, dict[str, float]]:
    with open(path) as f:
        raw = json.load(f)
    table = {}
    for model, rates in raw.items():
        if not isinstance(rates, dict):
            raise ValueError(f"legacy scalar throughput for {model}; rerun bench_throughput.sh")
        table[model] = {
            "prefill_tok_s": float(rates["prefill_tok_s"]),
            "decode_tok_s": float(rates["decode_tok_s"]),
        }
    return table


def throughput_for(throughput: dict[str, dict[str, float]], model: str) -> dict[str, float]:
    if model in throughput:
        return throughput[model]
    base = model.rstrip("/").split("/")[-1]
    for key, val in throughput.items():
        if key.rstrip("/").split("/")[-1] == base:
            return val
    raise KeyError(f"no throughput entry for {model}; available: {list(throughput)}")


def trajectory_cost(
    local_input_tokens: int,
    own_tokens: int,
    calls: list[dict],
    p_gpu_hr: float,
    prefill_tok_s: float,
    decode_tok_s: float,
    tiers: dict[int, dict],
) -> dict:
    local = cost_local(local_input_tokens, own_tokens, p_gpu_hr, prefill_tok_s, decode_tok_s) \
        if local_input_tokens or own_tokens else 0.0
    api = cost_api(calls, tiers)
    return {"cost_local": local, "cost_api": api, "cost_usd": local + api}
