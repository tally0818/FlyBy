'Call depth-tiered OpenRouter models with bounded cost and retries.'
from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Optional

import httpx

from . import protocol

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
UNAVAILABLE_TEXT = "The assistant is temporarily unavailable. Continue on your own."



WORDS_PER_TOKEN = 0.75


class BudgetTracker:
    'Thread-safe process-global USD budget. Prices are $/M tokens.'

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd
        self.usd_spent = 0.0
        self._lock = threading.Lock()

    def charge(self, prompt_tokens: int, completion_tokens: int,
               p_in: float, p_out: float) -> float:
        usd = (prompt_tokens * p_in + completion_tokens * p_out) / 1e6
        with self._lock:
            self.usd_spent += usd
        return usd

    @property
    def exceeded(self) -> bool:
        with self._lock:
            return self.usd_spent >= self.budget_usd


def normalize_tiers(raw: dict) -> dict[int, dict]:
    'Validate a config `depths:` block into {depth: {model, max_tokens, p_in, p_out}}.'
    tiers: dict[int, dict] = {}
    for key, spec in raw.items():
        depth = int(key)
        if depth not in protocol.DEPTHS:
            raise ValueError(f"depth {depth} is not one of {list(protocol.DEPTHS)}")
        missing = {"model", "max_tokens", "p_in", "p_out"} - set(spec)
        if missing:
            raise ValueError(f"depth {depth} tier missing {sorted(missing)}")
        tiers[depth] = {
            "model": str(spec["model"]),
            "max_tokens": int(spec["max_tokens"]),
            "p_in": float(spec["p_in"]),
            "p_out": float(spec["p_out"]),
        }
    if not tiers:
        raise ValueError("no depth tiers configured")
    return tiers


class LRMClient:
    def __init__(
        self,
        depth_tiers: dict,
        budget_usd: float = float("inf"),
        timeout_s: float = 180.0,
        max_retries: int = 2,
        concurrency: Optional[int] = None,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
        thinking: bool = False,
    ):
        self.tiers = normalize_tiers(depth_tiers)
        self.default_depth = max(self.tiers)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.thinking = thinking
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.budget = BudgetTracker(budget_usd)

        n = concurrency or int(os.environ.get("LRM_MAX_CONCURRENCY", "8"))
        self.concurrency = n
        self._sem = threading.Semaphore(n)

        self._client = httpx.Client(timeout=timeout_s, transport=transport)
        self._async_transport = async_transport
        self._aclient: Optional[httpx.AsyncClient] = None



    def tier(self, depth: Optional[int]) -> dict:
        return self.tiers[self.default_depth if depth is None else depth]

    def query(self, q: str, depth: int, problem_id: str) -> dict:
        tier = self.tier(depth)
        contract = protocol.query_contract(int(tier["max_tokens"] * WORDS_PER_TOKEN))
        messages = [{"role": "system", "content": contract}, {"role": "user", "content": q}]
        return self._call(messages, depth, tier, tier["max_tokens"])

    async def aquery(self, q: str, depth: int, problem_id: str) -> dict:
        return await asyncio.to_thread(self.query, q, depth, problem_id)

    def chat(self, messages: list[dict], problem_id: str, max_tokens: Optional[int] = None,
             depth: Optional[int] = None) -> dict:
        'Raw chat — offline data synthesis and the standalone API baseline.'
        tier = self.tier(depth)
        return self._call(messages, depth, tier, max_tokens or tier["max_tokens"])

    async def achat(self, messages: list[dict], problem_id: str,
                    max_tokens: Optional[int] = None, depth: Optional[int] = None) -> dict:
        return await asyncio.to_thread(self.chat, messages, problem_id, max_tokens, depth)



    def _unavailable(self, depth: Optional[int], tier: dict, reason: str) -> dict:
        return {
            "text": UNAVAILABLE_TEXT,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
            "model": tier["model"],
            "kind": protocol.KIND_QUERY,
            "depth": depth,
            "unavailable": True,
            "unavailable_reason": reason,
            "cost_usd": 0.0,
        }

    def _call(self, messages: list[dict], depth: Optional[int], tier: dict,
              max_tokens: int) -> dict:
        if self.budget.exceeded:
            return self._unavailable(depth, tier, "budget_exhausted")
        with self._sem:
            result = self._request_with_retries(messages, max_tokens, tier["model"])
        if result is None:
            return self._unavailable(depth, tier, "request_failed")

        usage = result.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        cost_usd = self.budget.charge(prompt_tokens, completion_tokens,
                                      tier["p_in"], tier["p_out"])
        return {
            "finish_reason": result.get("finish_reason"),
            "text": result["text"],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "model": result.get("model") or tier["model"],
            "kind": protocol.KIND_QUERY,
            "depth": depth,
            "unavailable": False,
            "cost_usd": cost_usd,
        }

    def _request_with_retries(
        self, messages: list[dict], max_tokens: int, model: str
    ) -> Optional[dict]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
        }


        payload["reasoning"] = {"enabled": bool(self.thinking)}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                return {
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "text": text,
                    "usage": data.get("usage", {}),
                    "model": data.get("model") or model,
                }
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        return None
