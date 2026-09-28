"""Resumable, seeded vLLM generation and bounded OpenRouter synthesis calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.io import read_json, write_json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def item_seed(seed: int, *parts) -> int:
    return int(digest([seed, *parts])[:8], 16)


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, stage: str, key, produce):
        path = self.root / stage / f"{digest(key)}.json"
        if path.exists():
            record = read_json(path)
            if record["key"] != key:
                raise ValueError(f"Cache key mismatch: {path}")
            return record["value"]
        value = produce()
        write_json(path, {"key": key, "value": value})
        return value


class Engine:
    """Load the GPU model lazily; cache hits need only the tokenizer."""
    def __init__(self, cfg: dict, tokenizer, cache: Cache):
        self.cfg, self.tokenizer, self.cache = cfg, tokenizer, cache
        self.llm = None

    def generate(self, prompt: str, n: int, seed: int, decode: dict, max_tokens=None):
        params = {key: decode[key] for key in ("temperature", "top_p", "top_k")}
        params.update(n=n, seed=seed, max_tokens=int(
            decode["max_new_tokens"] if max_tokens is None else max_tokens))
        if n <= 0 or params["max_tokens"] <= 0:
            raise ValueError("Generation requires positive sample and token counts")
        key = {"model": self.cfg, "prompt": prompt, "params": params}

        def produce():
            from vllm import LLM, SamplingParams

            if self.llm is None:
                self.llm = LLM(
                    model=self.cfg["name"], revision=self.cfg["revision"],
                    tokenizer_revision=self.cfg["revision"], **self.cfg["engine"])
            outputs = self.llm.generate([prompt], SamplingParams(**params), use_tqdm=False)
            texts = [out.text for out in outputs[0].outputs]
            if len(texts) != n:
                raise RuntimeError(f"Expected {n} continuations, got {len(texts)}")
            return texts

        texts = self.cache.get("generations", key, produce)
        if len(texts) != n:
            raise ValueError("Incomplete cached generation")
        return texts


class CachedLRM:
    """Cache successful calls only; API outages never become negative examples.

    The budget counts previously cached calls as well as this invocation's calls.
    Changing credentials, timeouts or the budget does not invalidate successful work.
    """
    def __init__(self, client, cache: Cache, stage: str):
        self.client, self.cache, self.stage = client, cache, stage
        client.budget.usd_spent = sum(
            float(read_json(path)["value"].get("cost_usd", 0))
            for path in (cache.root / stage).glob("*.json"))

    def _call(self, method, args, kwargs):
        key = {"method": method, "args": list(args), "kwargs": kwargs,
               "tiers": {str(k): v for k, v in self.client.tiers.items()},
               "thinking": self.client.thinking, "base_url": self.client.base_url}

        def produce():
            result = getattr(self.client, method)(*args, **kwargs)
            if result.get("unavailable"):
                raise RuntimeError(
                    f"Oracle unavailable ({result.get('unavailable_reason')}); "
                    "fix the API/budget and rerun. Successful work is cached.")
            return result

        return self.cache.get(self.stage, key, produce)

    def chat(self, *args, **kwargs):
        return self._call("chat", args, kwargs)

    def query(self, *args, **kwargs):
        return self._call("query", args, kwargs)
