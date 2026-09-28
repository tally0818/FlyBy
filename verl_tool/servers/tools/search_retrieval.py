'Search-R1 tool adapter for the retained verl-tool server interface.'
from __future__ import annotations

import os

from src.tools.search_r1 import (
    SearchR1Client,
    format_information,
    format_search_error,
    parse_search_call,
    system_prompt,
    truncate_result,
)

from .base import BaseTool, register_tool


def _env(name: str, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value not in (None, "") else default


@register_tool
class SearchRetrievalTool(BaseTool):
    'Execute ``<search>`` actions against the canonical Wiki-18 service.'

    tool_type = "search_retrieval"

    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.topk = _env("SEARCH_R1_TOPK", 3, int)
        self.max_query_chars = _env("SEARCH_R1_MAX_QUERY_CHARS", 300, int)
        self.max_result_chars = _env("SEARCH_R1_MAX_RESULT_CHARS", 6000, int)
        self.client = SearchR1Client(
            _env("SEARCH_R1_RETRIEVER_URL", "http://127.0.0.1:8000/retrieve"),
            topk=self.topk,
            timeout_s=_env("SEARCH_R1_TIMEOUT_S", 30.0, float),
            concurrency=_env("SEARCH_R1_CONCURRENCY", 16, int),
        )

    def get_usage_inst(self):
        return system_prompt(
            max_turns=_env("SEARCH_R1_MAX_TURNS", 4, int),
            topk=self.topk,
        )

    def parse_action(self, action: str):
        call = parse_search_call(action)
        return (call.query or "", bool(call and call.valid))

    def get_action_priority(self, action: str, extra_field: dict) -> int:
        return 100 if parse_search_call(action) is not None else -1

    def conduct_action(self, trajectory_id, action, extra_field):
        call = parse_search_call(action)
        env = self.load_env(trajectory_id)
        if call is None or not call.valid:
            reason = call.error if call else "no closed search call found"
            result = {
                "obs": format_search_error(reason),
                "invalid_call": True,
                "invalid_reason": reason,
                "search_usage": {"num_docs": 0, "result_chars": 0, "latency_ms": 0.0},
            }
            valid = False
            query = call.query if call else ""
        elif len(call.query or "") > self.max_query_chars:
            reason = f"search query exceeds {self.max_query_chars} characters"
            result = {
                "obs": format_search_error(reason),
                "invalid_call": True,
                "invalid_reason": reason,
                "search_usage": {"num_docs": 0, "result_chars": 0, "latency_ms": 0.0},
            }
            valid = False
            query = call.query or ""
        else:
            query = call.query or ""
            try:
                search = self.client.search(query)
                if not search["num_docs"] or not search["text"].strip():
                    raise RuntimeError("retriever returned no usable document text")
                rendered = truncate_result(search["text"], self.max_result_chars)
                result = {
                    "obs": format_information(rendered),
                    "search_usage": {
                        "num_docs": search["num_docs"],
                        "result_chars": len(rendered),
                        "latency_ms": search["latency_ms"],
                    },
                }
                valid = True
            except Exception as exc:
                reason = str(exc)
                result = {
                    "obs": format_search_error(reason),
                    "invalid_call": True,
                    "invalid_reason": reason,
                    "search_usage": {"num_docs": 0, "result_chars": 0, "latency_ms": 0.0},
                }
                valid = False

        self.update_env(trajectory_id, env, query, valid, extra_field or {}, result["obs"])
        self.save_env(trajectory_id, env)
        return result, False, valid
