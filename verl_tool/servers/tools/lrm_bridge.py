'effieReasoner LRM bridge tool: depth-tiered `query` backed by an OpenRouter LRM.'
import json
import os
import sys
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools import guard, protocol
from src.tools.lrm_client import LRMClient

from .base import BaseTool, register_tool


def _env(name, default, cast=str):
    val = os.environ.get(name)
    return cast(val) if val not in (None, "") else default


def _invalid_observation(
    reason: str, error_code: str, q: Optional[str] = None
) -> dict:
    'Return a rejected attempt with stable, machine-readable diagnostics.'
    return {
        "obs": protocol.format_tool_error(reason),
        "invalid_call": True,
        "invalid_reason": reason,
        "invalid_error_code": error_code,
        "invalid_category": protocol.invalid_category(error_code),
        "query_chars": len(q) if q is not None else 0,
    }


@register_tool
class LRMBridgeTool(BaseTool):
    tool_type = "lrm_bridge"

    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.q_max_chars = _env("EFFIE_Q_MAX_CHARS", guard.Q_MAX_CHARS, int)
        self.gold_redaction = _env("EFFIE_GOLD_REDACTION", "true").lower() in ("1", "true")
        tiers_json = os.environ.get("EFFIE_DEPTH_TIERS")
        if not tiers_json:
            raise RuntimeError(
                "EFFIE_DEPTH_TIERS is unset; the tool server and the reward manager "
                "must be given the same depth tiers or trajectories would be mispriced"
            )
        self.client = LRMClient(
            depth_tiers=json.loads(tiers_json),
            budget_usd=_env("EFFIE_BUDGET_USD", 500.0, float),
            timeout_s=_env("EFFIE_TIMEOUT_S", 180.0, float),
            max_retries=_env("EFFIE_MAX_RETRIES", 2, int),
            concurrency=_env("EFFIE_CONCURRENCY", 8, int),
            thinking=_env("EFFIE_THINKING", "false").lower() in ("1", "true"),
        )

    def get_usage_inst(self):
        return protocol.system_prompt()

    def parse_action(self, action: str):
        call = protocol.parse_tool_call(action)
        if call is None:
            return "", False

        return action, True

    def get_action_priority(self, action: str, extra_field: dict) -> int:
        return 1 if protocol.parse_tool_call(action) is not None else -1

    def _usage_meta(
        self, result: dict, depth: int, q_overlap: float, query_chars: int
    ) -> dict:
        return {
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "reasoning_tokens": result["usage"].get("reasoning_tokens", 0),
            "kind": protocol.KIND_QUERY,
            "depth": depth,
            "model": result.get("model"),
            "cost_usd": result.get("cost_usd", 0.0),
            "unavailable": result.get("unavailable", False),
            "q_overlap": q_overlap,
            "query_chars": query_chars,
        }

    def conduct_action(self, trajectory_id, action, extra_field):
        call = protocol.parse_tool_call(action)
        env = self.load_env(trajectory_id)
        extra_field = extra_field or {}
        problem_id = str(extra_field.get("id", trajectory_id))
        problem_text = extra_field.get("stem") or extra_field.get("question") or ""
        gold = str(extra_field.get("gold", "") or "")
        answer_format = extra_field.get("answer_format", "math")

        if call is None:

            obs = _invalid_observation(
                "no tool call found", protocol.ERROR_MISSING_CALL
            )
            valid = False
        elif not call.valid:
            reason = call.error or "malformed tool call"
            obs = _invalid_observation(
                reason,
                call.error_code or protocol.ERROR_MISSING_CALL,
                call.q,
            )
            valid = False
        else:
            text, valid, meta = self._do_query(call, problem_id, problem_text, gold, answer_format)
            obs = {"obs": text, "llm_usage": meta}
            if not valid:
                obs.update({
                    "invalid_call": True,
                    "invalid_reason": meta["rejected"],
                    "invalid_error_code": meta["rejection_code"],
                    "invalid_category": protocol.invalid_category(
                        meta["rejection_code"]
                    ),
                    "query_chars": meta["query_chars"],
                })

        self.update_env(trajectory_id, env, action, valid, extra_field, obs["obs"])
        self.save_env(trajectory_id, env)
        return obs, False, valid

    def _do_query(self, call, problem_id, problem_text, gold, answer_format):
        ok, reason = guard.validate_query(call.q, self.q_max_chars)
        rejection_code = None
        if not ok:
            rejection_code = (
                protocol.ERROR_EMPTY_QUERY
                if not call.q or not call.q.strip()
                else protocol.ERROR_QUERY_TOO_LONG
            )
        reject_overlap, overlap = guard.check_query_overlap(call.q or "", problem_text)
        if ok and reject_overlap:
            ok, reason = False, "query overlaps the problem statement too much; ask about concepts instead"
            rejection_code = protocol.ERROR_QUERY_OVERLAP
        if not ok:
            meta = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                    "kind": protocol.KIND_QUERY, "depth": call.depth,
                    "unavailable": False, "rejected": reason,
                    "rejection_code": rejection_code or protocol.ERROR_MISSING_CALL,
                    "q_overlap": overlap, "query_chars": len(call.q or "")}
            return protocol.format_tool_error(reason), False, meta
        result = self.client.query(call.q, call.depth, problem_id)
        text, _ = guard.filter_leakage(result["text"])
        if self.gold_redaction:
            text, _ = guard.redact_gold(text, gold, answer_format)
        return protocol.format_lrm_answer(text), True, self._usage_meta(
            result, call.depth, overlap, len(call.q)
        )
