'Compute correctness and cost-aware training rewards.'
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verl import DataProto
from verl.workers.reward_manager import register

from src.eval.cost import load_throughput, throughput_for, trajectory_cost
from src.eval.grader import extract_boxed, extract_letter, grade
from src.tools import protocol
from src.tools.lrm_client import normalize_tiers


COST_NORMALIZATION_EPS = 1e-8


def _all_wrong_group_mask(group_keys, correct: list[float]) -> list[bool]:
    'Mark samples whose prompt group contains no correct rollout.'
    if len(group_keys) != len(correct):
        raise ValueError(
            f"group key / correctness length mismatch: {len(group_keys)} != {len(correct)}"
        )
    group_has_correct = defaultdict(bool)
    for key, score in zip(group_keys, correct):
        group_has_correct[key] = group_has_correct[key] or bool(score)
    return [not group_has_correct[key] for key in group_keys]


@register("effi")
class EffiRewardManager:
    name = "effi"

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source",
                 lam=1.0, norm_cost=False, p_gpu_hr=3.50, depth_tiers=None,
                 throughput_json="outputs/bench/throughput.json",
                 throughput_model="Qwen/Qwen3-4B", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.lam = float(lam)
        self.norm_cost = bool(norm_cost)
        self.p_gpu_hr = float(p_gpu_hr)
        if not depth_tiers:
            raw = os.environ.get("EFFIE_DEPTH_TIERS")
            if not raw:
                raise ValueError(
                    "no depth tiers: pass `depth_tiers` or set EFFIE_DEPTH_TIERS. The cost "
                    "axis must match the tiers the tool server actually calls"
                )
            depth_tiers = json.loads(raw)
        self.tiers = normalize_tiers(depth_tiers)
        table = load_throughput(throughput_json)
        self.throughput = throughput_for(table, throughput_model)

    def _reward_components(self, correct: float, reward_cost: float) -> tuple[float, float]:
        'Return correct-only reward and its cost penalty.'
        cost_penalty = self.lam * reward_cost
        return correct * (1.0 - cost_penalty), cost_penalty

    @staticmethod
    def _task_minmax_costs(group_keys, costs: list[float]) -> list[float]:
        'Min-max normalize costs within each prompt group; equal costs map to zero.'
        if len(group_keys) != len(costs):
            raise ValueError(
                f"group key / cost length mismatch: {len(group_keys)} != {len(costs)}"
            )
        group_costs = defaultdict(list)
        for key, cost in zip(group_keys, costs):
            group_costs[key].append(float(cost))

        bounds = {
            key: (min(values), max(values)) for key, values in group_costs.items()
        }
        normalized = []
        for key, cost in zip(group_keys, costs):
            cost_min, cost_max = bounds[key]
            cost_range = cost_max - cost_min
            normalized.append(
                (float(cost) - cost_min) / (cost_range + COST_NORMALIZATION_EPS)
            )
        return normalized

    @staticmethod
    def _has_valid_answer_format(answer_format: str, response_text: str) -> bool:
        'Use the same accepted answer syntax as the shared evaluator.'
        if answer_format == "mcq":
            return bool(extract_letter(response_text))
        return bool(extract_boxed(response_text))

    @staticmethod
    def _prompt_group_keys(data) -> list:
        'Return the same per-prompt uid that GRPO uses to group rollouts.'
        raw_uids = data.non_tensor_batch.get("uid")
        if raw_uids is not None:
            uids = np.asarray(raw_uids, dtype=object)
            return [uids.item()] if uids.ndim == 0 else uids.tolist()



        raw_extras = data.non_tensor_batch.get("extra_info")
        if raw_extras is not None:
            extras = np.asarray(raw_extras, dtype=object)
            extras = [extras.item()] if extras.ndim == 0 else extras.tolist()
            if all(isinstance(extra, dict) and extra.get("id") is not None for extra in extras):
                return [str(extra["id"]) for extra in extras]

        raise ValueError(
            "group-gated Effi reward requires `uid` or `extra_info.id` for every rollout"
        )



    def _tool_stats(self, data_item) -> dict:
        infos = data_item.non_tensor_batch.get("tool_interact_info")
        if infos is None:
            infos = []
        elif isinstance(infos, np.ndarray):
            infos = infos.tolist()
        if isinstance(infos, dict):
            infos = [infos]


        if len(infos) == 1 and isinstance(infos[0], list):
            infos = infos[0]
        calls, q_overlap_max = [], 0.0
        depth_counts = {d: 0 for d in protocol.DEPTHS}
        invalid_by_category = {
            protocol.INVALID_SYNTAX: 0,
            protocol.INVALID_DEPTH: 0,
            protocol.INVALID_EMPTY: 0,
            protocol.INVALID_LENGTH: 0,
            protocol.INVALID_OVERLAP: 0,
            protocol.INVALID_OTHER: 0,
        }
        reasoning_tokens = invalid_calls = 0
        attempted_calls = 0
        invalid_seen = False
        retry_after_invalid = 0
        for info in infos:
            if not isinstance(info, dict):
                continue
            if invalid_seen:
                retry_after_invalid = 1
            attempted_calls += 1
            usage = info.get("llm_usage")
            if not usage:





                invalid_calls += 1
                invalid_seen = True
                category = info.get("invalid_category") or protocol.invalid_category(
                    info.get("invalid_error_code")
                )
                if category not in invalid_by_category:
                    category = protocol.INVALID_OTHER
                invalid_by_category[category] += 1
                continue


            q_overlap_max = max(q_overlap_max, float(usage.get("q_overlap") or 0.0))
            if usage.get("rejected"):
                invalid_calls += 1
                invalid_seen = True
                category = info.get("invalid_category") or protocol.invalid_category(
                    info.get("invalid_error_code") or usage.get("rejection_code")
                )


                if category == protocol.INVALID_OTHER:
                    reason = str(usage.get("rejected", "")).lower()
                    if "exceeds" in reason:
                        category = protocol.INVALID_LENGTH
                    elif "overlap" in reason:
                        category = protocol.INVALID_OVERLAP
                    elif "empty" in reason:
                        category = protocol.INVALID_EMPTY
                if category not in invalid_by_category:
                    category = protocol.INVALID_OTHER
                invalid_by_category[category] += 1
                continue
            depth = usage.get("depth")
            calls.append({"depth": depth,
                          "in_toks": usage.get("prompt_tokens", 0),
                          "out_toks": usage.get("completion_tokens", 0)})
            if depth in depth_counts:
                depth_counts[depth] += 1
            reasoning_tokens += int(usage.get("reasoning_tokens") or 0)
        depths = [c["depth"] for c in calls if c["depth"] is not None]
        metrics = data_item.non_tensor_batch.get("verl_tool_metrics") or {}
        return {
            "calls": calls,
            "q_overlap_max": q_overlap_max,
            "lrm_calls": len(calls),
            "attempted_calls": attempted_calls,
            "repeat_valid_calls": max(len(calls) - 1, 0),
            "repeat_call_trajectory": int(len(calls) > 1),
            "retry_after_invalid": retry_after_invalid,
            "depth_counts": depth_counts,
            "mean_depth": (sum(depths) / len(depths)) if depths else 0.0,
            "reasoning_tokens": reasoning_tokens,
            "invalid_calls": invalid_calls,
            **{
                f"invalid_{category}": count
                for category, count in invalid_by_category.items()
            },
            "generated_own_tokens": (
                int(metrics["generated_own_tokens"])
                if "generated_own_tokens" in metrics else None
            ),
            "local_input_tokens": int(metrics.get("local_input_tokens", 0)),
        }

    def _own_tokens(self, data_item, prompt_length: int, valid_response_length: int) -> tuple[int, int]:
        '(own_tokens, own_tokens_after_last_call).'
        if "response_mask" in data_item.batch.keys():
            mask = data_item.batch["response_mask"][:valid_response_length]
        else:
            mask = data_item.batch["attention_mask"][prompt_length:prompt_length + valid_response_length]
        own_kept = int(mask.sum().item())
        zeros = (mask == 0).nonzero()
        last_obs_end = int(zeros[-1].item()) + 1 if len(zeros) else 0
        own_after_last = int(mask[last_obs_end:].sum().item())
        return own_kept, own_after_last

    def __call__(self, data: DataProto, return_dict=False):




        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra = defaultdict(list)
        printed = 0
        trajectories = []

        for i in range(len(data)):
            item = data[i]
            prompt_length = item.batch["prompts"].shape[-1]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            response_str = self.tokenizer.decode(
                item.batch["responses"][:valid_response_length], skip_special_tokens=True
            )

            gold = item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = item.non_tensor_batch.get("extra_info") or {}
            answer_format = extra_info.get("answer_format", "math")
            correct = float(grade(answer_format, response_str, str(gold)))
            valid_answer_format = self._has_valid_answer_format(answer_format, response_str)

            stats = self._tool_stats(item)
            reconstructed_own_tokens, own_after_last = self._own_tokens(
                item, prompt_length, valid_response_length
            )
            own_tokens = (
                stats["generated_own_tokens"]
                if stats["generated_own_tokens"] is not None
                else reconstructed_own_tokens
            )
            local_input_tokens = stats["local_input_tokens"]
            costs = trajectory_cost(local_input_tokens, own_tokens, stats["calls"], self.p_gpu_hr,
                                    self.throughput["prefill_tok_s"],
                                    self.throughput["decode_tok_s"], self.tiers)

            trajectories.append({
                "index": i,
                "valid_response_length": valid_response_length,
                "response_str": response_str,
                "gold": gold,
                "correct": correct,
                "valid_answer_format": valid_answer_format,
                "unterminated_query_attempt": int(
                    protocol.has_unterminated_tool_attempt(response_str)
                ),
                "stats": stats,
                "own_tokens": own_tokens,
                "own_after_last": own_after_last,
                "costs": costs,
            })

        group_keys = self._prompt_group_keys(data)
        raw_costs = [trajectory["costs"]["cost_usd"] for trajectory in trajectories]
        reward_costs = (
            self._task_minmax_costs(group_keys, raw_costs)
            if self.norm_cost else raw_costs
        )
        for trajectory, reward_cost in zip(trajectories, reward_costs):
            shaped_reward, cost_penalty = self._reward_components(
                trajectory["correct"], reward_cost
            )
            trajectory["reward_cost"] = reward_cost
            trajectory["shaped_reward"] = shaped_reward
            trajectory["cost_penalty"] = cost_penalty

        all_wrong_mask = _all_wrong_group_mask(
            group_keys, [trajectory["correct"] for trajectory in trajectories]
        )
        is_validation = self.num_examine == 1 or bool(data.meta_info.get("validate", False))



        for trajectory, all_wrong_group in zip(trajectories, all_wrong_mask):
            if is_validation:
                reward = trajectory["correct"]
            elif all_wrong_group:
                reward = 0.0
            else:
                reward = trajectory["shaped_reward"]
            trajectory["reward"] = reward

        group_indices = defaultdict(list)
        for i, key in enumerate(group_keys):
            group_indices[key].append(i)

        group_diagnostics = [None] * len(trajectories)
        for indices in group_indices.values():
            correctness = np.asarray(
                [trajectories[i]["correct"] for i in indices], dtype=np.float64
            )
            rewards = np.asarray(
                [trajectories[i]["reward"] for i in indices], dtype=np.float64
            )
            cost_penalties = np.asarray(
                [trajectories[i]["cost_penalty"] for i in indices], dtype=np.float64
            )
            applied_total_penalties = (
                np.zeros_like(cost_penalties)
                if is_validation
                else correctness * cost_penalties
            )
            all_wrong_group = bool(np.all(correctness == 0.0))
            all_correct_group = bool(np.all(correctness == 1.0))
            mixed_group = not all_wrong_group and not all_correct_group
            valid_called = np.asarray(
                [trajectories[i]["stats"]["lrm_calls"] > 0 for i in indices],
                dtype=bool,
            )
            attempted = np.asarray(
                [trajectories[i]["stats"]["attempted_calls"] > 0 for i in indices],
                dtype=bool,
            )
            all_call_group = bool(np.all(valid_called))
            no_call_group = bool(np.all(~valid_called))
            mixed_call_group = not all_call_group and not no_call_group
            all_attempt_group = bool(np.all(attempted))
            no_attempt_group = bool(np.all(~attempted))
            mixed_attempt_group = not all_attempt_group and not no_attempt_group
            ddof = 1 if len(indices) > 1 else 0
            reward_mean = float(rewards.mean())
            reward_std = float(rewards.std(ddof=ddof))
            accuracy_std = float(correctness.std(ddof=ddof))
            cost_penalty_std = float(cost_penalties.std(ddof=ddof))
            applied_penalty_std = float(applied_total_penalties.std(ddof=ddof))
            penalty_to_accuracy_std_ratio = (
                applied_penalty_std / (accuracy_std + 1e-6) if mixed_group else 0.0
            )
            for i, reward in zip(indices, rewards):
                group_diagnostics[i] = {
                    "all_wrong_group": float(all_wrong_group),
                    "all_correct_group": float(all_correct_group),
                    "mixed_group": float(mixed_group),


                    "all_call_group": float(all_call_group),
                    "mixed_call_group": float(mixed_call_group),
                    "no_call_group": float(no_call_group),
                    "all_attempt_group": float(all_attempt_group),
                    "mixed_attempt_group": float(mixed_attempt_group),
                    "no_attempt_group": float(no_attempt_group),
                    "group_reward_mean": reward_mean,
                    "group_reward_std": reward_std,
                    "group_accuracy_std": accuracy_std,
                    "group_cost_penalty_std": cost_penalty_std,
                    "group_applied_penalty_std": applied_penalty_std,
                    "penalty_to_accuracy_std_ratio": penalty_to_accuracy_std_ratio,
                    "group_centered_score": float(reward - reward_mean),
                }

        for trajectory, diagnostics in zip(trajectories, group_diagnostics):
            i = trajectory["index"]
            correct = trajectory["correct"]
            reward = trajectory["reward"]



            cost_penalty = trajectory["cost_penalty"]
            valid_response_length = trajectory["valid_response_length"]
            reward_tensor[i, max(valid_response_length - 1, 0)] = reward

            extra["accuracy"].append(correct)
            extra["score"].append(reward)
            costs = trajectory["costs"]
            extra["cost_usd"].append(costs["cost_usd"])
            extra["cost_local"].append(costs["cost_local"])
            extra["cost_api"].append(costs["cost_api"])
            extra["reward_cost"].append(trajectory["reward_cost"])
            extra["cost_was_normalized"].append(float(self.norm_cost))
            extra["cost_penalty"].append(cost_penalty)
            extra["total_penalty"].append(cost_penalty)
            extra["applied_total_penalty"].append(
                0.0 if is_validation else correct * cost_penalty
            )
            for key, value in diagnostics.items():
                extra[key].append(value)
            extra["valid_answer_format"].append(
                float(trajectory["valid_answer_format"])
            )
            extra["unterminated_query_attempt"].append(
                trajectory["unterminated_query_attempt"]
            )
            extra["own_tokens"].append(trajectory["own_tokens"])
            stats = trajectory["stats"]
            extra["local_input_tokens"].append(stats["local_input_tokens"])
            extra["lrm_calls"].append(stats["lrm_calls"])
            extra["attempted_calls"].append(stats["attempted_calls"])
            extra["repeat_valid_calls"].append(stats["repeat_valid_calls"])
            extra["repeat_call_trajectory"].append(stats["repeat_call_trajectory"])
            extra["retry_after_invalid"].append(stats["retry_after_invalid"])
            for d in protocol.DEPTHS:
                extra[f"calls_depth{d}"].append(stats["depth_counts"][d])
            extra["mean_depth"].append(stats["mean_depth"])
            extra["reasoning_tokens"].append(stats["reasoning_tokens"])
            extra["invalid_calls"].append(stats["invalid_calls"])
            for category in (
                protocol.INVALID_SYNTAX,
                protocol.INVALID_DEPTH,
                protocol.INVALID_EMPTY,
                protocol.INVALID_LENGTH,
                protocol.INVALID_OVERLAP,
                protocol.INVALID_OTHER,
            ):
                extra[f"invalid_{category}"].append(
                    stats[f"invalid_{category}"]
                )
            extra["q_overlap_max"].append(stats["q_overlap_max"])
            extra["own_tokens_after_last_call"].append(trajectory["own_after_last"])

            if printed < self.num_examine:
                printed += 1
                print("[effi] gold:", trajectory["gold"], "| correct:", correct,
                      "| reward:", reward,
                      "| group:", "all-wrong" if diagnostics["all_wrong_group"] else (
                          "all-correct" if diagnostics["all_correct_group"] else "mixed"
                      ),
                      "| cost:", costs, "| cost penalty:", cost_penalty,
                      "| attempts:", stats["attempted_calls"],
                      "| calls:", stats["lrm_calls"],
                      "| invalid:", stats["invalid_calls"],
                      "by depth:", stats["depth_counts"])
                print("[response tail]", trajectory["response_str"][-500:])

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(extra)}
        return reward_tensor
