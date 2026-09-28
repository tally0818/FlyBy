import torch
import numpy as np
from typing import Any
from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics as verl_compute_data_metrics
from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val
from functools import partial
from collections import defaultdict


REWARD_EXTRA_METRIC_KEYS = (
    "accuracy",
    "score",
    "cost_usd",
    "cost_local",
    "cost_api",
    "cost_penalty",
    "call_penalty",
    "total_penalty",
    "applied_total_penalty",
    "all_wrong_group",
    "all_correct_group",
    "mixed_group",
    "all_call_group",
    "mixed_call_group",
    "no_call_group",
    "all_attempt_group",
    "mixed_attempt_group",
    "no_attempt_group",
    "group_reward_mean",
    "group_reward_std",
    "group_accuracy_std",
    "group_cost_penalty_std",
    "group_applied_penalty_std",
    "group_centered_score",
    "valid_answer_format",
    "unterminated_query_attempt",
    "own_tokens",
    "local_input_tokens",
    "lrm_calls",
    "attempted_calls",
    "repeat_valid_calls",
    "repeat_call_trajectory",
    "retry_after_invalid",
    "calls_depth1",
    "calls_depth2",
    "calls_depth3",
    "mean_depth",
    "reasoning_tokens",
    "invalid_calls",
    "invalid_syntax",
    "invalid_depth",
    "invalid_empty",
    "invalid_length",
    "invalid_overlap",
    "invalid_other",
    "q_overlap_max",
    "own_tokens_after_last_call",
)


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    'Computes various metrics from a batch of data for PPO training.'
    result = verl_compute_data_metrics(batch, use_critic)
    
    verl_tool_metrics = batch.non_tensor_batch.get("verl_tool_metrics", [])
    all_keys = []
    for x in verl_tool_metrics:
        all_keys.extend(list(x.keys()))
    all_keys = set(all_keys)
    for key in all_keys:
        values = np.array([float(m[key]) for m in verl_tool_metrics if key in m])
        result[f"verl_tool/{key}/mean"] = values.mean()
        result[f"verl_tool/{key}/max"] = values.max()
        result[f"verl_tool/{key}/min"] = values.min()



    for key in REWARD_EXTRA_METRIC_KEYS:
        if key not in batch.non_tensor_batch:
            continue
        values = np.asarray(batch.non_tensor_batch[key], dtype=np.float64)
        if values.size == 0:
            continue
        result[f"train/{key}/mean"] = float(values.mean())
        result[f"train/{key}/max"] = float(values.max())
        result[f"train/{key}/min"] = float(values.min())
        if key == "lrm_calls":


            result["train/call_rate"] = float((values > 0).mean())
        elif key == "attempted_calls":
            result["train/attempted_call_rate"] = float((values > 0).mean())
        elif key == "repeat_call_trajectory":
            result["train/repeat_call_rate"] = float(values.mean())
        elif key == "retry_after_invalid":
            result["train/retry_after_invalid_rate"] = float(values.mean())
        elif key == "unterminated_query_attempt":
            result["train/unterminated_query_rate"] = float(values.mean())

    if (
        "attempted_calls" in batch.non_tensor_batch
        and "invalid_calls" in batch.non_tensor_batch
    ):
        attempted = np.asarray(
            batch.non_tensor_batch["attempted_calls"], dtype=np.float64
        )
        invalid = np.asarray(
            batch.non_tensor_batch["invalid_calls"], dtype=np.float64
        )
        attempted_total = float(attempted.sum())
        result["train/invalid_attempt_fraction"] = (
            float(invalid.sum()) / attempted_total if attempted_total > 0.0 else 0.0
        )




    if "accuracy" in batch.non_tensor_batch:
        accuracy = np.asarray(batch.non_tensor_batch["accuracy"], dtype=np.float64)
        correct = accuracy > 0.5
        wrong = ~correct

        if "lrm_calls" in batch.non_tensor_batch:
            called = np.asarray(batch.non_tensor_batch["lrm_calls"], dtype=np.float64) > 0
            if correct.any():
                result["train/call_rate_given_correct"] = float(called[correct].mean())
            if wrong.any():
                result["train/call_rate_given_wrong"] = float(called[wrong].mean())
            if called.any():
                result["train/accuracy_given_call"] = float(correct[called].mean())
            if (~called).any():
                result["train/accuracy_given_no_call"] = float(correct[~called].mean())

        for depth in (1, 2, 3):
            key = f"calls_depth{depth}"
            if key not in batch.non_tensor_batch:
                continue
            used_depth = np.asarray(batch.non_tensor_batch[key], dtype=np.float64) > 0
            if used_depth.any():
                result[f"train/accuracy_given_depth{depth}_call"] = float(
                    correct[used_depth].mean()
                )

        if (
            "group_centered_score" in batch.non_tensor_batch
            and "mixed_group" in batch.non_tensor_batch
        ):
            centered = np.asarray(
                batch.non_tensor_batch["group_centered_score"], dtype=np.float64
            )
            mixed = np.asarray(batch.non_tensor_batch["mixed_group"], dtype=np.float64) > 0.5
            mixed_wrong = mixed & wrong
            mixed_correct = mixed & correct
            if mixed_wrong.any():
                result["train/wrong_positive_adv_rate"] = float(
                    (centered[mixed_wrong] > 0.0).mean()
                )
            if mixed_correct.any():
                result["train/correct_negative_adv_rate"] = float(
                    (centered[mixed_correct] < 0.0).mean()
                )

            if "penalty_to_accuracy_std_ratio" in batch.non_tensor_batch and mixed.any():
                ratios = np.asarray(
                    batch.non_tensor_batch["penalty_to_accuracy_std_ratio"], dtype=np.float64
                )
                result["train/mixed_penalty_to_accuracy_std_ratio"] = float(
                    ratios[mixed].mean()
                )
    return result


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    'Process validation metrics into a structured format with statistical analysis.'

    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])


    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                    
                var_vals = [x for x in var_vals if x is not None]
                if not var_vals:
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric


    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
