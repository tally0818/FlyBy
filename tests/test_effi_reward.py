import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_reward_manager_class():
    verl_module = ModuleType("verl")
    verl_module.DataProto = object
    workers_module = ModuleType("verl.workers")
    reward_module = ModuleType("verl.workers.reward_manager")
    reward_module.register = lambda _name: lambda cls: cls

    previous = {
        name: sys.modules.get(name)
        for name in ("verl", "verl.workers", "verl.workers.reward_manager")
    }
    sys.modules["verl"] = verl_module
    sys.modules["verl.workers"] = workers_module
    sys.modules["verl.workers.reward_manager"] = reward_module
    try:
        path = REPO_ROOT / "verl_tool/workers/reward_manager/effi.py"
        spec = importlib.util.spec_from_file_location("_effi_reward_test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.EffiRewardManager
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


EffiRewardManager = _load_reward_manager_class()


@pytest.mark.parametrize("keys,costs,expected", [
    (["a", "b", "a", "b", "a"], [1, 10, 3, 20, 2], [0, 0, 1, 1, 0.5]),
    (["same", "same", "single"], [7, 7, 9], [0, 0, 0]),
    ([], [], []),
])
def test_task_minmax_costs(keys, costs, expected):
    assert EffiRewardManager._task_minmax_costs(keys, costs) == pytest.approx(expected)


def test_task_minmax_costs_rejects_mismatched_groups():
    with pytest.raises(ValueError, match="length mismatch"):
        EffiRewardManager._task_minmax_costs(["a"], [1, 2])


class _FakeRewardBatch:
    def __init__(self):
        self.batch = {
            "prompts": torch.tensor([[9], [8], [9], [8]]),
            "responses": torch.tensor([[2], [2], [2], [1]]),
            "attention_mask": torch.ones((4, 2), dtype=torch.long),
            "response_mask": torch.ones((4, 1), dtype=torch.long),
            "rm_scores": torch.ones((4, 1), dtype=torch.float32),
        }
        self.non_tensor_batch = {
            "uid": np.array(["all-wrong", "mixed", "all-wrong", "mixed"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": "A"} for _ in range(4)], dtype=object
            ),
            "extra_info": np.array(
                [{"id": f"row-{i}", "answer_format": "mcq"} for i in range(4)],
                dtype=object,
            ),
            "tool_interact_info": np.array([[] for _ in range(4)], dtype=object),
            "verl_tool_metrics": np.array(
                [
                    {"generated_own_tokens": own_tokens, "local_input_tokens": 0}
                    for own_tokens in (1, 2, 3, 4)
                ],
                dtype=object,
            ),
        }
        self.meta_info = {"validate": False}

    def __len__(self):
        return self.batch["responses"].shape[0]

    def __getitem__(self, index):
        return SimpleNamespace(
            batch={key: value[index] for key, value in self.batch.items()},
            non_tensor_batch={
                key: value[index] for key, value in self.non_tensor_batch.items()
            },
        )


class _FakeTokenizer:
    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "\\boxed{A}" if int(token_ids[0]) == 1 else "\\boxed{B}"


def _grouped_reward_manager():
    manager = object.__new__(EffiRewardManager)
    manager.tokenizer = _FakeTokenizer()
    manager.num_examine = 0
    manager.lam = 0.1
    manager.norm_cost = False
    manager.p_gpu_hr = 3600.0
    manager.throughput = {"prefill_tok_s": 1.0, "decode_tok_s": 1.0}
    manager.tiers = {1: {"p_in": 0.0, "p_out": 0.0}}
    return manager


def test_correct_only_cost_zeros_wrong_trajectories_and_logs_group_diagnostics():
    result = _grouped_reward_manager()(_FakeRewardBatch(), return_dict=True)

    rewards = result["reward_tensor"].sum(dim=-1).tolist()
    assert rewards == pytest.approx([0.0, 0.0, 0.0, 0.6])
    assert result["reward_extra_info"]["all_wrong_group"] == [1.0, 0.0, 1.0, 0.0]
    assert result["reward_extra_info"]["mixed_group"] == [0.0, 1.0, 0.0, 1.0]
    assert result["reward_extra_info"]["all_correct_group"] == [0.0, 0.0, 0.0, 0.0]
    assert result["reward_extra_info"]["group_centered_score"] == pytest.approx(
        [0.0, -0.3, 0.0, 0.3]
    )
    assert result["reward_extra_info"]["group_reward_std"] == pytest.approx(
        [0.0, 0.6 / np.sqrt(2), 0.0, 0.6 / np.sqrt(2)]
    )
    assert result["reward_extra_info"]["cost_penalty"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert result["reward_extra_info"]["applied_total_penalty"] == pytest.approx(
        [0.0, 0.0, 0.0, 0.4]
    )
    assert result["reward_extra_info"]["valid_answer_format"] == [1.0] * 4
    assert result["reward_extra_info"]["no_call_group"] == [1.0] * 4
    assert result["reward_extra_info"]["no_attempt_group"] == [1.0] * 4


@pytest.mark.parametrize("all_correct", [False, True])
def test_grouped_reward_normalizes_cost_and_recomputes_cached_scores(all_correct):
    batch = _FakeRewardBatch()
    if all_correct:
        batch.batch["responses"].fill_(1)
    manager = _grouped_reward_manager()
    manager.norm_cost = True
    manager.lam = 0.2

    result = manager(batch, return_dict=True)
    extra = result["reward_extra_info"]

    assert result["reward_tensor"].sum(dim=-1).tolist() == pytest.approx(
        [1.0, 1.0, 0.8, 0.8] if all_correct else [0.0, 0.0, 0.0, 0.8]
    )
    assert extra["reward_cost"] == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert extra["cost_penalty"] == pytest.approx([0.0, 0.0, 0.2, 0.2])
    assert extra["cost_usd"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert extra["cost_was_normalized"] == [1.0] * 4


def test_equal_cost_correct_rollouts_receive_full_reward():
    batch = _FakeRewardBatch()
    batch.batch["responses"].fill_(1)
    for metrics in batch.non_tensor_batch["verl_tool_metrics"]:
        metrics["generated_own_tokens"] = 7
    manager = _grouped_reward_manager()
    manager.norm_cost = True

    result = manager(batch, return_dict=True)

    assert result["reward_tensor"].sum(dim=-1).tolist() == [1.0] * 4
    assert result["reward_extra_info"]["reward_cost"] == [0.0] * 4


@pytest.mark.parametrize("norm_cost", [False, True])
def test_tool_attempt_count_does_not_change_reward(norm_cost):
    batch = _FakeRewardBatch()
    batch.batch["responses"].fill_(1)
    manager = _grouped_reward_manager()
    manager.norm_cost = norm_cost
    baseline = manager(batch, return_dict=True)
    valid_call = {"llm_usage": {"depth": 1, "prompt_tokens": 0, "completion_tokens": 0}}
    rejected_call = {"invalid_call": True, "invalid_error_code": "bad_attributes"}
    batch.non_tensor_batch["tool_interact_info"] = np.array(
        [[], [valid_call], [valid_call, valid_call], [valid_call, rejected_call, rejected_call]],
        dtype=object,
    )

    result = manager(batch, return_dict=True)

    assert torch.equal(result["reward_tensor"], baseline["reward_tensor"])
    assert result["reward_extra_info"]["attempted_calls"] == [0, 1, 2, 3]
    assert result["reward_extra_info"]["lrm_calls"] == [0, 1, 2, 1]
    assert "call_penalty" not in result["reward_extra_info"]


def test_normalization_includes_api_cost():
    batch = _FakeRewardBatch()
    batch.batch["responses"].fill_(1)
    batch.non_tensor_batch["tool_interact_info"] = np.array(
        [[{"llm_usage": {"depth": 1, "prompt_tokens": 10, "completion_tokens": 0}}], [], [], []],
        dtype=object,
    )
    manager = _grouped_reward_manager()
    manager.norm_cost = True
    manager.tiers[1]["p_in"] = 1e6

    result = manager(batch, return_dict=True)

    assert result["reward_extra_info"]["cost_usd"] == pytest.approx([11, 2, 3, 4])
    assert result["reward_extra_info"]["reward_cost"] == pytest.approx([1, 0, 0, 1])
    assert result["reward_tensor"].sum(dim=-1).tolist() == pytest.approx([0.9, 1, 1, 0.9])


@pytest.mark.parametrize("norm_cost", [False, True])
@pytest.mark.parametrize("validation_flag", [False, True])
def test_validation_remains_accuracy_only(norm_cost, validation_flag, capsys):
    batch = _FakeRewardBatch()
    batch.meta_info["validate"] = validation_flag
    manager = _grouped_reward_manager()
    manager.norm_cost = norm_cost
    manager.num_examine = 0 if validation_flag else 1

    result = manager(batch, return_dict=True)

    assert result["reward_tensor"].sum(dim=-1).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert result["reward_extra_info"]["applied_total_penalty"] == [0.0] * 4
    assert result["reward_extra_info"]["group_applied_penalty_std"] == [0.0] * 4


def test_nearly_equal_costs_use_epsilon():
    costs = [0.001, 0.001 + 1e-10]
    normalized = EffiRewardManager._task_minmax_costs(["a", "a"], costs)
    assert normalized == pytest.approx([0.0, 1 / 101], rel=1e-6)
