import json
from argparse import Namespace
from collections import Counter

import pandas as pd
import pytest

from src.data import difficulty_filter as df
from src.data.build_all import commands
from src.data.build_rl import build_rl
from src.data.build_sft import build_dataframe, OBJECTIVE_INTEGRATION, OBJECTIVE_QUERY
from src.data.build_sources import select_sft_candidates
from src.data.curate_sft import (
    assemble, replay_candidate, select_no_call, select_strict, token_prefix,
)
from src.data.generation import Cache, CachedLRM
from src.data.synthesize_sft import candidate_states, synthesize_state


class Tokenizer:
    eos_token = "<|im_end|>"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n<think>\n"


TOK = Tokenizer()
CONFIG = {"failed_rollouts_per_problem": 1, "alphas": [0.4, 0.6, 0.8],
          "epistemic_window_chars": 512, "total_own_tokens": 16384,
          "decode": {"temperature": .6, "top_p": .95, "top_k": 20, "max_new_tokens": 16384}}
GOOD = "Integrate the fact. " * 12 + r"</think>\boxed{B}"
BAD = r"That did not help. </think>\boxed{C}"


def record(name="rescue", domain="science"):
    return {"id": name, "dataset": "supergpqa" if domain == "science" else "arxivmath_training",
            "domain": domain, "question": f"Problem {name}", "gold": "B", "answer_format": "mcq",
            "slice": "band", "p_solve": .5,
            "rollouts": [{"text": "I'm not sure about the theorem. Let's try another route. </think>wrong", "correct": False},
                         {"text": GOOD, "correct": True}]}


class Engine:
    tokenizer = TOK

    def __init__(self, successes):
        self.successes, self.calls = list(successes), []

    def generate(self, prompt, n, seed, decode, max_tokens):
        self.calls.append((prompt, n, max_tokens))
        k = self.successes.pop(0)
        return [GOOD] * k + [BAD] * (n - k)


class Client:
    def __init__(self, q="What is the relevant theorem?", observation="The theorem gives a useful relation."):
        self.q, self.observation, self.depths = q, observation, []

    def chat(self, messages, *args, **kwargs):
        self.messages = messages
        return {"unavailable": False, "text": json.dumps({"q": self.q})}

    def query(self, q, depth, problem_id):
        self.depths.append(depth)
        return {"unavailable": False, "text": self.observation, "cost_usd": .001}


def run_state(successes, rec=None, client=None):
    rec = rec or record()
    engine, client = Engine(successes), client or Client()
    state = next(candidate_states(rec, CONFIG))
    result = synthesize_state(rec, state, CONFIG, engine, client, 42)
    return result, engine, client


@pytest.mark.parametrize("successes, depths, strict", [
    ([0, 3], [1], True),
    ([0, 1, 3], [1, 2], True),
    ([0, 0, 1, 4], [1, 2, 3], True),
    ([0, 2], [1], False),  # collection gate stops, even though strict gate fails
    ([1, 3], [1], False),  # plain success prevents strict rescue
    ([2, 2, 3], [1, 2], False),
    ([0, 0, 1, 1], [1, 2, 3], False),
])
def test_paper_collection_and_strict_gates(successes, depths, strict):
    result, engine, client = run_state(successes)
    assert client.depths == depths
    assert result["audit"]["strict"] == strict
    assert all(n == 4 for _, n, _ in engine.calls)
    plain_prompt = engine.calls[0][0]
    assert all(prompt.startswith(plain_prompt) for prompt, _, _ in engine.calls[1:])
    if strict:
        assert result["continuation"] == GOOD


def test_query_validity_overlap_and_observation_redaction():
    invalid, engine, client = run_state([], client=Client(q="x" * 301))
    assert invalid["audit"]["status"].startswith("query_invalid")
    assert not engine.calls and not client.depths
    rec = record()
    rec["question"] = " ".join(f"word{i}" for i in range(25))
    invalid, engine, _ = run_state([], rec, Client(q=rec["question"]))
    assert invalid["audit"]["status"] == "query_overlaps_problem"
    assert not engine.calls
    result, engine, _ = run_state([0, 3], client=Client(observation=r"The answer is B. \boxed{B}"))
    assert "[redacted]" in result["observation"]
    assert r"\boxed{B}" not in engine.calls[1][0]
    assert result["audit"]["attempts"][0]["redactions"] >= 2


def test_uncertainty_dedup_and_fractional_fallback_before_final_answer():
    assert len(list(candidate_states(record(), CONFIG))) == 1
    rec = record()
    rec["rollouts"][0]["text"] = "0123456789" * 10 + "</think>" + "final" * 100
    states = list(candidate_states(rec, CONFIG))
    assert [s["cut"] for s in states] == [40, 60, 80]
    assert all("</think>" not in s["trace"] for s in states)


def test_dynamic_five_groups_masks_and_paired_unique_sources(tmp_path):
    results = [run_state([0, 3], record(f"rescue-{i}"))[0] for i in range(2)]
    strict = select_strict(results + results, TOK)
    assert len(strict) == 2
    records = [record(f"control-{i}") for i in range(2)]
    controls, quotas = select_no_call(records, [], TOK, 2, {"math": 41, "science": 54}, 42)
    assert quotas == {"math": 0, "science": 2}
    replay = []
    for i in range(2):
        entry = replay_candidate({"domain": "science", "conversations": [
            {"from": "human", "value": f"Replay problem {i}"},
            {"from": "gpt", "value": "<think>" + "Reason carefully. " * 20 + r"</think>\boxed{B}"},
        ]}, TOK)
        assert entry is not None
        assert (entry["prompt"] + entry["text"]).count("<think>") == 1
        replay.append({**entry, "row_index": i})
    rows = assemble(strict, controls, replay, TOK, {"hf": "test", "revision": "pinned"}, 42)
    source = pd.DataFrame(rows)
    output, report = build_dataframe(source, TOK)
    assert len(output) == 10
    assert set(report["objectives"].values()) == {2}
    assert report["masked_observations"] == 4
    assert report["supervised_actions_converted"] == 4
    assert report["masked_actions_converted"] == 2
    path = tmp_path / "sft.parquet"
    source.to_parquet(path, index=False)
    assert pd.read_parquet(path).to_dict("records") == rows
    for r in rows:
        parts, meta = json.loads(r["segments_json"]), json.loads(r["meta_json"])
        if meta["source_objective"] == OBJECTIVE_INTEGRATION:
            assert [p["train"] for p in parts] == [False, False, False, True]
            assert len(TOK.encode(parts[-1]["text"])) == 128
        if meta["source_objective"] == OBJECTIVE_QUERY:
            assert [p["train"] for p in parts] == [False, True]
    # A masked observation or integration action cannot silently become supervised.
    corrupt = source.copy(deep=True)
    idx = next(i for i, r in enumerate(rows) if r["kind"] == "integration_booster")
    parts = json.loads(corrupt.at[idx, "segments_json"])
    parts[1]["train"] = True
    corrupt.at[idx, "segments_json"] = json.dumps(parts)
    with pytest.raises(AssertionError, match="masks"):
        build_dataframe(corrupt, TOK)


def test_no_control_duplication_or_filter_relaxation():
    rec = record()
    rec["rollouts"] += rec["rollouts"]
    with pytest.raises(ValueError, match="unique verified"):
        select_no_call([rec], [], TOK, 2, {"science": 1}, 42)
    result = run_state([0, 3])[0]
    result["continuation"] += '<llm_query depth="2">Again?</llm_query>'
    assert select_strict([result], TOK) == []


def test_utf8_integration_prefix_respects_actual_token_count():
    text = "정리" * 100
    prefix = token_prefix(TOK, text)
    assert text.startswith(prefix) and 0 < len(TOK.encode(prefix)) <= 128


def test_chance_corrected_floor_requires_verified_lift():
    item = record()
    item["choices"] = list("ABCDEFGHIJ")
    cfg = {"chance": .25, "band_skill": [.25, .75],
           "oracle_gap": {"max_skill": 0, "min_lift": .125, "n_rollouts": 16}}
    band, hard, _ = df.classify_domain([item], [[GOOD] * 4 + [BAD] * 12], cfg)
    assert not band and not hard
    _, hard, _ = df.classify_domain([item], [[GOOD] + [BAD] * 15], cfg)
    assert len(hard) == 1 and hard[0]["p_solve"] == 1 / 16
    assert "oracle_lift" not in hard[0]
    for correct, expected in [(2, 0), (3, 1)]:
        kept, stats = df.probe_oracle_gap(hard, cfg, lambda _: ["A useful concept."],
                                        lambda _: [[GOOD] * correct + [BAD] * (16 - correct)])
        assert len(kept) == expected and stats["probed"] == 1
        if kept:
            assert kept[0]["oracle_lift"] == .125
            assert kept[0]["p_solve_briefed"] == 3 / 16
    with pytest.raises(ValueError, match="Incomplete oracle"):
        df.probe_oracle_gap(hard, cfg, lambda _: [], lambda _: [])
    with pytest.raises(ValueError, match="Incomplete oracle"):
        df.probe_oracle_gap(hard, cfg, lambda _: ["Concept"], lambda _: [[GOOD]])


def test_fixed_candidate_quota_after_difficulty():
    selected = [record(f"{domain}-{i}", domain) for domain in ("math", "science") for i in range(6)]
    cfg = {"per_domain": 4, "sources": {"math": "arxivmath_training", "science": "supergpqa"}}
    chosen = select_sft_candidates(selected, cfg, 42)
    assert Counter(x["domain"] for x in chosen) == {"math": 4, "science": 4}
    assert chosen == select_sft_candidates(selected, cfg, 42)
    with pytest.raises(ValueError, match="requires 4"):
        select_sft_candidates(selected[:3], cfg, 42)


def test_rejected_synthesis_problem_can_enter_rl_but_actual_sft_cannot():
    rows = []
    for domain in ("math", "science"):
        for index, rate in enumerate([.5, .625, .75, 0, 0]):
            item = record(f"{domain}-{index}", domain)
            item.update(p_solve=rate, choices=list("ABCD") if rate else [],
                        oracle_lift=.125, p_solve_briefed=rate+.125, slice="oracle_gap" if rate == 0 else "band")
            rows.append(item)
    # All ten were synthesis candidates. Only this extra problem actually trained SFT.
    used = record("used-for-no-call", "science")
    selected, report = build_rl(rows + [used], [used])
    assert {x["id"] for x in selected} == {x["id"] for x in rows}
    assert report["selection"]["excluded_sft"] == 1


def test_cache_resume_and_api_failure_not_cached(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get("test", {"id": 1}, lambda: ["value"]) == ["value"]
    assert cache.get("test", {"id": 1}, lambda: pytest.fail("must reuse cache")) == ["value"]
    with pytest.raises(RuntimeError):
        cache.get("failure", "key", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert not list((tmp_path / "failure").glob("*.json"))
    class API:
        tiers = {3: {"model": "teacher"}}
        thinking = False
        base_url = "https://test.invalid"
        budget = Namespace(usd_spent=0)
        calls = 0
        unavailable = True

        def chat(self, *args):
            self.calls += 1
            return {"unavailable": self.unavailable, "text": "cached", "cost_usd": .01}
    api = API()
    client = CachedLRM(api, cache, "api")
    with pytest.raises(RuntimeError, match="Oracle unavailable"):
        client.chat([], "problem")
    api.unavailable = False
    client.chat([], "problem")
    client.chat([], "problem")
    assert api.calls == 2
    CachedLRM(API(), cache, "api")
    assert api.budget.usd_spent == .01


def test_pipeline_eval_only_and_custom_paths():
    args = Namespace(config="configs/data.yaml", source_config="configs/data_generation.yaml",
                     split="eval", override=["paths.processed=custom/processed"],
                     source_override=["paths.source=custom/source"])
    assert len(commands(args)) == 1
    args.split = "all"
    stages = commands(args)
    assert len(stages) == 4
    assert "custom/source/sft_used.jsonl" in stages[-1]
    assert "custom/processed/train_rl.jsonl" in stages[-1]
    assert stages[1][-2:] == ["--override", "paths.source=custom/source"]


def test_full_source_build_with_fake_models_and_pinned_dataset(tmp_path, monkeypatch):
    """Exercise orchestration, oracle screening, curation, disk artifacts and RL together."""
    import datasets
    import transformers
    import yaml
    from src.data import build_sources
    from src.common.config import load_config
    from src.common.io import load_jsonl, read_json, write_jsonl

    cfg = load_config("configs/data_generation.yaml")
    raw_dir, source_dir = tmp_path / "raw", tmp_path / "source"
    cfg["paths"] = {"source": str(source_dir), "cache": str(tmp_path / "cache")}
    cfg["sft"]["per_domain"] = 3
    cfg["difficulty"]["chunk_items"] = 8
    for domain, donor in cfg["sft"]["sources"].items():
        cfg["difficulty"][domain]["source_targets"] = {donor: 8}
        cfg["difficulty"][domain]["oracle_gap"]["share"] = .25
        rows = []
        for index in range(8):
            r = record(f"{domain}-{index}", domain)
            r["question"] = f"{domain} CASE{index} {'GAP' if index >= 6 else 'BAND'}"
            r["choices"] = list("ABCDEFGHIJ")
            rows.append(r)
        write_jsonl(raw_dir / f"{domain}.jsonl", rows)

    class FakeEngine:
        calls = 0
        base_calls = 0

        def __init__(self, config, tokenizer, cache):
            self.tokenizer = tokenizer
            self.cache = cache

        def generate(self, prompt, n, seed, decode, max_tokens=None):
            def produce():
                FakeEngine.calls += 1
                gap = "GAP<|im_end|>" in prompt or "GAP\n<lrm_answer>" in prompt
                assisted = "<lrm_answer>" in prompt.split("<|im_start|>user\n", 1)[1]
                synthesis = prompt.startswith("<|im_start|>system")
                if synthesis:
                    correct = (2 if gap else 3) if assisted else 0
                else:
                    if not assisted:
                        FakeEngine.base_calls += 1
                    correct = n // 2 if assisted or not gap else 0
                return [GOOD] * correct + [BAD] * (n - correct)
            return self.cache.get("fake_generations", [prompt, n, seed, max_tokens], produce)

    class API(Client):
        def __init__(self, **kwargs):
            super().__init__()
            self.tiers = kwargs["depth_tiers"]
            self.budget = Namespace(usd_spent=0)
            self.thinking = False
            self.base_url = "https://test.invalid"

        def chat(self, messages, *args, **kwargs):
            if messages[0]["content"] == df.ORACLE_BRIEF_CONTRACT:
                return {"unavailable": False, "text": "A useful theorem.", "cost_usd": 0}
            assert "partial failed attempt" in messages[0]["content"]
            return {"unavailable": False, "text": json.dumps({"q": self.q}), "cost_usd": 0}

    replay_calls = []
    def load_dataset(name, **kwargs):
        replay_calls.append((name, kwargs))
        return iter({"domain": domain, "conversations": [
            {"from": "human", "value": f"{domain} replay-{i}"},
            {"from": "gpt", "value": "<think>" + "Reason carefully. " * 20 + r"</think>\boxed{B}"},
        ]} for domain in ("math", "science") for i in range(12))

    monkeypatch.setattr(build_sources, "Engine", FakeEngine)
    monkeypatch.setattr(build_sources, "LRMClient", API)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: TOK)
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    config_path, data_path = tmp_path / "generation.yaml", tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    data_path.write_text(yaml.safe_dump({"paths": {"raw": str(raw_dir)}}))
    monkeypatch.setattr("sys.argv", ["build_sources", "--config", str(config_path), "--data-config", str(data_path)])
    build_sources.main()
    assert FakeEngine.base_calls == 16  # exactly one no-tool rollout group per input
    candidates = load_jsonl(source_dir / "sft_candidates.jsonl")
    used = load_jsonl(source_dir / "sft_used.jsonl")
    assert len(candidates) == 6 and len(used) == 4
    assert len(pd.read_parquet(source_dir / "sft.parquet")) == 20
    report = read_json(source_dir / "build_report.json")
    assert report["strict_unique_problems"] == 4
    assert set(report["api_cost_usd"]) == {"probe", "synthesis"}
    selected, _ = build_rl(load_jsonl(source_dir / "rl.jsonl"), used)
    rejected = {r["id"] for r in candidates} - {r["id"] for r in used}
    assert rejected <= {r["id"] for r in selected}
    assert len(selected) == 10
    assert replay_calls[0][1]["revision"] == cfg["replay"]["revision"]
    # Budget-only overrides keep every cached expensive operation.
    before = FakeEngine.calls
    cfg["api"]["synthesis_budget_usd"] += 10
    config_path.write_text(yaml.safe_dump(cfg))
    build_sources.main()
    assert FakeEngine.calls == before
    assert len(replay_calls) == 1


def test_brief_sanitized_before_policy_rollout():
    item = {**record(), "p_solve": 0, "gold": "123456", "answer_format": "math"}
    cfg = {"oracle_gap": {"min_lift": .125, "n_rollouts": 8}}
    def rollouts(items):
        assert "123456" not in items[0]["question"]
        assert "<lrm_answer>" in items[0]["question"]
        return [[r"\boxed{123456}"] + [r"\boxed{0}"] * 7]
    kept, stats = df.probe_oracle_gap([item], cfg, lambda _: ["The constant is 123456."], rollouts)
    assert len(kept) == 1 and stats["redacted_briefs"] == 1


def test_rl_stage_receives_screen_configuration():
    args = Namespace(config="configs/data.yaml", source_config="configs/data_generation.yaml",
                     split="all", override=[],
                     source_override=["difficulty.math.oracle_gap.min_lift=0.25"])
    stage = commands(args)[-1]
    assert stage[stage.index("--config") + 1] == args.source_config
    assert stage[-2:] == ["--override", args.source_override[0]]


def test_gold_redaction_preserves_larger_numbers():
    from src.tools.guard import redact_gold
    cleaned, count = redact_gold("Value 12. Other values 12.5, 312 and 0.12.", "12")
    assert count == 1
    assert "12.5, 312 and 0.12." in cleaned
