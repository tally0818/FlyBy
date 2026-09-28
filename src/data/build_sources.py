"""Generate SFT/RL artifacts with chance-corrected and oracle-lift screening."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from ..common.config import load_config
from ..common.io import load_jsonl, write_json, write_jsonl
from ..common.run import load_dotenv
from ..tools.lrm_client import LRMClient
from . import difficulty_filter as df
from .generation import Cache, CachedLRM, Engine, digest, item_seed
from .synthesize_sft import candidate_states, synthesize_state


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def make_cache(cfg, raw_dir):
    """Invalidate changed inputs, models, parameters or filter/curation code.

    Budget/transport settings are deliberately excluded, allowing a stopped run
    to resume with a larger budget without throwing away successful generations.
    """
    src = Path(__file__).resolve().parents[1]
    code = sorted((src / "data").glob("*.py")) + [src / "eval/grader.py"]
    code += sorted((src / "tools").glob("*.py"))
    identity = {"config": {k: v for k, v in cfg.items() if k not in ("api", "paths")},
                "inputs": {d: file_digest(raw_dir / f"{d}.jsonl") for d in ("math", "science")},
                "code": {str(p.relative_to(src)): file_digest(p) for p in code}}
    root = Path(cfg["paths"]["cache"]) / digest(identity)
    write_json(root / "manifest.json", identity)
    return Cache(root)


def user_prompt(tokenizer, question):
    return tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                         tokenize=False, add_generation_prompt=True)


def run_difficulty(cfg, raw_dir, engine, cache, oracle):
    dcfg = cfg["difficulty"]
    all_selected, all_records, rl, reports = [], [], [], {}
    seed = cfg["seed"]
    for domain in ("math", "science"):
        settings = dcfg[domain]
        items = df.order_by_source_targets(load_jsonl(raw_dir / f"{domain}.jsonl"),
                                           settings["source_targets"], random.Random(seed))
        if not items or len({x["id"] for x in items}) != len(items):
            raise ValueError(f"{domain}: training source must contain unique nonempty problems")
        band, hard, gap, records = [], [], [], []
        probe_stats = Counter()
        selected = []
        step = dcfg["chunk_items"]
        for start in range(0, len(items), step):
            for item in items[start:start + step]:
                texts = engine.generate(user_prompt(engine.tokenizer, item["question"]),
                                        settings["n_rollouts"], item_seed(seed, "base", item["id"]),
                                        dcfg["decode"])
                b, h, r = df.classify_domain([item], [texts], settings)
                band += b
                hard += h
                records += r
                if h and probe_stats["probed"] < settings["oracle_gap"]["max_candidates"]:
                    def briefs(batch):
                        return [oracle.chat(df.brief_prompt(x), x["id"])["text"] for x in batch]

                    def briefed_rollouts(batch):
                        return [engine.generate(
                            user_prompt(engine.tokenizer, x["question"]),
                            settings["oracle_gap"]["n_rollouts"],
                            item_seed(seed, "brief", x["id"]), dcfg["decode"]) for x in batch]

                    kept, stats = df.probe_oracle_gap(h, settings, briefs, briefed_rollouts)
                    gap += kept
                    probe_stats.update(stats)
            selected, stats = df.select_from_pools(band, gap, settings, random.Random(seed))
            donor = cfg["sft"]["sources"][domain]
            donor_count = sum(x["dataset"] == donor for x in selected)
            reports[domain] = {**stats, "rolled_out": len(records), "hard_candidates": len(hard),
                               "sft_eligible": donor_count, **dict(probe_stats)}
            write_json(cache.root / "difficulty_report.partial.json", reports)
            print(f"[difficulty/{domain}] scanned={len(records)}/{len(items)}, "
                  f"selected={len(selected)}, SFT-eligible={donor_count}", flush=True)
            if (len(selected) >= sum(settings["source_targets"].values())
                    and donor_count >= cfg["sft"]["per_domain"]):
                break
        all_selected += selected
        all_records += records
        rl += band + gap
    return all_selected, rl, all_records, reports


def select_sft_candidates(selected, cfg, seed):
    """Enforce the full synthesis pool; no silent quota reduction or top-up."""
    from .build_dataset import stratified_take

    result = []
    for domain, source in cfg["sources"].items():
        eligible = [x for x in selected if x["domain"] == domain and x["dataset"] == source]
        n = cfg["per_domain"]
        if len(eligible) < n:
            raise ValueError(f"SFT requires {n} difficulty-filtered {source} problems; "
                             f"found {len(eligible)}. Inspect difficulty_report.json; "
                             "filters were not relaxed and the pool was not reduced.")
        result.extend(stratified_take(eligible, n, lambda x: x["slice"], random.Random(seed)))
    return result


def run_synthesis(records, cfg, engine, oracle, cache, source_dir):
    from .curate_sft import select_strict

    results = []
    strict_count = 0
    for index, record in enumerate(records):
        for state in candidate_states(record, cfg["synthesis"]):
            key = {"record": record, "state": state, "config": cfg["synthesis"], "seed": cfg["seed"]}
            result = cache.get("synthesis", key, lambda: synthesize_state(
                record, state, cfg["synthesis"], engine, oracle, cfg["seed"]))
            results.append(result)
            # One valid rescue per problem; no resampling to hit a target count.
            if select_strict([result], engine.tokenizer):
                strict_count += 1
                break
        write_jsonl(source_dir / "sft_counterfactual_audit.jsonl", [r["audit"] for r in results])
        print(f"[synthesis] problems={index + 1}/{len(records)}, "
              f"strict={strict_count}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_generation.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--data-override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    data_cfg = load_config(args.data_config, args.data_override)
    raw_dir = Path(data_cfg["paths"]["raw"])
    out = Path(cfg["paths"]["source"])
    load_dotenv()
    cache = make_cache(cfg, raw_dir)
    from transformers import AutoTokenizer
    from .curate_sft import select_strict, select_no_call, fetch_replay, assemble
    from .build_sft import build_dataframe
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
    engine = Engine(cfg["model"], tokenizer, cache)
    api = cfg["api"]
    def client(stage, budget, tiers):
        return CachedLRM(LRMClient(depth_tiers=tiers, budget_usd=budget,
                                  timeout_s=api["timeout_s"], max_retries=api["max_retries"],
                                  concurrency=api["concurrency"], thinking=False), cache, stage)

    probe = client("api/probe", api["probe_budget_usd"], {3: cfg["difficulty"]["probe_backend"]})
    selected, rl, all_records, report = run_difficulty(cfg, raw_dir, engine, cache, probe)
    write_json(out / "difficulty_report.json", report)
    candidates = select_sft_candidates(selected, cfg["sft"], cfg["seed"])
    ids = {item["id"] for item in candidates}
    by_id = {rec["id"]: rec for rec in all_records}
    records = [{**item, **by_id[item["id"]]} for item in candidates if item["id"] in by_id]
    if len(records) != len(ids):
        raise ValueError("Missing or duplicate rollout records for SFT candidates")
    write_jsonl(out / "sft_candidates.jsonl", candidates)
    write_jsonl(out / "rl.jsonl", rl)
    write_jsonl(out / "sft_base_rollouts.jsonl", records)
    # Full candidate IDs are audit-only; RL exclusion is derived from final SFT rows.
    oracle = client("api/synthesis", api["synthesis_budget_usd"], cfg["depths"])
    results = run_synthesis(records, cfg, engine, oracle, cache, out)
    strict = select_strict(results, tokenizer)
    n = len(strict)
    if not n:
        raise ValueError("No strict rescue in the 800-problem pool; thresholds were not relaxed")
    controls, quotas = select_no_call(records, results, tokenizer, n,
                                      cfg["sft"]["control_domain_weights"], cfg["seed"])
    replay = fetch_replay(tokenizer, quotas, cfg["replay"], cache, cfg["seed"])
    rows = assemble(strict, controls, replay, tokenizer, cfg["replay"], cfg["seed"])
    frame = pd.DataFrame(rows)
    _, validation = build_dataframe(frame, tokenizer)
    used_ids = {json.loads(r["meta_json"]).get("problem_id") for r in rows}
    used = [item for item in candidates if item["id"] in used_ids]
    if used_ids - {None} != {item["id"] for item in used}:
        raise ValueError("Final SFT contains an untracked training problem")
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "sft.parquet.tmp"
    frame.to_parquet(temporary, index=False)
    if pd.read_parquet(temporary).to_dict("records") != rows:
        raise ValueError("SFT parquet roundtrip mismatch")
    temporary.replace(out / "sft.parquet")
    write_jsonl(out / "sft_used.jsonl", used)
    write_json(out / "build_report.json", {
        "cache": str(cache.root), "config": cfg, "sft_candidates": len(candidates),
        "candidate_sources": dict(Counter(x["dataset"] for x in candidates)),
        "strict_unique_problems": n, "control_quotas": quotas, "validation": validation,
        "sft_used_problems": len(used), "rl_exclusion": "actual_sft_training_problems_only",
        "rl_source_rows": len(rl), "api_cost_usd": {
            "probe": probe.client.budget.usd_spent,
            "synthesis": oracle.client.budget.usd_spent},
    })
    print(f"[source] {len(candidates)} candidates -> {n} strict rescues -> {len(rows)} SFT rows", flush=True)


if __name__ == "__main__":
    main()
