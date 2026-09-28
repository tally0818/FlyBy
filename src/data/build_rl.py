'Build the released RL training dataset.'
from __future__ import annotations

import argparse
import hashlib
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..common.config import load_config
from .difficulty_filter import item_chance, skill_of
from ..common.io import load_jsonl, write_json, write_jsonl

VALID_SLICES = {"band", "oracle_gap"}


def _index_unique(items: Iterable[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    required = {"id", "dataset", "domain", "slice", "question", "gold"}
    for row_number, item in enumerate(items, start=1):
        missing = required - set(item)
        if missing:
            raise ValueError(f"{label} row {row_number} missing fields {sorted(missing)}")
        item_id = str(item["id"])
        if item_id in indexed:
            raise ValueError(f"duplicate id {item_id!r} in {label}")
        if item["slice"] not in VALID_SLICES:
            raise ValueError(
                f"{label} row {item_id!r} has invalid slice {item['slice']!r}"
            )
        indexed[item_id] = item
    return indexed


def _composition(items: Iterable[dict]) -> dict:
    rows = list(items)
    slice_counts = Counter(str(x["slice"]) for x in rows)
    return {
        "n": len(rows),
        "gap_share": slice_counts["oracle_gap"] / len(rows) if rows else 0.0,
        "by_slice": dict(sorted(slice_counts.items())),
        "by_domain_slice": {
            f"{domain}/{slice_}": n
            for (domain, slice_), n in sorted(
                Counter((str(x["domain"]), str(x["slice"])) for x in rows).items()
            )
        },
        "by_source": dict(
            sorted(Counter(str(x["dataset"]) for x in rows).items())
        ),
    }


def _stable_key(item: dict, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item['id']}".encode()).hexdigest()


def _proportional_take(
    items: list[dict],
    n: int,
    *,
    stratum_fn,
    rank_fn,
) -> list[dict]:
    'Take ``n`` rows while preserving the available stratum distribution.'
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []

    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(str(stratum_fn(item)), []).append(item)

    total = len(items)
    quotas = {key: int(n * len(group) / total) for key, group in groups.items()}
    remaining = n - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (
            -(n * len(groups[key]) / total - quotas[key]),
            key,
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1

    selected = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=rank_fn)[: quotas[key]])
    if len(selected) != n:
        raise AssertionError(f"stratified take selected {len(selected)} rows, expected {n}")
    return selected


def build_rl(
    rl_items: list[dict],
    sft_items: list[dict],
    *,
    seed: int = 42,
    difficulty: dict | None = None,
) -> tuple[list[dict], dict]:
    if difficulty is None:
        difficulty = load_config(Path(__file__).resolve().parents[2] / "configs/data_generation.yaml")["difficulty"]
    rl_by_id = _index_unique(rl_items, "RL input")
    sft_by_id = _index_unique(sft_items, "Actual SFT training problems")
    excluded_ids = set(sft_by_id)
    groups = {tier: [] for tier in ("band", "oracle_gap")}
    excluded_sft = excluded_rate = 0
    for item_id, item in sorted(rl_by_id.items()):
        if item_id in excluded_ids:
            excluded_sft += 1
            continue
        if item["domain"] not in ("math", "science"):
            raise ValueError(f"{item_id}: unsupported domain {item['domain']!r}")
        raw_rate = item.get("p_solve")
        if isinstance(raw_rate, bool) or raw_rate is None:
            raise ValueError(f"{item_id}: p_solve must be a probability")
        try:
            rate = float(raw_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item_id}: p_solve must be a probability") from exc
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError(f"{item_id}: p_solve must be in [0, 1]")
        settings = difficulty[item["domain"]]
        chance = item_chance(item, settings["chance"])
        skill = skill_of(rate, chance)
        lo, hi = settings["band_skill"]
        if lo <= skill <= hi:
            tier = "band"
        elif skill <= settings["oracle_gap"]["max_skill"]:
            briefed, lift = item.get("p_solve_briefed"), item.get("oracle_lift")
            if (isinstance(briefed, bool) or isinstance(lift, bool)
                    or not isinstance(briefed, (int, float)) or not isinstance(lift, (int, float))
                    or not math.isfinite(briefed) or not math.isfinite(lift)
                    or not 0 <= briefed <= 1
                    or not math.isclose(lift, briefed - rate, abs_tol=1e-12)
                    or lift < settings["oracle_gap"]["min_lift"]):
                excluded_rate += 1
                continue
            tier = "oracle_gap"
        else:
            excluded_rate += 1
            continue
        groups[tier].append({**item, "slice": tier, "skill": skill, "chance": chance})

    units = min(len(groups["band"]) // 3, len(groups["oracle_gap"]) // 2)
    if units == 0:
        available = {tier: len(rows) for tier, rows in groups.items()}
        raise ValueError(f"insufficient eligible data for global 6:4 band/oracle-gap mix: {available}")

    final_items = []
    for tier, rows in groups.items():
        final_items.extend(_proportional_take(
            rows,
            units * (3 if tier == "band" else 2),
            stratum_fn=lambda item: (
                item["domain"], item["dataset"],
                (item.get("meta") or {}).get("discipline", ""),
            ),
            rank_fn=lambda item: _stable_key(item, seed),
        ))
    random.Random(seed).shuffle(final_items)
    report = {
        "seed": seed,
        "inputs": {
            "rl": _composition(rl_by_id.values()),
            "sft_used": _composition(sft_by_id.values()),
        },
        "selection": {
            "excluded_sft": excluded_sft,
            "excluded_difficulty": excluded_rate,
            "eligible": _composition(item for rows in groups.values() for item in rows),
            "global_band": 3 * units,
            "global_oracle_gap": 2 * units,
        },
        "final": _composition(final_items),
    }
    return final_items, report


def _default_report_path(output: Path) -> Path:
    return output.with_suffix(".report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", default="data/source/rl.jsonl")
    parser.add_argument("--sft", default="data/source/sft_used.jsonl",
                        help="Actual SFT training problems, not the full synthesis candidate pool")
    parser.add_argument("--output", default="data/processed/train_rl.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--config", default="configs/data_generation.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    report_path = Path(args.report) if args.report else _default_report_path(output)
    final_items, report = build_rl(
        load_jsonl(args.rl),
        load_jsonl(args.sft),
        seed=args.seed,
        difficulty=load_config(args.config, args.override)["difficulty"],
    )
    report["paths"] = {
        "rl": args.rl,
        "sft": args.sft,
        "output": str(output),
    }
    write_jsonl(output, final_items)
    write_json(report_path, report)
    final = report["final"]
    print(
        f"wrote {final['n']} rows -> {output} "
        f"(oracle_gap={final['gap_share']:.1%}, report={report_path})"
    )


if __name__ == "__main__":
    main()
