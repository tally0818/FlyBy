"""Chance-corrected difficulty screening and oracle-lift verification."""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from ..eval.grader import grade
from ..tools import guard, protocol
from typing import Callable

def skill_of(p_solve: float, chance: float) -> float:
    return (p_solve - chance) / (1.0 - chance)


def item_chance(item: dict, default: float) -> float:
    """Use the item option count when available."""
    choices = item.get("choices")
    return 1.0 / len(choices) if choices else default


def _gap_cfg(dcfg: dict) -> dict:
    return dcfg.get("oracle_gap") or {}


def classify_domain(
    items: list[dict], rollouts: list[list[str]], dcfg: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return band items, unverified hard-floor candidates, and base rollouts."""
    default_chance = dcfg["chance"]
    lo, hi = dcfg["band_skill"]
    max_skill = float(_gap_cfg(dcfg).get("max_skill", 0.0))
    band, hard_pool, records = [], [], []
    if len(items) != len(rollouts) or any(not texts for texts in rollouts):
        raise ValueError("Incomplete difficulty rollouts")
    for item, texts in zip(items, rollouts):
        graded = [grade(item["answer_format"], t, item["gold"]) for t in texts]
        p_solve = sum(graded) / len(graded)
        chance = item_chance(item, default_chance)
        skill = skill_of(p_solve, chance)
        item = dict(item, p_solve=p_solve, skill=skill, chance=chance)
        records.append({
            "id": item["id"], "domain": item["domain"], "question": item["question"],
            "gold": item["gold"], "answer_format": item["answer_format"],
            "p_solve": p_solve, "skill": skill, "chance": chance,
            "rollouts": [{"text": t, "correct": bool(c)} for t, c in zip(texts, graded)],
        })
        if lo <= skill <= hi:
            band.append(dict(item, slice="band"))
        elif skill <= max_skill:
            hard_pool.append(item)
    return band, hard_pool, records


ORACLE_BRIEF_CONTRACT = (
    "You are a knowledge assistant. You are shown a problem that a small model is failing "
    "to solve. Write a short briefing of the concepts, theorems, definitions, formulas or "
    "facts that a solver would need to know to attempt it. State them in general form. "
    "Do NOT solve the problem, do NOT apply the results to it, do NOT work through any "
    "steps, and do NOT state a final answer, a numeric result, or a multiple-choice letter."
)


def brief_prompt(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": ORACLE_BRIEF_CONTRACT},
        {"role": "user", "content": item.get("stem") or item["question"]},
    ]


def sanitize_brief(brief: str, item: dict) -> tuple[str, int]:
    """Remove answer leakage and gold-answer occurrences."""
    text, n = guard.filter_leakage(brief or "")
    text, k = guard.redact_gold(text, str(item.get("gold", "")), item.get("answer_format", "math"))
    return text, n + k


def apply_brief(item: dict, brief: str) -> dict:
    """Item whose question carries the brief in the same shape RL will splice in."""
    return dict(item, question=item["question"] + protocol.format_lrm_answer(brief))


def probe_oracle_gap(
    candidates: list[dict],
    dcfg: dict,
    brief_fn: Callable[[list[dict]], list[str]],
    rollout_fn: Callable[[list[dict]], list[list[str]]],
) -> tuple[list[dict], dict]:
    """Keep hard-floor candidates whose pass rate an oracle brief actually lifts."""
    gcfg = _gap_cfg(dcfg)
    if not candidates or not gcfg.get("enabled", True):
        return [], {"probed": 0, "kept": 0, "redacted_briefs": 0}

    min_lift = float(gcfg.get("min_lift", 0.25))
    cap = int(gcfg.get("max_candidates", len(candidates)))
    candidates = candidates[:cap]

    briefs = brief_fn(candidates)
    if len(briefs) != len(candidates):
        raise ValueError("Incomplete oracle briefs")
    briefed, redacted = [], 0
    for item, brief in zip(candidates, briefs):
        text, n = sanitize_brief(brief, item)
        redacted += bool(n)
        briefed.append(apply_brief(item, text))

    kept = []
    rollouts = rollout_fn(briefed)
    expected = int(gcfg["n_rollouts"])
    if len(rollouts) != len(candidates) or any(len(t) != expected or not t for t in rollouts):
        raise ValueError("Incomplete oracle rollouts")
    for item, texts in zip(candidates, rollouts):
        graded = [grade(item["answer_format"], t, item["gold"]) for t in texts]
        p_briefed = sum(graded) / len(graded)
        lift = p_briefed - float(item["p_solve"])
        if lift >= min_lift:
            kept.append(dict(item, slice="oracle_gap",
                             p_solve_briefed=p_briefed, oracle_lift=lift))
    return kept, {
        "probed": len(candidates),
        "kept": len(kept),
        "redacted_briefs": redacted,
    }


# ---------- pooling and selection ----------


def _source_targets(dcfg: dict) -> dict:
    return dcfg.get("source_targets") or {}


def _domain_target(dcfg: dict, available: int | None = None) -> int:
    """How many items this domain should select: the sum of its per-source targets."""
    targets = _source_targets(dcfg)
    if targets:
        return int(sum(targets.values()))
    return int(dcfg.get("target_selected") or (available or 0))


def _pool_targets(dcfg: dict, available: int | None = None) -> tuple[int, int]:
    """(band_target, gap_target) implied by the domain target and oracle_gap.share."""
    target = _domain_target(dcfg, available)
    share = float(_gap_cfg(dcfg).get("share", 0.0))
    gap_target = int(round(target * share))
    return target - gap_target, gap_target


def _largest_remainder(want: dict, n: int) -> dict:
    """Scale `want` down to total n, distributing the rounding loss largest-first."""
    total = sum(want.values())
    if total <= 0:
        return {source: 0 for source in want}
    exact = {source: value * n / total for source, value in want.items()}
    out = {source: int(value) for source, value in exact.items()}
    for source in sorted(want, key=lambda s: exact[s] - int(exact[s]), reverse=True)[
        : n - sum(out.values())
    ]:
        out[source] += 1
    return out


def draw_by_source_targets(
    pool: list[dict], n: int, targets: dict, taken: dict, rng: random.Random
) -> list[dict]:
    """Draw by remaining source targets, filling shortfalls from available sources."""
    def _record(items: list[dict]) -> list[dict]:
        for item in items:
            source = item.get("dataset", "")
            taken[source] = taken.get(source, 0) + 1
        return items

    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))
    if not targets:
        return _record(rng.sample(pool, n))

    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in pool:
        by_source[item.get("dataset", "")].append(item)
    for items in by_source.values():
        rng.shuffle(items)

    want = {source: min(len(items), max(0, int(targets.get(source, 0)) - taken.get(source, 0)))
            for source, items in by_source.items()}
    if sum(want.values()) > n:
        want = _largest_remainder(want, n)

    drawn, leftover = [], []
    for source in sorted(by_source):
        k = want.get(source, 0)
        drawn += by_source[source][:k]
        leftover += by_source[source][k:]
    rng.shuffle(leftover)
    return _record(drawn + leftover[: n - len(drawn)])


def order_by_source_targets(items: list[dict], targets: dict, rng: random.Random) -> list[dict]:
    """Interleave shuffled sources in proportion to their targets."""
    out = list(items)
    rng.shuffle(out)
    if not targets:
        return out
    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in out:
        by_source[item.get("dataset", "")].append(item)
    total_w = sum(float(w) for w in targets.values()) or 1.0
    ranked = []
    for source, group in by_source.items():
        weight = float(targets.get(source, 0.0)) / total_w
        for k, item in enumerate(group):
            ranked.append((((k + 0.5) / weight) if weight > 0 else float("inf"), k, item))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in ranked]


def select_from_pools(
    band: list[dict], gap_pool: list[dict], dcfg: dict, rng: random.Random
) -> tuple[list[dict], dict]:
    """Sample the configured slice ratio, scaling down when either pool underfills."""
    band_target, gap_target = _pool_targets(dcfg, len(band) + len(gap_pool))
    share = float(_gap_cfg(dcfg).get("share", 0.0))
    targets = _source_targets(dcfg)

    band_n, gap_n = min(len(band), band_target), min(len(gap_pool), gap_target)
    if share > 0 and gap_target > 0 and band_target > 0:
        scale = min(band_n / band_target, gap_n / gap_target)
        band_n, gap_n = int(band_target * scale), int(gap_target * scale)

    taken: dict[str, int] = {}
    gap_items = draw_by_source_targets(gap_pool, gap_n, targets, taken, rng)
    band_items = draw_by_source_targets(band, band_n, targets, taken, rng)
    selected = band_items + gap_items
    total = band_n + gap_n
    per_source = Counter(x.get("dataset", "") for x in selected)
    return selected, {
        "band_pool": len(band),
        "oracle_gap_pool": len(gap_pool),
        "selected": total,
        "band": band_n,
        "oracle_gap": gap_n,
        "gap_share": (gap_n / total) if total else 0.0,
        "shortfall": max(0, (band_target + gap_target) - total),
        "by_source": dict(per_source),
        "by_source_slice": {
            f"{source}/{slice_}": n for (source, slice_), n in
            Counter((x.get("dataset", ""), x.get("slice") or "") for x in selected).items()
        },
        "source_shortfall": {source: int(target) - per_source.get(source, 0)
                             for source, target in targets.items()
                             if per_source.get(source, 0) < int(target)},
    }


