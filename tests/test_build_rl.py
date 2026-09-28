from collections import Counter

import pytest

from src.data.build_rl import build_rl as _build_rl

def build_rl(rows, sft, **kwargs):
    settings = {"chance": 0, "band_skill": [.25, .75],
                "oracle_gap": {"max_skill": 0, "min_lift": .125}}
    return _build_rl(rows, sft, difficulty={"math": settings, "science": settings}, **kwargs)


def row(name, domain, rate, source="source"):
    return {
        "id": name, "dataset": source, "domain": domain,
        "slice": "oracle_gap" if rate == 0 else "band",
        "p_solve": rate, "choices": [], "oracle_lift": .125, "p_solve_briefed": rate + .125, "chance": 0, "question": name, "gold": "1",
    }


def pool():
    rows = []
    for domain, band, gap in [("math", 17, 9), ("science", 31, 24)]:
        for tier, count, rate in [("band", band, 0.5), ("gap", gap, 0.0)]:
            rows.extend(row(f"{domain}-{tier}-{i}", domain, rate, f"source-{i % 2}")
                        for i in range(count))
    return rows


def test_exact_ratios_after_sft_exclusion():
    rows = pool()
    sft = [rows[0], rows[17], row("sft-only", "science", 0.0)]
    selected, report = build_rl(rows, sft)
    counts = Counter((item["domain"], item["slice"]) for item in selected)
    assert counts == {("math", "band"): 12, ("math", "oracle_gap"): 8,
                      ("science", "band"): 12, ("science", "oracle_gap"): 8}
    assert not {item["id"] for item in selected} & {item["id"] for item in sft}
    assert report["selection"]["excluded_sft"] == 2
    assert len({item["id"] for item in selected}) == 40


def test_reproducible_selection_independent_of_input_order():
    rows = pool()
    assert build_rl(rows, [], seed=42) == build_rl(list(reversed(rows)), [], seed=42)
    assert build_rl(rows, [], seed=42)[0] != build_rl(rows, [], seed=43)[0]


def test_probability_boundaries_and_stale_slice_labels():
    rows = [row(f"{domain}-{i}", domain, rate)
            for domain in ("math", "science")
            for i, rate in enumerate([0, 0, 0.25, 0.5, 0.75, 0.125, 0.8, 1])]
    for item in rows:
        item["slice"] = "band"
    selected, report = build_rl(rows, [])
    assert len(selected) == 10
    assert report["selection"]["excluded_difficulty"] == 6
    assert all(item["p_solve"] == 0 if item["slice"] == "oracle_gap"
               else 0.25 <= item["p_solve"] <= 0.75 for item in selected)


@pytest.mark.parametrize("rate", [None, True, "bad", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_solve_rate_fails(rate):
    rows = pool()
    rows[0]["p_solve"] = rate
    with pytest.raises(ValueError, match="p_solve"):
        build_rl(rows, [])


def test_insufficient_group_fails_instead_of_relaxing_ratios():
    rows = [item for item in pool() if item["domain"] != "science" or item["p_solve"] > 0]
    with pytest.raises(ValueError, match="insufficient eligible data"):
        build_rl(rows, [])


def test_duplicate_ids_fail():
    rows = pool()
    with pytest.raises(ValueError, match="duplicate id"):
        build_rl(rows + [rows[0]], [])


def test_unverified_and_inconsistent_lift_are_excluded():
    rows = pool()
    for domain in ("math", "science"):
        rows.extend([
            {**row(f"{domain}-missing", domain, 0), "oracle_lift": None},
            {**row(f"{domain}-weak", domain, 0), "oracle_lift": .0625, "p_solve_briefed": .0625},
            {**row(f"{domain}-inconsistent", domain, 0), "oracle_lift": .5},
        ])
    selected, report = build_rl(rows, [])
    assert report["selection"]["excluded_difficulty"] == 6
    assert all(x["oracle_lift"] >= .125 for x in selected if x["slice"] == "oracle_gap")
