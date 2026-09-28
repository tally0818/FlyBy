'Convert train.jsonl (and eval jsonl) into verl-tool parquet format.'
from __future__ import annotations

import argparse

from ..common.io import load_jsonl
from ..tools import protocol
from ..tools import search_r1


def to_rows(
    items: list[dict],
    split: str,
    *,
    with_tool: bool = True,
    search_r1_prompt: bool = False,
) -> list[dict]:
    if not with_tool and search_r1_prompt:
        raise ValueError("search_r1_prompt and with_tool=False are mutually exclusive")
    if search_r1_prompt:
        system = search_r1.system_prompt()
    else:
        system = protocol.system_prompt(with_tool=with_tool)
    rows = []
    for i, item in enumerate(items):
        rows.append({
            "data_source": item["dataset"],
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": item["question"]},
            ],
            "ability": item["domain"],
            "reward_model": {"style": "rule", "ground_truth": item["gold"]},
            "extra_info": {
                "id": item["id"],
                "split": split,
                "index": i,
                "question": item["question"],
                "stem": item.get("stem") or item["question"],
                "gold": item["gold"],
                "answer_format": item["answer_format"],
                "domain": item["domain"],
                "skill": item.get("skill"),
                "p_solve": item.get("p_solve"),
                "slice": item.get("slice"),
                "p_solve_briefed": item.get("p_solve_briefed"),
                "oracle_lift": item.get("oracle_lift"),
            },
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train.jsonl")
    parser.add_argument("--out", default="data/processed/train.parquet")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--no-tool",
        action="store_true",
        help="Use the plain self-reasoning system prompt without llm_query instructions.",
    )
    parser.add_argument(
        "--search-r1",
        action="store_true",
        help="Use the Search-R1 search/information protocol instead of llm_query.",
    )
    args = parser.parse_args()

    items = load_jsonl(args.input)
    if args.no_tool and args.search_r1:
        parser.error("--no-tool and --search-r1 are mutually exclusive")
    rows = to_rows(
        items,
        args.split,
        with_tool=not args.no_tool,
        search_r1_prompt=args.search_r1,
    )

    import pandas as pd

    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
