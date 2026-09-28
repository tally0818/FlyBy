'Build training and evaluation JSONL files from configured datasets.'
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..common.config import load_config
from ..common.io import write_jsonl, write_json
from . import mcq
from .decontaminate import decontaminate, item_text, stem_hash

QUESTION_KEYS = ["problem", "question", "Question", "prompt"]
ANSWER_KEYS = ["answer", "Answer", "solution", "final_answer", "gold"]





SLICE_KEYS = ("_hf_source", "competition", "discipline", "field", "subfield", "difficulty")


def _first(row: dict, keys: list[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _source_ids(item: dict, fields: list[str]) -> set[str]:
    'Every source identifier an item carries, across differently-named fields.'
    meta = item.get("meta") or {}
    return {str(meta[f]).strip() for f in fields if meta.get(f) not in (None, "")}


def _drop_source_id_overlap(
    items: list[dict], eval_source_ids: set[str], fields: list[str]
) -> tuple[list[dict], list[dict]]:
    'Remove training items whose source identifiers occur in evaluation data.'
    kept, removed = [], []
    for item in items:
        hit = _source_ids(item, fields) & eval_source_ids
        if hit:
            removed.append({"id": item["id"], "hit_source_id": sorted(hit)[0]})
        else:
            kept.append(item)
    return kept, removed


def stratified_take(items: list[dict], n: int, key_fn, rng: random.Random) -> list[dict]:
    'Draw `n` items, allocating across strata proportionally (largest remainder).'
    if n <= 0 or not items:
        return []
    n = min(n, len(items))
    strata: dict = defaultdict(list)
    for item in items:
        strata[key_fn(item)].append(item)
    keys = sorted(strata, key=str)
    for key in keys:
        rng.shuffle(strata[key])
    total = len(items)
    exact = {key: len(strata[key]) * n / total for key in keys}
    take = {key: int(exact[key]) for key in keys}
    for key in sorted(keys, key=lambda k: exact[k] - int(exact[k]), reverse=True)[
        : n - sum(take.values())
    ]:
        take[key] += 1
    out: list[dict] = []
    for key in keys:
        out += strata[key][: take[key]]
    return out


def _load_hf(spec: dict):
    'Rows for a spec. `hf` may be a list, whose datasets are concatenated.'
    from datasets import load_dataset

    names = spec["hf"] if isinstance(spec["hf"], list) else [spec["hf"]]
    revisions = spec.get("revision")
    if isinstance(revisions, list) and len(revisions) != len(names):
        raise ValueError("revision list must match the hf dataset list")
    config = spec.get("subset") or spec.get("hf_config")
    split = spec.get("split", "train")
    rows = []
    for index, name in enumerate(names):
        revision = revisions[index] if isinstance(revisions, list) else revisions
        if not revision:
            raise ValueError(f"{name}: a pinned revision is required")
        ds = load_dataset(name, config, revision=revision)
        part = split if split in ds else list(ds.keys())[0]
        for row in ds[part]:
            rows.append({**row, "_hf_source": name})
    return rows


def _is_english(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    return ascii_ratio > 0.9


def _dataset_name(spec: dict, fallback: str | None = None) -> str:
    if spec.get("name"):
        return spec["name"]
    if fallback:
        return fallback
    hf = spec["hf"]
    first = hf[0] if isinstance(hf, list) else hf
    return first.split("/")[-1].replace("-", "_").lower()


def _item_id(name: str, row: dict, i: int, multi_source: bool) -> str:
    'Stable unique id. Pooled sources get their release tag so ids cannot collide.'
    rid = row.get("id") or row.get("uuid") or row.get("Record ID") or i
    if multi_source:
        tag = str(row.get("_hf_source", "")).split("/")[-1]
        return f"{name}:{tag}:{rid}"
    return f"{name}:{rid}"


def _meta(row: dict, cfg: dict) -> dict:
    id_fields = cfg.get("decontam", {}).get("id_fields", [])
    meta = {f: row[f] for f in id_fields if f in row and row[f] is not None}
    for key in SLICE_KEYS:
        if row.get(key):
            meta[key.lstrip("_")] = row[key]
    return meta





def _math_items(rows: list[dict], name: str, domain: str, spec: dict, cfg: dict) -> list[dict]:
    multi = isinstance(spec.get("hf"), list) and len(spec["hf"]) > 1
    items = []
    for i, row in enumerate(rows):
        question = _first(row, QUESTION_KEYS)
        if isinstance(question, list):
            question = question[-1].get("content", "")
        gold = _first(row, ANSWER_KEYS)
        if isinstance(gold, dict):
            gold = gold.get("ground_truth", "")
        if not question or gold is None:
            continue
        if spec.get("filter_lang") == "en" and not _is_english(str(question)):
            continue
        items.append({
            "id": _item_id(name, row, i, multi),
            "dataset": name,
            "domain": domain,
            "question": str(question).strip(),
            "gold": str(gold).strip(),
            "answer_format": "math",
            "choices": None,
            "perm": None,
            "stem": str(question).strip(),

            "meta": _meta(row, cfg),
        })
    return items


def _render_mcq_item(
    qid: str, name: str, domain: str, stem: str,
    correct: str, incorrect: list[str], cfg: dict, meta: dict,
) -> dict:
    rendered = mcq.render_mcq(
        stem=stem, correct=correct, incorrect=incorrect,
        shuffle_seed=cfg["mcq"]["shuffle_seed"], qid=qid,
    )
    return {
        "id": qid,
        "dataset": name,
        "domain": domain,
        "question": rendered["question"],
        "gold": rendered["gold"],
        "answer_format": "mcq",
        "choices": rendered["choices"],
        "perm": rendered["perm"],
        "stem": rendered["stem"],
        "meta": meta,
    }


def _options_mcq_items(rows: list[dict], name: str, domain: str, spec: dict, cfg: dict) -> list[dict]:
    'Rows shaped {question, options[], answer letter} -> MCQ items (GooseReason).'
    multi = isinstance(spec.get("hf"), list) and len(spec["hf"]) > 1
    items = []
    for i, row in enumerate(rows):
        options = [str(o) for o in (row.get("options") or [])]
        answer = str(row.get("answer", "")).strip().upper()
        idx = ord(answer) - ord("A") if len(answer) == 1 else -1
        if not 0 <= idx < len(options) or not 2 <= len(options) <= len(mcq.LETTERS):
            continue
        qid = _item_id(name, row, i, multi)
        meta = _meta(row, cfg)
        meta.update({"source_index": i, "n_choices": len(options), "source_answer": answer})
        items.append(_render_mcq_item(
            qid, name, domain, str(row["question"]),
            options[idx], options[:idx] + options[idx + 1:], cfg, meta,
        ))
    return items


def _squash(text: str) -> str:
    return "".join(str(text).split())


def _supergpqa_items(rows: list[dict], name: str, domain: str, spec: dict, cfg: dict) -> list[dict]:
    'SuperGPQA rows -> MCQ items. 10-way, so `mcq.LETTERS` has to reach J.'
    multi = isinstance(spec.get("hf"), list) and len(spec["hf"]) > 1
    items, mismatched, unparsed = [], 0, 0
    for i, row in enumerate(rows):
        options = [str(o) for o in (row.get("options") or [])]
        letter = str(row.get("answer_letter") or "").strip().upper()
        idx = ord(letter) - ord("A") if len(letter) == 1 else -1
        stem = str(row.get("question") or "").strip()
        if not stem or not 0 <= idx < len(options) or not 2 <= len(options) <= len(mcq.LETTERS):
            unparsed += 1
            continue
        answer_text = str(row.get("answer") or "").strip()
        if answer_text and _squash(answer_text) != _squash(options[idx]):
            mismatched += 1
            continue
        qid = _item_id(name, row, i, multi)
        meta = _meta(row, cfg)
        meta["n_choices"] = len(options)
        items.append(_render_mcq_item(
            qid, name, domain, stem,
            options[idx], options[:idx] + options[idx + 1:], cfg, meta,
        ))
    if mismatched or unparsed:
        rate = mismatched / len(rows) if rows else 0.0
        print(f"[{name}] dropped {unparsed} unparseable, {mismatched} answer_letter/answer "
              f"mismatches ({rate:.2%})")




        assert rate < 0.05 or len(rows) < 50, (
            f"{name}: {rate:.1%} of rows disagree between answer_letter and answer — "
            f"inspect the release before trusting either field"
        )
    return items


def _gpqa_items(rows: list[dict], name: str, domain: str, spec: dict, cfg: dict) -> list[dict]:
    items = []
    for i, row in enumerate(rows):
        rid = str(row.get("Record ID", i))
        meta = _meta(row, cfg)
        meta["record_id"] = rid
        items.append(_render_mcq_item(
            f"{name}:{rid}", name, domain, row["Question"],
            row["Correct Answer"],
            [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]],
            cfg, meta,
        ))
    return items


_PARSERS = {
    "math": _math_items,
    "options_mcq": _options_mcq_items,
    "supergpqa": _supergpqa_items,
    "gpqa": _gpqa_items,
}


def apply_include(items: list[dict], include: dict | None, name: str = "") -> list[dict]:
    'Filter items by configured metadata values.'
    if not include:
        return items
    wanted = {f: {str(v).strip().lower() for v in vals} for f, vals in include.items()}
    kept = [x for x in items
            if all(str((x.get("meta") or {}).get(f, "")).strip().lower() in vals
                   for f, vals in wanted.items())]
    for field in wanted:
        seen = sorted({str((x.get("meta") or {}).get(field, "")) for x in items})
        print(f"[{name}] include.{field}: kept {len(kept)}/{len(items)}; "
              f"values present in source: {seen}")
    assert kept or not items, (
        f"{name}: include filter {include} matched nothing — compare the value spelling "
        f"against the values listed above"
    )
    return kept


def build_source(spec: dict, cfg: dict, name: str | None = None, domain: str | None = None) -> list[dict]:
    'Items for one configured source. `format` picks the row parser.'
    fmt = spec.get("format", "math")
    if fmt not in _PARSERS:
        raise ValueError(f"unknown source format {fmt!r}; expected one of {sorted(_PARSERS)}")
    name = _dataset_name(spec, name)
    domain = domain or spec.get("domain") or "math"
    items = _PARSERS[fmt](_load_hf(spec), name, domain, spec, cfg)
    return apply_include(items, spec.get("include"), name)


def build_eval_suite(name: str, spec: dict, cfg: dict) -> list[dict]:
    return build_source(spec, cfg, name=name)





def self_split(items: list[dict], spec: dict) -> tuple[list[dict], list[dict]]:
    'Create a deterministic stratified train/test split.'
    rng = random.Random(int(spec.get("seed", 0)))
    stratify_by = list(spec.get("stratify_by") or [])
    items = sorted(items, key=lambda x: str(x["id"]))

    def key(item):
        meta = item.get("meta") or {}
        return tuple(str(meta.get(k, "")) for k in stratify_by)

    test = stratified_take(items, int(spec["test_size"]), key, rng)
    test_ids = {x["id"] for x in test}
    remainder = [x for x in items if x["id"] not in test_ids]
    train_size = spec.get("train_size")
    train = (stratified_take(remainder, int(train_size), key, rng)
             if train_size is not None else remainder)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)

    raw_dir = Path(cfg["paths"]["raw"])
    eval_dir = Path(cfg["paths"]["eval"])



    self_train: dict[str, list[dict]] = defaultdict(list)
    self_test: dict[str, list[dict]] = {}
    for name, spec in (cfg.get("self_split") or {}).items():
        items = build_source(spec, cfg, name=name)
        train_half, test_half = self_split(items, spec)
        suite = spec.get("test_suite") or f"{name}_test"
        self_train[spec["domain"]] += train_half
        self_test[suite] = test_half
        write_json(raw_dir / f"{name}_split_manifest.json", {
            "hf": spec["hf"], "revision": spec["revision"],
            "seed": spec.get("seed"), "stratify_by": spec.get("stratify_by"),
            "total": len(items), "test_size": spec.get("test_size"),
            "train_size": spec.get("train_size"),
            "train": len(train_half), "test": len(test_half), "test_suite": suite,
            "test_ids": sorted(x["id"] for x in test_half),
            "train_ids": sorted(x["id"] for x in train_half),
        })
        print(f"[self-split] {name}: {len(items)} -> train {len(train_half)} / "
              f"test {len(test_half)} (suite {suite})")

    eval_items_by_suite: dict[str, list[dict]] = {}
    if args.split in ("eval", "all"):
        for name, spec in cfg["eval_suites"].items():
            eval_items_by_suite[name] = build_eval_suite(name, spec, cfg)
        eval_items_by_suite.update(self_test)
        for name, items in eval_items_by_suite.items():
            n = write_jsonl(eval_dir / f"{name}.jsonl", items)
            print(f"[eval] {name}: {n} items -> {eval_dir}/{name}.jsonl")
    else:
        from ..common.io import load_jsonl

        for name in list(cfg["eval_suites"]) + list(self_test):
            path = eval_dir / f"{name}.jsonl"
            if path.exists():
                eval_items_by_suite[name] = load_jsonl(path)
        if not eval_items_by_suite:
            raise SystemExit("train build needs eval suites for decontamination; run --split eval first")

    if args.split in ("train", "all"):
        eval_items = [x for items in eval_items_by_suite.values() for x in items]
        eval_stems = [item_text(x) for x in eval_items]

        for domain in ("math", "science"):
            items = list(self_train.get(domain, []))
            for spec in cfg["train_sources"].get(domain) or []:
                items += build_source(spec, cfg, domain=domain)
            if not items:
                print(f"[skip] no train sources configured for {domain}")
                continue

            kept, removed = decontaminate(items, eval_stems, n=cfg["decontam"]["ngram"])




            id_fields = cfg["decontam"].get("id_fields", [])
            eval_src = {sid for x in eval_items for sid in _source_ids(x, id_fields)} \
                if id_fields else set()
            kept, shared_source = _drop_source_id_overlap(kept, eval_src, id_fields)
            removed += shared_source


            eval_ids = {x["id"] for x in eval_items}
            eval_hashes = {stem_hash(item_text(x)) for x in eval_items}
            assert not ({x["id"] for x in kept} & eval_ids), f"{domain}: id overlap with eval"
            assert not ({stem_hash(item_text(x)) for x in kept} & eval_hashes), \
                f"{domain}: stem overlap with eval survived decontamination"
            assert not [x for x in kept if _source_ids(x, id_fields) & eval_src], \
                f"{domain}: source-id overlap survived filtering"

            by_source = Counter(x["dataset"] for x in kept)
            write_jsonl(raw_dir / f"{domain}.jsonl", kept)
            write_json(raw_dir / f"{domain}_decontam_report.json",
                       {"kept": len(kept), "removed": len(removed),
                        "removed_ngram": len(removed) - len(shared_source),
                        "removed_source_id": len(shared_source),
                        "by_source": dict(by_source), "removals": removed})
            print(f"[train] {domain}: kept {len(kept)}, removed {len(removed)} "
                  f"(ngram {len(removed) - len(shared_source)}, source-id {len(shared_source)}) "
                  f"-> {raw_dir}/{domain}.jsonl")
            print(f"        sources: {dict(by_source)}")


if __name__ == "__main__":
    main()
