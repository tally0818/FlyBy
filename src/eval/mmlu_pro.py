'Prepare the deterministic 500-question MMLU-Pro sample.'
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from ..common.io import load_jsonl, write_json, write_jsonl
from ..data.mcq import BOXED_LETTER_INSTRUCTION, LETTERS


def _mmlu_cfg(cfg: dict) -> dict:
    block = cfg.get("mmlu_pro")
    if not isinstance(block, dict):
        raise ValueError("config is missing the mmlu_pro block")
    return block


def _paths(cfg: dict) -> dict[str, Path]:
    raw = _mmlu_cfg(cfg)["paths"]
    return {key: Path(value) for key, value in raw.items()}


def _render_item(row: dict, source_index: int) -> dict:
    options = [str(option).strip() for option in row.get("options") or []]
    answer_index = int(row["answer_index"])
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(
            f"MMLU-Pro row {source_index} has {len(options)} options; expected 2-{len(LETTERS)}"
        )
    if not 0 <= answer_index < len(options):
        raise ValueError(
            f"MMLU-Pro row {source_index} has invalid answer_index={answer_index}"
        )
    source_answer = str(row.get("answer") or "").strip().upper()
    expected_answer = LETTERS[answer_index]
    if source_answer and source_answer != expected_answer:
        raise ValueError(
            f"MMLU-Pro row {source_index} disagrees on answer: "
            f"answer={source_answer!r}, answer_index={answer_index}"
        )

    stem = str(row["question"]).strip()
    lines = [stem, ""]
    lines.extend(f"({LETTERS[index]}) {option}" for index, option in enumerate(options))
    lines.extend(["", BOXED_LETTER_INSTRUCTION])
    question_id = row.get("question_id", source_index)
    category = str(row.get("category") or "unknown")
    return {
        "id": f"mmlu_pro:{question_id}",
        "dataset": "mmlu_pro",
        "domain": category,
        "question": "\n".join(lines),
        "gold": expected_answer,
        "answer_format": "mcq",
        "choices": options,
        "perm": list(range(len(options))),
        "stem": stem,
        "meta": {
            "question_id": question_id,
            "category": category,
            "src": row.get("src"),
            "source_answer": expected_answer,
            "source_answer_index": answer_index,
            "source_index": source_index,
        },
    }


def _sample_digest(items: Iterable[dict]) -> str:
    payload = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_sample(
    cfg: dict,
    *,
    force: bool = False,
    dataset_loader: Callable[..., object] | None = None,
) -> dict:
    'Download, shuffle and materialize the fixed MMLU-Pro subset.'
    mcfg = _mmlu_cfg(cfg)
    source = mcfg["dataset"]
    sample_cfg = mcfg["sample"]
    paths = _paths(cfg)
    manifest_path = paths["sample_manifest"]
    sample_path = paths["sample"]
    expected = {
        "dataset": str(source["name"]),
        "split": str(source.get("split", "test")),
        "revision": source.get("revision"),
        "seed": int(sample_cfg["seed"]),
        "sample_size": int(sample_cfg["size"]),
    }

    if sample_path.is_file() and manifest_path.is_file() and not force:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"existing sample manifest does not match config: {actual} != {expected}; "
                "rerun with --force"
            )
        items = load_jsonl(sample_path)
        if len(items) != expected["sample_size"] or _sample_digest(items) != manifest.get(
            "sample_id_sha256"
        ):
            raise RuntimeError(f"existing sample is incomplete or changed: {sample_path}")
        print(f"[prepare] validated existing {sample_path} ({len(items)} rows)")
        return manifest

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    kwargs = {
        "path": source["name"],
        "split": source.get("split", "test"),
    }
    revision = source.get("revision")
    if revision:
        kwargs["revision"] = revision
    dataset = dataset_loader(**kwargs)
    if len(dataset) < expected["sample_size"]:
        raise ValueError(
            f"requested {expected['sample_size']} rows from a split containing only {len(dataset)}"
        )

    indices = list(range(len(dataset)))
    random.Random(expected["seed"]).shuffle(indices)
    selected_indices = indices[: expected["sample_size"]]
    items = [_render_item(dict(dataset[index]), index) for index in selected_indices]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        duplicates = [item_id for item_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"MMLU-Pro question ids are not unique: {duplicates[:5]}")

    write_jsonl(sample_path, items)
    manifest = {
        **expected,
        "source_rows": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "sample_id_sha256": _sample_digest(items),
        "categories": dict(sorted(Counter(item["domain"] for item in items).items())),
        "sample_path": str(sample_path),
    }
    write_json(manifest_path, manifest)
    print(f"[prepare] wrote {len(items)} shuffled rows -> {sample_path}")
    return manifest
