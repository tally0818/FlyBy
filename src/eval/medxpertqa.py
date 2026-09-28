'Prepare the deterministic 500-question MedXpertQA sample.'
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from ..common.io import load_jsonl, write_json, write_jsonl
from ..data.mcq import BOXED_LETTER_INSTRUCTION, LETTERS


def _med_cfg(cfg: dict) -> dict:
    block = cfg.get("medxpertqa")
    if not isinstance(block, dict):
        raise ValueError("config is missing the medxpertqa block")
    return block


def _paths(cfg: dict) -> dict[str, Path]:
    return {key: Path(value) for key, value in _med_cfg(cfg)["paths"].items()}


def _question_stem(question: object) -> str:
    text = str(question or "").strip()
    marker = "Answer Choices:"
    return text.partition(marker)[0].rstrip() if marker in text else text


def _render_item(row: dict, source_index: int) -> dict:
    source_id = str(row.get("id") or "").strip()
    stem = _question_stem(row.get("question"))
    raw_options = row.get("options")
    if not source_id:
        raise ValueError(f"MedXpertQA row {source_index} is missing id")
    if not stem:
        raise ValueError(f"MedXpertQA row {source_index} is missing question")
    if not isinstance(raw_options, dict):
        raise ValueError(f"MedXpertQA row {source_index} options must be a mapping")

    option_keys = [letter for letter in LETTERS if letter in raw_options]
    unexpected = sorted(set(str(key) for key in raw_options) - set(option_keys))
    if option_keys != list(LETTERS) or unexpected:
        raise ValueError(
            f"MedXpertQA row {source_index} must contain options A-J; "
            f"found {sorted(str(key) for key in raw_options)}"
        )
    options = [str(raw_options[letter]).strip() for letter in option_keys]
    if any(not option for option in options):
        raise ValueError(f"MedXpertQA row {source_index} contains an empty option")

    gold = str(row.get("label") or "").strip().upper()
    if gold not in option_keys:
        raise ValueError(
            f"MedXpertQA row {source_index} has invalid label={row.get('label')!r}"
        )

    lines = [stem, ""]
    lines.extend(f"({letter}) {raw_options[letter]}" for letter in option_keys)
    lines.extend(["", BOXED_LETTER_INSTRUCTION])
    body_system = str(row.get("body_system") or "unknown").strip() or "unknown"
    return {
        "id": f"medxpertqa_text:{source_id}",
        "dataset": "medxpertqa_text",
        "domain": body_system,
        "question": "\n".join(lines),
        "gold": gold,
        "answer_format": "mcq",
        "choices": options,
        "perm": list(range(len(options))),
        "stem": stem,
        "meta": {
            "source_id": source_id,
            "source_index": source_index,
            "medical_task": row.get("medical_task"),
            "body_system": body_system,
            "question_type": row.get("question_type"),
            "source_label": gold,
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
    'Download and materialize the fixed MedXpertQA Text/test sample.'
    mcfg = _med_cfg(cfg)
    source = mcfg["dataset"]
    sample_cfg = mcfg["sample"]
    paths = _paths(cfg)
    sample_path = paths["sample"]
    manifest_path = paths["sample_manifest"]
    expected = {
        "dataset": str(source["name"]),
        "subset": str(source.get("subset", "Text")),
        "split": str(source.get("split", "test")),
        "data_file": str(source.get("data_file", "Text/test.jsonl")),
        "revision": source.get("revision"),
        "seed": int(sample_cfg["seed"]),
        "sample_size": int(sample_cfg["size"]),
    }

    if sample_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"existing MedXpertQA manifest does not match config: {actual} != {expected}; "
                "rerun with --force"
            )
        items = load_jsonl(sample_path)
        if len(items) != expected["sample_size"] or _sample_digest(items) != manifest.get(
            "sample_id_sha256"
        ):
            raise RuntimeError(
                f"existing MedXpertQA sample is incomplete or changed: {sample_path}"
            )
        print(f"[prepare] validated existing {sample_path} ({len(items)} rows)")
        return manifest

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    data_file = str(source.get("data_file", "Text/test.jsonl"))
    if Path(data_file).is_file():
        resolved_data_file = data_file
    else:
        revision = str(source.get("revision") or "main")
        resolved_data_file = (
            f"https://huggingface.co/datasets/{source['name']}/resolve/"
            f"{revision}/{data_file}"
        )


    kwargs = {"path": "json", "data_files": resolved_data_file, "split": "train"}
    dataset = dataset_loader(**kwargs)

    expected_source_rows = source.get("expected_source_rows")
    if expected_source_rows is not None and len(dataset) != int(expected_source_rows):
        raise RuntimeError(
            f"MedXpertQA Text/test changed: expected {expected_source_rows} rows, "
            f"found {len(dataset)}; pin/review the revision before evaluating"
        )
    if len(dataset) < expected["sample_size"]:
        raise ValueError(
            f"requested {expected['sample_size']} MedXpertQA problems from only {len(dataset)} rows"
        )

    indices = list(range(len(dataset)))
    random.Random(expected["seed"]).shuffle(indices)
    items = [
        _render_item(dict(dataset[index]), index)
        for index in indices[: expected["sample_size"]]
    ]
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        duplicates = [item_id for item_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"MedXpertQA ids are not unique: {duplicates[:5]}")

    write_jsonl(sample_path, items)
    manifest = {
        **expected,
        "source_rows": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "sample_id_sha256": _sample_digest(items),
        "body_systems": dict(sorted(Counter(item["domain"] for item in items).items())),
        "question_types": dict(
            sorted(Counter(str(item["meta"]["question_type"]) for item in items).items())
        ),
        "medical_tasks": dict(
            sorted(Counter(str(item["meta"]["medical_task"]) for item in items).items())
        ),
        "answer_format": "mcq_A_to_J",
        "grader": "exact_letter",
        "sample_path": str(sample_path),
    }
    write_json(manifest_path, manifest)
    print(f"[prepare] wrote {len(items)} MedXpertQA Text rows -> {sample_path}")
    return manifest
