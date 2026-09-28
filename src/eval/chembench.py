'Prepare the three ChemBench main-result domains.'
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from ..common.io import load_jsonl, write_json, write_jsonl
from ..data.mcq import BOXED_LETTER_INSTRUCTION, LETTERS, permute


class _ParquetRows(list):
    _fingerprint: str


def _load_parquet_dataset(*, path: str, data_files: str, split: str) -> _ParquetRows:
    "Read raw parquet without depending on datasets' feature-metadata version."
    del path, split
    import pyarrow.parquet as pq

    local_path = Path(data_files)
    if local_path.is_file():
        payload = local_path.read_bytes()
    else:
        with urllib.request.urlopen(data_files) as response:
            payload = response.read()
    rows = _ParquetRows(pq.read_table(io.BytesIO(payload)).to_pylist())
    rows._fingerprint = hashlib.sha256(payload).hexdigest()
    return rows


def _chem_cfg(cfg: dict) -> dict:
    block = cfg.get("chembench")
    if not isinstance(block, dict):
        raise ValueError("config is missing the chembench block")
    return block


def _paths(cfg: dict) -> dict[str, Path]:
    return {key: Path(value) for key, value in _chem_cfg(cfg)["paths"].items()}


def _sample_digest(items: Iterable[dict]) -> str:
    payload = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_target_scores(row: dict, subset: str, source_index: int) -> tuple[list[str], int] | str:
    examples = row.get("examples")
    try:
        if len(examples) != 1:
            return "not_one_example"
        example = dict(examples[0])
    except (TypeError, ValueError):
        return "malformed_examples"

    question = str(example.get("input") or "").strip()
    if not question:
        return "missing_question"
    raw_scores = example.get("target_scores")
    if raw_scores is None:
        return "open_ended"
    try:
        scores = json.loads(raw_scores) if isinstance(raw_scores, str) else dict(raw_scores)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"ChemBench {subset} row {source_index} has invalid target_scores"
        ) from exc
    if not isinstance(scores, dict) or not scores:
        raise ValueError(
            f"ChemBench {subset} row {source_index} target_scores must be a mapping"
        )

    choices = [str(choice).strip() for choice in scores]
    if any(not choice for choice in choices):
        raise ValueError(f"ChemBench {subset} row {source_index} has an empty choice")
    if not 2 <= len(choices) <= len(LETTERS):
        return "unsupported_choice_count"
    try:
        numeric_scores = [float(value) for value in scores.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ChemBench {subset} row {source_index} has non-numeric target scores"
        ) from exc
    if any(value not in (0.0, 1.0) for value in numeric_scores):
        return "non_binary_scores"
    correct = [index for index, value in enumerate(numeric_scores) if value == 1.0]
    if not correct:
        return "no_correct_choice"
    if len(correct) > 1:
        return "multiple_correct_choices"
    return choices, correct[0]


def _render_item(
    row: dict,
    subset: str,
    source_index: int,
    shuffle_seed: int,
) -> tuple[dict | None, str | None]:
    parsed = _parse_target_scores(row, subset, source_index)
    if isinstance(parsed, str):
        return None, parsed
    choices, correct_index = parsed
    source_id = str(row.get("uuid") or "").strip()
    if not source_id:
        raise ValueError(f"ChemBench {subset} row {source_index} is missing uuid")
    example = dict(row["examples"][0])
    stem = str(example["input"]).strip()
    qid = f"chembench:{subset}:{source_id}"
    permutation = permute(len(choices), shuffle_seed, qid)
    displayed = [choices[index] for index in permutation]
    gold = LETTERS[permutation.index(correct_index)]
    lines = [stem, ""]
    lines.extend(f"({LETTERS[index]}) {choice}" for index, choice in enumerate(displayed))
    lines.extend(["", BOXED_LETTER_INSTRUCTION])
    return (
        {
            "id": qid,
            "dataset": "chembench",
            "domain": subset,
            "question": "\n".join(lines),
            "gold": gold,
            "answer_format": "mcq",
            "choices": displayed,
            "perm": permutation,
            "stem": stem,
            "meta": {
                "source_uuid": source_id,
                "source_index": source_index,
                "source_name": row.get("name"),
                "source_correct_index": correct_index,
                "subfield": row.get("subfield"),
                "description": row.get("description"),
                "keywords": list(row.get("keywords") or []),
                "in_humansubset_w_tool": bool(row.get("in_humansubset_w_tool")),
                "in_humansubset_wo_tool": bool(row.get("in_humansubset_wo_tool")),
            },
        },
        None,
    )


def _expected_manifest(cfg: dict) -> dict:
    source = _chem_cfg(cfg)["dataset"]
    selection = _chem_cfg(cfg)["selection"]
    subsets = {
        str(name): {
            "suite": str(spec["suite"]),
            "expected_source_rows": int(spec["expected_source_rows"]),
            "expected_selected_rows": int(spec["expected_selected_rows"]),
        }
        for name, spec in source["subsets"].items()
    }
    return {
        "dataset": str(source["name"]),
        "revision": str(source["revision"]),
        "split": str(source.get("split", "train")),
        "selection": "single_answer_multiple_choice",
        "shuffle_seed": int(selection["shuffle_seed"]),
        "subsets_config": subsets,
    }


def _validate_existing(cfg: dict, manifest: dict, expected: dict) -> None:
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"existing ChemBench manifest does not match config: {actual} != {expected}; "
            "rerun with --force"
        )
    for subset, spec in expected["subsets_config"].items():
        sample_path = Path(cfg["suites"][spec["suite"]]["path"])
        items = load_jsonl(sample_path)
        recorded = manifest["subsets"][subset]
        if (
            len(items) != spec["expected_selected_rows"]
            or _sample_digest(items) != recorded.get("sample_id_sha256")
        ):
            raise RuntimeError(f"existing ChemBench sample is incomplete or changed: {sample_path}")


def prepare_sample(
    cfg: dict,
    *,
    force: bool = False,
    dataset_loader: Callable[..., object] | None = None,
) -> dict:
    'Download and materialize all selected single-answer MCQs.'
    chem_cfg = _chem_cfg(cfg)
    source = chem_cfg["dataset"]
    expected = _expected_manifest(cfg)
    manifest_path = _paths(cfg)["sample_manifest"]
    sample_paths = [
        Path(cfg["suites"][spec["suite"]]["path"])
        for spec in expected["subsets_config"].values()
    ]
    if manifest_path.is_file() and all(path.is_file() for path in sample_paths) and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing(cfg, manifest, expected)
        print(
            f"[prepare] validated existing ChemBench samples "
            f"({manifest['total_selected_rows']} rows)"
        )
        return manifest

    if dataset_loader is None:
        dataset_loader = _load_parquet_dataset

    revision = expected["revision"]
    split = expected["split"]
    shuffle_seed = expected["shuffle_seed"]
    prepared: dict[str, list[dict]] = {}
    subset_manifests = {}
    all_ids = []
    for subset, spec in source["subsets"].items():
        suite = str(spec["suite"])
        data_file = str(
            spec.get("data_file", f"{subset}/train-00000-of-00001.parquet")
        )
        resolved_data_file = (
            data_file
            if Path(data_file).is_file()
            else (
                f"https://huggingface.co/datasets/{source['name']}/resolve/"
                f"{revision}/{data_file}"
            )
        )
        dataset = dataset_loader(
            path="parquet", data_files=resolved_data_file, split=split
        )
        expected_source_rows = int(spec["expected_source_rows"])
        if len(dataset) != expected_source_rows:
            raise RuntimeError(
                f"ChemBench {subset} changed: expected {expected_source_rows} rows, "
                f"found {len(dataset)}; review and update the pinned revision"
            )

        items = []
        exclusions = Counter()
        for source_index, raw_row in enumerate(dataset):
            item, reason = _render_item(
                dict(raw_row), str(subset), source_index, shuffle_seed
            )
            if item is None:
                exclusions[str(reason)] += 1
            else:
                items.append(item)
                all_ids.append(item["id"])
        expected_selected_rows = int(spec["expected_selected_rows"])
        if len(items) != expected_selected_rows:
            raise RuntimeError(
                f"ChemBench {subset} selection changed: expected "
                f"{expected_selected_rows} single-answer MCQs, found {len(items)}"
            )
        prepared[suite] = items
        subset_manifests[str(subset)] = {
            "suite": suite,
            "source_rows": len(dataset),
            "selected_rows": len(items),
            "excluded_rows": len(dataset) - len(items),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "sample_path": str(cfg["suites"][suite]["path"]),
            "sample_id_sha256": _sample_digest(items),
            "subfields": dict(sorted(Counter(str(item["meta"]["subfield"]) for item in items).items())),
        }

    if len(all_ids) != len(set(all_ids)):
        duplicates = [item_id for item_id, count in Counter(all_ids).items() if count > 1]
        raise ValueError(f"ChemBench ids are not unique: {duplicates[:5]}")
    for suite, items in prepared.items():
        write_jsonl(cfg["suites"][suite]["path"], items)
        print(f"[prepare] wrote {len(items)} rows -> {cfg['suites'][suite]['path']}")
    manifest = {
        **expected,
        "total_source_rows": sum(value["source_rows"] for value in subset_manifests.values()),
        "total_selected_rows": sum(value["selected_rows"] for value in subset_manifests.values()),
        "subsets": subset_manifests,
        "answer_format": "mcq_A_to_J_variable_width",
        "grader": "exact_letter",
    }
    write_json(manifest_path, manifest)
    return manifest
