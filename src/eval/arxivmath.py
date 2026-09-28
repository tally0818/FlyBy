'Prepare the June 2026 source rows pooled into ArXivMath.'
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from ..common.io import load_jsonl, write_json, write_jsonl
from ..data.build_dataset import _math_items


class _ParquetRows(list):
    _fingerprint: str


def _load_parquet_dataset(*, path: str, data_files: str, split: str) -> _ParquetRows:
    'Read the pinned parquet directly, independent of Hub feature metadata.'
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


def _arxiv_cfg(cfg: dict) -> dict:
    block = cfg.get("arxivmath")
    if not isinstance(block, dict):
        raise ValueError("config is missing the arxivmath block")
    return block


def _paths(cfg: dict) -> dict[str, Path]:
    return {key: Path(value) for key, value in _arxiv_cfg(cfg)["paths"].items()}


def _sample_digest(items: Iterable[dict]) -> str:
    payload = "\n".join(str(item["id"]) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_manifest(cfg: dict) -> dict:
    benchmark = _arxiv_cfg(cfg)
    source = benchmark["dataset"]
    suites = list(benchmark["evaluation"]["suites"])
    if len(suites) != 1:
        raise ValueError("arxivmath.evaluation.suites must contain exactly one suite")
    return {
        "dataset": str(source["name"]),
        "revision": str(source["revision"]),
        "split": str(source.get("split", "train")),
        "data_file": str(source["data_file"]),
        "expected_source_rows": int(source["expected_source_rows"]),
        "suite": str(suites[0]),
    }


def _validate_existing(cfg: dict, manifest: dict, expected: dict) -> None:
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"existing ArXivMath 06/26 manifest does not match config: "
            f"{actual} != {expected}; rerun with --force"
        )
    items = load_jsonl(_paths(cfg)["june"])
    if (
        len(items) != expected["expected_source_rows"]
        or _sample_digest(items) != manifest.get("sample_id_sha256")
    ):
        raise RuntimeError("existing ArXivMath 06/26 sample is incomplete or changed")


def _write_pool(paths: dict[str, Path]) -> None:
    if not paths["base"].is_file():
        raise FileNotFoundError(
            f"missing {paths['base']}; run scripts/build_data.sh first"
        )
    items = load_jsonl(paths["base"]) + load_jsonl(paths["june"])
    ids = [str(item["id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("ArXivMath pooled sample contains duplicate IDs")
    write_jsonl(paths["sample"], items)
    print(f"[prepare] pooled {len(items)} ArXivMath rows -> {paths['sample']}")


def prepare_sample(
    cfg: dict,
    *,
    force: bool = False,
    dataset_loader: Callable[..., object] | None = None,
) -> dict:
    'Download and materialize all 49 questions from the pinned monthly release.'
    source = _arxiv_cfg(cfg)["dataset"]
    expected = _expected_manifest(cfg)
    paths = _paths(cfg)
    sample_path = paths["june"]
    manifest_path = paths["june_manifest"]
    if sample_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing(cfg, manifest, expected)
        _write_pool(paths)
        print(f"[prepare] validated existing {sample_path} ({len(load_jsonl(sample_path))} rows)")
        return manifest

    if dataset_loader is None:
        dataset_loader = _load_parquet_dataset
    data_file = expected["data_file"]
    resolved_data_file = (
        data_file
        if Path(data_file).is_file()
        else (
            f"https://huggingface.co/datasets/{expected['dataset']}/resolve/"
            f"{expected['revision']}/{data_file}"
        )
    )
    dataset = dataset_loader(
        path="parquet",
        data_files=resolved_data_file,
        split=expected["split"],
    )
    if len(dataset) != expected["expected_source_rows"]:
        raise RuntimeError(
            f"ArXivMath 06/26 changed: expected {expected['expected_source_rows']} rows, "
            f"found {len(dataset)}; review and update the pinned revision"
        )

    rows = [{**dict(row), "_hf_source": expected["dataset"]} for row in dataset]
    spec = {
        "hf": expected["dataset"],
        "split": expected["split"],
        "domain": "math",
        "format": "math",
    }
    items = _math_items(rows, expected["suite"], "math", spec, cfg)
    if len(items) != len(dataset):
        raise RuntimeError(
            f"ArXivMath 06/26 parser retained {len(items)}/{len(dataset)} rows; "
            "inspect missing problem/answer fields"
        )
    ids = [str(item["id"]) for item in items]
    if len(ids) != len(set(ids)):
        duplicates = [item_id for item_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"ArXivMath 06/26 ids are not unique: {duplicates[:5]}")

    write_jsonl(sample_path, items)
    manifest = {
        **expected,
        "source_rows": len(dataset),
        "selected_rows": len(items),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "sample_id_sha256": _sample_digest(items),
        "sample_path": str(sample_path),
        "answer_format": "math",
        "grader": "math-verify symbolic/numeric equivalence",
        "source_papers": len({item["meta"].get("source") for item in items}),
    }
    write_json(manifest_path, manifest)
    print(f"[prepare] wrote {len(items)} ArXivMath 06/26 rows -> {sample_path}")
    _write_pool(paths)
    return manifest
