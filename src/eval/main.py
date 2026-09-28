'Evaluate the main benchmark suite and report hard-set estimated pass@8.'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..common.config import load_config
from ..common.io import load_jsonl, read_json, write_json
from ..common.run import load_dotenv
from . import arxivmath, chembench, medxpertqa, mmlu_pro
from .run_eval import configure_deepseek
from .scheduler import (
    EvalTask,
    _filter_missing_local_models,
    _filter_missing_throughput,
    _gpu_is_idle,
    _pending_tasks,
    _run_scheduled,
    _start_gpu_reservations,
    _stop_gpu_reservations,
    _task_command,
)


BASE_ROLLOUTS = 16
HARD_MAX_CORRECT = 4
PASS_K = 8
MAIN_MODES = ("flyby", "ours", "vanilla", "search_r1", "deepseek")
BENCHMARK_ORDER = (
    "ArXivMath",
    "GPQA-Diamond",
    "SuperGPQA",
    "MedXpertQA",
    "MMLU-Pro",
    "ChemBench",
)


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    benchmark: str
    suite: str
    config_path: str


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("arxivmath", "ArXivMath", "arxivmath", "configs/eval.yaml"),
    ComponentSpec("gpqa", "GPQA-Diamond", "gpqa_diamond", "configs/eval.yaml"),
    ComponentSpec("supergpqa", "SuperGPQA", "supergpqa_test", "configs/eval.yaml"),
    ComponentSpec(
        "medxpertqa",
        "MedXpertQA",
        "medxpertqa_text_500",
        "configs/medxpertqa.yaml",
    ),
    ComponentSpec(
        "mmlu_pro",
        "MMLU-Pro",
        "mmlu_pro_500",
        "configs/mmlu_pro.yaml",
    ),
    ComponentSpec(
        "chem_organic",
        "ChemBench",
        "chembench_organic_chemistry_mcq_384",
        "configs/chembench.yaml",
    ),
    ComponentSpec(
        "chem_materials",
        "ChemBench",
        "chembench_materials_science_mcq_57",
        "configs/chembench.yaml",
    ),
    ComponentSpec(
        "chem_inorganic",
        "ChemBench",
        "chembench_inorganic_chemistry_mcq_49",
        "configs/chembench.yaml",
    ),
)


def estimated_pass_at_k(correct: int, total: int, k: int = PASS_K) -> float:
    if not 0 <= correct <= total:
        raise ValueError(f"correct must be in [0, {total}], got {correct}")
    if not 1 <= k <= total:
        raise ValueError(f"k must be in [1, {total}], got {k}")
    if correct == 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return float(1.0 - math.comb(total - correct, k) / math.comb(total, k))


def _id_digest(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_grouped(path: Path, expected_reps: int) -> dict[str, list[dict]]:
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, row in enumerate(load_jsonl(path), start=1):
        problem_id = str(row["id"])
        rep = int(row["rep"])
        if rep in grouped[problem_id]:
            raise ValueError(f"{path}:{line_number}: duplicate ({problem_id}, {rep})")
        if not isinstance(row.get("correct"), bool):
            raise ValueError(f"{path}:{line_number}: correct must be bool")
        grouped[problem_id][rep] = row
    expected = set(range(expected_reps))
    result = {}
    for problem_id, reps in grouped.items():
        if set(reps) != expected:
            raise ValueError(
                f"{path}: {problem_id} has reps {sorted(reps)}, expected "
                f"0..{expected_reps - 1}"
            )
        result[problem_id] = [reps[index] for index in range(expected_reps)]
    return result


def _hard_digest(components: Sequence[dict]) -> str:
    payload = [
        (
            row["key"],
            [(problem["id"], int(problem["base_correct"])) for problem in row["hard_problems"]],
        )
        for row in components
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_hard_manifest(manifest: dict) -> None:
    if manifest.get("version") != 1:
        raise ValueError("unsupported hard manifest version")
    if int(manifest.get("baseline_rollouts", 0)) != BASE_ROLLOUTS:
        raise ValueError("hard manifest must use exactly 16 baseline rollouts")
    if int(manifest.get("hard_max_correct", -1)) != HARD_MAX_CORRECT:
        raise ValueError("hard manifest threshold is not inclusive correct_count <= 4")
    components = manifest.get("components") or []
    if [row.get("key") for row in components] != [spec.key for spec in COMPONENTS]:
        raise ValueError("hard manifest components differ from the main-suite contract")
    for row in components:
        problems = row.get("hard_problems") or []
        ids = [str(problem["id"]) for problem in problems]
        if len(ids) != len(set(ids)) or len(ids) != int(row["n_hard"]):
            raise ValueError(f"{row['key']}: invalid hard ID list")
        if any(not 0 <= int(problem["base_correct"]) <= HARD_MAX_CORRECT for problem in problems):
            raise ValueError(f"{row['key']}: hard label lies outside 0..4 correct")
    if manifest.get("hard_set_sha256") != _hard_digest(components):
        raise ValueError("hard manifest digest mismatch")
    if int(manifest["n_full"]) != sum(int(row["n_full"]) for row in components):
        raise ValueError("hard manifest full count mismatch")
    if int(manifest["n_hard"]) != sum(int(row["n_hard"]) for row in components):
        raise ValueError("hard manifest hard count mismatch")


def _parse_gpus(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("--gpus must be a non-empty unique CSV list")
    return values


def _discover_gpu_ids() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"cannot discover GPUs with nvidia-smi: {detail}")
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return gpus


def _partition_gpu_pool(candidates: list[str], gpu_count: int) -> tuple[list[str], list[str]]:
    if gpu_count <= 0 or len(candidates) < gpu_count:
        raise ValueError(
            f"requested {gpu_count} GPUs but candidate pool has {len(candidates)}"
        )
    idle, busy = [], []
    for gpu in candidates:
        (idle if _gpu_is_idle(gpu) else busy).append(gpu)
    immediate = idle[:gpu_count]
    waiting = busy + idle[gpu_count:]
    print(
        f"[gpu-pool] candidates={candidates}; immediate={immediate}; "
        f"waiting={waiting}; limit={gpu_count}"
    )
    return immediate, waiting


def _health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _start_search_retriever(eval_cfg: dict, run_dir: Path):
    search_cfg = eval_cfg["search_r1"]
    health_url = str(search_cfg["health_url"])
    if _health_ok(health_url):
        print(f"[retriever] reusing {health_url}")
        return None, None
    from ..search_r1.setup_wiki18 import materialize

    materialize(Path("data/search_r1/wiki18"))
    log_path = run_dir / "logs" / "retriever.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        ["bash", "scripts/run_search_r1_retriever.sh", "--device", "cpu"],
        stdout=handle,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    deadline = time.monotonic() + float(search_cfg["startup_timeout_s"])
    while time.monotonic() < deadline:
        if process.poll() is not None:
            handle.close()
            raise RuntimeError(f"retriever failed; see {log_path}")
        if _health_ok(health_url):
            print(f"[retriever] ready at {health_url}")
            return process, handle
        time.sleep(2)
    process.terminate()
    handle.close()
    raise RuntimeError(f"retriever startup timed out; see {log_path}")


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("run id may contain only letters, numbers, '.', '_' and '-'")
    return value


def _model_label(model: str) -> str:
    parts = Path(model).parts
    for index, part in enumerate(parts):
        if part.startswith("global_step_") and index > 0:
            value = f"{parts[index - 1]}-{part}"
            break
    else:
        value = parts[-1] if parts else "model"
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "model"
    return value[:96]


def _default_run_id(model: str) -> str:
    return f"{_model_label(model)}-{time.strftime('%Y%m%d_%H%M%S')}"


def balance_components(components: Sequence[dict], workers: int) -> list[list[dict]]:
    'Greedy LPT partition by evaluated hard-problem count.'
    if workers <= 0:
        raise ValueError("workers must be positive")
    shards: list[list[dict]] = [[] for _ in range(min(workers, len(components)))]
    loads = [0] * len(shards)
    order = {str(row["key"]): index for index, row in enumerate(components)}
    for component in sorted(
        components, key=lambda row: (-int(row["n_eval"]), order[str(row["key"])])
    ):
        index = min(range(len(shards)), key=lambda value: (loads[value], value))
        shards[index].append(component)
        loads[index] += int(component["n_eval"])
    return shards


def _source_suite_path(spec: ComponentSpec) -> Path:
    cfg = load_config(spec.config_path)
    try:
        return Path(cfg["suites"][spec.suite]["path"])
    except KeyError as exc:
        raise ValueError(f"{spec.config_path} does not define suite {spec.suite}") from exc


def prepare_run(
    *,
    model: str,
    run_id: str,
    output_root: Path,
    hard_manifest_path: Path,
    num_gpus: int,
    rollouts: int,
    throughput_model: str | None,
    tool_budget_usd_per_worker: float,
    no_cost: bool,
    mode: str = "ours",
) -> tuple[Path, dict, dict]:
    hard_manifest = read_json(hard_manifest_path)
    _validate_hard_manifest(hard_manifest)
    hard_by_key = {str(row["key"]): row for row in hard_manifest["components"]}
    components = []
    suite_cfg = {}
    for spec in COMPONENTS:
        source_path = _source_suite_path(spec)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"missing full evaluation sample {source_path} ({spec.key}); "
                "run the corresponding prepare/build-data stage first"
            )
        rows = load_jsonl(source_path)
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{source_path}: duplicate problem IDs")
        frozen = hard_by_key[spec.key]
        digest_matches = (
            "all_ids_sha256" not in frozen
            or _id_digest(ids) == frozen["all_ids_sha256"]
        )
        if len(rows) != int(frozen["n_full"]) or not digest_matches:
            raise ValueError(
                f"{spec.key}: full sample differs from the frozen Qwen baseline population"
            )
        hard_ids = {str(problem["id"]) for problem in frozen["hard_problems"]}
        if not hard_ids <= set(ids):
            raise ValueError(f"{spec.key}: frozen hard IDs are missing from {source_path}")
        hard_rows = [row for row in rows if str(row["id"]) in hard_ids]
        hard_input = output_root / run_id / "hard_inputs" / f"{spec.suite}.jsonl"
        hard_input.parent.mkdir(parents=True, exist_ok=True)
        with hard_input.open("w", encoding="utf-8") as handle:
            for row in hard_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        components.append(
            {
                "key": spec.key,
                "benchmark": spec.benchmark,
                "suite": spec.suite,
                "source_path": str(source_path),
                "eval_path": str(hard_input),
                "n_eval": len(hard_rows),
                "n_hard": len(hard_ids),
            }
        )
        suite_cfg[spec.suite] = {"path": str(hard_input), "k": rollouts}

    worker_count = 1 if mode == "deepseek" else min(num_gpus, len(components))
    shards = balance_components(components, worker_count)
    shard_rows = [
        {
            "name": f"worker-{index:02d}",
            "suites": [str(row["suite"]) for row in shard],
            "components": [str(row["key"]) for row in shard],
            "n_eval": sum(int(row["n_eval"]) for row in shard),
        }
        for index, shard in enumerate(shards)
    ]

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = load_config("configs/eval.yaml")
    eval_cfg["suites"] = suite_cfg
    eval_cfg["output_root"] = str(run_dir / "workers")
    eval_cfg["tool"]["budget_usd"] = float(tool_budget_usd_per_worker)
    if mode == "deepseek":
        model = configure_deepseek(eval_cfg, model)
        throughput_model = None
    eval_config_path = run_dir / "eval_config.json"

    contract = {
        "model": model,
        "mode": mode,
        "rollouts": rollouts,
        "pass_k": PASS_K,
        "num_workers": len(shard_rows),
        "throughput_model": throughput_model,
        "tool_budget_usd_per_worker": float(tool_budget_usd_per_worker),
        "cost_accounting": not no_cost,
        "hard_set_sha256": hard_manifest["hard_set_sha256"],
        "generation_population": "frozen hard-set problems only",
    }
    if mode == "deepseek":
        contract["thinking"] = True
        contract["depth_tiers"] = json.loads(json.dumps(eval_cfg["tool"]["depths"]))
        contract["budgets"] = eval_cfg["budgets"]["deepseek"]
    run_manifest_path = run_dir / "manifest.json"
    if run_manifest_path.is_file():
        previous = read_json(run_manifest_path)
        if previous.get("contract") != contract:
            raise ValueError(
                f"cannot reuse {run_dir}: run contract changed; choose a new --run-id"
            )
    write_json(eval_config_path, eval_cfg)
    run_manifest = {
        "version": 1,
        "run_id": run_id,
        "contract": contract,
        "hard_manifest": str(hard_manifest_path),
        "eval_config": str(eval_config_path),
        "n_eval": sum(int(row["n_eval"]) for row in components),
        "n_hard": sum(int(row["n_hard"]) for row in components),
        "components": components,
        "shards": shard_rows,
    }
    write_json(run_manifest_path, run_manifest)
    return run_dir, run_manifest, eval_cfg


def _tasks(run_manifest: dict) -> list[EvalTask]:
    contract = run_manifest["contract"]
    return [
        EvalTask(
            name=str(shard["name"]),
            mode=str(contract["mode"]),
            model=str(contract["model"]),
            run_id=str(shard["name"]),
            suites=tuple(str(value) for value in shard["suites"]),
            repeats=int(contract["rollouts"]),
            needs_gpu=contract["mode"] != "deepseek",
            throughput_model=contract["throughput_model"],
            resume=True,
        )
        for shard in run_manifest["shards"]
    ]


def _print_plan(config_path: Path, tasks: Sequence[EvalTask], no_cost: bool) -> None:
    for task in tasks:
        command = _task_command(str(config_path), task, no_cost=no_cost)
        device = "CUDA_VISIBLE_DEVICES=<assigned> " if task.needs_gpu else ""
        print(
            f"[plan] {task.name}: {len(task.suites)} suites, "
            f"{device}{' '.join(command)}"
        )


def run_workers(
    run_dir: Path,
    run_manifest: dict,
    eval_cfg: dict,
    *,
    candidates: list[str],
    num_gpus: int,
    gpu_poll_s: float,
    no_cost: bool,
    gpu_guard: bool,
    dry_run: bool,
) -> None:
    tasks = _tasks(run_manifest)
    config_path = Path(run_manifest["eval_config"])
    pending = _pending_tasks(eval_cfg, tasks, force=False)
    pending, model_failures = _filter_missing_local_models(pending)
    if model_failures:
        raise FileNotFoundError(model_failures[0]["reason"])
    if pending and not no_cost:
        pending, throughput_failures = _filter_missing_throughput(eval_cfg, pending)
        if throughput_failures:
            raise RuntimeError(throughput_failures[0]["reason"])
    _print_plan(config_path, pending, no_cost)
    if dry_run or not pending:
        return

    load_dotenv()
    if run_manifest["contract"]["mode"] in ("ours", "deepseek") and not os.environ.get(
        "OPENROUTER_API_KEY"
    ):
        raise RuntimeError("OPENROUTER_API_KEY is required for API-backed evaluation")

    if all(not task.needs_gpu for task in pending):
        failures = _run_scheduled(
            str(config_path), pending, [], run_dir / "logs" / "workers", no_cost=no_cost,
        )
        write_json(run_dir / "failures.json", failures)
        if failures:
            raise RuntimeError(f"API evaluation failed; see {run_dir / 'failures.json'}")
        return

    retriever_process = None
    retriever_handle = None
    if run_manifest["contract"]["mode"] == "search_r1":
        retriever_process, retriever_handle = _start_search_retriever(eval_cfg, run_dir)

    claim_count = min(num_gpus, len(pending))
    immediate, waiting = _partition_gpu_pool(candidates, claim_count)
    reservations = []

    def reserve(gpus: list[str]) -> None:
        if gpu_guard and gpus:
            reservations.extend(
                _start_gpu_reservations(
                    gpus,
                    run_dir / "logs" / "gpu_guard",
                    memory_mib=512,
                    startup_timeout_s=60,
                )
            )

    def release(gpu: str) -> None:
        matched = [item for item in reservations if item.gpu == gpu]
        if not matched:
            return
        _stop_gpu_reservations(matched)
        for item in matched:
            reservations.remove(item)

    try:
        reserve(immediate)

        def activate_waiting(gpu: str) -> None:
            reserve([gpu])

        failures = _run_scheduled(
            str(config_path),
            pending,
            immediate,
            run_dir / "logs" / "workers",
            no_cost=no_cost,
            late_gpus=waiting,
            late_gpu_poll_s=gpu_poll_s,
            activate_late_gpu=activate_waiting,
            release_gpu=release,
            max_gpus=claim_count,
        )
        write_json(run_dir / "failures.json", failures)
        if failures:
            raise RuntimeError(
                f"{len(failures)} evaluation worker(s) failed; see {run_dir / 'failures.json'}"
            )
    finally:
        _stop_gpu_reservations(reservations)
        if retriever_process is not None:
            retriever_process.terminate()
            try:
                retriever_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                retriever_process.kill()
            retriever_handle.close()
            print("[retriever] stopped")


def _problem_point(problem_id: str, rows: Sequence[dict], *, hard: bool) -> dict:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "id": problem_id,
        "hard": hard,
        "correct": correct,
        "estimated_pass_at_8": estimated_pass_at_k(correct, total, PASS_K),
        "accuracy": correct / total,
        "pass_at_n": float(correct > 0),
        "call_rate": sum(int(row.get("lrm_calls", 0)) > 0 for row in rows) / total,
        "calls_per_rollout": sum(int(row.get("lrm_calls", 0)) for row in rows) / total,
        "cost_usd_per_rollout": sum(float(row.get("cost_usd", 0.0)) for row in rows) / total,
        "cost_usd_observed": sum(float(row.get("cost_usd", 0.0)) for row in rows),
    }


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def _summarize(points: Sequence[dict], rollouts: int) -> dict:
    if not points:
        raise ValueError("cannot summarize an empty problem set")
    return {
        "n_problems": len(points),
        "estimated_pass_at_8": _mean(float(row["estimated_pass_at_8"]) for row in points),
        f"accuracy_at_{rollouts}": _mean(float(row["accuracy"]) for row in points),
        f"pass_at_{rollouts}": _mean(float(row["pass_at_n"]) for row in points),
        "call_rate": _mean(float(row["call_rate"]) for row in points),
        "calls_per_rollout": _mean(float(row["calls_per_rollout"]) for row in points),
        "cost_usd_per_rollout": _mean(float(row["cost_usd_per_rollout"]) for row in points),
        f"cost_usd_per_problem_{rollouts}": _mean(
            float(row["cost_usd_observed"]) for row in points
        ),
        "estimated_cost_usd_per_problem_8": 8.0
        * _mean(float(row["cost_usd_per_rollout"]) for row in points),
    }


def _macro_summary(rows: Sequence[dict], rollouts: int) -> dict:
    keys = (
        "estimated_pass_at_8",
        f"accuracy_at_{rollouts}",
        f"pass_at_{rollouts}",
        "call_rate",
        "calls_per_rollout",
        "cost_usd_per_rollout",
        f"cost_usd_per_problem_{rollouts}",
        "estimated_cost_usd_per_problem_8",
    )
    return {
        "n_benchmarks": len(rows),
        "n_problems": sum(int(row["n_problems"]) for row in rows),
        **{key: _mean(float(row[key]) for row in rows) for key in keys},
    }


def aggregate_run(run_dir: Path) -> dict:
    run_manifest = read_json(run_dir / "manifest.json")
    hard_manifest = read_json(run_manifest["hard_manifest"])
    _validate_hard_manifest(hard_manifest)
    hard_rows = {str(row["key"]): row for row in hard_manifest["components"]}
    shard_by_component = {
        str(component): str(shard["name"])
        for shard in run_manifest["shards"]
        for component in shard["components"]
    }
    rollouts = int(run_manifest["contract"]["rollouts"])
    points_by_benchmark: dict[str, list[dict]] = defaultdict(list)

    for component in run_manifest["components"]:
        key = str(component["key"])
        suite = str(component["suite"])
        worker = shard_by_component[key]
        output_path = run_dir / "workers" / worker / f"{suite}.jsonl"
        grouped = _load_grouped(output_path, rollouts)
        hard_ids = {
            str(problem["id"]) for problem in hard_rows[key]["hard_problems"]
        }
        if set(grouped) != hard_ids:
            raise ValueError(f"{key}: evaluated IDs differ from the frozen hard set")
        points = [
            _problem_point(problem_id, rows, hard=True)
            for problem_id, rows in sorted(grouped.items())
        ]
        points_by_benchmark[str(component["benchmark"])].extend(points)

    scopes = {}
    csv_rows = []
    for scope in ("hard",):
        benchmark_rows = []
        all_points = []
        for benchmark in BENCHMARK_ORDER:
            selected = [
                row
                for row in points_by_benchmark[benchmark]
                if bool(row["hard"])
            ]
            all_points.extend(selected)
            summary = {"benchmark": benchmark, **_summarize(selected, rollouts)}
            benchmark_rows.append(summary)
            csv_rows.append({"scope": scope, "level": "benchmark", "name": benchmark, **summary})
        micro = _summarize(all_points, rollouts)
        macro = _macro_summary(benchmark_rows, rollouts)
        csv_rows.append({"scope": scope, "level": "aggregate", "name": "micro", **micro})
        csv_rows.append({"scope": scope, "level": "aggregate", "name": "macro_6", **macro})
        scopes[scope] = {
            "by_benchmark": benchmark_rows,
            "micro": micro,
            "macro_6": macro,
        }

    result = {
        "run_id": run_manifest["run_id"],
        "model": run_manifest["contract"]["model"],
        "mode": run_manifest["contract"]["mode"],
        "main_metric": "hard.macro_6.estimated_pass_at_8",
        "protocol": {
            "generated_population": "frozen hard-set problems only",
            "rollouts_per_problem": rollouts,
            "hard_definition": hard_manifest["hard_definition"],
            "headline_metric": "hard.macro_6.estimated_pass_at_8",
            "estimator": f"1 - C({rollouts}-c, 8) / C({rollouts}, 8)",
            "arxivmath": "pooled base and June 2026 releases",
            "chembench": "problem pool of Organic + Materials + Inorganic",
        },
        "hard": scopes["hard"],
    }
    write_json(run_dir / "results.json", result)
    columns = [
        "scope",
        "level",
        "name",
        "n_benchmarks",
        "n_problems",
        "estimated_pass_at_8",
        f"accuracy_at_{rollouts}",
        f"pass_at_{rollouts}",
        "call_rate",
        "calls_per_rollout",
        "cost_usd_per_rollout",
        f"cost_usd_per_problem_{rollouts}",
        "estimated_cost_usd_per_problem_8",
    ]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n[hard main result]")
    for row in scopes["hard"]["by_benchmark"]:
        print(
            f"  {row['benchmark']:<14} n={row['n_problems']:>4}  "
            f"estimated pass@8={100 * row['estimated_pass_at_8']:6.2f}%  "
            f"call={100 * row['call_rate']:6.2f}%"
        )
    headline = scopes["hard"]["macro_6"]
    print(
        f"  {'Macro-6':<14} n={headline['n_problems']:>4}  "
        f"estimated pass@8={100 * headline['estimated_pass_at_8']:6.2f}%"
    )
    print(f"[aggregate] wrote {run_dir / 'results.json'} and {run_dir / 'results.csv'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="checkpoint or model ID; optional for deepseek")
    parser.add_argument("--mode", choices=MAIN_MODES, default="ours")
    parser.add_argument("--run-id", help="restartable output id; generated from model+time by default")
    parser.add_argument("--gpus", type=_parse_gpus, default=None, help="candidate GPU IDs, e.g. 2,4")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-poll-s", type=float, default=30.0)
    parser.add_argument("--throughput-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tool-budget-usd-per-worker", type=float, default=1000.0)
    parser.add_argument("--hard-manifest", default="configs/main_hard_pass8.json")
    parser.add_argument("--output-root", default="outputs/evals")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--no-cost", action="store_true")
    parser.add_argument("--no-gpu-guard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def prepare_sources(*, force: bool = False) -> None:
    core = (
        Path("data/eval/arxivmath_base.jsonl"),
        Path("data/eval/gpqa_diamond.jsonl"),
        Path("data/eval/supergpqa_test.jsonl"),
    )
    missing = [str(path) for path in core if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing core evaluation data: " + ", ".join(missing)
            + "; run scripts/build_data.sh first"
        )
    arxivmath.prepare_sample(load_config("configs/arxivmath.yaml"), force=force)
    medxpertqa.prepare_sample(load_config("configs/medxpertqa.yaml"), force=force)
    mmlu_pro.prepare_sample(load_config("configs/mmlu_pro.yaml"), force=force)
    chembench.prepare_sample(load_config("configs/chembench.yaml"), force=force)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # ``flyby`` is the project-facing name for the existing tool-use mode.
    if args.mode == "flyby":
        args.mode = "ours"
    hard_manifest_path = Path(args.hard_manifest)
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if args.gpu_poll_s <= 0:
        parser.error("--gpu-poll-s must be positive")
    if args.tool_budget_usd_per_worker <= 0:
        parser.error("--tool-budget-usd-per-worker must be positive")

    output_root = Path(args.output_root)
    if args.aggregate_only:
        if not args.run_id:
            parser.error("--aggregate-only requires --run-id")
        aggregate_run(output_root / _safe_run_id(args.run_id))
        return
    if not args.model:
        if args.mode == "deepseek":
            args.model = load_config("configs/eval.yaml")["tool"]["depths"][3]["model"]
        else:
            parser.error("--model is required")
    load_dotenv()
    prepare_sources(force=args.force_prepare)
    try:
        run_id = _safe_run_id(args.run_id or _default_run_id(args.model))
    except ValueError as exc:
        parser.error(str(exc))
    run_dir, run_manifest, eval_cfg = prepare_run(
        model=args.model,
        run_id=run_id,
        output_root=output_root,
        hard_manifest_path=hard_manifest_path,
        num_gpus=args.num_gpus,
        rollouts=BASE_ROLLOUTS,
        throughput_model=args.throughput_model,
        tool_budget_usd_per_worker=args.tool_budget_usd_per_worker,
        no_cost=args.no_cost,
        mode=args.mode,
    )
    print(
        f"[prepare] run_id={run_id} hard={run_manifest['n_eval']} "
        f"workers={len(run_manifest['shards'])}"
    )
    for shard in run_manifest["shards"]:
        print(
            f"  {shard['name']}: n_hard={shard['n_eval']} "
            f"suites={','.join(shard['suites'])}"
        )
    if args.prepare_only:
        return

    candidates = []
    if args.mode != "deepseek":
        env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
        candidates = args.gpus or (
            _parse_gpus(env_gpus) if env_gpus else _discover_gpu_ids()
        )
        if len(candidates) < min(args.num_gpus, len(run_manifest["shards"])):
            parser.error(
                f"requested {args.num_gpus} GPUs but candidate pool has only {len(candidates)}"
            )
    run_workers(
        run_dir,
        run_manifest,
        eval_cfg,
        candidates=candidates,
        num_gpus=args.num_gpus,
        gpu_poll_s=args.gpu_poll_s,
        no_cost=args.no_cost,
        gpu_guard=not args.no_gpu_guard,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        aggregate_run(run_dir)


if __name__ == "__main__":
    main()
