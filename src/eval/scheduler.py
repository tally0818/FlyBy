"""Schedule main-suite evaluation workers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..common.io import load_jsonl
from .cost import load_throughput, throughput_for


@dataclass(frozen=True)
class EvalTask:
    name: str
    mode: str
    model: str | None
    run_id: str
    suites: tuple[str, ...]
    repeats: int
    needs_gpu: bool
    throughput_model: str | None = None
    resume: bool = False


@dataclass
class RunningTask:
    task: EvalTask
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    gpu: str | None


@dataclass
class GPUReservation:
    gpu: str
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    ready_path: Path

def _record_counts(path: Path) -> tuple[int, dict[str, set[int]]]:
    count = 0
    reps: dict[str, set[int]] = defaultdict(set)
    if not path.is_file():
        return count, reps
    for record in load_jsonl(path):
        count += 1
        reps[str(record["id"])].add(int(record["rep"]))
    return count, reps


def _is_complete(path: Path, items_path: Path, repeats: int) -> bool:
    if not path.is_file() or not items_path.is_file():
        return False
    expected_ids = {str(item["id"]) for item in load_jsonl(items_path)}
    count, reps = _record_counts(path)
    expected_reps = set(range(repeats))
    return (
        count == len(expected_ids) * repeats
        and set(reps) == expected_ids
        and all(values == expected_reps for values in reps.values())
    )


def _task_command(config_path: str, task: EvalTask, *, no_cost: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.eval.run_eval",
        "--config",
        config_path,
        "--mode",
        task.mode,
        "--suites",
        ",".join(task.suites),
        "--k",
        str(task.repeats),
        "--run-id",
        task.run_id,
    ]
    if task.model:
        command.extend(["--model", task.model])
    if task.throughput_model:
        command.extend(["--throughput-model", task.throughput_model])
    if task.resume:
        command.append("--resume")
    if no_cost:
        command.append("--no-cost")
    return command


def _launch(
    config_path: str,
    task: EvalTask,
    log_dir: Path,
    *,
    gpu: str | None,
    no_cost: bool = False,
) -> RunningTask:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.name}.log"
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    command = _task_command(config_path, task, no_cost=no_cost)
    log_handle.write(f"\n$ {' '.join(command)}\n")
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("PYTHONHASHSEED", str(42))
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    where = f"GPU {gpu}" if gpu is not None else "API/CPU"
    print(f"[launch] {task.name} on {where} (pid={process.pid}, log={log_path})")
    return RunningTask(task, process, log_handle, log_path, gpu)


def _terminate(running: Iterable[RunningTask]) -> None:
    values = list(running)
    for item in values:
        if item.process.poll() is None:
            item.process.terminate()
    for item in values:
        try:
            item.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            item.process.kill()
        item.log_handle.close()


def _start_gpu_reservations(
    gpus: list[str],
    log_dir: Path,
    *,
    memory_mib: int,
    startup_timeout_s: float,
) -> list[GPUReservation]:
    'Start one persistent CUDA marker per selected physical/visible GPU.'
    log_dir.mkdir(parents=True, exist_ok=True)
    reservations: list[GPUReservation] = []
    try:
        for gpu in gpus:
            gpu_label = "".join(char if char.isalnum() else "_" for char in str(gpu))
            log_path = log_dir / f"gpu_guard_{gpu_label}.log"
            ready_path = log_dir / f"gpu_guard_{gpu_label}.ready.json"
            ready_path.unlink(missing_ok=True)
            log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                "-m",
                "src.eval.gpu_reservation",
                "--memory-mib",
                str(memory_mib),
                "--ready-file",
                str(ready_path),
            ]
            log_handle.write(f"\n$ {' '.join(command)}\n")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command, stdout=log_handle, stderr=subprocess.STDOUT, env=env
            )
            reservations.append(
                GPUReservation(gpu, process, log_handle, log_path, ready_path)
            )

        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            failed = [item for item in reservations if item.process.poll() is not None]
            if failed:
                details = ", ".join(
                    f"GPU {item.gpu} exit={item.process.returncode} log={item.log_path}"
                    for item in failed
                )
                raise RuntimeError(f"GPU reservation process failed: {details}")
            if all(item.ready_path.is_file() for item in reservations):
                for item in reservations:
                    metadata = json.loads(item.ready_path.read_text(encoding="utf-8"))
                    print(
                        f"[gpu-guard] GPU {item.gpu} reserved by pid={metadata['pid']} "
                        f"({metadata['memory_mib']} MiB marker)"
                    )
                return reservations
            time.sleep(0.25)
        raise RuntimeError(
            f"GPU reservation processes were not ready within {startup_timeout_s}s"
        )
    except BaseException:
        _stop_gpu_reservations(reservations)
        raise


def _stop_gpu_reservations(reservations: Iterable[GPUReservation]) -> None:
    values = list(reservations)
    for item in values:
        if item.process.poll() is None:
            item.process.terminate()
    for item in values:
        try:
            item.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            item.process.kill()
        item.log_handle.close()
        item.ready_path.unlink(missing_ok=True)
        print(f"[gpu-guard] released GPU {item.gpu}")


def _gpu_is_idle(gpu: str) -> bool:
    'Whether nvidia-smi reports no compute process on a physical GPU.'
    command = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"cannot query GPU {gpu} availability: {detail}")
    entries = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "no running processes" not in line.lower()
    ]
    return not entries


def _run_scheduled(
    config_path: str,
    tasks: list[EvalTask],
    gpus: list[str],
    log_dir: Path,
    *,
    no_cost: bool = False,
    on_complete: Callable[[EvalTask], None] | None = None,
    late_gpus: list[str] | None = None,
    late_gpu_poll_s: float = 60.0,
    activate_late_gpu: Callable[[str], None] | None = None,
    release_gpu: Callable[[str], None] | None = None,
    max_gpus: int | None = None,
) -> list[dict]:
    gpu_pending = [task for task in tasks if task.needs_gpu]
    cpu_pending = [task for task in tasks if not task.needs_gpu]
    available_gpus = list(gpus)
    waiting_gpus = list(late_gpus or [])
    active_gpu: dict[str, RunningTask] = {}
    active_cpu: list[RunningTask] = []
    failures: list[dict] = []
    next_late_poll = 0.0

    if max_gpus is not None:
        if max_gpus <= 0:
            raise ValueError("max_gpus must be positive")
        if len(available_gpus) > max_gpus:
            raise ValueError("initial GPU count exceeds max_gpus")

    try:
        for task in cpu_pending:
            active_cpu.append(_launch(config_path, task, log_dir, gpu=None, no_cost=no_cost))
        while gpu_pending or active_gpu or active_cpu:
            now = time.monotonic()
            if gpu_pending and waiting_gpus and now >= next_late_poll:
                next_late_poll = now + late_gpu_poll_s
                for gpu in list(waiting_gpus):
                    if max_gpus is not None and len(available_gpus) >= max_gpus:
                        break
                    if not _gpu_is_idle(gpu):
                        print(f"[gpu-wait] GPU {gpu} is still busy; checking again later")
                        continue
                    try:
                        if activate_late_gpu is not None:
                            activate_late_gpu(gpu)
                    except Exception as exc:
                        print(f"[gpu-wait] GPU {gpu} became idle but guard failed: {exc}")
                        continue
                    waiting_gpus.remove(gpu)
                    available_gpus.append(gpu)
                    print(f"[gpu-join] GPU {gpu} joined the evaluation queue")

            for gpu in available_gpus:
                if gpu_pending and gpu not in active_gpu:
                    active_gpu[gpu] = _launch(
                        config_path, gpu_pending.pop(0), log_dir, gpu=gpu, no_cost=no_cost
                    )

            for gpu, running in list(active_gpu.items()):
                code = running.process.poll()
                if code is None:
                    continue
                running.log_handle.close()
                del active_gpu[gpu]
                if code:
                    failures.append(
                        {
                            "arm": running.task.name,
                            "stage": "evaluation_subprocess",
                            "exit_code": code,
                            "log_path": str(running.log_path),
                            "gpu": gpu,
                        }
                    )
                    print(f"[failed] {running.task.name} exit={code}; see {running.log_path}")
                else:
                    print(f"[done] {running.task.name}")
                    if on_complete is not None:
                        on_complete(running.task)
                if not gpu_pending:
                    available_gpus.remove(gpu)
                    if release_gpu is not None:
                        release_gpu(gpu)

            for running in list(active_cpu):
                code = running.process.poll()
                if code is None:
                    continue
                running.log_handle.close()
                active_cpu.remove(running)
                if code:
                    failures.append(
                        {
                            "arm": running.task.name,
                            "stage": "evaluation_subprocess",
                            "exit_code": code,
                            "log_path": str(running.log_path),
                            "gpu": None,
                        }
                    )
                    print(f"[failed] {running.task.name} exit={code}; see {running.log_path}")
                else:
                    print(f"[done] {running.task.name}")
                    if on_complete is not None:
                        on_complete(running.task)

            if gpu_pending or active_gpu or active_cpu:
                time.sleep(2)
    except BaseException:
        _terminate([*active_gpu.values(), *active_cpu])
        raise
    return failures


def _output_path(cfg: dict, task: EvalTask, suite: str) -> Path:
    return Path(cfg["output_root"]) / task.run_id / f"{suite}.jsonl"


def _input_path(cfg: dict, suite: str) -> Path:
    return Path(cfg["suites"][suite]["path"])


def _task_summary_matches(cfg: dict, task: EvalTask) -> bool:
    'Reject stale outputs produced with a different arm runtime or model.'
    summary_path = Path(cfg["output_root"]) / task.run_id / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if str(summary.get("mode")) != task.mode:
        return False
    if str(summary.get("model")) != str(task.model):
        return False
    if task.mode == "deepseek" and (
        summary.get("thinking") is not True
        or summary.get("budgets") != cfg["budgets"]["deepseek"]
        or summary.get("depth_tiers") != json.loads(json.dumps(cfg["tool"]["depths"]))
    ):
        return False
    return all(suite in (summary.get("suites") or {}) for suite in task.suites)


def _pending_tasks(cfg: dict, tasks: list[EvalTask], *, force: bool) -> list[EvalTask]:
    pending = []
    for task in tasks:
        outputs_complete = all(
            _is_complete(_output_path(cfg, task, suite), _input_path(cfg, suite), task.repeats)
            for suite in task.suites
        )
        metadata_matches = _task_summary_matches(cfg, task)
        complete = outputs_complete and metadata_matches
        if not force and complete:
            print(
                f"[skip] {task.name}: validated complete outputs for "
                f"{', '.join(task.suites)}"
            )
        else:
            if not force and outputs_complete and not metadata_matches:
                print(
                    f"[rerun] {task.name}: output exists but mode/model metadata is stale"
                )
            pending.append(task)
    return pending


def _filter_missing_local_models(
    tasks: list[EvalTask],
) -> tuple[list[EvalTask], list[dict]]:
    runnable = []
    failures = []
    for task in tasks:
        if not task.needs_gpu or not task.model:
            runnable.append(task)
            continue
        candidate = Path(task.model)
        if task.model.startswith(("outputs/", "./", "../", "/")) and not candidate.exists():
            failure = {
                "arm": task.name,
                "stage": "model_preflight",
                "reason": f"local checkpoint does not exist: {candidate}",
            }
            failures.append(failure)
            print(f"[failed] {task.name}: {failure['reason']}; continuing")
        else:
            runnable.append(task)
    return runnable, failures


def _filter_missing_throughput(
    cfg: dict, tasks: list[EvalTask]
) -> tuple[list[EvalTask], list[dict]]:
    if not any(task.needs_gpu and task.model for task in tasks):
        return list(tasks), []
    path = Path(cfg["pricing"]["throughput_json"])
    if not path.is_file():
        raise FileNotFoundError(
            f"throughput table does not exist: {path}; run scripts/bench_throughput.sh "
            "or pass --no-cost"
        )
    table = load_throughput(path)
    runnable = []
    failures = []
    for task in tasks:
        if not task.needs_gpu or not task.model:
            runnable.append(task)
            continue
        model = task.throughput_model or task.model
        try:
            throughput_for(table, model)
        except KeyError:
            failure = {
                "arm": task.name,
                "stage": "throughput_preflight",
                "reason": (
                    f"throughput table {path} has no entry for {model}; "
                    "run scripts/bench_throughput.sh or pass --no-cost"
                ),
            }
            failures.append(failure)
            print(f"[failed] {task.name}: {failure['reason']}; continuing")
        else:
            runnable.append(task)
    return runnable, failures
