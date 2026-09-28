'Run standalone model evaluation.'
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..common.config import load_config
from ..common.io import load_jsonl, write_jsonl, write_json
from ..common.run import load_dotenv, new_run_id, set_seed
from ..tools import guard, protocol
from ..tools.lrm_client import LRMClient, normalize_tiers
from ..tools import search_r1
from ..tools.search_r1 import SearchR1Client
from ..tools.token_budget import generation_allowance
from . import cost as cost_mod
from .grader import grade

MAX_MODEL_LEN = 32768 + 8192
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-4B"


def configure_deepseek(cfg: dict, model: str | None = None) -> str:
    """Select the standalone backend and enable its reasoning mode."""
    tier = normalize_tiers(cfg["tool"]["depths"])[3]
    if model is not None and model != tier["model"]:
        raise ValueError(f"deepseek mode uses the configured depth-3 backend: {tier['model']}")
    cfg["tool"] = {
        **cfg["tool"],
        "depths": {3: tier},
        "thinking": True,
        "timeout_s": cfg.get("deepseek", {}).get("timeout_s", 600),
    }
    cfg["strict_api_errors"] = True
    cfg["_throughput_model"] = None
    return tier["model"]


def _load_eval_config(path: str, overrides=()) -> dict:
    'Load an eval config and restore typed depth keys after JSON round trips.'
    cfg = load_config(path, overrides)
    cfg["tool"]["depths"] = normalize_tiers(cfg["tool"]["depths"])
    return cfg


def _model_run_label(model: str) -> str:
    parts = Path(model).parts
    for index, part in enumerate(parts):
        if part.startswith("global_step_") and index > 0:
            return f"{parts[index - 1]}-{part}"
    return parts[-1] if parts else model


def _run_id_prefix(mode: str, model: str | None) -> str:
    prefix = mode
    if mode == "vanilla":
        prefix += f"-{model or DEFAULT_LOCAL_MODEL}"
    elif mode in ("ours", "search_r1") and model:
        prefix += f"-{_model_run_label(model)}"
    return prefix


class VllmEngine:
    'Thin wrapper over vllm.LLM for batched multi-segment generation.'

    def __init__(self, model: str, sampling: dict, seed: int,
                 gpu_memory_utilization: float = 0.85):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model = model
        self.sampling = sampling
        self.seed = seed
        self.generation_index = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=MAX_MODEL_LEN, seed=seed)

    def chat_prompt(self, question: str, system: str | None = None,
                    enable_thinking: bool | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": question}
        ]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompts: list[str], max_tokens: list[int],
                 stops: list[list[str] | None] | None = None):
        from vllm import SamplingParams

        stops = stops or [None] * len(prompts)
        params = [
            SamplingParams(
                temperature=self.sampling["temperature"],
                top_p=self.sampling["top_p"],
                top_k=self.sampling["top_k"],
                max_tokens=mt,
                stop=st or None,
                include_stop_str_in_output=True,
                seed=self.seed + self.generation_index + i,
            )
            for i, (mt, st) in enumerate(zip(max_tokens, stops))
        ]
        self.generation_index += len(params)
        outs = self.llm.generate(prompts, params)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


@dataclass
class TrajState:
    item: dict
    rep: int
    prompt: str
    context: str
    full_gen: str = ""
    own_tokens: int = 0
    local_input_tokens: int = 0
    turns: int = 0
    calls: list = field(default_factory=list)
    done: bool = False
    tools_enabled: bool = True
    record_trace: bool = False
    trace_events: list[dict] = field(default_factory=list)


def _record_trace_event(state: TrajState, kind: str, text: str, **metadata) -> None:
    'Append an exact context fragment when trace capture was requested.'
    if not state.record_trace:
        return
    state.trace_events.append({"kind": kind, "text": text, **metadata})


def make_client(cfg: dict) -> LRMClient:
    tool = cfg["tool"]
    return LRMClient(
        depth_tiers=tool["depths"],
        budget_usd=tool.get("budget_usd", float("inf")), timeout_s=tool.get("timeout_s", 180),
        max_retries=tool.get("max_retries", 2), concurrency=tool.get("concurrency", 8),
        thinking=tool.get("thinking", False),
    )


def make_search_client(cfg: dict) -> SearchR1Client:
    search = cfg["search_r1"]
    return SearchR1Client(
        search["retriever_url"],
        topk=search.get("topk", 3),
        timeout_s=search.get("timeout_s", 30),
        concurrency=search.get("concurrency", 16),
    )


async def _do_tool_call(state: TrajState, call: protocol.ToolCall, client: LRMClient,
                        cfg: dict) -> None:
    tool_cfg = cfg["tool"]
    item = state.item
    depth = call.depth
    ok, reason = guard.validate_query(call.q, tool_cfg.get("q_max_chars", guard.Q_MAX_CHARS))
    problem_text = item.get("stem") or item["question"]
    reject_overlap, overlap = guard.check_query_overlap(call.q or "", problem_text)
    if ok and reject_overlap:
        ok, reason = False, "query overlaps the problem statement too much"
    if not ok:
        obs = protocol.format_tool_error(reason)
        call_record = {"kind": "query", "depth": depth, "in_toks": 0, "out_toks": 0,
                       "rejected": reason, "q_overlap": overlap}
    else:
        result = await client.aquery(call.q, depth, item["id"])
        text, _ = guard.filter_leakage(result["text"])
        obs = protocol.format_lrm_answer(text)
        call_record = {
            "kind": "query", "depth": depth, "requested_depth": call.depth,
            "in_toks": result["usage"]["prompt_tokens"],
            "out_toks": result["usage"]["completion_tokens"],
            "reasoning_toks": result["usage"].get("reasoning_tokens", 0),
            "model": result.get("model"), "cost_usd": result.get("cost_usd", 0.0),
            "unavailable": result.get("unavailable", False), "q_overlap": overlap,
        }
    if state.record_trace:
        call_record["query"] = call.q
    state.calls.append(call_record)
    state.context += obs
    _record_trace_event(
        state,
        "observation" if ok else "tool_error",
        obs,
        query=call.q,
        requested_depth=call.depth,
        effective_depth=depth,
        valid=bool(ok),
    )


async def _do_tool_calls_with_progress(
    pending: list[tuple[TrajState, protocol.ToolCall]],
    client: LRMClient,
    cfg: dict,
    round_index: int,
) -> None:
    'Run one tool round concurrently while exposing completion progress.'
    from tqdm.auto import tqdm

    effective_depths = [call.depth for _, call in pending]
    depth_summary = " ".join(
        f"d{depth}={effective_depths.count(depth)}" for depth in protocol.DEPTHS
        if depth in effective_depths
    )
    description = f"LRM tool calls round {round_index} [{depth_summary}]"
    tasks = [
        asyncio.create_task(_do_tool_call(state, call, client, cfg))
        for state, call in pending
    ]
    try:
        with tqdm(total=len(tasks), desc=description, unit="call", dynamic_ncols=True) as progress:
            for finished in asyncio.as_completed(tasks):
                await finished
                progress.update(1)
    finally:
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)


async def _do_search_call(
    state: TrajState,
    call: search_r1.SearchCall,
    client: SearchR1Client,
    cfg: dict,
) -> None:
    search_cfg = cfg["search_r1"]
    query = call.query or ""
    if len(query) > search_cfg.get("max_query_chars", 300):
        reason = f"search query exceeds {search_cfg.get('max_query_chars', 300)} characters"
        observation = search_r1.format_search_error(reason)
        record = {
            "kind": "search",
            "query_chars": len(query),
            "num_docs": 0,
            "result_chars": 0,
            "latency_ms": 0.0,
            "rejected": reason,
        }
        valid = False
    else:
        try:
            result = await client.asearch(query)
            if not result["num_docs"] or not result["text"].strip():
                raise RuntimeError("retriever returned no usable document text")
            rendered = search_r1.truncate_result(
                result["text"], search_cfg.get("max_result_chars", 6000)
            )
            observation = search_r1.format_information(rendered)
            record = {
                "kind": "search",
                "query_chars": len(query),
                "num_docs": result["num_docs"],
                "result_chars": len(rendered),
                "latency_ms": result["latency_ms"],
            }
            valid = True
        except Exception as exc:
            reason = str(exc)
            observation = search_r1.format_search_error(reason)
            record = {
                "kind": "search",
                "query_chars": len(query),
                "num_docs": 0,
                "result_chars": 0,
                "latency_ms": 0.0,
                "rejected": reason,
            }
            valid = False
    if state.record_trace:
        record["query"] = query
    state.calls.append(record)
    state.context += observation
    _record_trace_event(
        state,
        "observation" if valid else "tool_error",
        observation,
        query=query,
        valid=valid,
        tool="search",
    )


async def _do_search_calls_with_progress(
    pending: list[tuple[TrajState, search_r1.SearchCall]],
    client: SearchR1Client,
    cfg: dict,
    round_index: int,
) -> None:
    from tqdm.auto import tqdm

    tasks = [
        asyncio.create_task(_do_search_call(state, call, client, cfg))
        for state, call in pending
    ]
    try:
        with tqdm(
            total=len(tasks),
            desc=f"Wiki-18 searches round {round_index}",
            unit="call",
            dynamic_ncols=True,
        ) as progress:
            for finished in asyncio.as_completed(tasks):
                await finished
                progress.update(1)
    finally:
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)


def run_tool_loop(engine: VllmEngine, client: LRMClient, items: list[dict], k: int,
                  budgets: dict, cfg: dict, with_tool: bool = True,
                  save_traces: bool = False,
                  system_prompt: str | None = None) -> list[dict]:
    'Run the tool-augmented policy with a cumulative own-token budget.'
    own_total = budgets["own_tokens_total"]
    max_turns = budgets["max_turns"]
    stops = list(protocol.ACTION_STOP_TOKENS)
    system = system_prompt or protocol.system_prompt(with_tool=with_tool)

    states = [
        TrajState(
            item=item,
            rep=r,
            prompt=(p := engine.chat_prompt(item["question"], system)),
            context=p,
            tools_enabled=with_tool,
            record_trace=save_traces,
        )
        for item in items for r in range(k)
    ]
    tool_round = 0

    while True:
        active = [s for s in states if not s.done]
        if not active:
            break
        prompts, max_toks, stop_lists = [], [], []
        for s in active:
            context_tokens = engine.n_tokens(s.context)
            allowance = generation_allowance(
                total_budget=own_total,
                generated_tokens=s.own_tokens,
                live_capacity=MAX_MODEL_LEN - context_tokens - 64,
                per_turn_cap=own_total,
            )
            if allowance <= 0:
                s.done = True
                continue
            prompts.append(s.context)
            s.local_input_tokens += context_tokens
            max_toks.append(allowance)
            stop_lists.append(stops if s.tools_enabled else None)
        active = [s for s in active if not s.done]
        if not active:
            break

        outs = engine.generate(prompts, max_toks, stop_lists)

        pending: list[tuple[TrajState, protocol.ToolCall]] = []
        for s, (text, n_tok) in zip(active, outs):
            s.own_tokens += n_tok
            s.full_gen += text
            s.context += text
            _record_trace_event(
                s,
                "policy",
                text,
                own_tokens=n_tok,
                tools_enabled=bool(s.tools_enabled),
            )
            call = protocol.parse_tool_call(text) if s.tools_enabled and text.rstrip().endswith(tuple(stops)) else None
            if call is None:
                s.done = True
                continue




            s.turns += 1
            if s.turns > max_turns:
                obs = protocol.format_tool_error("tool limit reached; finish on your own")
                s.context += obs
                _record_trace_event(
                    s,
                    "tool_error",
                    obs,
                    query=call.q,
                    requested_depth=call.depth,
                    effective_depth=call.depth,
                    valid=False,
                )
                s.tools_enabled = False
                continue



            if s.turns == max_turns:
                s.tools_enabled = False
            if not call.valid:
                obs = protocol.format_tool_error(call.error)
                s.context += obs
                call_record = {"kind": call.kind, "depth": call.depth,
                               "in_toks": 0, "out_toks": 0,
                               "rejected": call.error, "q_overlap": 0.0}
                if s.record_trace:
                    call_record["query"] = call.q
                s.calls.append(call_record)
                _record_trace_event(
                    s,
                    "tool_error",
                    obs,
                    query=call.q,
                    requested_depth=call.depth,
                    effective_depth=call.depth,
                    valid=False,
                )
                continue
            pending.append((s, call))

        if pending:
            tool_round += 1
            asyncio.run(_do_tool_calls_with_progress(
                pending, client, cfg, tool_round
            ))

    return [finalize_record(s, cfg, engine.model) for s in states]


def run_search_loop(
    engine: VllmEngine,
    client: SearchR1Client,
    items: list[dict],
    k: int,
    budgets: dict,
    cfg: dict,
    *,
    save_traces: bool = False,
) -> list[dict]:
    'Search-R1 mid-think loop using the canonical Wiki-18/E5 backend.'
    own_total = budgets["own_tokens_total"]
    max_turns = budgets["max_turns"]
    system = search_r1.system_prompt(max_turns=max_turns, topk=client.topk)
    states = [
        TrajState(
            item=item,
            rep=rep,
            prompt=(prompt := engine.chat_prompt(item["question"], system)),
            context=prompt,
            tools_enabled=True,
            record_trace=save_traces,
        )
        for item in items
        for rep in range(k)
    ]
    search_round = 0

    while True:
        active = [state for state in states if not state.done]
        if not active:
            break
        prompts: list[str] = []
        max_tokens: list[int] = []
        stop_lists: list[list[str] | None] = []
        for state in active:
            context_tokens = engine.n_tokens(state.context)
            allowance = generation_allowance(
                total_budget=own_total,
                generated_tokens=state.own_tokens,
                live_capacity=MAX_MODEL_LEN - context_tokens - 64,
                per_turn_cap=own_total,
            )
            if allowance <= 0:
                state.done = True
                continue
            prompts.append(state.context)
            state.local_input_tokens += context_tokens
            max_tokens.append(allowance)
            stop_lists.append(search_r1.ACTION_STOP_TOKENS if state.tools_enabled else None)
        active = [state for state in active if not state.done]
        if not active:
            break

        outputs = engine.generate(prompts, max_tokens, stop_lists)
        pending: list[tuple[TrajState, search_r1.SearchCall]] = []
        for state, (text, n_tokens) in zip(active, outputs):
            state.own_tokens += n_tokens
            state.full_gen += text
            state.context += text
            _record_trace_event(
                state,
                "policy",
                text,
                own_tokens=n_tokens,
                tools_enabled=bool(state.tools_enabled),
            )
            call = (
                search_r1.parse_search_call(text)
                if state.tools_enabled and text.rstrip().endswith(search_r1.SEARCH_CLOSE)
                else None
            )
            if call is None:
                state.done = True
                continue
            state.turns += 1
            if state.turns > max_turns:
                observation = search_r1.format_search_error("search limit reached; finish on your own")
                state.context += observation
                state.tools_enabled = False
                _record_trace_event(
                    state, "tool_error", observation, query=call.query, valid=False, tool="search"
                )
                continue
            if state.turns == max_turns:
                state.tools_enabled = False
            if not call.valid:
                observation = search_r1.format_search_error(call.error or "malformed search call")
                state.context += observation
                record = {
                    "kind": "search",
                    "query_chars": len(call.query or ""),
                    "num_docs": 0,
                    "result_chars": 0,
                    "latency_ms": 0.0,
                    "rejected": call.error,
                }
                if state.record_trace:
                    record["query"] = call.query
                state.calls.append(record)
                _record_trace_event(
                    state, "tool_error", observation, query=call.query, valid=False, tool="search"
                )
                continue
            pending.append((state, call))

        if pending:
            search_round += 1
            asyncio.run(_do_search_calls_with_progress(pending, client, cfg, search_round))

    return [finalize_record(state, cfg, engine.model) for state in states]


def run_vanilla(engine: VllmEngine, items: list[dict], k: int, budgets: dict, cfg: dict,
                save_traces: bool = False) -> list[dict]:
    system = protocol.system_prompt(with_tool=False)
    states = [
        TrajState(
            item=item,
            rep=r,
            prompt=(p := engine.chat_prompt(item["question"], system)),
            context=p,
            record_trace=save_traces,
        )
        for item in items for r in range(k)
    ]
    outs = engine.generate([s.context for s in states],
                           [budgets["max_new_tokens"]] * len(states))
    for s, (text, n_tok) in zip(states, outs):
        s.local_input_tokens = engine.n_tokens(s.context)
        s.full_gen, s.own_tokens, s.done = text, n_tok, True


        s.context += text
        _record_trace_event(s, "policy", text, own_tokens=n_tok, tools_enabled=False)
    return [finalize_record(s, cfg, engine.model) for s in states]


def _api_generation_complete(result: dict) -> bool:
    # Exhausting the fixed token budget is a model outcome, not a transport error.
    return not result.get("unavailable") and (
        bool(result.get("text", "").strip()) or result.get("finish_reason") == "length"
    )


async def _run_deepseek_async(client: LRMClient, items: list[dict], n: int,
                              max_tokens: int, cfg: dict, save_traces: bool = False) -> list[dict]:
    strict = cfg.get("strict_api_errors", False)
    checkpoint_root = cfg.get("_api_checkpoint_root") if strict else None
    checkpoint_dir = None
    if checkpoint_root:
        # Bind cached generations to the exact inputs and evaluation settings.
        identity = {"cfg": {k: v for k, v in cfg.items() if k != "_api_checkpoint_root"},
                    "items": items, "n": n, "max_tokens": max_tokens,
                    "save_traces": save_traces}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        checkpoint_dir = Path(checkpoint_root) / digest
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    reused = 0
    print(f"[API start] samples={len(items) * n} max_tokens={max_tokens} "
          f"thinking={cfg.get('tool', {}).get('thinking', False)} "
          f"checkpoints={checkpoint_dir}", flush=True)

    async def generate(index, item, rep, messages):
        nonlocal reused
        path = checkpoint_dir / f"{index}.json" if checkpoint_dir else None
        if path and path.is_file():
            result = json.loads(path.read_text())
            if _api_generation_complete(result):
                reused += 1
                return index, item, rep, messages, result
        attempts = 3 if strict else 1
        for attempt in range(attempts):
            result = await client.achat(messages, item["id"], max_tokens=max_tokens)
            if not strict or _api_generation_complete(result):
                if result.get("finish_reason") == "length" and not result.get("text", "").strip():
                    print(f"[API truncated] {item['id']} rep={rep}: "
                          "empty final answer; counted incorrect, no retry", flush=True)
                if path:
                    write_json(path, result)
                return index, item, rep, messages, result
            # Transport failures already receive retries inside the client.
            if result.get("unavailable"):
                break
            print(f"[empty API response] {item['id']} rep={rep} "
                  f"attempt={attempt + 1}/{attempts} "
                  f"finish_reason={result.get('finish_reason')} usage={result.get('usage')}",
                  flush=True)
            if attempt + 1 < attempts:
                await asyncio.sleep(2 ** attempt)
        return index, item, rep, messages, result

    jobs = []
    for item in items:
        instruction = "\nPut your final answer within \\boxed{}."
        messages = [{"role": "user", "content": item["question"] + instruction}]
        for rep in range(n):
            jobs.append(asyncio.create_task(generate(len(jobs), item, rep, messages)))

    completed = []
    failures = []
    pending = set(jobs)
    while pending:
        # A reporting timeout must not cancel or retry an in-flight API request.
        done, pending = await asyncio.wait(
            pending, timeout=30, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            index, item, rep, messages, result = await task
            if strict and not _api_generation_complete(result):
                detail = (f"{item['id']} rep={rep}: "
                          f"{result.get('unavailable_reason', 'empty response')} "
                          f"finish_reason={result.get('finish_reason')} "
                          f"usage={result.get('usage')} error={result.get('request_error')}")
                failures.append(detail)
                print(f"[API generation failed] {detail}", flush=True)
            else:
                completed.append((index, item, rep, messages, result))
        print(f"[API progress] done={len(completed) + len(failures)}/{len(jobs)} "
              f"ok={len(completed)} failed={len(failures)} cached={reused} "
              f"pending={len(pending)} elapsed={time.monotonic() - started:.0f}s",
              flush=True)
    if failures:
        raise RuntimeError(
            f"API generation failed for {len(failures)} sample(s); "
            f"successful samples checkpointed={checkpoint_dir is not None}; "
            f"rerun this component. First failure: {failures[0]}"
        )
    records = []
    print(f"[API grading] samples={len(completed)}", flush=True)
    for index, item, rep, messages, result in sorted(completed, key=lambda row: row[0]):
        prompt = messages[0]["content"]
        state = TrajState(item=item, rep=rep, prompt=prompt,
                          context=prompt + result["text"], full_gen=result["text"],
                          record_trace=save_traces)
        _record_trace_event(state, "generation", result["text"])
        state.calls.append({
            "kind": "deepseek",
            # the standalone baseline runs on the deepest tier's backend; record the
            # depth explicitly so its price is not an implicit fallback in cost_api
            "depth": client.default_depth,
            "in_toks": result["usage"].get("prompt_tokens", 0),
            "out_toks": result["usage"].get("completion_tokens", 0),
            "model": result.get("model"),
            "cost_usd": result.get("cost_usd", 0.0),
            "finish_reason": result.get("finish_reason"),
            "reasoning_tokens": result["usage"].get("reasoning_tokens", 0),
            "q_overlap": None,
        })
        record = finalize_record(state, cfg, local_model=None)
        if result.get("finish_reason") == "length" and not result["text"].strip():
            record["correct"] = False
        records.append(record)
    return records


def run_deepseek(client: LRMClient, items: list[dict], budgets: dict, cfg: dict,
                 save_traces: bool = False) -> list[dict]:
    """Evaluate the standalone API model directly."""
    async def run():
        # achat uses to_thread: the default, CPU-dependent pool can otherwise
        # cap API concurrency below the requested value. asyncio.run owns and
        # shuts down this executor along with its event loop.
        concurrency = getattr(client, "concurrency", cfg.get("tool", {}).get("concurrency", 8))
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=concurrency))
        return await _run_deepseek_async(
            client, items, n=budgets.get("n", 16),
            max_tokens=budgets.get("max_tokens", 16384), cfg=cfg,
            save_traces=save_traces,
        )

    return asyncio.run(run())


def finalize_record(state: TrajState, cfg: dict, local_model: str | None) -> dict:
    pricing = cfg["pricing"]
    if cfg.get("_no_cost"):
        costs = {"cost_local": 0.0, "cost_api": 0.0, "cost_usd": 0.0}
    elif local_model and state.own_tokens:
        table = cost_mod.load_throughput(pricing["throughput_json"])
        throughput = cost_mod.throughput_for(table, cfg.get("_throughput_model") or local_model)
    else:
        throughput = {"prefill_tok_s": 1.0, "decode_tok_s": 1.0}


    api_priced_calls = [
        call for call in state.calls if call.get("kind") in ("query", "deepseek")
    ]
    if not cfg.get("_no_cost"):
        costs = cost_mod.trajectory_cost(
            state.local_input_tokens if local_model else 0,
            state.own_tokens if local_model else 0, api_priced_calls, pricing["p_gpu_hr"],
            throughput["prefill_tok_s"], throughput["decode_tok_s"],
            cfg["tool"]["depths"],
        )
    item = state.item
    lrm_calls = sum(
        c["kind"] == "query" and not c.get("rejected") for c in state.calls
    )
    invalid_calls = sum(
        c["kind"] == "query" and bool(c.get("rejected")) for c in state.calls
    )
    search_calls = sum(
        c["kind"] == "search" and not c.get("rejected") for c in state.calls
    )
    invalid_search_calls = sum(
        c["kind"] == "search" and bool(c.get("rejected")) for c in state.calls
    )
    record = {
        "id": item["id"], "rep": state.rep, "dataset": item["dataset"], "domain": item["domain"],


        "correct": bool(grade(item["answer_format"], state.full_gen, item["gold"])),
        "answer": state.full_gen[-400:],
        "own_tokens": state.own_tokens,
        "local_input_tokens": state.local_input_tokens,
        "lrm_in": sum(c.get("in_toks", 0) for c in state.calls),
        "lrm_out": sum(c.get("out_toks", 0) for c in state.calls),
        "lrm_calls": lrm_calls,
        "invalid_calls": invalid_calls,
        "search_calls": search_calls,
        "invalid_search_calls": invalid_search_calls,
        "retrieved_docs": sum(c.get("num_docs", 0) for c in state.calls if c["kind"] == "search"),
        "retrieved_chars": sum(c.get("result_chars", 0) for c in state.calls if c["kind"] == "search"),
        "retrieval_latency_ms": sum(c.get("latency_ms", 0.0) for c in state.calls if c["kind"] == "search"),
        "attempted_calls": lrm_calls + invalid_calls + search_calls + invalid_search_calls,
        "unterminated_query_attempt": bool(
            protocol.has_unterminated_tool_attempt(state.full_gen)
        ),
        "unterminated_search_attempt": bool(
            search_r1.has_unterminated_search_attempt(state.full_gen)
        ),
        "depth_hist": {d: sum(c.get("depth") == d and not c.get("rejected")
                              for c in state.calls) for d in protocol.DEPTHS},
        "first_valid_call_depth": next(
            (c.get("depth") for c in state.calls
             if c.get("kind") == "query" and not c.get("rejected")),
            None,
        ),
        "calls": state.calls,
        **costs,
    }
    if state.record_trace:
        record["problem"] = {
            key: item.get(key)
            for key in ("id", "dataset", "domain", "question", "stem", "gold", "answer_format")
            if key in item
        }
        record["trace"] = {
            "prompt": state.prompt,
            "events": state.trace_events,
            "full_generation": state.full_gen,
            "full_context": state.context,
        }
    return record


def summarize_records(records: list[dict]) -> dict:
    import numpy as np

    if not records:
        return {}
    summary = {
        "n": len(records),
        "accuracy": float(np.mean([r["correct"] for r in records])),
        "cost_usd_mean": float(np.mean([r["cost_usd"] for r in records])),
        "cost_local_mean": float(np.mean([r["cost_local"] for r in records])),
        "cost_api_mean": float(np.mean([r["cost_api"] for r in records])),
        "own_tokens_mean": float(np.mean([r["own_tokens"] for r in records])),
        "call_rate": float(np.mean([
            r.get("lrm_calls", 0) + r.get("search_calls", 0) > 0 for r in records
        ])),
        "lrm_calls_mean": float(np.mean([r["lrm_calls"] for r in records])),
        "search_calls_mean": float(np.mean([r.get("search_calls", 0) for r in records])),
        "search_call_rate": float(np.mean([r.get("search_calls", 0) > 0 for r in records])),
        "retrieved_docs_mean": float(np.mean([r.get("retrieved_docs", 0) for r in records])),
        "retrieved_chars_mean": float(np.mean([r.get("retrieved_chars", 0) for r in records])),
        "retrieval_latency_ms_mean": float(np.mean([
            r.get("retrieval_latency_ms", 0.0) for r in records
        ])),
        "attempted_call_rate": float(np.mean([r.get("attempted_calls", 0) > 0 for r in records])),
        "attempted_calls_mean": float(np.mean([r.get("attempted_calls", 0) for r in records])),
        "invalid_calls_mean": float(np.mean([r.get("invalid_calls", 0) for r in records])),
        "invalid_search_calls_mean": float(np.mean([
            r.get("invalid_search_calls", 0) for r in records
        ])),
        "unterminated_query_rate": float(np.mean([
            bool(r.get("unterminated_query_attempt", False)) for r in records
        ])),
        "unterminated_search_rate": float(np.mean([
            bool(r.get("unterminated_search_attempt", False)) for r in records
        ])),


        **{f"calls_depth{d}_mean": float(np.mean([r["depth_hist"].get(d, 0) for r in records]))
           for d in protocol.DEPTHS if "depth_hist" in records[0]},
        "mean_depth": float(np.mean([m for r in records
                                     for m in ([c["depth"] for c in r["calls"]
                                                if c.get("depth") and not c.get("rejected")])]
                                    or [0.0])),
    }
    if "opening_query_valid" in records[0]:
        summary.update({
            "opening_query_valid_rate": float(np.mean([
                bool(r["opening_query_valid"]) for r in records
            ])),
            "opening_query_tokens_mean": float(np.mean([
                r["opening_query_tokens"] for r in records
            ])),
            "opening_query_chars_mean": float(np.mean([
                r["opening_query_chars"] for r in records
            ])),
            "pre_query_reasoning_tokens_mean": float(np.mean([
                r["pre_query_reasoning_tokens"] for r in records
            ])),
        })
    return summary


def _records_complete(path: Path, items: list[dict], repeats: int) -> bool:
    'Validate an existing suite before allowing --resume to skip it.'
    if not path.is_file():
        return False
    expected_ids = {str(item["id"]) for item in items}
    expected_reps = set(range(repeats))
    reps = {problem_id: set() for problem_id in expected_ids}
    count = 0
    try:
        for record in load_jsonl(path):
            problem_id = str(record["id"])
            rep = int(record["rep"])
            if problem_id not in reps or rep in reps[problem_id]:
                return False
            reps[problem_id].add(rep)
            count += 1
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return (
        count == len(expected_ids) * repeats
        and set(reps) == expected_ids
        and all(values == expected_reps for values in reps.values())
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--mode", required=True, choices=["ours", "vanilla", "search_r1", "deepseek"])
    parser.add_argument(
        "--model",
        help="local model or checkpoint path; optional for deepseek",
    )
    parser.add_argument("--suites", help="comma-separated subset of suites")
    parser.add_argument("--k", type=int, help="override repeats per problem")
    parser.add_argument("--throughput-model", default=None,
                        help="throughput.json model key for local cost accounting")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip suites whose output has the exact expected IDs and repetitions",
    )
    parser.add_argument(
        "--no-cost",
        action="store_true",
        help="skip throughput/API cost accounting",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = _load_eval_config(args.config)
    cfg["_no_cost"] = bool(args.no_cost)
    set_seed(cfg["seed"])
    needs_local = args.mode != "deepseek"
    if needs_local:
        model = args.model or DEFAULT_LOCAL_MODEL
        cfg["_throughput_model"] = args.throughput_model or (
            DEFAULT_LOCAL_MODEL if args.mode in ("ours", "search_r1") else model
        )
    else:
        model = configure_deepseek(cfg, args.model)
        if not os.environ.get("OPENROUTER_API_KEY"):
            parser.error("OPENROUTER_API_KEY is required for deepseek evaluation")
    run_id = args.run_id or new_run_id(_run_id_prefix(args.mode, model))
    out_dir = Path(cfg["output_root"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_key = args.mode
    budgets = dict(cfg["budgets"][budget_key])

    suites = {name: spec for name, spec in cfg["suites"].items()
              if not args.suites or name in args.suites.split(",")}

    expected_summary = {
        "mode": args.mode,
        "model": model,
        "budgets": budgets,
        "save_traces": False,
        "cost_accounting": not args.no_cost,
        "throughput_model": cfg["_throughput_model"],
        "suites": {},
    }
    if args.mode in ("ours", "deepseek"):
        expected_summary["depth_tiers"] = json.loads(json.dumps(cfg["tool"]["depths"]))
    if args.mode == "deepseek":
        expected_summary["thinking"] = True
    summary_path = out_dir / "summary.json"
    summary = expected_summary
    if args.resume and summary_path.is_file():
        try:
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot resume from invalid {summary_path}") from exc
        metadata_keys = (
            "mode",
            "model",
            "budgets",
            "save_traces",
            "cost_accounting",
            "throughput_model",
            "depth_tiers",
            "thinking",
        )
        mismatches = [
            key
            for key in metadata_keys
            if previous_summary.get(key) != expected_summary.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"cannot resume {run_id}: metadata differs for {mismatches}; "
                "use a new --run-id or rerun without --resume"
            )
        summary = {**previous_summary, **expected_summary}
        summary["suites"] = dict(previous_summary.get("suites") or {})
    if args.mode == "ours":
        summary["depth_tiers"] = cfg["tool"]["depths"]
    if args.mode == "search_r1":
        summary["search_r1"] = dict(cfg["search_r1"])
        summary["search_reward"] = "binary_accuracy_only"
        summary["retrieval_dollar_cost_included"] = False




    write_json(summary_path, summary)
    prior_lrm_usd_spent = float(summary.get("lrm_usd_spent", 0.0))

    client = None
    if args.mode in ("ours", "deepseek"):
        client = make_client(cfg)
    search_client = make_search_client(cfg) if args.mode == "search_r1" else None
    if search_client:
        probe = search_client.search("Wikipedia artificial intelligence")
        if probe["num_docs"] <= 0:
            raise RuntimeError("Search-R1 retriever preflight returned no documents")
        summary["retriever_preflight"] = {
            key: probe[key] for key in ("num_docs", "result_chars", "latency_ms")
        }
        write_json(summary_path, summary)

    engine = None
    if needs_local:
        engine = VllmEngine(model, cfg["sampling"], cfg["seed"])

    for name, spec in suites.items():
        items = load_jsonl(spec["path"])
        k = args.k or spec.get("k", 1)
        output_path = out_dir / f"{name}.jsonl"
        if args.resume and _records_complete(output_path, items, k):
            if name not in summary["suites"]:
                summary["suites"][name] = summarize_records(load_jsonl(output_path))
                write_json(summary_path, summary)
            print(f"[skip] {name}: validated complete output ({len(items)} x {k})")
            continue

        if args.mode == "ours":
            records = run_tool_loop(engine, client, items, k, budgets, cfg,
                                    save_traces=False)
        elif args.mode == "search_r1":
            records = run_search_loop(
                engine,
                search_client,
                items,
                k,
                budgets,
                cfg,
                save_traces=False,
            )
        elif args.mode == "vanilla":
            cfg["_throughput_model"] = args.throughput_model or engine.model
            records = run_vanilla(engine, items, k, budgets, cfg, save_traces=False)
        elif args.mode == "deepseek":
            cfg["_api_checkpoint_root"] = str(out_dir / f"{name}.api-checkpoints")
            records = run_deepseek(client, items, {**budgets, "n": k}, cfg)

        write_jsonl(output_path, records)
        summary["suites"][name] = summarize_records(records)
        if client:
            summary["lrm_usd_spent"] = prior_lrm_usd_spent + client.budget.usd_spent
        write_json(summary_path, summary)
        print(f"[{name}] {summary['suites'][name]}")

    if search_client:
        search_client.close()

    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
