"""Five equally sized SFT objectives from unique strict single-call rescues."""
from __future__ import annotations

import json
import random
from collections import Counter

from .build_sft import (
    MAX_SEQ_LEN, NO_CALL_MAX_SEQ_LEN, SOURCE_DATASET, PROTOCOL_NAME,
    OBJECTIVE_STRICT_RESCUE, OBJECTIVE_ORIGINAL_NO_CALL, OBJECTIVE_REPLAY,
    OBJECTIVE_QUERY, OBJECTIVE_INTEGRATION,
)
from .generation import digest
from .synthesize_sft import RESERVED, token_len, tool_prompt

OBJECTIVES = (OBJECTIVE_STRICT_RESCUE, OBJECTIVE_ORIGINAL_NO_CALL, OBJECTIVE_REPLAY,
              OBJECTIVE_QUERY, OBJECTIVE_INTEGRATION)


def segment(text, train=False, role="full_trajectory"):
    return {"text": text, "train": train,
            **({"loss_weight": 1.0, "role": role} if train else {})}


def row(row_id, kind, parts, objective, **meta):
    return {"id": row_id, "kind": kind,
            "segments_json": json.dumps(parts, ensure_ascii=False),
            "meta_json": json.dumps({
                **meta, "source_dataset": SOURCE_DATASET, "source_protocol": PROTOCOL_NAME,
                "source_objective": objective, "sft_only": True, "rl_eligible": False,
            }, ensure_ascii=False)}


def sequence_length(tokenizer, parts):
    return sum(token_len(tokenizer, part["text"]) for part in parts)


def token_prefix(tokenizer, text, limit=128):
    """Select a prefix by token offsets while preserving the original characters."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        raise ValueError("Empty integration continuation")
    if len(offsets) <= limit:
        return text
    for index in range(limit - 1, -1, -1):
        prefix = text[:int(offsets[index][1])]
        if prefix and token_len(tokenizer, prefix) <= limit:
            return prefix
    raise ValueError("Cannot construct an integration prefix within the token limit")


def tool_rows(result, tokenizer):
    audit = result["audit"]
    prompt, trace = result["prompt"], result["trace"]
    action, observation, continuation = (result[k] for k in ("action", "observation", "continuation"))
    target = token_prefix(tokenizer, continuation)
    definitions = [
        ("query_chain", OBJECTIVE_STRICT_RESCUE,
         [segment(prompt), segment(trace + action, True), segment(observation),
          segment(continuation, True)]),
        ("query_booster", OBJECTIVE_QUERY,
         [segment(prompt + trace), segment(action, True, "query_booster")]),
        ("integration_booster", OBJECTIVE_INTEGRATION,
         [segment(prompt + trace), segment(action), segment(observation),
          segment(target, True, "integration_booster")]),
    ]
    return [row(f"{result['key']}:{kind}", kind, parts, objective,
                **audit, problem_id=audit["id"], source_id=result["key"])
            for kind, objective, parts in definitions]


def select_strict(results, tokenizer):
    chosen = {}
    for result in results:
        audit = result["audit"]
        if (not audit["strict"] or audit["plain_successes"] != 0
                or audit["tool_successes"] < 3 or audit["n_calls"] != 1):
            continue
        continuation = result["continuation"]
        if not continuation or any(tag in result["trace"] + continuation for tag in RESERVED):
            continue
        rows = tool_rows(result, tokenizer)
        if any(sequence_length(tokenizer, json.loads(r["segments_json"])) > MAX_SEQ_LEN for r in rows):
            continue
        chosen.setdefault(str(audit["id"]), result)
    return list(chosen.values())


def select_no_call(records, results, tokenizer, target, weights, seed):
    candidates = []
    for record in records:
        prompt = tool_prompt(tokenizer, record["question"])
        for index, rollout in enumerate(record["rollouts"]):
            if rollout["correct"]:
                candidates.append({"id": f"{record['id']}:correct:{index}",
                                   "problem_id": record["id"], "domain": record["domain"],
                                   "source": "correct_rollout", "prompt": prompt, "text": rollout["text"]})
    for result in results:
        if result["audit"]["accepted"]:
            continue
        for index, text in enumerate(result["plain_good"][:1]):
            audit = result["audit"]
            candidates.append({"id": f"{result['key']}:plain:{index}", "problem_id": audit["id"],
                               "domain": audit["domain"], "prompt": result["prompt"],
                               "source": "epistemic_matched" if audit["epistemic_signal"] != "none"
                               else "counterfactual", "text": result["trace"] + text})
    unique = {}
    for candidate in sorted(candidates, key=lambda c: (
            c["source"] != "epistemic_matched", digest([seed, c["id"]]))):
        text = candidate["text"]
        if (not text or any(tag in text for tag in RESERVED)
                or token_len(tokenizer, candidate["prompt"]) + token_len(tokenizer, text) > NO_CALL_MAX_SEQ_LEN):
            continue
        unique.setdefault(candidate["problem_id"], candidate)
    available = Counter(c["domain"] for c in unique.values())
    if sum(available.values()) < target:
        raise ValueError(f"Need {target} unique verified no-tool controls; found {dict(available)}")
    quotas = control_quotas(target, weights, available)
    selected = []
    for domain, n in quotas.items():
        selected.extend([c for c in unique.values() if c["domain"] == domain][:n])
    return selected, quotas


def control_quotas(target, weights, available):
    """Largest-remainder allocation capped by unique available problems."""
    if sum(available.get(d, 0) for d in weights) < target:
        raise ValueError("Insufficient unique controls for five equal groups")
    quotas = {domain: 0 for domain in weights}
    total_weight = sum(weights.values())
    for _ in range(target):
        domain = min((d for d in weights if quotas[d] < available.get(d, 0)),
                     key=lambda d: (quotas[d] - target * weights[d] / total_weight, d))
        quotas[domain] += 1
    return quotas


def replay_candidate(raw, tokenizer):
    """Filter replay examples by structure and length; answers are not gold-verified."""
    conversations = raw.get("conversations")
    if (raw.get("domain") not in ("math", "science") or not isinstance(conversations, list)
            or len(conversations) != 2 or not all(isinstance(x, dict) for x in conversations)):
        return None
    human, assistant = conversations
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        return None
    question, answer = human.get("value"), assistant.get("value")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    question, answer = question.strip(), answer.strip()
    if (not question or not answer.startswith("<think>") or answer.count("<think>") != 1
            or answer.count("</think>") != 1 or "\\boxed" not in answer.split("</think>")[1]
            or any(tag in question + answer for tag in RESERVED)):
        return None
    # The upstream answer contains its own <think> opening. Do not duplicate the
    # Qwen generation prompt's opening thought tag.
    prompt = tool_prompt(tokenizer, question)
    if prompt.endswith("<think>\n"):
        prompt = prompt[:-len("<think>\n")]
    if (token_len(tokenizer, answer) < 256
            or token_len(tokenizer, prompt) + token_len(tokenizer, answer) > NO_CALL_MAX_SEQ_LEN):
        return None
    return {"question_hash": digest(" ".join(question.split())), "prompt": prompt,
            "text": answer, "domain": raw["domain"], "upstream_source": raw.get("source")}


def fetch_replay(tokenizer, quotas, cfg, cache, seed):
    """Read the pinned HF release, cache only selected valid replay examples."""
    key = {"dataset": cfg, "quotas": quotas, "seed": seed}

    def produce():
        from datasets import load_dataset

        data = load_dataset(cfg["hf"], revision=cfg["revision"], split="train", streaming=True)
        pools = {domain: {} for domain in quotas}
        for index, raw in enumerate(data):
            domain = raw.get("domain")
            if domain not in pools or len(pools[domain]) >= quotas[domain] * 2:
                continue
            entry = replay_candidate(raw, tokenizer)
            if entry is not None:
                entry["row_index"] = index
                pools[domain].setdefault(entry["question_hash"], entry)
            if all(len(pools[d]) >= 2 * n for d, n in quotas.items()):
                break
            if index % 1000 == 0:
                print(f"[OpenThoughts] scanned={index + 1}, valid="
                      f"{ {d: len(p) for d, p in pools.items()} }", flush=True)
        selected = []
        for domain, n in quotas.items():
            pool = sorted(pools[domain].values(), key=lambda r: digest([seed, r["question_hash"]]))
            if len(pool) < n:
                raise ValueError(f"OpenThoughts {domain}: need {n}, found {len(pool)}")
            selected.extend(pool[:n])
        return selected

    return cache.get("replay", key, produce)


def assemble(strict, controls, replay, tokenizer, replay_cfg, seed):
    n = len(strict)
    if not n or len(controls) != n or len(replay) != n:
        raise ValueError("SFT needs N>0 unique rescues, N no-tool controls and N replay examples")
    rows = [r for result in strict for r in tool_rows(result, tokenizer)]
    for candidate in controls:
        rows.append(row(candidate["id"], "continue",
                        [segment(candidate["prompt"]), segment(candidate["text"], True)],
                        OBJECTIVE_ORIGINAL_NO_CALL, problem_id=candidate["problem_id"],
                        domain=candidate["domain"], continue_source=candidate["source"],
                        final_answer_verified=True))
    for entry in replay:
        rows.append(row(f"openthoughts:{entry['question_hash']}", "continue",
                        [segment(entry["prompt"]), segment(entry["text"], True)], OBJECTIVE_REPLAY,
                        domain=entry["domain"], openthoughts_dataset=replay_cfg["hf"],
                        openthoughts_revision=replay_cfg["revision"],
                        openthoughts_row_index=entry["row_index"], final_answer_verified=False))
    random.Random(seed).shuffle(rows)
    return rows
