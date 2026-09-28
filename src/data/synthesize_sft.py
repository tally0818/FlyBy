"""Single-call counterfactual synthesis with depth search and strict rescue selection."""
from __future__ import annotations
import json
import re
from typing import Any
from ..tools import guard, protocol
from ..eval.grader import grade
from .generation import digest, item_seed


RESERVED = ("<llm_query", "</llm_query>", "<lrm_answer>", "</lrm_answer>",
            "<|im_start|>", "<|im_end|>")


def tool_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": protocol.system_prompt()},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def candidate_states(record: dict, cfg: dict):
    """First failed rollout; uncertainty/doubt before deduplicated alpha fallbacks."""
    failed = [(i, r) for i, r in enumerate(record["rollouts"]) if not r["correct"]]
    seen = set()
    for index, rollout in failed[:cfg["failed_rollouts_per_problem"]]:
        # Intervene inside thinking, never after an already emitted final answer.
        text = rollout["text"].split("</think>", 1)[0]
        if len(text) < 2 or any(tag in text for tag in RESERVED):
            continue
        for alpha in cfg["alphas"]:
            info = _candidate_cut_info(text, alpha, cfg["epistemic_window_chars"])
            if info["cut"] in seen:
                continue
            seen.add(info["cut"])
            yield {"rollout_index": index, "trace": text[:info["cut"]], **info}


def synthesize_state(record, state, cfg, engine, client, seed, grade_fn=grade):
    """Compare four plain/four assisted continuations at exactly the same prefix."""
    tokenizer = engine.tokenizer
    prompt = tool_prompt(tokenizer, record["question"])
    trace = state["trace"]
    key = digest([record["id"], state["rollout_index"], state["cut"]])
    audit = {"id": record["id"], "domain": record["domain"], "state_key": key,
             **{k: v for k, v in state.items() if k != "trace"},
             "plain_successes": None, "tool_successes": None,
             "accepted": False, "strict": False, "n_calls": 0, "attempts": []}
    result = {"key": key, "audit": audit, "prompt": prompt, "trace": trace,
              "plain_good": [], "query": None, "observation": None, "continuation": None}
    q, reason = _synthesize_query(client, record["question"], trace, record["id"])
    if q is None or any(tag in q for tag in RESERVED):
        audit["status"] = reason or "query_reserved_tag"
        return result
    result["query"] = q
    budget = cfg["total_own_tokens"]
    own = token_len(tokenizer, trace)
    allowance = min(budget - own, cfg["decode"]["max_new_tokens"])
    if allowance <= 0:
        audit["status"] = "budget_exhausted"
        return result
    texts = engine.generate(prompt + trace, 4, item_seed(seed, key, "plain"),
                            cfg["decode"], allowance)
    if len(texts) != 4:
        raise ValueError("Plain comparison requires exactly four continuations")
    result["plain_good"] = [t for t in texts if grade_fn(record["answer_format"], t, record["gold"])]
    b0 = len(result["plain_good"])
    audit["plain_successes"] = b0
    for depth in protocol.DEPTHS:
        response = client.query(q, depth, record["id"])
        if response["unavailable"]:
            audit["attempts"].append({"depth": depth, "status": "unavailable"})
            continue
        text, structural = guard.filter_leakage(response["text"])
        text, gold = guard.redact_gold(text, str(record["gold"]), record["answer_format"])
        attempt = {"depth": depth, "redactions": structural + gold,
                   "cost_usd": response.get("cost_usd", 0), "model": response.get("model")}
        audit["attempts"].append(attempt)
        if not text.strip() or any(tag in text for tag in RESERVED):
            attempt["status"] = "invalid_observation"
            continue
        action = protocol.format_tool_call(q, depth)
        observation = protocol.format_lrm_answer(text)
        # Apply cumulative own-token and live-buffer caps.
        allowance = min(budget - token_len(tokenizer, trace + action),
                        budget - token_len(tokenizer, trace + action + observation),
                        cfg["decode"]["max_new_tokens"])
        if allowance <= 0:
            attempt["status"] = "budget_exhausted"
            break
        texts = engine.generate(prompt + trace + action + observation, 4,
                                item_seed(seed, key, depth), cfg["decode"], allowance)
        if len(texts) != 4:
            raise ValueError("Assisted comparison requires exactly four continuations")
        good = [t for t in texts if grade_fn(record["answer_format"], t, record["gold"])]
        bd = len(good)
        collection = bd >= 2 and bd - b0 >= 1
        attempt.update(tool_successes=bd, collection_accepted=collection,
                       status="accepted" if collection else "rejected")
        if collection:
            # Do not escalate a 2/4 rescue to seek a strict 3/4 rescue.
            audit.update(accepted=True, strict=b0 == 0 and bd >= 3,
                         plain_recovery=b0 / 4, tool_recovery=bd / 4,
                         tool_successes=bd, depth=depth, n_calls=1,
                         status="strict_rescue" if b0 == 0 and bd >= 3 else "collection_only")
            result.update(action=action, observation=observation, continuation=good[0])
            return result
    audit["status"] = "no_collection_rescue"
    return result


UNCERTAINTY_NGRAMS = [
    "i'm not sure", "i am not sure", "i don't remember", "i do not remember",
    "i don't know", "i do not know", "i can't recall", "cannot recall",
    "not certain", "i'm unsure", "unsure whether", "i forget",
]

EPISTEMIC_TOKENS = {
    "verify": {"check", "seems", "actually"},
    "doubt": {"maybe", "might", "perhaps"},
    "pivot": {"wait", "hmm", "alternatively"},
}
_EPISTEMIC_CLASS = {
    token: kind for kind, tokens in EPISTEMIC_TOKENS.items() for token in tokens
}
_EPISTEMIC_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in _EPISTEMIC_CLASS) + r")\b",
    flags=re.IGNORECASE,
)



def epistemic_profile(text: str) -> dict[str, Any]:
    """Count the reference reflection vocabulary and classify the local state."""
    counts = {kind: 0 for kind in EPISTEMIC_TOKENS}
    triggers: list[str] = []
    for match in _EPISTEMIC_PATTERN.finditer(text):
        token = match.group(1).lower()
        counts[_EPISTEMIC_CLASS[token]] += 1
        triggers.append(token)

    verify, doubt, pivot = counts["verify"], counts["doubt"], counts["pivot"]
    if doubt > verify:
        signal = "doubt"
    elif verify > doubt:
        signal = "verify"
    elif doubt or verify:
        signal = "balanced"
    elif pivot:
        signal = "pivot"
    else:
        signal = "none"
    return {
        "epistemic_signal": signal,
        "epistemic_trigger": triggers[0] if triggers else None,
        "epistemic_triggers": triggers,
        "verify_count": verify,
        "doubt_count": doubt,
        "pivot_count": pivot,
        "verify_doubt_ratio": verify / doubt if doubt else None,
    }


def _sentence_spans(text: str):
    """Yield non-empty sentence-like spans with character offsets."""
    for match in re.finditer(r"[^.!?\n]*(?:[.!?]+|\n+|$)", text):
        sentence = match.group(0)
        if sentence.strip():
            yield match.start(), match.end(), sentence


def cut_at_epistemic_doubt(text: str) -> tuple[int, str] | None:
    """Return the first sentence whose local reflection signal is doubt-dominated."""
    for _, end, sentence in _sentence_spans(text):
        if epistemic_profile(sentence)["epistemic_signal"] == "doubt":
            return end, sentence.strip()
    return None


def cut_at_uncertainty(text: str) -> tuple[int, str] | None:
    low = text.lower()
    hits = [(low.find(g), g) for g in UNCERTAINTY_NGRAMS if g in low]
    if not hits:
        return None
    pos, _ = min(hits)
    end = text.find(".", pos)
    end = end + 1 if end != -1 else min(len(text), pos + 200)
    sent_start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos)) + 1
    return end, text[sent_start:end].strip()



def _candidate_cut_info(
    text: str,
    alpha: float,
    profile_window_chars: int = 512,
) -> dict[str, Any]:
    """Where in `text` to interrupt and ask. Uncertainty first, alpha as fallback."""
    hit = cut_at_uncertainty(text)
    source, sentence = "explicit_uncertainty", ""
    if hit:
        cut, sentence = hit
    elif hit := cut_at_epistemic_doubt(text):
        cut, sentence = hit
        source = "epistemic_doubt"
    else:
        cut, source = int(len(text) * alpha), "alpha"
    return _cut_info(text, cut, sentence, source, profile_window_chars)



def _cut_info(
    text: str,
    cut: int,
    sentence: str,
    source: str,
    profile_window_chars: int,
) -> dict[str, Any]:
    cut = max(1, min(cut, len(text) - 1))
    # Profile only information available at the proposed state; never look past
    # the cut when assigning metadata or matched-negative strata.
    local_window = text[max(0, cut - max(1, profile_window_chars)):cut]
    profile = epistemic_profile(local_window)
    if sentence:
        # Attribute the trigger to the sentence that proposed the cut, while
        # retaining counts/ratio over the configured local prefix window.
        profile["epistemic_trigger"] = epistemic_profile(sentence)["epistemic_trigger"]
    return {
        "cut": cut,
        "sentence": sentence,
        "candidate_source": source,
        **profile,
    }



def _synthesize_query(
    client: Any, question: str, trace: str, problem_id: str
) -> tuple[str | None, str | None]:
    """Ask the teacher for an information-seeking subquestion, never an answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "Given a problem and a partial failed attempt, identify one missing concept, theorem, "
                "formula, or factual relation that would help a weaker model continue. Return JSON only "
                "as {\"q\": \"...\"}. The question must be self-contained, under 300 characters, and must "
                "not copy the problem, mention answer choices, request the final answer, or solve the problem."
            ),
        },
        {"role": "user", "content": f"Problem:\n{question}\n\nPartial attempt:\n{trace[-6000:]}"},
    ]
    result = client.chat(messages, problem_id, max_tokens=256)
    if result["unavailable"]:
        reason = result.get("unavailable_reason", "query_synthesis_unavailable")
        return None, f"unavailable: {reason}"
    raw = result["text"].strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        q = json.loads(match.group(0) if match else raw).get("q", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return None, "query_synthesis_malformed_json"
    valid, reason = guard.validate_query(q)
    if not valid:
        return None, f"query_invalid: {reason}"
    reject, _ = guard.check_query_overlap(q, question)
    if reject:
        return None, "query_overlaps_problem"
    return q, None
