'Validate and finalize five equally sized SFT groups.'
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from src.tools.protocol import DEPTHS, format_tool_call, system_prompt

DATASET_NAME = "sft"
SOURCE_DATASET = "sft_source_core"
PROTOCOL_NAME = "llm_query_depth_attribute_raw_body_v1"
ID_SUFFIX = ":sft"
MAX_SEQ_LEN = 24576
NO_CALL_MAX_SEQ_LEN = 16384
OBJECTIVE_ORIGINAL_NO_CALL = "correct_no_call_full_trajectory"
OBJECTIVE_REPLAY = "openthoughts_reasoning_replay"
OBJECTIVE_STRICT_RESCUE = "strict_rescue_full_trajectory"
OBJECTIVE_QUERY = "unit_query_booster"
OBJECTIVE_INTEGRATION = "post_observation_integration_booster"
CONTROLLED_TRAIN_EPOCHS = 2

CHAT_SYSTEM_OPEN = "<|im_start|>system\n"
CHAT_TURN_CLOSE = "<|im_end|>"
SOURCE_QUERY_OPEN = "<llm_query>"
QUERY_CLOSE = "</llm_query>"
SOURCE_CALL_RE = re.compile(r"<llm_query>([\s\S]*?)</llm_query>")
NEW_CALL_RE = re.compile(
    r'<llm_query depth="([1-3])">([\s\S]*?)</llm_query>'
)


def _segments(raw: str, row_id: str) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{row_id}: segments_json must be a non-empty list")
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            raise ValueError(f"{row_id}: segment {index} must be an object")
        if not isinstance(segment.get("text"), str) or not isinstance(
            segment.get("train"), bool
        ):
            raise ValueError(f"{row_id}: malformed segment {index}")
    return value


def _meta(raw: str, row_id: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{row_id}: meta_json must be an object")
    return value


def _write_pair(frame: pd.DataFrame, parquet_path: Path, jsonl_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    frame.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)


def source_system_prompt() -> str:
    return (
        "You are a careful reasoning assistant. Solve the problem step by step.\n"
        "While thinking, you may consult a stronger assistant model at any point:\n"
        '<llm_query>{"q": "your short question", "depth": 2}</llm_query>\n'
        "   Ask a SHORT question (under 300 characters) about a concept, theorem, formula\n"
        "   or fact you are unsure of. Do NOT paste the problem; the assistant cannot see\n"
        "   it and will not solve it for you. The reply appears as <lrm_answer>...</lrm_answer>\n"
        "   and then you continue reasoning.\n"
        "   `depth` picks how much oracle you are buying, and both fields are required:\n"
        "     depth 1: a quick factual check — a name, a constant, a definition you nearly have (cheapest)\n"
        "     depth 2: a concept, theorem or formula you half-remember and need stated properly (~10x depth 1)\n"
        "     depth 3: a knowledge gap you cannot bridge alone and that decides the problem (~50x depth 1)\n"
        "Calling costs you, and a deeper call costs much more. Ask only when the answer\n"
        "would actually change what you do next, and at the shallowest depth that would\n"
        "resolve it.\n"
        "Finish with your final answer within \\boxed{}."
    )


def sft_system_prompt() -> str:
    'Return the SFT system prompt.'
    return system_prompt()


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported dataframe format: {path}")


def _decode_source_call(inner: str, row_id: str) -> tuple[str, int]:
    try:
        payload = json.loads(inner)
    except json.JSONDecodeError as error:
        raise ValueError(f"{row_id}: malformed source tool JSON: {error.msg}") from error
    if not isinstance(payload, dict) or set(payload) != {"q", "depth"}:
        raise ValueError(f"{row_id}: source tool payload must contain only q and depth")

    question = payload["q"]
    depth = payload["depth"]
    if not isinstance(question, str) or not question or question != question.strip():
        raise ValueError(f"{row_id}: source tool q must be a non-empty trimmed string")
    if QUERY_CLOSE in question:
        raise ValueError(f"{row_id}: raw query contains the reserved closing tag")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in DEPTHS:
        raise ValueError(f"{row_id}: source tool depth must be an integer in {list(DEPTHS)}")
    return question, depth


def convert_action_spans(text: str, row_id: str) -> tuple[str, list[tuple[str, int]]]:
    'Decode and replace every source action span in one non-prompt segment.'
    if SOURCE_QUERY_OPEN not in text:
        calls = [(q, int(depth)) for depth, q in NEW_CALL_RE.findall(text)]
        if text.count("<llm_query") != len(calls) or text.count(QUERY_CLOSE) != len(calls):
            raise ValueError(f"{row_id}: malformed raw query action")
        return text, calls
    if text.count(QUERY_CLOSE) > text.count(SOURCE_QUERY_OPEN):
        raise ValueError(f"{row_id}: source query contains a reserved closing tag")
    calls: list[tuple[str, int]] = []

    def replace(match: re.Match[str]) -> str:
        question, depth = _decode_source_call(match.group(1), row_id)
        calls.append((question, depth))
        return format_tool_call(question, depth)

    converted = SOURCE_CALL_RE.sub(replace, text)
    if SOURCE_QUERY_OPEN in converted:
        raise ValueError(f"{row_id}: unparsed source llm_query tag remains")
    return converted, calls


def replace_system_prompt(text: str, row_id: str) -> str:
    'Replace the complete embedded source system message with protocol final.'
    if not text.startswith(CHAT_SYSTEM_OPEN):
        raise ValueError(f"{row_id}: first segment does not start with a system turn")
    body_start = len(CHAT_SYSTEM_OPEN)
    body_end = text.find(CHAT_TURN_CLOSE, body_start)
    if body_end < 0:
        raise ValueError(f"{row_id}: first segment has no system turn close")
    embedded = text[body_start:body_end]
    if embedded == sft_system_prompt():
        return text
    if embedded != source_system_prompt():
        raise ValueError(f"{row_id}: embedded system prompt is not the source prompt")
    replacement = sft_system_prompt()
    if replacement == embedded or SOURCE_QUERY_OPEN in replacement:
        raise RuntimeError("frozen final prompt does not use the raw-query protocol")
    return text[:body_start] + replacement + text[body_end:]


def convert_row(row: pd.Series | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    'Convert one source row to the training protocol.'
    source_id = str(row["id"])
    segments = _segments(str(row["segments_json"]), source_id)
    source_meta = _meta(str(row["meta_json"]), source_id)
    source_dataset = source_meta.get("source_dataset")
    if source_dataset != SOURCE_DATASET:
        raise ValueError(f"{source_id}: row is not from the released SFT source")
    if not segments:
        raise ValueError(f"{source_id}: row has no segments")

    output_segments: list[dict[str, Any]] = []
    action_calls: list[tuple[str, int]] = []
    supervised_actions = 0
    for index, segment in enumerate(segments):
        output_segment = dict(segment)
        text = str(segment["text"])
        if index == 0:
            text = replace_system_prompt(text, source_id)
            calls: list[tuple[str, int]] = []
        else:
            text, calls = convert_action_spans(text, source_id)
        output_segment["text"] = text
        output_segments.append(output_segment)
        action_calls.extend(calls)
        if bool(segment["train"]):
            supervised_actions += len(calls)



    expected_actions = 0 if str(row["kind"]) == "continue" else 1
    if len(action_calls) != expected_actions:
        raise ValueError(
            f"{source_id}: expected {expected_actions} trajectory actions, "
            f"found {len(action_calls)}"
        )

    output_meta = {
        key: value
        for key, value in source_meta.items()
        if not re.match(r"^v(?:0[0-9]?|[2-5])_", key)
    }
    output_meta.update({
        "sft_dataset": DATASET_NAME,
        "sft_objective": source_meta.get("sft_objective")
        or source_meta.get("source_objective"),
        "sft_source_dataset": SOURCE_DATASET,
        "sft_source_id": source_id,
        "sft_protocol": PROTOCOL_NAME,
        "sft_only": source_meta.get(
            "sft_only", source_meta.get("source_sft_only")
        ),
        "rl_eligible": source_meta.get(
            "rl_eligible",
            source_meta.get("source_rl_eligible"),
        ),
    })
    output = {
        "id": source_id + ID_SUFFIX,
        "kind": str(row["kind"]),
        "segments_json": json.dumps(output_segments, ensure_ascii=False),
        "meta_json": json.dumps(output_meta, ensure_ascii=False),
    }
    return output, {
        "trajectory_actions": len(action_calls),
        "supervised_actions": supervised_actions,
    }


def _sequence_tokens(tokenizer: Any, row: pd.Series | dict[str, Any]) -> int:
    return sum(
        len(tokenizer.encode(str(segment["text"]), add_special_tokens=False))
        for segment in _segments(str(row["segments_json"]), str(row["id"]))
    )


def _assert_preserved(
    source: pd.Series | dict[str, Any], converted: pd.Series | dict[str, Any]
) -> None:
    'Assert that only prompt/action text and final provenance were added.'
    source_id = str(source["id"])
    if str(converted["id"]) != source_id + ID_SUFFIX:
        raise AssertionError(f"{source_id}: unexpected converted id")
    if str(converted["kind"]) != str(source["kind"]):
        raise AssertionError(f"{source_id}: kind changed during conversion")

    before = _segments(str(source["segments_json"]), source_id)
    after = _segments(str(converted["segments_json"]), str(converted["id"]))
    if len(before) != len(after):
        raise AssertionError(f"{source_id}: segment count changed during conversion")
    for index, (old_segment, new_segment) in enumerate(zip(before, after)):
        old_non_text = {key: value for key, value in old_segment.items() if key != "text"}
        new_non_text = {key: value for key, value in new_segment.items() if key != "text"}
        if old_non_text != new_non_text:
            raise AssertionError(f"{source_id}: segment {index} mask/metadata changed")
        if index == 0:
            expected_text = replace_system_prompt(str(old_segment["text"]), source_id)
        else:
            expected_text, _ = convert_action_spans(str(old_segment["text"]), source_id)
        if str(new_segment["text"]) != expected_text:
            raise AssertionError(f"{source_id}: segment {index} changed beyond protocol conversion")
        if (
            index > 0
            and "<lrm_answer>" in str(old_segment["text"])
            and old_segment != new_segment
        ):
            raise AssertionError(f"{source_id}: external observation changed")



def validate_output(
    tokenizer: Any, source: pd.DataFrame, output: pd.DataFrame
) -> dict[str, Any]:
    source_objectives = Counter(
        _meta(str(row["meta_json"]), str(row["id"])).get("source_objective")
        or _meta(str(row["meta_json"]), str(row["id"])).get("sft_objective")
        for _, row in source.iterrows()
    )
    n = source_objectives[OBJECTIVE_STRICT_RESCUE]
    expected_rows = 5 * n
    if not n or len(source) != expected_rows or source["id"].nunique() != expected_rows:
        raise AssertionError(f"expected 5N unique source rows with N>0, got {len(source)}, N={n}")
    if len(output) != expected_rows or output["id"].nunique() != expected_rows:
        raise AssertionError(f"expected {expected_rows} unique final rows, got {len(output)}")

    objectives: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    actual_actions = supervised_actions = masked_actions = 0
    observations = masked_observations = 0
    max_sequence_tokens = rows_over_max = 0
    max_no_call_sequence_tokens = no_call_rows_over_max = 0
    paired_sources = {objective: set() for objective in (
        OBJECTIVE_STRICT_RESCUE, OBJECTIVE_QUERY, OBJECTIVE_INTEGRATION)}
    strict_problems = set()
    prompt_text = sft_system_prompt()
    prompt_calls = NEW_CALL_RE.findall(prompt_text)
    if len(prompt_calls) != 1:
        raise AssertionError("frozen final prompt must contain exactly one query example")

    for (_, old_row), (_, new_row) in zip(source.iterrows(), output.iterrows()):
        _assert_preserved(old_row, new_row)
        meta = _meta(str(new_row["meta_json"]), str(new_row["id"]))
        segments = _segments(str(new_row["segments_json"]), str(new_row["id"]))
        objectives[str(meta.get("sft_objective"))] += 1
        kinds[str(new_row["kind"])] += 1
        objective = meta.get("sft_objective")
        # Validate counterfactual evidence when present.
        if meta.get("source_protocol") == PROTOCOL_NAME:
            expected_kind, expected_masks = {
                OBJECTIVE_STRICT_RESCUE: ("query_chain", [False, True, False, True]),
                OBJECTIVE_ORIGINAL_NO_CALL: ("continue", [False, True]),
                OBJECTIVE_REPLAY: ("continue", [False, True]),
                OBJECTIVE_QUERY: ("query_booster", [False, True]),
                OBJECTIVE_INTEGRATION: ("integration_booster", [False, False, False, True]),
            }[objective]
            if (new_row["kind"] != expected_kind
                    or [s["train"] for s in segments] != expected_masks):
                raise AssertionError(f"{new_row['id']}: wrong objective masks or kind")
            if segments[0]["train"]:
                raise AssertionError(f"{new_row['id']}: prompt must be masked")
            trained = [s for s in segments if s["train"]]
            if not trained or any(s.get("loss_weight") != 1.0 for s in trained):
                raise AssertionError(f"{new_row['id']}: expected unit-weight supervision")
            if objective in paired_sources:
                if (meta.get("plain_successes") != 0 or meta.get("tool_successes", 0) < 3
                        or meta.get("n_calls") != 1):
                    raise AssertionError(f"{new_row['id']}: not a strict single-call rescue")
                paired_sources[objective].add(meta["source_id"])
                if objective == OBJECTIVE_STRICT_RESCUE:
                    strict_problems.add(meta["problem_id"])
            if objective == OBJECTIVE_INTEGRATION:
                if len(trained) != 1 or len(tokenizer.encode(
                        trained[0]["text"], add_special_tokens=False)) > 128:
                    raise AssertionError(f"{new_row['id']}: invalid integration supervision")
            if objective == OBJECTIVE_QUERY and not NEW_CALL_RE.fullmatch(trained[0]["text"]):
                raise AssertionError(f"{new_row['id']}: query booster must supervise only the action")
            if objective == OBJECTIVE_ORIGINAL_NO_CALL and meta.get("final_answer_verified") is not True:
                raise AssertionError(f"{new_row['id']}: unverified no-tool trajectory")
            if meta.get("rl_eligible") is not False:
                raise AssertionError(f"{new_row['id']}: actual SFT training problems must be excluded from RL")

        system_start = len(CHAT_SYSTEM_OPEN)
        system_end = str(segments[0]["text"]).find(CHAT_TURN_CLOSE, system_start)
        if str(segments[0]["text"])[system_start:system_end] != prompt_text:
            raise AssertionError(f"{new_row['id']}: final system prompt mismatch")

        for index, segment in enumerate(segments):
            text = str(segment["text"])
            if SOURCE_QUERY_OPEN in text:
                raise AssertionError(f"{new_row['id']}: source action remains")
            calls = NEW_CALL_RE.findall(text)

            if index == 0:
                if calls != prompt_calls:
                    raise AssertionError(f"{new_row['id']}: malformed prompt example")
                calls = []
            actual_actions += len(calls)
            if segment["train"]:
                supervised_actions += len(calls)
            else:
                masked_actions += len(calls)
            if index > 0 and "<lrm_answer>" in text:
                observations += 1
                masked_observations += not bool(segment["train"])

        sequence_tokens = _sequence_tokens(tokenizer, new_row)
        max_sequence_tokens = max(max_sequence_tokens, sequence_tokens)
        rows_over_max += sequence_tokens > MAX_SEQ_LEN
        if str(new_row["kind"]) == "continue":
            max_no_call_sequence_tokens = max(
                max_no_call_sequence_tokens, sequence_tokens
            )
            no_call_rows_over_max += sequence_tokens > NO_CALL_MAX_SEQ_LEN

    expected_objectives = Counter({
        OBJECTIVE_STRICT_RESCUE: n,
        OBJECTIVE_ORIGINAL_NO_CALL: n,
        OBJECTIVE_REPLAY: n,
        OBJECTIVE_QUERY: n,
        OBJECTIVE_INTEGRATION: n,
    })
    if objectives != expected_objectives:
        raise AssertionError(f"unexpected final objectives: {objectives}")
    if any(paired_sources.values()):
        if (len(strict_problems) != n or any(len(s) != n for s in paired_sources.values())
                or paired_sources[OBJECTIVE_STRICT_RESCUE] != paired_sources[OBJECTIVE_QUERY]
                or paired_sources[OBJECTIVE_STRICT_RESCUE] != paired_sources[OBJECTIVE_INTEGRATION]):
            raise AssertionError("Full/query/integration must cover the same N unique rescues")
    if kinds != Counter(source["kind"]):
        raise AssertionError(f"final kind counts changed: {kinds}")
    expected_actions = 3 * n
    if actual_actions != expected_actions:
        raise AssertionError(f"expected {expected_actions} actions, got {actual_actions}")
    if (
        supervised_actions != 2 * n
        or masked_actions != n
    ):
        raise AssertionError(
            f"unexpected action masks: supervised={supervised_actions}, "
            f"masked={masked_actions}"
        )
    expected_observations = 2 * n
    if observations != expected_observations or masked_observations != observations:
        raise AssertionError(
            f"unexpected observation masks: total={observations}, "
            f"masked={masked_observations}"
        )
    if rows_over_max:
        raise AssertionError(f"{rows_over_max} final rows exceed {MAX_SEQ_LEN} tokens")
    if no_call_rows_over_max:
        raise AssertionError(
            f"{no_call_rows_over_max} final no-call rows exceed "
            f"{NO_CALL_MAX_SEQ_LEN} tokens"
        )
    return {
        "rows": len(output),
        "objectives": dict(objectives),
        "kind_counts": dict(kinds),
        "system_prompts_validated": len(output),
        "trajectory_actions_validated": actual_actions,
        "supervised_actions_converted": supervised_actions,
        "masked_actions_converted": masked_actions,
        "observations_preserved": observations,
        "masked_observations": masked_observations,
        "max_sequence_tokens": max_sequence_tokens,
        "max_seq_len": MAX_SEQ_LEN,
        "max_no_call_sequence_tokens": max_no_call_sequence_tokens,
        "no_call_max_seq_len": NO_CALL_MAX_SEQ_LEN,
        "protocol": PROTOCOL_NAME,
    }


def build_dataframe(
    source: pd.DataFrame, tokenizer: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"id", "kind", "segments_json", "meta_json"}
    if set(source.columns) != required:
        raise ValueError("source frame must have id/kind/segments_json/meta_json")

    rows = [convert_row(row)[0] for _, row in source.iterrows()]
    output = pd.DataFrame(rows, columns=["id", "kind", "segments_json", "meta_json"])
    return output, validate_output(tokenizer, source, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input", default="data/source/sft.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--tokenizer-revision",
        default="1cfa9a7208912126459214e8b04321603b3df60c",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    source = _read_frame(Path(args.source_input))
    output, report = build_dataframe(source, tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "sft.parquet"
    jsonl_path = output_dir / "sft.jsonl"
    _write_pair(output, parquet_path, jsonl_path)
    print(json.dumps({
        "source_input": args.source_input,
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
        "cpu_only": True,
        "offline": False,
        "generation_or_pass_at_k_used": False,
        "row_order_preserved": True,
        "rows_per_epoch": len(output),
        "controlled_training_epochs": CONTROLLED_TRAIN_EPOCHS,
        "controlled_training_row_exposures": len(output) * CONTROLLED_TRAIN_EPOCHS,
        **report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
