'Tag protocol — the single source for tags, parsing, stop strings and the LRM contract.'
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional





QUERY_OPEN_PREFIX = "<llm_query"
QUERY_OPEN = QUERY_OPEN_PREFIX
QUERY_CLOSE = "</llm_query>"

_QUERY_OPEN_RE = re.compile(r"<llm_query\b(?P<attributes>[^<>]*)>")
_DEPTH_ATTRIBUTE_RE = re.compile(r'^\s+depth\s*=\s*"(?P<depth>[^"]*)"\s*$')

ACTION_STOP_TOKENS = [QUERY_CLOSE]


LRM_ANSWER_OPEN = "<lrm_answer>"
LRM_ANSWER_CLOSE = "</lrm_answer>"
TOOL_ERROR_OPEN = "<tool_error>"
TOOL_ERROR_CLOSE = "</tool_error>"

KIND_QUERY = "query"



ERROR_MISSING_CALL = "missing_call"
ERROR_MISSING_OPEN = "missing_open"
ERROR_BAD_ATTRIBUTES = "bad_attributes"
ERROR_BAD_DEPTH = "bad_depth"
ERROR_EMPTY_QUERY = "empty"
ERROR_RESERVED_TAG = "reserved"
ERROR_QUERY_TOO_LONG = "query_too_long"
ERROR_QUERY_OVERLAP = "query_overlap"

INVALID_SYNTAX = "syntax"
INVALID_DEPTH = "depth"
INVALID_EMPTY = "empty"
INVALID_LENGTH = "length"
INVALID_OVERLAP = "overlap"
INVALID_OTHER = "other"

_INVALID_CATEGORY_BY_ERROR_CODE = {
    ERROR_MISSING_CALL: INVALID_SYNTAX,
    ERROR_MISSING_OPEN: INVALID_SYNTAX,
    ERROR_BAD_ATTRIBUTES: INVALID_SYNTAX,
    ERROR_BAD_DEPTH: INVALID_DEPTH,
    ERROR_EMPTY_QUERY: INVALID_EMPTY,
    ERROR_RESERVED_TAG: INVALID_SYNTAX,
    ERROR_QUERY_TOO_LONG: INVALID_LENGTH,
    ERROR_QUERY_OVERLAP: INVALID_OVERLAP,
}





DEPTHS = (1, 2, 3)
DEPTH_MIN, DEPTH_MAX = DEPTHS[0], DEPTHS[-1]

DEPTH_LABELS = {
    1: "a quick factual check — a name, a constant, a definition you nearly have",
    2: "a concept, theorem or formula you half-remember and need stated properly",
    3: "a knowledge gap you cannot bridge alone and that decides the problem",
}



DEPTH_RELATIVE_COST = {1: "cheapest", 2: "~10x depth 1", 3: "~50x depth 1"}


def format_lrm_answer(text: str) -> str:
    return f"\n{LRM_ANSWER_OPEN}{text}{LRM_ANSWER_CLOSE}\n"


def format_tool_error(msg: str) -> str:
    return f"\n{TOOL_ERROR_OPEN}{msg}{TOOL_ERROR_CLOSE}\n"


def format_tool_call(q: str, depth: int) -> str:
    'Format an escape-free query action.'
    if not isinstance(q, str) or not q.strip() or q != q.strip():
        raise ValueError("query text must be a non-empty trimmed string")
    for reserved in (QUERY_OPEN_PREFIX, QUERY_CLOSE):
        if reserved in q:
            raise ValueError(f"query text must not contain reserved tag {reserved}")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in DEPTHS:
        raise ValueError(f"depth must be an integer in {list(DEPTHS)}")
    return f'<llm_query depth="{depth}">{q}</llm_query>'


@dataclass
class ToolCall:
    kind: str
    q: Optional[str] = None
    depth: Optional[int] = None
    valid: bool = True
    error: Optional[str] = None
    error_code: Optional[str] = None
    span: tuple = field(default=(0, 0))


def invalid_category(error_code: Optional[str]) -> str:
    'Map a stable parser/guard error code to a bounded metric category.'
    return _INVALID_CATEGORY_BY_ERROR_CODE.get(error_code, INVALID_OTHER)


def has_tool_attempt(text: str) -> bool:
    'Return whether generated text contains either edge of our action tag.'
    return QUERY_OPEN_PREFIX in text or QUERY_CLOSE in text


def has_unterminated_tool_attempt(text: str) -> bool:
    'Return whether generation contains more query openers than closers.'
    return text.count(QUERY_OPEN_PREFIX) > text.count(QUERY_CLOSE)


def _coerce_depth(raw) -> Optional[int]:
    'Return the depth as an int, or None when it is not a usable value.'
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def parse_tool_call(text: str) -> Optional[ToolCall]:
    'Parse the *last* tool call in `text`.'
    close = text.rfind(QUERY_CLOSE)
    if close < 0:
        return None

    end = close + len(QUERY_CLOSE)
    open_matches = list(_QUERY_OPEN_RE.finditer(text, 0, close))
    if not open_matches:
        return ToolCall(
            KIND_QUERY,
            valid=False,
            error="missing open tag",
            error_code=ERROR_MISSING_OPEN,
            span=(0, end),
        )

    opening = open_matches[-1]
    start = opening.start()
    q = text[opening.end():close].strip()
    depth_match = _DEPTH_ATTRIBUTE_RE.fullmatch(opening.group("attributes"))
    if depth_match is None:
        return ToolCall(
            KIND_QUERY,
            q=q or None,
            valid=False,
            error='opening tag must be exactly <llm_query depth="N">',
            error_code=ERROR_BAD_ATTRIBUTES,
            span=(start, end),
        )

    depth = _coerce_depth(depth_match.group("depth"))
    if depth is None or depth not in DEPTHS:
        return ToolCall(
            KIND_QUERY,
            q=q or None,
            valid=False,
            error=f'"depth" must be an integer in {list(DEPTHS)}',
            error_code=ERROR_BAD_DEPTH,
            span=(start, end),
        )

    if not q:
        return ToolCall(
            KIND_QUERY,
            depth=depth,
            valid=False,
            error="query text must be non-empty",
            error_code=ERROR_EMPTY_QUERY,
            span=(start, end),
        )
    for reserved in (QUERY_OPEN_PREFIX, QUERY_CLOSE):
        if reserved in q:
            return ToolCall(
                KIND_QUERY,
                q=q,
                depth=depth,
                valid=False,
                error=f"query text must not contain reserved tag {reserved}",
                error_code=ERROR_RESERVED_TAG,
                span=(start, end),
            )

    return ToolCall(KIND_QUERY, q=q, depth=depth, span=(start, end))





def _depth_block() -> str:
    return "\n".join(
        f"     depth {d}: {DEPTH_LABELS[d]} ({DEPTH_RELATIVE_COST[d]})"
        for d in DEPTHS
    )


def system_prompt(with_tool: bool = True) -> str:
    'Policy system prompt. `with_tool=False` gives the plain no-tool baseline.'
    if not with_tool:
        return (
            "You are a careful reasoning assistant. Solve the problem step by step.\n"
            "Finish with your final answer within \\boxed{}."
        )
    return (
        'You are a careful reasoning assistant. Solve the problem step by step.\n'
        '\n'
        'Use this optional, expensive tool only for a specific external knowledge gap:\n'
        '\n'
        '<llm_query depth="2">short question about the missing fact</llm_query>\n'
        '\n'
        'Use exactly one double-quoted `depth` attribute and raw question text.\n'
        'Do NOT use JSON or other attributes. LaTeX backslashes are literal, not\n'
        'doubled or escaped.\n'
        '\n'
        'Ask a SHORT question (under 300 characters) about a concept, theorem,\n'
        'formula, or fact you are unsure of. Do NOT paste the problem; the assistant\n'
        'cannot see it or solve it for you. Its reply appears as\n'
        '<lrm_answer>...</lrm_answer>.\n'
        '\n'
        'Normally one call is enough. After a reply, integrate it and finish.\n'
        'Never repeat or rephrase a query; call again only for a distinct gap that\n'
        'still blocks the answer. After a <tool_error>, do not repeat the same\n'
        'malformed call.\n'
        '\n'
        '`depth` buys the following oracle tier and response budget:\n'
        f"{_depth_block()}\n"
        '\n'
        'Calling costs you, and deeper calls cost much more. Ask only if the answer\n'
        'would change what you do next, using the shallowest sufficient depth.\n'
        '\n'
        'Finish with your final answer within \\boxed{}.'
    )



QUERY_CONTRACT = (
    "You are a knowledge assistant. A small model asks you a short question while solving a "
    "problem you CANNOT see. Answer with relevant concepts, theorems, formulas, or facts only. "
    "Do NOT attempt to solve any problem, do NOT give a final answer, a numeric result, or a "
    "multiple-choice letter."
)


def query_contract(word_cap: int) -> str:
    "Contract for one call, with the depth's length budget stated in words."
    return f"{QUERY_CONTRACT} Answer in at most {word_cap} words, and finish your sentence."
