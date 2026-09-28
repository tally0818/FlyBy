'Shared accuracy-only RLVR reward for baseline training.'
from __future__ import annotations

from typing import Any

from src.eval.grader import grade


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    del data_source
    answer_format = (extra_info or {}).get("answer_format", "math")
    accuracy = float(grade(answer_format, solution_str, str(ground_truth)))
    return {"score": accuracy, "accuracy": accuracy}
