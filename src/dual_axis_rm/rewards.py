from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

from swift.plugin import ORM, orms

RESPONSE_THINK_PATTERN = re.compile(
    r"<response think>.*?</response think>",
    re.DOTALL | re.IGNORECASE,
)
FLUENCY_THINK_PATTERN = re.compile(
    r"<fluency think>.*?</fluency think>",
    re.DOTALL | re.IGNORECASE,
)
SCORE_PATTERN = re.compile(
    r"<overall score>\s*([12])\s*</overall score>",
    re.DOTALL | re.IGNORECASE,
)


def extract_overall_score(text: str) -> Optional[int]:
    match = SCORE_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(1))


def normalize_solution(value: Any) -> Optional[int]:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, str):
        value = value.strip()
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in {1, 2} else None


def has_required_format(text: str) -> bool:
    return (
        bool(RESPONSE_THINK_PATTERN.search(text))
        and bool(FLUENCY_THINK_PATTERN.search(text))
        and extract_overall_score(text) is not None
    )


class DualAxisScoreAccuracy(ORM):
    def __call__(self, completions: Iterable[str], solution: Iterable[Any], **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion, target in zip(completions, solution):
            pred = extract_overall_score(completion)
            gold = normalize_solution(target)
            rewards.append(1.0 if pred is not None and gold is not None and pred == gold else 0.0)
        return rewards


class DualAxisFormatAccuracy(ORM):
    def __call__(self, completions: Iterable[str], **kwargs) -> List[float]:
        return [1.0 if has_required_format(completion) else 0.0 for completion in completions]


orms["dual_axis_score_acc"] = DualAxisScoreAccuracy
orms["dual_axis_format_acc"] = DualAxisFormatAccuracy
