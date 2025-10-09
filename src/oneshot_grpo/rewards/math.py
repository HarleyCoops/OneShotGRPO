"""Reward functions and helpers for math reasoning tasks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

AnswerExtractor = Callable[[str], str]


def _default_answer_extractor(text: str) -> str:
    """Pull the string inside <answer> tags; fall back to hash-colon patterns."""
    xml_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        return xml_match.group(1).strip()
    hash_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if hash_match:
        return hash_match.group(1).strip()
    return text.strip()


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_close(lhs: str, rhs: str, *, tol: float = 1e-6) -> bool:
    if not (_is_numeric(lhs) and _is_numeric(rhs)):
        return lhs == rhs
    return math.isclose(float(lhs), float(rhs), rel_tol=tol, abs_tol=tol)


def correctness_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    references: Sequence[str],
    *,
    answer_extractor: AnswerExtractor = _default_answer_extractor,
) -> List[float]:
    """Binary reward for matching the ground-truth answer."""
    rewards: List[float] = []
    for completion, reference in zip(completions, references):
        predicted = answer_extractor(completion)
        target = answer_extractor(reference)
        rewards.append(1.0 if _is_close(predicted, target) else 0.0)
    return rewards


def numeric_answer_reward(completions: Sequence[str], *, answer_extractor: AnswerExtractor = _default_answer_extractor) -> List[float]:
    """Reward 1 when the completion yields a numeric answer."""
    rewards: List[float] = []
    for completion in completions:
        predicted = answer_extractor(completion)
        rewards.append(1.0 if _is_numeric(predicted) else 0.0)
    return rewards


def xml_format_reward(completions: Sequence[str]) -> List[float]:
    """Reward 1 when required XML tags are present; partial credit for partial tags."""
    required_tags = ["<scratchpad>", "</scratchpad>", "<answer>", "</answer>"]
    rewards: List[float] = []
    for completion in completions:
        present = sum(1 for tag in required_tags if tag.lower() in completion.lower())
        rewards.append(present / len(required_tags))
    return rewards


@dataclass
class RewardBreakdown:
    total: List[float]
    components: Dict[str, List[float]]


def combine_rewards(component_rewards: Dict[str, List[float]], weights: Dict[str, float]) -> RewardBreakdown:
    """Merge component-wise rewards into a weighted total."""
    totals: List[float] = [0.0] * len(next(iter(component_rewards.values()), []))
    for name, values in component_rewards.items():
        weight = weights.get(name, 1.0)
        for idx, value in enumerate(values):
            totals[idx] += weight * value
    return RewardBreakdown(total=totals, components=component_rewards)


def evaluate_reward_stack(
    prompts: Sequence[str],
    completions: Sequence[str],
    references: Sequence[str],
    *,
    weights: Dict[str, float] | None = None,
) -> RewardBreakdown:
    """Compute a default reward stack used in math reasoning experiments."""
    weights = weights or {"correctness": 1.0, "numeric": 0.1, "xml": 0.1}
    components = {
        "correctness": correctness_reward(prompts, completions, references),
        "numeric": numeric_answer_reward(completions),
        "xml": xml_format_reward(completions),
    }
    return combine_rewards(components, weights)


__all__ = [
    "RewardBreakdown",
    "combine_rewards",
    "correctness_reward",
    "evaluate_reward_stack",
    "numeric_answer_reward",
    "xml_format_reward",
]
