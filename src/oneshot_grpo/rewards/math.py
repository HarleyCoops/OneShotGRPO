"""Reward functions and helpers for math reasoning tasks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

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


def _as_text(completion: Any) -> str:
    """Normalize chat/message completions to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, Sequence):
        if completion and isinstance(completion[0], dict):
            return str(completion[0].get("content", ""))
        if completion and isinstance(completion[0], str):
            return completion[0]
    return str(completion)


def _batch_as_text(completions: Sequence[Any]) -> List[str]:
    return [_as_text(completion) for completion in completions]


def correctness_reward(
    prompts: Sequence[str],
    completions: Sequence[Any],
    references: Sequence[str],
    *,
    answer_extractor: AnswerExtractor = _default_answer_extractor,
) -> List[float]:
    """Binary reward for matching the ground-truth answer."""
    completion_texts = _batch_as_text(completions)
    rewards: List[float] = []
    for completion, reference in zip(completion_texts, references):
        predicted = answer_extractor(completion)
        target = answer_extractor(reference)
        rewards.append(1.0 if _is_close(predicted, target) else 0.0)
    return rewards


def numeric_answer_reward(
    completions: Sequence[Any], *, answer_extractor: AnswerExtractor = _default_answer_extractor
) -> List[float]:
    """Reward 1 when the completion yields a numeric answer."""
    completion_texts = _batch_as_text(completions)
    rewards: List[float] = []
    for completion in completion_texts:
        predicted = answer_extractor(completion)
        rewards.append(1.0 if _is_numeric(predicted) else 0.0)
    return rewards


def xml_format_reward(completions: Sequence[Any]) -> List[float]:
    """Reward 1 when required XML tags are present; partial credit for partial tags."""
    completion_texts = _batch_as_text(completions)
    required_tags = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
    rewards: List[float] = []
    for completion in completion_texts:
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
    completions: Sequence[Any],
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


def correctness_reward_func(
    *,
    prompts: Sequence[str],
    completions: Sequence[Any],
    reference_answer: Sequence[str],
    **_: Any,
) -> List[float]:
    """Wrapper matching TRL's reward function signature."""
    return correctness_reward(prompts, completions, reference_answer)


def numeric_reward_func(
    *,
    completions: Sequence[Any],
    **_: Any,
) -> List[float]:
    """Wrapper matching TRL's reward function signature."""
    return numeric_answer_reward(completions)


def xml_reward_func(
    *,
    completions: Sequence[Any],
    **_: Any,
) -> List[float]:
    """Wrapper matching TRL's reward function signature."""
    return xml_format_reward(completions)


def default_reward_functions() -> List[Callable[..., List[float]]]:
    """Return the standard reward stack as a list of callable functions."""
    return [correctness_reward_func, numeric_reward_func, xml_reward_func]


__all__ = [
    "RewardBreakdown",
    "combine_rewards",
    "correctness_reward",
    "correctness_reward_func",
    "default_reward_functions",
    "evaluate_reward_stack",
    "numeric_answer_reward",
    "numeric_reward_func",
    "xml_format_reward",
    "xml_reward_func",
]
