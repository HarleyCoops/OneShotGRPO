"""Utilities for loading and preparing GSM8K data for GRPO training."""

from __future__ import annotations

import re
from typing import Mapping

try:
    from datasets import Dataset, load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `datasets` package is required to load GSM8K. Install it with `pip install datasets`."
    ) from exc


DEFAULT_SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
""".strip()


def load_gsm8k(split: str = "train", *, config: str = "main") -> Dataset:
    """Fetch the GSM8K split, defaulting to the main train partition."""
    return load_dataset("openai/gsm8k", config, split=split)


def extract_hash_answer(text: str) -> str:
    """Pull the numeric answer that follows the #### marker in GSM8K examples."""
    match = re.search(r"####\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_chat_prompt(question: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Compose the system + user chat messages for a math question."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def format_for_grpo(
    dataset: Dataset,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    """Attach chat prompts and reference answers ready for GRPO sampling."""

    def _format(example: Mapping[str, str]) -> Mapping[str, str]:
        question = example["question"]
        solution = example["answer"]
        example["prompt"] = build_chat_prompt(question, system_prompt=system_prompt)
        example["reference_solution"] = solution.strip()
        example["reference_answer"] = extract_hash_answer(solution)
        return example

    return dataset.map(_format)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "build_chat_prompt",
    "extract_hash_answer",
    "format_for_grpo",
    "load_gsm8k",
]
