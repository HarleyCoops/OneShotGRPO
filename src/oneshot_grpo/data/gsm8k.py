"""Utilities for loading and preparing GSM8K data."""

from __future__ import annotations

from typing import Iterable, Mapping

try:
    from datasets import Dataset, load_dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `datasets` package is required to load GSM8K. Install it with `pip install datasets`."
    ) from exc


def load_gsm8k(split: str = "train", *, config: str = "main") -> Dataset:
    """Fetch the GSM8K split, defaulting to the main train partition."""
    return load_dataset("gsm8k", config, split=split)


def format_for_grpo(
    dataset: Dataset,
    *,
    system_prompt: str,
    prompt_template: str,
    cot_format: str,
) -> Dataset:
    """Attach system prompts and formatted questions ready for sampling."""

    def _format(example: Mapping[str, str]) -> Mapping[str, str]:
        question = example["question"].strip()
        answer = example["answer"].strip()
        prompt = prompt_template.format(question=question, system_prompt=system_prompt, cot_format=cot_format)
        example["prompt"] = prompt
        example["reference_answer"] = answer
        return example

    return dataset.map(_format)


__all__ = ["format_for_grpo", "load_gsm8k"]
