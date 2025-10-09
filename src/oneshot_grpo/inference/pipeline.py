"""Inference helpers for running GRPO-tuned models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from transformers import pipeline

DEFAULT_MODEL_ID = "HarleyCooper/GRPOtuned"


def build_text_generator(model_id: str = DEFAULT_MODEL_ID, **pipeline_kwargs):
    """Create a Hugging Face text-generation pipeline for the tuned policy."""
    return pipeline("text-generation", model=model_id, **pipeline_kwargs)


def generate_math_solution(
    prompt: str,
    *,
    generator=None,
    max_new_tokens: int = 6000,
    temperature: float = 1.0,
    **generate_kwargs,
):
    """Produce a reasoning trace and answer for a math prompt."""
    text_generator = generator or build_text_generator()
    outputs = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **generate_kwargs,
    )
    if not outputs:
        raise ValueError("Text generator returned no completions.")
    completion = outputs[0]
    if isinstance(completion, dict):
        return completion.get("generated_text", "")
    return str(completion)


def append_jsonl(records: Iterable[dict], path: Path | str) -> None:
    """Persist one or more records to a JSONL file."""
    target = Path(path)
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def save_prompt_and_response(
    prompt: str,
    generated_output: str,
    *,
    path: Path | str = "math_results.jsonl",
    extra: Optional[dict] = None,
) -> None:
    """Save the prompt, generated answer, and optional metadata."""
    payload = {"prompt": prompt, "generated_output": generated_output}
    if extra:
        payload.update(extra)
    append_jsonl([payload], path=path)


__all__ = [
    "DEFAULT_MODEL_ID",
    "append_jsonl",
    "build_text_generator",
    "generate_math_solution",
    "save_prompt_and_response",
]
