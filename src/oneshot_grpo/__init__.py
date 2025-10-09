"""OneShot GRPO package for math-focused reinforcement learning workflows."""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TENSORFLOW", "1")
os.environ.setdefault("USE_TF", "0")

from .inference.pipeline import (
    DEFAULT_MODEL_ID,
    build_text_generator,
    generate_math_solution,
    save_prompt_and_response,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "build_text_generator",
    "generate_math_solution",
    "save_prompt_and_response",
]
