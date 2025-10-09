"""High-level helpers for math-focused GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from datasets import Dataset

from oneshot_grpo.data import DEFAULT_SYSTEM_PROMPT, format_for_grpo, load_gsm8k
from oneshot_grpo.rewards import default_reward_functions
from oneshot_grpo.train import TrainingArtifacts, build_grpo_trainer

DEFAULT_REWARD_WEIGHTS = [1.0, 0.1, 0.1]
DEFAULT_OUTPUT_DIR = "outputs/math-grpo"
DEFAULT_RUN_NAME = "math-grpo"
DEFAULT_GRPO_ARGS = {
    "output_dir": DEFAULT_OUTPUT_DIR,
    "run_name": DEFAULT_RUN_NAME,
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "bf16": False,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "generation_batch_size": 16,
    "num_generations": 16,
    "max_prompt_length": 256,
    "max_completion_length": 200,
    "num_train_epochs": 1,
    "save_steps": 100,
    "max_grad_norm": 0.1,
    "log_on_each_node": False,
    "report_to": "none",
}


@dataclass
class MathTrainingSetup:
    dataset: Dataset
    reward_funcs: List[Callable[..., List[float]]]
    artifacts: TrainingArtifacts
    grpo_args: Dict[str, Any]


def prepare_math_dataset(
    split: str = "train",
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    """Load GSM8K and attach prompts/answers compatible with the GRPO trainer."""
    dataset = load_gsm8k(split=split)
    return format_for_grpo(dataset, system_prompt=system_prompt)


def create_math_training_setup(
    *,
    model_name: str,
    split: str = "train",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    run_name: str = DEFAULT_RUN_NAME,
    report_to: str = "none",
    reward_weights: Optional[Sequence[float]] = None,
    grpo_args: Optional[Dict[str, Any]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> MathTrainingSetup:
    """Bundle dataset loading, reward selection, and trainer construction."""
    dataset = prepare_math_dataset(split=split, system_prompt=system_prompt)
    reward_funcs = default_reward_functions()
    full_grpo_args = dict(DEFAULT_GRPO_ARGS)
    full_grpo_args.update({"output_dir": output_dir, "run_name": run_name, "report_to": report_to})
    if grpo_args:
        full_grpo_args.update(grpo_args)
    full_grpo_args.setdefault("reward_weights", list(reward_weights or DEFAULT_REWARD_WEIGHTS))
    artifacts = build_grpo_trainer(
        model_name=model_name,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        grpo_args=full_grpo_args,
        trainer_kwargs=trainer_kwargs,
    )
    return MathTrainingSetup(
        dataset=dataset,
        reward_funcs=reward_funcs,
        artifacts=artifacts,
        grpo_args=full_grpo_args,
    )


__all__ = [
    "DEFAULT_REWARD_WEIGHTS",
    "MathTrainingSetup",
    "create_math_training_setup",
    "prepare_math_dataset",
]
