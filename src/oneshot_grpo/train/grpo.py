"""High-level utilities for configuring GRPO training runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass
class TrainingArtifacts:
    trainer: Any
    config: Any


def build_grpo_trainer(
    model_name: str,
    reward_funcs,
    train_dataset,
    *,
    grpo_args: Dict[str, Any] | None = None,
    trainer_kwargs: Dict[str, Any] | None = None,
):
    """Construct a GRPOTrainer with lazy imports to keep startup light."""
    from trl import GRPOConfig, GRPOTrainer  # defer import until needed

    config = GRPOConfig(**(grpo_args or {}))
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        args=config,
        **(trainer_kwargs or {}),
    )
    return TrainingArtifacts(trainer=trainer, config=config)


def run_training(artifacts: TrainingArtifacts) -> None:
    """Kick off training using the prepared trainer."""
    artifacts.trainer.train()


__all__ = ["TrainingArtifacts", "build_grpo_trainer", "run_training"]
