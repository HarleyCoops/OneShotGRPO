"""Command-line entry point for math GRPO training with built-in W&B and HF hooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from oneshot_grpo.data import DEFAULT_SYSTEM_PROMPT
from oneshot_grpo.train import create_math_training_setup, run_training


def _load_system_prompt(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    return path.read_text(encoding="utf-8").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a math GRPO model.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model identifier to fine-tune.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (defaults to 'train').",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=Path,
        help="Optional path to a custom system prompt template.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/math-grpo",
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--run-name",
        default="math-grpo",
        help="Training run name used for logging and checkpoint folders.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        help="W&B project name (required when --use-wandb is set).",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B entity/organization.",
    )
    parser.add_argument(
        "--wandb-run-name",
        help="Optional W&B run name override.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        help="Optional list of W&B tags.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push checkpoints to the Hugging Face Hub after training.",
    )
    parser.add_argument(
        "--hub-model-id",
        help="Repository name for the Hub upload (e.g., user/model-name).",
    )
    parser.add_argument(
        "--hub-token",
        help="Hugging Face token to authenticate uploads. Defaults to environment token.",
    )
    parser.add_argument(
        "--hub-strategy",
        default="checkpoint",
        help="Hub upload strategy (see transformers Trainer docs).",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create a private Hub repo when pushing checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system_prompt = _load_system_prompt(args.system_prompt_path)
    report_to = "wandb" if args.use_wandb else "none"

    hub_kwargs = {}
    if args.push_to_hub or args.hub_model_id:
        raw_hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": args.hub_model_id,
            "hub_strategy": args.hub_strategy,
            "hub_token": args.hub_token,
            "hub_private_repo": args.hub_private,
        }
        hub_kwargs = {key: value for key, value in raw_hub_kwargs.items() if value is not None}

    system_prompt_value = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    setup = create_math_training_setup(
        model_name=args.model_name,
        split=args.split,
        system_prompt=system_prompt_value,
        output_dir=args.output_dir,
        run_name=args.run_name,
        report_to=report_to,
        grpo_args=hub_kwargs if hub_kwargs else None,
    )

    if args.use_wandb:
        if not args.wandb_project:
            raise ValueError("--wandb-project must be provided when --use-wandb is set.")
        import wandb  # pylint: disable=import-outside-toplevel

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.run_name,
            tags=args.wandb_tags,
            config={
                "model_name": args.model_name,
                "split": args.split,
                "grpo_args": setup.grpo_args,
            },
        )

    run_training(setup.artifacts)

    if args.push_to_hub or args.hub_model_id:
        commit_message = f"Add GRPO checkpoint for {args.model_name}"
        setup.artifacts.trainer.push_to_hub(commit_message=commit_message)

    if args.use_wandb:
        import wandb  # pylint: disable=import-outside-toplevel

        wandb.finish()


if __name__ == "__main__":
    main()
