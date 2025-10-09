# Math GRPO Overview

This document tracks the math reasoning pipeline as it moves from notebooks into the reusable `oneshot_grpo` package.

## Quick Start

```python
from oneshot_grpo.train import create_math_training_setup, run_training

setup = create_math_training_setup(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    split="train",
)

run_training(setup.artifacts)
```

Behind the scenes:

- `oneshot_grpo.data.format_for_grpo` loads GSM8K, attaches the system prompt, and preserves both the full worked solution and the numeric answer needed for rewards.
- `oneshot_grpo.rewards.default_reward_functions` supplies the multi-signal reward stack (correctness, numeric guard, XML structure) so the training notebook can call into the package instead of redefining these helpers.
- `oneshot_grpo.train.math_pipeline.create_math_training_setup` builds a `GRPOTrainer`, wiring in reward weights, dataset references, and any overrides passed via `grpo_args` or `trainer_kwargs`.

Use this module from the notebook by importing `create_math_training_setup`, replacing local reward definitions with `default_reward_functions()`, and logging any experiment metadata directly through the returned `TrainingArtifacts`.

## CLI Runner

You can also launch training from the command line with W&B and Hugging Face Hub integration:

```bash
python -m oneshot_grpo.train.run_math_grpo \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --use-wandb \
  --wandb-project qwen-math-grpo \
  --push-to-hub \
  --hub-model-id your-hf-user/oneshot-grpo-math
```

The CLI delegates to `setup.reward_funcs`, so there is no need to redefine XML or numeric rewards—the same stack is reused across notebooks and scripts.
