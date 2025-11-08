# Weights & Biases Visualization Guide

##  Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Basic Logging](#basic-logging)
- [Advanced Visualizations](#advanced-visualizations)
- [3D Charts for RL Training](#3d-charts-for-rl-training)
- [Custom Dashboards](#custom-dashboards)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Weights & Biases (W&B) provides comprehensive experiment tracking and visualization for machine learning. This guide focuses on advanced monitoring techniques for GRPO training, including 3D visualizations of reward landscapes and training dynamics.

### Key Features for GRPO

- **Real-time Metrics**: Track loss, rewards, KL divergence live
- **3D Visualizations**: Plot reward landscapes and policy evolution
- **Artifact Tracking**: Version datasets, models, and checkpoints
- **Hyperparameter Sweeps**: Automatic optimization
- **Team Collaboration**: Share runs and dashboards
- **Model Comparison**: Side-by-side run analysis

---

## Setup

### Installation

```python
# Install W&B
!pip install -q wandb

# Import
import wandb
```

### Authentication

```python
# Option 1: Interactive login
wandb.login()

# Option 2: API key
import os
os.environ['WANDB_API_KEY'] = 'your-api-key-here'
wandb.login(key=os.environ['WANDB_API_KEY'])

# Option 3: Notebook secret (recommended for Colab)
from google.colab import userdata
wandb.login(key=userdata.get('WANDB_API_KEY'))
```

### Project Initialization

```python
# Initialize run with comprehensive config
run = wandb.init(
    project="grpo-math-reasoning",
    entity="your-username",  # Optional
    name="qwen-05b-gsm8k-v1",
    tags=["grpo", "math", "gsm8k", "qwen"],
    notes="GRPO training on GSM8K with multi-objective rewards",

    config={
        # Model
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_params": "500M",

        # Training
        "learning_rate": 5e-6,
        "batch_size": 4,
        "num_epochs": 1,
        "max_steps": 1000,

        # GRPO specific
        "num_generations": 16,
        "reward_weights": {
            "correctness": 2.0,
            "format": 1.5,
            "numeric": 0.5
        },

        # Dataset
        "dataset": "openai/gsm8k",
        "dataset_size": 7500,
    }
)
```

---

## Basic Logging

### Metrics Logging

```python
# Log single metric
wandb.log({"train/loss": 0.234})

# Log multiple metrics
wandb.log({
    "train/loss": 0.234,
    "train/learning_rate": 5e-6,
    "train/reward": 2.34,
    "step": 100
})

# Log with custom step
wandb.log({"metric": value}, step=custom_step)
```

### During GRPO Training

```python
from transformers import TrainerCallback

class WandBMetricsCallback(TrainerCallback):
    """Enhanced W&B logging for GRPO"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log all trainer metrics
            wandb.log(logs, step=state.global_step)

            # Compute and log custom metrics
            if 'rewards' in logs:
                rewards = logs['rewards']
                wandb.log({
                    "rewards/mean": np.mean(rewards),
                    "rewards/std": np.std(rewards),
                    "rewards/min": np.min(rewards),
                    "rewards/max": np.max(rewards),
                    "rewards/median": np.median(rewards),
                }, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Log evaluation metrics with "eval/" prefix
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(eval_metrics, step=state.global_step)

# Add to trainer
trainer = GRPOTrainer(
    # ... args
    callbacks=[WandBMetricsCallback()]
)
```

### Logging Rewards

```python
# Inside reward function
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Calculate rewards
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    # Log reward distribution
    step = kwargs.get('step', 0)
    if step % 10 == 0:  # Log every 10 steps
        wandb.log({
            "rewards/correctness_mean": np.mean(rewards),
            "rewards/correctness_rate": sum(r > 0 for r in rewards) / len(rewards),
        }, step=step)

    return rewards
```

---

## Advanced Visualizations

### Tables for Sample Outputs

```python
# Create table for model generations
columns = ["step", "question", "reasoning", "answer", "ground_truth", "reward"]
data = []

def log_generation_sample(step, question, response, ground_truth, reward):
    """Log model generation to W&B table"""
    # Parse response
    try:
        reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
    except:
        reasoning = "Parse error"
        answer = "Parse error"

    # Add to table
    data.append([step, question, reasoning, answer, ground_truth, reward])

    # Update table every 50 steps
    if len(data) % 50 == 0:
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"generations": table})

# Use in training loop
log_generation_sample(
    step=100,
    question="What is 2+2?",
    response="<reasoning>2 plus 2 equals 4</reasoning><answer>4</answer>",
    ground_truth="4",
    reward=4.0
)
```

### Histograms

```python
# Log reward distribution
wandb.log({
    "rewards/distribution": wandb.Histogram(reward_values),
    "step": step
})

# Log answer length distribution
answer_lengths = [len(extract_xml_answer(r)) for r in responses]
wandb.log({
    "generation/answer_length_dist": wandb.Histogram(answer_lengths)
})
```

### Charts

```python
# Line charts (automatic from wandb.log)
wandb.log({"train/loss": loss, "train/reward": reward})

# Bar charts
wandb.log({
    "rewards/breakdown": wandb.plot.bar(
        wandb.Table(
            columns=["reward_type", "value"],
            data=[
                ["correctness", 2.0],
                ["format", 1.5],
                ["numeric", 0.5]
            ]
        ),
        "reward_type",
        "value",
        title="Reward Breakdown"
    )
})
```

---

## 3D Charts for RL Training

### Reward Landscape Visualization

```python
import numpy as np

def log_reward_landscape_3d(step, rewards_history, steps_history, example_ids):
    """
    Create 3D visualization of reward landscape.

    X-axis: Training step
    Y-axis: Example ID
    Z-axis: Reward value
    """
    # Prepare data
    data = []
    for s, ex_id, reward in zip(steps_history, example_ids, rewards_history):
        data.append([s, ex_id, reward])

    # Create table
    table = wandb.Table(
        columns=["step", "example_id", "reward"],
        data=data
    )

    # Create 3D scatter plot
    wandb.log({
        "reward_landscape_3d": wandb.plot_table(
            "wandb/3d-scatter/v0",
            table,
            {"x": "step", "y": "example_id", "z": "reward"},
            {
                "title": "Reward Landscape Over Training",
                "x-axis-title": "Training Step",
                "y-axis-title": "Example ID",
                "z-axis-title": "Reward"
            }
        )
    })

# Collect data during training
rewards_history = []
steps_history = []
example_ids = []

# In training loop
for step in range(num_steps):
    # ... training code ...

    # Collect data
    for ex_id, reward in enumerate(batch_rewards):
        rewards_history.append(reward)
        steps_history.append(step)
        example_ids.append(ex_id)

    # Log every 100 steps
    if step % 100 == 0:
        log_reward_landscape_3d(step, rewards_history, steps_history, example_ids)
```

### Policy Evolution 3D

```python
def log_policy_evolution_3d(kl_divergences, rewards, steps):
    """
    Visualize policy evolution in 3D.

    X-axis: KL divergence from reference policy
    Y-axis: Average reward
    Z-axis: Training step (color coded)
    """
    data = [[kl, reward, step] for kl, reward, step in zip(kl_divergences, rewards, steps)]

    table = wandb.Table(
        columns=["kl_divergence", "reward", "step"],
        data=data
    )

    wandb.log({
        "policy_evolution_3d": wandb.plot_table(
            "wandb/3d-scatter/v0",
            table,
            {"x": "kl_divergence", "y": "reward", "z": "step"},
            {
                "title": "Policy Evolution (KL vs Reward vs Step)",
                "x-axis-title": "KL Divergence",
                "y-axis-title": "Average Reward",
                "z-axis-title": "Training Step"
            }
        )
    })
```

### Reward Component Breakdown 3D

```python
def log_reward_components_3d(step, component_rewards):
    """
    3D visualization of different reward components over time.

    component_rewards: dict of {component_name: [reward_values]}
    """
    data = []
    for component, rewards in component_rewards.items():
        for i, reward in enumerate(rewards):
            data.append([step, component, reward, i])

    table = wandb.Table(
        columns=["step", "component", "reward", "sample_id"],
        data=data
    )

    wandb.log({
        "reward_components_3d": wandb.plot_table(
            "wandb/3d-scatter/v0",
            table,
            {"x": "step", "y": "component", "z": "reward"},
            {
                "title": "Reward Components Breakdown",
                "x-axis-title": "Training Step",
                "y-axis-title": "Reward Component",
                "z-axis-title": "Reward Value"
            }
        )
    })

# Usage
component_rewards = {
    "correctness": [2.0, 0.0, 2.0, 2.0],
    "format": [1.5, 1.0, 1.5, 1.5],
    "numeric": [0.5, 0.5, 0.5, 0.5]
}
log_reward_components_3d(step=100, component_rewards=component_rewards)
```

### Surface Plots (Custom Vega)

```python
def create_reward_surface_plot(step_grid, example_grid, reward_grid):
    """
    Create 3D surface plot of reward landscape.

    Uses custom Vega specification for W&B.
    """
    # Flatten grids for Vega format
    data = []
    for i in range(len(step_grid)):
        for j in range(len(example_grid[0])):
            data.append({
                "step": step_grid[i],
                "example": example_grid[i][j],
                "reward": reward_grid[i][j]
            })

    # Vega specification
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 600,
        "height": 400,
        "data": [{"name": "table", "values": data}],
        "marks": [{
            "type": "rect",
            "from": {"data": "table"},
            "encode": {
                "enter": {
                    "x": {"scale": "xscale", "field": "step"},
                    "y": {"scale": "yscale", "field": "example"},
                    "fill": {"scale": "color", "field": "reward"},
                    "width": {"scale": "xscale", "band": 1},
                    "height": {"scale": "yscale", "band": 1}
                }
            }
        }],
        "scales": [
            {
                "name": "xscale",
                "type": "band",
                "domain": {"data": "table", "field": "step"},
                "range": "width"
            },
            {
                "name": "yscale",
                "type": "band",
                "domain": {"data": "table", "field": "example"},
                "range": "height"
            },
            {
                "name": "color",
                "type": "linear",
                "domain": {"data": "table", "field": "reward"},
                "range": {"scheme": "viridis"}
            }
        ]
    }

    wandb.log({"reward_surface": wandb.plots.HeatMap(
        x_labels=[str(x) for x in step_grid],
        y_labels=[str(y) for y in example_grid[:, 0]],
        matrix_values=reward_grid,
        show_text=False
    )})
```

---

## Custom Dashboards

### Create Custom Charts

```python
# 1. Log data with custom step
for step in range(1000):
    wandb.log({
        "custom/metric_a": metric_a,
        "custom/metric_b": metric_b,
        "custom/metric_c": metric_c,
    }, step=step)

# 2. In W&B UI:
# - Go to workspace
# - Click "Add visualization"
# - Choose "Line plot" or "Custom chart"
# - Select metrics
# - Save to dashboard
```

### Programmatic Dashboard Creation

```python
import wandb

# Create dashboard programmatically
api = wandb.Api()
entity = "your-entity"
project = "grpo-math-reasoning"

# Define dashboard panels
panels = [
    {
        "id": "panel1",
        "type": "line-plot",
        "config": {
            "metrics": ["train/loss", "train/reward"],
            "title": "Training Metrics"
        }
    },
    {
        "id": "panel2",
        "type": "scatter-plot",
        "config": {
            "x": "rewards/correctness",
            "y": "rewards/format",
            "title": "Reward Correlation"
        }
    },
    {
        "id": "panel3",
        "type": "parallel-coordinates",
        "config": {
            "columns": ["learning_rate", "batch_size", "num_generations", "final_reward"],
            "title": "Hyperparameter Importance"
        }
    }
]

# Create dashboard (requires W&B API)
# Note: This is conceptual - actual API may vary
workspace = api.workspace(entity, project)
workspace.create_dashboard("GRPO Training Dashboard", panels)
```

---

## Best Practices

### 1. Organized Metric Naming

Use hierarchical naming with `/`:

```python
wandb.log({
    # Training metrics
    "train/loss": loss,
    "train/grad_norm": grad_norm,
    "train/learning_rate": lr,

    # Reward metrics
    "rewards/total": total_reward,
    "rewards/correctness": correctness_reward,
    "rewards/format": format_reward,
    "rewards/mean": mean_reward,
    "rewards/std": std_reward,

    # Generation metrics
    "generation/length": avg_length,
    "generation/quality": quality_score,

    # RL specific
    "rl/kl_divergence": kl_div,
    "rl/policy_entropy": entropy,
    "rl/value_estimate": value,
})
```

### 2. Sampling for Large Datasets

Don't log every single sample:

```python
# Log samples at regular intervals
if step % 50 == 0:
    log_generation_sample(...)

# Or log a subset
import random
if random.random() < 0.1:  # 10% sampling
    log_generation_sample(...)
```

### 3. Artifacts for Versioning

```python
# Log dataset
dataset_artifact = wandb.Artifact(
    name="gsm8k-train",
    type="dataset",
    description="GSM8K training split with GRPO formatting"
)
dataset_artifact.add_file("gsm8k_formatted.jsonl")
wandb.log_artifact(dataset_artifact)

# Log model checkpoints
model_artifact = wandb.Artifact(
    name=f"qwen-grpo-checkpoint-{step}",
    type="model",
    description=f"GRPO checkpoint at step {step}"
)
model_artifact.add_dir("outputs/checkpoint-1000")
wandb.log_artifact(model_artifact)

# Log final model
final_artifact = wandb.Artifact(
    name="qwen-grpo-final",
    type="model",
    metadata={
        "final_loss": final_loss,
        "final_reward": final_reward,
        "training_steps": total_steps
    }
)
final_artifact.add_dir("outputs/final_model")
wandb.log_artifact(final_artifact)
```

### 4. Hyperparameter Sweeps

```python
# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'final_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 1e-6,
            'max': 1e-4,
            'distribution': 'log_uniform'
        },
        'num_generations': {
            'values': [8, 16, 32]
        },
        'weight_decay': {
            'min': 0.0,
            'max': 0.3
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="grpo-math-reasoning")

# Define training function
def train():
    run = wandb.init()
    config = wandb.config

    # Train with config
    trainer = GRPOTrainer(
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        weight_decay=config.weight_decay,
        # ...
    )
    trainer.train()

    # Log final metrics
    wandb.log({"final_reward": trainer.state.best_metric})

# Run sweep
wandb.agent(sweep_id, function=train, count=20)
```

### 5. Compare Runs

```python
# Get runs from API
api = wandb.Api()
runs = api.runs("your-entity/grpo-math-reasoning")

# Create comparison table
comparison_data = []
for run in runs:
    comparison_data.append([
        run.name,
        run.config.get('learning_rate'),
        run.config.get('num_generations'),
        run.summary.get('final_reward'),
        run.summary.get('train/loss')
    ])

# Log as table
table = wandb.Table(
    columns=["run_name", "lr", "num_gen", "final_reward", "final_loss"],
    data=comparison_data
)
wandb.log({"run_comparison": table})
```

---

## Troubleshooting

### Slow Logging

**Problem**: Logging slows down training

**Solutions**:
```python
# 1. Batch logs
log_buffer = {}
if step % 10 == 0:
    wandb.log(log_buffer, step=step)
    log_buffer.clear()
else:
    log_buffer.update({"metric": value})

# 2. Async logging
wandb.init(settings=wandb.Settings(start_method="thread"))

# 3. Reduce logging frequency
if step % 100 == 0:  # Instead of every step
    wandb.log(...)
```

### Large Tables

**Problem**: Tables consume too much memory/bandwidth

**Solutions**:
```python
# 1. Sample data
sampled_data = random.sample(all_data, k=1000)
table = wandb.Table(columns=columns, data=sampled_data)

# 2. Log incrementally
table = wandb.Table(columns=columns)
for i in range(0, len(data), 100):
    batch = data[i:i+100]
    for row in batch:
        table.add_data(*row)
    if i % 1000 == 0:
        wandb.log({"table": table})
```

### Connection Issues

**Problem**: W&B connection fails or times out

**Solutions**:
```python
# 1. Offline mode
wandb.init(mode="offline")

# Later sync:
# wandb sync wandb/offline-run-xxx

# 2. Retry logic
from wandb.errors import CommError
import time

def log_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            wandb.log(data)
            break
        except CommError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print("Failed to log after retries")
```

---

## Complete GRPO Integration Example

```python
import wandb
from transformers import TrainerCallback
import numpy as np

class GRPOWandBCallback(TrainerCallback):
    """Comprehensive W&B logging for GRPO training"""

    def __init__(self):
        self.rewards_history = []
        self.steps_history = []
        self.generation_samples = []

    def on_train_begin(self, args, state, control, **kwargs):
        # Log training config
        wandb.config.update({
            "output_dir": args.output_dir,
            "num_train_epochs": args.num_train_epochs,
            "save_steps": args.save_steps,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Basic metrics
            wandb.log(logs, step=state.global_step)

            # Reward analysis
            if 'rewards' in logs:
                self.log_reward_analysis(logs['rewards'], state.global_step)

    def log_reward_analysis(self, rewards, step):
        """Detailed reward analysis"""
        wandb.log({
            "rewards/mean": np.mean(rewards),
            "rewards/std": np.std(rewards),
            "rewards/min": np.min(rewards),
            "rewards/max": np.max(rewards),
            "rewards/median": np.median(rewards),
            "rewards/q25": np.percentile(rewards, 25),
            "rewards/q75": np.percentile(rewards, 75),
            "rewards/distribution": wandb.Histogram(rewards),
        }, step=step)

        # Collect for 3D viz
        self.rewards_history.extend(rewards)
        self.steps_history.extend([step] * len(rewards))

        # Create 3D visualization every 100 steps
        if step % 100 == 0 and len(self.rewards_history) > 0:
            self.log_3d_reward_landscape(step)

    def log_3d_reward_landscape(self, step):
        """Create 3D reward landscape"""
        data = [[s, i, r] for i, (s, r) in enumerate(zip(
            self.steps_history[-1000:],  # Last 1000 points
            self.rewards_history[-1000:]
        ))]

        table = wandb.Table(
            columns=["step", "example_id", "reward"],
            data=data
        )

        wandb.log({
            "reward_landscape_3d": wandb.plot_table(
                "wandb/3d-scatter/v0",
                table,
                {"x": "step", "y": "example_id", "z": "reward"},
                {"title": "Reward Landscape"}
            )
        }, step=step)

    def on_save(self, args, state, control, **kwargs):
        """Log checkpoint as artifact"""
        checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"

        artifact = wandb.Artifact(
            name=f"checkpoint-{state.global_step}",
            type="model",
            metadata={
                "step": state.global_step,
                "epoch": state.epoch,
            }
        )
        artifact.add_dir(checkpoint_path)
        wandb.log_artifact(artifact)

# Usage
wandb.init(project="grpo-math", name="qwen-05b-run1")

trainer = GRPOTrainer(
    # ... args
    callbacks=[GRPOWandBCallback()]
)

trainer.train()
wandb.finish()
```

---

## Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B RL Guide](https://wandb.ai/yashkotadia/rl-example)
- [Custom Charts](https://docs.wandb.ai/guides/app/features/custom-charts)
- [3D Visualizations](https://docs.wandb.ai/guides/track/log/plots)
- [W&B Course](https://wandb.ai/site/courses/101/)

---

For more guides, see:
- [Prime Intellect Integration](./PRIME_INTELLECT.md)
- [Google Cloud Storage](./GOOGLE_CLOUD_STORAGE.md)
- [Gradio Deployment](./GRADIO_DEPLOYMENT.md)
