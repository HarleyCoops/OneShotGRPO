# Prime Intellect Integration Guide

##  Table of Contents
- [Overview](#overview)
- [What is Prime Intellect?](#what-is-prime-intellect)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using the AQuA-RAT Environment](#using-the-aqua-rat-environment)
- [Custom Environments](#custom-environments)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Overview

Prime Intellect provides a scalable, fault-tolerant framework for reinforcement learning training of large language models. This guide shows you how to integrate Prime Intellect with your GRPO training pipeline.

### Key Benefits

- ** Scalability**: Train across multiple GPUs and nodes
- ** Fault Tolerance**: Automatic recovery from failures
- ** Environment Hub**: Pre-built RL environments
- ** Verifiers**: Modular reward functions
- ** Performance**: Optimized for distributed training

---

## What is Prime Intellect?

Prime Intellect consists of three main components:

### 1. **Prime-RL Framework**
- Async reinforcement learning at scale
- FSDP2 training with vLLM inference
- Rayless multi-node deployment
- Native integration with Verifiers

### 2. **Environments Hub**
- Community-driven RL environments
- Pre-built tasks for various domains
- Easy-to-use API
- Standardized reward functions

### 3. **Verifiers Library**
- Modular reward components
- Compatible with any OpenAI-like API
- Supports both RL training and evaluation
- Async GRPO implementation

---

## Installation

### Option 1: Quick Install (Recommended)

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

### Option 2: Manual Install

```bash
# Clone repository
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### System Requirements

- **GPU**: Minimum one NVIDIA GPU (RTX 3090/4090/5090, A100, H100, H200, B200)
- **Python**: 3.12
- **Flash Attention**: Required for optimal performance

### Verify Installation

```bash
# Test SFT trainer
uv run sft @ configs/debug/sft/train.toml

# Test RL trainer
uv run trainer @ configs/debug/rl/train.toml

# Test inference server
uv run inference @ configs/debug/infer.toml

# Test orchestrator
uv run orchestrator @ configs/debug/orch.toml

# Test evaluation
uv run eval @ configs/debug/eval.toml
```

---

## Quick Start

### Example 1: Simple Evaluation

```bash
# Evaluate GPT-4o-mini on AQuA-RAT (25 examples)
uv run vf-eval harleycooper/nanochatAquaRat -m gpt-4o-mini -n 25
```

### Example 2: GRPO Training

```bash
# Train with default GRPO config
uv run vf-rl @ configs/rl/nanochat.toml
```

### Example 3: Custom Training Script

```python
from prime_rl import GRPOTrainer, Environment

# Load environment
env = Environment.from_hub("harleycooper/nanochatAquaRat", {
    "num_train_examples": 2000,
    "num_eval_examples": 254,
    "seed": 42
})

# Configure trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    environment=env,
    learning_rate=2e-5,
    rollouts_per_example=8,
    max_steps=400
)

# Train
trainer.train()
```

---

## Using the AQuA-RAT Environment

### Environment Details

- **Hub ID**: `harleycooper/nanochatAquaRat`
- **Task**: Single-turn algebra questions
- **Format**: Multiple choice (A-E)
- **Dataset**: ~97k algebra word problems from deepmind/aqua_rat
- **Scoring**: Categorical accuracy

### Configuration

Create `configs/rl/aquarat.toml`:

```toml
model = "Qwen/Qwen2.5-7B-Instruct"

[env]
id = "harleycooper/nanochatAquaRat"

[env.args]
num_train_examples = 2000  # Use subset for faster iteration
num_eval_examples = 254
seed = 42
system_prompt = "You are an algebra tutor. Choose the correct answer (A-E)."
train_split = "train"
eval_split = "validation"
include_rationale_metadata = true

[trainer.args]
learning_rate = 2e-5
rollouts_per_example = 8
max_steps = 400
save_steps = 100
logging_steps = 10
```

### Run Training

```bash
uv run vf-rl @ configs/rl/aquarat.toml
```

### Environment Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `system_prompt` | str | Algebra instruction | Prepended system message |
| `train_split` | str | "train" | Dataset split for training |
| `eval_split` | str\|null | "validation" | Split for evaluation |
| `num_train_examples` | int | -1 | Limit training examples (-1 = all) |
| `num_eval_examples` | int | -1 | Limit eval examples (-1 = all) |
| `seed` | int\|null | 42 | Shuffle seed for reproducibility |
| `include_rationale_metadata` | bool | true | Include human rationale |
| `data_dir` | str\|null | null | Local data directory |
| `cache_dir` | str\|null | null | HuggingFace cache override |

### Reward Structure

The environment uses two reward signals:

1. **Exact Match** (weight: 1.0)
   - Full credit for correct letter (A-E)
   - Binary: 1.0 or 0.0

2. **Format Bonus** (weight: 0.1)
   - Partial credit for valid letter output
   - Encourages proper formatting

### Metrics

Monitor these metrics during training:

- `reward`: Weighted total reward
- `exact_match_reward`: Raw correctness signal
- `format_reward`: Valid letter bonus
- `rl/acc`: Training accuracy

---

## Custom Environments

### Creating Your Own Environment

```python
from verifiers import Environment, Parser, Rubric

class MathEnvironment(Environment):
    def __init__(self, dataset_name, split="train"):
        # Load dataset
        self.dataset = load_dataset(dataset_name, split=split)

        # Define parser (extracts answer from model output)
        self.parser = Parser(
            extract_fn=self.extract_answer,
            validate_fn=self.validate_format
        )

        # Define rubric (reward function)
        self.rubric = Rubric([
            ("correctness", self.check_correctness, 1.0),
            ("format", self.check_format, 0.1)
        ])

    def extract_answer(self, text):
        # Extract answer from XML tags
        return text.split("<answer>")[1].split("</answer>")[0].strip()

    def validate_format(self, text):
        # Check if response has proper format
        return "<answer>" in text and "</answer>" in text

    def check_correctness(self, prompt, completion, metadata):
        extracted = self.parser.extract(completion)
        return 1.0 if extracted == metadata["answer"] else 0.0

    def check_format(self, prompt, completion, metadata):
        return 0.1 if self.parser.validate(completion) else 0.0

# Use custom environment
env = MathEnvironment("openai/gsm8k", split="train")
```

### Registering Environment to Hub

```python
from verifiers import register_environment

# Register for community use
register_environment(
    name="yourname/custom-math-env",
    environment_class=MathEnvironment,
    description="Custom math reasoning environment",
    tags=["math", "reasoning", "grpo"]
)
```

---

## Advanced Configuration

### Multi-GPU Training

```toml
[trainer.args]
# Distributed training
num_gpus = 4
per_device_batch_size = 2
gradient_accumulation_steps = 4

# FSDP configuration
fsdp = "full_shard"
fsdp_transformer_layer_cls_to_wrap = "Qwen2DecoderLayer"
```

### LoRA Training

```toml
[peft]
type = "lora"
r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Custom Reward Weights

```toml
[env.reward_weights]
correctness = 2.0
format = 0.5
reasoning_quality = 1.0
```

### Logging and Monitoring

```bash
# Enable W&B
uv run wandb login

# Enable HuggingFace
uv run hf auth login

# Run with monitoring
uv run vf-rl @ configs/rl/aquarat.toml --wandb-project "my-grpo-project"
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce batch size or use gradient accumulation

```toml
[trainer.args]
per_device_batch_size = 1  # Reduce from 2
gradient_accumulation_steps = 8  # Increase from 4
```

#### 2. vLLM Initialization Fails

**Solution**: Adjust GPU memory allocation

```toml
[inference]
vllm_gpu_memory_utilization = 0.2  # Reduce from 0.3
```

#### 3. Environment Not Found

**Solution**: Ensure environment ID is correct

```bash
# List available environments
uv run vf-env list

# Check specific environment
uv run vf-env info harleycooper/nanochatAquaRat
```

#### 4. Slow Training

**Solutions**:
- Enable vLLM: `use_vllm = true`
- Use bf16: `bf16 = true`
- Increase generation batch size: `generation_batch_size = 32`
- Reduce number of rollouts: `rollouts_per_example = 4`

#### 5. Distributed Training Hangs

**Solution**: Check network configuration

```bash
# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Run with logging
uv run vf-rl @ configs/rl/aquarat.toml --log-level DEBUG
```

### Getting Help

- **GitHub Issues**: [prime-rl/issues](https://github.com/PrimeIntellect-ai/prime-rl/issues)
- **Discord**: [Prime Intellect Community](https://discord.gg/primeintellect)
- **Documentation**: [docs.primeintellect.ai](https://docs.primeintellect.ai/)

---

## Resources

### Documentation
- [Prime Intellect Docs](https://docs.primeintellect.ai/)
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
- [Environment Hub](https://www.primeintellect.ai/blog/environments)

### Papers
- [INTELLECT-2 Release](https://www.primeintellect.ai/blog/intellect-2-release)
- [Distributed RL Training](https://www.primeintellect.ai/blog/intellect-2)

### Examples
- [Reverse Text](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/reverse)
- [Wordle Multi-Turn](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/wordle)
- [Alphabet Sort](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/alphabet)

### Community
- [GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [Blog](https://www.primeintellect.ai/blog)
- [Twitter](https://twitter.com/PrimeIntellect)

---

## Integration with OneShotGRPO

### Converting GSM8K to Prime Intellect

```python
# Create Prime Intellect compatible environment from GSM8K
from oneshot_grpo.data import load_gsm8k, format_for_grpo
from verifiers import Environment

class GSM8KEnvironment(Environment):
    def __init__(self, split="train", num_examples=None):
        # Use existing OneShotGRPO data pipeline
        self.dataset = load_gsm8k(split)
        if num_examples:
            self.dataset = self.dataset.select(range(num_examples))

        # Convert to Prime Intellect format
        self.dataset = format_for_grpo(self.dataset)

        # Import reward functions from OneShotGRPO
        from oneshot_grpo.rewards import (
            correctness_reward,
            format_reward,
            numeric_reward
        )

        self.rubric = Rubric([
            ("correctness", correctness_reward, 2.0),
            ("format", format_reward, 1.0),
            ("numeric", numeric_reward, 0.5)
        ])
```

### Best Practices

1. **Start Small**: Test with `num_train_examples=100` first
2. **Monitor Early**: Check W&B after first 50 steps
3. **Checkpoint Often**: Set `save_steps=50` for experiments
4. **Compare Baselines**: Run both HF and Prime Intellect versions
5. **Scale Gradually**: Single GPU → Multi-GPU → Multi-node

---

## Next Steps

1.  Install Prime RL
2.  Test with example environments
3.  Run AQuA-RAT training
4.  Create custom environment
5.  Scale to multi-GPU
6.  Deploy to production

For more details on other integrations, see:
- [Google Cloud Storage](./GOOGLE_CLOUD_STORAGE.md)
- [Weights & Biases](./WANDB_VISUALIZATION.md)
- [Gradio Deployment](./GRADIO_DEPLOYMENT.md)
