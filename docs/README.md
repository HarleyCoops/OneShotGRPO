# OneShotGRPO Documentation

Welcome to the comprehensive documentation for OneShotGRPO! This guide will help you train, monitor, and deploy small language models using GRPO (Generative Reinforcement Policy Optimization).

##  Documentation Index

###  Getting Started

- **[Educational GRPO Notebook](../EducationalGRPO.ipynb)**: Start here! A comprehensive, step-by-step notebook that teaches you GRPO from scratch.
- **[CLAUDE.md](../CLAUDE.md)**: Quick reference guide for the codebase structure and core concepts.
- **[README2.md](../README2.md)**: Original project documentation with additional context.

###  Integration Guides

| Guide | Description | Level |
|-------|-------------|-------|
| [Prime Intellect Integration](./PRIME_INTELLECT.md) | Scale training across multiple GPUs/nodes with Prime RL | Advanced |
| [Google Cloud Storage](./GOOGLE_CLOUD_STORAGE.md) | Persistent checkpoint storage and management | Intermediate |
| [Weights & Biases Visualization](./WANDB_VISUALIZATION.md) | Advanced monitoring with 3D charts | Intermediate |
| [Gradio Deployment](./GRADIO_DEPLOYMENT.md) | Deploy chat interfaces to HF Spaces | Beginner |

---

##  Quick Navigation

### By Use Case

**I want to...**

- **Learn GRPO from scratch** → Start with [Educational Notebook](../EducationalGRPO.ipynb)
- **Scale to multiple GPUs** → See [Prime Intellect Guide](./PRIME_INTELLECT.md)
- **Save checkpoints reliably** → Read [GCS Guide](./GOOGLE_CLOUD_STORAGE.md)
- **Monitor training deeply** → Check [W&B Guide](./WANDB_VISUALIZATION.md)
- **Deploy a chat demo** → Follow [Gradio Guide](./GRADIO_DEPLOYMENT.md)
- **Understand the code** → Review [CLAUDE.md](../CLAUDE.md)

### By Experience Level

**Beginner:**
1. [Educational Notebook](../EducationalGRPO.ipynb) (Start here!)
2. [Gradio Deployment](./GRADIO_DEPLOYMENT.md) (Deploy your model)
3. [README2.md](../README2.md) (Learn about the project)

**Intermediate:**
1. [Google Cloud Storage](./GOOGLE_CLOUD_STORAGE.md) (Better checkpointing)
2. [Weights & Biases](./WANDB_VISUALIZATION.md) (Advanced monitoring)
3. [CLAUDE.md](../CLAUDE.md) (Code deep dive)

**Advanced:**
1. [Prime Intellect](./PRIME_INTELLECT.md) (Distributed training)
2. Source code in `src/oneshot_grpo/`
3. Custom environments and reward functions

---

##  Documentation Overview

### 1. Prime Intellect Integration

**File**: [PRIME_INTELLECT.md](./PRIME_INTELLECT.md)

Learn how to use Prime Intellect's distributed RL framework:
- Installation and setup
- Using pre-built environments (AQuA-RAT)
- Creating custom environments
- Multi-GPU training configuration
- Fault-tolerant training at scale

**Best for**: Teams needing to scale training beyond a single GPU, or those wanting access to pre-built RL environments.

### 2. Google Cloud Storage Integration

**File**: [GOOGLE_CLOUD_STORAGE.md](./GOOGLE_CLOUD_STORAGE.md)

Everything about checkpoint persistence:
- GCS setup and authentication
- Automatic checkpoint uploading
- Resuming from saved checkpoints
- Cost optimization strategies
- Lifecycle management

**Best for**: Anyone training for >2 hours or needing guaranteed checkpoint persistence.

### 3. Weights & Biases Visualization

**File**: [WANDB_VISUALIZATION.md](./WANDB_VISUALIZATION.md)

Advanced experiment tracking and visualization:
- Real-time metric logging
- 3D reward landscape plots
- Policy evolution visualization
- Custom dashboards
- Hyperparameter sweeps

**Best for**: Researchers wanting deep insights into training dynamics or comparing multiple runs.

### 4. Gradio Deployment

**File**: [GRADIO_DEPLOYMENT.md](./GRADIO_DEPLOYMENT.md)

Build and deploy chat interfaces:
- Quick chat interface creation
- HuggingFace Hub integration
- Deploying to HF Spaces
- Production considerations
- Custom themes and features

**Best for**: Anyone wanting to demo their trained model with a user-friendly interface.

---

##  Learning Paths

### Path 1: Quick Start (2-4 hours)

Perfect for getting your first GRPO model trained and deployed:

1. **Setup** (30 min)
   - Open [Educational Notebook](../EducationalGRPO.ipynb) in Colab
   - Get a GPU runtime (A100 recommended)
   - Install dependencies

2. **Training** (1-2 hours)
   - Follow notebook sections 1-7
   - Train on 1,000 GSM8K examples
   - Monitor with basic W&B

3. **Deployment** (30 min)
   - Push model to HuggingFace Hub
   - Create Gradio interface (Section 11)
   - Test with sample questions

4. **Result**: A working math reasoning model with chat interface!

### Path 2: Production Setup (1-2 days)

For serious projects requiring robust infrastructure:

1. **Day 1 Morning: Core Training**
   - Complete Educational Notebook (full dataset)
   - Set up [Google Cloud Storage](./GOOGLE_CLOUD_STORAGE.md)
   - Configure automatic checkpoint backups

2. **Day 1 Afternoon: Monitoring**
   - Implement [W&B visualization](./WANDB_VISUALIZATION.md)
   - Set up custom dashboards
   - Create 3D reward landscapes

3. **Day 2 Morning: Scaling** (Optional)
   - Set up [Prime Intellect](./PRIME_INTELLECT.md)
   - Configure multi-GPU training
   - Test fault tolerance

4. **Day 2 Afternoon: Deployment**
   - Create production [Gradio app](./GRADIO_DEPLOYMENT.md)
   - Deploy to HF Spaces with GPU
   - Set up monitoring and rate limiting

5. **Result**: Production-ready GRPO training pipeline!

### Path 3: Research Deep Dive (Ongoing)

For researchers extending GRPO or exploring RL:

1. **Week 1: Understanding**
   - Study GRPO paper and implementation
   - Read [CLAUDE.md](../CLAUDE.md) thoroughly
   - Examine source code in `src/`
   - Run experiments with different rewards

2. **Week 2: Experimentation**
   - Implement custom reward functions
   - Try different datasets
   - Use [W&B sweeps](./WANDB_VISUALIZATION.md#hyperparameter-sweeps)
   - Compare with baselines

3. **Week 3: Scaling**
   - Set up [Prime Intellect](./PRIME_INTELLECT.md)
   - Create custom environments
   - Scale to larger models
   - Optimize hyperparameters

4. **Week 4: Publication**
   - Write model cards
   - Create visualizations
   - Deploy demo apps
   - Share results

---

##  External Resources

### Official Documentation
- [TRL Documentation](https://huggingface.co/docs/trl)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Gradio Documentation](https://gradio.app/docs)
- [W&B Documentation](https://docs.wandb.ai/)
- [Prime Intellect Docs](https://docs.primeintellect.ai/)

### Research Papers
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [GSM8K Dataset](https://arxiv.org/abs/2110.14168)

### Community
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [TRL GitHub](https://github.com/huggingface/trl)
- [Prime Intellect Discord](https://discord.gg/primeintellect)

---

##  Getting Help

### Troubleshooting

Each guide has a dedicated troubleshooting section:
- [Prime Intellect Troubleshooting](./PRIME_INTELLECT.md#troubleshooting)
- [GCS Troubleshooting](./GOOGLE_CLOUD_STORAGE.md#troubleshooting)
- [W&B Troubleshooting](./WANDB_VISUALIZATION.md#troubleshooting)
- [Gradio Troubleshooting](./GRADIO_DEPLOYMENT.md#troubleshooting)

### Common Issues

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size, use gradient accumulation, or enable 8-bit quantization

**Issue**: Training too slow
- **Solution**: Enable vLLM, use bf16 precision, or scale to multiple GPUs with Prime Intellect

**Issue**: Checkpoints lost after disconnect
- **Solution**: Set up Google Cloud Storage integration

**Issue**: Can't monitor training well
- **Solution**: Enable Weights & Biases with custom dashboards

### Support Channels

1. **GitHub Issues**: [Report bugs or request features](https://github.com/HarleyCoops/OneShotGRPO/issues)
2. **Discussions**: [Ask questions or share ideas](https://github.com/HarleyCoops/OneShotGRPO/discussions)
3. **Email**: [Contact maintainers](mailto:your-email@example.com)

---

##  Contributing

We welcome contributions! Here's how:

1. **Documentation**: Found a typo or want to clarify something? Edit the docs!
2. **Code**: Improved a reward function? Created a new environment? Submit a PR!
3. **Examples**: Trained a cool model? Share your notebook!
4. **Guides**: Found a better way to do something? Write a guide!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

##  Citation

If you use OneShotGRPO in your research, please cite:

```bibtex
@misc{oneshotgrpo,
  title={OneShotGRPO: Educational Framework for GRPO Training},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/HarleyCoops/OneShotGRPO}}
}
```

---

##  License

This project is licensed under [LICENSE](../LICENSE).

Base model (Qwen) and other components have their own licenses. See individual files for details.

---

##  Acknowledgments

- **Base Model**: Qwen Team
- **Dataset**: OpenAI (GSM8K)
- **Frameworks**: HuggingFace (TRL, Transformers), vLLM, Gradio, W&B
- **Inspiration**: Will Brown's GRPO demo
- **Community**: All contributors and users!

---

**Happy Training! **

Start with the [Educational Notebook](../EducationalGRPO.ipynb) and build amazing math reasoning models!
