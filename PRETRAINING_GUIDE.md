# THONK Pretraining Guide

## Overview

This guide explains how to pretrain THONK models using high-quality datasets from HuggingFace's curated collection. These are the same datasets used to train successful models like SmolLM, Phi, and others.

## Pretraining Configurations

### 1. Full Pretraining (`configs/pretrain_thonk_base.yaml`)
- **Model Size**: ~110M parameters
- **Training Data**: ~15-20B tokens from multiple sources
- **Estimated Time**: 3-5 days on 8×A100 GPUs
- **Use Case**: Production-ready base model

### 2. Test Configuration (`configs/pretrain_thonk_base_test.yaml`)
- **Model Size**: Same architecture (110M parameters)
- **Training Steps**: 1,000 steps only
- **Use Case**: Testing pipeline, debugging, single GPU experiments

## Dataset Mix

Based on [HuggingFace's LLM pretraining collection](https://huggingface.co/collections/HuggingFaceTB/llm-pretraining-datasets-67caec1306e1ad6659d9c1cc), our pretraining uses:

### Primary Datasets (Full Config)

| Dataset | Weight | Description | Size |
|---------|--------|-------------|------|
| **FineWeb-Edu** | 40% | Educational web content, highly curated | 3.5B tokens |
| **SmolLM-Corpus** | 25% | Synthetic textbooks (Cosmopedia v2) + Python-Edu | 237M tokens |
| **The-Stack-v2** | 20% | High-quality GitHub code | 5.45B tokens |
| **FineMath** | 10% | Mathematical content from CommonCrawl | 48.3M pages |
| **FineWeb** | 5% | General web data for diversity | 52.5B tokens (sampled) |

### Why These Datasets?

1. **FineWeb-Edu**: Provides high-quality educational content that's been filtered for instructional value
2. **SmolLM-Corpus**: Synthetic textbooks provide structured knowledge and improve reasoning
3. **The-Stack-v2**: Code understanding is crucial for modern LLMs
4. **FineMath**: Mathematical reasoning improves logical thinking
5. **FineWeb**: General web text ensures linguistic diversity

## Quick Start

### 1. Test the Pipeline
```bash
# Run a quick test (1000 steps, ~30 minutes on single GPU)
python pretrain_thonk.py configs/pretrain_thonk_base_test.yaml
```

### 2. Full Pretraining
```bash
# Start full pretraining (requires significant compute)
python pretrain_thonk.py configs/pretrain_thonk_base.yaml
```

### 3. Resume from Checkpoint
```bash
# Modify the config file:
# resume_from_checkpoint: "./outputs/thonk-base-pretrain/checkpoint-10000"
python pretrain_thonk.py configs/pretrain_thonk_base.yaml
```

## Hardware Requirements

### Minimum (Test Config)
- GPU: 1× V100 (16GB) or RTX 3090 (24GB)
- RAM: 32GB
- Storage: 50GB for cached data

### Recommended (Full Pretraining)
- GPU: 8× A100 (40GB) or 4× A100 (80GB)
- RAM: 256GB
- Storage: 500GB for full datasets
- Network: Fast internet for streaming datasets

### Memory Optimization Tips
```yaml
# Enable these in your config if running out of memory:
gradient_checkpointing: true  # Saves ~30% memory
per_device_train_batch_size: 1  # Reduce batch size
gradient_accumulation_steps: 32  # Increase accumulation
fp16: true  # Or bf16: true on modern GPUs
```

## Monitoring Training

### Weights & Biases (Recommended)
```yaml
# In your config file:
report_to: "wandb"
wandb_project: "THONK-Pretraining"
wandb_name: "thonk-base-v1"
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir outputs/thonk-base-pretrain

# View at http://localhost:6006
```

## Expected Results

### Training Metrics (Approximate)
- **Step 10k**: Loss ~3.5, Perplexity ~33
- **Step 50k**: Loss ~2.8, Perplexity ~16
- **Step 100k**: Loss ~2.4, Perplexity ~11
- **Step 500k**: Loss ~2.0, Perplexity ~7.4

### Validation Performance
After full pretraining, expect:
- **HellaSwag**: ~45-50% accuracy
- **MMLU**: ~25-30% accuracy  
- **HumanEval**: ~10-15% pass@1

## Customizing the Dataset Mix

### Adding New Datasets
```yaml
datasets:
  - name: "your-org/your-dataset"
    split: "train"
    streaming: true  # Use for large datasets
    weight: 0.1  # Relative weight in the mix
    data_files: ["data/specific_files/*"]  # Optional filtering
    sample_ratio: 0.01  # Optional subsampling
```

### Adjusting Weights
Weights are normalized automatically. For example:
- Technical model: Increase code/math weights
- Creative model: Increase FineWeb/story datasets
- Instruction model: Add more instruction datasets

## Common Issues

### Out of Memory
```yaml
# Reduce memory usage:
per_device_train_batch_size: 1
gradient_checkpointing: true
block_size: 1024  # Shorter sequences
fp16: true  # Mixed precision
```

### Slow Data Loading
```yaml
# Increase workers and buffers:
preprocessing_num_workers: 16
dataloader_num_workers: 8
stream_buffer_size: 50000
```

### Unstable Training
```yaml
# Stabilize training:
learning_rate: 1e-4  # Lower LR
warmup_steps: 5000  # Longer warmup
max_grad_norm: 0.5  # Stronger clipping
```

## Next Steps

After pretraining:
1. **Fine-tune** on specific tasks (see `train_thonk.py`)
2. **Evaluate** on benchmarks
3. **Optimize** for inference (quantization, pruning)
4. **Deploy** using HuggingFace Transformers

## References

- [HuggingFace Pretraining Datasets](https://huggingface.co/collections/HuggingFaceTB/llm-pretraining-datasets-67caec1306e1ad6659d9c1cc)
- [SmolLM Paper](https://huggingface.co/blog/smollm)
- [FineWeb Blog Post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
