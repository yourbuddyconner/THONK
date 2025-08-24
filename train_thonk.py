#!/usr/bin/env python3
"""
Training script for THONK (Thinking Hierarchically: Optimized Neural Knowledge).

Train THONK models on text data using HuggingFace's Trainer API,
with support for instruction tuning and efficient learning from minimal data.
"""

import os
import sys
import logging
import json
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from transformers import (
    GPT2TokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset, Dataset
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.thonk_model import THONKConfig, THONK

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config we are going to train."""
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size of the model"}
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads"}
    )
    H_layers: int = field(
        default=4,
        metadata={"help": "Number of H-level (high-level) layers"}
    )
    L_layers: int = field(
        default=4,
        metadata={"help": "Number of L-level (low-level) layers"}
    )
    H_cycles: int = field(
        default=2,
        metadata={"help": "Number of H-level cycles"}
    )
    L_cycles: int = field(
        default=2,
        metadata={"help": "Number of L-level cycles"}
    )
    max_position_embeddings: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    use_act: bool = field(
        default=False,
        metadata={"help": "Whether to use Adaptive Computation Time"}
    )
    halt_max_steps: int = field(
        default=1,
        metadata={"help": "Maximum ACT steps (1 = no ACT)"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input to our model."""
    
    dataset_name: str = field(
        default="tatsu-lab/alpaca",
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "The split to use for training"}
    )
    validation_split: Optional[str] = field(
        default=None,
        metadata={"help": "The split to use for validation"}
    )
    max_train_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "For debugging, truncate the number of training examples"}
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={"help": "For debugging, truncate the number of evaluation examples"}
    )
    block_size: int = field(
        default=512,
        metadata={"help": "Optional input sequence length after tokenization"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for preprocessing"}
    )


def format_alpaca(example):
    """Format Alpaca dataset examples into text."""
    if example.get("input", ""):
        text = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        text = f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""
    
    return {"text": text}


def parse_yaml_config(yaml_path: str, parser: HfArgumentParser) -> tuple:
    """Parse YAML config file and distribute parameters to dataclasses."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Write the YAML config as a flat JSON file that HfArgumentParser can understand
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    
    try:
        # Parse the temporary JSON file
        model_args, data_args, training_args = parser.parse_json_file(json_file=temp_path)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
    
    return model_args, data_args, training_args


def main():
    """Main training function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # Check if we have a config file
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        config_file = os.path.abspath(sys.argv[1])
        if config_file.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=config_file)
        else:  # YAML file
            model_args, data_args, training_args = parse_yaml_config(config_file, parser)
    elif "--config" in sys.argv:
        # Handle --config flag
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = os.path.abspath(sys.argv[config_idx + 1])
            # Remove --config and the file path from sys.argv so HfArgumentParser doesn't complain
            sys.argv.pop(config_idx)  # Remove --config
            sys.argv.pop(config_idx)  # Remove the file path
            
            if config_file.endswith(".json"):
                model_args, data_args, training_args = parser.parse_json_file(json_file=config_file)
            else:  # YAML file
                model_args, data_args, training_args = parse_yaml_config(config_file, parser)
        else:
            raise ValueError("--config flag provided but no config file specified")
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Initialize W&B with THONK project name if enabled
    if "wandb" in training_args.report_to:
        wandb.init(
            project="THONK",
            name=training_args.run_name if training_args.run_name else None,
            config={
                "model_args": vars(model_args),
                "data_args": vars(data_args),
                "training_args": vars(training_args),
            },
            reinit=True,  # Allow multiple runs in same script
        )
    
    # Setup logging
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Log basic information
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    if data_args.dataset_name == "tatsu-lab/alpaca":
        # Load Alpaca dataset
        dataset = load_dataset(data_args.dataset_name)
        
        # Format the dataset
        dataset = dataset.map(format_alpaca, num_proc=data_args.preprocessing_num_workers)
        
        # Select train/validation splits
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        
        # Create validation split if not provided
        if data_args.validation_split is None:
            # Use last 10% as validation
            split_idx = int(len(train_dataset) * 0.9)
            eval_dataset = train_dataset.select(range(split_idx, len(train_dataset)))
            train_dataset = train_dataset.select(range(split_idx))
        else:
            eval_dataset = dataset[data_args.validation_split]
        
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
    
    elif data_args.dataset_name == "wikitext":
        # Load WikiText dataset
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name or "wikitext-2-raw-v1")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
    
    else:
        # Generic dataset loading
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        train_dataset = dataset[data_args.train_split]
        eval_dataset = dataset[data_args.validation_split] if data_args.validation_split else None
        
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        if eval_dataset and data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    # Tokenize datasets
    def tokenize_function(examples):
        """Tokenize text examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=data_args.block_size,
        )
    
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    
    tokenized_eval = None
    if eval_dataset:
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset",
        )
    
    # Initialize model
    logger.info("Initializing THONK model...")
    
    if model_args.model_name_or_path:
        # Load from checkpoint
        config = THONKConfig.from_pretrained(model_args.model_name_or_path)
        model = THONK.from_pretrained(model_args.model_name_or_path)
    else:
        # Create new model
        config = THONKConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=model_args.hidden_size,
            num_heads=model_args.num_heads,
            H_layers=model_args.H_layers,
            L_layers=model_args.L_layers,
            H_cycles=model_args.H_cycles,
            L_cycles=model_args.L_cycles,
            max_position_embeddings=model_args.max_position_embeddings,
            use_act=model_args.use_act,
            halt_max_steps=model_args.halt_max_steps,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        model = THONK(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    if training_args.do_train:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        
        # Save model
        trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluate
    if training_args.do_eval and tokenized_eval:
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()
        
        # Calculate perplexity
        try:
            perplexity = torch.exp(torch.tensor(metrics["eval_loss"])).item()
            metrics["perplexity"] = perplexity
        except OverflowError:
            metrics["perplexity"] = float("inf")
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
    
    # Push to hub if requested
    if training_args.push_to_hub:
        trainer.push_to_hub()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
