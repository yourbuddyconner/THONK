#!/usr/bin/env python3
"""
THONK Pretraining Script
Handles multiple datasets with streaming and weighted sampling
"""

import os
import sys
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from itertools import chain
import random

import torch
from datasets import load_dataset, interleave_datasets, Dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

from models.thonk_model import THONK, THONKConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model configuration file"}
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Pretrained tokenizer name or path"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    datasets_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to datasets configuration in the YAML file"}
    )
    block_size: int = field(
        default=2048,
        metadata={"help": "Block size for tokenization"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of processes for preprocessing"}
    )
    stream_buffer_size: int = field(
        default=10000,
        metadata={"help": "Buffer size for streaming datasets"}
    )
    shuffle_buffer_size: int = field(
        default=100000,
        metadata={"help": "Shuffle buffer size for streaming datasets"}
    )
    eval_dataset_size: int = field(
        default=10000,
        metadata={"help": "Number of samples to use for evaluation"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets"}
    )


class PretrainingDataset:
    """Handles loading and mixing multiple datasets for pretraining."""
    
    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        tokenizer,
        block_size: int,
        stream_buffer_size: int = 10000,
        shuffle_buffer_size: int = 100000,
    ):
        self.datasets_config = datasets_config
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stream_buffer_size = stream_buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        
    def load_and_process_dataset(self, config: Dict[str, Any]):
        """Load a single dataset and prepare it for training."""
        logger.info(f"Loading dataset: {config['name']}")
        
        # Load dataset with streaming
        dataset_kwargs = {
            "path": config["name"],
            "split": config.get("split", "train"),
            "streaming": config.get("streaming", True),
        }
        
        # Add config name if specified (for datasets with multiple configs)
        if "config" in config:
            dataset_kwargs["name"] = config["config"]
        
        # Add data files if specified (for filtering specific subsets)
        if "data_files" in config:
            dataset_kwargs["data_files"] = config["data_files"]
            
        dataset = load_dataset(**dataset_kwargs)
        
        # Apply sampling if specified
        if "sample_ratio" in config and config["sample_ratio"] < 1.0:
            # Note: For streaming datasets, this affects iteration probability
            dataset = dataset.shuffle(seed=42)
            # Sampling will be handled during interleaving
            
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize text examples."""
        # Handle different column names
        text_column = None
        for col in ["text", "content", "code", "instruction", "output"]:
            if col in examples:
                text_column = col
                break
        
        if text_column is None:
            # Try to combine instruction and output for instruction datasets
            if "instruction" in examples and "output" in examples:
                texts = [f"{inst}\n\n{out}" for inst, out in 
                        zip(examples["instruction"], examples["output"])]
            else:
                raise ValueError(f"No text column found in dataset: {examples.keys()}")
        else:
            texts = examples[text_column]
            
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
            return_tensors=None,
        )
        
        # Add labels (same as input_ids for language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def group_texts(self, examples):
        """Group texts into chunks of block_size."""
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the last chunk if it's smaller than block_size
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
            
        # Split by chunks of block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        
        return result
    
    def create_train_dataset(self):
        """Create the interleaved training dataset."""
        datasets = []
        weights = []
        
        # Check if we have any streaming datasets
        has_streaming = any(config.get("streaming", True) for config in self.datasets_config)
        
        for config in self.datasets_config:
            dataset = self.load_and_process_dataset(config)
            
            # Convert non-streaming datasets to streaming if needed
            if has_streaming and not hasattr(dataset, '__iter__'):
                # Convert regular Dataset to IterableDataset
                from datasets import IterableDataset
                dataset = dataset.to_iterable_dataset()
            
            # Tokenize the dataset
            if hasattr(dataset, 'column_names'):
                column_names = list(dataset.column_names) if dataset.column_names else []
            else:
                column_names = []
            dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=column_names if column_names else None,
            )
            
            # Group texts for efficient training
            dataset = dataset.map(
                self.group_texts,
                batched=True,
            )
            
            datasets.append(dataset)
            weights.append(config.get("weight", 1.0))
            
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        logger.info(f"Interleaving {len(datasets)} datasets with weights: {weights}")
        
        # Interleave datasets with specified weights
        train_dataset = interleave_datasets(
            datasets,
            probabilities=weights,
            seed=42,
            stopping_strategy="all_exhausted",
        )
        
        # Shuffle the interleaved dataset
        train_dataset = train_dataset.shuffle(
            seed=42,
            buffer_size=self.shuffle_buffer_size
        )
        
        return train_dataset
    
    def create_eval_dataset(self, size: int = 10000):
        """Create evaluation dataset by sampling from all datasets."""
        eval_samples = []
        samples_per_dataset = size // len(self.datasets_config)
        
        for config in self.datasets_config:
            # Load validation split if available, otherwise use train
            config_eval = config.copy()
            config_eval["split"] = "validation" if "validation" in config.get("available_splits", []) else "train"
            config_eval["streaming"] = False  # Load eval data fully
            
            dataset = self.load_and_process_dataset(config_eval)
            
            # Tokenize
            if hasattr(dataset, 'column_names'):
                column_names = list(dataset.column_names) if dataset.column_names else []
            else:
                column_names = []
            dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=column_names if column_names else None,
            )
            
            # Sample - only for non-streaming datasets
            if not config.get('streaming', True) and isinstance(dataset, Dataset):
                dataset_len = len(dataset)
                if dataset_len > samples_per_dataset:
                    indices = random.sample(range(dataset_len), samples_per_dataset)
                    dataset = dataset.select(indices)
            
            eval_samples.extend(dataset)
        
        # Create dataset from samples
        eval_dataset = Dataset.from_list(eval_samples[:size])
        
        return eval_dataset


def main():
    # Parse arguments  
    from typing import cast
    parser = HfArgumentParser(cast(tuple, (ModelArguments, DataArguments, TrainingArguments)))
    
    # Load from YAML config if provided
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        with open(sys.argv[1], "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Split config into appropriate dataclasses
        model_args_dict = {}
        data_args_dict = {}
        training_args_dict = {}
        
        for key, value in config_dict.items():
            if key in ["model_name_or_path", "tokenizer_name", "use_fast_tokenizer", "cache_dir"]:
                model_args_dict[key] = value
            elif key in ["datasets", "block_size", "preprocessing_num_workers", 
                        "stream_buffer_size", "shuffle_buffer_size", "eval_dataset_size"]:
                if key == "datasets":
                    data_args_dict["datasets_config"] = value
                else:
                    data_args_dict[key] = value
            elif key not in ["datasets", "H_layers", "L_layers", "H_cycles", "L_cycles", 
                            "hidden_size", "num_heads", "use_act", "halt_max_steps",
                            "max_position_embeddings", "vocab_size", "rope_theta", 
                            "rms_norm_eps", "expansion"]:
                training_args_dict[key] = value
        
        model_args = ModelArguments(**model_args_dict)
        data_args = DataArguments(**data_args_dict)
        training_args = TrainingArguments(**training_args_dict)
        
        # Store model config separately
        model_config_dict = {
            k: v for k, v in config_dict.items() 
            if k in ["vocab_size", "hidden_size", "num_heads", "H_layers", "L_layers",
                    "H_cycles", "L_cycles", "max_position_embeddings", "use_act",
                    "halt_max_steps", "rope_theta", "rms_norm_eps", "expansion"]
        }
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_config_dict = None
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    if model_args.model_name_or_path:
        # Load from checkpoint
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        model = THONK.from_pretrained(model_args.model_name_or_path)
    else:
        # Create new model
        logger.info("Creating new model from scratch")
        if model_config_dict:
            config = THONKConfig(**model_config_dict)
        else:
            config = THONKConfig()
        model = THONK(config)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model size: {total_params:,} total parameters, {trainable_params:,} trainable")
    logger.info(f"Model memory: ~{total_params * 4 / 1024**3:.2f} GB (fp32)")
    
    # Create datasets
    if data_args.datasets_config and isinstance(data_args.datasets_config, list):
        logger.info("Creating pretraining datasets...")
        dataset_handler = PretrainingDataset(
            datasets_config=data_args.datasets_config,
            tokenizer=tokenizer,
            block_size=data_args.block_size,
            stream_buffer_size=data_args.stream_buffer_size,
            shuffle_buffer_size=data_args.shuffle_buffer_size,
        )
        
        train_dataset = dataset_handler.create_train_dataset()
        eval_dataset = dataset_handler.create_eval_dataset(data_args.eval_dataset_size)
    else:
        raise ValueError("No datasets configuration provided!")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    
    # Check for checkpoint to resume from
    last_checkpoint = None
    if training_args.output_dir and os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training from {last_checkpoint}")
    
    # Train
    if training_args.do_train:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Save final model
        trainer.save_model()
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluate
    if training_args.do_eval:
        logger.info("Running evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
