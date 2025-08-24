#!/usr/bin/env python3
"""
Quick test training script for THONK.
This script runs a minimal training test with 1000 samples.
"""

import os
import sys
import torch
from transformers import (
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.thonk_model import THONKConfig, THONK


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


def main():
    """Run quick training test."""
    print("=" * 60)
    print("THONK - Quick Training Test")
    print("=" * 60)
    
    # Set seed
    set_seed(42)
    
    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # Load dataset
    print("\n2. Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")
    
    # Format and select subset
    dataset = dataset.map(format_alpaca)
    train_dataset = dataset["train"].select(range(1000))  # 1000 samples
    eval_dataset = dataset["train"].select(range(1000, 1100))  # 100 samples for eval
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    
    # Show sample
    print("\n   Sample data:")
    sample = train_dataset[0]
    print(f"   {sample['text'][:200]}...")
    
    # Tokenize
    print("\n3. Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )
    
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=eval_dataset.column_names,
    )
    
    # Initialize model
    print("\n4. Initializing THONK model...")
    config = THONKConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,  # Small model for testing
        num_heads=4,
        H_layers=2,
        L_layers=2,
        H_cycles=1,
        L_cycles=1,
        max_position_embeddings=512,
        use_act=True,  # ACT enabled - it's what makes THONK special!
        forward_dtype="float32",  # Use float32 for compatibility
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    
    model = THONK(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    # Setup training
    print("\n5. Setting up training...")
    training_args = TrainingArguments(
        output_dir="./outputs/thonk-quick-test",
        overwrite_output_dir=True,
        num_train_epochs=3,  # Just 3 epochs for quick test
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch = 16
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        learning_rate=6e-4,
        warmup_steps=50,
        logging_steps=10,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,  # Use fp16 if available
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb for quick test
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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
    
    # Get initial loss
    print("\n6. Initial evaluation...")
    initial_metrics = trainer.evaluate()
    print(f"   Initial loss: {initial_metrics['eval_loss']:.4f}")
    initial_perplexity = torch.exp(torch.tensor(initial_metrics['eval_loss'])).item()
    print(f"   Initial perplexity: {initial_perplexity:.2f}")
    
    # Train
    print("\n7. Starting training...")
    print("   This will take a few minutes...")
    print("-" * 40)
    
    train_result = trainer.train()
    
    # Final evaluation
    print("\n8. Final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"   Final loss: {final_metrics['eval_loss']:.4f}")
    final_perplexity = torch.exp(torch.tensor(final_metrics['eval_loss'])).item()
    print(f"   Final perplexity: {final_perplexity:.2f}")
    
    # Show improvement
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Loss improved from {initial_metrics['eval_loss']:.4f} to {final_metrics['eval_loss']:.4f}")
    print(f"  Perplexity improved from {initial_perplexity:.2f} to {final_perplexity:.2f}")
    print(f"  Training completed in {train_result.metrics['train_runtime']:.1f} seconds")
    print("=" * 60)
    
    # Save model
    print("\n9. Saving model...")
    trainer.save_model("./outputs/thonk-quick-test/final_model")
    tokenizer.save_pretrained("./outputs/thonk-quick-test/final_model")
    print("   Model saved to ./outputs/thonk-quick-test/final_model")
    
    print("\n✅ Quick training test completed successfully!")


if __name__ == "__main__":
    main()
