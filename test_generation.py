#!/usr/bin/env python3
"""
Test text generation with the trained THONK model.
"""

import sys
import os
import torch
from transformers import GPT2TokenizerFast

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.thonk_model import THONK


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, do_sample=True):
    """Generate text from a prompt."""
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    """Test generation with various prompts."""
    
    print("=" * 60)
    print("THONK - Generation Test")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if trained model exists
    model_path = "./outputs/thonk-quick-test/checkpoint-3000"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = THONK.from_pretrained(model_path)
    else:
        print("No trained model found. Using untrained model...")
        from models.thonk_model import THONKConfig
        config = THONKConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,
            num_heads=4,
            H_layers=2,
            L_layers=2,
            H_cycles=1,
            L_cycles=1,
            max_position_embeddings=512,
            forward_dtype="float32",
        )
        model = THONK(config)
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "### Instruction:\nWrite a haiku about programming.\n\n### Response:\n",
        "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",
        "Once upon a time,",
        "The quick brown fox",
    ]
    
    print("\n" + "=" * 60)
    print("GENERATION EXAMPLES:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print("-" * 40)
        
        # Generate with different settings
        # More deterministic generation
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        print(f"Generated:\n{generated}")
        print("-" * 40)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit):")
    print("=" * 60)
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            continue
        
        # Format as instruction if it looks like a question/command
        if not prompt.startswith("###"):
            prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True
        )
        
        print(f"\nGenerated:\n{generated}")
    
    print("\n✅ Generation test completed!")


if __name__ == "__main__":
    main()
