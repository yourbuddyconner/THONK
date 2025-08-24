#!/usr/bin/env python3
"""
Test script to verify tokenizer functionality with THONK model.

This script tests different tokenizer options and verifies compatibility
with the THONK language model.
"""

import torch
from transformers import (
    GPT2TokenizerFast,
    AutoTokenizer,
    LlamaTokenizer,
)


def test_gpt2_tokenizer():
    """Test GPT-2 tokenizer."""
    print("=" * 60)
    print("Testing GPT-2 Tokenizer")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Set padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Print tokenizer info
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Padding token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"BOS token: {tokenizer.bos_token}")
    
    # Test tokenization
    test_texts = [
        "Hello, this is a test of the THONK model.",
        "The hierarchical reasoning model can process text efficiently.",
        "What is the capital of France?",
        "Solve this equation: 2x + 5 = 13",
    ]
    
    print("\nTokenization examples:")
    for text in test_texts[:2]:
        tokens = tokenizer(text, return_tensors="pt")
        print(f"\nText: {text}")
        print(f"Token IDs shape: {tokens['input_ids'].shape}")
        print(f"Token IDs: {tokens['input_ids'][0][:20]}...")  # First 20 tokens
        
        # Decode back
        decoded = tokenizer.decode(tokens['input_ids'][0])
        print(f"Decoded: {decoded}")
    
    # Test batch tokenization
    print("\nBatch tokenization:")
    batch = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )
    
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    
    return tokenizer


def test_model_compatibility():
    """Test tokenizer compatibility with THONK model."""
    print("\n" + "=" * 60)
    print("Testing Model Compatibility")
    print("=" * 60)
    
    try:
        from models.thonk_model import THONKConfig, THONK
        
        # Load tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create model config
        config = THONKConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,  # Smaller for testing
            num_heads=4,
            H_layers=2,
            L_layers=2,
            H_cycles=1,
            L_cycles=1,
            max_position_embeddings=512,
            use_act=True,  # ACT enabled by default
        )
        
        print(f"Model config created successfully")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Max position: {config.max_position_embeddings}")
        
        # Initialize model
        print("\nInitializing THONK model...")
        model = THONK(config)
        print(f"Model initialized successfully!")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"Forward pass successful!")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  Expected shape: (batch_size=1, seq_len={inputs['input_ids'].shape[1]}, vocab_size={config.vocab_size})")
        
        # Test with labels (for loss calculation)
        print("\nTesting with labels...")
        inputs["labels"] = inputs["input_ids"].clone()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        if outputs.loss is not None:
            print(f"Loss calculation successful!")
            print(f"  Loss value: {outputs.loss.item():.4f}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during model compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_alternative_tokenizers():
    """Test alternative tokenizer options."""
    print("\n" + "=" * 60)
    print("Testing Alternative Tokenizers")
    print("=" * 60)
    
    # Test T5 tokenizer (requires sentencepiece)
    try:
        from transformers import T5TokenizerFast
        print("\nT5 Tokenizer:")
        t5_tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        print(f"  Vocab size: {t5_tokenizer.vocab_size}")
        print(f"  Model max length: {t5_tokenizer.model_max_length}")
        
        text = "Hello, world!"
        tokens = t5_tokenizer(text, return_tensors="pt")
        print(f"  Example tokenization: '{text}' -> {tokens['input_ids'][0].tolist()}")
    except Exception as e:
        print(f"  T5 tokenizer not available: {e}")
    
    # Test if tiktoken is available
    try:
        import tiktoken
        print("\nTiktoken (OpenAI's tokenizer):")
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding
        print(f"  Vocab size: {enc.n_vocab}")
        
        text = "Hello, world!"
        tokens = enc.encode(text)
        print(f"  Example tokenization: '{text}' -> {tokens}")
        print(f"  Decoded: '{enc.decode(tokens)}'")
    except ImportError:
        print("  Tiktoken not installed. Install with: pip install tiktoken")
    except Exception as e:
        print(f"  Tiktoken error: {e}")


def main():
    """Run all tokenizer tests."""
    print("THONK - Tokenizer Test Suite")
    print("=" * 60)
    
    # Test GPT-2 tokenizer
    tokenizer = test_gpt2_tokenizer()
    
    # Test model compatibility
    model, tokenizer = test_model_compatibility()
    
    # Test alternative tokenizers
    test_alternative_tokenizers()
    
    print("\n" + "=" * 60)
    print("Tokenizer tests completed!")
    
    if model is not None:
        print("\n✅ Model and tokenizer are compatible and ready for training!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
