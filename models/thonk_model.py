"""
THONK: Thinking Hierarchically - Optimized Neural Knowledge

A text generation model that combines hierarchical reasoning with adaptive
computation time, designed for efficient learning from minimal data.

This module provides HuggingFace compatibility for training and inference.
"""

from typing import Optional, Tuple, Union, Dict, List
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1Config
)

logger = logging.get_logger(__name__)


class THONKConfig(PretrainedConfig):
    """
    Configuration class for THONK (Thinking Hierarchically: Optimized Neural Knowledge).
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimensionality of the hidden states
        num_heads: Number of attention heads
        H_layers: Number of H-level (high-level) layers
        L_layers: Number of L-level (low-level) layers
        H_cycles: Number of H-level cycles
        L_cycles: Number of L-level cycles
        max_position_embeddings: Maximum sequence length
        halt_max_steps: Maximum ACT steps
        halt_exploration_prob: Exploration probability for ACT
        rope_theta: Base frequency for RoPE
        rms_norm_eps: Epsilon for RMS normalization
        expansion: Expansion factor for feedforward network
        use_act: Whether to use Adaptive Computation Time
        forward_dtype: Data type for forward pass
    """
    
    model_type = "thonk"
    
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 default vocab size
        hidden_size=512,
        num_heads=8,
        H_layers=4,
        L_layers=4,
        H_cycles=2,
        L_cycles=2,
        max_position_embeddings=2048,
        halt_max_steps=8,  # Allow up to 8 thinking steps
        halt_exploration_prob=0.1,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        expansion=2.66667,  # Standard for SwiGLU (8/3)
        use_act=True,  # ACT enabled by default - it's our key innovation!
        forward_dtype="float32",  # Use float32 for compatibility (bfloat16 not supported on MPS)
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        use_cache=True,  # Add use_cache for generation
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.max_position_embeddings = max_position_embeddings
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.expansion = expansion
        self.use_act = use_act
        self.forward_dtype = forward_dtype
        self.use_cache = use_cache


class THONK(PreTrainedModel, GenerationMixin):
    """
    THONK: Thinking Hierarchically - Optimized Neural Knowledge
    
    A language model that learns efficiently from minimal data through
    hierarchical reasoning and (optionally) adaptive computation time.
    """
    
    config_class = THONKConfig
    base_model_prefix = "thonk"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HierarchicalReasoningModel_ACTV1Block"]
    
    def __init__(self, config: THONKConfig):
        super().__init__(config)
        
        # Create HRM config dict from HuggingFace config
        hrm_config_dict = {
            "batch_size": 1,  # Will be set dynamically
            "seq_len": config.max_position_embeddings,
            "puzzle_emb_ndim": 0,  # No puzzle embeddings for text
            "num_puzzle_identifiers": 1,  # Dummy value
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "expansion": config.expansion,
            "num_heads": config.num_heads,
            "pos_encodings": "rope",  # Always use RoPE for text
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "H_cycles": config.H_cycles,
            "L_cycles": config.L_cycles,
            "H_layers": config.H_layers,
            "L_layers": config.L_layers,
            "halt_max_steps": config.halt_max_steps if config.use_act else 1,
            "halt_exploration_prob": config.halt_exploration_prob,
            "forward_dtype": config.forward_dtype,
        }
        
        # Convert to HRM config object
        hrm_config = HierarchicalReasoningModel_ACTV1Config(**hrm_config_dict)
        
        # Initialize the core HRM model
        self.hrm = HierarchicalReasoningModel_ACTV1_Inner(hrm_config)
        
        # Store config for easy access
        self.hrm_config = hrm_config
        self.use_act = config.use_act
        
        # Initialize carry state storage for generation
        self.past_carry = None
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        """Get input embeddings layer."""
        return self.hrm.embed_tokens
    
    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        self.hrm.embed_tokens = value
    
    def get_output_embeddings(self):
        """Get output embeddings layer (language model head)."""
        return self.hrm.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings layer."""
        self.hrm.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        # If we have past_key_values (carry state), we only need the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }
        
        return model_inputs
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass compatible with HuggingFace transformers.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (currently unused, HRM handles internally)
            position_ids: Position IDs (unused, HRM uses RoPE)
            past_key_values: Past key values (used as carry state for HRM)
            inputs_embeds: Input embeddings (alternative to input_ids)
            labels: Labels for language modeling loss
            use_cache: Whether to return past_key_values
            output_attentions: Whether to return attention weights (not supported)
            output_hidden_states: Whether to return hidden states (not supported)
            return_dict: Whether to return a ModelOutput object
        
        Returns:
            CausalLMOutputWithPast containing loss, logits, and optionally past_key_values
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Get batch size and sequence length
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("You must provide input_ids")
        
        # Pad input_ids to max_position_embeddings if needed (HRM expects fixed size)
        if seq_length < self.config.max_position_embeddings:
            padding_length = self.config.max_position_embeddings - seq_length
            # Pad with the pad_token_id or 0
            pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
            input_ids = F.pad(input_ids, (0, padding_length), value=pad_id)
        elif seq_length > self.config.max_position_embeddings:
            # Truncate if too long
            input_ids = input_ids[:, :self.config.max_position_embeddings]
        
        # Prepare HRM batch format
        batch = {
            "inputs": input_ids,
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=input_ids.device),
        }
        
        # Initialize or retrieve carry state
        if past_key_values is not None:
            # Use stored carry state from previous forward pass
            carry = past_key_values
        else:
            # Create fresh carry state
            # Update batch size in config (hacky but necessary)
            self.hrm_config.batch_size = batch_size
            
            # Ensure model is on the same device as input
            device = input_ids.device
            if self.hrm.H_init.device != device:
                self.hrm = self.hrm.to(device)
            
            carry = self.hrm.empty_carry(batch_size)
            carry = self.hrm.reset_carry(
                torch.ones(batch_size, dtype=torch.bool, device=device),
                carry
            )
        
        # Forward through HRM
        new_carry, output, (q_halt_logits, q_continue_logits) = self.hrm(carry, batch)
        
        # Output has shape: (batch_size, max_position_embeddings, vocab_size)
        # We need to extract only the original sequence length
        logits = output[:, :seq_length, :]  # Shape: (batch_size, seq_length, vocab_size)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        # Prepare output
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (new_carry,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_carry if use_cache else None,
            hidden_states=None,  # Not supported yet
            attentions=None,  # Not supported yet
        )
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: Optional[int] = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text using the HRM model.
        
        This is a simplified generation method. For full functionality,
        use HuggingFace's generate() method which will call our forward().
        """
        
        # Use HuggingFace's generation method
        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id if pad_token_id is not None else self.config.pad_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else self.config.eos_token_id,
            use_cache=True,  # Always use cache for generation
            **kwargs
        )
    
    @torch.no_grad()
    def generate_with_act(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, List[int]]:
        """
        Generate text with Adaptive Computation Time tracking.
        
        Returns:
            generated_ids: The generated token IDs
            act_steps: List of ACT steps used for each token
        """
        
        if not self.use_act:
            raise ValueError("ACT is not enabled in this model configuration")
        
        # TODO: Implement ACT-aware generation
        # This would track the number of computation steps used for each token
        
        raise NotImplementedError("ACT-aware generation not yet implemented")
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resize token embeddings matrix to new vocabulary size.
        """
        model_embeds = self.hrm.embed_tokens
        if new_num_tokens is None:
            return model_embeds
        
        old_num_tokens = model_embeds.num_embeddings
        if old_num_tokens == new_num_tokens:
            return model_embeds
        
        # Build new embeddings
        new_embeddings = self._get_resized_embeddings(model_embeds, new_num_tokens)
        self.hrm.embed_tokens = new_embeddings
        
        # Update config
        self.config.vocab_size = new_num_tokens
        self.hrm_config.vocab_size = new_num_tokens
        
        # Resize output embeddings
        self.hrm.lm_head = self._get_resized_lm_head(self.hrm.lm_head, new_num_tokens)
        
        return model_embeds
    
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """Build resized embeddings."""
        if new_num_tokens is None:
            return old_embeddings
        
        old_num_tokens, embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        return new_embeddings
    
    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None
    ) -> nn.Linear:
        """Build resized language model head."""
        if new_num_tokens is None:
            return old_lm_head
        
        old_num_tokens, hidden_size = old_lm_head.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_lm_head
        
        # Create new lm_head
        new_lm_head = nn.Linear(hidden_size, new_num_tokens, bias=False)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        
        return new_lm_head


# Register the model with HuggingFace
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("thonk", THONKConfig)
AutoModelForCausalLM.register(THONKConfig, THONK)
