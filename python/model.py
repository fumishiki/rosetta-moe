# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""MoE Transformer model implementation.

Assembles the full model pipeline:
  token_ids -> Embedding -> [TransformerBlock x N] -> RMSNorm -> Linear -> logits

Each TransformerBlock contains MQA + MoE (see moe.py).
The model exposes multiple generation methods (greedy, temperature, top-k, top-p).
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .generate import generate, GreedySampling
from .layers import Embedding, Linear, RMSNorm
from .moe import TransformerBlock
from .tensor import Tensor


class MoETransformer:
    """Full MoE Transformer model.

    Forward pipeline:
      1. Embedding lookup: token_ids -> hidden states [batch, seq, hidden_dim]
      2. N transformer blocks: each applies MQA attention + MoE FFN with residual
      3. Final RMSNorm
      4. LM head (Linear): hidden_dim -> vocab_size logits
    """

    def __init__(self, config: Config):
        """Initialize transformer model."""
        self.config = config

        self.embedding = Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = Linear(config.hidden_dim, config.vocab_size)

    @classmethod
    def from_config(cls, config: Config) -> MoETransformer:
        """Create model from config."""
        return cls(config)

    @classmethod
    def tiny(cls) -> MoETransformer:
        """Create tiny model for testing."""
        return cls(Config.tiny())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Pipeline: Embed -> Blocks -> Norm -> LM Head

        Args:
            x: Token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Embedding: token IDs -> dense vectors
        h = self.embedding.forward(x)

        # Transformer blocks (each: pre-norm attention + pre-norm MoE + residuals)
        for block in self.blocks:
            h = block.forward(h)

        # Final norm and LM head projection to vocabulary
        h = self.final_norm.forward(h)
        logits = self.lm_head.forward(h)

        return logits

    def forward_ids(
        self, token_ids: list[int], batch: int = 1, seq_len: int | None = None
    ) -> Tensor:
        """Forward pass with token ID list.

        Convenience wrapper that reshapes a flat list of token IDs into
        [batch, seq_len] before calling forward().

        Args:
            token_ids: List of token IDs
            batch: Batch size
            seq_len: Sequence length (defaults to len(token_ids) // batch)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if seq_len is None:
            seq_len = len(token_ids) // batch

        x = Tensor.from_numpy(
            np.array(token_ids, dtype=np.int64).reshape((batch, seq_len))
        )
        return self.forward(x)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through all layers (reverse of forward).

        Propagates gradients: LM head -> final norm -> blocks (reversed) -> embedding.
        Each layer's backward() stores parameter gradients on param._grad.
        """
        grad = self.lm_head.backward(grad_output)
        grad = self.final_norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        self.embedding.backward(grad)
        return grad

    def parameters(self) -> list[Tensor]:
        """Return all model parameters."""
        params = self.embedding.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def total_aux_loss(self, alpha: float = 0.01) -> float:
        """Return total auxiliary loss from all MoE layers."""
        return sum(block.aux_loss(alpha) for block in self.blocks)

    def generate(self, prompt: list[int], max_len: int) -> list[int]:
        """Generate tokens using greedy decoding."""
        return generate(self, prompt, max_len, GreedySampling())

    def generate_greedy(self, prompt: list[int], max_len: int) -> list[int]:
        """Generate tokens using greedy (argmax) decoding."""
        return generate(self, prompt, max_len, GreedySampling())

    def generate_sample(
        self,
        prompt: list[int],
        max_len: int,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> list[int]:
        """Generate tokens using temperature sampling."""
        from .generate import TemperatureSampling
        return generate(self, prompt, max_len, TemperatureSampling(temperature, seed))

    def generate_top_k(
        self,
        prompt: list[int],
        max_len: int,
        k: int = 50,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> list[int]:
        """Generate tokens using top-k sampling."""
        from .generate import TopKSampling
        return generate(self, prompt, max_len, TopKSampling(k, temperature, seed))

    def generate_top_p(
        self,
        prompt: list[int],
        max_len: int,
        top_p: float = 0.9,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> list[int]:
        """Generate tokens using top-p (nucleus) sampling."""
        from .generate import TopPSampling
        return generate(self, prompt, max_len, TopPSampling(top_p, temperature, seed))


def tiny_model() -> MoETransformer:
    """Create a tiny model for testing."""
    return MoETransformer(Config.tiny())


def default_model() -> MoETransformer:
    """Create the default 6.9B model."""
    return MoETransformer(Config.default_6_9b())
