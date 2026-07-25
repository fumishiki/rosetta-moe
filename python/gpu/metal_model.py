# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU MoE Transformer model.

Assembles the full model pipeline on GPU:
  token_ids -> Embedding -> [TransformerBlock x N] -> RMSNorm -> Linear -> logits

All intermediate activations stay as MetalTensors. Only token IDs (input)
and logits (output) cross the CPU/GPU boundary.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .metal_tensor import MetalContext, MetalTensor
from .metal_layers import MetalEmbedding, MetalRMSNorm, MetalLinear
from .metal_moe import MetalTransformerBlock


class MetalMoETransformer:
    """Full MoE Transformer model on GPU.

    Forward pipeline:
      1. Embedding: token_ids (CPU) -> GPU buffer
      2. N transformer blocks: all on GPU (MPS matmul + MSL kernels)
      3. Final RMSNorm: MSL kernel
      4. LM head: MPS matmul
      5. Read logits back to numpy

    Only step 1 (token ID upload) and step 5 (logits readback) cross
    the CPU/GPU boundary. All intermediate computation stays on GPU.
    """

    def __init__(self, ctx: MetalContext, config: Config):
        self.ctx = ctx
        self.config = config

        self.embedding = MetalEmbedding(ctx, config.vocab_size, config.hidden_dim)
        self.blocks = [MetalTransformerBlock(ctx, config) for _ in range(config.n_layers)]
        self.final_norm = MetalRMSNorm(ctx, config.hidden_dim)
        self.lm_head = MetalLinear(ctx, config.hidden_dim, config.vocab_size)

    @classmethod
    def from_config(cls, ctx: MetalContext, config: Config) -> MetalMoETransformer:
        """Create model from config."""
        return cls(ctx, config)

    @classmethod
    def tiny(cls, ctx: MetalContext) -> MetalMoETransformer:
        """Create tiny model for testing."""
        return cls(ctx, Config.tiny())

    def forward_tokens_tensor(self, token_ids_mt: MetalTensor, batch: int, seq_len: int) -> MetalTensor:
        """Forward pass with GPU-resident token IDs."""
        batch_seq = batch * seq_len

        # 1. Embedding
        h = self.embedding.forward_tensor(token_ids_mt, batch_seq)

        # 2. Transformer blocks (all GPU)
        for block in self.blocks:
            h = block.forward(h, batch, seq_len)

        # 3. Final RMSNorm (GPU)
        h = self.final_norm.forward(h, batch_seq)

        # 4. LM Head (GPU)
        logits = self.lm_head.forward(h, batch_seq)

        return logits

    def forward(self, token_ids: np.ndarray) -> MetalTensor:
        """Forward pass from CPU token IDs."""
        batch, seq_len = token_ids.shape
        ids_f32 = token_ids.astype(np.float32, copy=False)
        token_ids_mt = MetalTensor(self.ctx, data=ids_f32)
        return self.forward_tokens_tensor(token_ids_mt, batch, seq_len)

    def forward_logits_numpy(self, token_ids: np.ndarray, copy: bool = True) -> np.ndarray:
        """Forward pass returning logits as numpy array.

        Convenience method that exposes logits via unified-memory view.

        Args:
            token_ids: numpy int array [batch, seq_len]
            copy: If True, return an owning copy; if False, return zero-copy view.

        Returns:
            numpy array [batch, seq_len, vocab_size]
        """
        batch, seq_len = token_ids.shape
        logits_mt = self.forward(token_ids)
        logits = logits_mt.numpy_view().reshape(batch, seq_len, self.config.vocab_size)
        return logits.copy() if copy else logits
