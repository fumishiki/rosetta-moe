# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU Multi-Query Attention with RoPE for MoE Transformer.

All heavy-path ops are dispatched on GPU:
  - Q/K/V projections (MPS matmul)
  - RoPE rotation (MSL rope_inplace)
  - attention scores + causal mask (MSL)
  - softmax (MSL)
  - weighted sum (MSL)
  - output projection (MPS matmul)

No host-side tensor views are used in forward.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from .metal_tensor import MetalContext, MetalTensor
from .metal_layers import MetalLinear


class MetalMQAttention:
    """Multi-Query Attention with RoPE, GPU-only forward path."""

    def __init__(self, ctx: MetalContext, config: Config):
        self.ctx = ctx
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        self.w_q = MetalLinear(ctx, config.hidden_dim, config.n_heads * config.head_dim)
        self.w_k = MetalLinear(ctx, config.hidden_dim, config.n_kv_heads * config.head_dim)
        self.w_v = MetalLinear(ctx, config.hidden_dim, config.n_kv_heads * config.head_dim)
        self.w_o = MetalLinear(ctx, config.n_heads * config.head_dim, config.hidden_dim)

        self.rope_base = config.rope_base
        self.rope_alpha = config.rope_alpha

        self._cached_rope_cos: MetalTensor | None = None
        self._cached_rope_sin: MetalTensor | None = None
        self._cached_rope_len: int = 0
        # Per-forward scratch buffers (resized on demand, reused across calls)
        self._buf_scores: MetalTensor | None = None
        self._buf_probs: MetalTensor | None = None
        self._buf_attn_out: MetalTensor | None = None

    def _ensure_rope_cache(self, seq_len: int) -> None:
        """Build and upload RoPE cache tables for current sequence length."""
        if self._cached_rope_len == seq_len and self._cached_rope_cos is not None:
            return

        base = self.rope_base
        if self.rope_alpha > 1.0:
            base = self.rope_base * (
                self.rope_alpha ** (self.head_dim / (self.head_dim - 2))
            )

        half = self.head_dim // 2
        i = np.arange(half, dtype=np.float32)
        freqs = (1.0 / (base ** (2 * i / self.head_dim))).astype(np.float32)
        positions = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)

        cos_np = np.cos(angles).astype(np.float32)
        sin_np = np.sin(angles).astype(np.float32)
        self._cached_rope_cos = MetalTensor.from_numpy(self.ctx, cos_np)
        self._cached_rope_sin = MetalTensor.from_numpy(self.ctx, sin_np)
        self._cached_rope_len = seq_len

    def forward(self, x: MetalTensor, batch: int, seq_len: int) -> MetalTensor:
        """Multi-query attention forward pass on GPU only."""
        batch_seq = batch * seq_len
        half_dim = self.head_dim // 2

        q = self.w_q.forward(x, batch_seq)
        k = self.w_k.forward(x, batch_seq)
        v = self.w_v.forward(x, batch_seq)

        self._ensure_rope_cache(seq_len)
        if self._cached_rope_cos is None or self._cached_rope_sin is None:
            raise RuntimeError("RoPE cache was not initialized")

        # In-place RoPE rotation on Q/K.
        self.ctx.dispatch_kernel(
            "rope_inplace",
            [q, self._cached_rope_cos, self._cached_rope_sin, self.n_heads, self.head_dim, seq_len],
            grid_size=batch_seq * self.n_heads * half_dim,
        )
        self.ctx.dispatch_kernel(
            "rope_inplace",
            [k, self._cached_rope_cos, self._cached_rope_sin, self.n_kv_heads, self.head_dim, seq_len],
            grid_size=batch_seq * self.n_kv_heads * half_dim,
        )

        scores_shape = [batch * self.n_heads, seq_len, seq_len]
        if self._buf_scores is None or self._buf_scores.shape != scores_shape:
            self._buf_scores = MetalTensor.empty(self.ctx, scores_shape)
        scores = self._buf_scores
        # scores: [batch * n_heads, seq_len, seq_len]
        self.ctx.dispatch_kernel(
            "attention_scores",
            [
                q,
                k,
                scores,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                seq_len,
                self.scale,
            ],
            grid_size=batch * self.n_heads * seq_len * seq_len,
        )

        # Causal mask (in-place)
        self.ctx.dispatch_kernel(
            "causal_mask_fill",
            [scores, seq_len],
            grid_size=batch * self.n_heads * seq_len * seq_len,
        )

        probs_shape = [batch * self.n_heads, seq_len, seq_len]
        if self._buf_probs is None or self._buf_probs.shape != probs_shape:
            self._buf_probs = MetalTensor.empty(self.ctx, probs_shape)
        probs = self._buf_probs
        # Softmax across key axis (row length = seq_len).
        self.ctx.dispatch_kernel(
            "softmax",
            [scores, probs, seq_len],
            grid_size=batch * self.n_heads * seq_len * 256,
            threadgroup_size=256,
        )

        attn_out_shape = [batch_seq, self.n_heads * self.head_dim]
        if self._buf_attn_out is None or self._buf_attn_out.shape != attn_out_shape:
            self._buf_attn_out = MetalTensor.empty(self.ctx, attn_out_shape)
        attn_out = self._buf_attn_out
        # Weighted sum with V -> [batch_seq, n_heads * head_dim]
        self.ctx.dispatch_kernel(
            "attention_weighted_sum",
            [
                probs,
                v,
                attn_out,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                seq_len,
            ],
            grid_size=batch_seq * self.n_heads * self.head_dim,
        )

        return self.w_o.forward(attn_out, batch_seq)
