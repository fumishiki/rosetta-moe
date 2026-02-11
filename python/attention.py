# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Multi-Query Attention with RoPE for MoE Transformer.

Implements scaled dot-product attention with:
  - Multi-Query Attention (MQA): n_kv_heads << n_heads, KV heads shared
  - Rotary Position Embedding (RoPE) with optional NTK-aware scaling
  - Causal (autoregressive) masking

Attention: scores = Q @ K^T / sqrt(d_k), weights = softmax(scores + mask), output = weights @ V
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config
from .layers import Linear
from .tensor import Tensor


class MQAttention:
    """Multi-Query Attention with RoPE.

    MQA uses a single (or few) KV head(s) shared across all Q heads.
    This reduces the KV cache size by n_heads/n_kv_heads during inference
    while preserving most of the model quality of full MHA.
    """

    def __init__(self, config: Config):
        """Initialize attention layer."""
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        # Scaling factor: 1/sqrt(d_k) prevents dot products from growing
        # too large, which would push softmax into saturated regions.
        self.scale = 1.0 / math.sqrt(config.head_dim)

        # Projections — Q has n_heads, K/V have n_kv_heads (MQA)
        self.w_q = Linear(config.hidden_dim, config.n_heads * config.head_dim)
        self.w_k = Linear(config.hidden_dim, config.n_kv_heads * config.head_dim)
        self.w_v = Linear(config.hidden_dim, config.n_kv_heads * config.head_dim)
        self.w_o = Linear(config.n_heads * config.head_dim, config.hidden_dim)

        # RoPE parameters
        self.rope_base = config.rope_base
        self.rope_alpha = config.rope_alpha

        # Causal mask cache — avoids re-allocating a [seq, seq] mask every forward call.
        # Invalidated when seq_len changes.
        self._cached_mask: np.ndarray | None = None
        self._cached_mask_len: int = 0

        # KV head expansion indices — cached to avoid recomputation per forward call
        self._kv_indices = np.arange(self.n_heads) % self.n_kv_heads

    def _compute_rope_freqs(self, seq_len: int) -> np.ndarray:
        """Compute RoPE frequency bands.

        RoPE frequencies: freq_i = 1 / (base^(2i/d))  for i in [0, d/2)

        With NTK scaling (alpha > 1):
          base' = base * alpha^(d/(d-2))
        This stretches the frequency spectrum to support longer sequences
        without retraining.
        """
        base = self.rope_base
        if self.rope_alpha > 1.0:
            # NTK-aware scaling: base' = base * alpha^(d/(d-2))
            base = self.rope_base * (
                self.rope_alpha ** (self.head_dim / (self.head_dim - 2))
            )

        i = np.arange(self.head_dim // 2, dtype=np.float32)
        return (1.0 / (base ** (2 * i / self.head_dim))).astype(np.float32)

    def _apply_rope(
        self, q: np.ndarray, k: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Rotary Position Embedding to Q and K.

        RoPE rotation for each pair of dimensions (2i, 2i+1):
          [x_even, x_odd] -> [x_even*cos(theta) - x_odd*sin(theta),
                               x_even*sin(theta) + x_odd*cos(theta)]
        where theta = position * freq_i.

        This encodes absolute position through relative-position-aware
        dot products: q_m . k_n depends only on (m - n).
        """
        freqs = self._compute_rope_freqs(seq_len)

        # angles[pos, i] = pos * freq_i   shape: [seq_len, head_dim/2]
        positions = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)

        # Broadcast to [1, seq_len, 1, half_dim] for batched Q/K
        cos_vals = np.cos(angles)[None, :, None, :]
        sin_vals = np.sin(angles)[None, :, None, :]

        # Split even/odd dims: q is [batch, seq, heads, head_dim]
        q_even = q[:, :, :, 0::2]
        q_odd = q[:, :, :, 1::2]
        k_even = k[:, :, :, 0::2]
        k_odd = k[:, :, :, 1::2]

        # Apply 2D rotation per pair
        q_out = np.empty_like(q)
        q_out[:, :, :, 0::2] = q_even * cos_vals - q_odd * sin_vals
        q_out[:, :, :, 1::2] = q_even * sin_vals + q_odd * cos_vals

        k_out = np.empty_like(k)
        k_out[:, :, :, 0::2] = k_even * cos_vals - k_odd * sin_vals
        k_out[:, :, :, 1::2] = k_even * sin_vals + k_odd * cos_vals

        return q_out, k_out

    def forward(self, x: Tensor) -> Tensor:
        """Multi-query attention forward pass.

        Full formula:
          Q, K, V = x @ W_q, x @ W_k, x @ W_v
          Q, K    = RoPE(Q), RoPE(K)
          scores  = (Q @ K^T) / sqrt(d_k) + causal_mask
          weights = softmax(scores)
          output  = (weights @ V) @ W_o

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.w_q.forward(x)  # [batch, seq_len, n_heads * head_dim]
        k = self.w_k.forward(x)  # [batch, seq_len, n_kv_heads * head_dim]
        v = self.w_v.forward(x)  # [batch, seq_len, n_kv_heads * head_dim]

        # Reshape to [batch, seq_len, n_heads, head_dim] using raw numpy (no Tensor overhead)
        q_data = q.data.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k_data = k.data.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v_data = v.data.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q_data, k_data = self._apply_rope(q_data, k_data, seq_len)

        # GQA/MQA head expansion: map each Q head to its corresponding KV head.
        # With n_kv_heads=1 (MQA), all Q heads share the same single KV head.
        # Uses cached _kv_indices to avoid recomputation per forward call.
        k_expanded = k_data[:, :, self._kv_indices, :]
        v_expanded = v_data[:, :, self._kv_indices, :]

        # Transpose to [batch, heads, seq, dim] for batched matmul
        q_t = np.transpose(q_data, (0, 2, 1, 3))
        k_t = np.transpose(k_expanded, (0, 2, 1, 3))
        v_t = np.transpose(v_expanded, (0, 2, 1, 3))

        # Attention scores: (Q @ K^T) / sqrt(d_k)
        # np.matmul dispatches to BLAS for the batched [heads, seq, dim] @ [heads, dim, seq] matmul
        scores = np.matmul(q_t, k_t.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask: upper-triangular -inf prevents attending to future tokens.
        # Cached to avoid re-allocation when seq_len is unchanged between calls.
        if self._cached_mask_len != seq_len:
            self._cached_mask = np.where(
                np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) > 0,
                float('-inf'),
                0.0,
            )
            self._cached_mask_len = seq_len
        scores += self._cached_mask

        # Numerically stable softmax: subtract max before exp to prevent overflow
        # Softmax: p_i = exp(s_i - max(s)) / sum_j(exp(s_j - max(s)))
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        attn_weights = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

        # Store for backward pass
        self.last_attn_weights = attn_weights
        self.last_q = q_data
        self.last_k = k_data
        self.last_v = v_data
        self.last_input = x

        # Weighted sum of values: weights @ V
        # np.matmul dispatches to BLAS for [heads, seq, seq] @ [heads, seq, dim]
        output = np.matmul(attn_weights, v_t)
        # Transpose [batch, heads, seq, dim] -> [batch, seq, heads, dim] and flatten heads
        output = np.ascontiguousarray(np.transpose(output, (0, 2, 1, 3)))
        output = output.reshape(batch, seq_len, self.n_heads * self.head_dim)
        return self.w_o.forward(Tensor.from_numpy(output))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through attention layer.

        Propagates gradients through: W_o -> attention -> W_q, W_k, W_v.
        Uses stored attention weights and Q/K/V from the forward pass.

        The attention backward computes:
          grad w.r.t. V:  attn_weights^T @ grad_attn_out
          grad w.r.t. attn_weights: grad_attn_out @ V^T
          grad w.r.t. scores: softmax jacobian (simplified as grad_weights * weights * (1 - weights))
          grad w.r.t. Q: grad_scores @ K
          grad w.r.t. K: grad_scores^T @ Q
        Then propagates through the projection layers for weight gradient accumulation.
        """
        batch, seq_len, _ = grad_output.shape

        # 1. Backward through W_o
        grad_o_input = self.w_o.backward(grad_output)

        # Reshape grad to [batch, seq_len, n_heads, head_dim]
        grad_o_reshaped = grad_o_input.data.reshape(batch, seq_len, self.n_heads, self.head_dim)
        # Transpose to [batch, n_heads, seq_len, head_dim]
        grad_attn_out = np.transpose(grad_o_reshaped, (0, 2, 1, 3))

        # 2. Backward through attention: output = attn_weights @ V
        attn_weights = self.last_attn_weights  # [batch, n_heads, seq, seq]
        v_data = self.last_v  # [batch, seq, n_kv_heads, head_dim]
        v_expanded = v_data[:, :, self._kv_indices, :]  # [batch, seq, n_heads, head_dim]
        v_t = np.transpose(v_expanded, (0, 2, 1, 3))  # [batch, n_heads, seq, head_dim]

        # grad_V = attn_weights^T @ grad_attn_out  -> [batch, n_heads, seq, head_dim]
        grad_v_t = np.matmul(attn_weights.transpose(0, 1, 3, 2), grad_attn_out)

        # grad_attn_weights = grad_attn_out @ V^T  -> [batch, n_heads, seq, seq]
        grad_attn_w = np.matmul(grad_attn_out, v_t.transpose(0, 1, 3, 2))

        # 3. Backward through softmax: simplified Jacobian
        # d_scores = attn_weights * (grad_attn_w - sum(grad_attn_w * attn_weights, axis=-1, keepdims))
        sum_term = np.sum(grad_attn_w * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn_w - sum_term)

        # Apply scale factor
        grad_scores *= self.scale

        # 4. Backward through Q @ K^T
        q_data = self.last_q  # [batch, seq, n_heads, head_dim]
        k_data = self.last_k  # [batch, seq, n_kv_heads, head_dim]
        k_expanded = k_data[:, :, self._kv_indices, :]
        q_t = np.transpose(q_data, (0, 2, 1, 3))  # [batch, n_heads, seq, head_dim]
        k_t = np.transpose(k_expanded, (0, 2, 1, 3))

        # grad_Q = grad_scores @ K  -> [batch, n_heads, seq, head_dim]
        grad_q_t = np.matmul(grad_scores, k_t)
        # grad_K = grad_scores^T @ Q  -> [batch, n_heads, seq, head_dim]
        grad_k_t = np.matmul(grad_scores.transpose(0, 1, 3, 2), q_t)

        # Transpose back to [batch, seq, n_heads, head_dim]
        grad_q = np.transpose(grad_q_t, (0, 2, 1, 3)).reshape(batch, seq_len, -1)
        grad_k = np.transpose(grad_k_t, (0, 2, 1, 3))
        grad_v = np.transpose(grad_v_t, (0, 2, 1, 3))

        # Reduce expanded heads back to kv_heads by summing
        # Head mapping is interleaved: [0,1,...,kv-1, 0,1,...,kv-1, ...]
        # Reshape to [batch, seq, heads_per_kv, n_kv_heads, head_dim] and sum
        heads_per_kv = self.n_heads // self.n_kv_heads
        grad_k_reduced = grad_k.reshape(
            batch, seq_len, heads_per_kv, self.n_kv_heads, self.head_dim
        ).sum(axis=2)
        grad_v_reduced = grad_v.reshape(
            batch, seq_len, heads_per_kv, self.n_kv_heads, self.head_dim
        ).sum(axis=2)

        grad_k_flat = grad_k_reduced.reshape(batch, seq_len, -1)
        grad_v_flat = grad_v_reduced.reshape(batch, seq_len, -1)

        # 5. Backward through projection layers (accumulates weight gradients)
        grad_q_tensor = Tensor.from_numpy(grad_q)
        grad_k_tensor = Tensor.from_numpy(grad_k_flat)
        grad_v_tensor = Tensor.from_numpy(grad_v_flat)

        # Set _last_input for Q/K/V projections (they all used the same input x)
        self.w_q._last_input = self.last_input
        self.w_k._last_input = self.last_input
        self.w_v._last_input = self.last_input

        grad_x_q = self.w_q.backward(grad_q_tensor)
        grad_x_k = self.w_k.backward(grad_k_tensor)
        grad_x_v = self.w_v.backward(grad_v_tensor)

        # Sum gradients from all projection paths
        grad_x = grad_x_q.data + grad_x_k.data + grad_x_v.data
        return Tensor.from_numpy(grad_x)

    def parameters(self) -> list[Tensor]:
        """Return all parameters (Q, K, V, O projections)."""
        return (
            self.w_q.parameters()
            + self.w_k.parameters()
            + self.w_v.parameters()
            + self.w_o.parameters()
        )
