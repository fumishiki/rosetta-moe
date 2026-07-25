# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU layers for MoE Transformer.

All operations stay on GPU (MetalTensor). No host readback in compute.
Weight initialization uses the same portable LCG as the CPU implementation
for cross-language reproducibility, then uploads weights to GPU once.

Layers:
  - MetalEmbedding: table lookup (CPU-side indexing, GPU buffer output)
  - MetalRMSNorm: custom MSL kernel
  - MetalLinear: MPS matmul (GEMM)
  - MetalSwiGLU: MPS matmul + MSL silu + MSL mul
"""

from __future__ import annotations

import math

import numpy as np

from .metal_tensor import MetalContext, MetalTensor


# --- Portable LCG (same as cpu/tensor.py for reproducible weights) ---
_LCG_MULT = 6_364_136_223_846_793_005
_LCG_MASK = (1 << 64) - 1
_lcg_state: int = 42
_lcg_cache: float | None = None


def seed_rng(seed: int) -> None:
    """Seed the global LCG for reproducible weight initialization."""
    global _lcg_state, _lcg_cache
    _lcg_state = seed & _LCG_MASK
    _lcg_cache = None


def _lcg_uniform() -> float:
    global _lcg_state
    _lcg_state = (_lcg_state * _LCG_MULT + 1) & _LCG_MASK
    u = _lcg_state / 18_446_744_073_709_551_615
    return max(u, 1e-10)


def _lcg_normal() -> float:
    global _lcg_cache
    if _lcg_cache is not None:
        val = _lcg_cache
        _lcg_cache = None
        return val
    u1 = _lcg_uniform()
    u2 = _lcg_uniform()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    _lcg_cache = r * math.sin(theta)
    return r * math.cos(theta)


def _randn_numpy(shape: tuple[int, ...], std: float = 1.0) -> np.ndarray:
    """Generate random normal values using portable LCG."""
    n = 1
    for s in shape:
        n *= s
    data = np.empty(n, dtype=np.float32)
    for i in range(n):
        data[i] = _lcg_normal() * std
    return data.reshape(shape)


class MetalEmbedding:
    """Token embedding layer on GPU.

    Weight table lives on GPU. Forward uses an MSL gather kernel:
      output[token, dim] = weight[token_ids[token], dim]
    """

    def __init__(self, ctx: MetalContext, vocab_size: int, hidden_dim: int):
        self.ctx = ctx
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        std = math.sqrt(2.0 / hidden_dim)
        w_np = _randn_numpy((vocab_size, hidden_dim), std)
        self.weight = MetalTensor.from_numpy(ctx, w_np)

    def forward_tensor(self, token_buf: MetalTensor, batch_seq: int) -> MetalTensor:
        """Lookup embeddings from a GPU-resident token ID buffer."""
        out = MetalTensor.empty(self.ctx, [batch_seq, self.hidden_dim])
        self.ctx.dispatch_kernel(
            "embedding_gather",
            [token_buf, self.weight, out, self.hidden_dim],
            grid_size=batch_seq * self.hidden_dim,
        )
        return out

    def forward(self, token_ids: np.ndarray) -> MetalTensor:
        """Lookup embeddings for CPU token IDs."""
        ids_f32 = token_ids.astype(np.float32, copy=False).reshape(-1)
        token_buf = MetalTensor.from_numpy(self.ctx, ids_f32)
        return self.forward_tensor(token_buf, int(ids_f32.shape[0]))


class MetalRMSNorm:
    """RMSNorm using custom MSL kernel.

    RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma
    """

    def __init__(self, ctx: MetalContext, dim: int, eps: float = 1e-6):
        self.ctx = ctx
        self.dim = dim
        self.eps = eps
        self.weight = MetalTensor.from_numpy(ctx, np.ones(dim, dtype=np.float32))

    def forward(self, x: MetalTensor, n_rows: int) -> MetalTensor:
        """Apply RMS normalization on GPU.

        Args:
            x: Input MetalTensor [n_rows, dim] (flattened)
            n_rows: Number of rows (batch * seq_len)

        Returns:
            MetalTensor [n_rows, dim]
        """
        out = MetalTensor.empty(self.ctx, [n_rows, self.dim])
        self.ctx.dispatch_kernel(
            "rmsnorm",
            [x, out, self.weight, self.dim, self.eps],
            grid_size=n_rows * 256,
            threadgroup_size=256,
        )
        return out


class MetalLinear:
    """Linear layer using MPS matmul: Y = X @ W^T.

    Weight is stored as [out_features, in_features] on GPU.
    We pre-transpose and store W^T = [in_features, out_features] to avoid
    runtime transposition.
    """

    def __init__(self, ctx: MetalContext, in_features: int, out_features: int):
        self.ctx = ctx
        self.in_features = in_features
        self.out_features = out_features

        std = math.sqrt(2.0 / in_features)
        w_np = _randn_numpy((out_features, in_features), std)
        # Store transposed weight on GPU: [in, out] for direct X @ W_T matmul
        self.weight_t = MetalTensor.from_numpy(ctx, w_np.T.copy())

    def forward(self, x: MetalTensor, batch_seq: int) -> MetalTensor:
        """Linear forward: Y = X @ W^T.

        Args:
            x: Input [batch_seq, in_features]
            batch_seq: Number of rows

        Returns:
            MetalTensor [batch_seq, out_features]
        """
        out = MetalTensor.empty(self.ctx, [batch_seq, self.out_features])
        self.ctx.mps_matmul(
            x, self.weight_t, out,
            M=batch_seq, N=self.out_features, K=self.in_features,
        )
        return out


class MetalSwiGLU:
    """SwiGLU on GPU: output = down(silu(gate(x)) * up(x)).

    Three MPS matmuls + MSL silu + MSL mul, all on GPU.
    """

    def __init__(self, ctx: MetalContext, hidden_dim: int, ffn_dim: int):
        self.ctx = ctx
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        self.gate = MetalLinear(ctx, hidden_dim, ffn_dim)
        self.up = MetalLinear(ctx, hidden_dim, ffn_dim)
        self.down = MetalLinear(ctx, ffn_dim, hidden_dim)

    def forward(self, x: MetalTensor, batch_seq: int) -> MetalTensor:
        """SwiGLU forward: down(silu(gate(x)) * up(x)).

        Args:
            x: Input [batch_seq, hidden_dim]
            batch_seq: Number of tokens

        Returns:
            MetalTensor [batch_seq, hidden_dim]
        """
        gate_out = self.gate.forward(x, batch_seq)
        up_out = self.up.forward(x, batch_seq)

        # SiLU on gate output
        n = batch_seq * self.ffn_dim
        silu_out = MetalTensor.empty(self.ctx, [batch_seq, self.ffn_dim])
        self.ctx.dispatch_kernel("silu", [gate_out, silu_out], grid_size=n)

        # Elementwise multiply: silu(gate) * up
        fused = MetalTensor.empty(self.ctx, [batch_seq, self.ffn_dim])
        self.ctx.dispatch_kernel("mul", [silu_out, up_out, fused], grid_size=n)

        # Down projection
        return self.down.forward(fused, batch_seq)
