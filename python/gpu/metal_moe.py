# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU MoE (Mixture of Experts) for MoE Transformer.

Router and expert dispatch are fully GPU-dispatched:
  - gate projection (MPS matmul)
  - gate softmax (MSL)
  - top-k routing (MSL)
  - per-expert routing-weight extraction (MSL)
  - expert SwiGLU (MPS+MSL)
  - weighted accumulation (MSL row_scale + add)

No host-side gather/scatter views are used in forward.
"""

from __future__ import annotations

from ..config import Config
from .metal_tensor import MetalContext, MetalTensor
from .metal_layers import MetalLinear, MetalSwiGLU, MetalRMSNorm


class MetalRouter:
    """MoE Router with GPU top-k selection."""

    def __init__(self, ctx: MetalContext, hidden_dim: int, n_experts: int, top_k: int):
        self.ctx = ctx
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = MetalLinear(ctx, hidden_dim, n_experts)
        self._buf_probs: MetalTensor | None = None
        self._buf_topk_indices: MetalTensor | None = None
        self._buf_topk_weights: MetalTensor | None = None

    def forward(self, x: MetalTensor, num_tokens: int) -> tuple[MetalTensor, MetalTensor]:
        """Return top-k routing weights and indices for each token.

        Returns:
            topk_weights: [num_tokens, top_k]
            topk_indices: [num_tokens, top_k] (float-encoded uint indices)
        """
        logits = self.gate.forward(x, num_tokens)

        probs_shape = [num_tokens, self.n_experts]
        if self._buf_probs is None or self._buf_probs.shape != probs_shape:
            self._buf_probs = MetalTensor.empty(self.ctx, probs_shape)
        probs = self._buf_probs
        self.ctx.dispatch_kernel(
            "softmax",
            [logits, probs, self.n_experts],
            grid_size=num_tokens * 256,
            threadgroup_size=256,
        )

        topk_shape = [num_tokens, self.top_k]
        if self._buf_topk_indices is None or self._buf_topk_indices.shape != topk_shape:
            self._buf_topk_indices = MetalTensor.empty(self.ctx, topk_shape)
        if self._buf_topk_weights is None or self._buf_topk_weights.shape != topk_shape:
            self._buf_topk_weights = MetalTensor.empty(self.ctx, topk_shape)
        topk_indices = self._buf_topk_indices
        topk_weights = self._buf_topk_weights
        self.ctx.dispatch_kernel(
            "moe_topk",
            [probs, topk_indices, topk_weights, self.n_experts, self.top_k],
            grid_size=num_tokens,
            threadgroup_size=1,
        )

        return topk_weights, topk_indices


class MetalMoELayer:
    """Mixture-of-Experts layer on GPU."""

    def __init__(self, ctx: MetalContext, config: Config):
        self.ctx = ctx
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.n_experts = config.n_experts
        self.top_k = config.top_k_experts

        self.router = MetalRouter(ctx, config.hidden_dim, config.n_experts, config.top_k_experts)
        self.experts = [
            MetalSwiGLU(ctx, config.hidden_dim, config.ffn_dim)
            for _ in range(config.n_experts)
        ]
        # scratch buffers reused per forward
        self._buf_moe_acc: MetalTensor | None = None
        self._buf_moe_tmp: MetalTensor | None = None
        self._buf_token_weights: MetalTensor | None = None
        self._buf_scaled: MetalTensor | None = None

    def forward(self, x: MetalTensor, batch: int, seq_len: int) -> MetalTensor:
        """MoE forward pass with GPU-native routing and accumulation."""
        num_tokens = batch * seq_len

        topk_weights, topk_indices = self.router.forward(x, num_tokens)

        moe_shape = [num_tokens, self.hidden_dim]
        if self._buf_moe_acc is None or self._buf_moe_acc.shape != moe_shape:
            self._buf_moe_acc = MetalTensor.empty(self.ctx, moe_shape)
        if self._buf_moe_tmp is None or self._buf_moe_tmp.shape != moe_shape:
            self._buf_moe_tmp = MetalTensor.empty(self.ctx, moe_shape)
        moe_out = self._buf_moe_acc
        # zero-initialize accumulator
        self.ctx.dispatch_kernel("scale", [x, moe_out, 0.0], grid_size=num_tokens * self.hidden_dim)

        for expert_idx in range(self.n_experts):
            if self._buf_token_weights is None or self._buf_token_weights.shape != [num_tokens]:
                self._buf_token_weights = MetalTensor.empty(self.ctx, [num_tokens])
            token_weights = self._buf_token_weights
            self.ctx.dispatch_kernel(
                "moe_topk_extract_weight",
                [topk_indices, topk_weights, token_weights, self.top_k, expert_idx],
                grid_size=num_tokens,
            )

            expert_out = self.experts[expert_idx].forward(x, num_tokens)

            if self._buf_scaled is None or self._buf_scaled.shape != moe_shape:
                self._buf_scaled = MetalTensor.empty(self.ctx, moe_shape)
            scaled = self._buf_scaled
            self.ctx.dispatch_kernel(
                "row_scale",
                [expert_out, token_weights, scaled, self.hidden_dim],
                grid_size=num_tokens * self.hidden_dim,
            )

            updated = self._buf_moe_tmp
            self.ctx.dispatch_kernel(
                "add",
                [moe_out, scaled, updated],
                grid_size=num_tokens * self.hidden_dim,
            )
            moe_out, self._buf_moe_tmp = updated, moe_out  # swap buffers

        return moe_out


class MetalTransformerBlock:
    """Single transformer block with MQA and MoE, on GPU."""

    def __init__(self, ctx: MetalContext, config: Config):
        self.ctx = ctx
        self.config = config

        from .metal_attention import MetalMQAttention

        self.attn_norm = MetalRMSNorm(ctx, config.hidden_dim)
        self.attention = MetalMQAttention(ctx, config)
        self.ffn_norm = MetalRMSNorm(ctx, config.hidden_dim)
        self.moe = MetalMoELayer(ctx, config)

    def forward(self, x: MetalTensor, batch: int, seq_len: int) -> MetalTensor:
        """Forward pass with residual connections, all on GPU."""
        batch_seq = batch * seq_len
        n = batch_seq * self.config.hidden_dim

        normed = self.attn_norm.forward(x, batch_seq)
        attn_out = self.attention.forward(normed, batch, seq_len)

        h = MetalTensor.empty(self.ctx, [batch_seq, self.config.hidden_dim])
        self.ctx.dispatch_kernel("add", [x, attn_out, h], grid_size=n)

        normed2 = self.ffn_norm.forward(h, batch_seq)
        moe_out = self.moe.forward(normed2, batch, seq_len)

        out = MetalTensor.empty(self.ctx, [batch_seq, self.config.hidden_dim])
        self.ctx.dispatch_kernel("add", [h, moe_out, out], grid_size=n)

        return out
