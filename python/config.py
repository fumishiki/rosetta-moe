# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Model configuration for MoE Transformer.

Defines the hyperparameters for a 6.9B-total / 1.8B-active MoE Transformer
(the "default" config) and a tiny variant for unit tests and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Model configuration.

    Architecture: Multi-Query Attention + Mixture-of-Experts FFN.
    The default config targets 6.9B total params with 1.8B active per token
    (top-4 out of 16 experts).

    Key relationships:
      Q projection size  = n_heads    * head_dim
      KV projection size = n_kv_heads * head_dim   (MQA: n_kv_heads << n_heads)
      Expert FFN params  = hidden_dim * ffn_dim * 3  (gate + up + down)
    """

    hidden_dim: int = 768
    n_layers: int = 30
    n_heads: int = 12
    n_kv_heads: int = 1  # MQA: single KV head shared across all Q heads
    n_experts: int = 16
    top_k_experts: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 32768
    ffn_dim: int = 6144
    head_dim: int = 64
    rope_base: float = 10000.0
    # NTK-aware RoPE scaling: base' = base * alpha^(d/(d-2))
    # alpha=8 extends context from 32K training to ~256K inference
    rope_alpha: float = 8.0

    @classmethod
    def default_6_9b(cls) -> Config:
        """Return default 6.9B model configuration."""
        return cls()

    @classmethod
    def tiny(cls) -> Config:
        """Return tiny model configuration for testing.

        Small enough for fast CPU tests (~seconds), but preserves
        the full architecture (MQA + MoE + RoPE).
        """
        return cls(
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            n_experts=4,
            top_k_experts=2,
            vocab_size=1000,
            max_seq_len=512,
            ffn_dim=256,
            head_dim=16,
            rope_base=10000.0,
            rope_alpha=1.0,
        )

    @classmethod
    def small(cls) -> Config:
        """Return small model configuration for scale comparison benchmarks.

        Same structure as tiny but 4x larger hidden dimension (256 vs 64).
        """
        return cls(
            hidden_dim=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            n_experts=4,
            top_k_experts=2,
            vocab_size=1000,
            max_seq_len=512,
            ffn_dim=1024,
            head_dim=64,
            rope_base=10000.0,
            rope_alpha=1.0,
        )

    def total_params(self) -> int:
        """Estimate total parameters.

        Breakdown per layer:
          attention = Q(h*h) + O(h*h) + K(h*d_kv) + V(h*d_kv)
          router    = h * n_experts
          expert_ffn = h * ffn * 3 * n_experts  (gate + up + down per expert)
          norms     = h * 2  (attn_norm + ffn_norm)
        """
        embedding = self.vocab_size * self.hidden_dim
        attention = (
            self.hidden_dim * self.hidden_dim * 2        # W_q + W_o
            + self.hidden_dim * self.head_dim * 2         # W_k + W_v (MQA)
        )
        router = self.hidden_dim * self.n_experts
        expert_ffn = self.hidden_dim * self.ffn_dim * 3 * self.n_experts
        norms = self.hidden_dim * 2
        per_layer = attention + router + expert_ffn + norms
        lm_head = self.hidden_dim * self.vocab_size
        return embedding + per_layer * self.n_layers + lm_head

    def active_params(self) -> int:
        """Estimate active parameters per token.

        Same as total_params but counts only top_k_experts instead of
        all n_experts in the FFN.
        """
        embedding = self.vocab_size * self.hidden_dim
        attention = (
            self.hidden_dim * self.hidden_dim * 2
            + self.hidden_dim * self.head_dim * 2
        )
        active_ffn = self.hidden_dim * self.ffn_dim * 3 * self.top_k_experts
        norms = self.hidden_dim * 2
        per_layer = attention + active_ffn + norms
        lm_head = self.hidden_dim * self.vocab_size
        return embedding + per_layer * self.n_layers + lm_head
