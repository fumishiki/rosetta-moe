"""MoE Transformer model implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .layers import Embedding, Linear, RMSNorm, SwiGLU
from .tensor import DType, Tensor


@dataclass
class Config:
    """Model configuration."""

    hidden_dim: int = 768
    n_layers: int = 30
    n_heads: int = 12
    n_kv_heads: int = 1  # MQA
    n_experts: int = 16
    top_k_experts: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 32768
    ffn_dim: int = 6144
    head_dim: int = 64
    rope_base: float = 10000.0
    rope_alpha: float = 8.0  # NTK scaling for 256K inference

    @classmethod
    def default_6_9b(cls) -> Config:
        """Return default 6.9B model configuration."""
        return cls()

    @classmethod
    def tiny(cls) -> Config:
        """Return tiny model configuration for testing."""
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

    def total_params(self) -> int:
        """Estimate total parameters."""
        embedding = self.vocab_size * self.hidden_dim
        attention = (
            self.hidden_dim * self.hidden_dim * 2
            + self.hidden_dim * self.head_dim * 2
        )
        router = self.hidden_dim * self.n_experts
        expert_ffn = self.hidden_dim * self.ffn_dim * 3 * self.n_experts
        norms = self.hidden_dim * 2
        per_layer = attention + router + expert_ffn + norms
        lm_head = self.hidden_dim * self.vocab_size
        return embedding + per_layer * self.n_layers + lm_head

    def active_params(self) -> int:
        """Estimate active parameters per token."""
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


class MQAttention:
    """Multi-Query Attention with RoPE."""

    def __init__(self, config: Config):
        """Initialize attention layer."""
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        # Projections
        self.w_q = Linear(config.hidden_dim, config.n_heads * config.head_dim)
        self.w_k = Linear(config.hidden_dim, config.n_kv_heads * config.head_dim)
        self.w_v = Linear(config.hidden_dim, config.n_kv_heads * config.head_dim)
        self.w_o = Linear(config.n_heads * config.head_dim, config.hidden_dim)

        # RoPE parameters
        self.rope_base = config.rope_base
        self.rope_alpha = config.rope_alpha

    def _compute_rope_freqs(self, seq_len: int) -> np.ndarray:
        """Compute RoPE frequencies."""
        base = self.rope_base
        if self.rope_alpha > 1.0:
            # NTK scaling
            base = self.rope_base * (
                self.rope_alpha ** (self.head_dim / (self.head_dim - 2))
            )

        freqs = np.array(
            [
                1.0 / (base ** (2 * i / self.head_dim))
                for i in range(self.head_dim // 2)
            ]
        )
        return freqs

    def _apply_rope(
        self, q: np.ndarray, k: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Rotary Position Embedding."""
        freqs = self._compute_rope_freqs(seq_len)
        positions = np.arange(seq_len)

        # q: [batch, seq_len, n_heads, head_dim]
        # k: [batch, seq_len, n_kv_heads, head_dim]
        q_out = np.zeros_like(q)
        k_out = np.zeros_like(k)

        for s in range(seq_len):
            for i in range(self.head_dim // 2):
                angle = positions[s] * freqs[i]
                cos_val = np.cos(angle)
                sin_val = np.sin(angle)

                # Apply to Q
                q_out[:, s, :, 2 * i] = (
                    q[:, s, :, 2 * i] * cos_val - q[:, s, :, 2 * i + 1] * sin_val
                )
                q_out[:, s, :, 2 * i + 1] = (
                    q[:, s, :, 2 * i] * sin_val + q[:, s, :, 2 * i + 1] * cos_val
                )

                # Apply to K
                k_out[:, s, :, 2 * i] = (
                    k[:, s, :, 2 * i] * cos_val - k[:, s, :, 2 * i + 1] * sin_val
                )
                k_out[:, s, :, 2 * i + 1] = (
                    k[:, s, :, 2 * i] * sin_val + k[:, s, :, 2 * i + 1] * cos_val
                )

        return q_out, k_out

    def forward(self, x: Tensor) -> Tensor:
        """Multi-query attention forward pass.

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

        # Reshape to [batch, seq_len, n_heads, head_dim]
        q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))
        k = k.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))
        v = v.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))

        # Apply RoPE
        q_data, k_data = self._apply_rope(q.data, k.data, seq_len)

        # Compute attention scores with causal mask
        output = np.zeros((batch, seq_len, self.n_heads, self.head_dim))

        for b in range(batch):
            for h in range(self.n_heads):
                kv_head = h % self.n_kv_heads

                # Attention scores [seq_len, seq_len]
                scores = np.zeros((seq_len, seq_len))
                for qi in range(seq_len):
                    for ki in range(seq_len):
                        if ki > qi:  # Causal mask
                            scores[qi, ki] = float("-inf")
                        else:
                            dot = np.dot(
                                q_data[b, qi, h], k_data[b, ki, kv_head]
                            )
                            scores[qi, ki] = dot * self.scale

                # Softmax per query
                for qi in range(seq_len):
                    row = scores[qi, : qi + 1]
                    max_val = np.max(row)
                    exp_vals = np.exp(row - max_val)
                    scores[qi, : qi + 1] = exp_vals / np.sum(exp_vals)
                    scores[qi, qi + 1 :] = 0

                # Compute output
                for qi in range(seq_len):
                    for ki in range(qi + 1):
                        output[b, qi, h] += scores[qi, ki] * v.data[b, ki, kv_head]

        # Reshape and project output
        output = output.reshape((batch, seq_len, self.n_heads * self.head_dim))
        return self.w_o.forward(Tensor.from_numpy(output))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Simplified backward pass."""
        return self.w_o.backward(grad_output)

    def parameters(self) -> list[Tensor]:
        """Return all parameters."""
        return (
            self.w_q.parameters()
            + self.w_k.parameters()
            + self.w_v.parameters()
            + self.w_o.parameters()
        )


class Router:
    """MoE Router with top-k selection."""

    def __init__(self, hidden_dim: int, n_experts: int, top_k: int):
        """Initialize router."""
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = Linear(hidden_dim, n_experts)

        # Cache for backward and aux loss
        self._last_input: Tensor | None = None
        self._last_weights: np.ndarray | None = None
        self._last_indices: list[list[int]] | None = None
        self._last_gate_probs: np.ndarray | None = None

    def forward(self, x: Tensor) -> tuple[np.ndarray, list[list[int]]]:
        """Compute router weights and expert indices.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            weights: [num_tokens, top_k]
            indices: List of expert indices per token
        """
        self._last_input = x.clone()

        batch, seq_len, _ = x.shape
        num_tokens = batch * seq_len

        # Flatten to [num_tokens, hidden_dim]
        flat_x = x.reshape((num_tokens, self.hidden_dim))

        # Gate logits
        gate_logits = self.gate.forward(flat_x)

        # Softmax
        gate_probs = gate_logits.softmax()
        self._last_gate_probs = gate_probs.data

        # Top-k selection
        weights = np.zeros((num_tokens, self.top_k))
        indices = []

        for t in range(num_tokens):
            probs = gate_probs.data[t]
            top_k_idx = np.argsort(probs)[-self.top_k :][::-1]
            indices.append(list(top_k_idx))

            # Normalize top-k weights
            top_k_probs = probs[top_k_idx]
            weights[t] = top_k_probs / np.sum(top_k_probs)

        self._last_weights = weights
        self._last_indices = indices
        return weights, indices

    def backward(self, grad_output: Tensor) -> Tensor:
        """Simplified backward - returns zeros of input shape."""
        if self._last_input is None:
            raise RuntimeError("backward called before forward")
        return Tensor.zeros(self._last_input.shape)

    def parameters(self) -> list[Tensor]:
        """Return gate parameters."""
        return self.gate.parameters()

    def compute_aux_loss(self, alpha: float = 0.01) -> float:
        """Compute load balancing auxiliary loss."""
        if self._last_gate_probs is None or self._last_indices is None:
            return 0.0

        num_tokens = len(self._last_indices)

        # Count tokens per expert
        expert_counts = np.zeros(self.n_experts)
        expert_probs = np.zeros(self.n_experts)

        for t in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = self._last_indices[t][k]
                expert_counts[expert_idx] += 1
            expert_probs += self._last_gate_probs[t]

        # Normalize
        total_assignments = num_tokens * self.top_k
        expert_counts /= total_assignments
        expert_probs /= num_tokens

        # Aux loss = alpha * N * sum(f_i * P_i)
        aux_loss = alpha * self.n_experts * np.sum(expert_counts * expert_probs)
        return float(aux_loss)


class MoELayer:
    """Mixture of Experts layer."""

    def __init__(self, config: Config):
        """Initialize MoE layer."""
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.n_experts = config.n_experts
        self.top_k = config.top_k_experts

        self.router = Router(config.hidden_dim, config.n_experts, config.top_k_experts)
        self.experts = [
            SwiGLU(config.hidden_dim, config.ffn_dim) for _ in range(config.n_experts)
        ]

    def forward(self, x: Tensor) -> Tensor:
        """MoE forward pass.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape
        num_tokens = batch * seq_len

        # Get router weights and indices
        weights, indices = self.router.forward(x)

        # Flatten input
        flat_x = x.reshape((num_tokens, self.hidden_dim))

        # Initialize output
        output = np.zeros((num_tokens, self.hidden_dim))

        # Process each token
        for t in range(num_tokens):
            token_input = Tensor.from_numpy(
                flat_x.data[t : t + 1]
            )  # [1, hidden_dim]

            for k in range(self.top_k):
                expert_idx = indices[t][k]
                weight = weights[t, k]

                expert_out = self.experts[expert_idx].forward(token_input)
                output[t] += weight * expert_out.data[0]

        return Tensor.from_numpy(output.reshape((batch, seq_len, self.hidden_dim)))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Simplified backward through router."""
        return self.router.backward(grad_output)

    def parameters(self) -> list[Tensor]:
        """Return all parameters."""
        params = self.router.parameters()
        for expert in self.experts:
            params.extend(expert.parameters())
        return params

    def aux_loss(self, alpha: float = 0.01) -> float:
        """Return auxiliary loss."""
        return self.router.compute_aux_loss(alpha)


class TransformerBlock:
    """Single transformer block with MQA and MoE."""

    def __init__(self, config: Config):
        """Initialize transformer block."""
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attention = MQAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.moe = MoELayer(config)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        # Pre-norm attention
        normed = self.attn_norm.forward(x)
        attn_out = self.attention.forward(normed)
        x = x + attn_out  # Residual

        # Pre-norm FFN (MoE)
        normed = self.ffn_norm.forward(x)
        moe_out = self.moe.forward(normed)
        x = x + moe_out  # Residual

        return x

    def backward(self, grad_output: Tensor) -> Tensor:
        """Simplified backward."""
        grad_moe = self.moe.backward(grad_output)
        grad_attn = self.attention.backward(grad_moe)
        return grad_attn

    def parameters(self) -> list[Tensor]:
        """Return all parameters."""
        return (
            self.attn_norm.parameters()
            + self.attention.parameters()
            + self.ffn_norm.parameters()
            + self.moe.parameters()
        )

    def aux_loss(self, alpha: float = 0.01) -> float:
        """Return MoE auxiliary loss."""
        return self.moe.aux_loss(alpha)


class MoETransformer:
    """Full MoE Transformer model."""

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

        Args:
            x: Token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Embedding
        h = self.embedding.forward(x)

        # Transformer blocks
        for block in self.blocks:
            h = block.forward(h)

        # Final norm and LM head
        h = self.final_norm.forward(h)
        logits = self.lm_head.forward(h)

        return logits

    def forward_ids(
        self, token_ids: list[int], batch: int = 1, seq_len: int | None = None
    ) -> Tensor:
        """Forward pass with token ID list.

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
        """Backward pass through all blocks."""
        grad = grad_output
        for block in reversed(self.blocks):
            grad = block.backward(grad)
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
