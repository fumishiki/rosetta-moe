# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Router, MoELayer, and TransformerBlock for MoE Transformer.

Implements the Mixture-of-Experts pattern:
  MoE: output = sum_k(gate_k * Expert_k(x))   for top-k experts

The router selects top-k experts per token via softmax gating, and
the MoE layer dispatches tokens to experts using vectorized NumPy
operations (np.argpartition, np.add.at) instead of Python loops over tokens.
"""

from __future__ import annotations

import numpy as np

from .attention import MQAttention
from .config import Config
from .layers import Linear, RMSNorm, SwiGLU
from .tensor import Tensor


class Router:
    """MoE Router with top-k selection.

    Router: gate_probs = softmax(x @ W_gate)
            top_k_indices, top_k_weights = top_k(gate_probs, k)
            weights = top_k_weights / sum(top_k_weights)  (renormalize)
    """

    def __init__(self, hidden_dim: int, n_experts: int, top_k: int):
        """Initialize router."""
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k

        self.gate = Linear(hidden_dim, n_experts)

        # Cache for backward and aux loss
        self._last_input: Tensor | None = None
        self._last_weights: np.ndarray | None = None
        self._last_indices: np.ndarray | None = None
        self._last_gate_probs: np.ndarray | None = None

    def forward(self, x: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Compute router weights and expert indices.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            weights: [num_tokens, top_k] — renormalized gate probabilities
            indices: np.ndarray [num_tokens, top_k] — expert indices per token
        """
        self._last_input = x

        batch, seq_len, _ = x.shape
        num_tokens = batch * seq_len

        # Flatten to [num_tokens, hidden_dim] using raw numpy to avoid Tensor overhead
        flat_x = Tensor.from_numpy(x.data.reshape(num_tokens, self.hidden_dim))

        # Gate logits -> softmax probabilities
        gate_logits = self.gate.forward(flat_x)
        gate_probs = gate_logits.softmax()
        self._last_gate_probs = gate_probs.data

        # Top-k selection via np.argpartition: O(N) average vs O(N log N) for full sort.
        # argpartition only guarantees the top-k elements are in the first k positions,
        # so we sort within the top-k afterwards for deterministic ordering.
        probs_2d = gate_probs.data  # [num_tokens, n_experts]
        part_idx = np.argpartition(-probs_2d, self.top_k, axis=-1)[:, :self.top_k]
        top_weights_unsorted = np.take_along_axis(probs_2d, part_idx, axis=-1)
        # Sort the top-k by descending probability for deterministic ordering
        sort_idx = np.argsort(-top_weights_unsorted, axis=-1)
        top_indices = np.take_along_axis(part_idx, sort_idx, axis=-1)
        top_weights = np.take_along_axis(top_weights_unsorted, sort_idx, axis=-1)
        # Renormalize so top-k weights sum to 1 per token
        top_weights /= np.sum(top_weights, axis=-1, keepdims=True)

        # Keep as numpy arrays (avoid .tolist() Python list conversion overhead)
        self._last_weights = top_weights
        self._last_indices = top_indices
        return top_weights, top_indices

    def backward(self, grad_output: Tensor) -> Tensor:
        """Simplified backward - returns zeros of input shape."""
        if self._last_input is None:
            raise RuntimeError("backward called before forward")
        return Tensor.zeros(self._last_input.shape)

    def parameters(self) -> list[Tensor]:
        """Return gate parameters."""
        return self.gate.parameters()

    def compute_aux_loss(self, alpha: float = 0.01) -> float:
        """Compute load balancing auxiliary loss.

        Aux loss: L_aux = alpha * N * sum_i(f_i * P_i)

        where:
          N = number of experts
          f_i = fraction of tokens routed to expert i
          P_i = mean gate probability for expert i

        This encourages uniform expert utilization. When all experts are
        equally used, f_i = P_i = 1/N and L_aux = alpha.
        """
        if self._last_gate_probs is None or self._last_indices is None:
            return 0.0

        num_tokens = self._last_indices.shape[0]

        # f_i: fraction of tokens assigned to each expert (vectorized via np.bincount)
        # _last_indices is already a numpy array, just ravel it
        flat_indices = self._last_indices.ravel()
        expert_counts = np.bincount(flat_indices, minlength=self.n_experts).astype(np.float32)

        total_assignments = num_tokens * self.top_k
        expert_counts /= total_assignments

        # P_i: mean gate probability per expert
        expert_probs = np.sum(self._last_gate_probs, axis=0) / num_tokens

        # L_aux = alpha * N * dot(f, P)
        aux_loss = alpha * self.n_experts * float(np.dot(expert_counts, expert_probs))
        return aux_loss


class MoELayer:
    """Mixture of Experts layer.

    For each token, the router selects top-k experts and their weights.
    The token is then processed by each selected expert (a SwiGLU FFN),
    and the outputs are combined as a weighted sum.

    MoE: output_t = sum_{e in top_k(t)} w_{t,e} * Expert_e(x_t)

    Dispatch strategy: iterate over experts (not tokens), batching all
    tokens assigned to each expert into a single forward call.  This is
    more cache-friendly and enables BLAS-level batching inside SwiGLU.
    """

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

        MoE: output_t = sum_{e in top_k(t)} w_{t,e} * Expert_e(x_t)

        Expert-batched dispatch: iterate over experts (not tokens), batching all
        tokens assigned to each expert into a single forward call. This enables
        BLAS-level batching inside SwiGLU.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape
        num_tokens = batch * seq_len

        # Get router weights and indices (both are numpy arrays now)
        weights, indices_arr = self.router.forward(x)

        # Flatten input using raw numpy to avoid Tensor overhead
        flat_x = x.data.reshape(num_tokens, self.hidden_dim)

        # Cache for backward pass
        self._last_input = x
        self._last_flat_x = flat_x
        self._last_weights = weights
        self._last_indices = indices_arr
        self._last_batch_shape = (batch, seq_len)

        # Expert-batched dispatch: iterate over experts, not tokens.
        output = np.zeros((num_tokens, self.hidden_dim), dtype=np.float32)

        for expert_idx in range(self.n_experts):
            # Vectorized: boolean mask of all (token, k) pairs assigned to this expert
            mask = indices_arr == expert_idx  # [num_tokens, top_k]
            token_mask = np.any(mask, axis=1)  # [num_tokens] — True if any k-slot hits this expert
            if not np.any(token_mask):
                continue

            token_indices = np.where(token_mask)[0]
            # Sum weights for this expert per token (handles rare duplicate assignments)
            token_weights_arr = np.sum(weights[token_mask] * mask[token_mask], axis=1, keepdims=True).astype(np.float32)

            batch_input = Tensor.from_numpy(flat_x[token_indices])
            batch_output = self.experts[expert_idx].forward(batch_input)

            # Weighted accumulation using np.add.at for scatter-add semantics.
            np.add.at(output, token_indices, token_weights_arr * batch_output.data)

        return Tensor.from_numpy(output.reshape((batch, seq_len, self.hidden_dim)))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through MoE layer.

        Propagates gradients through each expert for the tokens it processed,
        weighted by the router weights. Accumulates expert parameter gradients.
        """
        batch, seq_len = self._last_batch_shape
        num_tokens = batch * seq_len

        flat_grad = grad_output.data.reshape(num_tokens, self.hidden_dim)
        weights = self._last_weights
        indices_arr = self._last_indices
        flat_x = self._last_flat_x

        grad_input = np.zeros((num_tokens, self.hidden_dim), dtype=np.float32)

        for expert_idx in range(self.n_experts):
            mask = indices_arr == expert_idx
            token_mask = np.any(mask, axis=1)
            if not np.any(token_mask):
                continue

            token_indices = np.where(token_mask)[0]
            token_weights_arr = np.sum(
                weights[token_mask] * mask[token_mask], axis=1, keepdims=True
            ).astype(np.float32)

            # Weighted gradient for this expert's output
            expert_grad = Tensor.from_numpy(flat_grad[token_indices] * token_weights_arr)

            # Set _last_input for the expert's sub-layers (re-run input through cache)
            expert_input = Tensor.from_numpy(flat_x[token_indices])
            self.experts[expert_idx]._last_input = expert_input
            self.experts[expert_idx].gate._last_input = expert_input
            self.experts[expert_idx].up._last_input = expert_input

            # Backward through the expert (accumulates weight gradients)
            grad_expert_input = self.experts[expert_idx].backward(expert_grad)

            # Accumulate input gradient (scatter-add for multiple experts per token)
            np.add.at(grad_input, token_indices, grad_expert_input.data)

        return Tensor.from_numpy(grad_input.reshape(batch, seq_len, self.hidden_dim))

    def parameters(self) -> list[Tensor]:
        """Return all parameters (router + all experts)."""
        params = self.router.parameters()
        for expert in self.experts:
            params.extend(expert.parameters())
        return params

    def aux_loss(self, alpha: float = 0.01) -> float:
        """Return auxiliary loss."""
        return self.router.compute_aux_loss(alpha)


class TransformerBlock:
    """Single transformer block with MQA and MoE.

    Architecture (pre-norm residual):
      h = x + Attention(RMSNorm(x))      -- pre-norm attention + residual
      h = h + MoE(RMSNorm(h))            -- pre-norm MoE FFN + residual
    """

    def __init__(self, config: Config):
        """Initialize transformer block."""
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attention = MQAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.moe = MoELayer(config)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connections.

        h = x + Attention(RMSNorm(x))
        out = h + MoE(RMSNorm(h))

        Creates new tensors for residual connections (matching Rust) to
        avoid corrupting RMSNorm cached inputs needed for backward.

        Args:
            x: Input [batch, seq_len, hidden_dim]

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        # Pre-norm attention with residual
        normed = self.attn_norm.forward(x)
        attn_out = self.attention.forward(normed)
        h = Tensor.from_numpy(x.data + attn_out.data)

        # Pre-norm FFN (MoE) with residual
        normed = self.ffn_norm.forward(h)
        moe_out = self.moe.forward(normed)
        return Tensor.from_numpy(h.data + moe_out.data)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through transformer block.

        Forward: h = x + Attention(RMSNorm_attn(x))
                 out = h + MoE(RMSNorm_ffn(h))

        Backward (matching Rust):
          grad_ffn = ffn_norm.backward(grad_output)
          grad_moe = moe.backward(grad_ffn)
          grad_h   = grad_output + grad_moe                  (residual)

          grad_attn_norm = attn_norm.backward(grad_h)
          grad_attn      = attention.backward(grad_attn_norm)
          grad_x         = grad_h + grad_attn                (residual)
        """
        # MoE residual path
        grad_ffn = self.ffn_norm.backward(grad_output)
        grad_moe = self.moe.backward(grad_ffn)
        grad_h = Tensor.from_numpy(grad_output.data + grad_moe.data)

        # Attention residual path
        grad_attn_norm = self.attn_norm.backward(grad_h)
        grad_attn = self.attention.backward(grad_attn_norm)
        grad_x = Tensor.from_numpy(grad_h.data + grad_attn.data)

        return grad_x

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
