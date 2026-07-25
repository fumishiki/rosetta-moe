# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Router, MoELayer, and TransformerBlock for MoE Transformer.

Implements the Mixture-of-Experts pattern:
  MoE: output = sum_k(gate_k * Expert_k(x))   for top-k experts

The router selects top-k experts per token via softmax gating, and
the MoE layer dispatches tokens to experts using vectorized NumPy
operations (np.argpartition, np.add.at) instead of Python loops over tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .attention import MQAttention
from .config import Config
from .layers import Linear, RMSNorm, SwiGLU
from .tensor import Tensor

if TYPE_CHECKING:
    from .train import RoutingMode


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
        self._last_logits: np.ndarray | None = None
        self._probs_buf: np.ndarray | None = None

        # Phase 2: Routing mode extensions
        from .train import RoutingMode
        self.routing_mode: RoutingMode = RoutingMode.TOPK
        self.expert_bias: np.ndarray = np.zeros(n_experts, dtype=np.float32)
        self.relu_lambda_l1: float = 0.01
        self.last_avg_active: float = 0.0
        self._last_expert_counts: np.ndarray = np.zeros(n_experts, dtype=np.float32)
        self.last_relu_sum: float = 0.0

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

        # Gate logits -> softmax probabilities (in-place into reusable buffer)
        gate_logits = self.gate.forward(flat_x).data
        # Store pre-softmax logits for z-loss and ReLU mode
        self._last_logits = gate_logits.copy()
        if self._probs_buf is None or self._probs_buf.shape != gate_logits.shape:
            self._probs_buf = np.empty_like(gate_logits)
        probs_2d = self._probs_buf
        np.subtract(gate_logits, np.max(gate_logits, axis=-1, keepdims=True), out=probs_2d)
        np.exp(probs_2d, out=probs_2d)
        np.divide(probs_2d, np.sum(probs_2d, axis=-1, keepdims=True), out=probs_2d)
        self._last_gate_probs = probs_2d

        # Branch by routing mode
        from .train import RoutingMode
        if self.routing_mode == RoutingMode.TOPK:
            # Original TopK routing
            part_idx = np.argpartition(-probs_2d, self.top_k - 1, axis=-1)[:, :self.top_k]
            top_weights_unsorted = np.take_along_axis(probs_2d, part_idx, axis=-1)
            sort_idx = np.argsort(-top_weights_unsorted, axis=-1)
            top_indices = np.take_along_axis(part_idx, sort_idx, axis=-1)
            top_weights = np.take_along_axis(top_weights_unsorted, sort_idx, axis=-1)
            top_weights /= np.maximum(np.sum(top_weights, axis=-1, keepdims=True), 1e-12)
            # Track expert counts for aux_loss gradient
            flat_indices = top_indices.ravel()
            expert_counts = np.bincount(flat_indices, minlength=self.n_experts).astype(np.float32)
            self._last_expert_counts = expert_counts

        elif self.routing_mode == RoutingMode.BIAS_FREE:
            # BiasFree routing (DeepSeek-V3)
            # Selection scores = probs + expert_bias
            selection_scores = probs_2d + self.expert_bias[None, :]
            # Select top-k using selection_scores
            part_idx = np.argpartition(-selection_scores, self.top_k - 1, axis=-1)[:, :self.top_k]
            sort_idx_sel = np.argsort(-np.take_along_axis(selection_scores, part_idx, axis=-1), axis=-1)
            top_indices = np.take_along_axis(part_idx, sort_idx_sel, axis=-1)
            # Extract weights from ORIGINAL probs (not selection_scores)
            top_weights = np.take_along_axis(probs_2d, top_indices, axis=-1)
            top_weights /= np.maximum(np.sum(top_weights, axis=-1, keepdims=True), 1e-12)
            # Track expert counts for bias update
            flat_indices = top_indices.ravel()
            expert_counts = np.bincount(flat_indices, minlength=self.n_experts).astype(np.float32)
            self._last_expert_counts = expert_counts

        elif self.routing_mode == RoutingMode.RELU:
            # ReMoE (ReLU routing)
            gate_weights = np.maximum(0.0, gate_logits)
            top_indices = np.zeros((num_tokens, self.top_k), dtype=np.int64)
            top_weights = np.zeros((num_tokens, self.top_k), dtype=np.float32)
            active_count = 0
            relu_sum = 0.0

            for t in range(num_tokens):
                active = np.where(gate_weights[t] > 0)[0]
                if len(active) == 0:
                    # Fallback: use best logit
                    best = np.argmax(gate_logits[t])
                    active = np.array([best])
                    gate_weights[t, best] = 1.0
                if len(active) > self.top_k:
                    # Limit to top-k by gate_weights
                    top_active_idx = np.argsort(-gate_weights[t, active])[:self.top_k]
                    active = active[top_active_idx]

                # Renormalize
                weights_active = gate_weights[t, active]
                sum_w = np.sum(weights_active)
                if sum_w > 1e-12:
                    weights_active /= sum_w

                # Pad to top_k
                k = len(active)
                top_indices[t, :k] = active
                top_weights[t, :k] = weights_active
                if k < self.top_k:
                    # Pad with first expert, zero weight
                    top_indices[t, k:] = 0
                    top_weights[t, k:] = 0.0

                active_count += k
                relu_sum += np.sum(np.maximum(0.0, gate_logits[t]))

            self.last_avg_active = active_count / max(num_tokens, 1)
            self.last_relu_sum = relu_sum
        else:
            raise ValueError(f"Unknown routing mode: {self.routing_mode}")

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

    def compute_z_loss_with_grad(self, z_weight: float = 0.01) -> float:
        """Compute router z-loss with gradient backprop to gate weights.

        Z-loss: L_z = z_weight * (1/B) * sum_i(logsumexp(logits_i)^2)

        where logits_i is the pre-softmax gate logits for token i.

        The gradient w.r.t. logits is:
          dL_z/d(logits[i,j]) = z_weight * (2/B) * logsumexp(logits_i) * gate_probs[i,j]

        This gradient is then backpropagated to gate weights via:
          gate_weight_grad[e,h] += sum_t(grad_logits[t,e] * last_input[t,h])
        """
        if self._last_logits is None or self._last_gate_probs is None or self._last_input is None:
            return 0.0

        logits = self._last_logits
        gate_probs = self._last_gate_probs
        num_tokens = logits.shape[0]

        # Compute logsumexp for each token using max-subtract trick
        max_vals = np.max(logits, axis=-1, keepdims=True)
        lse = max_vals.ravel() + np.log(np.sum(np.exp(logits - max_vals), axis=-1))

        # Z-loss: (1/B) * sum(lse^2)
        z_loss = z_weight * float(np.sum(lse ** 2)) / num_tokens

        # Gradient w.r.t. logits: (2/B) * lse * gate_probs
        grad_logits = (z_weight * 2.0 / num_tokens) * lse[:, None] * gate_probs

        # Backprop to gate weights: gate_weight_grad = grad_logits.T @ last_input
        # Flatten last_input from [batch, seq_len, hidden_dim] to [num_tokens, hidden_dim]
        input_shape = self._last_input.shape
        batch, seq_len, hidden_dim = input_shape
        flat_input = self._last_input.data.reshape(num_tokens, hidden_dim)

        # Accumulate gradient to gate weight
        grad_weight = np.matmul(grad_logits.T, flat_input)
        if self.gate.weight._grad is None:
            self.gate.weight._grad = grad_weight.astype(np.float32)
        else:
            self.gate.weight._grad += grad_weight.astype(np.float32)

        # If gate has bias, accumulate gradient to bias
        if self.gate.bias is not None:
            grad_bias = np.sum(grad_logits, axis=0)
            if self.gate.bias._grad is None:
                self.gate.bias._grad = grad_bias.astype(np.float32)
            else:
                self.gate.bias._grad += grad_bias.astype(np.float32)

        return z_loss

    def compute_aux_loss_with_grad(self, alpha: float = 0.01) -> float:
        """Compute aux loss with gradient backprop to gate weights.

        Aux loss: L_aux = alpha * N * sum_e(f_e * P_e)

        where:
          f_e = fraction of tokens assigned to expert e (computed from top-k selections)
          P_e = mean gate probability for expert e (from softmax output)
          N = number of experts

        The gradient w.r.t. logits is:
          dL_aux/d(logits[t,e]) = alpha * N / batch_seq * probs[t,e] * (f_e - dot(f, probs[t]))

        This gradient is then backpropagated to gate weights via:
          gate_weight_grad[e,h] += sum_t(grad_logits[t,e] * last_input[t,h])
        """
        if self._last_gate_probs is None or self._last_indices is None or self._last_input is None:
            return 0.0

        gate_probs = self._last_gate_probs
        num_tokens = self._last_indices.shape[0]

        # Compute f_e: fraction of tokens assigned to each expert
        flat_indices = self._last_indices.ravel()
        expert_counts = np.bincount(flat_indices, minlength=self.n_experts).astype(np.float32)
        total_assignments = num_tokens * self.top_k
        if total_assignments < 1e-12:
            return 0.0
        f_e = expert_counts / total_assignments

        # Compute P_e: mean gate probability per expert
        P_e = np.sum(gate_probs, axis=0) / num_tokens

        # Aux loss: alpha * N * dot(f, P)
        aux_loss = alpha * self.n_experts * float(np.dot(f_e, P_e))

        # Gradient w.r.t. logits:
        # For each token t: grad_logits[t,e] = alpha * N / batch_seq * probs[t,e] * (f[e] - dot(f, probs[t]))
        # dot_fp_t = sum_e'(f[e'] * probs[t,e']) for each token t
        dot_fp = np.dot(gate_probs, f_e)  # [num_tokens]
        grad_logits = (alpha * self.n_experts / num_tokens) * gate_probs * (f_e[None, :] - dot_fp[:, None])

        # Backprop to gate weights: gate_weight_grad = grad_logits.T @ last_input
        # Flatten last_input from [batch, seq_len, hidden_dim] to [num_tokens, hidden_dim]
        input_shape = self._last_input.shape
        batch, seq_len, hidden_dim = input_shape
        flat_input = self._last_input.data.reshape(num_tokens, hidden_dim)

        # Accumulate gradient to gate weight
        grad_weight = np.matmul(grad_logits.T, flat_input)
        if self.gate.weight._grad is None:
            self.gate.weight._grad = grad_weight.astype(np.float32)
        else:
            self.gate.weight._grad += grad_weight.astype(np.float32)

        # If gate has bias, accumulate gradient to bias
        if self.gate.bias is not None:
            grad_bias = np.sum(grad_logits, axis=0)
            if self.gate.bias._grad is None:
                self.gate.bias._grad = grad_bias.astype(np.float32)
            else:
                self.gate.bias._grad += grad_bias.astype(np.float32)

        return aux_loss

    def update_expert_bias(self, gamma: float) -> None:
        """Update expert bias for BiasFree routing mode.

        Args:
            gamma: Learning rate for bias update
        """
        from .train import RoutingMode
        if self.routing_mode != RoutingMode.BIAS_FREE:
            return
        total_assignments = np.sum(self._last_expert_counts)
        if total_assignments < 1e-12:
            return
        f_e = self._last_expert_counts / total_assignments
        target = 1.0 / self.n_experts
        self.expert_bias += gamma * np.sign(target - f_e)

    def compute_relu_l1_loss_with_grad(self) -> float:
        """Compute ReLU L1 regularization loss with gradient backprop.

        Returns:
            L1 loss value
        """
        from .train import RoutingMode
        if self.routing_mode != RoutingMode.RELU or self._last_logits is None or self._last_input is None:
            return 0.0

        logits = self._last_logits
        num_tokens = logits.shape[0]
        relu_vals = np.maximum(0.0, logits)
        l1_loss = self.relu_lambda_l1 * float(np.mean(np.sum(relu_vals, axis=-1)))

        # Gradient: lambda * (1/batch_seq) * (logits > 0)
        grad_logits = (self.relu_lambda_l1 / num_tokens) * (logits > 0).astype(np.float32)

        # Backprop to gate weights
        input_shape = self._last_input.shape
        batch, seq_len, hidden_dim = input_shape
        flat_input = self._last_input.data.reshape(num_tokens, hidden_dim)
        grad_weight = np.matmul(grad_logits.T, flat_input)

        if self.gate.weight._grad is None:
            self.gate.weight._grad = grad_weight.astype(np.float32)
        else:
            self.gate.weight._grad += grad_weight.astype(np.float32)

        if self.gate.bias is not None:
            grad_bias = np.sum(grad_logits, axis=0)
            if self.gate.bias._grad is None:
                self.gate.bias._grad = grad_bias.astype(np.float32)
            else:
                self.gate.bias._grad += grad_bias.astype(np.float32)

        return l1_loss


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
        self._last_dispatch_tokens: list[np.ndarray | None] = [None] * config.n_experts
        self._last_dispatch_weights: list[np.ndarray | None] = [None] * config.n_experts
        self._token_ids_cache: np.ndarray | None = None

    def _token_ids(self, num_tokens: int) -> np.ndarray:
        """Return cached [0..num_tokens-1] token ids."""
        if self._token_ids_cache is None or self._token_ids_cache.shape[0] != num_tokens:
            self._token_ids_cache = np.arange(num_tokens, dtype=np.int64)
        return self._token_ids_cache

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

        self._last_batch_shape = (batch, seq_len)

        # Expert-batched dispatch (Rust-style inversion):
        #   token -> top-k experts  =>  expert -> assigned token list
        output = np.zeros((num_tokens, self.hidden_dim), dtype=np.float32)
        token_ids = self._token_ids(num_tokens)
        flat_token_ids = np.repeat(token_ids, self.top_k)
        flat_experts = indices_arr.reshape(-1)
        flat_weights = weights.reshape(-1).astype(np.float32, copy=False)
        order = np.argsort(flat_experts, kind="stable")
        sorted_experts = flat_experts[order]
        sorted_tokens = flat_token_ids[order]
        sorted_weights = flat_weights[order]
        bounds = np.searchsorted(sorted_experts, np.arange(self.n_experts + 1), side="left")

        # Reset dispatch cache
        self._last_dispatch_tokens = [None] * self.n_experts
        self._last_dispatch_weights = [None] * self.n_experts

        for expert_idx in range(self.n_experts):
            start = bounds[expert_idx]
            end = bounds[expert_idx + 1]
            if start == end:
                continue

            token_indices = sorted_tokens[start:end]
            token_weights = sorted_weights[start:end]
            self._last_dispatch_tokens[expert_idx] = token_indices
            self._last_dispatch_weights[expert_idx] = token_weights

            batch_input = Tensor.from_numpy(flat_x[token_indices])
            batch_output = self.experts[expert_idx].forward(batch_input)

            # token_indices are unique for a given expert (top-k has unique experts per token),
            # so direct indexed accumulation is safe and faster than np.add.at.
            output[token_indices] += token_weights[:, None] * batch_output.data

        return Tensor.from_numpy(output.reshape((batch, seq_len, self.hidden_dim)))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through MoE layer.

        Propagates gradients through each expert for the tokens it processed,
        weighted by the router weights. Accumulates expert parameter gradients.
        """
        batch, seq_len = self._last_batch_shape
        num_tokens = batch * seq_len

        flat_grad = grad_output.data.reshape(num_tokens, self.hidden_dim)

        grad_input = np.zeros((num_tokens, self.hidden_dim), dtype=np.float32)

        for expert_idx in range(self.n_experts):
            token_indices = self._last_dispatch_tokens[expert_idx]
            token_weights = self._last_dispatch_weights[expert_idx]
            if token_indices is None or token_weights is None:
                continue

            # Weighted gradient for this expert's output
            expert_grad = Tensor.from_numpy(flat_grad[token_indices] * token_weights[:, None])

            # Backward through the expert (accumulates weight gradients)
            grad_expert_input = self.experts[expert_idx].backward(expert_grad)

            # token_indices are unique for this expert, so direct accumulation is safe.
            grad_input[token_indices] += grad_expert_input.data

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
