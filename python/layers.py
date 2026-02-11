# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Neural network layers for MoE Transformer.

Provides the foundational layers: Embedding, RMSNorm, Linear, and SwiGLU.
Every layer stores activations for the backward pass and exposes a
``parameters()`` method for optimizer integration.

All weight initializations use Kaiming (He) normal: std = sqrt(2/fan_in),
matching the Go and Julia implementations in this repo.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from .tensor import DType, Tensor


class Layer(ABC):
    """Base class for layers."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        ...

    @abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass."""
        ...

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        """Return layer parameters."""
        ...


class Embedding(Layer):
    """Token embedding layer.

    Embedding: output[i] = W[token_id[i]]   (table lookup, no matmul)

    The gradient w.r.t. the weight matrix is sparse: only the rows
    corresponding to the input token IDs receive non-zero gradients.
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        """Initialize embedding layer with Kaiming normal init."""
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Kaiming (He) initialization: std = sqrt(2 / fan_in)
        std = math.sqrt(2.0 / hidden_dim)
        self.weight = Tensor.randn_std((vocab_size, hidden_dim), DType.F32, std)

        # Cache for backward
        self._last_indices: np.ndarray | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Lookup embeddings for token IDs.

        Args:
            x: Token IDs [batch, seq_len]

        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        indices = x.data.astype(np.int64)
        self._last_indices = indices
        # Fancy indexing: weight[indices] gathers rows from the embedding table
        return Tensor.from_numpy(self.weight.data[indices])

    def forward_ids(self, token_ids: list[int]) -> Tensor:
        """Lookup embeddings for token ID list.

        Args:
            token_ids: List of token IDs

        Returns:
            Embeddings [len(token_ids), hidden_dim]
        """
        indices = np.array(token_ids, dtype=np.int64)
        self._last_indices = indices
        return Tensor.from_numpy(self.weight.data[indices])

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass — accumulates gradient to embedding weight.

        Embedding gradient: grad_weight[token_id] += grad_output for each
        occurrence of token_id in the input indices. np.add.at handles
        duplicate indices correctly (unlike fancy indexing assignment).

        Returns zeros because there is no meaningful gradient w.r.t.
        discrete token IDs (the embedding input).
        """
        if self._last_indices is not None:
            grad_w = np.zeros_like(self.weight.data)
            # Scatter-add: accumulate gradients for each token ID
            np.add.at(grad_w, self._last_indices, grad_output.data)
            if self.weight._grad is None:
                self.weight._grad = grad_w
            else:
                self.weight._grad += grad_w
        return Tensor.zeros(grad_output.shape)

    def parameters(self) -> list[Tensor]:
        """Return embedding weight."""
        return [self.weight]


class RMSNorm(Layer):
    """Root Mean Square Layer Normalization.

    RMSNorm: y = x * (1 / sqrt(mean(x^2) + eps)) * gamma

    Unlike LayerNorm, RMSNorm omits the mean-subtraction step and has
    no bias term, making it cheaper to compute while empirically
    performing comparably.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm."""
        self.dim = dim
        self.eps = eps
        self.weight = Tensor.ones((dim,))  # gamma, initialized to 1

        # Cache for backward
        self._last_input: Tensor | None = None
        self._last_rms: np.ndarray | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        RMSNorm: y = (x / rms) * gamma
        where rms = sqrt(mean(x^2) + eps)

        Uses fused multiply to avoid intermediate array for normalized * gamma.

        Args:
            x: Input [..., dim]

        Returns:
            Normalized output [..., dim]
        """
        # Store reference (no clone needed — backward is simplified)
        self._last_input = x

        xd = x.data
        # rms = sqrt(mean(x^2, axis=-1) + eps)
        rms = np.sqrt(np.mean(xd * xd, axis=-1, keepdims=True) + self.eps)
        self._last_rms = rms

        # y = (x / rms) * gamma — fused into one output array
        # Using x * (gamma / rms) avoids the intermediate 'normalized' array
        out = xd * (self.weight.data / rms)
        return Tensor.from_numpy(out)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass for RMSNorm.

        y = (x / rms) * gamma
        grad_gamma = sum(grad_output * x / rms, over batch dims)
        grad_input = grad_output * gamma / rms
                     - x * dot(grad_output * gamma, x) / (dim * rms^3)

        The second term is the correction from the full normalization Jacobian.
        Without it, gradients are incorrect and training fails to converge.
        """
        if self._last_input is None:
            raise RuntimeError("backward called before forward")

        x_data = self._last_input.data
        rms = self._last_rms
        x_norm = x_data / rms  # x / rms
        go = grad_output.data

        # grad_gamma: sum over all dims except last (the feature dim)
        grad_gamma = np.sum(go * x_norm, axis=tuple(range(go.ndim - 1)))
        if self.weight._grad is None:
            self.weight._grad = grad_gamma.astype(np.float32)
        else:
            self.weight._grad += grad_gamma.astype(np.float32)

        # Full grad_input with correction term (matching Go):
        #   g = grad_output * gamma
        #   dot_term = sum(g * x, axis=-1, keepdims=True)
        #   grad_input = g / rms - x * dot_term / (dim * rms^3)
        g = go * self.weight.data  # grad_output * gamma
        dot_term = np.sum(g * x_data, axis=-1, keepdims=True)
        rms3 = rms * rms * rms
        dim = np.float32(x_data.shape[-1])
        grad_input = g / rms - x_data * dot_term / (dim * rms3)
        return Tensor.from_numpy(grad_input)

    def parameters(self) -> list[Tensor]:
        """Return weight parameter."""
        return [self.weight]


class Linear(Layer):
    """Fully connected layer.

    Linear: y = x @ W^T + b

    Weight shape is (out_features, in_features) so that the forward pass
    computes x @ W^T, which is the standard convention matching PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """Initialize linear layer with Kaiming normal init."""
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Kaiming (He) initialization: std = sqrt(2 / fan_in)
        std = math.sqrt(2.0 / in_features)
        self.weight = Tensor.randn_std((out_features, in_features), DType.F32, std)

        if bias:
            self.bias = Tensor.zeros((out_features,))
        else:
            self.bias = None

        # Cache for backward
        self._last_input: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Linear transformation: y = x @ W^T + b.

        Handles arbitrary batch dimensions by flattening to 2D for
        the matmul, then reshaping back.

        Args:
            x: Input [..., in_features]

        Returns:
            Output [..., out_features]
        """
        # Store reference (no clone needed — backward is simplified)
        self._last_input = x

        # Flatten leading dims to 2D for matmul
        orig_shape = x.shape
        batch_dims = orig_shape[:-1]
        batch_size = int(np.prod(batch_dims))

        flat_x = x.data.reshape(batch_size, self.in_features)

        # y = x @ W^T  (dispatches to BLAS sgemm via np.matmul)
        # .T returns a view (zero-copy) — numpy/BLAS handles the strided access
        result = np.matmul(flat_x, self.weight.data.T)

        if self.bias is not None:
            result = result + self.bias.data

        return Tensor.from_numpy(result.reshape((*batch_dims, self.out_features)))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass for linear layer.

        y = x @ W^T + b
        grad_input  = grad_output @ W
        grad_weight = grad_output^T @ input    (accumulated, not overwritten)
        grad_bias   = sum(grad_output, dim=0)  (if bias exists)
        """
        if self._last_input is None:
            raise RuntimeError("backward called before forward")

        input_shape = self._last_input.shape
        batch_dims = grad_output.shape[:-1]
        batch_size = int(np.prod(batch_dims))

        flat_grad = grad_output.data.reshape(batch_size, self.out_features)
        flat_input = self._last_input.data.reshape(batch_size, self.in_features)

        # grad_input = grad_output @ W
        grad_input = np.matmul(flat_grad, self.weight.data)

        # grad_weight = grad_output^T @ input  shape: [out_features, in_features]
        grad_w = np.matmul(flat_grad.T, flat_input)
        if self.weight._grad is None:
            self.weight._grad = grad_w.astype(np.float32)
        else:
            self.weight._grad += grad_w.astype(np.float32)

        # grad_bias = sum over batch dims
        if self.bias is not None:
            grad_b = np.sum(flat_grad, axis=0)
            if self.bias._grad is None:
                self.bias._grad = grad_b.astype(np.float32)
            else:
                self.bias._grad += grad_b.astype(np.float32)

        return Tensor.from_numpy(grad_input.reshape(input_shape))

    def parameters(self) -> list[Tensor]:
        """Return weight and optionally bias."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


class SwiGLU(Layer):
    """SwiGLU activation with gated linear unit.

    SwiGLU: output = down(silu(gate(x)) * up(x))

    Expanded:
      gate_out = x @ W_gate^T
      up_out   = x @ W_up^T
      hidden   = (gate_out * sigmoid(gate_out)) * up_out    -- SiLU(gate) * up
      output   = hidden @ W_down^T

    This is the standard FFN replacement used in LLaMA/Mistral-style models.
    The "3" in parameter count comes from 3 linear projections: gate, up, down.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        """Initialize SwiGLU layer."""
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        self.gate = Linear(hidden_dim, ffn_dim, bias=False)
        self.up = Linear(hidden_dim, ffn_dim, bias=False)
        self.down = Linear(ffn_dim, hidden_dim, bias=False)

        # Cache for backward
        self._last_gate_out: Tensor | None = None
        self._last_up_out: Tensor | None = None
        self._last_gate_pre_silu: np.ndarray | None = None
        self._last_input: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU forward: down(silu(gate(x)) * up(x)).

        Caches pre-silu gate output for backward pass, then uses in-place
        ops to avoid temporary allocations.

        Args:
            x: Input [..., hidden_dim]

        Returns:
            Output [..., hidden_dim]
        """
        self._last_input = x
        gate_out = self.gate.forward(x)
        up_out = self.up.forward(x)

        self._last_up_out = up_out
        # Cache pre-silu gate output for backward (before in-place mutation)
        self._last_gate_pre_silu = gate_out.data.copy()

        # In-place SiLU on gate, then in-place multiply with up projection
        # Saves 2 temporary Tensor allocations vs gate_out.silu() * up_out
        gate_out.silu_inplace()
        self._last_gate_out = Tensor.from_numpy(gate_out.data.copy())  # silu(gate)
        gate_out.mul_inplace(up_out)

        # Down projection back to hidden_dim
        return self.down.forward(gate_out)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass for SwiGLU.

        Forward: hidden = silu(gate(x)) * up(x),  output = down(hidden)
        Backward:
          grad_hidden = down.backward(grad_output)
          grad_silu_gate = grad_hidden * up(x)
          grad_up_out    = grad_hidden * silu(gate(x))
          grad_gate_out  = grad_silu_gate * silu'(gate(x))
            where silu'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
                           = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
          grad_input = gate.backward(grad_gate_out) + up.backward(grad_up_out)
        """
        # Backprop through down projection (also stores its weight grad)
        grad_hidden = self.down.backward(grad_output)

        # Split grad to gate and up paths
        up_data = self._last_up_out.data
        silu_gate_data = self._last_gate_out.data  # silu(gate(x))
        gate_pre_silu = self._last_gate_pre_silu   # gate(x) before silu

        gh = grad_hidden.data

        # grad w.r.t. up output:  d(silu(g) * u)/du = silu(g)
        grad_up_out = Tensor.from_numpy(gh * silu_gate_data)

        # grad w.r.t. silu(gate) output:  d(silu(g) * u)/d(silu(g)) = u
        grad_silu_out = gh * up_data

        # silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        sigmoid_g = 1.0 / (1.0 + np.exp(-gate_pre_silu))
        dsilu = sigmoid_g * (1.0 + gate_pre_silu * (1.0 - sigmoid_g))
        grad_gate_out = Tensor.from_numpy(grad_silu_out * dsilu)

        # Backprop through gate and up Linear layers
        # Need to temporarily set _last_input for gate and up since forward
        # may have overwritten it (both used x as input)
        self.gate._last_input = self._last_input
        self.up._last_input = self._last_input

        grad_x_gate = self.gate.backward(grad_gate_out)
        grad_x_up = self.up.backward(grad_up_out)

        # Sum gradients from both paths
        return Tensor.from_numpy(grad_x_gate.data + grad_x_up.data)

    def parameters(self) -> list[Tensor]:
        """Return all parameters (gate + up + down projections)."""
        return self.gate.parameters() + self.up.parameters() + self.down.parameters()
