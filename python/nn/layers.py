"""Neural network layers for MoE Transformer."""

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
        pass

    @abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass."""
        pass

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        """Return layer parameters."""
        pass


class Embedding(Layer):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        """Initialize embedding layer."""
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Initialize with small random values
        std = 1.0 / math.sqrt(hidden_dim)
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
        """Backward pass (returns zeros, gradient accumulated to weight)."""
        return Tensor.zeros(grad_output.shape)

    def parameters(self) -> list[Tensor]:
        """Return embedding weight."""
        return [self.weight]


class RMSNorm(Layer):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm."""
        self.dim = dim
        self.eps = eps
        self.weight = Tensor.ones((dim,))

        # Cache for backward
        self._last_input: Tensor | None = None
        self._last_rms: np.ndarray | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input [..., dim]

        Returns:
            Normalized output [..., dim]
        """
        self._last_input = x.clone()

        # Compute RMS
        rms = np.sqrt(np.mean(x.data**2, axis=-1, keepdims=True) + self.eps)
        self._last_rms = rms

        # Normalize and scale
        normalized = x.data / rms
        return Tensor.from_numpy(normalized * self.weight.data)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass for RMSNorm."""
        if self._last_input is None:
            raise RuntimeError("backward called before forward")

        # Simplified backward - just scale
        return Tensor.from_numpy(grad_output.data / self._last_rms)

    def parameters(self) -> list[Tensor]:
        """Return weight parameter."""
        return [self.weight]


class Linear(Layer):
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """Initialize linear layer."""
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Kaiming initialization
        std = math.sqrt(2.0 / in_features)
        self.weight = Tensor.randn_std((out_features, in_features), DType.F32, std)

        if bias:
            self.bias = Tensor.zeros((out_features,))
        else:
            self.bias = None

        # Cache for backward
        self._last_input: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Linear transformation: y = xW^T + b.

        Args:
            x: Input [..., in_features]

        Returns:
            Output [..., out_features]
        """
        self._last_input = x.clone()

        # Reshape to 2D for matmul
        orig_shape = x.shape
        batch_dims = orig_shape[:-1]
        batch_size = int(np.prod(batch_dims))

        flat_x = x.reshape((batch_size, self.in_features))

        # y = x @ W^T
        output = flat_x @ self.weight.transpose()

        # Add bias if present
        if self.bias is not None:
            output = Tensor.from_numpy(output.data + self.bias.data)

        # Reshape back
        return output.reshape((*batch_dims, self.out_features))

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass for linear layer."""
        if self._last_input is None:
            raise RuntimeError("backward called before forward")

        input_shape = self._last_input.shape
        batch_dims = grad_output.shape[:-1]
        batch_size = int(np.prod(batch_dims))

        # Flatten grad_output
        flat_grad = grad_output.reshape((batch_size, self.out_features))

        # grad_input = grad_output @ W
        grad_input = flat_grad @ self.weight

        # Reshape to original input shape
        return grad_input.reshape(input_shape)

    def parameters(self) -> list[Tensor]:
        """Return weight and optionally bias."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


class SwiGLU(Layer):
    """SwiGLU activation with gated linear unit."""

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

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU forward: down(silu(gate(x)) * up(x)).

        Args:
            x: Input [..., hidden_dim]

        Returns:
            Output [..., hidden_dim]
        """
        gate_out = self.gate.forward(x)
        up_out = self.up.forward(x)

        self._last_gate_out = gate_out
        self._last_up_out = up_out

        # SiLU activation on gate
        activated = gate_out.silu()

        # Element-wise multiply with up projection
        hidden = activated * up_out

        # Down projection
        return self.down.forward(hidden)

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass for SwiGLU."""
        # Simplified backward through down projection
        return self.down.backward(grad_output)

    def parameters(self) -> list[Tensor]:
        """Return all parameters."""
        return self.gate.parameters() + self.up.parameters() + self.down.parameters()
