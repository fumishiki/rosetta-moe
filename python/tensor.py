# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Tensor operations for MoE Transformer.

Thin wrapper around numpy ndarrays providing a typed Tensor API.
On macOS >= 14, numpy's np.matmul dispatches to Apple Accelerate (BLAS/AMX),
so all matmul calls in this project benefit from hardware-accelerated linear
algebra without any explicit FFI.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np


class DType(Enum):
    """Supported data types for tensors.

    Maps our enum variants to numpy dtypes.  BF16 is emulated as float32
    because numpy has no native bfloat16 support.
    """

    F32 = "float32"
    F16 = "float16"
    BF16 = "bfloat16"
    I32 = "int32"
    I64 = "int64"

    def to_numpy(self) -> np.dtype:
        """Convert to numpy dtype."""
        if self == DType.BF16:
            # numpy doesn't have bfloat16, use float32 as a stand-in
            return np.float32
        return np.dtype(self.value)


class Tensor:
    """Tensor class wrapping numpy arrays.

    Design notes:
    - __init__ avoids a copy when the source array already has the target
      dtype (``data.dtype == target``).  This matters in hot paths where
      Tensor.from_numpy is called with pre-cast arrays.
    - All arithmetic methods return new Tensor instances (immutable style).
    """

    def __init__(self, data: np.ndarray, dtype: DType = DType.F32):
        """Create tensor from numpy array.

        Skips the dtype cast (and therefore the copy) when ``data.dtype``
        already matches the target numpy dtype.  This is intentional: most
        internal code constructs arrays in the correct dtype up-front, so
        the branch avoids a redundant allocation on every layer output.
        """
        target = dtype.to_numpy()
        # Skip copy when dtype already matches — avoids allocation overhead
        self._data = data if data.dtype == target else data.astype(target)
        self._dtype = dtype
        # Gradient storage for backward pass — set by layer backward methods
        self._grad: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape."""
        return self._data.shape

    @property
    def dtype(self) -> DType:
        """Return tensor dtype."""
        return self._dtype

    @property
    def numel(self) -> int:
        """Return number of elements."""
        return self._data.size

    @property
    def data(self) -> np.ndarray:
        """Return underlying numpy array."""
        return self._data

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: DType = DType.F32) -> Tensor:
        """Create tensor of zeros."""
        return cls(np.zeros(shape, dtype=dtype.to_numpy()), dtype)

    @classmethod
    def ones(cls, shape: Sequence[int], dtype: DType = DType.F32) -> Tensor:
        """Create tensor of ones."""
        return cls(np.ones(shape, dtype=dtype.to_numpy()), dtype)

    @classmethod
    def randn(cls, shape: Sequence[int], dtype: DType = DType.F32) -> Tensor:
        """Create tensor with random normal values (mean=0, std=1)."""
        return cls(np.random.randn(*shape).astype(dtype.to_numpy()), dtype)

    @classmethod
    def randn_std(
        cls, shape: Sequence[int], dtype: DType = DType.F32, std: float = 1.0
    ) -> Tensor:
        """Create tensor with random normal values and specific std."""
        return cls((np.random.randn(*shape) * std).astype(dtype.to_numpy()), dtype)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, dtype: DType = DType.F32) -> Tensor:
        """Create tensor from numpy array."""
        return cls(arr, dtype)

    def clone(self) -> Tensor:
        """Create a deep copy of the tensor."""
        return Tensor(self._data.copy(), self._dtype)

    def reshape(self, shape: Sequence[int]) -> Tensor:
        """Reshape tensor (returns a view when possible, copy otherwise)."""
        return Tensor(self._data.reshape(shape), self._dtype)

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        """Transpose tensor."""
        if axes is None:
            # Default: reverse all axes
            axes = tuple(range(len(self.shape) - 1, -1, -1))
        return Tensor(np.transpose(self._data, axes), self._dtype)

    def add(self, other: Tensor) -> Tensor:
        """Element-wise addition."""
        return Tensor(self._data + other._data, self._dtype)

    def sub(self, other: Tensor) -> Tensor:
        """Element-wise subtraction."""
        return Tensor(self._data - other._data, self._dtype)

    def mul(self, other: Tensor) -> Tensor:
        """Element-wise multiplication."""
        return Tensor(self._data * other._data, self._dtype)

    def div(self, other: Tensor) -> Tensor:
        """Element-wise division."""
        return Tensor(self._data / other._data, self._dtype)

    def scale(self, scalar: float) -> Tensor:
        """Scale tensor by scalar."""
        return Tensor(self._data * scalar, self._dtype)

    def neg(self) -> Tensor:
        """Negate tensor."""
        return Tensor(-self._data, self._dtype)

    def exp(self) -> Tensor:
        """Element-wise exponential."""
        return Tensor(np.exp(self._data), self._dtype)

    def log(self) -> Tensor:
        """Element-wise natural logarithm."""
        return Tensor(np.log(self._data), self._dtype)

    def sqrt(self) -> Tensor:
        """Element-wise square root."""
        return Tensor(np.sqrt(self._data), self._dtype)

    def pow(self, exponent: float) -> Tensor:
        """Element-wise power."""
        return Tensor(np.power(self._data, exponent), self._dtype)

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Sum along axis."""
        return Tensor(np.sum(self._data, axis=axis, keepdims=keepdims), self._dtype)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Mean along axis."""
        return Tensor(np.mean(self._data, axis=axis, keepdims=keepdims), self._dtype)

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Max along axis."""
        return Tensor(np.max(self._data, axis=axis, keepdims=keepdims), self._dtype)

    def argmax(self, axis: int = -1) -> Tensor:
        """Argmax along axis."""
        return Tensor(np.argmax(self._data, axis=axis), DType.I64)

    def silu(self) -> Tensor:
        """SiLU (Swish) activation.

        SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
        """
        sigmoid = 1.0 / (1.0 + np.exp(-self._data))
        return Tensor(self._data * sigmoid, self._dtype)

    def silu_inplace(self) -> Tensor:
        """In-place SiLU (Swish) activation. Returns self for chaining."""
        sigmoid = 1.0 / (1.0 + np.exp(-self._data))
        np.multiply(self._data, sigmoid, out=self._data)
        return self

    def mul_inplace(self, other: Tensor) -> Tensor:
        """In-place element-wise multiplication. Returns self for chaining."""
        np.multiply(self._data, other._data, out=self._data)
        return self

    def add_inplace(self, other: Tensor) -> Tensor:
        """In-place element-wise addition. Returns self for chaining."""
        np.add(self._data, other._data, out=self._data)
        return self

    def softmax(self, axis: int = -1) -> Tensor:
        """Softmax along axis.

        Softmax: p_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

        Subtracting max(x) before exp() prevents overflow (numerical
        stability trick).  The result is mathematically identical because
        exp(x_i - c) / sum(exp(x_j - c)) = exp(x_i) / sum(exp(x_j)).
        """
        shifted = self._data - np.max(self._data, axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        return Tensor(exp_vals / np.sum(exp_vals, axis=axis, keepdims=True), self._dtype)

    def matmul(self, other: Tensor) -> Tensor:
        """Matrix multiplication via NumPy/BLAS.

        On macOS with numpy linked against Accelerate, this dispatches to
        AMX-backed sgemm for float32 — no explicit FFI required.
        """
        return Tensor(np.matmul(self._data, other._data), self._dtype)

    def __add__(self, other: Tensor) -> Tensor:
        return self.add(other)

    def __sub__(self, other: Tensor) -> Tensor:
        return self.sub(other)

    def __mul__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            return self.mul(other)
        return self.scale(other)

    def __rmul__(self, other: float) -> Tensor:
        return self.scale(other)

    def __neg__(self) -> Tensor:
        return self.neg()

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self._dtype.name})"

    def __getitem__(self, key):
        """Index into tensor."""
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return Tensor(result, self._dtype)
        return result
