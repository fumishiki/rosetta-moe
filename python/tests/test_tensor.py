"""Tests for tensor module."""

import numpy as np
import pytest

from nn.tensor import DType, Tensor


class TestDType:
    """Tests for DType enum."""

    def test_dtype_values(self):
        assert DType.F32.value == "float32"
        assert DType.F16.value == "float16"
        assert DType.I32.value == "int32"

    def test_dtype_to_numpy(self):
        assert DType.F32.to_numpy() == np.float32
        assert DType.I32.to_numpy() == np.int32


class TestTensor:
    """Tests for Tensor class."""

    def test_zeros(self):
        t = Tensor.zeros((2, 3))
        assert t.shape == (2, 3)
        assert t.numel == 6
        assert np.allclose(t.data, 0)

    def test_ones(self):
        t = Tensor.ones((2, 3))
        assert t.shape == (2, 3)
        assert np.allclose(t.data, 1)

    def test_randn(self):
        t = Tensor.randn((100, 100))
        assert t.shape == (100, 100)
        # Random normal should have mean ~0 and std ~1
        assert abs(np.mean(t.data)) < 0.1
        assert abs(np.std(t.data) - 1.0) < 0.1

    def test_randn_std(self):
        t = Tensor.randn_std((100, 100), std=0.5)
        assert abs(np.std(t.data) - 0.5) < 0.1

    def test_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = Tensor.from_numpy(arr)
        assert t.shape == (2, 2)
        assert np.allclose(t.data, arr)

    def test_clone(self):
        t1 = Tensor.ones((2, 3))
        t2 = t1.clone()
        t1._data[0, 0] = 99
        assert t2.data[0, 0] == 1  # Clone is independent

    def test_reshape(self):
        t = Tensor.randn((2, 3, 4))
        reshaped = t.reshape((6, 4))
        assert reshaped.shape == (6, 4)
        assert reshaped.numel == t.numel

    def test_transpose(self):
        t = Tensor.randn((2, 3))
        transposed = t.transpose()
        assert transposed.shape == (3, 2)

    def test_add(self):
        a = Tensor.ones((2, 3))
        b = Tensor.ones((2, 3))
        c = a + b
        assert np.allclose(c.data, 2)

    def test_sub(self):
        a = Tensor.ones((2, 3)) * 3
        b = Tensor.ones((2, 3))
        c = a - b
        assert np.allclose(c.data, 2)

    def test_mul(self):
        a = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        b = Tensor.from_numpy(np.array([2, 3, 4], dtype=np.float32))
        c = a * b
        assert np.allclose(c.data, [2, 6, 12])

    def test_scale(self):
        t = Tensor.ones((2, 3))
        scaled = t.scale(5.0)
        assert np.allclose(scaled.data, 5)

    def test_silu(self):
        t = Tensor.from_numpy(np.array([0, 1, -1], dtype=np.float32))
        result = t.silu()
        # SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269
        assert abs(result.data[0]) < 1e-6
        assert abs(result.data[1] - 0.731) < 0.01
        assert abs(result.data[2] + 0.269) < 0.01

    def test_softmax(self):
        t = Tensor.from_numpy(np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32))
        result = t.softmax()
        # Softmax sums to 1 along last axis
        row_sums = np.sum(result.data, axis=-1)
        assert np.allclose(row_sums, 1)

    def test_matmul(self):
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = Tensor.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = a @ b
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        assert np.allclose(c.data, expected)

    def test_sum(self):
        t = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert t.sum().data == 10
        assert np.allclose(t.sum(axis=0).data, [4, 6])
        assert np.allclose(t.sum(axis=1).data, [3, 7])

    def test_mean(self):
        t = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert t.mean().data == 2.5

    def test_argmax(self):
        t = Tensor.from_numpy(np.array([[1, 3, 2], [5, 1, 4]], dtype=np.float32))
        result = t.argmax(axis=-1)
        assert list(result.data) == [1, 0]
