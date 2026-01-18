"""Tests for CUDA Python bindings.

These tests verify:
1. CPU fallback implementations work correctly
2. Module imports correctly
3. All exported functions are available
"""

import numpy as np
import pytest

from . import (
    CudaError,
    cuda_available,
    silu,
    add,
    mul,
    scale,
    softmax,
    rmsnorm,
    gemm,
    cross_entropy_forward,
    adamw_step,
    argmax,
    sample,
    topk_sample,
    topp_sample,
)


class TestModuleImports:
    """Test that all module components are importable."""

    def test_cuda_error_exists(self):
        assert CudaError is not None
        assert issubclass(CudaError, Exception)

    def test_cuda_available_function(self):
        # Should return bool
        result = cuda_available()
        assert isinstance(result, bool)

    def test_all_functions_importable(self):
        # All functions should be callable
        assert callable(silu)
        assert callable(add)
        assert callable(mul)
        assert callable(scale)
        assert callable(softmax)
        assert callable(rmsnorm)
        assert callable(gemm)
        assert callable(cross_entropy_forward)
        assert callable(adamw_step)
        assert callable(argmax)
        assert callable(sample)
        assert callable(topk_sample)
        assert callable(topp_sample)


class TestCPUFallback:
    """Test CPU fallback implementations."""

    def test_silu(self):
        input_arr = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        output = np.zeros_like(input_arr)

        silu(input_arr, output)

        # SiLU(x) = x * sigmoid(x)
        expected = input_arr * (1.0 / (1.0 + np.exp(-input_arr)))
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_add(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        output = np.zeros_like(a)

        add(a, b, output)

        expected = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        np.testing.assert_allclose(output, expected)

    def test_mul(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        output = np.zeros_like(a)

        mul(a, b, output)

        expected = np.array([5.0, 12.0, 21.0, 32.0], dtype=np.float32)
        np.testing.assert_allclose(output, expected)

    def test_scale(self):
        input_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        output = np.zeros_like(input_arr)

        scale(input_arr, output, 2.0)

        expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        np.testing.assert_allclose(output, expected)

    def test_softmax(self):
        batch, dim = 2, 4
        input_arr = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        output = np.zeros_like(input_arr)

        softmax(input_arr, output, batch, dim)

        # Verify softmax properties
        output_2d = output.reshape(batch, dim)
        # Sum should be 1 for each row
        np.testing.assert_allclose(output_2d.sum(axis=-1), np.ones(batch), rtol=1e-5)
        # All values should be positive
        assert np.all(output >= 0)

    def test_rmsnorm(self):
        batch, dim = 2, 4
        input_arr = np.ones(batch * dim, dtype=np.float32)
        weight = np.ones(dim, dtype=np.float32)
        output = np.zeros_like(input_arr)

        rmsnorm(input_arr, weight, output, batch, dim, eps=1e-6)

        # With all ones input and weight, RMS = 1, normalized = 1
        expected = np.ones(batch * dim, dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_gemm(self):
        m, n, k = 2, 3, 4
        # A: (m, k) = (2, 4)
        a = np.arange(m * k, dtype=np.float32)
        # B: (k, n) = (4, 3)
        b = np.ones(k * n, dtype=np.float32)
        # C: (m, n) = (2, 3)
        c = np.zeros(m * n, dtype=np.float32)

        gemm(a, b, c, m, n, k, alpha=1.0, beta=0.0)

        # Verify using numpy
        a_mat = a.reshape(m, k)
        b_mat = b.reshape(k, n)
        expected = (a_mat @ b_mat).ravel()
        np.testing.assert_allclose(c, expected, rtol=1e-5)

    def test_gemm_with_beta(self):
        m, n, k = 2, 2, 2
        a = np.ones(m * k, dtype=np.float32)
        b = np.ones(k * n, dtype=np.float32)
        c = np.ones(m * n, dtype=np.float32) * 10.0

        gemm(a, b, c, m, n, k, alpha=1.0, beta=0.5)

        # C = 1.0 * (1 @ 1) + 0.5 * 10 = 2 + 5 = 7
        expected = np.full(m * n, 7.0, dtype=np.float32)
        np.testing.assert_allclose(c, expected, rtol=1e-5)

    def test_cross_entropy_forward(self):
        batch, vocab_size = 2, 5
        logits = np.random.randn(batch * vocab_size).astype(np.float32)
        targets = np.array([0, 3], dtype=np.int32)
        loss = np.zeros(1, dtype=np.float32)
        log_probs = np.zeros(batch * vocab_size, dtype=np.float32)

        cross_entropy_forward(logits, targets, loss, log_probs, batch, vocab_size)

        # Loss should be positive
        assert loss[0] > 0
        # Log probs should be <= 0
        assert np.all(log_probs <= 0)

    def test_adamw_step(self):
        size = 4
        param = np.ones(size, dtype=np.float32)
        grad = np.ones(size, dtype=np.float32) * 0.1
        m = np.zeros(size, dtype=np.float32)
        v = np.zeros(size, dtype=np.float32)

        adamw_step(
            param, grad, m, v,
            lr=0.001, beta1=0.9, beta2=0.999,
            eps=1e-8, weight_decay=0.01, step=1
        )

        # Params should have changed
        assert not np.allclose(param, np.ones(size))
        # m and v should be updated
        assert not np.allclose(m, np.zeros(size))
        assert not np.allclose(v, np.zeros(size))

    def test_argmax(self):
        batch, vocab_size = 2, 5
        logits = np.array([
            [0.1, 0.2, 0.9, 0.3, 0.4],  # max at index 2
            [0.5, 0.1, 0.2, 0.8, 0.1],  # max at index 3
        ], dtype=np.float32).ravel()
        output = np.zeros(batch, dtype=np.int32)

        argmax(logits, output, batch, vocab_size)

        expected = np.array([2, 3], dtype=np.int32)
        np.testing.assert_array_equal(output, expected)

    def test_sample(self):
        batch, vocab_size = 2, 5
        # Make one token have very high probability
        logits = np.full((batch, vocab_size), -100.0, dtype=np.float32)
        logits[:, 2] = 100.0  # Token 2 should be selected
        logits = logits.ravel()
        output = np.zeros(batch, dtype=np.int32)
        seeds = np.array([42, 123], dtype=np.uint64)

        sample(logits, output, seeds, batch, vocab_size, temperature=1.0)

        # With such extreme logits, token 2 should always be selected
        expected = np.array([2, 2], dtype=np.int32)
        np.testing.assert_array_equal(output, expected)

    def test_topk_sample(self):
        np.random.seed(42)
        batch, vocab_size, k = 2, 10, 3
        logits = np.random.randn(batch * vocab_size).astype(np.float32)
        output = np.zeros(batch, dtype=np.int32)
        seeds = np.array([42, 123], dtype=np.uint64)

        topk_sample(logits, output, seeds, batch, vocab_size, k, temperature=1.0)

        # Output should be valid indices
        assert np.all(output >= 0)
        assert np.all(output < vocab_size)

    def test_topp_sample(self):
        np.random.seed(42)
        batch, vocab_size = 2, 10
        logits = np.random.randn(batch * vocab_size).astype(np.float32)
        output = np.zeros(batch, dtype=np.int32)
        seeds = np.array([42, 123], dtype=np.uint64)

        topp_sample(logits, output, seeds, batch, vocab_size, top_p=0.9, temperature=1.0)

        # Output should be valid indices
        assert np.all(output >= 0)
        assert np.all(output < vocab_size)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_silu_zero_input(self):
        input_arr = np.zeros(4, dtype=np.float32)
        output = np.zeros_like(input_arr)

        silu(input_arr, output)

        # SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        np.testing.assert_allclose(output, np.zeros(4))

    def test_softmax_numerical_stability(self):
        # Large values that could cause overflow
        batch, dim = 1, 4
        input_arr = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=np.float32)
        output = np.zeros_like(input_arr)

        softmax(input_arr, output, batch, dim)

        # Should not have NaN or Inf
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        # Sum should still be 1
        np.testing.assert_allclose(output.sum(), 1.0, rtol=1e-5)

    def test_rmsnorm_small_input(self):
        batch, dim = 1, 4
        input_arr = np.full(dim, 1e-10, dtype=np.float32)
        weight = np.ones(dim, dtype=np.float32)
        output = np.zeros_like(input_arr)

        rmsnorm(input_arr, weight, output, batch, dim, eps=1e-6)

        # Should not have NaN or Inf
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
