#!/usr/bin/env python3
"""Cross-language benchmark: Python implementation using NumPy"""

import numpy as np
import timeit
import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchResult:
    name: str
    size: str
    mean_us: float
    std_us: float
    iterations: int


def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Naive matmul (for comparison with numpy optimized)"""
    m, k = a.shape
    _, n = b.shape
    c = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            for p in range(k):
                c[i, j] += a[i, p] * b[p, j]
    return c


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy optimized matmul (BLAS backend)"""
    return a @ b


def softmax_naive(x: np.ndarray) -> np.ndarray:
    """Naive softmax implementation"""
    rows, cols = x.shape
    output = np.zeros_like(x)
    for r in range(rows):
        max_val = np.max(x[r])
        exp_vals = np.exp(x[r] - max_val)
        output[r] = exp_vals / np.sum(exp_vals)
    return output


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy vectorized softmax"""
    max_vals = np.max(x, axis=1, keepdims=True)
    exp_vals = np.exp(x - max_vals)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def silu_naive(x: np.ndarray) -> np.ndarray:
    """Naive SiLU: x * sigmoid(x)"""
    output = np.zeros_like(x)
    for i in range(len(x)):
        output[i] = x[i] * (1.0 / (1.0 + np.exp(-x[i])))
    return output


def silu_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy vectorized SiLU"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def rmsnorm_naive(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Naive RMSNorm"""
    n, dim = x.shape
    output = np.zeros_like(x)
    for i in range(n):
        sum_sq = np.sum(x[i] ** 2)
        rms = np.sqrt(sum_sq / dim + eps)
        output[i] = (x[i] / rms) * weight
    return output


def rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NumPy vectorized RMSNorm"""
    rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + eps)
    return (x / rms) * weight


def benchmark(func: Callable, setup: Callable, iterations: int = 100) -> tuple[float, float]:
    """Run benchmark and return (mean_us, std_us)"""
    args = setup()

    # Warmup
    for _ in range(min(10, iterations)):
        func(*args)

    # Measure
    times = []
    for _ in range(iterations):
        start = timeit.default_timer()
        func(*args)
        end = timeit.default_timer()
        times.append((end - start) * 1e6)  # Convert to microseconds

    return np.mean(times), np.std(times)


def run_benchmarks() -> list[BenchResult]:
    results = []
    np.random.seed(42)

    # Matmul benchmarks
    print("Running matmul benchmarks...")
    for size in [64, 128, 256]:  # Skip 512 for naive (too slow)
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        # Naive (only small sizes)
        if size <= 128:
            mean, std = benchmark(matmul_naive, lambda: (a.copy(), b.copy()), iterations=10)
            results.append(BenchResult("matmul_naive", str(size), mean, std, 10))

        # NumPy
        mean, std = benchmark(matmul_numpy, lambda: (a.copy(), b.copy()), iterations=100)
        results.append(BenchResult("matmul_numpy", str(size), mean, std, 100))

    # Large matmul (numpy only)
    size = 512
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    mean, std = benchmark(matmul_numpy, lambda: (a.copy(), b.copy()), iterations=100)
    results.append(BenchResult("matmul_numpy", str(size), mean, std, 100))

    # Softmax benchmarks
    print("Running softmax benchmarks...")
    for rows, cols in [(64, 1024), (128, 1024), (256, 1024), (512, 32000)]:
        x = np.random.randn(rows, cols).astype(np.float32)

        mean, std = benchmark(softmax_numpy, lambda: (x.copy(),), iterations=100)
        results.append(BenchResult("softmax_numpy", f"{rows}x{cols}", mean, std, 100))

    # SiLU benchmarks
    print("Running silu benchmarks...")
    for size in [1024, 4096, 16384, 65536]:
        x = np.random.randn(size).astype(np.float32)

        mean, std = benchmark(silu_numpy, lambda: (x.copy(),), iterations=100)
        results.append(BenchResult("silu_numpy", str(size), mean, std, 100))

    # RMSNorm benchmarks
    print("Running rmsnorm benchmarks...")
    for batch_seq, dim in [(64, 768), (128, 768), (256, 768), (512, 768)]:
        x = np.random.randn(batch_seq, dim).astype(np.float32)
        weight = np.random.randn(dim).astype(np.float32)

        mean, std = benchmark(rmsnorm_numpy, lambda: (x.copy(), weight.copy()), iterations=100)
        results.append(BenchResult("rmsnorm_numpy", f"{batch_seq}x{dim}", mean, std, 100))

    return results


def print_results(results: list[BenchResult]):
    print("\n" + "=" * 60)
    print("Python Benchmark Results")
    print("=" * 60)
    print(f"{'Name':<20} {'Size':<15} {'Mean (µs)':<15} {'Std (µs)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r.name:<20} {r.size:<15} {r.mean_us:<15.2f} {r.std_us:<15.2f}")
    print("=" * 60)


def export_json(results: list[BenchResult], filepath: str):
    data = [
        {
            "name": r.name,
            "size": r.size,
            "mean_us": r.mean_us,
            "std_us": r.std_us,
            "iterations": r.iterations,
        }
        for r in results
    ]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    results = run_benchmarks()
    print_results(results)
    export_json(results, "results_python.json")
    print(f"\nResults exported to results_python.json")
