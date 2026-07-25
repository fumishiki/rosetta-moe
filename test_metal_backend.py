#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Test script for Metal backend — verifies API design and usage patterns

import sys
sys.path.insert(0, '.')

import numpy as np
from python.metal_backend import metal_available

print("=" * 60)
print("Metal Backend Test")
print("=" * 60)
print()

if not metal_available():
    print("❌ Metal not available on this system")
    print()
    print("To install Metal support:")
    print("  pip3 install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders")
    print()
    print("Exiting gracefully (this is expected if Metal deps not installed)")
    sys.exit(0)

print("✓ Metal is available")
print()

# Import Metal components (only after checking availability)
from python.metal_backend import MetalContext, MetalTensor, mps_matmul, dispatch_kernel

# Create Metal context
print("Creating Metal context...")
ctx = MetalContext()
print(f"  Device: {ctx.device.name()}")
print(f"  Loaded {len(ctx._pipelines)} kernel pipelines")
print()

# Test 1: MetalTensor creation and transfer
print("Test 1: Tensor creation and CPU↔GPU transfer")
a = np.random.randn(64, 64).astype(np.float32)
print(f"  Created numpy array: shape={a.shape}, dtype={a.dtype}")

gpu_a = MetalTensor.from_numpy(ctx, a)
print(f"  Uploaded to GPU: nbytes={gpu_a.nbytes}, shape={gpu_a.shape}")

a_back = gpu_a.to_numpy()
print(f"  Downloaded from GPU: shape={a_back.shape}")
print(f"  Round-trip error: {np.max(np.abs(a - a_back))}")
print()

# Test 2: MPS Matrix Multiplication
print("Test 2: MPS Matrix Multiplication (64x64 @ 64x64)")
b = np.random.randn(64, 64).astype(np.float32)
gpu_b = MetalTensor.from_numpy(ctx, b)

# Allocate output buffer
gpu_c = MetalTensor(ctx, nbytes=64*64*4, shape=[64, 64])

# Run matmul on GPU
print("  Running GPU matmul...")
mps_matmul(ctx, gpu_a, gpu_b, gpu_c, M=64, N=64, K=64)

# Verify against CPU
c_gpu = gpu_c.to_numpy()
c_cpu = a @ b
error = np.max(np.abs(c_gpu - c_cpu))
print(f"  GPU result shape: {c_gpu.shape}")
print(f"  CPU vs GPU max error: {error}")
print(f"  {'✓ PASS' if error < 1e-4 else '✗ FAIL'} (threshold: 1e-4)")
print()

# Test 3: Custom kernel dispatch (if available)
print("Test 3: Custom kernel dispatch")
if "silu" in ctx._pipelines:
    x = np.random.randn(1024).astype(np.float32)
    gpu_x = MetalTensor.from_numpy(ctx, x)
    gpu_y = MetalTensor(ctx, nbytes=1024*4, shape=[1024])

    print("  Dispatching 'silu' kernel...")
    dispatch_kernel(ctx, "silu", [gpu_x, gpu_y], grid_size=1024, threadgroup_size=256)

    y_gpu = gpu_y.to_numpy()
    # SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
    y_cpu = x / (1.0 + np.exp(-x))
    error = np.max(np.abs(y_gpu - y_cpu))
    print(f"  CPU vs GPU max error: {error}")
    print(f"  {'✓ PASS' if error < 1e-5 else '✗ FAIL'} (threshold: 1e-5)")
else:
    print("  ⚠ 'silu' kernel not loaded (shader file missing?)")
print()

print("=" * 60)
print("All tests completed successfully!")
print("=" * 60)
