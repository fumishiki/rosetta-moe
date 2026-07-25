# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU backend for MoE Transformer using PyObjC.

Provides GPU acceleration via Apple Metal framework for forward/backward passes
and custom compute kernels (softmax, rmsnorm, silu, RoPE).

Design notes:
  - PyObjC method names follow Objective-C selector conventions with underscores
  - MTLResourceStorageModeShared = unified memory (zero-copy on Apple Silicon)
  - Metal API requires explicit resource management (no GC for GPU buffers)
  - Shader loading from .metal files in shaders/ directory
"""

from __future__ import annotations

import ctypes
import os
import struct
from typing import Any

import numpy as np


def metal_available() -> bool:
    """Check if Metal is available on this system.

    Returns True if Metal framework can be imported and a default GPU device
    is available, False otherwise.
    """
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except ImportError:
        return False


# Conditional imports — only load Metal modules if available
try:
    import Metal
    import MetalPerformanceShaders as MPS
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


class MetalContext:
    """Metal GPU context managing device, command queue, and shader pipelines.

    Loads MSL shader files from shaders/ directory and compiles them into
    Metal pipeline states for custom kernels.
    """

    def __init__(self):
        if not _HAS_METAL:
            raise RuntimeError("Metal not available — install pyobjc-framework-Metal")

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found")

        self.queue = self.device.newCommandQueue()
        self._pipelines: dict[str, Any] = {}
        self._load_shaders()

    def _load_shaders(self):
        """Load MSL shader files and compile into pipeline states."""
        # Find shaders directory relative to this file
        shader_dir = os.path.join(os.path.dirname(__file__), "..", "shaders")

        # Map shader file to kernel function names
        shader_kernels = {
            "softmax": ["softmax"],
            "rmsnorm": ["rmsnorm"],
            "silu": ["silu"],
            "elementwise": ["add", "mul", "scale", "add_scale"],
            "rope": ["rope_forward"],
        }

        for shader_name, kernel_names in shader_kernels.items():
            path = os.path.join(shader_dir, f"{shader_name}.metal")
            if not os.path.exists(path):
                continue

            with open(path) as f:
                source = f.read()

            # Compile Metal library from source
            lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
            if err:
                raise RuntimeError(f"Failed to compile {shader_name}.metal: {err}")

            # Create pipeline state for each kernel function
            for fn_name in kernel_names:
                fn = lib.newFunctionWithName_(fn_name)
                if fn is None:
                    continue

                pipeline, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if err:
                    raise RuntimeError(f"Failed to create pipeline for {fn_name}: {err}")

                self._pipelines[fn_name] = pipeline


class MetalTensor:
    """GPU tensor backed by a Metal buffer.

    Uses MTLResourceStorageModeShared for unified memory — on Apple Silicon,
    CPU and GPU share the same physical memory, so no explicit copy is needed.
    """

    def __init__(
        self,
        ctx: MetalContext,
        data: np.ndarray | None = None,
        shape: list[int] | None = None,
        nbytes: int | None = None,
    ):
        """Create a Metal tensor.

        Args:
            ctx: Metal context
            data: Optional numpy array to upload (will be copied to GPU buffer)
            shape: Shape of the tensor (required if nbytes is specified)
            nbytes: Size in bytes (for allocating empty buffer)
        """
        self.ctx = ctx

        if data is not None:
            # Upload from numpy array
            arr = np.ascontiguousarray(data, dtype=np.float32)
            self.shape = list(arr.shape)
            self.nbytes = arr.nbytes

            # Create buffer with initial data (MTLResourceStorageModeShared)
            # PyObjC 12+ requires bytes, not ctypes.c_void_p
            self.buffer = ctx.device.newBufferWithBytes_length_options_(
                arr.tobytes(),
                arr.nbytes,
                Metal.MTLResourceStorageModeShared
            )
        elif nbytes is not None:
            # Allocate empty buffer
            self.shape = shape or []
            self.nbytes = nbytes
            self.buffer = ctx.device.newBufferWithLength_options_(
                nbytes,
                Metal.MTLResourceStorageModeShared
            )
        else:
            raise ValueError("Must provide either data or nbytes")

    def to_numpy(self) -> np.ndarray:
        """Read GPU buffer back to numpy array.

        Returns a copy of the buffer contents (not a view).
        """
        # Get pointer to buffer contents (PyObjC 12+ returns objc.varlist)
        ptr = self.buffer.contents()

        # PyObjC varlist.as_buffer(count) takes byte count, returns a memoryview
        arr = np.frombuffer(ptr.as_buffer(self.nbytes), dtype=np.float32).copy()

        return arr.reshape(self.shape) if self.shape else arr

    @classmethod
    def from_numpy(cls, ctx: MetalContext, arr: np.ndarray) -> MetalTensor:
        """Create MetalTensor from numpy array."""
        return cls(ctx, data=arr)


def mps_matmul(
    ctx: MetalContext,
    a_tensor: MetalTensor,
    b_tensor: MetalTensor,
    out_tensor: MetalTensor,
    M: int,
    N: int,
    K: int
):
    """Matrix multiplication using MetalPerformanceShaders.

    Computes: out = a @ b
    where a is (M, K), b is (K, N), out is (M, N).

    Args:
        ctx: Metal context
        a_tensor: Left matrix (M, K)
        b_tensor: Right matrix (K, N)
        out_tensor: Output matrix (M, N)
        M: Number of rows in a
        N: Number of columns in b
        K: Inner dimension (columns of a, rows of b)
    """
    # Create matrix descriptors
    # NOTE: rowBytes = columns * sizeof(float32) = columns * 4
    desc_a = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        M, K, K * 4, MPS.MPSDataTypeFloat32
    )
    desc_b = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        K, N, N * 4, MPS.MPSDataTypeFloat32
    )
    desc_c = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        M, N, N * 4, MPS.MPSDataTypeFloat32
    )

    # Wrap buffers as MPSMatrix
    mat_a = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(a_tensor.buffer, desc_a)
    mat_b = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(b_tensor.buffer, desc_b)
    mat_c = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(out_tensor.buffer, desc_c)

    # Create matmul kernel: C = alpha * A @ B + beta * C
    # Set beta=0 to compute C = A @ B (no accumulation)
    matmul = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        ctx.device,
        False,  # transposeLeft
        False,  # transposeRight
        M,      # resultRows
        N,      # resultColumns
        K,      # interiorColumns
        1.0,    # alpha
        0.0     # beta
    )

    # Encode and execute
    cmd_buf = ctx.queue.commandBuffer()
    matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        cmd_buf, mat_a, mat_b, mat_c
    )
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()


def dispatch_kernel(
    ctx: MetalContext,
    kernel_name: str,
    buffers: list[MetalTensor | int | float],
    grid_size: int,
    threadgroup_size: int = 256
):
    """Dispatch a custom MSL kernel.

    Args:
        ctx: Metal context
        kernel_name: Name of the kernel function (e.g., "softmax")
        buffers: List of arguments (MetalTensor, int, or float)
        grid_size: Total number of threads to dispatch
        threadgroup_size: Threads per threadgroup (default 256)
    """
    pipeline = ctx._pipelines.get(kernel_name)
    if pipeline is None:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    cmd_buf = ctx.queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    # Bind buffers and scalar arguments
    for i, arg in enumerate(buffers):
        if isinstance(arg, MetalTensor):
            encoder.setBuffer_offset_atIndex_(arg.buffer, 0, i)
        elif isinstance(arg, int):
            # Create buffer from packed uint32 bytes
            scalar_buf = ctx.device.newBufferWithBytes_length_options_(
                struct.pack('<I', arg), 4, Metal.MTLResourceStorageModeShared
            )
            encoder.setBuffer_offset_atIndex_(scalar_buf, 0, i)
        elif isinstance(arg, float):
            # Create buffer from packed float32 bytes
            scalar_buf = ctx.device.newBufferWithBytes_length_options_(
                struct.pack('<f', arg), 4, Metal.MTLResourceStorageModeShared
            )
            encoder.setBuffer_offset_atIndex_(scalar_buf, 0, i)

    # Dispatch threads
    threads_per_grid = Metal.MTLSizeMake(grid_size, 1, 1)
    threads_per_group = Metal.MTLSizeMake(min(threadgroup_size, grid_size), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_group)
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()


def gpu_linear(ctx: MetalContext, input_mt: MetalTensor, weight_data, out_features: int, in_features: int) -> MetalTensor:
    """GPU Linear: Y = X @ W^T using MPS matmul.

    weight_data is numpy array of shape [out_features, in_features].
    Transposes weight on CPU, then uses MPS matmul.
    """
    batch_seq = input_mt.shape[0] if len(input_mt.shape) == 2 else (input_mt.shape[0] * input_mt.shape[1])

    # Transpose weight [out, in] → [in, out]
    w_t = weight_data.reshape(out_features, in_features).T.copy()
    w_mt = MetalTensor.from_numpy(ctx, w_t)

    out_mt = MetalTensor(ctx, shape=[batch_seq, out_features], nbytes=batch_seq * out_features * 4)
    mps_matmul(ctx, input_mt, w_mt, out_mt, batch_seq, out_features, in_features)
    return out_mt


def gpu_rmsnorm(ctx: MetalContext, input_mt: MetalTensor, weight_data, n_rows: int, hidden_dim: int, eps: float = 1e-6) -> MetalTensor:
    """GPU RMSNorm using MSL kernel."""
    weight_mt = MetalTensor.from_numpy(ctx, weight_data.reshape(hidden_dim).copy())
    out_mt = MetalTensor(ctx, shape=[n_rows, hidden_dim], nbytes=n_rows * hidden_dim * 4)

    # rmsnorm kernel: 1 threadgroup per row, 256 threads per threadgroup
    dispatch_kernel(ctx, "rmsnorm", [input_mt, out_mt, weight_mt, hidden_dim, eps],
                    grid_size=n_rows * 256, threadgroup_size=256)
    return out_mt


def gpu_silu(ctx: MetalContext, input_mt: MetalTensor) -> MetalTensor:
    """GPU SiLU using MSL kernel."""
    n = 1
    for s in input_mt.shape:
        n *= s
    out_mt = MetalTensor(ctx, shape=input_mt.shape, nbytes=n * 4)
    dispatch_kernel(ctx, "silu", [input_mt, out_mt], grid_size=n, threadgroup_size=256)
    return out_mt


def gpu_add(ctx: MetalContext, a_mt: MetalTensor, b_mt: MetalTensor) -> MetalTensor:
    """GPU elementwise add using MSL kernel."""
    n = 1
    for s in a_mt.shape:
        n *= s
    out_mt = MetalTensor(ctx, shape=a_mt.shape, nbytes=n * 4)
    dispatch_kernel(ctx, "add", [a_mt, b_mt, out_mt], grid_size=n, threadgroup_size=256)
    return out_mt


def gpu_mul(ctx: MetalContext, a_mt: MetalTensor, b_mt: MetalTensor) -> MetalTensor:
    """GPU elementwise mul using MSL kernel."""
    n = 1
    for s in a_mt.shape:
        n *= s
    out_mt = MetalTensor(ctx, shape=a_mt.shape, nbytes=n * 4)
    dispatch_kernel(ctx, "mul", [a_mt, b_mt, out_mt], grid_size=n, threadgroup_size=256)
    return out_mt


def gpu_forward(ctx: MetalContext, model, input_tensor):
    """GPU forward pass: layer-by-layer with GPU matmuls and kernels.

    Hybrid approach:
    - Embedding: CPU (table lookup)
    - RMSNorm: GPU (MSL kernel)
    - Linear (Q/K/V/O, expert FFN, lm_head): GPU (MPS matmul)
    - Attention scores: CPU (causal mask + GQA complexity)
    - MoE routing: CPU (top-k selection)
    - Expert SwiGLU: GPU (matmul + silu + mul)
    - Residual adds: GPU (MSL kernel)
    """
    from .tensor import Tensor

    cfg = model.config
    batch = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    hidden = cfg.hidden_dim
    batch_seq = batch * seq_len

    # 1. Embedding (CPU)
    emb_out = model.embedding.forward(input_tensor)
    emb_data = emb_out.data.reshape(batch_seq, hidden).astype(np.float32)
    x = MetalTensor.from_numpy(ctx, emb_data)

    # 2. Transformer blocks
    for block in model.blocks:
        # RMSNorm (attn)
        normed = gpu_rmsnorm(ctx, x, block.attn_norm.weight.data, batch_seq, hidden, block.attn_norm.eps)

        # Attention (CPU fallback)
        normed_data = normed.to_numpy().reshape(batch, seq_len, hidden)
        normed_cpu = Tensor.from_numpy(normed_data)
        attn_out_cpu = block.attention.forward(normed_cpu)
        attn_data = attn_out_cpu.data.reshape(batch_seq, hidden).astype(np.float32)
        attn_mt = MetalTensor.from_numpy(ctx, attn_data)

        # Residual
        x = gpu_add(ctx, x, attn_mt)

        # RMSNorm (ffn)
        normed2 = gpu_rmsnorm(ctx, x, block.ffn_norm.weight.data, batch_seq, hidden, block.ffn_norm.eps)

        # MoE (hybrid: routing CPU, expert SwiGLU GPU)
        normed2_data = normed2.to_numpy()
        normed2_cpu = Tensor.from_numpy(normed2_data.reshape(batch, seq_len, hidden))

        # Router on CPU
        top_k = block.moe.top_k
        weights, indices = block.moe.router.forward(normed2_cpu)

        # Build inverted index: expert → tokens
        n_experts = len(block.moe.experts)
        expert_tokens = [[] for _ in range(n_experts)]
        expert_weight_idx = [[] for _ in range(n_experts)]
        for t in range(batch_seq):
            for k_idx in range(top_k):
                e_idx = indices[t, k_idx]
                expert_tokens[e_idx].append(t)
                expert_weight_idx[e_idx].append(k_idx)

        # Per-expert GPU SwiGLU
        moe_out_data = np.zeros((batch_seq, hidden), dtype=np.float32)

        for e_idx in range(n_experts):
            tokens = expert_tokens[e_idx]
            if not tokens:
                continue
            n_tok = len(tokens)

            # Gather token vectors
            batch_data = np.zeros((n_tok, hidden), dtype=np.float32)
            for i, t in enumerate(tokens):
                batch_data[i] = normed2_data[t]
            batch_mt = MetalTensor.from_numpy(ctx, batch_data)

            # GPU SwiGLU
            expert = block.moe.experts[e_idx]
            gate = gpu_linear(ctx, batch_mt, expert.gate.weight.data,
                            expert.gate.out_features, expert.gate.in_features)
            gate_silu = gpu_silu(ctx, gate)
            up = gpu_linear(ctx, batch_mt, expert.up.weight.data,
                          expert.up.out_features, expert.up.in_features)
            fused = gpu_mul(ctx, gate_silu, up)
            expert_out = gpu_linear(ctx, fused, expert.down.weight.data,
                                   expert.down.out_features, expert.down.in_features)
            e_out = expert_out.to_numpy()

            # Weighted scatter-add
            for i, t in enumerate(tokens):
                k_idx = expert_weight_idx[e_idx][i]
                alpha = weights[t, k_idx]
                moe_out_data[t] += alpha * e_out[i]

        moe_mt = MetalTensor.from_numpy(ctx, moe_out_data)
        x = gpu_add(ctx, x, moe_mt)

    # 3. Final RMSNorm
    normed_final = gpu_rmsnorm(ctx, x, model.final_norm.weight.data, batch_seq, hidden, model.final_norm.eps)

    # 4. LM Head
    vocab = cfg.vocab_size
    logits_mt = gpu_linear(ctx, normed_final, model.lm_head.weight.data, vocab, hidden)

    # 5. Read back to CPU Tensor
    logits_data = logits_mt.to_numpy().reshape(batch, seq_len, vocab)
    return Tensor.from_numpy(logits_data)


def gpu_train_step(ctx: MetalContext, trainer, input_tensor, targets):
    """GPU train step: forward on GPU, backward + optimizer on CPU.

    Uses M1 unified memory for zero-copy CPU↔GPU access.
    CPU forward runs first to populate backward caches, then GPU forward
    computes accelerated logits, then CPU backward + optimizer runs.

    Args:
        ctx: Metal context
        trainer: Trainer instance with model + optimizer state
        input_tensor: Input tensor [batch, seq_len]
        targets: Target tensor [batch, seq_len]

    Returns:
        Loss value (float)
    """
    # CPU forward to populate backward caches (needed for backward pass)
    _ = trainer.model.forward(input_tensor)

    # GPU forward for accelerated logits
    logits = gpu_forward(ctx, trainer.model, input_tensor)

    # CPU backward + optimizer using GPU logits + CPU caches
    return trainer.train_step_from_logits(logits, targets)
