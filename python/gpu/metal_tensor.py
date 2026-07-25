# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU tensor and context for MoE Transformer.

Provides GPU-resident tensor type backed by Metal buffers.
On Apple Silicon, MTLResourceStorageModeShared means unified memory --
CPU and GPU share the same physical memory, so buffer creation is zero-copy.

Design:
  - MetalTensor wraps a MTLBuffer with shape metadata
  - MetalContext owns device, command queue, and compiled shader pipelines
  - All compute kernels dispatch through MetalContext
  - MPS (MetalPerformanceShaders) used for matmul (GEMM)
"""

from __future__ import annotations

import os
import struct
from functools import reduce
from typing import Any

import numpy as np

try:
    import Metal
    import MetalPerformanceShaders as MPS
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


def metal_available() -> bool:
    """Check if Metal is available on this system."""
    if not _HAS_METAL:
        return False
    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except Exception:
        return False


class MetalContext:
    """Metal GPU context managing device, command queue, and shader pipelines."""

    def __init__(self):
        if not _HAS_METAL:
            raise RuntimeError("Metal not available -- install pyobjc-framework-Metal")

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found")

        self.queue = self.device.newCommandQueue()
        self._pipelines: dict[str, Any] = {}
        self._load_shaders()

    def _load_shaders(self):
        """Load MSL shader files and compile into pipeline states."""
        shader_dir = os.path.join(os.path.dirname(__file__), "..", "..", "shaders")

        shader_kernels = {
            "softmax": ["softmax"],
            "rmsnorm": ["rmsnorm"],
            "silu": ["silu"],
            "elementwise": ["add", "mul", "scale", "add_scale"],
            "embedding": ["embedding_gather"],
            "attention": [
                "rope_inplace",
                "causal_mask_fill",
                "attention_scores",
                "attention_weighted_sum",
                "moe_topk",
                "moe_topk_extract_weight",
                "row_scale",
                "moe_scatter_add",
                "moe_gather",
                "transpose_2d",
                "cross_entropy_forward",
                "cross_entropy_backward",
                "reduce_sum",
            ],
        }

        for shader_name, kernel_names in shader_kernels.items():
            path = os.path.join(shader_dir, f"{shader_name}.metal")
            if not os.path.exists(path):
                continue

            with open(path) as f:
                source = f.read()

            lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
            if err:
                raise RuntimeError(f"Failed to compile {shader_name}.metal: {err}")

            for fn_name in kernel_names:
                fn = lib.newFunctionWithName_(fn_name)
                if fn is None:
                    continue
                pipeline, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
                if err:
                    raise RuntimeError(f"Failed to create pipeline for {fn_name}: {err}")
                self._pipelines[fn_name] = pipeline

    def dispatch_kernel(
        self,
        kernel_name: str,
        buffers: list,
        grid_size: int,
        threadgroup_size: int = 256,
    ):
        """Dispatch a custom MSL kernel.

        Args:
            kernel_name: Name of the kernel function
            buffers: List of (MetalTensor | int | float) arguments
            grid_size: Total number of threads
            threadgroup_size: Threads per threadgroup
        """
        pipeline = self._pipelines.get(kernel_name)
        if pipeline is None:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        cmd_buf = self.queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        for i, arg in enumerate(buffers):
            if isinstance(arg, MetalTensor):
                encoder.setBuffer_offset_atIndex_(arg.buffer, 0, i)
            elif isinstance(arg, int):
                scalar_buf = self.device.newBufferWithBytes_length_options_(
                    struct.pack('<I', arg), 4, Metal.MTLResourceStorageModeShared
                )
                encoder.setBuffer_offset_atIndex_(scalar_buf, 0, i)
            elif isinstance(arg, float):
                scalar_buf = self.device.newBufferWithBytes_length_options_(
                    struct.pack('<f', arg), 4, Metal.MTLResourceStorageModeShared
                )
                encoder.setBuffer_offset_atIndex_(scalar_buf, 0, i)

        threads_per_grid = Metal.MTLSizeMake(grid_size, 1, 1)
        threads_per_group = Metal.MTLSizeMake(min(threadgroup_size, grid_size), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_group)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

    def mps_matmul(
        self,
        a: MetalTensor,
        b: MetalTensor,
        out: MetalTensor,
        M: int,
        N: int,
        K: int,
        transpose_left: bool = False,
        transpose_right: bool = False,
    ):
        """Matrix multiplication using MPS: out = a @ b.

        a is (M, K), b is (K, N), out is (M, N).
        """
        row_bytes_a = (M if transpose_left else K) * 4
        row_bytes_b = (K if transpose_right else N) * 4

        desc_a = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            K if transpose_left else M,
            M if transpose_left else K,
            row_bytes_a,
            MPS.MPSDataTypeFloat32,
        )
        desc_b = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            N if transpose_right else K,
            K if transpose_right else N,
            row_bytes_b,
            MPS.MPSDataTypeFloat32,
        )
        desc_c = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            M, N, N * 4, MPS.MPSDataTypeFloat32,
        )

        mat_a = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(a.buffer, desc_a)
        mat_b = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(b.buffer, desc_b)
        mat_c = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(out.buffer, desc_c)

        matmul = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
            self.device,
            transpose_left,
            transpose_right,
            M, N, K,
            1.0, 0.0,
        )

        cmd_buf = self.queue.commandBuffer()
        matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
            cmd_buf, mat_a, mat_b, mat_c,
        )
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()


class MetalTensor:
    """GPU tensor backed by a Metal buffer.

    Uses MTLResourceStorageModeShared for unified memory on Apple Silicon.
    """

    __slots__ = ("ctx", "shape", "nbytes", "buffer", "numel", "grad")

    def __init__(
        self,
        ctx: MetalContext,
        data: np.ndarray | None = None,
        shape: list[int] | None = None,
        nbytes: int | None = None,
    ):
        self.ctx = ctx
        self.grad: MetalTensor | None = None

        if data is not None:
            arr = np.ascontiguousarray(data, dtype=np.float32)
            self.shape = list(arr.shape)
            self.nbytes = arr.nbytes
            self.numel = arr.size
            self.buffer = ctx.device.newBufferWithBytes_length_options_(
                arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared,
            )
        elif nbytes is not None:
            self.shape = shape or []
            self.nbytes = nbytes
            self.numel = nbytes // 4
            self.buffer = ctx.device.newBufferWithLength_options_(
                nbytes, Metal.MTLResourceStorageModeShared,
            )
        else:
            raise ValueError("Must provide either data or nbytes")

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """Read GPU buffer to numpy array.

        Args:
            copy: If True (default), return an owning copy. If False, return
                a zero-copy view over unified memory (StorageModeShared).
        """
        ptr = self.buffer.contents()
        arr = np.frombuffer(ptr.as_buffer(self.nbytes), dtype=np.float32)
        if copy:
            arr = arr.copy()
        return arr.reshape(self.shape) if self.shape else arr

    def numpy_view(self) -> np.ndarray:
        """Return a zero-copy numpy view over unified memory."""
        return self.to_numpy(copy=False)

    @classmethod
    def from_numpy(cls, ctx: MetalContext, arr: np.ndarray) -> MetalTensor:
        """Create MetalTensor from numpy array."""
        return cls(ctx, data=arr)

    @classmethod
    def zeros(cls, ctx: MetalContext, shape: list[int]) -> MetalTensor:
        """Create zero-initialized MetalTensor."""
        n = reduce(lambda a, b: a * b, shape, 1)
        return cls(ctx, data=np.zeros(shape, dtype=np.float32))

    @classmethod
    def empty(cls, ctx: MetalContext, shape: list[int]) -> MetalTensor:
        """Create uninitialized MetalTensor."""
        n = reduce(lambda a, b: a * b, shape, 1)
        return cls(ctx, shape=shape, nbytes=n * 4)
