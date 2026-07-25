# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""GPU (Metal) MoE Transformer implementation.

Uses PyObjC Metal/MPS for GPU-accelerated forward pass.
All intermediate activations stay as MetalTensors -- no NumPy arrays
in the compute path.

Requires: pyobjc-framework-Metal, pyobjc-framework-MetalPerformanceShaders
"""

from .metal_tensor import MetalContext, MetalTensor, metal_available
from .metal_layers import (
    MetalEmbedding,
    MetalRMSNorm,
    MetalLinear,
    MetalSwiGLU,
    seed_rng,
)
from .metal_attention import MetalMQAttention
from .metal_moe import MetalRouter, MetalMoELayer, MetalTransformerBlock
from .metal_model import MetalMoETransformer
from .metal_train import MetalTrainer

__all__ = [
    "MetalContext",
    "MetalTensor",
    "metal_available",
    "MetalEmbedding",
    "MetalRMSNorm",
    "MetalLinear",
    "MetalSwiGLU",
    "seed_rng",
    "MetalMQAttention",
    "MetalRouter",
    "MetalMoELayer",
    "MetalTransformerBlock",
    "MetalMoETransformer",
    "MetalTrainer",
]
