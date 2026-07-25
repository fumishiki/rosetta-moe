# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""CPU-only MoE Transformer implementation.

Pure Python + NumPy implementation. On macOS >= 14, NumPy dispatches matmul
to Apple Accelerate (BLAS/AMX) for hardware-accelerated linear algebra.
No Metal or PyObjC imports.
"""

from .tensor import Tensor, DType, seed_rng
from .config import Config
from .layers import Embedding, RMSNorm, Linear, SwiGLU
from .attention import MQAttention
from .moe import Router, MoELayer, TransformerBlock
from .model import MoETransformer, tiny_model, default_model
from .generate import (
    SamplingStrategy,
    GreedySampling,
    TemperatureSampling,
    TopKSampling,
    TopPSampling,
    generate,
    generate_greedy,
    generate_sample,
    generate_top_k,
    generate_top_p,
)
from .train import (
    TrainConfig,
    Trainer,
    CheckpointStorage,
    CheckpointContext,
    LossScaleMode,
    LossScaler,
    MixedPrecisionConfig,
    MasterWeights,
    clip_grad_by_global_norm,
)

__all__ = [
    "Tensor",
    "DType",
    "seed_rng",
    "Config",
    "Embedding",
    "RMSNorm",
    "Linear",
    "SwiGLU",
    "MQAttention",
    "Router",
    "MoELayer",
    "TransformerBlock",
    "MoETransformer",
    "tiny_model",
    "default_model",
    "SamplingStrategy",
    "GreedySampling",
    "TemperatureSampling",
    "TopKSampling",
    "TopPSampling",
    "generate",
    "generate_greedy",
    "generate_sample",
    "generate_top_k",
    "generate_top_p",
    "TrainConfig",
    "Trainer",
    "CheckpointStorage",
    "CheckpointContext",
    "LossScaleMode",
    "LossScaler",
    "MixedPrecisionConfig",
    "MasterWeights",
    "clip_grad_by_global_norm",
]
