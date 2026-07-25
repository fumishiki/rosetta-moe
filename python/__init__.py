# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""MoE Transformer neural network module.

Pure Python + NumPy implementation of a Mixture-of-Experts Transformer.
No framework dependencies (PyTorch, JAX, etc.) -- all operations and
gradients are hand-implemented using NumPy as the numerical backend.

Subpackages:
  - cpu/: CPU-only NumPy implementation (default)
  - gpu/: Metal GPU implementation using PyObjC (requires macOS + Apple Silicon)

Key components:
  - Tensor: thin wrapper around numpy ndarrays (tensor.py)
  - Layers: Embedding, RMSNorm, Linear, SwiGLU (layers.py)
  - Attention: Multi-Query Attention with RoPE (attention.py)
  - MoE: Router + expert dispatch + TransformerBlock (moe.py)
  - Model: full MoETransformer pipeline (model.py)
  - Generate: sampling strategies for text generation (generate.py)
  - Train: AdamW optimizer, loss, LR scheduling (train.py)
"""

# Re-export top-level API from existing files (backward compat)
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
    # Subpackages
    "cpu",
    "gpu",
    # Backward-compatible top-level exports
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
