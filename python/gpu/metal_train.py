# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Metal GPU training helpers for MoE Transformer.

This module keeps forward and loss computation on GPU kernels.
By default, `train_step(..., readback=False)` performs no host readback and
returns a GPU scalar buffer. Optional scalar readback is available for logging.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .metal_tensor import MetalContext, MetalTensor
from .metal_model import MetalMoETransformer


class MetalTrainer:
    """GPU trainer wrapper for forward + cross-entropy loss."""

    def __init__(self, gpu_model: MetalMoETransformer):
        self.gpu_model = gpu_model
        self.ctx = gpu_model.ctx

    @classmethod
    def create(
        cls,
        ctx: MetalContext,
        model_config: Config,
    ) -> MetalTrainer:
        """Create a GPU trainer with GPU-only model state."""
        from .metal_layers import seed_rng as gpu_seed_rng

        gpu_seed_rng(42)
        gpu_model = MetalMoETransformer(ctx, model_config)
        return cls(gpu_model)

    def _cross_entropy_loss_sum_tensor(
        self,
        logits_mt: MetalTensor,
        targets_mt: MetalTensor,
        num_tokens: int,
    ) -> MetalTensor:
        """Compute cross-entropy loss sum on GPU and keep result on device."""
        vocab_size = self.gpu_model.config.vocab_size

        per_token = MetalTensor.empty(self.ctx, [num_tokens])
        self.ctx.dispatch_kernel(
            "cross_entropy_forward",
            [logits_mt, targets_mt, per_token, vocab_size],
            grid_size=num_tokens * 256,
            threadgroup_size=256,
        )

        loss_sum = MetalTensor.empty(self.ctx, [1])
        self.ctx.dispatch_kernel(
            "reduce_sum",
            [per_token, loss_sum, num_tokens],
            grid_size=256,
            threadgroup_size=256,
        )

        return loss_sum

    def _cross_entropy_mean_loss_tensor(
        self,
        logits_mt: MetalTensor,
        targets: np.ndarray,
    ) -> tuple[MetalTensor, int]:
        """Compute CE loss sum tensor and token count from CPU targets."""
        batch, seq_len = targets.shape
        num_tokens = batch * seq_len
        targets_flat = targets.astype(np.float32, copy=False).reshape(-1)
        targets_mt = MetalTensor(self.ctx, data=targets_flat)
        return self._cross_entropy_loss_sum_tensor(logits_mt, targets_mt, num_tokens), num_tokens

    def train_step_gpu_tensors(
        self,
        token_ids_mt: MetalTensor,
        targets_mt: MetalTensor,
        *,
        batch: int,
        seq_len: int,
        readback: bool = False,
    ) -> MetalTensor | float:
        """GPU forward + CE loss from GPU-resident input/targets."""
        logits_mt = self.gpu_model.forward_tokens_tensor(token_ids_mt, batch, seq_len)
        num_tokens = batch * seq_len
        loss_sum = self._cross_entropy_loss_sum_tensor(logits_mt, targets_mt, num_tokens)

        if not readback:
            return loss_sum

        return float(loss_sum.numpy_view().reshape(-1)[0]) / float(num_tokens)

    def train_step(
        self,
        token_ids: np.ndarray,
        targets: np.ndarray,
        *,
        readback: bool = False,
    ) -> MetalTensor | float:
        """GPU forward + GPU CE loss.

        Args:
            token_ids: Input [batch, seq_len] token ids.
            targets: Target [batch, seq_len] token ids.
            readback: If True, read mean loss scalar to CPU.

        Returns:
            `MetalTensor([1])` loss sum when `readback=False`.
            `float` mean loss when `readback=True`.
        """
        logits_mt = self.gpu_model.forward(token_ids)
        loss_sum, num_tokens = self._cross_entropy_mean_loss_tensor(logits_mt, targets)

        if not readback:
            return loss_sum

        loss = float(loss_sum.numpy_view().reshape(-1)[0]) / float(num_tokens)
        return loss
