# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Training utilities for MoE Transformer.

Provides:
  - AdamW optimizer with bias correction and weight decay
  - Cross-entropy loss with softmax gradient
  - LR scheduling: linear warmup + cosine decay
  - Gradient clipping by global L2 norm
  - Activation checkpointing scaffolding
  - Mixed-precision training scaffolding (loss scaler, master weights)

All gradients are hand-derived (no autograd framework).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .model import MoETransformer
from .tensor import DType, Tensor


def clip_grad_by_global_norm(grad: Tensor, clip_norm: float) -> Tensor:
    """Clip gradient by global L2 norm.

    GradClip: if ||g||_2 > clip_norm then g' = g * clip_norm / (||g||_2 + eps)

    The epsilon guard (1e-12) prevents division by zero when ||g|| ~ 0.
    """
    if clip_norm <= 0.0:
        return grad.clone()
    norm = float(np.sqrt(np.sum(grad.data ** 2)))
    if norm <= clip_norm:
        return grad.clone()
    return grad.scale(clip_norm / (norm + 1e-12))


class CheckpointStorage:
    """Block-level activation checkpoint storage.

    Stores intermediate activations at transformer block boundaries so
    they can be re-used (or recomputed from) during the backward pass.
    """

    def __init__(self, enabled: bool = True):
        self._checkpoints: dict[int, Tensor] = {}
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save(self, block_idx: int, tensor: Tensor) -> None:
        if self._enabled:
            self._checkpoints[block_idx] = tensor

    def get(self, block_idx: int) -> Tensor | None:
        return self._checkpoints.get(block_idx)

    def clear(self) -> None:
        self._checkpoints.clear()

    def __len__(self) -> int:
        return len(self._checkpoints)


class CheckpointContext:
    """Manages when to checkpoint based on segment_size.

    Every ``segment_size`` blocks, the activation is saved.  During
    backward, if a block's activation isn't saved, it can be recomputed
    from the nearest checkpoint.
    """

    def __init__(self, segment_size: int = 1):
        size = max(segment_size, 1)
        self.storage = CheckpointStorage(enabled=segment_size > 0)
        self.segment_size = size

    @classmethod
    def disabled(cls) -> CheckpointContext:
        ctx = cls.__new__(cls)
        ctx.storage = CheckpointStorage(enabled=False)
        ctx.segment_size = 1
        return ctx

    def should_checkpoint(self, block_idx: int) -> bool:
        return self.storage.enabled and (block_idx % self.segment_size == 0)

    def maybe_save(self, block_idx: int, tensor: Tensor) -> None:
        if self.should_checkpoint(block_idx):
            self.storage.save(block_idx, tensor)

    def get_checkpoint(self, block_idx: int) -> Tensor | None:
        cp_idx = (block_idx // self.segment_size) * self.segment_size
        return self.storage.get(cp_idx)

    def clear(self) -> None:
        self.storage.clear()


class LossScaleMode(Enum):
    """Loss scaling strategy for mixed-precision training."""
    STATIC = auto()
    DYNAMIC = auto()


class LossScaler:
    """Loss scaler for mixed-precision training.

    Dynamic scaling: starts with a large scale (65536), halves on overflow,
    doubles every ``scale_window`` steps without overflow.  This maximizes
    the use of fp16's dynamic range while recovering from gradient overflow.
    """

    def __init__(
        self,
        mode: LossScaleMode = LossScaleMode.DYNAMIC,
        init_scale: float = 65536.0,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
    ):
        self.mode = mode
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._scale = init_scale
        self._growth_tracker = 0
        self._overflow = False

    @classmethod
    def static(cls, scale: float = 1.0) -> LossScaler:
        return cls(mode=LossScaleMode.STATIC, init_scale=scale)

    @classmethod
    def dynamic(cls) -> LossScaler:
        return cls()

    @property
    def scale(self) -> float:
        return self._scale

    def scale_loss(self, loss: float) -> float:
        return loss * self._scale

    def unscale_grads(self, grad: float) -> float:
        return grad / self._scale

    def check_overflow(self, grads: list[float]) -> bool:
        self._overflow = any(not math.isfinite(g) for g in grads)
        return self._overflow

    def update(self) -> None:
        """Update scale based on overflow history."""
        if self.mode == LossScaleMode.STATIC:
            return
        if self._overflow:
            # Overflow detected: halve scale and reset growth counter
            self._scale /= self.scale_factor
            self._growth_tracker = 0
            self._overflow = False
            return
        # No overflow: try to grow scale after enough clean steps
        self._growth_tracker += 1
        if self._growth_tracker >= self.scale_window:
            self._scale *= self.scale_factor
            self._growth_tracker = 0

    def should_skip_step(self) -> bool:
        return self._overflow


@dataclass
class MixedPrecisionConfig:
    """Mixed-precision training configuration."""

    enabled: bool = False
    compute_dtype: DType = DType.F16
    loss_scale_mode: LossScaleMode = LossScaleMode.DYNAMIC
    fp32_layers: list[str] = field(default_factory=lambda: ["final_norm", "lm_head"])

    @classmethod
    def fp16(cls) -> MixedPrecisionConfig:
        return cls(enabled=True, compute_dtype=DType.F16)

    @classmethod
    def disabled(cls) -> MixedPrecisionConfig:
        return cls()

    def is_fp32_layer(self, layer_name: str) -> bool:
        return any(s in layer_name for s in self.fp32_layers)


class MasterWeights:
    """FP32 master copies of parameters for mixed-precision training.

    In mixed-precision, weights are stored in fp16 for compute but
    optimizer state and updates are applied to fp32 master copies
    to preserve precision in small gradient updates.
    """

    def __init__(self, params: list[Tensor] | None = None):
        if params is not None:
            self._weights = [Tensor.zeros(p.shape, DType.F32) for p in params]
        else:
            self._weights = []

    @property
    def weights(self) -> list[Tensor]:
        return self._weights

    def __len__(self) -> int:
        return len(self._weights)


@dataclass
class TrainConfig:
    """Training configuration."""

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    total_steps: int = 100000
    grad_clip: float = 1.0
    aux_loss_alpha: float = 0.01

    @classmethod
    def default(cls) -> TrainConfig:
        """Return default training config."""
        return cls()


class AdamWState:
    """AdamW optimizer state for a single parameter.

    Stores first moment (m) and second moment (v) estimates,
    both initialized to zeros.
    """

    def __init__(self, shape: tuple[int, ...]):
        """Initialize optimizer state."""
        self.m = np.zeros(shape, dtype=np.float32)  # First moment (mean of gradients)
        self.v = np.zeros(shape, dtype=np.float32)  # Second moment (mean of squared gradients)
        # Scratch buffer reused in every optimizer step to avoid temporary arrays.
        self.tmp = np.zeros(shape, dtype=np.float32)


class Trainer:
    """Training loop with AdamW optimizer and LR scheduling.

    Combines:
      - Cross-entropy loss with softmax gradient
      - AdamW optimizer (decoupled weight decay)
      - Linear warmup + cosine decay LR schedule
      - MoE auxiliary loss for load balancing
      - Global gradient norm clipping
    """

    def __init__(self, model: MoETransformer, config: TrainConfig):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.step = 0

        # Parameters are static after model construction; cache once.
        self.params = model.parameters()
        # One AdamW state per parameter tensor
        self.states = [AdamWState(p.shape) for p in self.params]

        # Reusable buffers for cross-entropy (shape: [num_tokens, vocab_size]).
        self._loss_probs: np.ndarray | None = None
        self._loss_row_idx: np.ndarray | None = None

    def _ensure_loss_buffers(
        self, num_tokens: int, vocab_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get reusable loss buffers sized for the current batch."""
        if self._loss_probs is None or self._loss_probs.shape != (num_tokens, vocab_size):
            self._loss_probs = np.empty((num_tokens, vocab_size), dtype=np.float32)
        if self._loss_row_idx is None or self._loss_row_idx.shape[0] != num_tokens:
            self._loss_row_idx = np.arange(num_tokens, dtype=np.int64)
        return self._loss_probs, self._loss_row_idx

    def get_lr(self) -> float:
        """Get current learning rate with warmup and cosine decay.

        LR schedule:
          if step < warmup_steps:
            lr = base_lr * step / warmup_steps           (linear warmup)
          else:
            progress = (step - warmup) / (total - warmup)
            lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * progress))  (cosine decay)
          where min_lr = 0.1 * base_lr
        """
        if self.step < self.config.warmup_steps:
            # Linear warmup: lr ramps from 0 to base_lr
            return self.config.lr * self.step / self.config.warmup_steps

        # Cosine decay from base_lr to min_lr (10% of base_lr)
        progress = (self.step - self.config.warmup_steps) / (
            self.config.total_steps - self.config.warmup_steps
        )
        progress = min(progress, 1.0)
        min_lr = self.config.lr * 0.1
        return min_lr + 0.5 * (self.config.lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> tuple[float, Tensor]:
        """Compute cross entropy loss and gradient.

        CrossEntropy: L = -mean_t(log(softmax(logits_t)[target_t]))

        Gradient (combined softmax + cross-entropy):
          d_logits = softmax(logits) - one_hot(targets)
        This elegant form arises because d/dx_i[-log(softmax(x)_c)] = p_i - 1{i=c}.

        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]

        Returns:
            loss: Scalar loss value
            grad: Gradient of logits [batch, seq_len, vocab_size]
        """
        batch, seq_len, vocab_size = logits.shape

        flat_logits = logits.data.reshape(-1, vocab_size)
        flat_targets = targets.data.reshape(-1).astype(np.int64)
        num_tokens = batch * seq_len

        # Numerically stable softmax into reusable buffer:
        # probs = exp(logits - max(logits)) / sum(exp(logits - max(logits)))
        probs, row_idx = self._ensure_loss_buffers(num_tokens, vocab_size)
        max_logits = np.max(flat_logits, axis=-1, keepdims=True)
        np.subtract(flat_logits, max_logits, out=probs)
        np.exp(probs, out=probs)
        np.divide(probs, np.sum(probs, axis=-1, keepdims=True), out=probs)

        # Cross entropy: L = -mean(log(p[target]))
        # Clamp probs to avoid log(0)
        loss = -np.mean(np.log(np.maximum(probs[row_idx, flat_targets], 1e-12)))

        # Gradient: d_logits = (softmax(logits) - one_hot(targets)) / num_tokens.
        # Reuse probs in-place to avoid a full-size copy.
        probs[row_idx, flat_targets] -= 1.0
        probs /= num_tokens

        grad = probs.reshape((batch, seq_len, vocab_size))
        return float(loss), Tensor.from_numpy(grad)

    def _adamw_step(
        self,
        param: Tensor,
        grad: np.ndarray,
        state: AdamWState,
        lr: float,
        bias_c1_inv: float,
        bias_c2_inv: float,
    ) -> None:
        """Perform AdamW optimizer step (in-place).

        AdamW update rule:
          m = beta1 * m + (1 - beta1) * g              (first moment update)
          v = beta2 * v + (1 - beta2) * g^2            (second moment update)
          m_hat = m / (1 - beta1^t)                     (bias correction)
          v_hat = v / (1 - beta2^t)                     (bias correction)
          w -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)

        Note: weight decay is decoupled (applied to w directly, not to m_hat).
        This is the key difference between AdamW and L2-regularized Adam.
        """
        m = state.m
        v = state.v
        tmp = state.tmp

        # Update moments (in-place):
        # m = beta1 * m + (1 - beta1) * grad
        np.multiply(grad, (1.0 - self.config.beta1), out=tmp)
        np.multiply(m, self.config.beta1, out=m)
        np.add(m, tmp, out=m)

        # v = beta2 * v + (1 - beta2) * grad^2
        np.multiply(grad, grad, out=tmp)
        np.multiply(tmp, (1.0 - self.config.beta2), out=tmp)
        np.multiply(v, self.config.beta2, out=v)
        np.add(v, tmp, out=v)

        # Build inv_denom = 1 / (sqrt(v_hat) + eps) in tmp.
        np.multiply(v, bias_c2_inv, out=tmp)
        np.sqrt(tmp, out=tmp)
        tmp += self.config.eps
        np.reciprocal(tmp, out=tmp)

        # tmp = m_hat / (sqrt(v_hat) + eps), where m_hat = m * bias_c1_inv
        np.multiply(m, tmp, out=tmp)
        np.multiply(tmp, bias_c1_inv, out=tmp)

        # Decoupled weight decay + Adam update (both in-place on parameter data).
        if self.config.weight_decay != 0.0:
            param._data *= (1.0 - lr * self.config.weight_decay)
        param._data -= lr * tmp

    def train_step(self, input_ids: Tensor, targets: Tensor) -> float:
        """Perform single training step.

        Pipeline:
          1. Increment step counter (matching Go: step is 1-indexed)
          2. Zero gradients on all parameters
          3. Forward pass -> logits
          4. Cross-entropy loss + gradient
          5. Add MoE auxiliary loss (load balancing)
          6. Backward pass (computes per-parameter gradients)
          7. Global gradient norm clipping
          8. AdamW update for each parameter

        Args:
            input_ids: Input token IDs [batch, seq_len]
            targets: Target token IDs [batch, seq_len]

        Returns:
            Loss value (cross-entropy + aux loss)
        """
        # Increment step counter before update (matching Go: first step = 1)
        self.step += 1

        # Zero gradients before forward/backward
        params = self.params
        for param in params:
            param._grad = None

        # Forward pass
        logits = self.model.forward(input_ids)

        # Compute loss and gradient
        loss, grad_logits = self._compute_loss(logits, targets)

        # Add auxiliary loss from MoE routers (load balancing regularization)
        aux_loss = self.model.total_aux_loss(self.config.aux_loss_alpha)
        total_loss = loss + aux_loss

        # Backward pass — computes and stores per-parameter gradients on param._grad
        self.model.backward(grad_logits)

        # Global gradient norm clipping across all parameters
        clip_norm = self.config.grad_clip
        if clip_norm > 0:
            # Compute global L2 norm of all gradients
            total_norm_sq = 0.0
            for param in params:
                if param._grad is not None:
                    total_norm_sq += float(np.sum(param._grad ** 2))
            global_norm = np.sqrt(total_norm_sq)

            # If needed, scale all gradients in-place once (Rust-style).
            if global_norm > clip_norm:
                clip_coeff = clip_norm / (global_norm + 1e-12)
                for param in params:
                    if param._grad is not None:
                        param._grad *= clip_coeff

        # Compute step-wise constants once (instead of per parameter).
        lr = self.get_lr()
        beta1_pow_t = self.config.beta1**self.step
        beta2_pow_t = self.config.beta2**self.step
        bias_c1_inv = 1.0 / max(1.0 - beta1_pow_t, 1e-12)
        bias_c2_inv = 1.0 / max(1.0 - beta2_pow_t, 1e-12)

        # AdamW update
        for param, state in zip(params, self.states):
            if param._grad is None:
                continue
            self._adamw_step(param, param._grad, state, lr, bias_c1_inv, bias_c2_inv)

        return total_loss

    def get_step(self) -> int:
        """Return current step."""
        return self.step
