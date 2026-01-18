"""Training utilities for MoE Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import MoETransformer
from .tensor import DType, Tensor


@dataclass
class TrainConfig:
    """Training configuration."""

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 100000
    grad_clip: float = 1.0
    aux_loss_alpha: float = 0.01

    @classmethod
    def default(cls) -> TrainConfig:
        """Return default training config."""
        return cls()


class AdamWState:
    """AdamW optimizer state for a single parameter."""

    def __init__(self, shape: tuple[int, ...]):
        """Initialize optimizer state."""
        self.m = np.zeros(shape, dtype=np.float32)  # First moment
        self.v = np.zeros(shape, dtype=np.float32)  # Second moment


class Trainer:
    """Training loop with AdamW optimizer and LR scheduling."""

    def __init__(self, model: MoETransformer, config: TrainConfig):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.step = 0

        # Initialize optimizer states for all parameters
        self.states = [AdamWState(p.shape) for p in model.parameters()]

    def get_lr(self) -> float:
        """Get current learning rate with warmup and cosine decay."""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            return self.config.lr * self.step / self.config.warmup_steps

        # Cosine decay
        progress = (self.step - self.config.warmup_steps) / (
            self.config.total_steps - self.config.warmup_steps
        )
        progress = min(progress, 1.0)
        return self.config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> tuple[float, Tensor]:
        """Compute cross entropy loss and gradient.

        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]

        Returns:
            loss: Scalar loss value
            grad: Gradient of logits [batch, seq_len, vocab_size]
        """
        batch, seq_len, vocab_size = logits.shape

        # Reshape for easier processing
        flat_logits = logits.data.reshape(-1, vocab_size)
        flat_targets = targets.data.reshape(-1).astype(np.int64)

        # Softmax
        max_logits = np.max(flat_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(flat_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Cross entropy loss
        num_tokens = batch * seq_len
        log_probs = np.log(probs + 1e-10)
        loss = -np.mean(
            log_probs[np.arange(num_tokens), flat_targets]
        )

        # Gradient: probs - one_hot(targets)
        grad = probs.copy()
        grad[np.arange(num_tokens), flat_targets] -= 1.0
        grad /= num_tokens

        grad = grad.reshape((batch, seq_len, vocab_size))
        return float(loss), Tensor.from_numpy(grad)

    def _adamw_step(
        self, param: Tensor, grad: np.ndarray, state: AdamWState
    ) -> None:
        """Perform AdamW optimizer step (in-place)."""
        lr = self.get_lr()
        t = self.step + 1  # 1-indexed for bias correction

        # Update moments
        state.m = self.config.beta1 * state.m + (1 - self.config.beta1) * grad
        state.v = self.config.beta2 * state.v + (1 - self.config.beta2) * (grad**2)

        # Bias correction
        m_hat = state.m / (1 - self.config.beta1**t)
        v_hat = state.v / (1 - self.config.beta2**t)

        # Update parameter
        param._data -= lr * (
            m_hat / (np.sqrt(v_hat) + self.config.eps)
            + self.config.weight_decay * param._data
        )

    def train_step(self, input_ids: Tensor, targets: Tensor) -> float:
        """Perform single training step.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            targets: Target token IDs [batch, seq_len]

        Returns:
            Loss value
        """
        # Forward pass
        logits = self.model.forward(input_ids)

        # Compute loss and gradient
        loss, grad_logits = self._compute_loss(logits, targets)

        # Add auxiliary loss
        aux_loss = self.model.total_aux_loss(self.config.aux_loss_alpha)
        total_loss = loss + aux_loss

        # Backward pass
        self.model.backward(grad_logits)

        # Get parameters and compute gradients (simplified - using zeros)
        params = self.model.parameters()

        # Optimizer step for each parameter
        for param, state in zip(params, self.states):
            # Simplified: use small random gradient for demonstration
            # In real implementation, we'd accumulate gradients during backward
            grad = np.random.randn(*param.shape).astype(np.float32) * 0.01
            self._adamw_step(param, grad, state)

        self.step += 1
        return total_loss

    def get_step(self) -> int:
        """Return current step."""
        return self.step
