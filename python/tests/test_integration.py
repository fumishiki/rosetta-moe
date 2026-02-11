# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Integration tests for MoE Transformer."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from python.tensor import Tensor, DType
from python.layers import Linear, RMSNorm, SwiGLU
from python.config import Config
from python.attention import MQAttention
from python.moe import Router, MoELayer, TransformerBlock
from python.model import MoETransformer
from python.generate import (
    SamplingStrategy,
    GreedySampling,
    TemperatureSampling,
    TopKSampling,
    TopPSampling,
    generate,
    generate_greedy,
)
from python.train import (
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

np.random.seed(42)


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class TestTensor:
    def test_zeros(self):
        t = Tensor.zeros((2, 3))
        assert t.shape == (2, 3)
        assert np.all(t.data == 0.0)

    def test_ones(self):
        t = Tensor.ones((2, 3))
        assert t.shape == (2, 3)
        assert np.all(t.data == 1.0)

    def test_from_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = Tensor.from_numpy(arr)
        assert t.shape == (2, 2)
        np.testing.assert_allclose(t.data, arr)

    def test_add(self):
        a = Tensor.ones((2, 2))
        b = Tensor.ones((2, 2))
        c = a.add(b)
        assert np.all(c.data == 2.0)

    def test_scale(self):
        t = Tensor.ones((2, 2)).scale(3.0)
        assert np.all(t.data == 3.0)

    def test_silu(self):
        t = Tensor.from_numpy(np.array([0.0, 1.0, -1.0]))
        out = t.silu()
        assert out.shape == (3,)
        assert np.all(np.isfinite(out.data))

    def test_softmax(self):
        t = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]]))
        out = t.softmax()
        assert out.shape == (1, 3)
        np.testing.assert_allclose(np.sum(out.data), 1.0, atol=1e-6)

    def test_matmul(self):
        a = Tensor.randn((2, 3))
        b = Tensor.randn((3, 4))
        c = a.matmul(b)
        assert c.shape == (2, 4)

    def test_transpose(self):
        t = Tensor.zeros((2, 3))
        out = t.transpose()
        assert out.shape == (3, 2)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class TestLayers:
    def test_linear_batch(self):
        layer = Linear(8, 16)
        x = Tensor.randn((2, 4, 8))
        out = layer.forward(x)
        assert out.shape == (2, 4, 16)

    def test_rmsnorm(self):
        norm = RMSNorm(8)
        x = Tensor.randn((2, 4, 8))
        out = norm.forward(x)
        assert out.shape == (2, 4, 8)
        assert np.all(np.isfinite(out.data))

    def test_swiglu(self):
        layer = SwiGLU(8, 32)
        x = Tensor.randn((2, 4, 8))
        out = layer.forward(x)
        assert out.shape == (2, 4, 8)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestModel:
    def test_config_tiny(self):
        cfg = Config.tiny()
        assert cfg.hidden_dim == 64
        assert cfg.n_layers == 2
        assert cfg.n_heads == 4
        assert cfg.vocab_size == 1000

    def test_model_creation(self):
        model = MoETransformer.tiny()
        assert model is not None

    def test_forward(self):
        model = MoETransformer.tiny()
        x = Tensor.from_numpy(np.array([[1, 2, 3, 4]]))
        out = model.forward(x)
        assert out.shape == (1, 4, 1000)

    def test_generation(self):
        model = MoETransformer.tiny()
        tokens = model.generate([1, 2, 3], 6)
        assert isinstance(tokens, list)
        assert len(tokens) == 6

    def test_block_shape(self):
        cfg = Config.tiny()
        block = TransformerBlock(cfg)
        x = Tensor.randn((1, 4, 64))
        out = block.forward(x)
        assert out.shape == (1, 4, 64)

    def test_forward_backward(self):
        model = MoETransformer.tiny()
        x = Tensor.from_numpy(np.array([[1, 2, 3, 4]]))
        logits = model.forward(x)
        grad = model.backward(logits)
        assert grad is not None


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class TestMoE:
    def test_router(self):
        router = Router(64, 4, 2)
        x = Tensor.randn((1, 4, 64))
        weights, indices = router.forward(x)
        assert weights.shape == (4, 2)
        assert len(indices) == 4

    def test_moe_layer(self):
        cfg = Config.tiny()
        layer = MoELayer(cfg)
        x = Tensor.randn((1, 4, 64))
        out = layer.forward(x)
        assert out.shape == (1, 4, 64)

    def test_aux_loss(self):
        router = Router(64, 4, 2)
        x = Tensor.randn((1, 4, 64))
        router.forward(x)
        loss = router.compute_aux_loss()
        assert loss >= 0


# ---------------------------------------------------------------------------
# Generate / SamplingStrategy
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_with_greedy_strategy(self):
        model = MoETransformer.tiny()
        tokens = generate(model, [1, 2, 3], 6, GreedySampling())
        assert isinstance(tokens, list)
        assert len(tokens) == 6

    def test_generate_greedy_compat(self):
        model = MoETransformer.tiny()
        tokens = generate_greedy(model, [1, 2, 3], 6)
        assert len(tokens) == 6

    def test_greedy_matches_model_generate(self):
        model = MoETransformer.tiny()
        a = model.generate([1, 2, 3], 6)
        b = generate(model, [1, 2, 3], 6, GreedySampling())
        assert a == b


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrain:
    def test_cross_entropy(self):
        model = MoETransformer.tiny()
        config = TrainConfig.default()
        trainer = Trainer(model, config)

        x = Tensor.from_numpy(np.array([[1, 2, 3, 4]]))
        logits = model.forward(x)
        targets = Tensor.from_numpy(np.array([[2, 3, 4, 5]]))
        loss, grad = trainer._compute_loss(logits, targets)
        assert np.isfinite(loss)
        assert grad.shape == logits.shape

    def test_training_pipeline(self):
        model = MoETransformer.tiny()
        config = TrainConfig.default()
        trainer = Trainer(model, config)

        input_ids = Tensor.from_numpy(np.array([[1, 2, 3, 4]]))
        targets = Tensor.from_numpy(np.array([[2, 3, 4, 5]]))
        loss = trainer.train_step(input_ids, targets)
        assert np.isfinite(loss)

    def test_multi_step(self):
        model = MoETransformer.tiny()
        config = TrainConfig.default()
        trainer = Trainer(model, config)

        input_ids = Tensor.from_numpy(np.array([[1, 2, 3, 4]]))
        targets = Tensor.from_numpy(np.array([[2, 3, 4, 5]]))

        losses = []
        for _ in range(3):
            loss = trainer.train_step(input_ids, targets)
            losses.append(loss)

        assert all(np.isfinite(l) for l in losses)

    def test_lr_schedule(self):
        config = TrainConfig(lr=1e-3, warmup_steps=100, total_steps=1000)
        model = MoETransformer.tiny()
        trainer = Trainer(model, config)

        # step 0 => lr == 0
        trainer.step = 0
        assert trainer.get_lr() == 0.0

        # at warmup_steps => lr == config.lr
        trainer.step = config.warmup_steps
        np.testing.assert_allclose(trainer.get_lr(), config.lr, rtol=1e-6)

        # well past total_steps => lr >= config.lr * 0.1
        trainer.step = config.total_steps * 10
        assert trainer.get_lr() >= config.lr * 0.1

    def test_grad_clip(self):
        big = Tensor.from_numpy(np.ones((10,)) * 100)
        clipped = clip_grad_by_global_norm(big, 1.0)
        norm = float(np.sqrt(np.sum(clipped.data ** 2)))
        assert norm <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_storage(self):
        store = CheckpointStorage()
        t = Tensor.ones((4,))
        store.save(0, t)
        assert store.get(0) is not None
        assert len(store) == 1
        store.clear()
        assert len(store) == 0

    def test_context(self):
        ctx = CheckpointContext(segment_size=2)
        assert ctx.should_checkpoint(0) is True
        assert ctx.should_checkpoint(1) is False
        assert ctx.should_checkpoint(2) is True
