"""Tests for model module."""

import numpy as np
import pytest

from nn.model import (
    Config,
    MoETransformer,
    MQAttention,
    Router,
    MoELayer,
    TransformerBlock,
)
from nn.tensor import Tensor


class TestConfig:
    """Tests for Config."""

    def test_default_config(self):
        cfg = Config.default_6_9b()
        assert cfg.hidden_dim == 768
        assert cfg.n_layers == 30
        assert cfg.n_heads == 12
        assert cfg.n_kv_heads == 1
        assert cfg.n_experts == 16
        assert cfg.top_k_experts == 4

    def test_tiny_config(self):
        cfg = Config.tiny()
        assert cfg.hidden_dim == 64
        assert cfg.n_layers == 2
        assert cfg.n_heads == 4
        assert cfg.n_experts == 4
        assert cfg.top_k_experts == 2

    def test_params_estimation(self):
        cfg = Config.tiny()
        total = cfg.total_params()
        active = cfg.active_params()
        assert total > 0
        assert active > 0
        assert active <= total


class TestRouter:
    """Tests for Router."""

    def test_router_forward(self):
        router = Router(hidden_dim=64, n_experts=4, top_k=2)
        x = Tensor.randn((1, 4, 64))
        weights, indices = router.forward(x)

        assert weights.shape == (4, 2)  # [num_tokens, top_k]
        assert len(indices) == 4
        assert all(len(idx) == 2 for idx in indices)

        # Weights should sum to ~1 per token
        for t in range(4):
            assert abs(np.sum(weights[t]) - 1.0) < 1e-5

    def test_router_backward(self):
        router = Router(hidden_dim=64, n_experts=4, top_k=2)
        x = Tensor.randn((1, 4, 64))
        router.forward(x)

        grad_out = Tensor.randn((1, 4, 64))
        grad_in = router.backward(grad_out)
        assert grad_in.shape == (1, 4, 64)

    def test_router_aux_loss(self):
        router = Router(hidden_dim=64, n_experts=4, top_k=2)
        x = Tensor.randn((1, 4, 64))
        router.forward(x)

        aux_loss = router.compute_aux_loss(alpha=0.01)
        assert aux_loss >= 0


class TestMoELayer:
    """Tests for MoELayer."""

    def test_moe_forward(self):
        cfg = Config.tiny()
        moe = MoELayer(cfg)
        x = Tensor.randn((1, 4, cfg.hidden_dim))
        output = moe.forward(x)
        assert output.shape == (1, 4, cfg.hidden_dim)

    def test_moe_backward(self):
        cfg = Config.tiny()
        moe = MoELayer(cfg)
        x = Tensor.randn((1, 4, cfg.hidden_dim))
        moe.forward(x)

        grad_out = Tensor.randn((1, 4, cfg.hidden_dim))
        grad_in = moe.backward(grad_out)
        assert grad_in.shape == (1, 4, cfg.hidden_dim)


class TestMQAttention:
    """Tests for MQAttention."""

    def test_attention_forward(self):
        cfg = Config.tiny()
        attn = MQAttention(cfg)
        x = Tensor.randn((1, 4, cfg.hidden_dim))
        output = attn.forward(x)
        assert output.shape == (1, 4, cfg.hidden_dim)

    def test_attention_backward(self):
        cfg = Config.tiny()
        attn = MQAttention(cfg)
        x = Tensor.randn((1, 4, cfg.hidden_dim))
        attn.forward(x)

        grad_out = Tensor.randn((1, 4, cfg.hidden_dim))
        grad_in = attn.backward(grad_out)
        assert grad_in.shape == (1, 4, cfg.hidden_dim)


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_block_forward(self):
        cfg = Config.tiny()
        block = TransformerBlock(cfg)
        x = Tensor.randn((1, 4, cfg.hidden_dim))
        output = block.forward(x)
        assert output.shape == (1, 4, cfg.hidden_dim)

    def test_block_backward(self):
        cfg = Config.tiny()
        block = TransformerBlock(cfg)
        x = Tensor.randn((1, 4, cfg.hidden_dim))
        block.forward(x)

        grad_out = Tensor.randn((1, 4, cfg.hidden_dim))
        grad_in = block.backward(grad_out)
        assert grad_in.shape == (1, 4, cfg.hidden_dim)


class TestMoETransformer:
    """Tests for MoETransformer."""

    def test_model_creation(self):
        model = MoETransformer.tiny()
        params = model.parameters()
        assert len(params) > 0

    def test_model_forward(self):
        model = MoETransformer.tiny()
        token_ids = [1, 2, 3, 4]
        logits = model.forward_ids(token_ids, batch=1, seq_len=4)
        assert logits.shape == (1, 4, model.config.vocab_size)

    def test_model_backward(self):
        model = MoETransformer.tiny()
        token_ids = [1, 2, 3, 4]
        logits = model.forward_ids(token_ids, batch=1, seq_len=4)

        grad_out = Tensor.randn(logits.shape)
        grad_in = model.backward(grad_out)
        # Backward returns gradient through blocks
        assert grad_in.shape[0] == 1

    def test_model_aux_loss(self):
        model = MoETransformer.tiny()
        token_ids = [1, 2, 3, 4]
        model.forward_ids(token_ids, batch=1, seq_len=4)

        aux_loss = model.total_aux_loss()
        assert aux_loss >= 0
