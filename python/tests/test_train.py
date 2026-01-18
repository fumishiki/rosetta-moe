"""Tests for training module."""

import numpy as np
import pytest

from nn.model import Config, MoETransformer
from nn.train import TrainConfig, Trainer
from nn.tensor import Tensor


class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_config(self):
        cfg = TrainConfig.default()
        assert cfg.lr == 1e-4
        assert cfg.beta1 == 0.9
        assert cfg.beta2 == 0.95
        assert cfg.warmup_steps == 2000


class TestTrainer:
    """Tests for Trainer."""

    def test_trainer_creation(self):
        model = MoETransformer.tiny()
        cfg = TrainConfig.default()
        trainer = Trainer(model, cfg)
        assert trainer.step == 0

    def test_lr_schedule_warmup(self):
        model = MoETransformer.tiny()
        cfg = TrainConfig(warmup_steps=100, total_steps=1000)
        trainer = Trainer(model, cfg)

        # At step 0, LR should be 0
        assert trainer.get_lr() == 0

        # At step 50, should be halfway through warmup
        trainer.step = 50
        assert abs(trainer.get_lr() - cfg.lr * 0.5) < 1e-8

        # At step 100, should be at max LR
        trainer.step = 100
        assert abs(trainer.get_lr() - cfg.lr) < 1e-8

    def test_lr_schedule_decay(self):
        model = MoETransformer.tiny()
        cfg = TrainConfig(warmup_steps=100, total_steps=1000)
        trainer = Trainer(model, cfg)

        # LR should decrease after warmup
        trainer.step = 100
        lr_at_warmup = trainer.get_lr()

        trainer.step = 500
        lr_mid = trainer.get_lr()

        assert lr_mid < lr_at_warmup

    def test_train_step(self):
        model = MoETransformer.tiny()
        cfg = TrainConfig.default()
        trainer = Trainer(model, cfg)

        # Create input
        batch, seq_len = 2, 8
        input_ids = Tensor.from_numpy(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int64)
        )
        targets = Tensor.from_numpy(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int64)
        )

        loss = trainer.train_step(input_ids, targets)
        assert loss >= 0
        assert trainer.step == 1

    def test_multiple_train_steps(self):
        model = MoETransformer.tiny()
        cfg = TrainConfig.default()
        trainer = Trainer(model, cfg)

        batch, seq_len = 1, 4
        input_ids = Tensor.from_numpy(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int64)
        )
        targets = Tensor.from_numpy(
            np.random.randint(0, 100, (batch, seq_len)).astype(np.int64)
        )

        losses = []
        for _ in range(5):
            loss = trainer.train_step(input_ids, targets)
            losses.append(loss)

        assert trainer.step == 5
        assert all(l >= 0 for l in losses)
