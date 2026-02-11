#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Loss convergence verification for Python MoE Transformer."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.config import Config
from python.tensor import Tensor
from python.model import MoETransformer
from python.train import Trainer, TrainConfig

def main():
    np.random.seed(42)
    model = MoETransformer.tiny()
    cfg = TrainConfig(
        lr=1e-3,
        warmup_steps=10,
        total_steps=1200,
    )
    trainer = Trainer(model, cfg)

    batch, seq = 2, 8
    input_data = np.array([i % 1000 for i in range(batch * seq)], dtype=np.float32).reshape(batch, seq)
    target_data = np.array([(i + 1) % 1000 for i in range(batch * seq)], dtype=np.float32).reshape(batch, seq)
    input_ids = Tensor.from_numpy(input_data)
    targets = Tensor.from_numpy(target_data)

    n_steps = 1000
    losses = []
    for _ in range(n_steps):
        loss = trainer.train_step(input_ids, targets)
        losses.append(float(loss))

    print(json.dumps({
        "language": "python",
        "steps": n_steps,
        "losses": [round(l, 6) for l in losses],
    }))

if __name__ == "__main__":
    main()
