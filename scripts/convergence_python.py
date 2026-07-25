#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Loss convergence verification for Python MoE Transformer."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.config import Config
from python.tensor import Tensor, seed_rng
from python.model import MoETransformer
from python.train import Trainer, TrainConfig, RoutingMode

def main():
    parser = argparse.ArgumentParser(description="Python MoE convergence test")
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="topk",
        choices=["topk", "biasfree", "relu"],
        help="Routing mode: topk, biasfree, or relu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)"
    )
    args = parser.parse_args()

    # Parse routing mode
    routing_mode = RoutingMode(args.routing_mode.lower())

    np.random.seed(args.seed)
    seed_rng(args.seed)
    model = MoETransformer.tiny()
    model.set_routing_mode(routing_mode)

    cfg = TrainConfig(
        lr=1e-3,
        warmup_steps=50,
        total_steps=600,
        grad_clip=0.5,
        routing_mode=routing_mode,
    )
    trainer = Trainer(model, cfg)

    batch, seq = 2, 8
    input_data = np.array([i % 1000 for i in range(batch * seq)], dtype=np.float32).reshape(batch, seq)
    target_data = np.array([(i + 1) % 1000 for i in range(batch * seq)], dtype=np.float32).reshape(batch, seq)
    input_ids = Tensor.from_numpy(input_data)
    targets = Tensor.from_numpy(target_data)

    n_steps = 500
    losses = []
    for _ in range(n_steps):
        loss = trainer.train_step(input_ids, targets)
        losses.append(float(loss))

    print(json.dumps({
        "language": "python",
        "routing_mode": args.routing_mode,
        "steps": n_steps,
        "losses": [round(l, 6) for l in losses],
    }))

if __name__ == "__main__":
    main()
