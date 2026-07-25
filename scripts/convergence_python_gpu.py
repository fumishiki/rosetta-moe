#!/usr/bin/env python3
"""GPU loss convergence verification for Python MoE Transformer."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.config import Config
from python.gpu import MetalContext, MetalTrainer, metal_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Python GPU MoE convergence test")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="topk",
        choices=["topk", "biasfree", "relu"],
        help="Routing mode placeholder (kept for CLI compatibility)",
    )
    args = parser.parse_args()

    if not metal_available():
        raise RuntimeError("Metal is not available on this system")

    np.random.seed(args.seed)

    cfg = Config.tiny()
    ctx = MetalContext()
    trainer = MetalTrainer.create(ctx, cfg)

    batch, seq = 2, 8
    input_data = np.array([i % cfg.vocab_size for i in range(batch * seq)], dtype=np.float32).reshape(batch, seq)
    target_data = np.array([(i + 1) % cfg.vocab_size for i in range(batch * seq)], dtype=np.float32).reshape(batch, seq)

    n_steps = 500
    losses: list[float] = []
    for _ in range(n_steps):
        loss = trainer.train_step(input_data, target_data, readback=True)
        losses.append(float(loss))

    print(
        json.dumps(
            {
                "language": "python",
                "backend": "gpu_metal",
                "steps": n_steps,
                "losses": [round(l, 6) for l in losses],
            }
        )
    )


if __name__ == "__main__":
    main()
