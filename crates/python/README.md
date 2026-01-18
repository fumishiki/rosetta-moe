# Python Implementation

MoE Transformer (6.9B/1.8B) の Python 実装。

## Structure

```
python/
├── nn/
│   ├── tensor.py    # Tensor operations (numpy backend)
│   ├── layers.py    # NN layers (Embedding, RMSNorm, Linear, SwiGLU)
│   ├── model.py     # Model (Attention, Router, MoE, Transformer)
│   └── train.py     # Training (Trainer, AdamW, LR scheduler)
├── cuda/
│   └── bindings.py  # ctypes CUDA bindings (with CPU fallback)
├── tests/           # pytest tests
└── pyproject.toml   # Package config
```

## Install

```bash
cd crates/python
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Usage

```python
from nn import MoETransformer, TrainConfig, Trainer

# Create tiny model for testing
model = MoETransformer.tiny()

# Forward pass
token_ids = [1, 2, 3, 4]
logits = model.forward_ids(token_ids, batch=1, seq_len=4)
print(f"Output shape: {logits.shape}")

# Training
config = TrainConfig.default()
trainer = Trainer(model, config)
# ... training loop
```

## CUDA Bindings

The `cuda` package provides Python bindings to shared CUDA kernels:

| Function | Description |
|----------|-------------|
| silu | SiLU activation |
| add/mul/scale | Element-wise ops |
| softmax | Softmax |
| rmsnorm | RMS normalization |
| gemm | Matrix multiplication |
| cross_entropy_forward | Loss computation |
| adamw_step | Optimizer step |
| argmax | Greedy decoding |
| sample | Multinomial sampling |
| topk_sample | Top-k sampling |
| topp_sample | Nucleus sampling |

All functions have CPU fallback when CUDA is not available.

## Dependencies

- Python 3.10+
- numpy >= 1.24.0
- pytest (for testing)
