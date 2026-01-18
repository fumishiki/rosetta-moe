# cuda-nn Documentation

## Overview

MoE Transformer (6.9B total / 1.8B active) multi-language implementation.
Full-scratch implementation in Rust + Go + Python + CUDA.

---

## Document List

| Document | Content |
|----------|---------|
| [1-model.md](1-model.md) | Model Architecture Design |
| [2-learn.md](2-learn.md) | Training System Design |

---

## Project Structure

```
machine_learning/
├── rust/               # Rust implementation
│   ├── nn-core/        # Model, tensor, training
│   └── nn-ffi/         # CUDA FFI bridge
├── go/                 # Go implementation
│   ├── tensor/         # Tensor operations
│   ├── cuda/           # cgo CUDA bindings
│   ├── layer/          # NN layers
│   ├── model/          # MoE model
│   └── train/          # Training pipeline
├── python/             # Python implementation
│   ├── nn/             # NN modules
│   ├── cuda/           # ctypes CUDA bindings
│   └── tests/          # pytest tests
├── cuda/               # Shared CUDA kernels (9 files)
│   ├── kernels/        # .cu kernel files
│   └── src/            # stub.c (CPU fallback)
├── docs-jp/            # Japanese documentation
└── docs-en/            # English documentation
```

---

## Implementation Language Comparison

| Item | Rust | Go | Python |
|------|------|-----|--------|
| Tensor | Custom type + Error type | Custom type | numpy backend |
| CUDA bindings | FFI (build.rs) | cgo (Makefile) | ctypes |
| CPU fallback | stub.c | stub.c | numpy |
| Test count | 53 | 31 | 42 |
| Advanced optimization | CUDA Graph, etc. | - | - |

---

## Quick Start

### Rust

```bash
cargo build --release
cargo test
```

### Go

```bash
cd go
go test ./...
```

### Python

```bash
cd python
pip install -e ".[dev]"
pytest
```

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Total parameters | ~6.9B |
| Active parameters | ~1.8B |
| Hidden dim | 768 |
| Layers | 30 |
| Attention | MQA (12Q/1KV) |
| Experts | 16 total, top-4 active |
| FFN dim | 6144 |
| Vocab size | 32,000 |
| Context | 32K train → 256K inference (NTK RoPE) |

---

## Main Components

### Model Layers

- **Embedding**: Token embedding (32K × 768)
- **RMSNorm**: Root Mean Square normalization
- **MQA Attention**: Multi-Query Attention (12Q/1KV)
- **MoE Layer**: Router + 16 Experts (top-4 selection)
- **SwiGLU FFN**: Gated Linear Unit (768 → 6144 → 768)
- **LM Head**: Output projection (768 → 32K)

### CUDA Kernels

| File | Kernels |
|------|---------|
| elementwise.cu | silu, add, mul, scale |
| softmax.cu | softmax, softmax_topk |
| rmsnorm.cu | rmsnorm, rmsnorm_residual |
| gemm.cu | gemm, gemm_batched |
| rope.cu | rope_freqs, rope_forward |
| attention.cu | attention_scores, flash_attention |
| loss.cu | cross_entropy, aux_loss |
| optimizer.cu | adamw_step, grad_clip, scatter_add |
| decode.cu | argmax, sample, topk_sample, topp_sample |

### Training Features

- **Loss**: CrossEntropy + MoE AuxLoss (load balancing)
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **LR Schedule**: Warmup + Cosine Decay
- **Decode**: Greedy, Sample, Top-K, Top-P

---

## Test Status

| Language | Test Count | Status |
|----------|------------|--------|
| Rust | 53 | ✅ |
| Go | 31 | ✅ |
| Python | 42 | ✅ |
| **Total** | **126** | ✅ |

---

## License

MIT OR Apache-2.0
