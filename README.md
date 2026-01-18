# cuda-nn

[![CI](https://github.com/pikafumi/machine_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/pikafumi/machine_learning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange)
![Go](https://img.shields.io/badge/Go-1.22-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)

**MoE Transformer (6.9B / 1.8B active) — Rust + Go + Python + CUDA from scratch**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE Transformer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Embedding  │→ │ Blocks × 30 │→ │  LM Head    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Transformer Block                        │  │
│  │  ┌─────────┐    ┌───────────┐    ┌─────────┐         │  │
│  │  │ RMSNorm │ →  │    MQA    │ →  │ RMSNorm │ → MoE   │  │
│  │  └─────────┘    │ (12Q/1KV) │    └─────────┘         │  │
│  │       ↑         └───────────┘         ↑     ↓        │  │
│  │       └──────── residual ─────────────┴─ residual ───│  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   MoE Layer                           │  │
│  │  ┌────────┐   ┌─────────────────────────────────┐    │  │
│  │  │ Router │ → │ Expert 0  │ ... │  Expert 15   │    │  │
│  │  │ top-4  │   │ (SwiGLU)  │     │  (SwiGLU)    │    │  │
│  │  └────────┘   └─────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Model Specs

| Parameter | Value |
|-----------|-------|
| Hidden dim | 768 |
| Layers | 30 |
| Attention | MQA (12 Q heads, 1 KV head) |
| Experts | 16 total, top-4 active |
| FFN dim | 6144 |
| Vocab size | 32,000 |
| Context | 32K train → 256K inference (NTK) |
| **Total params** | **~6.9B** |
| **Active params** | **~1.8B** |

## Project Structure

```
machine_learning/
├── crates/
│   ├── cuda/         # Shared CUDA kernels (9 files)
│   ├── rust/         # Rust implementation
│   │   ├── nn-core/  # Model, tensor ops, training
│   │   └── nn-ffi/   # FFI bridge + GpuTrainer
│   ├── go/           # Go implementation
│   │   ├── tensor/   # Tensor operations
│   │   ├── cuda/     # cgo CUDA bindings
│   │   ├── layer/    # Neural network layers
│   │   ├── model/    # MoE Transformer model
│   │   └── train/    # Training pipeline
│   └── python/       # Python implementation
│       ├── nn/       # Neural network module
│       ├── cuda/     # ctypes CUDA bindings
│       └── tests/    # pytest tests
├── docs/             # Design documents
└── Cargo.toml        # Rust workspace
```

## Language Implementations

### Rust (crates/rust/)

Pure Rust implementation with `#![forbid(unsafe_code)]` in nn-core.

- **nn-core**: Tensor ops, layers, attention, MoE, training
- **nn-ffi**: FFI bridge, GpuTensor, GpuTrainer, CUDA Graph

### Go (crates/go/)

Go implementation with cgo bindings to shared CUDA kernels.

- **tensor**: Shape, DType, Tensor operations
- **layer**: Embedding, RMSNorm, Linear, SwiGLU
- **model**: MQAttention, Router, MoELayer, TransformerBlock, MoETransformer
- **train**: AdamW optimizer, LR scheduler, training loop

### Python (crates/python/)

Python implementation with numpy backend and ctypes CUDA bindings.

- **nn.tensor**: Tensor operations (numpy backend)
- **nn.layers**: Embedding, RMSNorm, Linear, SwiGLU
- **nn.model**: MQAttention, Router, MoELayer, TransformerBlock, MoETransformer
- **nn.train**: AdamW optimizer, LR scheduler, training loop
- **cuda**: ctypes bindings with CPU fallback

### CUDA Kernels (crates/cuda/)

Shared CUDA kernels used by both Rust and Go.

| File | Kernels |
|------|---------|
| elementwise.cu | silu, add, mul, scale |
| softmax.cu | softmax, top-k |
| rmsnorm.cu | rmsnorm, fused residual |
| gemm.cu | tiled GEMM (32×32), batched |
| rope.cu | NTK RoPE frequencies |
| attention.cu | MQA, FlashAttention-style |
| loss.cu | CrossEntropy, AuxLoss |
| optimizer.cu | AdamW, grad clip, scatter_add |
| decode.cu | argmax, sample, top-k, top-p |

**GPU Support**: sm_70 (V100), sm_75 (Turing), sm_80 (A100), sm_86 (Ampere), sm_89 (Ada), sm_90 (Hopper)

## Quick Start

### Rust

```bash
# Build
cargo build --release

# Test (53 tests)
cargo test

# Clippy
cargo clippy --all-targets
```

### Go

```bash
cd crates/go

# Test
go test ./...

# Build (requires CUDA library)
go build ./...
```

### Python

```bash
cd crates/python

# Install
pip install -e ".[dev]"

# Test (42 tests)
pytest
```

### CUDA (Optional)

CUDA is auto-detected. Without CUDA toolkit, CPU stubs are linked.

```bash
# Force CPU-only (Rust)
CUDA_PATH="" cargo build --release
```

## Usage

### Rust

```rust
// Create tiny model for testing
let model = MoETransformer::tiny();

// Forward pass
let token_ids = vec![1, 2, 3, 4];
let logits = model.forward_ids(&token_ids, 1, 4);
// → [1, 4, vocab_size] logits
```

### Go

```go
// Create tiny model for testing
model := model.NewTiny()

// Forward pass
tokenIDs := []int{1, 2, 3, 4}
logits := model.ForwardIDs(tokenIDs, 1, 4)
// → [1, 4, vocab_size] logits
```

### Python

```python
from nn import MoETransformer

# Create tiny model for testing
model = MoETransformer.tiny()

# Forward pass
token_ids = [1, 2, 3, 4]
logits = model.forward_ids(token_ids, batch=1, seq_len=4)
# → [1, 4, vocab_size] logits
```

## Implementation Status

| Component | Rust | Go | Python |
|-----------|------|-----|--------|
| Tensor ops | ✅ | ✅ | ✅ |
| Embedding | ✅ | ✅ | ✅ |
| RMSNorm | ✅ | ✅ | ✅ |
| Linear | ✅ | ✅ | ✅ |
| MQA Attention | ✅ | ✅ | ✅ |
| MoE Router | ✅ | ✅ | ✅ |
| Expert FFN | ✅ | ✅ | ✅ |
| Full model forward | ✅ | ✅ | ✅ |
| CUDA bindings | ✅ FFI | ✅ cgo | ✅ ctypes |
| Training loop | ✅ | ✅ | ✅ |
| GPU decode | ✅ | ✅ | ✅ |
| GpuTrainer | ✅ | - | - |

## Design Principles

- **Type safety**: `#![forbid(unsafe_code)]` in Rust nn-core
- **Shared CUDA**: Single CUDA kernel source for all languages
- **Multi-language**: Rust (FFI) + Go (cgo) + Python (ctypes)
- **Manual autograd**: Educational, full control
- **MQA**: Memory efficient (1 KV head)
- **NTK RoPE**: 32K→256K extrapolation
- **GPU-resident**: Minimal CPU↔GPU transfer (loss only)

## Docs

- [docs/00-index.md](docs/00-index.md) - Documentation index
- [docs/1-model.md](docs/1-model.md) - Model architecture
- [docs/2-learn.md](docs/2-learn.md) - Training system

## License

[MIT](LICENSE)
