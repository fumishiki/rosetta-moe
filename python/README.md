<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# Python MoE Transformer

Pure Python + NumPy implementation of a Mixture-of-Experts Transformer. No PyTorch, no JAX -- every operation and every gradient is hand-implemented using NumPy as the sole numerical backend. The benchmark uses a tiny model configuration for cross-language comparison; the educational value is in the explicit math, not the scale.

## Architecture Overview

```text
python/
├── cpu/             # CPU implementation package (NumPy / Accelerate BLAS)
│   ├── tensor.py
│   ├── config.py
│   ├── layers.py
│   ├── attention.py
│   ├── moe.py
│   ├── model.py
│   ├── generate.py
│   └── train.py
├── gpu/             # GPU implementation package (Metal / MPS via PyObjC)
│   ├── metal_tensor.py
│   ├── metal_layers.py
│   ├── metal_attention.py
│   ├── metal_moe.py
│   ├── metal_model.py
│   └── metal_train.py
├── bench_cpu.py     # CPU benchmark harness (axes 1-5); JSON output to stdout
├── bench_gpu.py     # GPU benchmark harness (axis 6, Metal/MPS); JSON output to stdout
├── bench.py         # compatibility dispatcher (ROSETTA_CPU_ONLY / ROSETTA_GPU_ONLY)
├── metal_backend.py # low-level Metal/MPS bridge
├── __init__.py      # package facade (backward-compatible re-exports)
└── tests/
    └── test_integration.py
```

Bench paths import `python.cpu` / `python.gpu` only. Top-level modules (`python/tensor.py` etc.) remain as backward-compatible facades. `bench.py` is kept as a compatibility dispatcher (`ROSETTA_GPU_ONLY=1` for GPU, default CPU).

### Forward pipeline

```
token_ids [batch, seq]
  -> Embedding lookup         [batch, seq, hidden_dim]
  -> N x TransformerBlock:
       RMSNorm -> MQ Attention (RoPE) -> residual
       RMSNorm -> MoE (Router + top-k SwiGLU experts) -> residual
  -> Final RMSNorm
  -> Linear (LM head)         [batch, seq, vocab_size]
```

## Equation-to-Code Map

| Formula | File | Function/Method |
|---------|------|-----------------|
| `Embedding: out[i] = W[token_id[i]]` | `cpu/layers.py` | `Embedding.forward` |
| `RMSNorm: y = x * (1/sqrt(mean(x^2) + eps)) * gamma` | `cpu/layers.py` | `RMSNorm.forward` |
| `Linear: y = x @ W^T + b` | `cpu/layers.py` | `Linear.forward` |
| `SiLU: y = x * sigmoid(x)` | `cpu/tensor.py` | `Tensor.silu` |
| `SwiGLU: y = down(silu(gate(x)) * up(x))` | `cpu/layers.py` | `SwiGLU.forward` |
| `Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))` | `cpu/tensor.py` | `Tensor.softmax` |
| `RoPE: [x0,x1] -> [x0*cos(t) - x1*sin(t), x0*sin(t) + x1*cos(t)]` | `cpu/attention.py` | `MQAttention._apply_rope` |
| `RoPE freq: f_i = 1 / (base^(2i/d))` | `cpu/attention.py` | `MQAttention._compute_rope_freqs` |
| `NTK scaling: base' = base * alpha^(d/(d-2))` | `cpu/attention.py` | `MQAttention._compute_rope_freqs` |
| `Attention: scores = Q@K^T/sqrt(d_k), weights = softmax(scores+mask), out = weights@V` | `cpu/attention.py` | `MQAttention.forward` |
| `MoE: output = sum_k(gate_k * Expert_k(x))` | `cpu/moe.py` | `MoELayer.forward` |
| `Router: probs = softmax(x@W), top_k -> renormalize` | `cpu/moe.py` | `Router.forward` |
| `Aux loss: L = alpha * N * sum(f_i * P_i)` | `cpu/moe.py` | `Router.compute_aux_loss` |
| `CrossEntropy: L = -mean(log(softmax(logits)[target]))` | `cpu/train.py` | `Trainer._compute_loss` |
| `CE gradient: d_logits = softmax(logits) - one_hot(targets)` | `cpu/train.py` | `Trainer._compute_loss` |
| `AdamW: m=b1*m+(1-b1)*g, v=b2*v+(1-b2)*g^2, w-=lr*(m_hat/(sqrt(v_hat)+eps)+wd*w)` | `cpu/train.py` | `Trainer._adamw_step` |
| `Grad clip: if \|\|g\|\|>c then g' = g*c/(\|\|g\|\|+eps)` | `cpu/train.py` | `clip_grad_by_global_norm` |
| `LR warmup: lr = base_lr * step / warmup_steps` | `cpu/train.py` | `Trainer.get_lr` |
| `LR cosine: lr = min_lr + 0.5*(base_lr-min_lr)*(1+cos(pi*progress))` | `cpu/train.py` | `Trainer.get_lr` |
| `Temperature: p = softmax(logits / T)` | `cpu/generate.py` | `_sample_from_logits` |
| `Top-p: accumulate sorted probs until sum >= p, renormalize` | `cpu/generate.py` | `_sample_top_p` |

## Implementation Notes

### Pure Python + NumPy (no framework)

Every forward and backward operation is implemented from scratch. There is no autograd tape -- all gradients are derived by hand and coded explicitly. This makes the implementation fully transparent and educational, at the cost of not scaling to production training.

### np.matmul -> Apple Accelerate (BLAS/AMX)

On macOS >= 14, NumPy links against Apple's Accelerate framework. All `np.matmul` calls (used in Linear, Attention, and MoE layers) automatically dispatch to hardware-accelerated `sgemm` backed by AMX units. No explicit FFI or library loading is required -- it happens transparently through NumPy's BLAS backend.

### Vectorized MoE Dispatch

The MoE layer (`python/cpu/moe.py:MoELayer.forward`) uses three key NumPy patterns:

- **`np.argpartition`** for O(N) top-k selection (vs O(N log N) full sort)
- **Boolean mask broadcasting** (`indices_arr == expert_idx`) to find all tokens assigned to each expert in one vectorized operation
- **`np.add.at`** for scatter-add accumulation -- unlike `output[indices] += values`, `np.add.at` is unbuffered and correctly handles duplicate indices

### tracemalloc Limitations

`tracemalloc` (used in `bench_cpu.py`) only tracks allocations made through Python's memory allocator (`PyMem_Malloc` / `PyObject_Malloc`). NumPy array data is allocated via libc `malloc`, which tracemalloc does NOT see. The `alloc_bytes` metric in benchmarks significantly undercounts actual memory usage. Use `peak_rss_bytes` (from `resource.getrusage`) for a more accurate picture.

### ProcessPoolExecutor for Parallelism

Python's GIL prevents true thread parallelism for CPU-bound code. The CPU benchmark (`bench_cpu.py`) uses `ProcessPoolExecutor` to fork separate processes, each with its own GIL and its own model.

The pool is created once with an `initializer=` callback that builds the model and input tensor in each worker process at startup. Warmup and trial loops only measure dispatch + forward pass time, not process creation overhead. The ~113ms fork/import cost is paid once at pool creation, not per trial.

This matters: the previous version created and destroyed the pool per trial, measuring 26 inf/s at T4. With the pool pre-created, T4 throughput is 1,968 inf/s -- tied with Go's goroutine-based parallelism.

## Performance Characteristics

Latest benchmark results (M1, batch=2, seq=32, hidden=64):

| Metric | Value | Rank (of 4 languages) |
|--------|-------|------|
| Forward pass | 2.53ms | 4th (Rust 0.56ms, Julia 0.59ms, Go 1.14ms) |
| Train step | 10.22ms | 4th |
| Matmul 64x64 | 164 GFLOPS | 4th |
| Softmax kernel | 16.17us | 4th (Rust 1.83us) |
| RMSNorm kernel | 23.38us | 4th (Rust 3.38us) |
| Peak RSS | 61MB | 3rd |
| T4 throughput | 1,859 inf/s | 4th |
| T4 train | 503 trn/s | 4th |

**NumPy closes the BLAS gap**: `np.matmul` dispatches to Accelerate with zero developer effort. No FFI wrapper needed.

**CPython interpreter is the bottleneck**: softmax (16.17us) is 8.8x Rust (1.83us), rmsnorm (23.38us) is 6.9x. Every non-BLAS operation pays the interpreter dispatch tax: bytecode decode, dynamic type check, reference counting. In-place optimizations help 10-15% but cannot overcome this fundamental overhead. At h=64, **90% of every training step** is interpreter overhead (see root README for production-scale cost analysis).

**~2ms fixed overhead**: At small inputs (batch=1, seq=8), Python takes ~1.9ms while Rust takes 0.16ms. This interpreter overhead does not scale with workload size.

**Python's real value is not in this benchmark**: The same model prototype here can be validated with NumPy, then deployed with PyTorch/JAX on GPU — where torch.compile and CUDA Graphs bypass the interpreter entirely. See root README for why these are "escape hatches from Python."

## Gotchas / Pitfalls

### NumPy vs Python loop performance

Operations like softmax and RMSNorm are implemented as NumPy vectorized calls (fast). However, some paths -- notably `_sample_from_probs` in `generate.py` -- use Python `for` loops over array elements, which are orders of magnitude slower than C-level iterations. These are acceptable for single-token generation but would bottleneck batch inference.

### tracemalloc does not track NumPy data buffers

As noted above, `tracemalloc.take_snapshot()` captures Python heap only. A model with 1M float32 parameters uses ~4MB of NumPy array storage that is completely invisible to tracemalloc. Always cross-reference with RSS for memory profiling.

### Tensor.__init__ copy behavior

`Tensor.__init__` skips the `astype` copy when `data.dtype` already matches the target dtype. This is a deliberate optimization: most internal code produces arrays in the correct dtype, so the branch avoids a redundant allocation on every layer output. However, it means the Tensor may share memory with the source array (a view, not a copy). Mutations to the original array would be visible through the Tensor.

### Backward pass: every gradient is manually derived

There is no autograd. Every layer (Embedding, Linear, RMSNorm, SwiGLU, Attention, MoE) implements a full `backward()` method that computes both `d_loss/d_input` and accumulates `d_loss/d_weight` into `param._grad`. The combined softmax + cross-entropy gradient (`probs - one_hot`) is propagated through every layer in reverse via chain rule. AdamW reads each parameter's accumulated gradient to update weights.

Key implementation details:
- `Tensor._grad`: per-parameter gradient buffer, accumulated during backward, consumed by AdamW
- `RMSNorm.backward()`: includes full Jacobian correction term (`- x * dot / (dim * rms^3)`), not just the simplified `g * gamma / rms`
- `Linear.backward()`: `dW = grad_output^T @ input` via `np.matmul`, `dX = grad_output @ W`
- `Embedding.backward()`: scatter-add grad_output into weight grad at token indices

### Kaiming initialization

All weights use Kaiming (He) normal initialization with `std = sqrt(2/fan_in)`, matching the Go and Julia implementations in this repository. This is important for numerical stability at initialization -- without it, activations can explode or vanish through deep networks.

### Causal mask caching

The attention layer caches the causal mask (`_cached_mask`) keyed by sequence length. When `seq_len` changes between calls (e.g., during autoregressive generation where the sequence grows by 1 each step), the mask is regenerated. For fixed-length training, the mask is computed once and reused.

## GPU (Metal/MPS)

Requires macOS with Apple Silicon and PyObjC Metal bindings:

```bash
pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders
```

Run GPU benchmarks:
```bash
python3 bench_gpu.py
```

`bench_gpu.py` is a GPU-only path: benchmark loops avoid CPU↔GPU data transfers and host tensor views.

The Metal backend (metal_backend.py) uses PyObjC to call Metal and MPS APIs.
If PyObjC/Metal is unavailable, `bench_gpu.py` exits; use `bench_cpu.py` for CPU runs.

## Dependencies

- Python 3.10+
- numpy >= 1.24.0
- pytest (for testing)

## Usage

```bash
# Install
cd python && pip install -e ".[dev]"

# Run tests
python3 -m pytest tests/ -v

# Run CPU benchmark
python3 bench_cpu.py > results_cpu.json

# Run GPU benchmark (Metal)
python3 bench_gpu.py > results_gpu.json

# Compatibility dispatcher (defaults to CPU)
ROSETTA_GPU_ONLY=1 python3 bench.py > results_gpu_via_dispatcher.json

# Override trials/warmup (defaults: 10/3)
ROSETTA_BENCH_TRIALS=30 ROSETTA_BENCH_WARMUP=3 python3 bench_cpu.py

# Project-root one-shot runs
cd .. && make bench-all
cd .. && make bench-all-30
```

```python
from python import MoETransformer, TrainConfig, Trainer

# Create tiny model for testing
model = MoETransformer.tiny()

# Forward pass
logits = model.forward_ids([1, 2, 3, 4], batch=1, seq_len=4)
print(f"Output shape: {logits.shape}")  # (1, 4, 1000)

# Generate tokens
tokens = model.generate([1, 2, 3], max_len=10)

# Training step
from python.tensor import Tensor
import numpy as np

trainer = Trainer(model, TrainConfig.default())
input_ids = Tensor.from_numpy(np.array([[1, 2, 3, 4]]))
targets = Tensor.from_numpy(np.array([[2, 3, 4, 5]]))
loss = trainer.train_step(input_ids, targets)
```
