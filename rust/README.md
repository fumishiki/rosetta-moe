<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# Rust MoE Transformer (`nn-core`)

Educational Mixture-of-Experts Transformer in Rust with explicit CPU/GPU separation. CPU kernels use Apple Accelerate BLAS, and GPU paths are provided via Metal. Part of a 4-language (Rust, Julia, Go, Python) benchmark suite comparing language characteristics for ML workloads.

## Architecture Overview

### File Structure

```
rust/
├── Cargo.toml              # Crate config: [lib] + [[bin]] (bench/convergence)
├── src/
│   ├── lib.rs              # Public API facade (sole export boundary)
│   ├── config.rs           # Model hyperparameter configs
│   ├── cpu/
│   │   ├── tensor.rs       # CPU tensor + matmul/softmax kernels
│   │   ├── layers.rs       # Embedding, RMSNorm, Linear, SwiGLU
│   │   ├── attention.rs    # Multi-Query Attention with RoPE
│   │   ├── moe.rs          # Router, MoELayer, TransformerBlock
│   │   ├── model.rs        # CPU MoETransformer
│   │   ├── generate.rs     # CPU sampling/generation
│   │   ├── train.rs        # CPU training (AdamW, CE, clipping)
│   │   ├── simd.rs         # NEON SIMD intrinsics
│   │   └── accelerate.rs   # Apple Accelerate BLAS FFI
│   └── gpu/
│       ├── metal_tensor.rs # Metal tensor/buffer wrappers
│       ├── metal_layers.rs # Metal kernels (softmax/rmsnorm/etc.)
│       ├── metal_model.rs  # GPU model forward path
│       └── metal_train.rs  # GPU train proxy/update path
└── benches/
    └── bench.rs            # Unified benchmark harness (CPU+GPU, up to 33 scenarios)
```

### Dependency DAG

```
lib.rs (facade)
  ├── cpu::model
  │     ├── cpu::generate
  │     ├── cpu::layers ──────── cpu::tensor, cpu::simd
  │     ├── cpu::moe
  │     │     ├── cpu::attention ── cpu::layers
  │     │     └── cpu::layers
  │     └── config
  ├── cpu::train ──── cpu::model, cpu::layers, cpu::tensor, cpu::simd
  └── gpu::{metal_tensor, metal_layers, metal_model, metal_train}

benches/bench.rs ── nn_core (uses lib.rs public API)
```

CPU data flow: `cpu::model -> cpu::moe -> cpu::attention -> cpu::layers -> cpu::tensor -> cpu::accelerate`. GPU data flow is isolated under `src/gpu/*`.
`benches/bench.rs` supports split execution with `ROSETTA_CPU_ONLY=1` / `ROSETTA_GPU_ONLY=1` (both set is an error).

## Equation-to-Code Map

| Math Formula | File | Function/Location |
|---|---|---|
| `Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))` | `src/cpu/tensor.rs` | `softmax_into_slice`, `softmax_in_place`, `Tensor::softmax_into` |
| `SiLU: silu(x) = x / (1 + exp(-x))` | `src/cpu/tensor.rs` | `Tensor::silu` |
| `Matmul: C = A @ B` (batched) | `src/cpu/tensor.rs` | `Tensor::try_matmul` -> `accelerate::sgemm` |
| `Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)` | `src/cpu/tensor.rs` | `Tensor::randn` |
| `Embedding: out = W[token_id, :]` | `src/cpu/layers.rs` | `Embedding::forward_with_ids` |
| `RMSNorm: y = x * (1/sqrt(mean(x^2) + eps)) * gamma` | `src/cpu/layers.rs` | `RMSNorm::forward` |
| `Linear: Y = X @ W^T` | `src/cpu/layers.rs` | `Linear::forward` -> `accelerate::sgemm_transb` |
| `Linear backward: dX = dY @ W` | `src/cpu/layers.rs` | `Linear::backward` -> `accelerate::sgemm` |
| `SwiGLU: out = Down(SiLU(Gate(x)) * Up(x))` | `src/cpu/layers.rs` | `SwiGLU::forward` |
| `RoPE: [x0,x1] -> [x0*cos(t) - x1*sin(t), x0*sin(t) + x1*cos(t)]` | `src/cpu/attention.rs` | `MQAttention::apply_rope` |
| `Attention: scores = Q@K^T/sqrt(d_k), out = softmax(scores+mask)@V` | `src/cpu/attention.rs` | `MQAttention::forward` |
| `MoE: output = sum_k(gate_k * Expert_k(x))` for top-k | `src/cpu/moe.rs` | `MoELayer::forward` |
| `Router: probs = softmax(x @ W), top-k select + renormalize` | `src/cpu/moe.rs` | `Router::route` |
| `Aux Loss: L = alpha * N * sum_e(f_e * p_e)` | `src/cpu/moe.rs` | `MoELayer::aux_loss`, `compute_aux_loss` |
| `Cross-entropy: L = -(1/N) * sum(log(softmax(logits)[target]))` | `src/cpu/train.rs` | `CrossEntropyLoss::forward` |
| `CE gradient: dL = (1/N) * (softmax(logits) - one_hot(target))` | `src/cpu/train.rs` | `CrossEntropyLoss::backward` |
| `AdamW: m=b1*m+(1-b1)*g, v=b2*v+(1-b2)*g^2, w-=lr*(m_hat*rsqrt(v_hat+eps)+wd*w)` | `src/cpu/train.rs` | `AdamW::step` -> `simd::adamw_step_simd` |
| `Fast rsqrt: 1/sqrt(x) ≈ vrsqrteq_f32(x) * (1.5 - 0.5*x*est*est)` | `src/cpu/simd.rs` | `fast_rsqrt_slice` |
| `Gradient clipping: g = g * clip/norm if norm > clip` | `src/cpu/train.rs` | `clip_grad_by_global_norm_inplace` |
| `Cosine LR: lr = min_lr + 0.5*(lr-min_lr)*(1+cos(pi*progress))` | `src/cpu/train.rs` | `Trainer::get_lr` |
| `Temperature sampling: p = softmax(logits / T)` | `src/cpu/generate.rs` | `sample_from_logits` |
| `Top-p (nucleus): keep smallest set with cumulative prob >= p` | `src/cpu/generate.rs` | `sample_top_p` |

## Implementation Notes

### Ownership Model for Tensor

Tensor owns its data via `Vec<f32>`. Key patterns:

- **Move semantics in the forward pass**: `let mut x = embed(...); x = block.forward(&x);` -- each layer takes `&Tensor` (borrowed) and returns a new `Tensor` (owned). The old `x` is dropped when reassigned, so only one hidden-state tensor is live per layer.
- **Gradient storage**: `grad: Option<Box<Tensor>>` is colocated inside the parameter Tensor itself. Each layer's `backward()` accumulates per-parameter gradients into `param.grad` during the backward pass, and AdamW reads them via `parameters_mut()`.
- **Borrow splitting in AdamW**: The optimizer needs `param.grad()` (immutable borrow) and `param.data_mut()` (mutable borrow) simultaneously. Since Rust won't allow both, the grad data is copied to a temporary `Vec<f32>` first.

### BLAS FFI Pattern

All matrix multiplications route through `src/cpu/accelerate.rs`, which wraps `cblas_sgemm` from Apple's Accelerate framework (uses AMX coprocessor on Apple Silicon):

```
rust/src/cpu/accelerate.rs
  #[link(name = "Accelerate", kind = "framework")]
  extern "C" { fn cblas_sgemm(...) }
```

- **Safety boundary**: `#![deny(unsafe_code)]` at crate root; only `src/cpu/accelerate.rs` and `src/cpu/simd.rs` have `#![allow(unsafe_code)]`. In `src/cpu/accelerate.rs`, all unsafe is in two functions: `sgemm` and `sgemm_transb`. In `src/cpu/simd.rs`, unsafe is confined to NEON intrinsic calls (`vrsqrteq_f32`, `vld1q_f32`, `vst1q_f32`, etc.) with documented SAFETY invariants per block.
- **Framework linking**: `#[link(name = "Accelerate", kind = "framework")]` tells the linker to use `-framework Accelerate`. No build.rs or pkg-config needed on macOS.
- **SAFETY invariants** (documented in each unsafe block):
  1. Slice bounds verified by `debug_assert!(a.len() >= m*k)` etc.
  2. Non-overlapping A, B, C regions (guaranteed by Rust's borrow rules: `&[f32]` for A/B, `&mut [f32]` for C)
  3. Leading dimensions match row-major layout

### `from_vec()` vs `from_slice()` -- When to Use Which

| Method | Allocation | Use when |
|---|---|---|
| `Tensor::from_vec(data, shape)` | Zero-copy (takes ownership of the Vec) | You already own the buffer (e.g., freshly computed `Vec<f32>`) |
| `Tensor::from_slice(data, shape)` | Copies the slice into a new Vec | You're borrowing data from another Tensor (e.g., slicing a chunk) |

**Rule of thumb**: If you just built the data with `vec![...]` or `.collect()`, use `from_vec`. If you're extracting a sub-slice from an existing tensor's data, use `from_slice`.

Example in `src/cpu/moe.rs`: Each token chunk is a borrowed slice of the input tensor, so `from_slice` is required:
```rust
// chunk borrows input.data() -- must copy
let token = Tensor::from_slice(chunk, Shape::new(&[1, 1, hidden]));
```

### Pre-allocated Output Buffers (`_into` Pattern)

Functions suffixed with `_into` write results into a caller-provided buffer instead of allocating:
- `Tensor::softmax_into(&self, out: &mut Tensor)` -- avoids allocation per softmax call
- `Tensor::add_into(&self, other: &Tensor, out: &mut Tensor)` -- avoids allocation per residual add

This pattern is critical in hot loops (attention score computation) where allocation overhead dominates small-tensor operations.

### NEON SIMD Optimization

The `src/cpu/simd.rs` module provides NEON intrinsic-based fast paths for two hot loops:

1. **AdamW inner loop** (`adamw_step_simd`): Processes 4 parameters per cycle using NEON vector registers. The key optimization replaces `m_hat / (sqrt(v_hat) + eps)` with `m_hat * rsqrt(v_hat + eps)`, avoiding the expensive scalar `sqrt()` call. The rsqrt uses `vrsqrteq_f32` (~12-bit hardware estimate) + one Newton-Raphson refinement (`vrsqrtsq_f32`) for ~23-bit accuracy.

2. **RMSNorm normalization** (`fast_rsqrt_slice`): Computes `1/sqrt(mean(x^2) + eps)` using the same approximate rsqrt path, called from `src/cpu/layers.rs::RMSNorm::forward`.

This matches Julia's `@fastmath` approach, which lowers to ARM NEON `frsqrte` + Newton-Raphson. Both have `#[cfg(not(target_arch = "aarch64"))]` scalar fallbacks for non-ARM targets. Tail elements (when length is not a multiple of 4) also use scalar fallback.

### In-Place Gradient Clipping

`clip_grad_by_global_norm_inplace()` consumes the gradient tensor and scales in-place, replacing the previous `clip_grad_by_global_norm()` which cloned the tensor. This avoids a 64KB allocation per train step (for the gradient tensor copy).

### Row-Major Matmul Stride Calculation

CBLAS uses "leading dimension" (lda, ldb, ldc) to express stride between consecutive rows:

```
Row-major, no transpose:
  A[m,k] -> lda = k    (k elements per row)
  B[k,n] -> ldb = n    (n elements per row)
  C[m,n] -> ldc = n    (n elements per row)

Row-major, B transposed:
  A[m,k] -> lda = k
  B[n,k] -> ldb = k    (stored as [n,k], but BLAS reads it transposed)
  C[m,n] -> ldc = n
```

The key gotcha: when `trans_b = CblasTrans`, `ldb` is the stride of B's **storage layout** (not the transposed view). Since B is stored as `[n,k]`, `ldb = k`.

## Performance Characteristics

Latest benchmark results (M1, batch=2, seq=32, hidden=64):

| Metric | Value | Rank (of 4 languages) |
|--------|-------|------|
| Forward pass | 0.56ms | **1st** (Julia 0.59ms) |
| Train step | 1.44ms | 2nd (Julia 0.98ms) |
| Matmul 64x64 | 629 GFLOPS | 1st* (tied with Julia 629, within noise) |
| Softmax kernel | 1.83us | **1st** (Julia 4.83us, Go 5.67us) |
| RMSNorm kernel | 3.38us | **1st** (Julia 4.17us, Go 8.75us) |
| Peak RSS | 19MB | **1st** (Go 31MB, Python 61MB, Julia 490MB) |
| Train alloc | 1.4MB | 2nd (Julia 3.0MB) |
| T4 throughput | 4,732 inf/s | **1st** (Julia 4,226) |
| T4 train | 1,309 trn/s | 2nd (Julia 1,558) |
| GC throughput | 1.000 | N/A (no GC exists) |

**Why Rust leads non-BLAS kernels**: LLVM AOT produces the tightest scalar loops -- bounds checks eliminated via iterator chains, aggressive inlining, auto-vectorization where possible.

**Why Julia leads training by 1.47x**: Julia's `@.` broadcast fusion is a **language-level fusion compilation** capability that fuses multiple element-wise operations into a single pass, eliminating intermediate allocations. On CPU, this manifests as `@.` broadcast; on GPU, the same capability manifests as Reactant.jl (XLA backend) for automatic whole-graph kernel fusion. Rust has no equivalent automatic fusion -- each element-wise backward operation runs as a separate loop. Rust's NEON rsqrt matches Julia for AdamW sqrt, but the backward pass architecture gap persists.

**RSS advantage**: 19MB total = no runtime, no GC, no JIT. Model weights (~0.1MB at hidden=64) + stack. Julia's equivalent performance costs 490MB for the JIT runtime.

## Gotchas / Pitfalls

### 1. `usize` to `i32` Overflow in BLAS Dimension Args

CBLAS takes `i32` dimensions. Rust's `usize` is 64-bit on Apple Silicon. A naive `v as i32` silently truncates large values, which would cause BLAS to read/write wrong memory (undefined behavior). The `dim_i32()` function guards against this:

```rust
fn dim_i32(v: usize) -> i32 {
    assert!(v <= i32::MAX as usize, "matrix dimension {v} exceeds i32::MAX");
    v as i32
}
```

This is a hard `assert!` (not `debug_assert!`) because incorrect dimensions in a BLAS call are a safety violation.

### 2. Tensor Construction Overhead in Benchmarks

The kernel benchmarks (matmul, softmax) call `sgemm` directly instead of `Tensor::try_matmul` to avoid measuring Tensor overhead:
- `try_matmul` does `batch_dims().to_vec()` comparison (heap allocation)
- Shape construction via `Vec::with_capacity` + `extend_from_slice`
- `dim_i32()` conversion per call

For a 64x64 matmul, this overhead can be comparable to the actual BLAS time. Direct `sgemm` calls isolate the AMX coprocessor cost for fair cross-language comparison.

### 3. Mutable Borrow Splitting for In-Place Operations

Rust's borrow checker prevents simultaneous `&self` and `&mut self` access. This affects:

**RoPE application**: `apply_rope` takes `q_data: &mut [f32]` and `k_data: &mut [f32]` as separate slices rather than `&mut Tensor` references, because Q and K are independent tensors and we need to mutate both in the same function call.

**AdamW step**: Cannot call `param.grad()` (borrows `param`) and `param.data_mut()` (mutably borrows `param`) at the same time. Solution in `step()`: copy the gradient to a temporary `Vec<f32>` before taking the mutable borrow.

**MoELayer route caching**: The `Layer` trait requires `forward(&self, ...)`, but MoELayer needs to cache routing decisions for `aux_loss()`. Solution: `RefCell<Option<RouteData>>` for interior mutability, which is safe because forward is always single-threaded.

### 4. `RefCell` Makes `MoETransformer` `!Sync`

The `RefCell` in `MoELayer::last_route` makes the entire model `!Sync`. For the parallel benchmark, each thread needs its own model instance. The benchmark wraps each model in `Mutex<MoETransformer>` -- not for contention (each thread has its own Mutex), but purely for `Send`/`Sync` trait compliance.

### 5. Softmax Numerical Stability

The max-subtraction trick (`exp(x - max(x))`) prevents overflow, but the denominator can still be zero if all inputs are `-inf` (happens with causal masking when softmax is applied to a single element). The `sum.max(1e-12)` clamp prevents `1/0 = inf` from propagating.

## GPU (Metal/MPS)

Requires macOS with Apple Silicon. Build with Metal feature:

```bash
cargo build --release --features metal
cargo run --release --features metal --bin bench
# GPU-only benchmark JSON
ROSETTA_GPU_ONLY=1 cargo run --release --features metal --bin bench
```

The Metal backend uses a C/Objective-C bridge (metal-bridge/) compiled via `cc` crate.
GPU benchmarks are feature-gated and only included when `--features metal` is specified.
Timed GPU scenarios avoid host readback in the hot path (for example `gpu_train_step_no_readback`).

## Build & Run

```bash
# Tests
cargo test

# Release build
cargo build --release

# CPU-only benchmark JSON
ROSETTA_CPU_ONLY=1 cargo run --release --bin bench

# Override trials/warmup (defaults: 10/3)
ROSETTA_BENCH_TRIALS=30 ROSETTA_BENCH_WARMUP=3 ROSETTA_CPU_ONLY=1 cargo run --release --bin bench

# Project-root one-shot runs
cd .. && make bench-all
cd .. && make bench-all-30
```
