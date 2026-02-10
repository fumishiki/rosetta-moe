# Julia MoE Transformer

Educational Mixture-of-Experts Transformer implementation in pure Julia. Part of a 4-language benchmark suite (Rust, Go, Python, Julia) for comparing language characteristics in ML workloads.

## Architecture Overview

```
MoETransformer.jl          Module entry point, exports, BLAS backend setup
  tensor.jl                Tensor type, element-wise ops, BLAS matmul, softmax
  config.jl                Model hyperparameters (Config struct)
  layers.jl                Embedding, Linear, RMSNorm, SwiGLU
  attention.jl             Multi-Query Attention with RoPE
  moe.jl                   Router (top-k gating), MoELayer, TransformerBlock
  model.jl                 MoETransformer assembly, forward/backward
  generate.jl              Sampling strategies (greedy, temperature, top-k, top-p)
  train.jl                 AdamW optimizer, cross-entropy, checkpointing, loss scaling
  bench.jl                 Benchmark harness (5 axes, 22 scenarios, JSON output)
  test/runtests.jl         Test suite
```

All source files are included into a single `MoETransformer` module. The include order in `MoETransformer.jl` defines the dependency chain.

## Equation-to-Code Map

| Formula | File | Function |
|---------|------|----------|
| `Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))` | `tensor.jl` | `softmax!`, `softmax_in_place!`, `_softmax_arr` |
| `SiLU: f(x) = x / (1 + exp(-x))` | `tensor.jl` | `silu`, `silu_in_place!`, `_silu_in_place_arr!` |
| `L1 Normalize: x_i = x_i / sum(x)` | `tensor.jl` | `normalize_in_place!` |
| `Matmul: C = A × B` (batched BLAS) | `tensor.jl` | `tensor_matmul`, `tensor_matmul!`, `_tensor_matmul_2d` |
| `Embedding: out = W[:, token_id]` | `layers.jl` | `forward(::Embedding, ...)` |
| `Linear: y = x @ W^T + b` | `layers.jl` | `forward(::Linear, ...)` via `matmul_transposed_b!` |
| `RMSNorm: y = x * (1/sqrt(mean(x^2) + eps)) * gamma` | `layers.jl` | `_rmsnorm_forward!` |
| `RMSNorm backward: d_x = d_y*gamma/rms - x*(sum(d_y*gamma*x/rms))/(dim*rms^3)` | `layers.jl` | `backward(::RMSNorm, ...)` |
| `SwiGLU: output = W_down @ (silu(W_gate @ x) * (W_up @ x))` | `layers.jl` | `forward(::SwiGLU, ...)` |
| `RoPE: [x0,x1] -> [x0*cos(t)-x1*sin(t), x0*sin(t)+x1*cos(t)]` | `attention.jl` | `_rotate_half!` |
| `Attention: scores = Q@K^T/sqrt(d_k), weights = softmax(scores+mask), out = weights@V` | `attention.jl` | `forward(::MQAttention, ...)` |
| `Router: probs = softmax(x @ W), top-k select + renormalize` | `moe.jl` | `forward(::Router, ...)` |
| `MoE: output = sum_k(gate_k * Expert_k(x)) for top-k experts` | `moe.jl` | `_moelayer_forward!`, `_moe_accumulate!` |
| `Aux Loss: L_aux = alpha * N * sum_e(f_e * P_e)` | `moe.jl` | `compute_aux_loss` |
| `CrossEntropy: L = -mean(logits[target] - log(sum(exp(logits))))` | `train.jl` | `cross_entropy_loss` |
| `CE Gradient: d_logits = (softmax(logits) - one_hot(target)) / N` | `train.jl` | `_cross_entropy_grad_into!` |
| `AdamW: m=b1*m+(1-b1)*g, v=b2*v+(1-b2)*g^2, w-=lr*(m_hat/(sqrt(v_hat)+eps)+wd*w)` | `train.jl` | `train_step!` |
| `LR Schedule: cosine decay with linear warmup` | `train.jl` | `get_lr` |
| `Gradient Clip: g *= clip_norm/norm(g) if norm(g) > clip_norm` | `train.jl` | `clip_grad_by_global_norm!` |
| `Temperature sampling: p = softmax(logits / T)` | `generate.jl` | `pick_token(::TemperatureSampling, ...)` |
| `LCG PRNG: state = state * 6364136223846793005 + 1` | `generate.jl` | `next_rand01` |

## Implementation Notes

### Column-Major Layout and BLAS

Julia stores arrays in **column-major** (Fortran) order: the first index varies fastest in memory. A `Matrix{Float32}(undef, M, K)` has contiguous columns of length M.

This is the natural layout for BLAS `sgemm`, so `LinearAlgebra.mul!(C, A, B)` on 2D matrices dispatches directly to BLAS with no copy or transpose. This is the opposite of row-major languages (C, Python/NumPy, Rust `ndarray`) where the last index is contiguous.

**Shape convention**: Shapes are written as `(batch, seq_len, hidden_dim)` throughout the codebase. In column-major, this means `batch` is the fastest-varying (contiguous) index. The shape notation is the same as PyTorch's `(batch, seq_len, hidden_dim)`, but the memory layout is reversed: Julia has `batch` contiguous while PyTorch (row-major) has `hidden_dim` contiguous.

### LBT -> AppleAccelerate.jl for BLAS

Julia uses **libblastrampoline (LBT)** as a BLAS dispatch layer. At load time, `AppleAccelerate.jl` (loaded in `MoETransformer.jl`) replaces the default OpenBLAS backend via LBT, routing all `mul!` calls to Apple's Accelerate framework.

On Apple Silicon, Accelerate uses the AMX (Apple Matrix eXtension) coprocessor, giving **7-14x speedup** over NEON-only OpenBLAS for matrix multiplication. This is a load-time side-effect -- no code changes are needed in downstream `mul!` calls.

### Zero-Alloc Forward Pass

The forward pass achieves **0 GC pauses** in steady state through pre-allocated buffers:

- Each layer struct has `Union{Buffer, Nothing}` fields (e.g., `matmul_buf`, `output_buf`, `scores_buf`)
- On the first call, buffers are allocated and stored in the struct
- On subsequent calls, buffers are reused if the shape matches
- `reshape()` in Julia creates a view (zero-copy), not a new allocation

This means after JIT warmup, the forward pass allocates 0 bytes and triggers 0 GC pauses.

### Multiple Dispatch for Type Specialization

Public `forward(layer, input::Tensor)` methods delegate to inner `_*_forward!(layer, data::Array{Float32,N}, ...)` functions. This pattern gives the Julia compiler a **concrete type** (`Array{Float32,N}` where N is known) instead of `Array{Float32}` (abstract), enabling better type inference and avoiding dynamic dispatch overhead.

Sampling strategies also use multiple dispatch: `pick_token(::GreedySampling, logits)` vs `pick_token(::TopKSampling, logits)` -- adding a new strategy requires only a new struct + method, no if/else chains.

### JIT Warmup Cost

Julia uses JIT (Just-In-Time) compilation. The **first call** to any new method specialization triggers compilation. The benchmark measures this via:

- `dispatch_cold`: constructs a new model + runs first forward each trial (includes JIT)
- `dispatch_warm`: reuses model after warmup (JIT already done)

The difference between cold and warm dispatch timings shows the JIT compilation cost. After warmup, Julia code runs at native speed.

### `@views` Gotcha: 3D Non-Contiguous Slices

For a 3D array `A` of shape `(B, M, K)` in column-major layout:
- `@view A[batch, :, :]` yields a 2D slice with `stride[1] = B` (not 1)
- BLAS `sgemm` requires `stride[1] = 1` (contiguous first dimension)
- Julia **silently falls back** to generic matrix multiply for non-contiguous views (~10x slower)

The solution in `_tensor_matmul_3d` is to manually copy each batch slice into a contiguous `Matrix{Float32}` buffer before calling `mul!`. This ensures real BLAS dispatch at the cost of a copy.

### `@fastmath` SIMD Optimization

`@fastmath` is the single most impactful optimization for training performance. It allows the compiler to use approximate SIMD intrinsics by relaxing IEEE 754 guarantees (NaN propagation, floating-point associativity). For ML training, where gradient noise already exceeds floating-point error, this is standard practice.

**AdamW inner loop** (`_adamw_update_param!` in `train.jl`): The `sqrt(v_hat)` in the denominator becomes an approximate SIMD square root -- on ARM, this is NEON `frsqrte` (reciprocal sqrt estimate) + Newton-Raphson refinement, processing 4 Float32 elements per instruction. Result: **4.58x speedup** on AdamW parameter updates.

**Cross-entropy loss/gradient** (`cross_entropy_loss`, `_cross_entropy_grad_permuted!` in `train.jl`): `exp()` and `log()` in the softmax and log-sum-exp computations become approximate SIMD variants. Result: **1.6-1.8x speedup**.

This is a one-line annotation (`@fastmath` before the loop) vs Rust's ~40 lines of explicit NEON intrinsics for the same approximation -- a significant ergonomic advantage. The trade-off is implicit: the programmer must know that `@fastmath` changes numerical semantics.

## Performance Characteristics

Latest benchmark results (M1, batch=2, seq=32, hidden=64):

| Metric | Value | Rank (of 4 languages) |
|--------|-------|------|
| Forward pass | 0.59ms | 2nd (Rust 0.56ms) |
| Train step | **0.98ms** | **1st** (Rust 1.44ms, Go 3.28ms) |
| Matmul 64x64 | 629 GFLOPS | 1st* (tied with Rust 629, within noise) |
| Softmax kernel | 4.83us | 2nd (Rust 1.83us) |
| Peak RSS | 490MB | 4th (JIT runtime) |
| T4 throughput | 4,226 inf/s | 2nd (Rust 4,732) |
| T4 scaling | **2.88x** | **1st** |
| T4 train | **1,558 trn/s** | **1st** (Rust 1,309) |
| GC throughput | 1.000 | 1st (zero GC across all 22 scenarios) |
| Train alloc | 3.0MB | 2nd (Rust 1.4MB) |

**Fastest training thanks to broadcast fusion**: 0.98ms/step. Julia's `@.` broadcast is not merely a CPU optimization — it is a **language-level fusion compilation** capability. The compiler fuses multiple element-wise operations into a single pass, eliminating intermediate allocations. On GPU, the same capability manifests as [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (XLA backend) for automatic whole-graph kernel fusion. Rust has no equivalent automatic fusion.

**Zero GC is real**: After type stability optimization (function barriers, pre-allocated buffers, `@.` broadcast fusion), Julia achieves gc_throughput = 1.000 across ALL 22 benchmark scenarios. The JIT specializes code paths to eliminate boxing and allocation entirely.

**Column-major BLAS alignment**: Julia's column-major default matches Fortran-order sgemm. At seq=64, Julia is close to Rust (1.29ms vs 1.47ms) as larger matrices exploit this alignment.

**RSS is the price of JIT**: ~490MB = LLVM JIT compiler + type inference + method cache + GC infrastructure. The model itself is ~0.1MB. This is architectural -- reducible to ~30-60MB with `juliac --trim` (Julia 1.12+) or 0.5-2MB with StaticCompiler.jl (requires complete rewrite).

**Best parallel scaling**: JIT-specialized per-thread dispatch paths. 2.88x T4/T1 scaling (best ratio). Training T4: 1,558 trn/s (1st).

## Gotchas / Pitfalls

### Training Is the Fastest (472x Improvement from Initial)

The backward pass was massively optimized from 231ms to **0.98ms** (236x faster), with GC pauses dropping from 62ms to **0ms**. Key optimizations:

- **Function barrier for AdamW**: `_adamw_update_param!` dispatches on concrete `Array{Float32,N}`, enabling LLVM to emit SIMD-vectorized code. Without this, the inner loop falls back to generic (unvectorized) iteration.
- **`@fastmath` for SIMD approximations**: Enables approximate `sqrt` (ARM NEON `frsqrte` + Newton-Raphson) in AdamW, and approximate `exp`/`log` in cross-entropy. See the `@fastmath` implementation note above.
- **Column-major stride fix in cross-entropy**: `permutedims!` to `(vocab, batch, seq_len)` makes vocab the contiguous axis, eliminating stride-`batch*seq_len` access in the inner loop (26x speedup).
- **RMSNorm backward function barrier**: `_rmsnorm_backward!` dispatches on concrete `Array{Float32,N1}` / `Array{Float32,N2}` for type-stable SIMD code generation.
- **Pre-allocated gradient buffers**: Every layer now has `grad_buf` fields (`Union{Array{Float32}, Nothing}`) that are allocated once and reused, matching the forward pass pattern.

Julia is the fastest of all 4 languages for training (0.98ms vs Rust 1.44ms). The broadcast fusion advantage is not CPU-specific — it reflects Julia's language-level fusion compilation that also applies on GPU via Reactant.jl.

### JULIA_NUM_THREADS Must Be Set Before Launch

Julia's thread count is fixed at process startup. Setting `JULIA_NUM_THREADS` after Julia has launched has no effect:

```bash
JULIA_NUM_THREADS=4 julia --project=. bench.jl
```

The parallel benchmark scenarios test scaling with `Threads.@threads`, but actual parallelism depends on this environment variable.

### Column-Major Shapes Are "Reversed" vs PyTorch

In PyTorch (row-major), a tensor of shape `(batch, seq, hidden)` has `hidden` as the contiguous dimension. In Julia (column-major), the same shape `(batch, seq, hidden)` has `batch` as the contiguous dimension.

This means:
- Weight matrices are stored as `(out_features, in_features)` (same shape as PyTorch, but transposed in memory)
- `matmul_transposed_b!(out, input, weight)` computes `input @ weight^T` to match PyTorch's `F.linear`
- Permutation to `(head_dim, heads, seq, batch)` in attention puts `head_dim` as the fastest-varying dimension for cache-friendly dot products

### `@view` of Non-Contiguous 3D Slice Falls Back to Generic Multiply

As described above, `@view arr[batch, :, :]` for a 3D column-major array has non-unit stride in the first dimension. Passing this view to `mul!` will silently use Julia's generic matrix multiply instead of BLAS sgemm. The fix is explicit copying into contiguous buffers. See `_tensor_matmul_3d` in `tensor.jl`.

### ~~`partialsortperm` in Router Allocates~~ (Fixed)

The Router no longer uses `partialsortperm`. It now uses a manual O(n_experts * top_k) partial selection sort with a pre-allocated `perm_buf::Vector{Int}`. For the typical case of n_experts=4-16 and top_k=2-4, this is faster than `partialsortperm` and eliminates all allocation in the routing path. See `_router_softmax_inner!` in `moe.jl`.

## Running

```bash
# Tests
cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia --project=. test/runtests.jl

# Benchmark
JULIA_NUM_THREADS=4 julia --project=. bench.jl > results.json
```
