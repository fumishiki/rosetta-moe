<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# rosetta-moe

[![CI](https://github.com/fumishiki/rosetta-moe/actions/workflows/ci.yml/badge.svg)](https://github.com/fumishiki/rosetta-moe/actions/workflows/ci.yml)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange)
![Go](https://img.shields.io/badge/Go-1.22-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Julia](https://img.shields.io/badge/Julia-1.10+-purple)

**Same MoE Transformer. 4 languages. 22 benchmark scenarios. One hardware.**

Rust, Go, Python, Julia — each implements the same MoE Transformer from scratch (forward, backward, optimizer, inference). All matmul hits the same Apple Accelerate BLAS on the same M1. No frameworks — every gradient is hand-derived.

## Results

| | Forward | Train | T4 Inference | T4 Training | softmax | RSS |
|---|---|---|---|---|---|---|
| **Rust** | **0.57 ms** | 1.31 ms | **4,738 inf/s** | 1,309 trn/s | **1.83 us** | **20 MB** |
| **Julia** | 0.61 ms | **1.00 ms** | 4,205 inf/s | **1,558 trn/s** | 4.83 us | 488 MB |
| **Go** | 1.14 ms | 3.33 ms | 2,121 inf/s | 867 trn/s | 5.67 us | 32 MB |
| **Python** | 2.37 ms | 10.03 ms | 1,898 inf/s | 503 trn/s | 16.2 us | 63 MB |

Rust leads inference and kernels. Julia leads training. Both hit >37% of M1 AMX peak on matmul. At hidden=256, the forward gap already narrows from 4.2x to 1.9x.

<details>
<summary>Parallel scaling detail</summary>

### Inference (T4/T1)

| | Julia | Rust | Go | Python |
|---|---|---|---|---|
| T1 | 1,458 inf/s | **1,742** | 874 | 768 |
| T4 | 4,205 inf/s | **4,738** | 2,121 | 1,898 |
| Speedup | **2.88x** | 2.72x | 2.43x | 2.47x |

### Training (T4/T1)

| | Julia | Rust | Go | Python |
|---|---|---|---|---|
| T1 | **943 trn/s** | 758 | 311 | 175 |
| T4 | **1,558 trn/s** | 1,309 | 867 | 503 |
| Speedup | 1.65x | 1.73x | 2.79x | **2.88x** |

Training scaling is lower because backward pass triples AMX bus transactions. [Full AMX analysis](docs/analysis-parallel-training.md)

</details>

Full results with 22 scenarios across 5 axes: [`docs/bench-results.md`](docs/bench-results.md)

## Quick Start

```bash
make test          # test all 4 languages
make bench         # benchmark + summary table
make convergence   # verify loss convergence
```

<details>
<summary>Per-language commands & setup</summary>

```bash
make test-rust / make test-go / make test-python / make test-julia
make bench-rust / make bench-julia

# First-time setup
cd python && pip install -e ".[dev]" && cd ..
cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
```

</details>

## Why This Exists

| Question | Finding |
|----------|---------|
| **Is GC actually a problem for ML?** | No. Go: 29M alloc/train, gc_throughput 0.996. Julia: 0 GC pauses. |
| **Does BLAS FFI overhead matter?** | Rust/Julia both >37% M1 peak. Language only matters for non-BLAS code. |
| **Rust vs Julia for ML?** | Rust leads inference (0.57ms). Julia leads training (1.00ms via broadcast fusion). |
| **Does column-major help?** | Yes for training. No for inference (Rust's optimization overcomes layout disadvantage). |
| **Do language gaps persist at scale?** | Forward spread: 4.2x (h=64) → 1.9x (h=256). BLAS share grows, gap shrinks. |

## What Makes This Different

| | This repo | Benchmarks Game | ML framework benchmarks | Blog "X vs Y" posts |
|---|---|---|---|---|
| Workload | Full MoE Transformer (fwd + bwd + opt) | Toy algorithms | Framework-level API | Cherry-picked microbenchmarks |
| BLAS control | All languages share same BLAS (Apple Accelerate) | N/A | Each framework bundles own BLAS | Uncontrolled |
| GC analysis | `gc_throughput`, pause count, per-scenario instrumentation | None | None | "Rust has no GC" (hand-wave) |
| Methodology | 5 axes, 22 scenarios, N=10 median, literature-backed metrics | Single metric | Wall time only | Single run |
| Parallel model | std::thread / goroutine / ProcessPool / Threads.@threads | Varies | Framework-managed | Rarely tested |
| Reproducibility | Fixed seed, fixed input, getrusage, JSON output | Varies | Docker-dependent | Not reproducible |
| Math traceability | 21-entry equation-to-code map per language | None | None | None |

## 5 Research Axes

| Axis | What it isolates | Key finding |
|------|-----------------|-------------|
| **Memory** | Ownership vs GC vs RC under scaling | Rust 20MB/0.57ms fwd. Julia 0 GC forward + training (0 pauses, 0ns). Go 29M alloc/train, invisible GC (0.996) |
| **Compiler** | BLAS dispatch + loop optimization | Rust/Julia both >37% M1 peak. Rust leads softmax (1.83us) and rmsnorm (3.38us) |
| **Type System** | Dispatch overhead | Rust 0.57ms warm, Julia 0.61ms. Rust leads forward, Julia leads training |
| **Parallel** | Thread scaling (inference + training) | Rust leads inference T4 (4,738 inf/s). Julia leads training T4 (1,558 trn/s). Training scaling limited by AMX contention |
| **Scale** | Convergence at larger model | Forward spread: 4.2x (h=64) → 1.9x (h=256) |

## Architecture

All 4 implementations share the same model contract:

```
Input token_ids [batch, seq]
  → Embedding lookup                    [batch, seq, hidden]
  → N × TransformerBlock:
      RMSNorm → MQ Attention (RoPE)   → Residual add
      RMSNorm → MoE (Router + SwiGLU) → Residual add
  → Final RMSNorm
  → Linear (LM head)                   [batch, seq, vocab]
```

### Model Specification (benchmark config)

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 64 | Intentionally small to expose language overhead vs compute |
| n_layers | 2 | Transformer blocks |
| n_heads | 4 | Multi-Query Attention (1 KV head) |
| head_dim | 16 | hidden / n_heads |
| n_experts | 4 | Mixture-of-Experts per block |
| top_k | 2 | Active experts per token |
| intermediate_dim | 128 | SwiGLU hidden (2 × hidden) |
| vocab_size | 1000 | Embedding table rows |
| RoPE base | 10000 | With NTK-aware scaling |
| Total params | ~0.1MB | Tiny model — overhead dominates, not compute |

## Equation-to-Code Traceability

Every math formula in the model is mapped to concrete source code:

| Category | Equations |
|----------|-----------|
| Core layers | Embedding, RMSNorm, Linear, SwiGLU, Softmax, SiLU |
| Attention | RoPE rotation, Q@K^T/√d attention scores, causal masking |
| MoE | Router gating (softmax + top-k), weighted expert combine, auxiliary load-balancing loss |
| Training | Cross-entropy loss + gradient, AdamW (bias-corrected moments), gradient clipping, LR schedule (warmup + cosine) |
| Inference | Temperature sampling, top-p (nucleus) sampling |

All source files are annotated with inline comments mapping math notation to code. See each language's README for the full table.

## Technology Stack

| Language | BLAS Integration | Non-BLAS Optimization | Parallelism |
|----------|-----------------|----------------------|-------------|
| **Rust** | Direct FFI (`extern "C"` → Accelerate) | NEON SIMD rsqrt (`std::arch::aarch64`) | `std::thread::scope` |
| **Go** | CGO bridge (`#cgo LDFLAGS: -framework Accelerate`) | Go compiler (no SIMD) | goroutine + `sync.WaitGroup` |
| **Python** | NumPy → Accelerate (automatic on macOS ≥ 14) | NumPy vectorization | `ProcessPoolExecutor` |
| **Julia** | `LinearAlgebra.mul!` via LBT → `AppleAccelerate.jl` | `@fastmath` approximate SIMD | `Threads.@threads` |

All matrix multiplications route through Apple Accelerate's `cblas_sgemm`, which uses the AMX coprocessor on Apple Silicon. Non-matmul operations (softmax, RMSNorm, attention masking, MoE routing, AdamW optimizer) are hand-written loops — these are where language differences show up most.

## Key Findings

1. **BLAS levels the playing field.** All 4 languages hit the same AMX — the language only matters for non-BLAS code.
2. **GC is not the enemy.** Go gc_throughput 0.985-1.000 across all scenarios. Julia: zero GC pauses on training.
3. **Broadcast fusion > zero-alloc.** Julia's `@.` fused backward (1.00ms) beats Rust's hand-written zero-alloc backward (1.31ms).
4. **AMX contention is the parallel ceiling.** Inference T4/T1: 2.4-2.9x. Training T4/T1: 1.7x (3x more AMX bus transactions).
5. **Language gaps collapse at scale.** Forward spread: 4.2x (h=64) → 1.9x (h=256). BLAS share grows with model size.
6. **RSS inversely correlates with ergonomics.** Rust 20MB (manual everything) → Julia 488MB (JIT buys dispatch + fusion + zero-GC).

## Why Julia Beats Rust at Training

Rust leads forward (0.56ms vs 0.59ms) — but Julia leads training (0.98ms vs 1.44ms). The reversal comes from **backward pass implementation**.

```
Before: Julia 7.36ms (Rust was already ~1.4ms — Rust winning by 5x)
After:  Julia 0.98ms (Rust 1.44ms — Julia winning by 1.5x)
```

The fix was one pattern change — `@simd` → `@.` broadcast:

```julia
# Before: @simd — 9K-11K heap allocations per loop (Julia 1.12 SIMD lane metadata)
@inbounds @simd for i in eachindex(grad)
    grad[i] = grad[i] * mask[i]
end

# After: @. broadcast — compiler fuses into single loop, zero intermediate allocation
@. grad = grad * mask
```

| Metric | Before (`@simd`) | After (`@.`) | Change |
|--------|-----------------|-------------|--------|
| Train alloc | 55.5 MB/step | 3.0 MB/step | **-94%** |
| Train time | 7.36 ms | 0.98 ms | **-87%** |
| Fwd:Bwd ratio | 12:1 | 1.7:1 | -- |
| GC pauses | >0 | **0** | -- |

**Why Rust can't match this.** Rust's backward is zero-alloc (1.4 MB/step) — every gradient buffer is pre-allocated. But without broadcast fusion, each element-wise operation runs as a separate loop. Julia's compiler fuses `grad = grad * mask * scale + bias` into a single pass over memory. Rust would need a hand-written fused kernel for each combination — possible, but trades developer velocity for performance.

**The insight:** Zero allocation is not the same as zero overhead. Julia's broadcast fusion eliminates not just allocations but also loop overhead, cache misses from multiple passes, and function call boundaries. This is why a GC language beats a zero-GC language on training — the compiler does work that the programmer can't practically do by hand for every operation combination.

Rust still leads inference (no backward pass), all non-BLAS kernels (softmax 1.83us vs 4.83us), and memory footprint (20MB vs 523MB).

## When to Choose What

| Use Case | Recommended | Rationale |
|----------|-------------|-----------|
| Production inference (latency) | Rust | 0.57ms forward, fastest single-thread inference |
| Training pipeline | Julia | 1.00ms/step, optimized backward with zero GC |
| Concurrent inference | Rust | 4,738 inf/s at T4, best absolute throughput |
| Research prototyping | Python | PyTorch/JAX ecosystem, fastest time-to-experiment |
| Scientific computing (BLAS-heavy) | Julia | Column-major BLAS alignment, 1.00ms train, best scaling ratio |
| Edge / embedded / WASM | Rust | 20MB RSS, no runtime, `no_std` possible |
| Microservice / API backend | Go | Simple deploy, invisible GC, fast compilation |
| Maximum kernel throughput | Rust | softmax 1.83us — tightest scalar loops |
| Correctness-critical numerics | Rust | IEEE 754 by default, each approximation auditable |
| Rapid numerical experimentation | Julia | Column-major + REPL + JIT = iterate fast |
| Minimum memory footprint | Rust | 20MB RSS, no runtime overhead |

<details>
<summary>Per-language profiles</summary>

### Rust — Precision by Default, Speed by Choice

| Metric | Value | Rank |
|--------|-------|------|
| Forward | **0.57ms** | **1st** |
| Train | 1.31ms | 2nd |
| softmax | **1.83us** | **1st** |
| rmsnorm | **3.38us** | **1st** |
| RSS | **20MB** | **1st** |

**Strengths**: Leads forward (0.57ms), parallel T4 (4,738 inf/s), and all non-BLAS kernels. Lowest RSS (no runtime/GC/JIT). Zero GC by construction.

**Weakness**: Training 1.31ms — 31% behind Julia (1.00ms) after Julia backward optimization.

### Julia — JIT Compilation as Investment

| Metric | Value | Rank |
|--------|-------|------|
| Forward | 0.61ms | 2nd |
| Train | **1.00ms** | **1st** |
| T4 throughput | 4,205 inf/s | 2nd |
| GC pauses | 0 (forward + training) | 1st |
| RSS | 488MB | 4th |

**Strengths**: Fastest training (1.00ms). Column-major BLAS alignment + JIT specialization. Zero GC on both forward and training. Backward optimized via `@.` broadcast replacing `@simd` loops — train alloc dropped 7.2x. Best parallel scaling ratio at T4 (2.88x).

**Weakness**: ~488MB RSS (JIT runtime + LLVM + method cache). True cold start ~300ms+.

### Go — Simplicity as Competitive Advantage

| Metric | Value | Rank |
|--------|-------|------|
| Forward | 1.14ms | 3rd |
| GC throughput | 0.985-0.996 | Best among GC langs |
| RSS | 32MB | 2nd |
| Compilation | ~1s | Fastest |

**Strengths**: GC is invisible (minimal overhead under 29M alloc/fwd). Simple parallelism (goroutine + WaitGroup). Lean runtime. Fastest compilation.

**Weakness**: 3rd on most compute metrics. Go compiler lacks LLVM-level optimization and SIMD. CGO bridge ~1us per BLAS call.

### Python — Ecosystem Over Execution

| Metric | Value | Rank |
|--------|-------|------|
| Forward | 2.37ms | 4th |
| T4 throughput | 1,898 inf/s | 4th |
| Ecosystem | Unmatched | — |

**Strengths**: NumPy → Accelerate with zero FFI effort. ProcessPoolExecutor provides reasonable parallel scaling. Ecosystem (PyTorch/JAX/HuggingFace) for production.

**Weakness**: CPython interpreter 8-10x slower than Rust on non-BLAS code. ~2ms fixed overhead per call.

</details>

<details>
<summary>Optimization Journey</summary>

| Language | Before | After | Speedup | Key optimization |
|----------|--------|-------|---------|-----------------|
| **Julia forward** | 231 ms | **0.61 ms** | **379x** | Type stability (function barriers), column-major stride fix, router alloc elimination |
| **Julia backward** | 7.36 ms train | **1.00 ms** train | **7.4x** | `@.` broadcast replacing `@simd` loops (Julia 1.12 `@simd` = 9K-11K micro-alloc/loop). Train alloc: 21.5M → 3.0M (7.2x) |
| **Python T4** | 26 inf/s | **1,898 inf/s** | **73x** | ProcessPoolExecutor pre-created pool + initializer (was measuring fork overhead) |
| **Rust forward** | 1.43 ms | **0.57 ms** | **2.5x** | Per-token → batched expert dispatch, `silu_in_place`, `add_in_place`, inference mode |

</details>

## Scaling Behavior

| Scale | Forward Spread | Train Spread | Status |
|-------|---------------|-------------|--------|
| hidden=64 | **4.2×** | **10.0×** | Measured |
| hidden=256 | **1.9×** | **5.2×** | Measured |

As hidden size grows, BLAS (AMX) share of total compute increases and language overhead shrinks. At h=256, forward spread already halves from 4.2x to 1.9x.

### How Much GPU Time Does Python Waste?

Our benchmark isolates the overhead gap **outside** BLAS compute — the host-language tax on every training step. All 4 languages call the same `cblas_sgemm`, so the difference is pure language overhead: interpreter dispatch, GC pauses, FFI bridge cost, and memory management.

| Language | Step Time | BLAS | Overhead | GPU Utilization |
|---|---|---|---|---|
| **Julia** | 0.98 ms | 0.96 ms | 0.02 ms (2%) | **98.0%** |
| **Rust** | 1.44 ms | 0.98 ms | 0.46 ms (32%) | **68.1%** |
| **Go** | 3.28 ms | 0.98 ms | 2.30 ms (70%) | **29.9%** |
| **Python** | 10.22 ms | 0.98 ms | 9.24 ms (90%) | **9.6%** |

> h=64, train step, Apple M1 (AMX) measured. BLAS = step time − overhead. GPU utilization = BLAS / step time.
>
> **Note:** Julia's 98% utilization reflects broadcast fusion (`@.`) on M1 AMX — but this is not merely a CPU trick. Julia's advantage is **language-level fusion compilation**: the compiler fuses multiple element-wise operations into a single pass, eliminating intermediate allocations. On GPU, this same capability manifests as [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (XLA backend) — whole-graph compilation and automatic kernel fusion without manual CUDA code. Rust has no equivalent automatic fusion; GPU acceleration requires hand-written CUDA kernels or frameworks like Burn. Rust's 32% overhead here is dominated by per-parameter gradient update allocations (a CPU memory access pattern), and its zero-copy abstraction (`&mut [f32]`, no GC, no runtime) would close the **kernel launch overhead** gap on GPU — but not the **fusion** gap. **These figures are M1 CPU results; treat the absolute numbers as reference, not GPU predictions.**

Python spends **90% of every training step** waiting for the CPython interpreter. The GPU sits idle. At h=64 this means 10.4x slower training than Julia for the same matmul work.

**Switching from Python — what you save at 13T training scale:**

| Switch to | Speedup | GPU-hours saved per $100M | Cost for same work | You save |
|---|---|---|---|---|
| **Julia** | **10.4×** | 90.2M H100-hours | **$9.8M** | **$90.2M** |
| **Rust** | **7.1×** | 85.9M H100-hours | **$14.1M** | **$85.9M** |
| **Go** | **3.1×** | 67.9M H100-hours | **$32.1M** | **$67.9M** |

> Based on measured h=64 M1 CPU overhead ratios — reference estimates, not production GPU predictions. On production GPUs, Rust's kernel launch overhead would match Julia's (both zero-GC, direct memory control), but Julia retains an architectural advantage: Reactant.jl compiles the full compute graph into fused XLA kernels automatically, whereas Rust requires manual kernel fusion or framework support. The Python vs {Rust, Julia, Go} gap is the robust signal; the Rust vs Julia gap is M1-specific and would narrow (but not necessarily close) on GPU.

In concrete terms: a GPT-4-class run (13T tokens, ~$100M) wastes **~$90M on the CPython interpreter** — enough to fund the entire training run again in Julia or Rust.

**"But torch.compile fixes this"**

Yes — and that's exactly the point. The Python ML ecosystem has spent enormous engineering effort building layers to **work around** the language:
- **CUDA Graphs** — bypass Python by replaying GPU command buffers directly
- **torch.compile / XLA** — trace the compute graph to eliminate per-op Python dispatch
- **C++ DataLoaders** — hide the GIL behind multiprocessing
- **NCCL** — GPU-direct all-reduce so Python never touches gradient data

These are not features of Python. They are **escape hatches from Python**. Every one of these optimizations exists because CPython is too slow to be in the critical path. The $90M waste is real — it's just that PyTorch has learned to avoid paying most of it by doing the real work in C++/CUDA.

The question this benchmark answers: what if the host language were fast enough that you didn't need escape hatches?

<details>
<summary>Measured h=256 data</summary>

| Language | Forward h=64 | Forward h=256 | Train h=64 | Train h=256 |
|----------|-------------|--------------|-----------|------------|
| **Rust** | 0.56 ms | 2.91 ms | 1.44 ms | 16.90 ms |
| **Julia** | 0.59 ms | 3.47 ms | 0.98 ms | 9.58 ms |
| **Go** | 1.14 ms | 5.26 ms | 3.28 ms | 38.33 ms |
| **Python** | 2.53 ms | 5.11 ms | 10.22 ms | 49.07 ms |

</details>

## Loss Convergence

| Language | Initial Loss | Final Loss | Reduction |
|----------|-------------|------------|-----------|
| **Julia** | 7.25 | **0.021** | **-99.7%** |
| **Go** | 8.47 | **0.022** | **-99.7%** |
| **Python** | 7.34 | **0.023** | **-99.7%** |
| **Rust** | 7.82 | **0.024** | **-99.7%** |

All 4 implementations converge to ~0.02 (from ~7.5), confirming gradient correctness. Run `make convergence` to verify.

## Verification Environment

| Item | Value |
|------|-------|
| Hardware | MacBook Air M1 (8-core: 4 perf + 4 eff, 16GB unified memory) |
| OS | macOS (Darwin 25.2.0) |
| Rust | rustc 1.91.1 (2024 Edition) |
| Go | go1.25.6 |
| Python | 3.13.5 + NumPy (Accelerate-linked) |
| Julia | 1.12.4 + AppleAccelerate.jl |
| BLAS | Apple Accelerate (AMX, ~1.49 TFLOPS f32 theoretical peak) |
| Methodology | N=10 trials, 3 warmup, median reported, fixed seed (42) |

## Known Limitations

- **Small model size**: hidden=64 amplifies per-call overhead. At hidden=256, BLAS share already grows significantly (see [Scaling Behavior](#scaling-behavior) above).
- **Single hardware**: Apple M1 only. Results may differ on x86 (no AMX), NVIDIA GPU, or different Apple Silicon generations.
- **CPU only**: No GPU benchmarks. At scale, GPU compute (A100: ~312 TFLOPS f32) dwarfs CPU (~1.5 TFLOPS).
- **alloc_bytes not comparable**: Each language measures allocation differently. Use `peak_rss_bytes` for cross-language comparison.

## Project Structure

```text
rosetta-moe/
├── rust/                 # Rust: LLVM AOT + NEON SIMD + Accelerate FFI
│   └── src/bin/convergence.rs  # Loss convergence verification (Rust)
├── go/                   # Go: CGO Accelerate bridge + goroutines
│   └── nn_test.go              # TestConvergence (Go convergence verification)
├── python/               # Python: NumPy → Accelerate + ProcessPoolExecutor
├── julia/                # Julia: LBT → AppleAccelerate.jl + @fastmath SIMD
├── benchmarks/           # Raw JSON outputs (4 languages)
├── scripts/
│   ├── summary.py                # Benchmark summary table generator
│   ├── convergence_python.py     # Loss convergence verification (Python)
│   └── convergence_julia.jl      # Loss convergence verification (Julia)
├── docs/
│   ├── spec.md           # Spec: requirements, evaluation matrix, acceptance criteria
│   ├── bench-results.md  # Results: methodology, 5-axis data, per-language deep analysis
│   └── analysis-parallel-training.md  # AMX architecture analysis of parallel training scaling
├── Makefile              # make test / make bench / make convergence / make verify
└── Cargo.toml
```

Each language directory contains its own README with architecture overview, 21-entry equation-to-code map, implementation notes, performance characteristics, and gotchas.

## License

Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [LICENSE](LICENSE) for details.
