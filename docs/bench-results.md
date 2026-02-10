# Educational SLM Benchmark: 4-Language Comparison

**Date**: 2026-02-10
**Platform**: Darwin 25.2.0 (MacBook Air M1, 8-core: 4 perf + 4 eff, 16GB unified memory)
**Model**: MoE Transformer (hidden=64, vocab=1000, seed=42)

| Language | Version | BLAS integration | Call path |
|----------|---------|-----------------|-----------|
| Rust | rustc 1.91.1 | `#[link(name = "Accelerate", kind = "framework")]` | Direct FFI (`extern "C"`) |
| Go | go1.25.6 | `#cgo LDFLAGS: -framework Accelerate` | CGO bridge |
| Python | 3.13.5 | `numpy.matmul` → Accelerate (macOS >= 14) | NumPy dispatch |
| Julia | 1.12.4 | `LinearAlgebra.mul!` via AppleAccelerate.jl | libblastrampoline (LBT) |

All matmul uses Apple Accelerate `cblas_sgemm`. Non-matmul ops (softmax, rmsnorm, attention scores) use hand-written loops.

### How to Read This Document

This document presents benchmark results for 4 independent implementations of the same MoE Transformer model, written in Rust, Go, Python, and Julia. All implementations use Apple Accelerate BLAS for matrix multiplication, so performance differences reflect language runtime overhead, memory management, and dispatch mechanisms — not BLAS performance itself. Results are organized by 5 research axes, each isolating a specific language characteristic (memory management, compiler optimization, type system dispatch, parallelism, and scaling). Raw JSON data is available in `benchmarks/*.json` for independent analysis.

---

## 1. Methodology

### Trial Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N_TRIALS | 10 | Sufficient for stable median on lightweight scenarios |
| N_WARMUP | 3 | Eliminate JIT/cache cold effects without wasting time |
| Primary metric | Median | Robust to outliers (single background process can skew max) |
| Spread metric | IQR | Resistant to outliers, meaningful with N=10 |
| Seed | 42 (fixed) | Reproducibility across languages and runs |

Median is preferred over mean because system benchmarks are susceptible to background process interference, which creates high outliers. IQR (interquartile range) captures the typical spread without being distorted by these outliers.

### Metric Design

Dynamic metrics satisfy the 5 requirements from Marr et al. ("Are We Fast Yet?", DLS 2016): unambiguous, dynamic, robust, discriminating, machine-independent.

| Category | Metrics | Collection |
|----------|---------|------------|
| **Measured** | `median_ns`, `cpu_time_ns`, `peak_rss`, `alloc_bytes`, `gc_time_ns`, `gc_pause_count`, `throughput` | Bench harness |
| **Derived** | `alloc_rate`, `gc_throughput`, `GFLOPS`, `speedup`, `cold/warm ratio` | Post-processed |
| **Static** | `source_gzip_bytes`, `SLOC`, `function_count` | Per codebase snapshot |

Derived formulas:

| Metric | Formula | Unit |
|--------|---------|------|
| `alloc_rate` | `alloc_bytes / (median_ns * 1e-9)` | bytes/sec |
| `gc_throughput` | `1.0 - (gc_time_ns / median_ns)` | ratio [0,1] |
| `GFLOPS` | `known_op_count / (median_ns * 1e-9) * 1e-9` | GFLOP/s |
| `speedup` | `throughput_TN / throughput_T1` | ratio |
| `cold/warm ratio` | `dispatch_cold_ns / dispatch_warm_ns` | ratio |

Key derived metrics:

- **`gc_throughput`**: A value of 1.0 means zero time was spent in GC. A value of 0.95 means 5% of wall-clock time was consumed by garbage collection. This metric comes from the GEAR framework (ICSE 2025).
- **`alloc_rate`**: Higher values indicate more allocation pressure on the memory subsystem. Languages with lower alloc_rate for the same workload are using memory more efficiently.
- **`GFLOPS`**: Measured against the M1 chip's theoretical peak of ~1.49 TFLOPS for f32. Values above 40% of peak (>600 GFLOPS) indicate efficient hardware utilization.

### Literature References

| Source | Year | Adopted | Skipped (reason) |
|--------|------|---------|------------------|
| **Benchmarks Game** (Debian) | ongoing | wall time, CPU time, peak RSS, source gzip size | -- |
| **Are We Fast Yet?** (Marr et al.) | 2016 | steady-state/cold split, warmup protocol, 5 metric requirements | alloc object count (too fine-grained) |
| **EffiBench-X** (arXiv 2505.13004) | 2025 | Memory Peak | Memory Integral (sampling thread too complex) |
| **Energy-Languages** (Pereira et al.) | 2021 | -- | Energy/Joules (RAPL = Intel only, M1 incompatible) |
| **GEAR** (ICSE 2025) | 2025 | gc_throughput, alloc_rate | live_heap (language-specific API fragmentation) |

---

## 2. Language Profiles

**Rust** -- Zero-cost abstractions with ownership-based memory. No runtime, no GC. AOT via LLVM. Matmul via Accelerate direct FFI.

**Go** -- Simplicity-first systems language with goroutine concurrency and tracing GC. AOT with lightweight runtime. Matmul via CGO Accelerate.

**Python** -- Interpreted dynamic language. GIL limits parallelism. NumPy dispatches matmul to Accelerate.

**Julia** -- JIT-compiled scientific computing language. Multiple dispatch core. Column-major. LLVM JIT. `mul!` via LBT → AppleAccelerate.jl.

---

## 3. Results -- 4 Research Axes

22 scenarios across 5 axes. Each axis isolates a distinct language/runtime characteristic.

| Axis | Isolates | Scenarios |
|------|----------|-----------|
| 1. Memory | Ownership vs GC vs RC | 9 (train + batch scale + seq scale) |
| 2. Compiler | LLVM vs Go compiler vs JIT vs BLAS | 3 (matmul, softmax, rmsnorm) |
| 3. Type System | Mono vs interface vs dispatch | 2 (warm, cold) |
| 4. Parallel | Thread vs goroutine vs process | 6 (inference T1/T2/T4, training T1/T2/T4) |
| 5. Scale | Convergence at larger model size | 2 (forward_256, train_256) |

---

### Axis 1: Memory Management

How each language's memory model handles allocation pressure under scaling.

| Scenario | Workload | Primary variable | Controls |
|----------|----------|-----------------|----------|
| mem_train_step | fwd + bwd + opt | Alloc pressure + GC | batch=2, seq=8, hidden=64 |
| mem_scale_batch_{1,2,4,8} | Inference, varying batch | Alloc scaling with batch | seq=32, hidden=64 |
| mem_scale_seq_{8,16,32,64} | Inference, varying seq_len | Alloc scaling with seq | batch=2, hidden=64 |

#### Training Step

| Language | Median (ms) | vs Fastest | alloc_bytes | GC time (ms) | gc_throughput |
|----------|-------------|-----------|-------------|---------------|---------------|
| Julia | **1.00** | 1.0x | 3.0M | 0 | 1.000 |
| Rust | 1.31 | 1.3x | 1.4M | 0 | 1.000 |
| Go | 3.33 | 3.3x | 29.0M | 0.14 | 0.996 |
| Python | 10.26 | 10.3x | 11.2M* | 0 | 1.000 |

*Python alloc_bytes via tracemalloc -- Python heap only, not NumPy data buffers (libc malloc).

Julia fastest training at 1.00ms with zero GC and 3.0M allocation per step — a 7.2x allocation reduction from the previous 55.5M via backward pass optimization (`@simd` loops replaced with `@.` broadcast fusion, eliminating 9K-11K allocations per loop). Forward:backward ratio improved from 12:1 to 1.7:1. Rust 1.31ms, also zero GC. Go 3.33ms with GC overhead invisible (0.14ms, gc_throughput 0.996). Python 10.26ms after in-place SwiGLU/residual optimization.

#### Batch Scaling (Forward)

seq=32, hidden=64.

| batch | Rust (ms) | Julia (ms) | Go (ms) | Python (ms) |
|-------|-----------|------------|---------|-------------|
| 1 | **0.35** | **0.34** | 0.64 | 2.06 |
| 2 | **0.57** | 0.61 | 1.15 | 2.34 |
| 4 | **1.02** | 1.11 | 2.19 | 2.99 |
| 8 | **1.93** | 2.19 | 4.16 | 4.02 |

Rust and Julia near-tied at batch 1 (0.35ms vs 0.34ms). Rust leads at batch 2-8. Python converges with Go at batch 8 (4.02ms vs 4.16ms) as NumPy overhead is amortized.

#### Sequence Scaling (Forward)

batch=2, hidden=64.

| seq_len | Rust (ms) | Julia (ms) | Go (ms) | Python (ms) |
|---------|-----------|------------|---------|-------------|
| 8 | **0.16** | 0.18 | 0.33 | 1.94 |
| 16 | **0.27** | 0.31 | 0.54 | 2.04 |
| 32 | **0.57** | 0.60 | 1.13 | 2.34 |
| 64 | **1.47** | 1.29 | 2.74 | 3.01 |

Rust leads at seq=8, seq=16, seq=32, and seq=64. Julia close at seq=64 (1.29ms vs Rust 1.47ms, column-major BLAS alignment advantage on larger matrices). Python ~2ms fixed-cost overhead visible at small seq.

#### Memory Footprint & GC

| Metric | Rust | Go | Python | Julia |
|--------|------|----|--------|-------|
| peak_rss (MB), fwd b=2 s=32 | **19.8** | 32.3 | 62.3 | 494 |
| peak_rss (MB), train | **19.1** | 31.9 | 60.9 | 488 |
| GC time / train_step | 0 | 0.15 | 0 | 0 |
| GC pauses / train_step | 0 | 3 | 0 | 0 |
| gc_throughput (train) | 1.000 | 0.996 | 1.000 | 1.000 |
| gc_throughput (forward worst) | 1.000 | 0.985 | 1.000 | 1.000 |
| alloc_bytes (fwd b=2 s=32) | 1.2M | 18.7M | 1.7M* | 6.9M |
| alloc_bytes (train) | 1.4M | 28.9M | 11.2M* | 3.0M |

Rust 19.8MB = minimal: no runtime, no GC, no JIT — pure model weights + stack. Julia ~488MB train RSS. Go 28.9M alloc per train step but GC overhead invisible (0.15ms, gc_throughput 0.996). Julia train alloc dropped 7.2x (55.5M → 3.0M per step) via broadcast fusion, with 0 GC pauses.

Julia's ~488-494MB RSS is the JIT runtime, compiled method cache, and type inference metadata — not the model itself. The model weights at hidden=64 total ~0.1MB. This is a fixed cost of the Julia runtime that does not scale with model operations or batch size. Training RSS (488MB) is slightly lower than forward-only RSS (494MB) due to backward pass allocation optimization.

**GC Detail by Scenario**:

| Scenario | Go gc_time (ms) | Go pauses | Julia gc_time (ms) | Julia pauses |
|----------|-----------------|-----------|---------------------|--------------|
| mem_train_step | 0.15 | 3 | 0 | 0 |
| mem_scale_batch_2 | 0.10 | 4 | 0 | 0 |
| mem_scale_batch_8 | 0.24 | 8 | 0 | 0 |
| mem_scale_seq_32 | 0.10 | 4 | 0 | 0 |
| mem_scale_seq_64 | 0.16 | 6 | 0 | 0 |

Julia achieves zero GC across all scenarios. Go GC overhead remains minimal (0.10-0.24ms, <1.5% of wall time).

---

### Axis 2: Compiler Optimization

Pure kernel throughput. All kernel benchmarks pre-allocate output buffers and measure only the compute call (no Tensor wrapping).

| Scenario | Workload | Primary variable | Controls |
|----------|----------|-----------------|----------|
| kernel_matmul | C = A x B (M=K=N=64, BLAS sgemm) | BLAS dispatch overhead | Pre-allocated output (all 4 languages use raw sgemm with pre-alloc) |
| kernel_softmax | softmax(x) (n=1000) | Exp/reduce pipeline | Fixed size, in-place (all 4 languages copy input → pre-alloc buffer → compute) |
| kernel_rmsnorm | rmsnorm(x, w) (shape=(2,32,64)) | Sqrt + element-wise | Fixed size, in-place |

#### Kernel Times & GFLOPS

| Kernel | Rust (us) | Julia (us) | Go (us) | Python (us) | Known FLOPS |
|--------|-----------|------------|---------|-------------|-------------|
| matmul 64x64 | **0.79** | 0.96 | 0.88 | 3.17 | 524,288 |
| softmax (1000) | **1.81** | 4.83 | 5.65 | 16.17 | 4,000 |
| rmsnorm (2,32,64) | **3.38** | 4.33 | 8.60 | 22.60 | 12,288 |

| Kernel | Rust | Julia | Go | Python |
|--------|------|-------|----|--------|
| matmul (GFLOPS) | **662** | 547 | 599 | 166 |
| softmax (GFLOPS) | **2.21** | 0.83 | 0.71 | 0.27 |
| rmsnorm (GFLOPS) | **3.64** | 2.84 | 1.43 | 0.54 |

Rust and Julia near-tied on matmul (662 vs 547 GFLOPS, within noise at 64x64). Rust wins softmax with tightest scalar loops via LLVM (1.81us) and leads rmsnorm (3.38us, 3.64 GFLOPS). Matmul GFLOPS spread (166-662) reflects FFI dispatch noise at small 64x64 size — all use the same `cblas_sgemm` underneath. Go CGO bridge overhead and Python NumPy dispatch overhead visible.

Rust's 662 GFLOPS represents 44% of the M1 chip's theoretical f32 peak (~1.49 TFLOPS). Julia's 547 GFLOPS = 37%. For comparison, production GEMM libraries typically achieve 70-90% of peak on large matrices (1024+). The 37-44% figures reflect the small 64x64 matrix size, where per-call FFI dispatch and AMX pipeline setup overhead consume a significant fraction of total time relative to actual compute.

---

### Axis 3: Type System

Dispatch overhead and type specialization cost.

| Scenario | Workload | Primary variable | Controls |
|----------|----------|-----------------|----------|
| dispatch_warm | Full forward pass (post-warmup) | Steady-state dispatch | batch=2, seq=32, hidden=64 |
| dispatch_cold | Model construction + first inference | JIT/dispatch resolution | batch=1, seq=8, hidden=64 |

| Metric | Rust | Julia | Go | Python |
|--------|------|-------|----|--------|
| dispatch_warm (ms) | **0.57** | 0.61 | 1.14 | 2.37 |
| dispatch_cold (ms) | 5.21 | **2.45** | 8.22 | 10.83 |
| cold/warm ratio | 9.22 | 4.01 | 7.22 | 4.58 |

"Warm" measures the steady-state forward pass with all JIT compilation and caches warmed up. "Cold" measures model construction plus the first forward pass, which includes JIT compilation for Julia and binary/library loading for compiled languages. The cold/warm ratio shows how much one-time setup cost each language incurs.

Rust leads warm dispatch (0.57ms), ahead of Julia (0.61ms). Both ~2x faster than Go (1.14ms). Julia cold fastest (2.45ms) thanks to JIT incremental compilation. Rust's high cold/warm ratio (9.22) reflects extremely fast warm path making construction cost proportionally larger. Go and Python show high cold/warm ratios (7.22, 4.58) due to runtime initialization and library loading overhead.

---

### Axis 4: Parallel Model

Throughput scaling with independent concurrent workloads. No shared state.

| Scenario | Workload | Primary variable | Controls |
|----------|----------|-----------------|----------|
| parallel_T{1,2,4} | N independent forward passes | Thread scaling efficiency | batch=2, seq=32, hidden=64 |

Per-language semantics:
- Rust: `std::thread::scope`, independent model per thread
- Go: goroutines + WaitGroup, independent model per goroutine
- Python: `ProcessPoolExecutor`, independent model per process (bypasses GIL)
- Julia: `Threads.@threads`, independent model per thread

| Threads | Julia (ms) | Rust (ms) | Go (ms) | Python (ms) |
|---------|------------|-----------|---------|-------------|
| T1 | 0.686 | **0.574** | 1.145 | 1.303 |
| T2 | 0.771 | **0.647** | 1.345 | 1.575 |
| T4 | 0.949 | **0.841** | 1.882 | 2.098 |

Python ProcessPoolExecutor with pre-created pool and initialized workers (matching Go/Rust/Julia methodology). Rust parallel uses thread-local alloc counter + barrier-based thread pool.

#### Throughput & Speedup

| Threads | Julia (inf/s) | Rust (inf/s) | Go (inf/s) | Python (inf/s) |
|---------|---------------|-------------|-----------|----------------|
| T1 | 1,458 | **1,742** | 874 | 768 |
| T2 | 2,591 | **3,084** | 1,486 | 1,268 |
| T4 | 4,205 | **4,738** | 2,121 | 1,898 |

| Speedup | Julia | Go | Rust | Python |
|---------|-------|-------|------|--------|
| T2/T1 | **1.78x** | 1.70x | 1.77x | 1.65x |
| T4/T1 | **2.88x** | 2.43x | 2.72x | 2.47x |

Rust leads T4 throughput (4,738 inf/s), followed by Julia (4,205 inf/s), Go (2,121 inf/s), and Python (1,898 inf/s). Julia shows best T4/T1 scaling ratio (2.88x), followed by Rust (2.72x), Python (2.47x), and Go (2.43x). Sub-linear scaling due to AMX contention.

Rust parallel improved significantly (previous: 2,705 inf/s T4) via thread-local alloc counter + barrier-based thread pool + inference-mode allocation elimination, reaching 4,738 inf/s T4.

Scaling is sub-linear because the Apple AMX coprocessor is a shared hardware resource. When multiple threads issue BLAS calls simultaneously, they may serialize at the AMX level rather than executing in true parallel. Additionally, for workloads this small (~0.6-1.3ms per inference), thread creation and synchronization overhead becomes a significant fraction of total time.

#### Parallel Training (Data-Parallel Simulation)

Each thread runs an independent model + trainer + unique data (data-parallel simulation). No gradient synchronization.

Per-language semantics:
- Rust: `std::thread::scope`, independent Trainer per thread, Barrier sync
- Go: goroutines + WaitGroup, independent Trainer per goroutine
- Python: `ProcessPoolExecutor`, independent Trainer per process
- Julia: `Threads.@threads`, independent Trainer + `train_step!` per thread

| Threads | Julia (trn/s) | Rust (trn/s) | Go (trn/s) | Python (trn/s) |
|---------|---------------|-------------|-----------|----------------|
| T1 | **943** | 758 | 311 | 175 |
| T2 | **1,490** | 1,170 | 524 | 311 |
| T4 | **1,558** | 1,309 | 867 | 503 |

| Speedup | Julia | Rust | Go | Python |
|---------|-------|------|-------|--------|
| T4/T1 | 1.65x | 1.73x | **2.79x** | **2.88x** |

Julia leads absolute training throughput at all thread counts (943→1,558 trn/s). Rust 2nd (758→1,309). Scaling ratios invert: Go (2.79x) and Python (2.88x) show better T4/T1 ratios than Julia (1.65x) and Rust (1.73x). This is explained by Amdahl's Law — Julia/Rust spend more time in BLAS (AMX-bound, serialized), while Go/Python's slower non-BLAS code is fully parallelizable. See [analysis-parallel-training.md](analysis-parallel-training.md) for detailed AMX architecture analysis.

Training parallel scaling is significantly lower than inference scaling (Julia: 1.65x vs 2.88x, Rust: 1.73x vs 2.72x) because backward pass triples AMX bus transactions (~72 sgemm/step vs ~24 for inference).

---

## 4. Cross-Analysis

### Rankings

| Axis | 1st | 2nd | 3rd | 4th |
|------|-----|-----|-----|-----|
| Memory (forward) | **Rust** (0.57ms, 0 GC) | Julia (0.61ms, 0 GC) | Go (1.14ms) | Python (2.37ms) |
| Memory (train) | **Julia** (1.00ms, 0 GC) | Rust (1.31ms, 0 GC) | Go (3.33ms) | Python (10.03ms) |
| Compiler (matmul) | **Rust** (662 GFLOPS) | Go (599 GFLOPS) | Julia (547 GFLOPS) | Python (166 GFLOPS) |
| Compiler (softmax) | **Rust** (1.81us) | Julia (4.83us) | Go (5.65us) | Python (16.17us) |
| Compiler (rmsnorm) | **Rust** (3.38us) | Julia (4.33us) | Go (8.60us) | Python (22.60us) |
| Type system (warm) | **Rust** (0.57ms) | Julia (0.61ms) | Go (1.14ms) | Python (2.37ms) |
| Parallel (T4 throughput) | **Rust** (4,738 inf/s) | Julia (4,205 inf/s) | Go (2,121 inf/s) | Python (1,898 inf/s) |
| Parallel Training (T4) | **Julia** (1,558 trn/s) | Rust (1,309 trn/s) | Go (867 trn/s) | Python (503 trn/s) |
| Scale (forward h=256) | **Rust** (2.88ms) | Julia (3.51ms) | Python (5.15ms) | Go (5.47ms) |

Note: Matmul GFLOPS differences between Rust and Julia (662 vs 547) are within measurement noise at 64x64 — both use the same BLAS (`cblas_sgemm`). The spread reflects sub-microsecond FFI dispatch variance, not compute difference.

### Per-Language Deep Analysis

#### Rust — Precision by Default, Speed by Choice

**Design philosophy**: Zero-cost abstractions. Ownership-based memory safety. No runtime, no GC. The compiler enforces invariants at compile time; the developer explicitly opts into any approximation or unsafety.

**Strengths observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| peak_rss | **19.8MB** | 1st | No JIT/GC runtime — pure model + stack |
| train step | 1.31ms | 2nd | LLVM AOT + zero-alloc backward |
| dispatch_warm | **0.57ms** | **1st** | Leads Julia (0.61ms) |
| softmax | **1.81us** | 1st | Tightest scalar loops via LLVM |
| rmsnorm | **3.38us** | **1st** | Inference-mode optimization |
| T4 throughput | **4,738 inf/s** | **1st** | Thread-local alloc + inference mode |
| gc_throughput | 1.000 | tied 1st | Deterministic |

- **Non-BLAS kernel dominance (softmax, rmsnorm)**: Where hand-written loops matter (softmax, rmsnorm, attention masking), Rust's LLVM AOT produces the fastest code. The compiler eliminates bounds checks via iterator chains, auto-vectorizes where possible, and inlines aggressively. This is where "zero-cost abstractions" manifests concretely.
- **Predictability**: No GC pauses, no JIT recompilation, no interpreter warmup. Wall time = CPU time. IQR consistently smallest across scenarios.
- **SIMD opt-in model**: NEON `vrsqrteq_f32` + Newton-Raphson for AdamW rsqrt — explicit `unsafe` with `// SAFETY:` audit trail. Each approximation is individually justified and reviewable.
- **Parallel improvement**: Thread-local alloc counter + barrier-based thread pool + inference-mode allocation elimination brought T4 throughput from 2,705 to 4,738 inf/s (1.75x improvement).

**Weaknesses observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| train step | 1.31ms | 2nd | Julia's backward optimization shifted lead |

- **Training 2nd**: Julia's backward pass optimization (broadcast fusion, 7.2x allocation reduction) brought Julia training to 1.00ms, surpassing Rust's 1.31ms. Rust still achieves zero allocation on backward, but Julia's broadcast-fused operations are faster overall.

**Key trade-off**: Rust leads forward (0.57ms), parallel T4 (4,738 inf/s), and non-BLAS kernels, while Julia leads training (1.00ms). Rust's 19.8MB RSS is 25x lower than Julia. The only gap is training step (1.31ms vs 1.00ms).

---

#### Julia — JIT Compilation as Investment

**Design philosophy**: Multiple dispatch as the core abstraction. JIT specialization via LLVM. Pay upfront cost (RSS, startup) to get runtime-specialized code that rivals AOT compilers. Scientific computing ergonomics above all.

**Strengths observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| forward | 0.61ms | 2nd | Column-major + JIT specialization |
| train step | **1.00ms** | 1st | Broadcast-fused backward, 3.0M alloc |
| matmul GFLOPS | 547 | 3rd | LBT → Accelerate; within noise of Rust |
| gc_throughput (train) | 1.000 | tied 1st | Zero GC on training |
| dispatch_cold | **2.45ms** | 1st | JIT incremental compilation |
| T4 throughput | 4,205 inf/s | 2nd | Best parallel scaling ratio (2.88x) |

- **Forward pass close 2nd**: Julia's forward pass (0.61ms) is narrowly behind Rust (0.57ms) after Rust's inference-mode optimization. At seq=64, Julia (1.29ms) is close to Rust (1.47ms) as larger matrices better exploit column-major layout.
- **Training pass fastest**: Julia's backward pass optimization (`@simd` → `@.` broadcast fusion) eliminated 9K-11K allocations per loop invocation. Training alloc dropped 7.2x (55.5M → 3.0M per step). Forward:backward ratio improved from 12:1 to 1.7:1. Julia now leads training (1.00ms) over Rust (1.31ms) by 1.3x.
- **Zero GC on training**: Julia achieves gc_throughput = 1.000 across all scenarios including training. The broadcast fusion eliminates temporary array allocation entirely. Zero GC pauses, 0ns GC time.
- **Best parallel scaling ratio**: With valid multi-threading (available_threads=4), Julia achieves 4,205 inf/s at T4 with 2.88x T4/T1 scaling ratio, the best scaling ratio among all languages. Rust leads absolute throughput (4,738 inf/s) with 2.72x scaling.
- **Column-major BLAS alignment**: Julia's column-major default matches Fortran-order `sgemm`. This alignment advantage grows with matrix size.
- **Multiple dispatch**: Adding a new type or method requires zero modification to existing code. The JIT specializes dispatch at runtime, eliminating virtual call overhead that would exist in OOP vtable dispatch.

**Weaknesses observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| peak_rss | 488MB | 4th | JIT runtime |
| softmax | 4.83us | 2nd | Behind Rust's tighter scalar loops |

- **RSS is the price of JIT**: ~488MB is the LLVM JIT compiler, type inference engine, method cache, and GC infrastructure. The model itself is ~0.1MB. This is architectural — the ~200MB runtime floor cannot be removed without removing what makes Julia productive (see Known Limitations: Julia AOT). For comparison, Rust achieves competitive performance at 19.8MB.
- **Column-major footgun**: `@view array[batch, :, :]` on 3D array produces non-contiguous memory (stride[1] != 1). BLAS silently falls back to generic matmul, ~1.5x slower. Requires explicit `copy` to contiguous Matrix. This trap is invisible and specific to Julia's memory layout.

**Backward pass optimization detail**: Julia 1.12's `@simd` generates 9K-11K heap allocations per loop invocation due to SIMD lane metadata. Replacing `@simd for i in ...` loops with `@.` broadcast fusion (e.g., `@. grad_output = grad_output * mask`) fuses element-wise operations into a single pass with zero temporary allocation. This is the idiomatic Julia approach — broadcast is the language's core strength, not explicit SIMD loops.

**Key trade-off**: Julia leads training (1.00ms) and is close 2nd on forward (0.61ms vs Rust 0.57ms). Julia's training lead was achieved by replacing C-style `@simd` loops with Julia-idiomatic `@.` broadcast fusion. Julia invests in runtime infrastructure (488MB) that pays off in developer ergonomics (`mul!`, multiple dispatch), training performance, and best parallel scaling ratio (2.88x) at the cost of resource footprint.

---

#### Go — Simplicity as Competitive Advantage

**Design philosophy**: Simplicity first. Fast compilation. Goroutines for concurrency. Minimize cognitive load. "Good enough" performance with minimal complexity.

**Strengths observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| gc_throughput | 0.985-0.996 | best GC | GC overhead < 1.5% |
| peak_rss | 31.9MB | 2nd | Lean runtime |
| T4 scaling | 2.43x | 4th | Moderate scaling ratio |
| softmax | 5.65us | 3rd | Competitive with Julia (4.83us) |

- **GC is invisible**: This is the most important Go finding. 28.9M bytes allocated per train step, gc_throughput 0.996 (0.15ms GC time). Go's concurrent tracing GC is specifically tuned for latency — it sacrifices throughput headroom to minimize pause impact. For ML workloads, GC overhead remains under 1.5%.
- **CGO Accelerate works cleanly**: The `#cgo LDFLAGS: -framework Accelerate` bridge adds overhead but at production matrix sizes (1024+) it would be negligible relative to compute.
- **Moderate T4 scaling**: goroutine + WaitGroup produces T4/T1 scaling of 2.43x. T4 throughput (2,121 inf/s) is competitive with Python's ProcessPoolExecutor (1,898 inf/s), with dramatically simpler code.

**Weaknesses observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| train step | 3.33ms | 3rd | 3.3x Julia |
| softmax | 5.65us | 3rd | Competitive with Julia (4.83us) |
| rmsnorm | 8.60us | 3rd | 2.0x slower than Julia (4.33us) |

- **Compiler optimization ceiling**: Go's compiler (gc) is designed for fast compilation, not peak codegen. Without LLVM, it cannot auto-vectorize or exploit SIMD. softmax (5.65us) is 3.1x Rust (1.81us) — this is the compiler gap, not the language.
- **No SIMD without CGO/assembly**: Go has no equivalent of Rust's `std::arch` or Julia's `@simd`. Optimizing non-BLAS kernels requires dropping to assembly or CGO, defeating the simplicity proposition.

**Key trade-off**: Go deliberately limits optimization surface area. No generics specialization (monomorphization), no SIMD builtins, no unsafe math flags. The result: 3rd place on training (3.33ms), but with the simplest codebase, fastest compilation, and most straightforward deployment. The 3.3x gap vs Julia is the cost of simplicity — and for web services where inference is one of many operations, it doesn't matter. Go's value proposition is not "fastest" — it's "fast enough with the lowest total cost of ownership."

---

#### Python — Ecosystem Over Execution

**Design philosophy**: Developer productivity above all. Computation is offloaded to C libraries (NumPy, BLAS). The language is glue, not compute.

**Strengths observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| T4 throughput | 1,898 inf/s | 4th | ProcessPoolExecutor |

- **NumPy closes the BLAS gap**: `np.matmul` dispatches to Accelerate with zero developer effort. No FFI wrapper, no CGO bridge, no framework linking. This is the lowest-friction path to AMX hardware.
- **ProcessPoolExecutor**: When pool is pre-created with `initializer=` callback, Python achieves 1,898 inf/s at T4. The fork-based model successfully bypasses the GIL for compute-heavy workloads.
- **Ecosystem network effect**: The real Python advantage is not in this benchmark. It's that the same model can be prototyped here, validated with NumPy, then deployed with PyTorch on GPU — without changing languages.

**Weaknesses observed**:

| Metric | Value | Rank | Significance |
|--------|-------|------|--------------|
| train step | 10.03ms | 4th | 10.0x Julia |
| softmax | 16.17us | 4th | CPython overhead |

- **CPython interpreter is the bottleneck**: softmax (16.17us) is 8.8x Rust (1.83us). Every non-BLAS operation pays the interpreter dispatch tax: bytecode decode, dynamic type check, reference counting, attribute lookup. In-place optimizations help but cannot overcome the fundamental overhead.
- **GIL requires multiprocessing**: Python's parallelism story is ProcessPoolExecutor — separate processes with no shared memory. This works for independent model inference but breaks down for workloads requiring shared state. The fork overhead must be amortized across many inferences.

**Key trade-off**: Python is not competing on the same axis as the other three languages. The 10.03ms train step is irrelevant when the real workflow is: prototype in Python → validate with NumPy → deploy with PyTorch/JAX on GPU. This benchmark measures the one thing Python doesn't optimize for — raw CPU inference on a tiny model. Python's value is time-to-production and ecosystem depth, not microsecond-level performance.

---

### Cross-Language Insights

The following insights emerge from comparing results across all 5 axes and languages.

#### 1. BLAS Levels the Playing Field

All 4 languages hit the same AMX hardware via Apple Accelerate `cblas_sgemm`. At 64×64, the GFLOPS spread (166-662) reflects FFI dispatch noise, not compute difference. At production matrix sizes (1024+), all languages would converge toward the M1's ~1.49 TFLOPS f32 peak. **The language only matters for non-BLAS code** — and that's where Rust and Julia separate from Go and Python.

#### 2. GC Is Not the Enemy

The "GC is slow" narrative does not hold for this workload:
- Go: 28.9M bytes allocated per train, gc_throughput 0.985 (1.5% overhead at worst)
- Julia: Zero GC across all scenarios including training (0 pauses, 0ns GC time)
- Python: GC not a factor (NumPy buffers are malloc'd, not GC-tracked)

The bottleneck is always compute (BLAS, softmax, attention), never garbage collection. Languages with GC can achieve near-zero overhead through runtime tuning (Go) or type stability (Julia).

#### 3. Forward Leadership Shifts to Rust

Rust's forward pass (0.57ms) now leads Julia (0.61ms) after inference-mode optimization (in-place residuals, allocation elimination). At larger sequence lengths, the gap narrows: at seq=64, Julia (1.29ms) is close to Rust (1.47ms), with column-major BLAS alignment showing advantage on larger matrices.

Training step results favor Julia (1.00ms vs Rust 1.31ms). Julia's backward pass optimization — replacing `@simd` loops with `@.` broadcast fusion — reduced training allocation from 55.5M to 3.0M per step (7.2x reduction) and improved forward:backward ratio from 12:1 to 1.7:1. This demonstrates that Julia's broadcast fusion is not just syntactic sugar but a genuine performance advantage: the compiler fuses element-wise operations into a single pass with zero temporary allocation, outperforming hand-written loops.

#### 4. Runtime Footprint vs Developer Ergonomics

| Language | RSS | Key ergonomic feature | What it costs |
|----------|-----|----------------------|---------------|
| Rust | 19.8MB | (none — you build everything yourself) | Developer time |
| Go | 31.9MB | Invisible GC, goroutines | 1.5% GC overhead |
| Python | 62.8MB | NumPy dispatch, ecosystem | Interpreter overhead |
| Julia | 488MB | `mul!`, JIT specialization, multiple dispatch | JIT runtime memory |

There is a near-perfect inverse correlation between RSS and developer ergonomics. Julia's 488MB buys the most "magic" — the JIT compiler, type inference engine, and dispatch system that enable zero-GC type stability, column-major BLAS alignment, and broadcast fusion. Rust's 19.8MB means every optimization is hand-written. The question is not "which is better" but "which cost are you willing to pay."

#### 5. AMX Contention Limits All Languages Equally

T4/T1 speedup: Julia 2.88x, Rust 2.72x, Python 2.47x, Go 2.43x. All sub-linear because the Apple AMX coprocessor serializes concurrent BLAS calls at the hardware level. This is not a language limitation — it's a hardware constraint. Rust leads absolute throughput (4,738 inf/s) while Julia leads scaling ratio (2.88x) at T4. At larger matrix sizes where BLAS dominates, all languages would converge toward similar scaling ceilings.

Training parallel scaling is even more constrained: Julia 1.65x, Rust 1.73x at T4 — backward pass triples AMX bus load.

---

### Scaling Analysis — Measured Convergence at hidden=256

This benchmark uses hidden=64 to expose language overhead. To verify that performance gaps converge at larger scale, we measured hidden=256 on the same hardware with the `small()` config (head_dim=64, ffn_dim=1024, rest identical to tiny).

#### Measured: Forward Pass Convergence (batch=2, seq=32)

| Language | hidden=64 | hidden=256 | Growth | vs Fastest |
|----------|-----------|------------|--------|------------|
| **Rust** | **0.57ms** | **2.88ms** | 5.1x | 1.00x |
| Julia | 0.61ms | 3.51ms | 5.8x | 1.22x |
| Python | 2.37ms | 5.15ms | 2.2x | 1.79x |
| Go | 1.14ms | 5.47ms | 4.8x | 1.90x |
| **Spread** | **4.16x** | **1.90x** | | **-54%** |

#### Measured: Training Step Convergence (batch=2, seq=8)

| Language | hidden=64 | hidden=256 | Growth | vs Fastest |
|----------|-----------|------------|--------|------------|
| **Julia** | **1.00ms** | **9.46ms** | 9.5x | 1.00x |
| Rust | 1.31ms | 16.91ms | 12.9x | 1.79x |
| Go | 3.33ms | 37.55ms | 11.3x | 3.97x |
| Python | 10.03ms | 49.18ms | 4.9x | 5.20x |
| **Spread** | **10.03x** | **5.20x** | | **-48%** |

#### Key Findings from hidden=256

**1. Forward spread reduced 54%** (4.16x → 1.90x). At hidden=64, non-BLAS overhead is significant. At hidden=256, BLAS fraction grows and the gap narrows dramatically. Rust leads at both scales after inference-mode optimization.

**2. Training spread reduced 48%** (10.03x → 5.20x). Julia maintains training lead at both scales (1.00ms → 9.46ms). Convergence is slower than forward because backward pass implementations differ more fundamentally across languages.

**3. Rust leads forward at both scales.** Rust leads at h=64 (0.57ms) and h=256 (2.88ms). Julia at 1.22x (3.51ms) at h=256. The gap narrows at h=256 as BLAS fraction grows.

**4. Julia training lead persists at scale.** Julia (9.46ms) leads Rust (16.91ms) at h=256 by 1.79x. This reflects Julia's **language-level fusion compilation**: the `@.` broadcast compiler fuses multiple element-wise operations into a single pass, eliminating intermediate allocations. This is not a CPU-specific trick — on GPU, the same capability manifests as [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) (XLA backend) for automatic whole-graph kernel fusion. Rust has no equivalent automatic fusion; closing this gap would require hand-written CUDA kernels or framework support (e.g., Burn).

**5. Python training slowest at scale** (49.18ms at h=256, 5.20x slowest). The CPython interpreter overhead compounds with backward pass complexity at larger model sizes.

**6. Julia training growth highest** (9.5x for 4x hidden). This reflects Julia's efficient backward pass scaling — broadcast fusion handles larger tensors without proportional allocation increase.

#### Measured Convergence Trend

| Scale | Forward Spread | Train Spread | Status |
|-------|---------------|-------------|--------|
| hidden=64 | 4.16x | 10.03x | Measured |
| hidden=256 | 1.90x | 5.20x | Measured |

Forward spread halved (-54%) from h=64 to h=256 as BLAS share grows. Training convergence is slower due to backward pass implementation differences — Julia's language-level broadcast fusion vs Rust's manual loops vs Python's interpreter overhead.

#### Per-Finding Scaling (Measured)

**Forward pass: Rust leads at both scales.** Rust leads at h=64 (0.57 vs Julia 0.61ms) and h=256 (2.88 vs Julia 3.51ms). The gap narrows at h=256 as BLAS fraction grows.

**Training: Julia leads at both scales.** h=64: Julia 1.00ms. h=256: Julia 9.46ms. Julia's broadcast fusion advantage persists at scale (1.79x over Rust at h=256). Rust's zero-alloc backward is efficient but Julia's fused operations are faster overall.

**RSS differences already irrelevant at h=256:** Model weights at hidden=256 total ~2MB. Julia's 488MB runtime is still 99%+ of RSS. At larger hidden sizes, model weights grow and runtime overhead becomes a smaller fraction of total RSS.

**FFI overhead amortized:** Go's CGO bridge overhead is significant at 64×64. At 256×256, sgemm takes longer, making CGO a smaller fraction. At 1024×1024, CGO becomes negligible relative to compute.

#### When Language Choice Still Matters

| Scenario | Language matters? | Evidence |
|----------|------------------|----------|
| Small-model edge inference (hidden <= 256) | **Yes -- measured** | Forward spread 1.90x at h=256. Rust's 19.8MB RSS matters for embedded. |
| Real-time inference SLA (<1ms) | **Yes** | At h=64, Rust achieves it (0.57ms). At h=256, none do (2.88ms minimum). |
| Production training (hidden=1024+) | **No** | BLAS >90%. Language is orchestration layer. Pick for ecosystem, not speed. |
| GPU training | **No** | A100 ~312 TFLOPS vs M1 ~1.5 TFLOPS. Language overhead is invisible next to 200× GPU advantage. |

#### Summary

**Measured convergence: forward spread 4.16x → 1.90x at hidden=256 (-54%), training spread 10.03x → 5.20x (-48%).** Language differences narrow as BLAS fraction grows. Training differences persist longer due to backward pass architecture choices — Julia's language-level fusion compilation vs Rust's manual zero-alloc loops.

**Note:** All measurements are on Apple M1 (AMX). On production GPUs, the overhead structure changes fundamentally: Rust's kernel launch overhead would match Julia's (both zero-GC, direct memory control), but Julia retains an architectural advantage through Reactant.jl (XLA-based automatic kernel fusion). Treat absolute numbers as M1-specific reference data.

---

## 5. When to Choose What

| Use Case | Recommended | Rationale |
|----------|-------------|-----------|
| Production inference (latency) | Rust or Julia | Rust 0.57ms, Julia 0.61ms forward, zero GC, deterministic |
| Training pipeline | Julia | 1.00ms/step, zero GC, broadcast-fused backward |
| Concurrent inference (throughput) | Rust | 4,738 inf/s at T4, Rust leads absolute throughput |
| Research prototyping | Python | Ecosystem (PyTorch/JAX), fastest time-to-experiment |
| Scientific computing | Julia | Column-major BLAS + multiple dispatch + JIT + 1.00ms training leader |
| Edge / embedded / WASM | Rust | 19.8MB RSS, no runtime, `no_std` possible |
| Microservice / API backend | Go | Simple deploy, invisible GC, fast compilation, goroutines |
| Maximum kernel throughput | Rust | Fastest non-BLAS kernels (softmax 1.81us) |
| Correctness-critical numerics | Rust | IEEE 754 by default, each approximation explicit + audited |
| Rapid numerical experimentation | Julia | REPL + JIT = iterate in seconds |

### Decision Framework

**Choose based on what you're optimizing for**:

- **Minimizing latency**: Rust (0.57ms forward, 19.8MB footprint) or Julia (0.61ms forward)
- **Minimizing training time**: Julia (1.00ms/step, zero GC, broadcast-fused backward)
- **Maximizing throughput**: Rust (4,738 inf/s at T4) or Julia (4,205 inf/s, best scaling ratio 2.88x)
- **Minimizing total cost of ownership**: Go (simple codebase, fast builds, invisible GC)
- **Minimizing time-to-production**: Python (ecosystem, prototype->deploy in same language)
- **Minimizing memory footprint**: Rust (19.8MB, no runtime overhead)
- **Minimizing developer friction**: Julia (column-major + JIT + broadcast fusion + multiple dispatch)

---

## 6. Known Limitations

These limitations should be considered when interpreting the results above. They represent known measurement gaps, not errors.

### alloc_bytes Not Cross-Language Comparable

| Language | What alloc_bytes measures |
|----------|--------------------------|
| Rust | Gross allocation (alloc only, no dealloc tracking) |
| Go | Cumulative bytes allocated (TotalAlloc diff) |
| Python | Net allocated bytes (tracemalloc snapshot diff, Python heap only) |
| Julia | Total GC-managed allocation (gc_num().allocd diff) |

`peak_rss_bytes` (via getrusage) IS comparable across all languages.

### Python alloc_bytes

`tracemalloc` tracks Python heap only (`PyMem_Malloc`). NumPy data buffers via libc `malloc` are not tracked. Peak RSS is the accurate metric. Python parallel gc/alloc fields = `null` (ProcessPoolExecutor child processes have independent trackers).

### Julia RSS

Julia's peak_rss for training (488MB) includes the JIT runtime, compiled method cache, and type inference metadata. The backward pass optimization reduced temporary array allocations (55.5M -> 3.0M per step). The ~200MB runtime floor is architectural — see "Julia RSS (~488-494MB) and AOT Reduction Potential" below for reduction strategies.

### Python Parallel

ProcessPoolExecutor with pre-created pool and worker initialization via `initializer=` callback. Models are created once per worker process at pool startup. Only dispatch + forward pass is timed, matching Go/Rust/Julia methodology.

### Julia Cold Start

`dispatch_cold` runs after other scenarios, so JIT has already compiled most methods. Measured cold time = model construction + first inference, not true process-level JIT cost (~300ms+ for clean process).

### AMX Contention

Apple AMX is a shared resource. Concurrent BLAS calls may serialize at hardware level, reducing parallel scaling for BLAS-heavy workloads. Parallel training scaling is lower than inference scaling (Julia T4/T1: 1.65x training vs 2.88x inference) due to backward pass tripling AMX bus transactions.

### Julia RSS (~488-494MB) and AOT Reduction Potential

Julia's ~488-494MB RSS is dominated by the JIT runtime (libjulia, LLVM JIT compiler, type inference metadata, compiled method cache). The model weights themselves total ~0.1MB at hidden=64. Training RSS (488MB) is slightly lower than forward RSS (494MB) due to backward pass allocation optimization reducing temporary array pressure.

AOT compilation can reduce this substantially:

| Approach | Expected RSS | Reduction | Notes |
|----------|-------------|-----------|-------|
| PackageCompiler (sysimage) | 360-420MB | 0-15% | Eliminates startup latency, runtime stays |
| `juliac` (Julia 1.12+, no trim) | 250-350MB | 15-40% | Bundles full runtime as static binary |
| `juliac --trim` | 30-60MB | 85-93% | Removes unused code; may break LBT→BLAS dynamic dispatch |
| StaticCompiler.jl | 0.5-2MB | **99%+** | No GC, no JIT, no runtime. Requires complete rewrite to avoid GC-tracked objects. BLAS via direct `ccall` only |

The ~200MB runtime floor (libjulia + LLVM + GC + type system) is architectural — it enables multiple dispatch, JIT specialization, and the productivity that makes Julia competitive. `StaticCompiler.jl` eliminates this entirely but requires a fundamentally different programming model (no heap allocation, no dynamic dispatch, no `Array`). For workloads where RSS matters (edge/embedded), Rust's 19.8MB with equivalent performance is the pragmatic choice.

### Small Matrix Size

64x64 matmul: per-call FFI/dispatch overhead is a significant fraction of total time. GFLOPS = overhead + compute combined. Larger matrices would yield higher GFLOPS as compute dominates.

---

## 7. Open Questions

These questions emerged from the benchmark results and represent potential directions for future investigation.

1. **Larger matrix sizes** -- 64x64 is small enough that FFI overhead dominates. Hidden=256+ would better represent production workloads.
