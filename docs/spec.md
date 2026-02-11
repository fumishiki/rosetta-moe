<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# Educational SLM Comparison Spec (CPU-Only)

## 1. Purpose

This repository is an educational and verification-focused implementation of a Small Language Model (SLM) across four languages:

- Python
- Go
- Rust
- Julia

The goal is to compare language/runtime tradeoffs with a shared model contract under reproducible local CPU conditions.

Most ML benchmarks compare frameworks (PyTorch vs JAX vs Flux) running on different runtimes but written in the same host language. This project inverts that: four languages implement the identical MoE Transformer from scratch, so observed differences reflect the language and runtime layer — memory model, compiler, type system, concurrency — rather than framework-level choices. The result is a controlled experiment that isolates what the language itself contributes to (or detracts from) ML workload performance.

## 2. Primary Questions

This project answers the following:

1. How do Python, Go, Rust, and Julia differ in implementation style for the same Transformer architecture?
2. What is the runtime cost profile on local CPU, including GC overhead where applicable?
3. How much source volume and complexity is required per language for equivalent functionality?
4. Which design choices are language-specific vs architecture-invariant?

## 3. Scope

In scope:

- CPU-only local execution
- Shared Transformer/SLM feature set across 4 languages
- Unified correctness tests and benchmark scenarios
- Runtime and memory profiling, including GC metrics
- Source size and maintainability metrics
- Educational documentation that maps equations to implementation units

Out of scope:

- Accelerator-specific kernel development
- Distributed training
- Production-scale dataset training
- Vendor-specific backend tuning

## 4. Repository Positioning

Positioning:

- This is a learning and systems-comparison repository.
- Runtime comparison is standardized on CPU to keep results reproducible and portable.

Policy:

- Main acceptance gates are CPU correctness + CPU benchmark reproducibility.

## 5. Consolidated Requirements

Implementation requirements for all language tracks.

### 5.1 Shared Contract Invariants (MUST)

A "shared contract" means all four languages implement the exact same model architecture with identical hyperparameters, layer ordering, and numerical behavior. This eliminates model-level confounds: any performance difference between, say, Rust and Go reflects language/runtime characteristics, not architectural choices.

All language implementations must preserve these interfaces and behaviors.

| Area | Required Contract |
|---|---|
| Tensor semantics | Row-major (Julia: column-major native), explicit shape checks, deterministic broadcasting rules |
| Core blocks | Embedding, Attention, FFN, Normalization, Output head |
| Training pieces | Forward, backward/gradient flow, optimizer step (minimal) |
| Inference pieces | `SamplingStrategy` dispatch: greedy, temperature, top-k, top-p |
| Numerical tolerance | Consistent tolerance windows for float comparisons |
| Seed control | Reproducible initialization and sampling per language |

### 5.2 Functional Requirements (MUST)

These requirements ensure the four implementations are functionally equivalent — same layer order, same dispatch logic, same loss terms — so benchmark comparisons are apples-to-apples.

| ID | Requirement |
|---|---|
| FR-01 | Transformer block order must be `RMSNorm -> Attention (with RoPE) -> Residual -> RMSNorm -> MoE/FFN -> Residual`. |
| FR-02 | Attention path must support multi-head query with reduced KV-head configuration (MQA/GQA-style contract). |
| FR-03 | MoE path must include router softmax + top-k dispatch + weighted combine. |
| FR-04 | Expert FFN must support SwiGLU-style gating (`gate`, `up`, `down`). |
| FR-05 | Training loop must cover `Forward -> Loss -> Backward -> Optimizer -> Step`. |
| FR-06 | Loss must include cross-entropy and configurable auxiliary load-balancing loss for MoE. |
| FR-07 | Optimizer baseline must include AdamW with configurable learning-rate schedule and gradient clipping hook. |
| FR-08 | Inference path must provide `SamplingStrategy` type hierarchy with `pick_token` dispatch, unified `generate(model, prompt, max_len, strategy)` entry point, and backward-compatible wrappers (`generate_greedy`, `generate_sample`, `generate_topk`, `generate_topp`). |
| FR-09 | Validation path must include explicit shape/error checks and deterministic seed handling. |

### 5.3 Architecture and Hyperparameter Profile

| Item | Tiny (benchmark) | Full (default_6_9b) |
|---|---|---|
| Layers | 2 | 30 |
| Hidden dim | 64 | 768 |
| Vocab size | 1,000 | 32,000 |
| Experts / top-k | 4 / 2 | 16 / 4 |
| Attention | 4 Q / 1 KV | 12 Q / 1 KV |
| Head dim | 16 | 64 |
| FFN dim | 256 | 6,144 |
| Max seq len | 512 | 32,768 |
| RoPE base / alpha | 10,000 / 1.0 | 10,000 / 8.0 |
| Precision | float32 | float32 |

The evaluation matrix spans 5 axes:

| Axis | Focus | Scenarios |
|------|-------|-----------|
| 1. Memory | Alloc/GC overhead across scaling dimensions | 9 (train step + batch×4 + seq×4) |
| 2. Compiler | Raw kernel optimization (BLAS baseline) | 3 (matmul, softmax, rmsnorm) |
| 3. Type System | Dispatch mechanisms (cold vs warm) | 2 (cold, warm) |
| 4. Parallel | Concurrency scaling (no shared state) | 6 (inference×3 + train×3) |
| 5. Scale | Convergence at larger model | 2 (forward + train at hidden=256) |

The **Tiny** profile is used for all benchmarks — small enough to run 10 trials with warmup in seconds, yet large enough to exercise every code path (MoE routing, GQA attention, SwiGLU gating). The **Full** profile is the reference architecture at real-world scale: 6.9B total parameters with 1.8B active per token via MoE top-k routing. It exists to verify that implementations handle realistic dimensions without overflow or shape errors, but is not benchmarked.

### 5.4 Training System Requirements (MUST/SHOULD)

| Priority | Requirement |
|---|---|
| MUST | Provide `train_step` interface with loss output and step progression. |
| MUST | Expose auxiliary MoE loss contribution as configurable coefficient (alpha). |
| MUST | Keep per-language optimizer state semantics consistent (AdamW moments and step). |
| SHOULD | Support warmup + cosine learning-rate schedule for cross-language parity. |
| SHOULD | Provide optional features such as gradient checkpointing and mixed precision when language/runtime supports them. (Not implemented.) |

### 5.5 Non-Goals

- Accelerator-specific kernel implementation
- GPU-resident decode/training pipelines
- Distributed training or production-scale data
- Vendor-specific backend tuning

## 6. Evaluation Matrix — 4 Research Axes

Performance evaluation is structured around 4 orthogonal axes that isolate distinct language/runtime characteristics. Non-performance evaluation (correctness) follows.

The 4-axis design ensures each axis isolates ONE language characteristic — memory model, compiler optimization, type system dispatch, or concurrency scaling — so that observed performance differences can be attributed to a specific language feature rather than confounded across multiple factors. By keeping workloads small and targeted per axis, we avoid the "everything is different" problem that plagues most cross-language comparisons.

### Metric Design Principles

All dynamic metrics satisfy the 5 requirements from Marr et al. ("Are We Fast Yet?", DLS 2016):
unambiguous, dynamic, robust, discriminating, machine-independent.

Metrics are categorized as:

| Category | Collection | Examples |
|----------|------------|---------|
| **Measured** | Instrumented in bench harness | `median_ns`, `cpu_time_ns`, `peak_rss`, `alloc_bytes`, `gc_time_ns`, `gc_pause_count` |
| **Derived** | Post-processed from measured | `throughput`, `alloc_rate`, `gc_throughput`, `GFLOPS`, `speedup`, `cold/warm ratio` |
| **Static** | Computed once per codebase | `source_gzip_bytes`, `SLOC`, `function_count` |

Cross-language comparability: `peak_rss` (via `getrusage`) is comparable. `alloc_bytes` is NOT — each language measures different allocation scopes. See bench-results.md for full rationale and literature references.

### 6.1 Axis 1: Memory Management (Ownership / GC / RC)

Does garbage collection actually hurt ML performance? This axis measures allocation pressure and GC overhead under realistic training and inference loads, scaling batch size and sequence length independently. Rust (ownership, no GC), Go (concurrent GC), Python (refcount + cycle GC), and Julia (generational GC) each handle memory differently — this axis quantifies the cost.

| Scenario | Workload | Parameters |
|----------|----------|------------|
| mem_train_step | 1 train step (fwd + bwd + opt) | batch=2, seq=8, hidden=64 |
| mem_scale_batch_{1,2,4,8} | Full inference, varying batch | seq=32, hidden=64 |
| mem_scale_seq_{8,16,32,64} | Full inference, varying seq_len | batch=2, hidden=64 |

| Metric | Type | Description |
|--------|------|-------------|
| median_ns | measured | Median wall-clock time |
| cpu_time_ns | measured | User + system CPU time (getrusage) |
| alloc_bytes | measured | Total heap allocation per scenario |
| gc_pause_count | measured | Number of GC pauses during scenario |
| gc_time_ns | measured | Total GC wall-clock time |
| peak_rss | measured | Peak resident set size (getrusage) |
| throughput | derived | Tokens/sec or samples/sec |
| alloc_rate | derived | `alloc_bytes / (median_ns × 1e-9)` — allocation pressure (bytes/sec) |
| gc_throughput | derived | `1.0 - (gc_time_ns / median_ns)` — fraction of time NOT in GC [0,1] (GEAR, ICSE 2025) |

### 6.2 Axis 2: Compiler Optimization (LLVM vs Go compiler vs JIT vs BLAS)

How much does the language's compiler affect raw computation speed? This axis benchmarks individual math kernels (matmul, softmax, rmsnorm) that bypass all framework and allocation overhead, isolating the compiler's ability to vectorize and optimize tight loops. Matmul uses Apple Accelerate (`cblas_sgemm`) in all languages to provide a BLAS-level baseline; softmax and rmsnorm use hand-written loops where compiler differences are fully exposed.

| Scenario | Workload | Parameters |
|----------|----------|------------|
| kernel_matmul | C = A * B^T | M=K=N=64 |
| kernel_softmax | softmax(x) | n=1000 |
| kernel_rmsnorm | rmsnorm(x, w) | shape=(2,32,64) |

| Metric | Type | Description |
|--------|------|-------------|
| median_ns | measured | Median wall-clock time |
| cpu_time_ns | measured | User + system CPU time |
| GFLOPS | derived | `known_op_count / (median_ns × 1e-9) × 1e-9` |

### 6.3 Axis 3: Type System (Multiple Dispatch vs Monomorphization)

Julia's multiple dispatch vs Rust's monomorphization vs Go's interface dispatch vs Python's dynamic lookup — which type system produces faster code for Transformer workloads? This axis measures the cold-start cost (where JIT compilation and monomorphization happen) against warm steady-state throughput, revealing the runtime tax of each language's type resolution strategy.

| Scenario | Workload | Parameters |
|----------|----------|------------|
| dispatch_warm | Full forward pass (post-warmup) | batch=2, seq=32, hidden=64 |
| dispatch_cold | Model construction + first forward | batch=1, seq=8, hidden=64 |

| Metric | Type | Description |
|--------|------|-------------|
| median_ns | measured | Median wall-clock time |
| cpu_time_ns | measured | User + system CPU time |
| cold/warm ratio | derived | `dispatch_cold_ns / dispatch_warm_ns` — JIT/monomorphization overhead |

### 6.4 Axis 4: Parallel Model (std::thread vs goroutine vs ProcessPool vs Threads.@threads)

How well does each language's concurrency model scale? This axis runs independent model instances in parallel (no shared state) to test the overhead of threads, goroutines, processes, and Julia tasks. By eliminating synchronization, it isolates the raw concurrency primitive cost — scheduler overhead, memory duplication, and OS thread management.

| Scenario | Workload | Parameters |
|----------|----------|------------|
| parallel_T{1,2,4} | N independent forward passes, concurrent | batch=2, seq=32, hidden=64 |
| parallel_train_T{1,2,4} | N independent train steps, concurrent | batch=2, seq=8, hidden=64 |

| Metric | Type | Description |
|--------|------|-------------|
| median_ns | measured | Median wall-clock time |
| cpu_time_ns | measured | User + system CPU time (reveals CPU utilization) |
| throughput | derived | Inferences/sec |
| speedup | derived | `throughput_TN / throughput_T1` |

### 6.5 Axis 5: Scale (Hidden Dimension Convergence)

Do language differences persist at larger model sizes? This axis measures forward pass and training step at hidden=256 (4× the benchmark config) to observe performance convergence as BLAS fraction grows. If BLAS dominates at larger sizes, all languages should converge toward similar times.

| Scenario | Workload | Parameters |
|----------|----------|------------|
| scale_forward_256 | Full forward pass | batch=2, seq=32, hidden=256 |
| scale_train_256 | 1 train step (fwd + bwd + opt) | batch=2, seq=8, hidden=256 |

| Metric | Type | Description |
|--------|------|-------------|
| median_ns | measured | Median wall-clock time |
| cpu_time_ns | measured | User + system CPU time |
| throughput | derived | Tokens/sec |

### 6.6 Correctness

| Category | Metric |
|---|---|
| Forward parity | Output deltas vs reference within tolerance |
| Training parity | Loss trend and gradient sanity checks |
| Shape safety | Invalid input rejection behavior |
| Determinism | Repeated same-seed outputs |

## 7. Measurement Methodology

### Environment Control

- Same machine, local CPU only
- Fixed OS and compiler/interpreter versions recorded in result metadata
- Fixed seed (42) and fixed input fixtures
- N_TRIALS=10, N_WARMUP=3, median reported with IQR

Median with IQR is chosen over mean with stddev because system benchmarks are prone to outliers (background daemons, thermal throttling, GC storms). Median is robust to these — a single 10x outlier does not skew the result. N=10 trials provides a stable median for CPU-only workloads; 3 warmup iterations eliminate JIT compilation latency (Julia), CPU cache cold-start effects, and lazy initialization overhead across all languages.

### Scenario Groups (5 Axes)

1. **Axis 1 — Memory Management** (9 scenarios):
   mem_train_step + mem_scale_batch_{1,2,4,8} + mem_scale_seq_{8,16,32,64}
2. **Axis 2 — Compiler Optimization** (3 scenarios):
   kernel_matmul + kernel_softmax + kernel_rmsnorm
3. **Axis 3 — Type System** (2 scenarios):
   dispatch_warm + dispatch_cold
4. **Axis 4 — Parallel Model** (6 scenarios):
   parallel_T{1,2,4} + parallel_train_T{1,2,4}
5. **Axis 5 — Scale** (2 scenarios):
   scale_forward_256 + scale_train_256

Total: 22 scenarios (max batch=8, max seq=64).

### Reporting

- Record raw measurements and summarized tables
- Separate warmup from steady-state
- Explicitly flag non-comparable measurements (e.g., alloc_bytes semantics differ across languages)
- Record metadata: OS, CPU model, thread setting, runtime/compiler versions
- GFLOPS for Axis 2 computed from known operation counts

## 8. Documentation Requirements

Each language track must provide:

1. Equation-to-code map:
   formula unit -> file -> function
2. Runtime notes:
   where allocations happen, where GC pressure is created or avoided
3. Tradeoff notes:
   readability, safety, performance implications

Repository-level docs must provide:

- Cross-language comparison tables
- Reproducible benchmark procedure
- Interpretation guidelines and known limitations

## 9. Deliverables

Mandatory deliverables:

1. CPU benchmark report for Python/Go/Rust/Julia
2. GC overhead comparison table (where runtime uses GC)
3. Equation-to-code correspondence (21-entry mapping tables in each language README, with inline math-to-code comments in all source files)
4. Reproducible runbook for local verification
5. Numerical stability report (NaN/Inf and divergence checks)
6. Scaling report (`seq_len`, `batch_size`, hidden dimension)
7. Parallel efficiency report (thread scaling)

## 10. Acceptance Criteria

1. All four implementations pass shared correctness scenarios.
2. CPU benchmark runs are reproducible from documented commands.
3. GC and memory metrics are collected and published in a comparable format.
4. Numerical stability checks pass under predefined stress inputs.
5. Scaling and parallel efficiency results are published with raw data.
6. Error semantics are validated for invalid-shape/index/empty-input cases.
7. Equation-to-code mapping tables (21 entries per language) are present in each language README, with corresponding inline comments in source files.
8. Functional requirements in Section 5.2 are implemented across all language tracks.

## 11. Risks and Mitigations

Cross-language benchmarking is inherently noisy — different runtimes, different memory models, different measurement granularities. These are the known risks and our mitigations.

| Risk | Impact | Mitigation |
|---|---|---|
| Runtime measurement bias | Misleading comparisons | Fixed environment, repeated runs, raw data publication |
| Feature drift across languages | Invalid parity claims | Shared contract and parity tests first |
| Overfitting to microbenchmarks | Wrong conclusions | Include both micro and mini end-to-end scenarios |
| Runtime instrumentation mismatch | Incomparable metrics | Define metric schema and collectors before benchmarking |
| Parallel benchmark noise | False scaling conclusions | Pin thread policy and run repeated trials |
| Documentation drift | Reduced educational value | Update docs as acceptance gate |

## 12. Execution Plan

Phase 1 (DONE):

- Lock common SLM contract and fixtures
- Align core module boundaries across Python/Go/Rust/Julia

Completed artifacts:

- Unified file structure: `tensor`, `config`, `layers`, `attention`, `moe`, `model`, `generate`, `train`
- Unified type names: `Config`, `SwiGLU`, `MQAttention`, `MoETransformer`, `SamplingStrategy`
- Include/import DAG: `tensor -> config -> layers -> attention -> moe -> model -> generate -> train`
- `SamplingStrategy` type hierarchy with `pick_token` dispatch in all 4 languages
- All test suites passing: Julia 71, Rust 21, Go 36, Python 31

Phase 2 (DONE):

- Benchmark harness implemented per language (`python/bench.py`, `rust/src/bin/bench.rs`, `go/bench_test.go`, `julia/bench.jl`)
- Memory/GC data collection via `getrusage` (peak_rss), language-specific alloc tracking, GC callback instrumentation
- All 22 scenarios × 5 axes producing unified JSON output
- BLAS unified: all languages use Apple Accelerate (`cblas_sgemm`) for matmul

Phase 3 (DONE):

- Benchmark results and methodology published in `docs/bench-results.md`
- Raw JSON outputs in `benchmarks/{python,rust,go,julia}.json`

Phase 4 (DONE): Equation-to-code mapping tables (21 entries) and math-to-code comments added to all source files across 4 languages.
