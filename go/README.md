<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# Go MoE Transformer

From-scratch Mixture-of-Experts Transformer in Go.
Zero external dependencies beyond the standard library and Apple Accelerate via CGO.

## Architecture

```
go/
├── tensor.go       # Tensor type, flat []float32 storage, pure-f32 math (exp/sqrt/log/sin/cos)
├── config.go       # Model hyperparameters (Default6_9B, Tiny presets)
├── layers.go       # Embedding, Linear, RMSNorm, SwiGLU
├── attention.go    # Multi-Query Attention with RoPE (causal, NTK-aware scaling)
├── moe.go          # Router (top-K gating), MoELayer (sparse dispatch), TransformerBlock
├── model.go        # MoETransformer (full model assembly)
├── generate.go     # Sampling strategies (greedy, temperature, top-K, top-P)
├── train.go        # CrossEntropy loss, AdamW optimizer, LR schedule, checkpointing, loss scaling
├── sgemm.go        # Apple Accelerate CGO wrapper (cblas_sgemm)
├── bench_test.go   # Benchmark harness (5 axes: memory, compiler, type system, parallel, scale)
├── nn_test.go      # Unit and integration tests
└── go.mod          # Module definition (Go 1.22+)
```

All files belong to package `nn`. No subdirectories.

## Equation-to-Code Map

| Formula | File | Function |
|---------|------|----------|
| `C = A @ B` (GEMM) | `sgemm.go` | `sgemm()` |
| `C = A @ B^T` | `sgemm.go` | `sgemmTransB()` |
| `Softmax: p_i = exp(x_i - max) / sum(exp(x_j - max))` | `tensor.go` | `Softmax()` |
| `SiLU(x) = x / (1 + exp(-x))` | `tensor.go` | `SiLU()` |
| `RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma` | `layers.go` | `RMSNorm.Forward()` |
| `Linear: y = x @ W^T + b` | `layers.go` | `Linear.Forward()` |
| `Embedding: out[b,s] = W[token_id]` | `layers.go` | `Embedding.Forward()` |
| `SwiGLU: out = W_down @ (SiLU(W_gate @ x) * W_up @ x)` | `layers.go` | `SwiGLU.Forward()` |
| `RoPE: [x0,x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]` | `attention.go` | `applyRoPE()` |
| `Attention: softmax(Q@K^T/sqrt(d)) @ V` | `attention.go` | `MQAttention.Forward()` |
| `MoE: out = sum_k(w_k * Expert_k(x))` | `moe.go` | `MoELayer.Forward()` |
| `Gate: probs = softmax(W_gate @ x), top-K select` | `moe.go` | `Router.Forward()` |
| `AuxLoss: alpha * N * sum(f_e * P_e)` | `moe.go` | `Router.ComputeAuxLoss()` |
| `CrossEntropy: L = -mean(log(softmax(logits)[target]))` | `train.go` | `crossEntropyLoss()` |
| `CE Grad: (softmax(logits) - one_hot) / N` | `train.go` | `crossEntropyGrad()` |
| `AdamW: m=b1*m+(1-b1)*g, v=b2*v+(1-b2)*g^2, w-=lr*(m_hat/(sqrt(v_hat)+e)+wd*w)` | `train.go` | `Trainer.TrainStep()` |
| `LR Schedule: warmup linear + cosine decay` | `train.go` | `Trainer.GetLR()` |
| `Grad Clip: t *= clip/(norm+eps) if norm > clip` | `train.go` | `clipTensorByGlobalNorm()` |
| `exp(x) = 2^k * Horner(r)` | `tensor.go` | `ExpF32()` |
| `sqrt(x) = x * fast_inv_sqrt(x)` (Quake III + Newton) | `tensor.go` | `SqrtF32()` |
| `ln(x) = e*ln2 + atanh_poly(m)` | `tensor.go` | `LogF32()` |

## Implementation Notes

### CGO Accelerate Integration (`sgemm.go`)

Matrix multiplication is delegated to Apple Accelerate's `cblas_sgemm`, which routes
through the AMX coprocessor on Apple Silicon (7-14x faster than NEON SIMD).

The CGO bridge is minimal: two thin wrappers (`sgemm` and `sgemmTransB`) that convert
Go types to C types. Key details:

- **Framework linking**: `#cgo LDFLAGS: -framework Accelerate` -- no static library needed
- **Pointer conversion**: `(*C.float)(unsafe.Pointer(&slice[0]))` extracts the raw data
  pointer from the Go slice header. CGO pins this pointer for the duration of the C call.
- **Empty slice guard**: Both functions return early if any dimension is 0. Without this,
  `&slice[0]` panics on nil/empty slices -- the most common CGO pitfall.
- **No `CblasTrans` for A**: Weight is stored as `[out, in]` so `sgemmTransB` applies
  `CblasTrans` only on B, avoiding a separate transpose allocation.

### Flat `[]float32` as Tensor Storage

There is no generic tensor library. `Tensor` is simply:
```go
type Tensor struct {
    data  []float32  // contiguous row-major storage
    shape Shape      // dimension metadata
    dtype DType      // type tag (only F32 used at runtime)
    Grad  []float32  // per-parameter gradient (nil until first backward)
}
```

`Grad` is lazily allocated: it stays nil until the first backward pass accumulates a gradient into it via `AccumulateGrad()`. `ZeroGrad()` zeros in place if allocated, stays nil otherwise. AdamW reads each parameter's `Grad` to update weights.

Index calculations use row-major strides. For a `[batch, seq, hidden]` tensor, the flat
offset of element `[b, s, h]` is `b*seq*hidden + s*hidden + h`. The `splitLast` helper peels
off leading dimensions so that all matmul operations work on 2D `[batch_total, features]`.

In-place variants (`AddInPlace`, `MulInPlace`, `SiLUInPlace`, `ScaleInPlace`) mutate the
receiver tensor directly, eliminating temporary allocations in the SwiGLU and attention
hot paths. Combined with `SoftmaxInto` (pre-allocated output), these keep per-forward
allocation pressure manageable for the GC.

### Goroutine Parallel Model

The benchmark's parallel axis uses one independent model per goroutine with `sync.WaitGroup`:

```go
var wg sync.WaitGroup
wg.Add(N)
for i := 0; i < N; i++ {
    m, inp := models[i], inputs[i]  // capture before goroutine
    go func() {
        defer wg.Done()
        m.Forward(inp)  // no shared state
    }()
}
wg.Wait()
```

No locks, no channels for data -- each goroutine owns its model and input exclusively.
This tests pure goroutine scheduling overhead and GC behavior under parallel allocation.

### GC Behavior Under ML Workload

The benchmark measures `gc_throughput = 1.0 - (gc_time / total_compute_time)`. Typical
values are 0.987-0.993 across all scenarios, meaning GC consumes less than 1.3% of runtime at worst. This is because:

- Tensor data is large contiguous `[]float32` slices (easy for GC to scan)
- Most allocations are short-lived intermediates (per-layer outputs) that die young
- Hot-path layers reuse buffers across calls (`RMSNorm.lastRMS`, `Router.softmaxBuf`, `MoELayer.outBuf`, `MQAttention.scoresBuf/attnOutBuf`) to reduce allocation frequency

## Performance Characteristics

Latest benchmark results (M1, batch=2, seq=32, hidden=64):

| Metric | Value | Rank (of 4 languages) |
|--------|-------|------|
| Forward pass | 1.14ms | 3rd (Rust 0.56ms, Julia 0.59ms) |
| Train step | 3.28ms | 3rd (Julia 0.98ms, Rust 1.44ms) |
| Matmul 64×64 | 536 GFLOPS | 3rd |
| Softmax kernel | 5.67us | 3rd (Rust 1.83us, Julia 4.83us) |
| RMSNorm kernel | 8.75us | 3rd (Rust 3.38us, Julia 4.17us) |
| Peak RSS | 31MB | 2nd (Rust 19MB) |
| T4 throughput | 2,189 inf/s | 3rd |
| T4 train | 867 trn/s | 3rd |
| GC throughput | 0.985-0.996 | Best among GC languages |

**GC is invisible**: 28.9M bytes allocated per train step, yet GC throughput never drops below 0.985. Go's concurrent tracing GC is tuned for latency — it sacrifices throughput headroom to minimize pause impact. Under this ML workload, GC is a non-issue.

**CGO overhead is the bottleneck**: ~1us per BLAS call. At 64×64 this is ~2x Rust's direct FFI overhead. At production matrix sizes, the overhead would be negligible.

**Compiler optimization ceiling**: Go's compiler (gc) is designed for fast compilation, not peak codegen. Without LLVM, it cannot auto-vectorize or exploit SIMD. softmax (5.67us) is 3.1x Rust (1.83us) — this is the compiler gap, not the language.

**Go's value proposition is not "fastest" — it's "fast enough with lowest total cost of ownership."** Simplest codebase, fastest compilation, most straightforward deployment.

## Gotchas / Pitfalls

### Empty Slice Panic at CGO Boundary

```go
// PANICS if data is nil or empty:
(*C.float)(unsafe.Pointer(&data[0]))

// SAFE: always check length first
if len(data) == 0 {
    return
}
```

Both `sgemm()` and `sgemmTransB()` have this guard. Any new CGO wrapper must include it.

### CGO Overhead for Small Matrices

Each CGO call costs ~100ns-1us (goroutine stack switch + runtime state save). For a 64x64
matmul (~524K FLOPS), the compute time is ~1us, so CGO overhead is ~50% of total time.
For production-size matrices (512+ dimensions), the overhead is negligible (<1%).

The benchmark's `kernel_matmul` scenario specifically measures this to quantify CGO
amortization.

### Allocation Rate Measurement

The benchmark snapshots `runtime.MemStats.TotalAlloc` before and after the timed trials
(not including warmup). The per-trial allocation rate is:

```
alloc_rate = (total_alloc_after - total_alloc_before) / num_trials / median_wall_seconds
```

This gives bytes/sec of heap allocation during steady-state operation, excluding one-time
setup costs.

## Build and Test

```bash
# Test (excludes bench by default since TestBench is slow)
go test -run 'Test[^B]' ./...

# Full test including benchmark
go test -v -count=1 -timeout 300s ./...

# Build check
go build ./...
```

Requires:
- Go 1.22+
- macOS with Xcode Command Line Tools (for Accelerate framework)
- CGO enabled (default on macOS)
