# Multi-Language Architecture Rationale

## Design Philosophy

従来の「各言語から個別にGPUコードを書く」設計ではなく、各言語の強みに基づいた**役割分担型アーキテクチャ**に移行。GPU処理はJuliaが一手に担い、カーネルは自動生成される。

## Language Selection Rationale

### Why Julia for Training?
- Multiple dispatch enables clean mathematical abstractions
- Type-stable JIT achieves near-C performance (50-1000x faster than Python for numerical code)
- Auto-GPU fusion via CUDA.jl eliminates manual kernel writing
- Lux+Reactant leverage MLIR/XLA for graph compilation

**NOT** Rust (verbose math syntax), Python (1000x slower), Go (no autodiff ecosystem)

### Why Rust for Production?
- Ownership system eliminates memory bugs at compile time
- Zero-cost abstractions match C performance without unsafe code complexity
- Fearless concurrency via type system (Send/Sync traits)
- jlrs provides <1ms FFI latency to Julia

**NOT** Julia (1-30s startup time), Go (GC pauses in hot paths), Zig (manual memory management overhead)

### Why Python for Prototyping?
- 10+ years of data science ecosystem maturity (NumPy/Pandas/Matplotlib)
- REPL-driven development accelerates iteration
- Ubiquitous in ML research (easy team onboarding)

**NOT** Julia (fewer libraries), Rust (slow compilation), Mojo (pre-1.0 instability)

### Why Go for Orchestration?
- Goroutines provide M:N scheduling with minimal overhead
- 2-second compilation enables fast iteration
- Single binary deployment simplifies ops

**NOT** Rust (lifetime complexity overkill), Elixir (OTP overhead for single-node), Julia (startup latency)

### Why TypeScript for Visualization?
- Browser runtime provides universal access (no installation)
- React ecosystem offers battle-tested UI components
- D3/Plotly.js enable sophisticated interactive charts

**NOT** Python (Dash is server-side, slower), Julia (Makie lacks web maturity), Rust (WASM friction)

### Why Elixir for Resilience?
- OTP supervision trees provide automatic fault recovery
- Hot code reload enables zero-downtime updates (40 years of telecom reliability)
- BEAM VM handles multi-node clustering natively

**NOT** Go (manual recovery logic), Kubernetes (infrastructure complexity), Rust (no runtime support)

### Why Zig for Kernels?
- `comptime` enables compile-time SIMD selection without macros
- C interop requires zero bindings (direct compatibility)
- Explicit allocator interface provides first-class memory control

**NOT** Rust (borrow checker friction in tight loops), C (unsafe by default), Julia (GC overhead in critical paths)

### Why Mojo Experimental?
- Python syntax compatibility enables gradual migration
- Targets 10-100x speedup for hot paths (when stable)
- Pre-1.0 status requires conservative adoption (basic syntax only)

**NOT** production-critical paths (use Rust/Julia), GPU kernels (use Julia), anything requiring MLIR features (wait for v1.0)

## Performance Comparisons

### Training Performance (Julia vs Python)
- Julia (Lux): 50-1000x faster for numerical code
- Automatic GPU fusion eliminates kernel launch overhead
- Multiple dispatch enables zero-cost abstractions for math

### Inference Latency (Rust vs Python)
- Rust (Burn): <1ms p99 latency
- Zero GC pauses (predictable performance)
- Static binary deployment (<50MB)

### Concurrency (Go vs Rust vs Elixir)
- Go: Simple goroutines, fast compile, 2-20ms GC pauses
- Rust: Complex lifetimes, perfect for <1ms requirements
- Elixir: Actor model overhead, best for distributed systems

## Anti-Pattern Explanations

### Performance Mismatches
- **Python for inference**: 100-1000x slower, GIL bottleneck, unpredictable GC → Use Rust (Burn)
- **Go for heavy math**: No SIMD, weak numeric types, GC in loops → Use Julia
- **Julia for web APIs**: 1-30s startup, 200MB+ binaries → Use Rust (axum)
- **TypeScript for compute**: Single-threaded Node.js, no SIMD → Use Rust/Julia

### Complexity Mismatches
- **Rust for prototyping**: Slow compile, borrow checker friction → Use Python/Julia
- **Elixir for single-node**: OTP overhead, unnecessary actors → Use Go/Rust
- **Zig for application logic**: Manual memory, pre-1.0 stability → Use Rust/Go

### Ecosystem Mismatches
- **Manual CUDA kernels**: Maintenance nightmare → Use Julia (CUDA.jl auto-generation)
- **TensorFlow/PyTorch**: Python overhead, opaque graphs → Use Julia Lux (from scratch)
- **Python→Julia FFI**: PyCall overhead, GIL interaction → Use Rust hub (jlrs)

### Maturity Mismatches (2026)
- **Mojo for production**: Pre-1.0, unstable API → Wait for v1.0, trial basic syntax only
- **Zig for critical paths**: Pre-1.0 stability → Use Rust (allow experimental kernels)

## Migration Path Rationale

### Phase 1-3: Core Foundation (Q1 2026)
Establish Rust-Julia FFI as project "spine." Validates architecture before expanding.

### Phase 4: TypeScript Dashboard (Q1-Q2 2026)
Immediate user value. Web visualization requires no infrastructure changes.

### Phase 5-6: Conditional Expansion (Q2-Q4 2026)
Add Elixir only if multi-node needed. Add Zig only if custom SIMD proves necessary.

### Phase 7-8: Experimental (2027)
Trial Mojo conservatively. Require >10x speedup vs Python to justify adoption.

## Why 8 Languages?

**Baseline (4)**: Julia/Rust/Python/Go = fully functional
**Production (5)**: +TypeScript = user-facing system
**Maximal (8)**: +Elixir/Zig/Mojo = "interesting configuration" (complexity cost)

Each language has non-overlapping primary use case. No redundancy.

## Decision Tree

```
Need real-time web UI?
  → Yes: Add TypeScript (Phase 4)
  → No: Skip

Need multi-node resilience?
  → Yes: Add Elixir (Phase 5)
  → No: Use Go for single-node

Need custom SIMD optimization?
  → Yes: Add Zig (Phase 6)
  → No: Julia auto-kernels sufficient

Python hot-path bottleneck + Mojo v1.0 stable?
  → Yes: Trial Mojo (Phase 7)
  → No: Keep Python or migrate to Rust/Julia
```

## Key Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| Julia for training | 50-1000x speedup, auto-GPU | 1-30s startup, 200MB+ binaries |
| Rust for production | <1ms latency, zero GC | Slow compilation, complex lifetimes |
| 8 languages total | Each optimized for role | 8x toolchain complexity |
| Mojo experimental | Future Python speedup | Pre-1.0 instability risk |
| From-scratch DL | Full control, no Python | Reimplement training infrastructure |
