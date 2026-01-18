# CUDA Kernels

共有CUDAカーネル実装。Rust/Go/Python から FFI 経由で呼び出される。

## Structure

```
cuda/
├── kernels/           # CUDA kernel implementations (.cu)
│   ├── elementwise.cu # SiLU, Add, Mul, Scale
│   ├── softmax.cu     # Softmax, TopK
│   ├── rmsnorm.cu     # RMS normalization
│   ├── gemm.cu        # Matrix multiplication
│   ├── rope.cu        # Rotary position embedding
│   ├── attention.cu   # MQA, Flash Attention
│   ├── loss.cu        # Cross entropy, Aux loss
│   ├── optimizer.cu   # AdamW, Grad clip
│   └── decode.cu      # Argmax, Sampling
└── stub.c             # CPU fallback (returns -1)
```

## FFI Design Principles

### 1. C ABI Compatibility

全関数は C ABI (`extern "C"`) で公開される:

```c
int32_t cuda_silu(
    const float* input,
    float* output,
    int64_t n,
    void* stream
);
```

**Conventions:**
- Return type: `int32_t` (0=success, -1=not available, other=error)
- Pointers: `const` for input, mutable for output
- Stream: `void*` (nullable, NULL for default stream)
- Sizes: `int64_t` for large counts, `int` for dimensions

### 2. Language Bindings

```
┌─────────────────────────────────────────────────────────────┐
│                     cuda/kernels/*.cu                       │
│                     cuda/stub.c                             │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   rust/nn-cuda  │  │   go/cuda       │  │  python/cuda    │
│   build.rs +    │  │   Makefile +    │  │  ctypes +       │
│   cc crate      │  │   cgo           │  │  CPU fallback   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

| Language | Binding Method | Library Type |
|----------|---------------|--------------|
| Rust | cc crate in build.rs | Static (.a) |
| Go | cgo + Makefile | Static (.a) |
| Python | ctypes.CDLL | Shared (.so/.dylib) |

### 3. Stub Fallback

CUDA未インストール環境ではstub.cが全関数を実装:

```c
#define STUB_IMPL(name, ...) \
    int32_t name(__VA_ARGS__) { return -1; }

STUB_IMPL(cuda_silu, const float* input, float* output, int64_t n, void* stream)
STUB_IMPL(cuda_add, const float* a, const float* b, float* output, int64_t n, void* stream)
// ...
```

**Important:** stub.cは全言語のFFI宣言と**シグネチャが一致**する必要がある。

## Function Categories

### Elementwise Operations

| Function | Signature |
|----------|-----------|
| `cuda_silu` | `(input, output, n, stream)` |
| `cuda_add` | `(a, b, output, n, stream)` |
| `cuda_mul` | `(a, b, output, n, stream)` |
| `cuda_scale` | `(input, output, scale, n, stream)` |

### Normalization

| Function | Signature |
|----------|-----------|
| `cuda_softmax` | `(input, output, batch, dim, stream)` |
| `cuda_rmsnorm` | `(input, weight, output, batch, dim, eps, stream)` |

### Matrix Operations

| Function | Signature |
|----------|-----------|
| `cuda_gemm` | `(A, B, C, M, N, K, alpha, beta, stream)` |
| `cuda_gemm_batched` | `(A, B, C, batch, M, N, K, alpha, beta, stream)` |

### Attention

| Function | Signature |
|----------|-----------|
| `cuda_rope` | `(q, k, freqs, batch, seq_len, n_heads, head_dim, stream)` |
| `cuda_mqa_attention` | `(Q, K, V, output, mask, batch, seq_len, n_heads, head_dim, scale, stream)` |
| `cuda_flash_attention` | `(Q, K, V, output, batch, seq_len, n_heads, head_dim, scale, is_causal, stream)` |

### Training

| Function | Signature |
|----------|-----------|
| `cuda_cross_entropy_forward` | `(logits, targets, loss, log_probs, batch, vocab_size, stream)` |
| `cuda_adamw_step` | `(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step, size, stream)` |

### Decoding

| Function | Signature |
|----------|-----------|
| `cuda_argmax` | `(logits, output, batch, vocab_size, stream)` |
| `cuda_sample` | `(logits, output, seeds, batch, vocab_size, temperature, stream)` |
| `cuda_topk_sample` | `(logits, output, seeds, batch, vocab_size, k, temperature, stream)` |
| `cuda_topp_sample` | `(logits, output, seeds, batch, vocab_size, top_p, temperature, stream)` |

## Adding New Kernels

### 1. Implement in kernels/*.cu

```cuda
// cuda/kernels/new_op.cu
extern "C" int32_t cuda_new_op(
    const float* input,
    float* output,
    int64_t n,
    void* stream
) {
    // CUDA implementation
    return 0;
}
```

### 2. Add stub in stub.c

```c
STUB_IMPL(cuda_new_op, const float* input, float* output, int64_t n, void* stream)
```

### 3. Add bindings in each language

**Rust (nn-cuda/src/lib.rs):**
```rust
extern "C" {
    fn cuda_new_op(input: *const f32, output: *mut f32, n: i64, stream: *mut c_void) -> i32;
}
```

**Go (go/cuda/cuda.go):**
```go
/*
extern int32_t cuda_new_op(const float* input, float* output, int64_t n, void* stream);
*/
import "C"
```

**Python (python/cuda/bindings.py):**
```python
def new_op(input_arr, output):
    if _lib is not None:
        _check_result(_lib.cuda_new_op(...))
    else:
        # CPU fallback
        ...
```

## Testing

各言語でstub動作を検証:

```bash
# Rust
cargo test -p nn-cuda

# Go
cd go/cuda && make && go test -v ./...

# Python
pytest python/cuda/test_bindings.py -v
```

## GPU Architectures

Supported compute capabilities (Makefile/build.rs):

| Architecture | Compute | GPUs |
|-------------|---------|------|
| Volta | 7.0 | V100 |
| Turing | 7.5 | RTX 20xx, T4 |
| Ampere | 8.0, 8.6 | A100, RTX 30xx |
| Ada Lovelace | 8.9 | RTX 40xx |
| Hopper | 9.0 | H100 |

---

## FFI Technical Reference

### Memory Layout Requirements

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Alignment                         │
├─────────────────────────────────────────────────────────────┤
│  float*   → 4-byte aligned (alignof(float) = 4)             │
│  int32_t* → 4-byte aligned                                  │
│  int64_t* → 8-byte aligned                                  │
│  void*    → pointer-size aligned (8 bytes on 64-bit)        │
└─────────────────────────────────────────────────────────────┘
```

**Alignment Guarantees:**
- CUDA `cudaMalloc` returns 256-byte aligned memory
- Host arrays must be contiguous (C-order, not Fortran-order)
- numpy/Go/Rust slices are naturally aligned

### Data Type Mapping

| C Type | Rust | Go | Python (ctypes) | Size |
|--------|------|-----|-----------------|------|
| `float` | `f32` | `float32` | `c_float` | 4 bytes |
| `int32_t` | `i32` | `int32` | `c_int32` | 4 bytes |
| `int64_t` | `i64` | `int64` | `c_int64` | 8 bytes |
| `uint64_t` | `u64` | `uint64` | `c_uint64` | 8 bytes |
| `void*` | `*mut c_void` | `unsafe.Pointer` | `c_void_p` | 8 bytes |

### Return Code Convention

```c
// Return codes (all functions)
#define CUDA_SUCCESS           0   // Operation completed successfully
#define CUDA_NOT_AVAILABLE    -1   // CUDA not installed / stub mode
#define CUDA_INVALID_VALUE    -2   // Invalid argument
#define CUDA_LAUNCH_FAILED    -3   // Kernel launch failed
#define CUDA_OUT_OF_MEMORY    -4   // GPU memory exhausted
```

**Error Handling Strategy:**
```
┌─────────────────────────────────────────────────────────────┐
│  Caller (Rust/Go/Python)                                    │
│  ├── Check return code immediately                          │
│  ├── code == 0  → Success, output buffer is valid           │
│  ├── code == -1 → Stub mode, use CPU fallback               │
│  └── code < -1  → Error, do NOT read output buffer          │
└─────────────────────────────────────────────────────────────┘
```

### Thread Safety

| Scenario | Safe? | Notes |
|----------|-------|-------|
| Multiple threads, same stream | No | Serialize with mutex |
| Multiple threads, different streams | Yes | Streams are independent |
| Multiple threads, stream=NULL | No | Default stream is shared |
| Single thread, multiple calls | Yes | Sequential execution |

**Recommendation:** Create one CUDA stream per thread for parallel execution.

```c
// Thread-safe pattern
void* stream_per_thread = create_stream();
cuda_silu(input, output, n, stream_per_thread);
synchronize(stream_per_thread);
```

### Memory Ownership

```
┌─────────────────────────────────────────────────────────────┐
│                   Ownership Rules                           │
├─────────────────────────────────────────────────────────────┤
│  1. Caller allocates ALL buffers (input, output, temp)      │
│  2. Kernel does NOT allocate memory                         │
│  3. Kernel does NOT free memory                             │
│  4. Caller must ensure buffers outlive kernel execution     │
│  5. Async kernels: buffers must outlive synchronization     │
└─────────────────────────────────────────────────────────────┘
```

**Anti-pattern (dangerous):**
```c
void bad_example() {
    float* temp = malloc(1024);
    cuda_silu(temp, output, 256, stream);
    free(temp);  // BUG: kernel may still be running!
}
```

**Correct pattern:**
```c
void good_example() {
    float* temp = malloc(1024);
    cuda_silu(temp, output, 256, stream);
    cudaStreamSynchronize(stream);  // Wait for completion
    free(temp);  // Safe: kernel finished
}
```

### Common Pitfalls

#### 1. Signature Mismatch

```c
// stub.c declares:
int32_t cuda_scale(const float* input, float* output, float scale, int64_t n, void* stream);

// Go declares (WRONG order):
extern int32_t cuda_scale(const float* input, float scale, float* output, int64_t n, void* stream);
// → Linker succeeds but runtime corruption!
```

**Prevention:** Single source of truth in header file, auto-generate bindings.

#### 2. Buffer Size Mismatch

```c
// Caller allocates 100 floats
float output[100];
// But passes n=200
cuda_silu(input, output, 200, stream);  // Buffer overflow!
```

**Prevention:** Validate `n <= buffer_size / sizeof(float)` before FFI call.

#### 3. Uninitialized Output Buffers

```c
float* output;  // Uninitialized pointer
cuda_silu(input, output, n, stream);  // Segfault or corruption
```

**Prevention:** Always allocate before FFI call.

#### 4. Mixing Host and Device Pointers

```c
float* host_ptr = malloc(1024);
float* device_ptr;
cudaMalloc(&device_ptr, 1024);

// WRONG: passing host pointer where device expected
cuda_silu(host_ptr, device_ptr, 256, stream);  // Crash
```

**Prevention:** Document pointer requirements clearly. Use separate types if possible.

### Performance Considerations

#### Kernel Launch Overhead

```
┌─────────────────────────────────────────────────────────────┐
│  Operation                          │ Typical Latency       │
├─────────────────────────────────────┼───────────────────────┤
│  FFI call overhead (Rust/Go)        │ ~10-50 ns             │
│  CUDA kernel launch                 │ ~5-10 μs              │
│  Small kernel execution             │ ~10-100 μs            │
│  cudaStreamSynchronize              │ ~1-5 μs (if no work)  │
└─────────────────────────────────────┴───────────────────────┘
```

**Optimization:** Batch small operations to amortize launch overhead.

#### Memory Transfer

```
┌─────────────────────────────────────────────────────────────┐
│  PCIe 4.0 x16 Bandwidth: ~25 GB/s                           │
│  GPU Memory Bandwidth: ~900 GB/s (A100)                     │
│                                                             │
│  Rule: Minimize Host ↔ Device transfers                     │
│  Keep data on GPU across multiple kernel calls              │
└─────────────────────────────────────────────────────────────┘
```

### Debugging FFI Issues

#### 1. Symbol Not Found

```
error: undefined reference to `cuda_new_op`
```

**Cause:** Function not in stub.c or kernel not compiled.
**Fix:** Add to stub.c, rebuild library.

#### 2. Segmentation Fault

```
SIGSEGV in cuda_silu
```

**Cause:** NULL pointer, misaligned pointer, or buffer overflow.
**Fix:** Add NULL checks, validate alignment, check buffer sizes.

#### 3. Wrong Results (Silent Corruption)

**Cause:** Signature mismatch between declaration and implementation.
**Fix:** Verify exact parameter order and types match.

#### 4. Memory Leak

**Cause:** CUDA resources not freed (streams, memory).
**Fix:** Call `cudaFree`, `cudaStreamDestroy` on cleanup.

### Versioning and ABI Stability

```
┌─────────────────────────────────────────────────────────────┐
│                   ABI Stability Rules                       │
├─────────────────────────────────────────────────────────────┤
│  ✓ Adding new functions          → ABI compatible           │
│  ✓ Adding parameters at end      → ABI compatible*          │
│  ✗ Removing functions            → ABI break                │
│  ✗ Changing parameter order      → ABI break                │
│  ✗ Changing parameter types      → ABI break                │
│  ✗ Changing return type          → ABI break                │
├─────────────────────────────────────────────────────────────┤
│  * Only if callers updated simultaneously                   │
└─────────────────────────────────────────────────────────────┘
```

**Recommendation:** Version the API, deprecate before removing.
