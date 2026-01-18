# Go Implementation

MoE Transformer (6.9B/1.8B) の Go 実装。

## Structure

```
go/
├── tensor/     # Tensor operations (Shape, DType, Tensor)
├── cuda/       # cgo CUDA bindings + Makefile
├── layer/      # NN layers (Embedding, RMSNorm, Linear, SwiGLU)
├── model/      # Model (Attention, Router, MoE, Transformer)
├── train/      # Training (Trainer, AdamW, LR scheduler)
└── go.mod      # Module definition
```

## Build

### 1. Build CUDA Library

```bash
cd cuda
make
```

This builds `lib/libcudann.a`:
- **With CUDA**: Compiles `.cu` kernels from `../../cuda/kernels/`
- **Without CUDA**: Compiles `stub.c` (CPU fallback)

### 2. Build Go

```bash
go build ./...
```

### 3. Test

```bash
go test ./...
```

## Dependencies

- Go 1.22+
- GCC (for cgo)
- CUDA Toolkit (optional, for GPU acceleration)

## Usage

```go
package main

import (
    "fmt"
    "github.com/fumi-engineer/machine_learning/go/model"
)

func main() {
    // Create tiny model for testing
    m := model.NewTiny()

    // Forward pass
    tokenIDs := []int{1, 2, 3, 4}
    logits := m.ForwardIDs(tokenIDs, 1, 4)

    fmt.Printf("Output shape: %v\n", logits.Shape())
}
```

## CUDA Functions

The `cuda` package provides Go bindings to:

| Function | Description |
|----------|-------------|
| SiLU | SiLU activation |
| Add/Mul/Scale | Element-wise ops |
| Softmax | Softmax |
| RMSNorm | RMS normalization |
| GEMM | Matrix multiplication |
| CrossEntropyForward | Loss computation |
| AdamWStep | Optimizer step |
| Argmax | Greedy decoding |
| Sample | Multinomial sampling |
| TopKSample | Top-k sampling |
| TopPSample | Nucleus sampling |

## FFI Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Go Application                          │
├─────────────────────────────────────────────────────────────┤
│  go/cuda/cuda.go                                            │
│  ├── cgo declarations (extern "C" function prototypes)      │
│  ├── Go wrapper functions (SiLU, Add, GEMM, etc.)           │
│  └── Error handling (ErrCudaNotAvailable)                   │
├─────────────────────────────────────────────────────────────┤
│  go/cuda/lib/libcudann.a                                    │
│  └── Static library (CUDA kernels or stub)                  │
├─────────────────────────────────────────────────────────────┤
│  ../../cuda/kernels/*.cu  OR  ../../cuda/stub.c             │
│  └── Actual implementations                                 │
└─────────────────────────────────────────────────────────────┘
```

### cgo Binding Details

```go
/*
#cgo CFLAGS: -I${SRCDIR}/../../cuda/kernels
#cgo LDFLAGS: -L${SRCDIR}/lib -lcudann

extern int32_t cuda_silu(const float* input, float* output, int64_t n, void* stream);
*/
import "C"
```

**Key Points:**
- `#cgo CFLAGS`: Include path for kernel headers
- `#cgo LDFLAGS`: Link against `libcudann.a`
- All FFI functions return `int32_t` (0=success, -1=not available)
- `void* stream` parameter for CUDA stream (nil for default)

### Error Handling

```go
var ErrCudaNotAvailable = errors.New("CUDA not available")

func checkResult(code C.int32_t) error {
    if code != 0 {
        return ErrCudaNotAvailable
    }
    return nil
}
```

### Testing

```bash
# Build stub library and run tests
cd go/cuda && make && go test -v ./...
```

Tests verify:
1. Input validation (length mismatch errors)
2. Stub returns `ErrCudaNotAvailable` when CUDA unavailable
3. All exported functions have correct signatures

## Notes

- CPU implementation works without CUDA (stub returns error)
- GPU functions require CUDA library to be built with nvcc
- Static library `.a` is linked at compile time via cgo

---

## FFI Technical Reference

### cgo Fundamentals

```
┌─────────────────────────────────────────────────────────────┐
│                     cgo Call Flow                           │
├─────────────────────────────────────────────────────────────┤
│  Go code                                                    │
│    ↓                                                        │
│  cgo runtime (stack switch, GC coordination)                │
│    ↓                                                        │
│  C function call                                            │
│    ↓                                                        │
│  Return to cgo runtime                                      │
│    ↓                                                        │
│  Go code continues                                          │
└─────────────────────────────────────────────────────────────┘
```

**Overhead:** Each cgo call costs ~100-200ns due to:
- Stack switching (Go stack → C stack)
- Saving/restoring Go runtime state
- Potential GC coordination

### Memory Pinning

```go
// CRITICAL: Go slices must not move during C call
// The Go runtime may move memory during GC

// SAFE: Pointer derived from slice element
func SiLU(input, output []float32, stream Stream) error {
    // &input[0] pins the backing array during this call
    return checkResult(C.cuda_silu(
        (*C.float)(&input[0]),   // Pinned by cgo
        (*C.float)(&output[0]),  // Pinned by cgo
        C.int64_t(len(input)),
        unsafe.Pointer(stream),
    ))
}

// UNSAFE: Don't store pointer for later use
func Bad() *C.float {
    data := []float32{1, 2, 3}
    ptr := (*C.float)(&data[0])
    return ptr  // DANGER: data may move after function returns!
}
```

**cgo Pointer Rules:**
1. Go pointer passed to C is valid only during the call
2. C cannot store Go pointers beyond the call duration
3. Go cannot pass pointers to Go memory that itself contains pointers

### Type Conversions

```go
import "C"
import "unsafe"

// Slice to C pointer
func sliceToPtr(s []float32) *C.float {
    if len(s) == 0 {
        return nil  // Handle empty slice
    }
    return (*C.float)(&s[0])
}

// Go types to C types
var (
    _ C.float   = C.float(float32(0))    // float32 → C.float
    _ C.int     = C.int(int32(0))        // int32 → C.int
    _ C.int32_t = C.int32_t(int32(0))    // int32 → C.int32_t
    _ C.int64_t = C.int64_t(int64(0))    // int64 → C.int64_t
    _ C.uint64_t = C.uint64_t(uint64(0)) // uint64 → C.uint64_t
)

// Opaque pointer (void*)
func ptrToVoid(p unsafe.Pointer) unsafe.Pointer {
    return p  // Go's unsafe.Pointer ≡ C's void*
}
```

### Goroutine Considerations

```
┌─────────────────────────────────────────────────────────────┐
│              Goroutines and cgo                             │
├─────────────────────────────────────────────────────────────┤
│  ✓ Multiple goroutines can call cgo concurrently           │
│  ✓ Each cgo call gets its own OS thread                    │
│  ✗ cgo calls block the OS thread (not just the goroutine)  │
│  ⚠ Many concurrent cgo calls = many OS threads             │
└─────────────────────────────────────────────────────────────┘
```

```go
// Pattern: Limit concurrent cgo calls
var cgoSemaphore = make(chan struct{}, runtime.NumCPU())

func SiLUThrottled(input, output []float32, stream Stream) error {
    cgoSemaphore <- struct{}{}        // Acquire
    defer func() { <-cgoSemaphore }() // Release

    return SiLU(input, output, stream)
}
```

### CUDA Stream Management

```go
// Stream represents a CUDA stream (nil for default stream)
type Stream unsafe.Pointer

// DefaultStream is the default CUDA stream
var DefaultStream Stream = nil

// Best Practice: One stream per goroutine for parallel execution
func ParallelKernels(inputs [][]float32, outputs [][]float32) error {
    var wg sync.WaitGroup
    errCh := make(chan error, len(inputs))

    for i := range inputs {
        wg.Add(1)
        go func(idx int) {
            defer wg.Done()
            // Each goroutine uses DefaultStream
            // Kernels execute sequentially on GPU
            if err := SiLU(inputs[idx], outputs[idx], DefaultStream); err != nil {
                errCh <- err
            }
        }(i)
    }

    wg.Wait()
    close(errCh)

    for err := range errCh {
        if err != nil {
            return err
        }
    }
    return nil
}
```

### Error Handling Patterns

```go
// Sentinel error for CUDA unavailable
var ErrCudaNotAvailable = errors.New("CUDA not available")

// Detailed error type (optional)
type CudaError struct {
    Code    int
    Message string
}

func (e CudaError) Error() string {
    return fmt.Sprintf("CUDA error %d: %s", e.Code, e.Message)
}

// Error code mapping
func checkResultDetailed(code C.int32_t) error {
    switch code {
    case 0:
        return nil
    case -1:
        return ErrCudaNotAvailable
    case -2:
        return CudaError{Code: -2, Message: "invalid argument"}
    case -3:
        return CudaError{Code: -3, Message: "kernel execution failed"}
    case -4:
        return CudaError{Code: -4, Message: "out of GPU memory"}
    default:
        return CudaError{Code: int(code), Message: "unknown error"}
    }
}

// Usage with errors.Is
func Example() {
    err := SiLU(input, output, DefaultStream)
    if errors.Is(err, ErrCudaNotAvailable) {
        // Fall back to CPU implementation
        cpuSiLU(input, output)
    } else if err != nil {
        log.Fatal(err)
    }
}
```

### Input Validation

```go
// Validate before cgo call to prevent buffer overflows
func SiLU(input, output []float32, stream Stream) error {
    // Length validation
    if len(input) != len(output) {
        return errors.New("input and output must have same length")
    }

    // Empty slice check (avoid nil pointer)
    if len(input) == 0 {
        return nil  // No-op for empty input
    }

    return checkResult(C.cuda_silu(
        (*C.float)(&input[0]),
        (*C.float)(&output[0]),
        C.int64_t(len(input)),
        unsafe.Pointer(stream),
    ))
}

// Matrix dimension validation
func GEMM(A, B, C []float32, M, N, K int, alpha, beta float32, stream Stream) error {
    // Validate buffer sizes
    if len(A) < M*K {
        return fmt.Errorf("A buffer too small: need %d, got %d", M*K, len(A))
    }
    if len(B) < K*N {
        return fmt.Errorf("B buffer too small: need %d, got %d", K*N, len(B))
    }
    if len(C) < M*N {
        return fmt.Errorf("C buffer too small: need %d, got %d", M*N, len(C))
    }

    return checkResult(C.cuda_gemm(
        (*C.float)(&A[0]),
        (*C.float)(&B[0]),
        (*C.float)(&C[0]),
        C.int(M), C.int(N), C.int(K),
        C.float(alpha), C.float(beta),
        unsafe.Pointer(stream),
    ))
}
```

### Build System (Makefile)

```makefile
# go/cuda/Makefile

CUDA_PATH ?= /usr/local/cuda
CUDA_KERNELS = ../../cuda/kernels
STUB_SRC = ../../cuda/stub.c

# Detect CUDA
ifeq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
    USE_CUDA = 0
else
    USE_CUDA = 1
endif

# GPU architectures (match Rust build.rs)
GPU_ARCHS = -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_89,code=sm_89

ifeq ($(USE_CUDA),1)
# With CUDA: compile .cu files
$(LIB_DIR)/$(LIB_NAME): $(CUDA_OBJS)
    ar rcs $@ $^
else
# Without CUDA: compile stub
$(LIB_DIR)/$(LIB_NAME): $(LIB_DIR)/stub.o
    ar rcs $@ $^
endif
```

### Common Pitfalls

#### 1. Slice Header vs Data

```go
// WRONG: Passing slice header (struct) instead of data pointer
func bad(data []float32) {
    C.kernel(unsafe.Pointer(&data))  // Passes slice header!
}

// CORRECT: Pass pointer to first element
func good(data []float32) {
    C.kernel((*C.float)(&data[0]))  // Passes data pointer
}
```

#### 2. Empty Slice Panic

```go
// WRONG: Panics on empty slice
func bad(data []float32) {
    C.kernel((*C.float)(&data[0]))  // panic: index out of range
}

// CORRECT: Handle empty slice
func good(data []float32) {
    if len(data) == 0 {
        return nil
    }
    C.kernel((*C.float)(&data[0]))
}
```

#### 3. Goroutine Leak with Long-Running C Calls

```go
// PROBLEM: C call blocks forever
func bad() {
    go func() {
        C.blocking_kernel()  // Never returns
    }()
}

// SOLUTION: Use context for cancellation
func good(ctx context.Context) error {
    done := make(chan error, 1)
    go func() {
        code := C.kernel_with_timeout()
        done <- checkResult(code)
    }()

    select {
    case err := <-done:
        return err
    case <-ctx.Done():
        // Cannot actually cancel C call, but can stop waiting
        return ctx.Err()
    }
}
```

#### 4. CGO_ENABLED=0 Breaks Build

```bash
# This fails because cgo is disabled
CGO_ENABLED=0 go build ./...

# Must enable cgo for CUDA package
CGO_ENABLED=1 go build ./...
```

### Performance Optimization

```go
// Batch operations to amortize cgo overhead
func BatchSiLU(inputs, outputs [][]float32, stream Stream) error {
    for i := range inputs {
        if err := SiLU(inputs[i], outputs[i], stream); err != nil {
            return err
        }
    }
    return nil
}

// Better: Single kernel for batched input
func SiLUBatched(input, output []float32, batchSize, dim int, stream Stream) error {
    // Single cgo call for entire batch
    return checkResult(C.cuda_silu(
        (*C.float)(&input[0]),
        (*C.float)(&output[0]),
        C.int64_t(batchSize * dim),
        unsafe.Pointer(stream),
    ))
}
```

### Testing Patterns

```go
func TestSiLUStub(t *testing.T) {
    input := []float32{1.0, 2.0, 3.0, 4.0}
    output := make([]float32, 4)

    err := SiLU(input, output, DefaultStream)

    // With stub, should return ErrCudaNotAvailable
    if !errors.Is(err, ErrCudaNotAvailable) {
        t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
    }
}

func TestInputValidation(t *testing.T) {
    tests := []struct {
        name   string
        input  []float32
        output []float32
        want   string
    }{
        {"length mismatch", []float32{1, 2, 3}, []float32{0, 0}, "same length"},
        {"empty slices", []float32{}, []float32{}, ""},  // Should succeed (no-op)
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := SiLU(tt.input, tt.output, DefaultStream)
            if tt.want != "" && (err == nil || !strings.Contains(err.Error(), tt.want)) {
                t.Errorf("expected error containing %q, got %v", tt.want, err)
            }
        })
    }
}

// Benchmark cgo overhead
func BenchmarkCgoOverhead(b *testing.B) {
    input := make([]float32, 1024)
    output := make([]float32, 1024)

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = SiLU(input, output, DefaultStream)
    }
}
```
