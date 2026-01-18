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
    "github.com/pikafumi/machine_learning/crates/go/model"
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

## Notes

- CPU implementation works without CUDA (stub returns error)
- GPU functions require CUDA library to be built with nvcc
