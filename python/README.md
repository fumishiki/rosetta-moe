# Python Implementation

MoE Transformer (6.9B/1.8B) の Python 実装。

## Structure

```
python/
├── nn/
│   ├── tensor.py    # Tensor operations (numpy backend)
│   ├── layers.py    # NN layers (Embedding, RMSNorm, Linear, SwiGLU)
│   ├── model.py     # Model (Attention, Router, MoE, Transformer)
│   └── train.py     # Training (Trainer, AdamW, LR scheduler)
├── cuda/
│   └── bindings.py  # ctypes CUDA bindings (with CPU fallback)
├── tests/           # pytest tests
└── pyproject.toml   # Package config
```

## Install

```bash
cd python
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Usage

```python
from nn import MoETransformer, TrainConfig, Trainer

# Create tiny model for testing
model = MoETransformer.tiny()

# Forward pass
token_ids = [1, 2, 3, 4]
logits = model.forward_ids(token_ids, batch=1, seq_len=4)
print(f"Output shape: {logits.shape}")

# Training
config = TrainConfig.default()
trainer = Trainer(model, config)
# ... training loop
```

## CUDA Bindings

The `cuda` package provides Python bindings to shared CUDA kernels:

| Function | Description |
|----------|-------------|
| silu | SiLU activation |
| add/mul/scale | Element-wise ops |
| softmax | Softmax |
| rmsnorm | RMS normalization |
| gemm | Matrix multiplication |
| cross_entropy_forward | Loss computation |
| adamw_step | Optimizer step |
| argmax | Greedy decoding |
| sample | Multinomial sampling |
| topk_sample | Top-k sampling |
| topp_sample | Nucleus sampling |

All functions have CPU fallback when CUDA is not available.

## FFI Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Python Application                        │
├─────────────────────────────────────────────────────────────┤
│  python/cuda/bindings.py                                    │
│  ├── ctypes.CDLL loading (.so/.dylib)                       │
│  ├── Python wrapper functions (silu, add, gemm, etc.)       │
│  ├── numpy ↔ C pointer conversion (_as_float_ptr)           │
│  └── CPU fallback implementations                           │
├─────────────────────────────────────────────────────────────┤
│  ../go/cuda/lib/libcudann.{so,dylib}                        │
│  └── Shared library (reuses Go build)                       │
├─────────────────────────────────────────────────────────────┤
│  ../cuda/kernels/*.cu  OR  ../cuda/stub.c                   │
│  └── Actual implementations                                 │
└─────────────────────────────────────────────────────────────┘
```

### ctypes Binding Details

```python
import ctypes
from ctypes import c_float, c_int64, POINTER

_lib = ctypes.CDLL("path/to/libcudann.so")

def _as_float_ptr(arr: np.ndarray) -> POINTER(c_float):
    return arr.ctypes.data_as(POINTER(c_float))

def silu(input_arr, output):
    if _lib is not None:
        _check_result(_lib.cuda_silu(
            _as_float_ptr(input_arr),
            _as_float_ptr(output),
            c_int64(input_arr.size),
            None,  # stream
        ))
    else:
        # CPU fallback
        sigmoid = 1.0 / (1.0 + np.exp(-input_arr))
        np.copyto(output, input_arr * sigmoid)
```

**Key Points:**
- ctypes requires **shared library** (.so/.dylib), not static (.a)
- numpy arrays converted via `.ctypes.data_as(POINTER(c_float))`
- CPU fallback when library not loaded (CUDA unavailable)
- All FFI functions return `int32_t` (0=success, -1=not available)

### Library Loading Priority

```python
_lib_paths = [
    Path(__file__).parent.parent.parent / "go/cuda/lib/libcudann.so",
    Path(__file__).parent.parent.parent / "go/cuda/lib/libcudann.dylib",
]
```

**Note:** Static libraries (`.a`) cannot be loaded by ctypes. Build a shared library for Python use, or rely on CPU fallback.

### Error Handling

```python
class CudaError(Exception):
    """CUDA operation error."""
    pass

def _check_result(result: int) -> None:
    if result != 0:
        raise CudaError("CUDA operation failed")
```

### Testing

```bash
pytest python/cuda/test_bindings.py -v
```

Tests verify:
1. Module imports correctly (CudaError, all functions)
2. CPU fallback implementations produce correct results
3. Edge cases (zero input, numerical stability, small values)

## Dependencies

- Python 3.10+
- numpy >= 1.24.0
- pytest (for testing)

---

## FFI Technical Reference

### ctypes Fundamentals

```
┌─────────────────────────────────────────────────────────────┐
│                   ctypes Call Flow                          │
├─────────────────────────────────────────────────────────────┤
│  Python code                                                │
│    ↓                                                        │
│  ctypes wrapper (type conversion, GIL handling)             │
│    ↓                                                        │
│  libffi (Foreign Function Interface library)                │
│    ↓                                                        │
│  C function in shared library                               │
│    ↓                                                        │
│  Return to Python                                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Difference from Rust/Go:** ctypes uses dynamic linking at runtime, not compile time.

### Library Loading

```python
import ctypes
from pathlib import Path

class LibraryLoader:
    """Cross-platform library loader."""

    @staticmethod
    def load() -> ctypes.CDLL | None:
        """Load CUDA library with fallback."""
        lib_names = {
            "linux": "libcudann.so",
            "darwin": "libcudann.dylib",
            "win32": "cudann.dll",
        }

        import sys
        lib_name = lib_names.get(sys.platform)
        if lib_name is None:
            return None

        # Search paths (priority order)
        search_paths = [
            Path(__file__).parent / "lib" / lib_name,      # Local
            Path(__file__).parent.parent.parent / "go/cuda/lib" / lib_name,
            Path(f"/usr/local/lib/{lib_name}"),            # System
        ]

        for path in search_paths:
            if path.exists():
                try:
                    return ctypes.CDLL(str(path))
                except OSError as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    continue

        return None  # Fall back to CPU

# Module-level loading
_lib = LibraryLoader.load()
```

### Static vs Shared Libraries

```
┌─────────────────────────────────────────────────────────────┐
│                Library Type Comparison                      │
├─────────────────────────────────────────────────────────────┤
│  Static (.a)          │  Shared (.so/.dylib/.dll)          │
├───────────────────────┼─────────────────────────────────────┤
│  Linked at compile    │  Loaded at runtime                  │
│  Rust/Go: ✓           │  Python (ctypes): ✓                 │
│  Python: ✗            │  Rust/Go: ✓ (also supported)        │
│  Larger binary        │  Smaller binary                     │
│  No runtime deps      │  Library must exist at runtime      │
└─────────────────────────────────────────────────────────────┘
```

**Building Shared Library for Python:**

```bash
# Linux
gcc -shared -fPIC -o libcudann.so stub.c

# macOS
gcc -shared -fPIC -o libcudann.dylib stub.c

# With CUDA (Linux)
nvcc -shared -Xcompiler -fPIC -o libcudann.so kernels/*.cu
```

### Type Mapping

```python
from ctypes import (
    c_float,    # float → 4 bytes
    c_int,      # int → platform-dependent (usually 4 bytes)
    c_int32,    # int32_t → 4 bytes
    c_int64,    # int64_t → 8 bytes
    c_uint64,   # uint64_t → 8 bytes
    c_void_p,   # void* → pointer size (8 bytes on 64-bit)
    POINTER,    # Creates pointer type
)

# Type mapping table
TYPE_MAP = {
    "float*": POINTER(c_float),
    "const float*": POINTER(c_float),  # ctypes doesn't distinguish const
    "int32_t*": POINTER(c_int32),
    "int64_t": c_int64,
    "int": c_int,
    "float": c_float,
    "void*": c_void_p,
}
```

### numpy Integration

```python
import numpy as np
from numpy.typing import NDArray

def _as_float_ptr(arr: NDArray[np.float32]) -> POINTER(c_float):
    """Convert numpy array to C float pointer.

    Requirements:
    - Array must be contiguous (C-order)
    - Array must be float32 dtype
    - Array must not be empty
    """
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError("Array must be C-contiguous")
    if arr.dtype != np.float32:
        raise ValueError(f"Expected float32, got {arr.dtype}")
    if arr.size == 0:
        return None  # Handle empty arrays

    return arr.ctypes.data_as(POINTER(c_float))

# Safe wrapper with validation
def silu(input_arr: NDArray[np.float32], output: NDArray[np.float32]) -> None:
    """SiLU activation with safety checks."""
    # Validate inputs
    if input_arr.shape != output.shape:
        raise ValueError(f"Shape mismatch: {input_arr.shape} vs {output.shape}")

    if not input_arr.flags['C_CONTIGUOUS']:
        input_arr = np.ascontiguousarray(input_arr)

    if not output.flags['C_CONTIGUOUS']:
        raise ValueError("Output must be C-contiguous (cannot copy)")

    if _lib is not None:
        _check_result(_lib.cuda_silu(
            _as_float_ptr(input_arr),
            _as_float_ptr(output),
            c_int64(input_arr.size),
            None,
        ))
    else:
        # CPU fallback
        sigmoid = 1.0 / (1.0 + np.exp(-input_arr))
        np.copyto(output, input_arr * sigmoid)
```

### Memory Layout Requirements

```python
# C-order (row-major) vs F-order (column-major)

# C-order (default in numpy, required for C FFI)
c_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order='C')
# Memory: [1, 2, 3, 4, 5, 6]

# F-order (Fortran, common in scientific computing)
f_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order='F')
# Memory: [1, 4, 2, 5, 3, 6]

# Check and convert
def ensure_c_contiguous(arr: np.ndarray) -> np.ndarray:
    if arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr)

# DANGER: Views may not be contiguous
a = np.zeros((10, 10), dtype=np.float32)
view = a[::2, ::2]  # Non-contiguous view!
assert not view.flags['C_CONTIGUOUS']
```

### GIL and Threading

```python
from concurrent.futures import ThreadPoolExecutor
import threading

# GIL (Global Interpreter Lock) considerations
#
# ctypes releases the GIL during C function calls by default.
# This allows true parallelism in C code.

def parallel_silu(inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
    """Run SiLU in parallel threads."""
    def worker(i: int) -> None:
        silu(inputs[i], outputs[i])

    with ThreadPoolExecutor(max_workers=4) as executor:
        # C code runs in parallel (GIL released)
        list(executor.map(worker, range(len(inputs))))

# Explicit GIL control (advanced)
# If C code calls back into Python, you need to acquire GIL:
#
# PyGILState_STATE gstate = PyGILState_Ensure();
# // Python API calls here
# PyGILState_Release(gstate);
```

### Function Signature Declaration

```python
# Explicit type declarations for safety and performance
def setup_function_signatures(lib: ctypes.CDLL) -> None:
    """Declare function signatures for type checking."""

    # cuda_silu(const float*, float*, int64_t, void*) -> int32_t
    lib.cuda_silu.argtypes = [
        POINTER(c_float),  # input
        POINTER(c_float),  # output
        c_int64,           # n
        c_void_p,          # stream
    ]
    lib.cuda_silu.restype = c_int32

    # cuda_gemm(float*, float*, float*, int, int, int, float, float, void*) -> int32_t
    lib.cuda_gemm.argtypes = [
        POINTER(c_float),  # A
        POINTER(c_float),  # B
        POINTER(c_float),  # C
        c_int,             # M
        c_int,             # N
        c_int,             # K
        c_float,           # alpha
        c_float,           # beta
        c_void_p,          # stream
    ]
    lib.cuda_gemm.restype = c_int32

# Apply signatures after loading
if _lib is not None:
    setup_function_signatures(_lib)
```

### Error Handling Patterns

```python
from enum import IntEnum
from typing import NoReturn

class CudaErrorCode(IntEnum):
    SUCCESS = 0
    NOT_AVAILABLE = -1
    INVALID_VALUE = -2
    LAUNCH_FAILED = -3
    OUT_OF_MEMORY = -4

class CudaError(Exception):
    """CUDA operation error with code."""

    def __init__(self, code: int, message: str | None = None):
        self.code = CudaErrorCode(code) if code in CudaErrorCode._value2member_map_ else code
        self.message = message or self._default_message()
        super().__init__(f"CUDA error {self.code}: {self.message}")

    def _default_message(self) -> str:
        messages = {
            CudaErrorCode.NOT_AVAILABLE: "CUDA not available",
            CudaErrorCode.INVALID_VALUE: "Invalid argument",
            CudaErrorCode.LAUNCH_FAILED: "Kernel launch failed",
            CudaErrorCode.OUT_OF_MEMORY: "Out of GPU memory",
        }
        return messages.get(self.code, "Unknown error")

def _check_result(result: int) -> None:
    """Check CUDA result code and raise on error."""
    if result != 0:
        raise CudaError(result)

# Usage with context manager
from contextlib import contextmanager

@contextmanager
def cuda_error_context(operation: str):
    """Context manager for better error messages."""
    try:
        yield
    except CudaError as e:
        raise CudaError(e.code, f"{operation}: {e.message}") from e
```

### CPU Fallback Strategy

```python
def cuda_available() -> bool:
    """Check if CUDA library is loaded."""
    return _lib is not None

# Strategy 1: Automatic fallback (current implementation)
def silu_auto(input_arr: np.ndarray, output: np.ndarray) -> None:
    if _lib is not None:
        _check_result(_lib.cuda_silu(...))
    else:
        # CPU fallback
        sigmoid = 1.0 / (1.0 + np.exp(-input_arr))
        np.copyto(output, input_arr * sigmoid)

# Strategy 2: Explicit fallback (user controls)
def silu_gpu(input_arr: np.ndarray, output: np.ndarray) -> None:
    if _lib is None:
        raise CudaError(CudaErrorCode.NOT_AVAILABLE)
    _check_result(_lib.cuda_silu(...))

def silu_cpu(input_arr: np.ndarray, output: np.ndarray) -> None:
    sigmoid = 1.0 / (1.0 + np.exp(-input_arr))
    np.copyto(output, input_arr * sigmoid)

# Strategy 3: Hybrid with preference
def silu(input_arr: np.ndarray, output: np.ndarray, prefer_gpu: bool = True) -> bool:
    """Returns True if GPU was used, False if CPU fallback."""
    if prefer_gpu and _lib is not None:
        try:
            _check_result(_lib.cuda_silu(...))
            return True
        except CudaError:
            pass  # Fall through to CPU
    silu_cpu(input_arr, output)
    return False
```

### Common Pitfalls

#### 1. Dtype Mismatch

```python
# WRONG: float64 passed to C expecting float32
arr = np.array([1.0, 2.0, 3.0])  # Default float64!
silu(arr, output)  # Corrupted data!

# CORRECT: Explicit dtype
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
silu(arr, output)

# Even better: Type hints and validation
def silu(input_arr: NDArray[np.float32], ...) -> None:
    assert input_arr.dtype == np.float32
    ...
```

#### 2. Non-Contiguous Arrays

```python
# WRONG: Sliced array may not be contiguous
data = np.zeros((100, 100), dtype=np.float32)
slice_view = data[::2, ::2]  # Non-contiguous!
silu(slice_view, output)  # Garbage results!

# CORRECT: Force contiguous
slice_copy = np.ascontiguousarray(slice_view)
silu(slice_copy, output)
```

#### 3. Garbage Collection Race

```python
# WRONG: Array may be collected during C call
def bad():
    arr = np.array([1, 2, 3], dtype=np.float32)
    ptr = arr.ctypes.data_as(POINTER(c_float))
    del arr  # Array freed!
    return ptr  # Dangling pointer!

# CORRECT: Keep reference alive
def good():
    arr = np.array([1, 2, 3], dtype=np.float32)
    # Pass directly to C function in same scope
    _lib.cuda_silu(arr.ctypes.data_as(POINTER(c_float)), ...)
    # arr still in scope, not collected
```

#### 4. Buffer Size Mismatch

```python
# WRONG: Output buffer too small
input_arr = np.zeros(100, dtype=np.float32)
output = np.zeros(50, dtype=np.float32)  # Too small!
silu(input_arr, output)  # Buffer overflow!

# CORRECT: Validate sizes
def silu_safe(input_arr: np.ndarray, output: np.ndarray) -> None:
    if input_arr.size != output.size:
        raise ValueError(f"Size mismatch: {input_arr.size} vs {output.size}")
    ...
```

### Performance Considerations

```python
import time

# ctypes overhead: ~1-10μs per call
# numpy operation overhead: ~1-5μs per call
# Actual computation (depending on size): varies

def benchmark_overhead():
    """Measure ctypes call overhead."""
    arr = np.zeros(10, dtype=np.float32)
    out = np.zeros(10, dtype=np.float32)

    # Warm up
    for _ in range(100):
        silu(arr, out)

    # Measure
    n = 10000
    start = time.perf_counter()
    for _ in range(n):
        silu(arr, out)
    elapsed = time.perf_counter() - start

    print(f"Average call time: {elapsed/n*1e6:.2f}μs")

# Optimization: Batch small operations
def silu_batched(inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
    """Batch multiple arrays into single call."""
    # Concatenate
    all_input = np.concatenate(inputs)
    all_output = np.concatenate(outputs)

    # Single FFI call
    silu(all_input, all_output)

    # Split back (views, no copy)
    offset = 0
    for i, inp in enumerate(inputs):
        outputs[i][:] = all_output[offset:offset+inp.size]
        offset += inp.size
```

### Testing Best Practices

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestCudaBindings:
    """Test suite for CUDA Python bindings."""

    def test_silu_basic(self):
        """Test basic SiLU computation."""
        input_arr = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        output = np.zeros_like(input_arr)

        silu(input_arr, output)

        # Compare with reference implementation
        expected = input_arr * (1.0 / (1.0 + np.exp(-input_arr)))
        assert_allclose(output, expected, rtol=1e-5)

    def test_silu_numerical_stability(self):
        """Test with extreme values."""
        input_arr = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
        output = np.zeros_like(input_arr)

        silu(input_arr, output)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_dtype_validation(self):
        """Test that wrong dtype raises error."""
        input_arr = np.array([1.0, 2.0], dtype=np.float64)  # Wrong!
        output = np.zeros(2, dtype=np.float32)

        with pytest.raises((ValueError, TypeError)):
            silu(input_arr, output)

    def test_contiguity_validation(self):
        """Test that non-contiguous arrays are handled."""
        data = np.zeros((10, 10), dtype=np.float32)
        non_contiguous = data[::2, ::2]

        # Should either:
        # 1. Raise ValueError, or
        # 2. Automatically make contiguous copy
        output = np.zeros(non_contiguous.shape, dtype=np.float32)
        silu(non_contiguous, output.ravel())  # May need to handle

    @pytest.mark.parametrize("size", [0, 1, 100, 10000])
    def test_various_sizes(self, size: int):
        """Test with different input sizes."""
        input_arr = np.random.randn(size).astype(np.float32)
        output = np.zeros_like(input_arr)

        silu(input_arr, output)

        if size > 0:
            expected = input_arr * (1.0 / (1.0 + np.exp(-input_arr)))
            assert_allclose(output, expected, rtol=1e-5)
```
