# Rust Implementation

MoE Transformer (6.9B/1.8B) の Rust 実装。

## Structure

```
rust/
├── nn-core/     # Core tensor ops & model (pure Rust)
├── nn-cuda/     # CUDA FFI bindings (cc/bindgen)
└── nn-ffi/      # FFI bridge connecting nn-core & nn-cuda
```

## Build

```bash
# From workspace root
cargo build --release

# Run tests
cargo test
```

## Usage

```rust
use nn_core::{Tensor, MoETransformer};

fn main() {
    // Create tiny model for testing
    let model = MoETransformer::tiny();

    // Forward pass
    let token_ids = vec![1, 2, 3, 4];
    let logits = model.forward_ids(&token_ids, 1, 4);

    println!("Output shape: {:?}", logits.shape());
}
```

## FFI Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Rust Application                          │
├─────────────────────────────────────────────────────────────┤
│  nn-ffi (GpuTrainer, hybrid training)                       │
│  └── Uses nn-cuda for GPU ops, nn-core for model            │
├─────────────────────────────────────────────────────────────┤
│  nn-cuda/src/lib.rs                                         │
│  ├── extern "C" function bindings                           │
│  ├── Safe Rust wrappers (elementwise, gemm, etc.)           │
│  ├── CudaError enum with is_not_available()                 │
│  └── Stream abstraction                                     │
├─────────────────────────────────────────────────────────────┤
│  nn-cuda/build.rs                                           │
│  ├── CUDA detection (nvcc availability)                     │
│  ├── Kernel compilation with cc crate                       │
│  └── Stub fallback when CUDA unavailable                    │
├─────────────────────────────────────────────────────────────┤
│  ../../cuda/kernels/*.cu  OR  ../../cuda/stub.c             │
│  └── Actual implementations                                 │
└─────────────────────────────────────────────────────────────┘
```

### FFI Binding Details

```rust
// nn-cuda/src/lib.rs

extern "C" {
    fn cuda_silu(
        input: *const f32,
        output: *mut f32,
        n: i64,
        stream: *mut c_void,
    ) -> i32;
}

pub mod elementwise {
    pub fn silu(
        input: *const f32,
        output: *mut f32,
        n: i64,
        stream: Stream,
    ) -> Result<(), CudaError> {
        let result = unsafe { cuda_silu(input, output, n, stream.as_ptr()) };
        CudaError::from_code(result)
    }
}
```

**Key Points:**
- `extern "C"` blocks declare raw FFI functions
- Safe wrappers convert raw result codes to `Result<(), CudaError>`
- `Stream` type wraps raw CUDA stream pointer
- `CudaError::NotAvailable` returned when stub is used

### Build System (build.rs)

```rust
fn main() {
    let cuda_dir = PathBuf::from("../../cuda");

    if cuda_available() {
        // Compile CUDA kernels with nvcc
        cc::Build::new()
            .cuda(true)
            .files(glob("../../cuda/kernels/*.cu"))
            .compile("nn_cuda_kernels");
    } else {
        // Compile stub (returns -1 for all functions)
        cc::Build::new()
            .file(cuda_dir.join("stub.c"))
            .compile("nn_cuda_kernels");
    }
}
```

**Key Points:**
- `cc` crate handles CUDA compilation
- Stub compiled when nvcc not found
- Static library linked into Rust binary

### Error Handling

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaError {
    NotAvailable,
    ExecutionFailed,
    InvalidArgument,
}

impl CudaError {
    pub fn is_not_available(&self) -> bool {
        matches!(self, CudaError::NotAvailable)
    }

    pub fn from_code(code: i32) -> Result<(), Self> {
        match code {
            0 => Ok(()),
            -1 => Err(CudaError::NotAvailable),
            _ => Err(CudaError::ExecutionFailed),
        }
    }
}
```

### Testing

```bash
cargo test -p nn-cuda
```

Tests verify:
1. `CudaError` display and construction
2. `Stream` type safety
3. Stub functions return `CudaError::NotAvailable`
4. All FFI wrapper functions have correct signatures

## Crates

| Crate | Description |
|-------|-------------|
| nn-core | Pure Rust tensor ops, layers, model |
| nn-cuda | CUDA FFI bindings |
| nn-ffi | FFI bridge for GPU-accelerated training |

## Dependencies

- Rust 1.75+ (edition 2024)
- cc crate (for build.rs compilation)
- CUDA Toolkit (optional, for GPU acceleration)

---

## FFI Technical Reference

### Unsafe Boundary Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Safety Hierarchy                         │
├─────────────────────────────────────────────────────────────┤
│  Level 3: Safe Public API (nn-core, nn-ffi)                 │
│    └── No unsafe, fully checked                             │
│  Level 2: Safe Wrappers (nn-cuda pub functions)             │
│    └── Encapsulates unsafe, validates inputs                │
│  Level 1: Raw FFI (extern "C" block)                        │
│    └── unsafe, direct C ABI calls                           │
│  Level 0: C/CUDA Code (stub.c, kernels/*.cu)                │
│    └── Outside Rust's control                               │
└─────────────────────────────────────────────────────────────┘
```

**Design Principle:** Minimize unsafe surface area. All unsafe code concentrated in `nn-cuda/src/lib.rs`.

### Pointer Validity Contract

```rust
// SAFETY requirements for FFI calls:
// 1. input: valid for reads of `n * sizeof(f32)` bytes
// 2. output: valid for writes of `n * sizeof(f32)` bytes
// 3. input and output: properly aligned (4-byte for f32)
// 4. input and output: no aliasing (or both are same buffer)
// 5. Buffers must remain valid until kernel completes

pub fn silu(
    input: *const f32,
    output: *mut f32,
    n: i64,
    stream: Stream,
) -> Result<(), CudaError> {
    // SAFETY: Caller guarantees pointer validity per contract above
    let result = unsafe { cuda_silu(input, output, n, stream.as_ptr()) };
    CudaError::from_code(result)
}
```

### Safe Wrapper Pattern

```rust
/// Safe wrapper that takes slices instead of raw pointers
pub fn silu_safe(input: &[f32], output: &mut [f32], stream: Stream) -> Result<(), CudaError> {
    // Compile-time guarantee: slices are valid, aligned, non-null
    assert_eq!(input.len(), output.len(), "length mismatch");

    // SAFETY:
    // - input.as_ptr(): valid for input.len() elements
    // - output.as_mut_ptr(): valid for output.len() elements
    // - Both derived from slices, guaranteed aligned
    // - No aliasing: &[f32] and &mut [f32] cannot overlap
    unsafe {
        silu(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as i64,
            stream,
        )
    }
}
```

### Panic Safety Across FFI

```
┌─────────────────────────────────────────────────────────────┐
│                 Panic Boundary Rules                        │
├─────────────────────────────────────────────────────────────┤
│  ✗ Panic in Rust called FROM C      → Undefined Behavior    │
│  ✓ Panic in Rust calling INTO C     → Unwinds normally      │
│  ✗ C code calling panic!() macro    → Impossible            │
│  ✓ C code returns error code        → Rust handles safely   │
└─────────────────────────────────────────────────────────────┘
```

**Current Design:** C code never calls back into Rust, so panic safety is guaranteed.

```rust
// Safe: panic before FFI call
fn example(input: &[f32], output: &mut [f32]) -> Result<(), CudaError> {
    if input.len() != output.len() {
        panic!("length mismatch");  // OK: panics in Rust context
    }
    unsafe { silu(input.as_ptr(), output.as_mut_ptr(), input.len() as i64, Stream::DEFAULT) }
}
```

### Lifetime Considerations

```rust
/// Stream borrows the underlying CUDA stream pointer
/// The CUDA stream must outlive the Stream wrapper
pub struct Stream(*mut c_void);

impl Stream {
    /// SAFETY: raw must remain valid for the lifetime of Stream
    pub unsafe fn from_raw(raw: *mut c_void) -> Self {
        Stream(raw)
    }

    /// Default stream (NULL) is always valid
    pub const DEFAULT: Stream = Stream(std::ptr::null_mut());
}

// Lifetime example with async operations
fn async_example<'a>(
    input: &'a [f32],      // Must live until sync
    output: &'a mut [f32], // Must live until sync
    stream: Stream,
) -> Result<(), CudaError> {
    // Kernel may still be running after this returns!
    silu_safe(input, output, stream)?;

    // Caller MUST synchronize before dropping input/output
    // stream.synchronize()?;
    Ok(())
}
```

### Type Mapping Details

```rust
// C to Rust type mapping in extern "C" block
extern "C" {
    fn cuda_example(
        // const float* → *const f32 (immutable pointer)
        input: *const f32,

        // float* → *mut f32 (mutable pointer)
        output: *mut f32,

        // int64_t → i64 (guaranteed same size)
        n: i64,

        // int → c_int (platform-dependent, usually i32)
        batch: std::ffi::c_int,

        // float → f32 (IEEE 754 single precision)
        scale: f32,

        // void* → *mut c_void (opaque pointer)
        stream: *mut std::ffi::c_void,
    ) -> i32;  // int32_t → i32
}
```

### Build Script Details

```rust
// build.rs - Full implementation

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cuda_dir = manifest_dir.join("../../cuda");

    // Rerun if CUDA sources change
    println!("cargo:rerun-if-changed=../../cuda/stub.c");
    println!("cargo:rerun-if-changed=../../cuda/kernels/");

    if cuda_available() {
        build_cuda_kernels(&cuda_dir);
    } else {
        build_stub(&cuda_dir);
    }
}

fn cuda_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}

fn build_stub(cuda_dir: &PathBuf) {
    cc::Build::new()
        .file(cuda_dir.join("stub.c"))
        .warnings(true)
        .compile("nn_cuda_kernels");
}

fn build_cuda_kernels(cuda_dir: &PathBuf) {
    cc::Build::new()
        .cuda(true)
        .cudart("static")  // Link CUDA runtime statically
        .flag("-gencode=arch=compute_70,code=sm_70")  // V100
        .flag("-gencode=arch=compute_80,code=sm_80")  // A100
        .flag("-gencode=arch=compute_89,code=sm_89")  // RTX 40xx
        .files(glob::glob(cuda_dir.join("kernels/*.cu").to_str().unwrap()).unwrap())
        .compile("nn_cuda_kernels");
}
```

### Error Type Design

```rust
/// Semantic error types instead of raw codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaError {
    /// CUDA not installed, using stub
    NotAvailable,
    /// Kernel execution failed
    ExecutionFailed,
    /// Invalid argument (NULL pointer, negative size, etc.)
    InvalidArgument,
    /// Out of GPU memory
    OutOfMemory,
    /// Unknown error code
    Unknown(i32),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::NotAvailable => write!(f, "CUDA not available"),
            CudaError::ExecutionFailed => write!(f, "CUDA kernel execution failed"),
            CudaError::InvalidArgument => write!(f, "Invalid argument to CUDA function"),
            CudaError::OutOfMemory => write!(f, "CUDA out of memory"),
            CudaError::Unknown(code) => write!(f, "Unknown CUDA error: {}", code),
        }
    }
}

impl std::error::Error for CudaError {}

impl CudaError {
    pub fn from_code(code: i32) -> Result<(), Self> {
        match code {
            0 => Ok(()),
            -1 => Err(CudaError::NotAvailable),
            -2 => Err(CudaError::InvalidArgument),
            -3 => Err(CudaError::ExecutionFailed),
            -4 => Err(CudaError::OutOfMemory),
            other => Err(CudaError::Unknown(other)),
        }
    }
}
```

### Common Pitfalls in Rust FFI

#### 1. Forgetting `#[repr(C)]` on Structs

```rust
// WRONG: Rust may reorder fields
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
}

// CORRECT: C-compatible layout
#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
}
```

#### 2. String Handling

```rust
// WRONG: Rust strings are not null-terminated
fn bad(s: &str) {
    unsafe { c_function(s.as_ptr() as *const i8) }; // Missing NUL!
}

// CORRECT: Use CString
fn good(s: &str) -> Result<(), std::ffi::NulError> {
    let c_str = std::ffi::CString::new(s)?;
    unsafe { c_function(c_str.as_ptr()) };
    Ok(())
}
```

#### 3. Ownership Confusion

```rust
// WRONG: Vec dropped while C still using it
fn bad() {
    let data = vec![1.0f32, 2.0, 3.0];
    unsafe { async_kernel(data.as_ptr(), data.len()) };
    // data dropped here, kernel may still be running!
}

// CORRECT: Ensure data lives long enough
fn good() {
    let data = vec![1.0f32, 2.0, 3.0];
    unsafe { async_kernel(data.as_ptr(), data.len()) };
    synchronize(); // Wait for kernel
    // Now safe to drop data
}
```

#### 4. Aliasing Violations

```rust
// WRONG: Same buffer as both input and output
fn bad(buf: &mut [f32]) {
    unsafe {
        // UB: input and output alias!
        kernel(buf.as_ptr(), buf.as_mut_ptr(), buf.len());
    }
}

// May be OK if kernel supports in-place operation
// Document clearly if aliasing is allowed
```

### Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Test that stub returns NotAvailable
    #[test]
    fn test_stub_returns_not_available() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];

        let result = silu_safe(&input, &mut output, Stream::DEFAULT);

        // In CI without CUDA, this should be NotAvailable
        // With CUDA, this should be Ok
        match result {
            Ok(()) => {
                // Verify output is correct (SiLU computation)
                for (i, &x) in input.iter().enumerate() {
                    let expected = x / (1.0 + (-x).exp());
                    assert!((output[i] - expected).abs() < 1e-5);
                }
            }
            Err(CudaError::NotAvailable) => {
                // Expected when running with stub
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    /// Test error types
    #[test]
    fn test_error_display() {
        assert_eq!(
            CudaError::NotAvailable.to_string(),
            "CUDA not available"
        );
    }
}
```
