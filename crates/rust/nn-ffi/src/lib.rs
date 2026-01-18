//! FFI bridge connecting nn-core and nn-cuda
//!
//! Provides:
//! - GPU memory management (DeviceBuffer)
//! - GpuTensor for GPU-resident tensors
//! - High-level operations bridging Rust and CUDA
//! - CUDA Graph optimization support

#![allow(dead_code)]
#![allow(clippy::manual_slice_size_calculation)]

use std::ffi::c_void;
use std::ptr;

pub use nn_cuda::{CudaResult, Stream};

pub mod cuda_graph;
pub use cuda_graph::{CudaGraph, GraphExecutor, GraphExecutionMode};

pub mod trainer;
pub use trainer::{GpuTrainer, TrainerConfig, DecodingStrategy, StepMetrics};

/// FFI error
#[derive(Debug, Clone, Copy)]
pub struct FfiError(pub i32);

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FFI error code: {}", self.0)
    }
}

impl std::error::Error for FfiError {}

// =============================================================================
// CUDA Runtime FFI
// =============================================================================

#[link(name = "nn_cuda_kernels")]
unsafe extern "C" {
    // These are provided by CUDA runtime, linked via nn-cuda
}

/// Result type for FFI operations
pub type FfiResult<T> = Result<T, FfiError>;

// Stub implementations for non-CUDA environments
mod runtime {
    use std::ffi::c_void;
    use super::{FfiError, FfiResult};

    pub unsafe fn malloc(_size: usize) -> FfiResult<*mut c_void> {
        Err(FfiError(-1)) // CUDA not available
    }

    pub unsafe fn free(_ptr: *mut c_void) -> FfiResult<()> {
        Err(FfiError(-1))
    }

    pub unsafe fn memcpy_h2d(_dst: *mut c_void, _src: *const c_void, _size: usize) -> FfiResult<()> {
        Err(FfiError(-1))
    }

    pub unsafe fn memcpy_d2h(_dst: *mut c_void, _src: *const c_void, _size: usize) -> FfiResult<()> {
        Err(FfiError(-1))
    }

    pub unsafe fn memcpy_d2d(_dst: *mut c_void, _src: *const c_void, _size: usize) -> FfiResult<()> {
        Err(FfiError(-1))
    }

    pub unsafe fn memset(_ptr: *mut c_void, _value: i32, _size: usize) -> FfiResult<()> {
        Err(FfiError(-1))
    }
}

// =============================================================================
// Device Buffer
// =============================================================================

/// Raw GPU memory buffer
pub struct DeviceBuffer {
    ptr: *mut c_void,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate GPU memory
    pub fn new(size: usize) -> FfiResult<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                size: 0,
            });
        }

        let ptr = unsafe { runtime::malloc(size) }?;
        Ok(Self { ptr, size })
    }

    /// Create from raw pointer (takes ownership)
    ///
    /// # Safety
    /// - `ptr` must be a valid CUDA device pointer allocated with cudaMalloc
    /// - `size` must match the allocation size
    pub unsafe fn from_raw(ptr: *mut c_void, size: usize) -> Self {
        Self { ptr, size }
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get typed pointer
    pub fn as_ptr_typed<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy from host to device
    pub fn copy_from_host<T>(&mut self, data: &[T]) -> FfiResult<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            return Err(FfiError(1)); // Invalid size
        }
        unsafe { runtime::memcpy_h2d(self.ptr, data.as_ptr() as *const c_void, size) }
    }

    /// Copy from device to host
    pub fn copy_to_host<T>(&self, data: &mut [T]) -> FfiResult<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            return Err(FfiError(1)); // Invalid size
        }
        unsafe { runtime::memcpy_d2h(data.as_mut_ptr() as *mut c_void, self.ptr, size) }
    }

    /// Fill with zeros
    pub fn zero(&mut self) -> FfiResult<()> {
        if self.size > 0 {
            unsafe { runtime::memset(self.ptr, 0, self.size) }
        } else {
            Ok(())
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { runtime::free(self.ptr) };
        }
    }
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

// =============================================================================
// GPU Tensor
// =============================================================================

/// Data type for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Check if this is a reduced precision type
    pub fn is_reduced_precision(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }
}

/// Shape of a tensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}

/// GPU-resident tensor
pub struct GpuTensor {
    buffer: DeviceBuffer,
    shape: Shape,
    dtype: DType,
}

impl GpuTensor {
    /// Create a new GPU tensor filled with zeros
    pub fn zeros(shape: Shape, dtype: DType) -> FfiResult<Self> {
        let size = shape.numel() * dtype.size_bytes();
        let mut buffer = DeviceBuffer::new(size)?;
        buffer.zero()?;
        Ok(Self { buffer, shape, dtype })
    }

    /// Create from host data
    pub fn from_slice(data: &[f32], shape: Shape) -> FfiResult<Self> {
        assert_eq!(data.len(), shape.numel());
        let size = data.len() * std::mem::size_of::<f32>();
        let mut buffer = DeviceBuffer::new(size)?;
        buffer.copy_from_host(data)?;
        Ok(Self {
            buffer,
            shape,
            dtype: DType::F32,
        })
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get raw pointer
    pub fn as_ptr(&self) -> *const f32 {
        self.buffer.as_ptr_typed()
    }

    /// Get mutable raw pointer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.buffer.as_ptr_typed()
    }

    /// Copy to host
    pub fn to_vec(&self) -> FfiResult<Vec<f32>> {
        let mut data = vec![0.0f32; self.numel()];
        self.buffer.copy_to_host(&mut data)?;
        Ok(data)
    }
}

// =============================================================================
// High-Level Operations
// =============================================================================

/// Execute RMSNorm on GPU
pub fn rmsnorm(
    input: &GpuTensor,
    weight: &GpuTensor,
    output: &mut GpuTensor,
    eps: f32,
    stream: Stream,
) -> CudaResult<()> {
    let dims = input.shape().dims();
    let rows = dims[..dims.len() - 1].iter().product::<usize>() as i32;
    let hidden_dim = *dims.last().unwrap() as i32;

    unsafe {
        nn_cuda::rmsnorm::forward(
            input.as_ptr(),
            weight.as_ptr(),
            output.as_mut_ptr(),
            rows,
            hidden_dim,
            eps,
            stream,
        )
    }
}

/// Execute GEMM on GPU: C = A @ B^T
pub fn gemm(
    a: &GpuTensor,
    b: &GpuTensor,
    c: &mut GpuTensor,
    stream: Stream,
) -> CudaResult<()> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    let m = a_dims[0] as i32;
    let k = a_dims[1] as i32;
    let n = b_dims[0] as i32;

    unsafe {
        nn_cuda::gemm::matmul(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m, n, k,
            stream,
        )
    }
}

/// Execute SiLU activation on GPU
pub fn silu(
    input: &GpuTensor,
    output: &mut GpuTensor,
    stream: Stream,
) -> CudaResult<()> {
    let n = input.numel() as i64;
    unsafe {
        nn_cuda::elementwise::silu(input.as_ptr(), output.as_mut_ptr(), n, stream)
    }
}

/// Execute Softmax on GPU
pub fn softmax(
    input: &GpuTensor,
    output: &mut GpuTensor,
    stream: Stream,
) -> CudaResult<()> {
    let dims = input.shape().dims();
    let rows = dims[..dims.len() - 1].iter().product::<usize>() as i32;
    let cols = *dims.last().unwrap() as i32;

    unsafe {
        nn_cuda::softmax::softmax(input.as_ptr(), output.as_mut_ptr(), rows, cols, stream)
    }
}

/// Execute cross entropy forward on GPU
pub fn cross_entropy_forward(
    logits: &GpuTensor,
    targets: &GpuTensor,
    loss: &mut GpuTensor,
    log_probs: &mut GpuTensor,
    stream: Stream,
) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch_seq = dims[..dims.len() - 1].iter().product::<usize>() as i32;
    let vocab_size = *dims.last().unwrap() as i32;

    unsafe {
        nn_cuda::loss::cross_entropy_forward(
            logits.as_ptr(),
            targets.as_ptr() as *const i32,
            loss.as_mut_ptr(),
            log_probs.as_mut_ptr(),
            batch_seq,
            vocab_size,
            stream,
        )
    }
}

/// Execute AdamW optimizer step on GPU
#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    param: &mut GpuTensor,
    grad: &GpuTensor,
    m: &mut GpuTensor,
    v: &mut GpuTensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: i32,
    stream: Stream,
) -> CudaResult<()> {
    let size = param.numel() as i64;

    unsafe {
        nn_cuda::optimizer::adamw_step(
            param.as_mut_ptr(),
            grad.as_ptr(),
            m.as_mut_ptr(),
            v.as_mut_ptr(),
            lr, beta1, beta2, eps, weight_decay, step, size,
            stream,
        )
    }
}

// =============================================================================
// GPU-side Decoding Operations
// =============================================================================

/// Greedy decode on GPU: select argmax token for each batch item
pub fn argmax(
    logits: &GpuTensor,
    output: &mut GpuTensor,
    stream: Stream,
) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch = dims[0] as i32;
    let vocab_size = dims[1] as i32;

    unsafe {
        nn_cuda::decode::argmax(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            batch,
            vocab_size,
            stream,
        )
    }
}

/// Multinomial sampling on GPU with temperature
pub fn sample(
    logits: &GpuTensor,
    output: &mut GpuTensor,
    seeds: &GpuTensor,
    temperature: f32,
    stream: Stream,
) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch = dims[0] as i32;
    let vocab_size = dims[1] as i32;

    unsafe {
        nn_cuda::decode::sample(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            seeds.as_ptr() as *const u64,
            batch,
            vocab_size,
            temperature,
            stream,
        )
    }
}

/// Top-k sampling on GPU
pub fn topk_sample(
    logits: &GpuTensor,
    output: &mut GpuTensor,
    seeds: &GpuTensor,
    k: i32,
    temperature: f32,
    stream: Stream,
) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch = dims[0] as i32;
    let vocab_size = dims[1] as i32;

    unsafe {
        nn_cuda::decode::topk_sample(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            seeds.as_ptr() as *const u64,
            batch,
            vocab_size,
            k,
            temperature,
            stream,
        )
    }
}

/// Nucleus (top-p) sampling on GPU
pub fn topp_sample(
    logits: &GpuTensor,
    output: &mut GpuTensor,
    seeds: &GpuTensor,
    top_p: f32,
    temperature: f32,
    stream: Stream,
) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch = dims[0] as i32;
    let vocab_size = dims[1] as i32;

    unsafe {
        nn_cuda::decode::topp_sample(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            seeds.as_ptr() as *const u64,
            batch,
            vocab_size,
            top_p,
            temperature,
            stream,
        )
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I32.size_bytes(), 4);
    }

    #[test]
    fn test_dtype_properties() {
        assert!(DType::F32.is_float());
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::I32.is_float());

        assert!(!DType::F32.is_reduced_precision());
        assert!(DType::F16.is_reduced_precision());
        assert!(DType::BF16.is_reduced_precision());
    }

    #[test]
    fn test_device_buffer_zero_size() {
        // Zero-size allocation should succeed
        let buf = DeviceBuffer::new(0);
        assert!(buf.is_ok());
    }
}
