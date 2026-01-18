//! GPU Tensor types and high-level operations
//!
//! Provides Shape, DType, DeviceBuffer, GpuTensor and tensor_* operations.

use std::ffi::c_void;
use std::ptr;

use crate::{CudaError, CudaResult, Stream};
use crate::ffi::{runtime, elementwise, softmax, rmsnorm, gemm, loss, optimizer, decode};

// =============================================================================
// Data Types
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
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

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
    pub fn new(size: usize) -> CudaResult<Self> {
        if size == 0 {
            return Ok(Self { ptr: ptr::null_mut(), size: 0 });
        }
        let ptr = unsafe { runtime::malloc(size) }?;
        Ok(Self { ptr, size })
    }

    /// Create from raw pointer (takes ownership)
    ///
    /// # Safety
    /// - `ptr` must be a valid CUDA device pointer
    /// - `size` must match the allocation size
    pub unsafe fn from_raw(ptr: *mut c_void, size: usize) -> Self {
        Self { ptr, size }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn as_ptr_typed<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn copy_from_host<T>(&mut self, data: &[T]) -> CudaResult<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            return Err(CudaError(1));
        }
        unsafe { runtime::memcpy_h2d(self.ptr, data.as_ptr() as *const c_void, size) }
    }

    pub fn copy_to_host<T>(&self, data: &mut [T]) -> CudaResult<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            return Err(CudaError(1));
        }
        unsafe { runtime::memcpy_d2h(data.as_mut_ptr() as *mut c_void, self.ptr, size) }
    }

    pub fn zero(&mut self) -> CudaResult<()> {
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

/// GPU-resident tensor
pub struct GpuTensor {
    buffer: DeviceBuffer,
    shape: Shape,
    dtype: DType,
}

impl GpuTensor {
    pub fn zeros(shape: Shape, dtype: DType) -> CudaResult<Self> {
        let size = shape.numel() * dtype.size_bytes();
        let mut buffer = DeviceBuffer::new(size)?;
        buffer.zero()?;
        Ok(Self { buffer, shape, dtype })
    }

    pub fn from_slice(data: &[f32], shape: Shape) -> CudaResult<Self> {
        assert_eq!(data.len(), shape.numel());
        let size = data.len() * std::mem::size_of::<f32>();
        let mut buffer = DeviceBuffer::new(size)?;
        buffer.copy_from_host(data)?;
        Ok(Self { buffer, shape, dtype: DType::F32 })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.buffer.as_ptr_typed()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.buffer.as_ptr_typed()
    }

    pub fn to_vec(&self) -> CudaResult<Vec<f32>> {
        let mut data = vec![0.0f32; self.numel()];
        self.buffer.copy_to_host(&mut data)?;
        Ok(data)
    }
}

// =============================================================================
// High-Level Operations
// =============================================================================

/// Execute RMSNorm on GPU
pub fn tensor_rmsnorm(
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
        rmsnorm::forward(input.as_ptr(), weight.as_ptr(), output.as_mut_ptr(), rows, hidden_dim, eps, stream)
    }
}

/// Execute GEMM on GPU: C = A @ B^T
pub fn tensor_gemm(
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

    unsafe { gemm::matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, stream) }
}

/// Execute SiLU activation on GPU
pub fn tensor_silu(input: &GpuTensor, output: &mut GpuTensor, stream: Stream) -> CudaResult<()> {
    let n = input.numel() as i64;
    unsafe { elementwise::silu(input.as_ptr(), output.as_mut_ptr(), n, stream) }
}

/// Execute Softmax on GPU
pub fn tensor_softmax(input: &GpuTensor, output: &mut GpuTensor, stream: Stream) -> CudaResult<()> {
    let dims = input.shape().dims();
    let rows = dims[..dims.len() - 1].iter().product::<usize>() as i32;
    let cols = *dims.last().unwrap() as i32;

    unsafe { softmax::softmax(input.as_ptr(), output.as_mut_ptr(), rows, cols, stream) }
}

/// Execute cross entropy forward on GPU
pub fn tensor_cross_entropy_forward(
    logits: &GpuTensor,
    targets: &GpuTensor,
    loss_out: &mut GpuTensor,
    log_probs: &mut GpuTensor,
    stream: Stream,
) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch_seq = dims[..dims.len() - 1].iter().product::<usize>() as i32;
    let vocab_size = *dims.last().unwrap() as i32;

    unsafe {
        loss::cross_entropy_forward(
            logits.as_ptr(),
            targets.as_ptr() as *const i32,
            loss_out.as_mut_ptr(),
            log_probs.as_mut_ptr(),
            batch_seq,
            vocab_size,
            stream,
        )
    }
}

/// Execute AdamW optimizer step on GPU
pub fn tensor_adamw_step(
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
        optimizer::adamw_step(
            param.as_mut_ptr(), grad.as_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
            lr, beta1, beta2, eps, weight_decay, step, size, stream,
        )
    }
}

/// Greedy decode on GPU: select argmax token for each batch item
pub fn tensor_argmax(logits: &GpuTensor, output: &mut GpuTensor, stream: Stream) -> CudaResult<()> {
    let dims = logits.shape().dims();
    let batch = dims[0] as i32;
    let vocab_size = dims[1] as i32;

    unsafe { decode::argmax(logits.as_ptr(), output.as_mut_ptr() as *mut i32, batch, vocab_size, stream) }
}

/// Multinomial sampling on GPU with temperature
pub fn tensor_sample(
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
        decode::sample(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            seeds.as_ptr() as *const u64,
            batch, vocab_size, temperature, stream,
        )
    }
}

/// Top-k sampling on GPU
pub fn tensor_topk_sample(
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
        decode::topk_sample(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            seeds.as_ptr() as *const u64,
            batch, vocab_size, k, temperature, stream,
        )
    }
}

/// Nucleus (top-p) sampling on GPU
pub fn tensor_topp_sample(
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
        decode::topp_sample(
            logits.as_ptr(),
            output.as_mut_ptr() as *mut i32,
            seeds.as_ptr() as *const u64,
            batch, vocab_size, top_p, temperature, stream,
        )
    }
}
