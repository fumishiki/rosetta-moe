//! FFI bridge for MoE Transformer GPU operations
//!
//! This crate provides:
//! - CUDA kernel FFI bindings (elementwise, softmax, rmsnorm, gemm, rope, attention, loss, optimizer, decode)
//! - GPU memory management (DeviceBuffer)
//! - GpuTensor for GPU-resident tensors
//! - High-level operations bridging Rust and CUDA
//! - CUDA Graph optimization support
//! - GPU-resident training support

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::ptr;

// =============================================================================
// Core Types
// =============================================================================

/// CUDA operation result
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub(crate) i32);

impl CudaError {
    pub const NOT_AVAILABLE: CudaError = CudaError(-1);

    pub fn code(&self) -> i32 {
        self.0
    }

    pub fn is_not_available(&self) -> bool {
        self.0 == -1
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_not_available() {
            write!(f, "CUDA not available")
        } else {
            write!(f, "CUDA error code: {}", self.0)
        }
    }
}

impl std::error::Error for CudaError {}

/// CUDA stream handle (opaque pointer)
#[derive(Debug, Clone, Copy)]
pub struct Stream(*mut c_void);

impl Stream {
    pub const DEFAULT: Stream = Stream(ptr::null_mut());

    /// # Safety
    /// The pointer must be a valid CUDA stream or null.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Stream(ptr)
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.0
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

// =============================================================================
// Modules
// =============================================================================

mod ffi;
mod tensor;

pub mod cuda_graph;
pub mod trainer;

// =============================================================================
// Re-exports: FFI operations
// =============================================================================

pub use ffi::{elementwise, softmax, rmsnorm, gemm, rope, attention, loss, optimizer, decode};

// =============================================================================
// Re-exports: Tensor types and operations
// =============================================================================

pub use tensor::{DType, Shape, DeviceBuffer, GpuTensor};
pub use tensor::{
    tensor_rmsnorm, tensor_gemm, tensor_silu, tensor_softmax,
    tensor_cross_entropy_forward, tensor_adamw_step,
    tensor_argmax, tensor_sample, tensor_topk_sample, tensor_topp_sample,
};

// =============================================================================
// Re-exports: CUDA Graph
// =============================================================================

pub use cuda_graph::{CudaGraph, GraphExecutor, GraphExecutionMode, GraphState, ExecutionHandle};

// =============================================================================
// Re-exports: Trainer
// =============================================================================

pub use trainer::{GpuTrainer, TrainerConfig, DecodingStrategy, StepMetrics};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_error() {
        let err = CudaError::NOT_AVAILABLE;
        assert!(err.is_not_available());
        assert_eq!(err.code(), -1);
    }

    #[test]
    fn test_cuda_error_display() {
        let err = CudaError::NOT_AVAILABLE;
        assert_eq!(format!("{}", err), "CUDA not available");

        let err2 = CudaError(42);
        assert_eq!(format!("{}", err2), "CUDA error code: 42");
    }

    #[test]
    fn test_stream_default() {
        let stream = Stream::DEFAULT;
        assert!(stream.as_ptr().is_null());
    }

    #[test]
    fn test_stream_from_raw() {
        let ptr = 0x1234 as *mut c_void;
        let stream = unsafe { Stream::from_raw(ptr) };
        assert_eq!(stream.as_ptr(), ptr);
    }

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
        let buf = DeviceBuffer::new(0);
        assert!(buf.is_ok());
    }

    mod stub_tests {
        use super::*;

        #[test]
        fn test_elementwise_silu_stub() {
            let input = [1.0f32; 4];
            let mut output = [0.0f32; 4];
            let result = unsafe {
                elementwise::silu(input.as_ptr(), output.as_mut_ptr(), 4, Stream::DEFAULT)
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_elementwise_add_stub() {
            let a = [1.0f32; 4];
            let b = [2.0f32; 4];
            let mut output = [0.0f32; 4];
            let result = unsafe {
                elementwise::add(a.as_ptr(), b.as_ptr(), output.as_mut_ptr(), 4, Stream::DEFAULT)
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_softmax_stub() {
            let input = [1.0f32; 8];
            let mut output = [0.0f32; 8];
            let result = unsafe {
                softmax::softmax(input.as_ptr(), output.as_mut_ptr(), 2, 4, Stream::DEFAULT)
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_rmsnorm_stub() {
            let input = [1.0f32; 8];
            let weight = [1.0f32; 4];
            let mut output = [0.0f32; 8];
            let result = unsafe {
                rmsnorm::forward(
                    input.as_ptr(), weight.as_ptr(), output.as_mut_ptr(),
                    2, 4, 1e-6, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_gemm_stub() {
            let a = [1.0f32; 4];
            let b = [1.0f32; 4];
            let mut c = [0.0f32; 4];
            let result = unsafe {
                gemm::matmul(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, Stream::DEFAULT)
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_rope_freqs_stub() {
            let mut freqs = [0.0f32; 64];
            let result = unsafe {
                rope::compute_freqs(freqs.as_mut_ptr(), 8, 16, 10000.0, 1.0, Stream::DEFAULT)
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_attention_scores_stub() {
            let q = [1.0f32; 64];
            let k = [1.0f32; 64];
            let mut scores = [0.0f32; 16];
            let result = unsafe {
                attention::compute_scores(
                    q.as_ptr(), k.as_ptr(), scores.as_mut_ptr(),
                    1, 1, 4, 16, 0.25, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_loss_cross_entropy_stub() {
            let logits = [1.0f32; 32];
            let targets = [0i32; 4];
            let mut loss = 0.0f32;
            let mut log_probs = [0.0f32; 32];
            let result = unsafe {
                loss::cross_entropy_forward(
                    logits.as_ptr(), targets.as_ptr(),
                    &mut loss, log_probs.as_mut_ptr(),
                    4, 8, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_optimizer_adamw_stub() {
            let mut param = [1.0f32; 4];
            let grad = [0.1f32; 4];
            let mut m = [0.0f32; 4];
            let mut v = [0.0f32; 4];
            let result = unsafe {
                optimizer::adamw_step(
                    param.as_mut_ptr(), grad.as_ptr(),
                    m.as_mut_ptr(), v.as_mut_ptr(),
                    1e-4, 0.9, 0.999, 1e-8, 0.01,
                    1, 4, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_decode_argmax_stub() {
            let logits = [1.0f32; 32];
            let mut output = [0i32; 4];
            let result = unsafe {
                decode::argmax(logits.as_ptr(), output.as_mut_ptr(), 4, 8, Stream::DEFAULT)
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_decode_sample_stub() {
            let logits = [1.0f32; 32];
            let mut output = [0i32; 4];
            let seeds = [12345u64; 4];
            let result = unsafe {
                decode::sample(
                    logits.as_ptr(), output.as_mut_ptr(), seeds.as_ptr(),
                    4, 8, 1.0, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_decode_topk_stub() {
            let logits = [1.0f32; 32];
            let mut output = [0i32; 4];
            let seeds = [12345u64; 4];
            let result = unsafe {
                decode::topk_sample(
                    logits.as_ptr(), output.as_mut_ptr(), seeds.as_ptr(),
                    4, 8, 3, 1.0, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }

        #[test]
        fn test_decode_topp_stub() {
            let logits = [1.0f32; 32];
            let mut output = [0i32; 4];
            let seeds = [12345u64; 4];
            let result = unsafe {
                decode::topp_sample(
                    logits.as_ptr(), output.as_mut_ptr(), seeds.as_ptr(),
                    4, 8, 0.9, 1.0, Stream::DEFAULT
                )
            };
            assert!(result.is_err());
            assert!(result.unwrap_err().is_not_available());
        }
    }
}
