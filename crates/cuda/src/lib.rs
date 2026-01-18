//! CUDA kernel bindings for MoE Transformer
//!
//! This crate provides Rust FFI bindings to CUDA kernels for:
//! - Element-wise operations (SiLU, add, mul, scale)
//! - Softmax with optional top-k selection
//! - RMSNorm
//! - GEMM (matrix multiplication)
//! - RoPE (Rotary Position Embedding)
//! - Multi-Query Attention

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::ptr;

// Error Handling
/// CUDA operation result
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(i32);

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

// =============================================================================
// Stream Handle
// =============================================================================

/// CUDA stream handle (opaque pointer)
#[derive(Debug, Clone, Copy)]
pub struct Stream(*mut c_void);

impl Stream {
    /// Default stream (nullptr)
    pub const DEFAULT: Stream = Stream(ptr::null_mut());

    /// Create from raw pointer
    ///
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
// FFI Declarations
// =============================================================================

#[link(name = "nn_cuda_kernels")]
unsafe extern "C" {
    // Element-wise operations
    fn cuda_silu(input: *const f32, output: *mut f32, n: i64, stream: *mut c_void) -> i32;
    fn cuda_add(a: *const f32, b: *const f32, output: *mut f32, n: i64, stream: *mut c_void) -> i32;
    fn cuda_mul(a: *const f32, b: *const f32, output: *mut f32, n: i64, stream: *mut c_void) -> i32;
    fn cuda_scale(input: *const f32, scale: f32, output: *mut f32, n: i64, stream: *mut c_void) -> i32;

    // Softmax
    fn cuda_softmax(input: *const f32, output: *mut f32, rows: i32, cols: i32, stream: *mut c_void) -> i32;
    fn cuda_softmax_topk(
        input: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        rows: i32,
        cols: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;

    // RMSNorm
    fn cuda_rmsnorm(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: i32,
        hidden_dim: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_rmsnorm_residual(
        input: *const f32,
        residual: *const f32,
        weight: *const f32,
        output: *mut f32,
        residual_out: *mut f32,
        rows: i32,
        hidden_dim: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;

    // GEMM
    fn cuda_gemm(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_gemm_beta(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_batched_gemm(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;

    // RoPE
    fn cuda_rope_freqs(
        freqs: *mut f32,
        max_seq_len: i32,
        head_dim: i32,
        base: f32,
        alpha: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_rope_forward(
        input: *const f32,
        freqs: *const f32,
        output: *mut f32,
        batch: i32,
        seq_len: i32,
        n_heads: i32,
        head_dim: i32,
        offset: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_rope_qk(
        q: *const f32,
        k: *const f32,
        freqs: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        batch: i32,
        seq_len: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        offset: i32,
        stream: *mut c_void,
    ) -> i32;

    // Attention
    fn cuda_attention_scores(
        q: *const f32,
        k: *const f32,
        scores: *mut f32,
        batch: i32,
        n_heads: i32,
        seq_len: i32,
        head_dim: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_attention_output(
        weights: *const f32,
        v: *const f32,
        output: *mut f32,
        batch: i32,
        n_heads: i32,
        seq_len: i32,
        head_dim: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_flash_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        batch: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        seq_len: i32,
        head_dim: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;

    // Loss
    fn cuda_cross_entropy_forward(
        logits: *const f32,
        targets: *const i32,
        loss: *mut f32,
        log_probs: *mut f32,
        batch_seq: i32,
        vocab_size: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_cross_entropy_backward(
        log_probs: *const f32,
        targets: *const i32,
        d_logits: *mut f32,
        batch_seq: i32,
        vocab_size: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_aux_loss_forward(
        router_probs: *const f32,
        expert_indices: *const i32,
        aux_loss: *mut f32,
        f_counts: *mut f32,
        p_means: *mut f32,
        batch_seq: i32,
        n_experts: i32,
        top_k: i32,
        alpha: f32,
        stream: *mut c_void,
    ) -> i32;

    // Optimizer
    fn cuda_adamw_step(
        param: *mut f32,
        grad: *const f32,
        m: *mut f32,
        v: *mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
        size: i64,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_zero_grad(grad: *mut f32, size: i64, stream: *mut c_void) -> i32;
    fn cuda_grad_clip(
        grad: *mut f32,
        partial_norms: *mut f32,
        total_norm: *mut f32,
        clip_norm: f32,
        size: i64,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_scatter_add(
        grad_output: *const f32,
        indices: *const i32,
        grad_weight: *mut f32,
        num_indices: i32,
        embedding_dim: i32,
        stream: *mut c_void,
    ) -> i32;

    // Decode (GPU-side token generation)
    fn cuda_argmax(
        logits: *const f32,
        output: *mut i32,
        batch: i32,
        vocab_size: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_sample(
        logits: *const f32,
        output: *mut i32,
        seeds: *const u64,
        batch: i32,
        vocab_size: i32,
        temperature: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_topk_sample(
        logits: *const f32,
        output: *mut i32,
        seeds: *const u64,
        batch: i32,
        vocab_size: i32,
        k: i32,
        temperature: f32,
        stream: *mut c_void,
    ) -> i32;
    fn cuda_topp_sample(
        logits: *const f32,
        output: *mut i32,
        seeds: *const u64,
        batch: i32,
        vocab_size: i32,
        top_p: f32,
        temperature: f32,
        stream: *mut c_void,
    ) -> i32;
}

// =============================================================================
// Safe Rust API
// =============================================================================

fn check_result(code: i32) -> CudaResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError(code))
    }
}

/// Element-wise operations
pub mod elementwise {
    use super::*;

    /// SiLU activation: y = x * sigmoid(x)
    ///
    /// # Safety
    /// - `input` and `output` must be valid device pointers
    /// - `n` must match the tensor size
    pub unsafe fn silu(input: *const f32, output: *mut f32, n: i64, stream: Stream) -> CudaResult<()> {
        check_result(unsafe { cuda_silu(input, output, n, stream.as_ptr()) })
    }

    /// Element-wise addition: output = a + b
    ///
    /// # Safety
    /// - All pointers must be valid device pointers
    /// - `n` must match the tensor size
    pub unsafe fn add(a: *const f32, b: *const f32, output: *mut f32, n: i64, stream: Stream) -> CudaResult<()> {
        check_result(unsafe { cuda_add(a, b, output, n, stream.as_ptr()) })
    }

    /// Element-wise multiplication: output = a * b
    ///
    /// # Safety
    /// - All pointers must be valid device pointers
    /// - `n` must match the tensor size
    pub unsafe fn mul(a: *const f32, b: *const f32, output: *mut f32, n: i64, stream: Stream) -> CudaResult<()> {
        check_result(unsafe { cuda_mul(a, b, output, n, stream.as_ptr()) })
    }

    /// Scalar multiplication: output = input * scale
    ///
    /// # Safety
    /// - `input` and `output` must be valid device pointers
    /// - `n` must match the tensor size
    pub unsafe fn scale(input: *const f32, scale: f32, output: *mut f32, n: i64, stream: Stream) -> CudaResult<()> {
        check_result(unsafe { cuda_scale(input, scale, output, n, stream.as_ptr()) })
    }
}

/// Softmax operations
pub mod softmax {
    use super::*;

    /// Row-wise softmax
    ///
    /// # Safety
    /// - `input` and `output` must be valid device pointers of shape [rows, cols]
    pub unsafe fn softmax(input: *const f32, output: *mut f32, rows: i32, cols: i32, stream: Stream) -> CudaResult<()> {
        check_result(unsafe { cuda_softmax(input, output, rows, cols, stream.as_ptr()) })
    }

    /// Softmax with top-k selection (for router)
    ///
    /// # Safety
    /// - `input` must be valid device pointer of shape [rows, cols]
    /// - `weights` and `indices` must be valid device pointers of shape [rows, k]
    pub unsafe fn softmax_topk(
        input: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        rows: i32,
        cols: i32,
        k: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_softmax_topk(input, weights, indices, rows, cols, k, stream.as_ptr()) })
    }
}

/// RMSNorm operations
pub mod rmsnorm {
    use super::*;

    /// RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
    ///
    /// # Safety
    /// - `input` must be valid device pointer of shape [rows, hidden_dim]
    /// - `weight` must be valid device pointer of shape [hidden_dim]
    /// - `output` must be valid device pointer of shape [rows, hidden_dim]
    pub unsafe fn forward(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: i32,
        hidden_dim: i32,
        eps: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_rmsnorm(input, weight, output, rows, hidden_dim, eps, stream.as_ptr()) })
    }

    /// Fused RMSNorm with residual connection
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn forward_residual(
        input: *const f32,
        residual: *const f32,
        weight: *const f32,
        output: *mut f32,
        residual_out: *mut f32,
        rows: i32,
        hidden_dim: i32,
        eps: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_rmsnorm_residual(
                input, residual, weight, output, residual_out, rows, hidden_dim, eps, stream.as_ptr(),
            )
        })
    }
}

/// Matrix multiplication operations
pub mod gemm {
    use super::*;

    /// GEMM: C = A @ B^T
    ///
    /// # Safety
    /// - `a` must be valid device pointer of shape [M, K]
    /// - `b` must be valid device pointer of shape [N, K]
    /// - `c` must be valid device pointer of shape [M, N]
    pub unsafe fn matmul(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_gemm(a, b, c, m, n, k, stream.as_ptr()) })
    }

    /// GEMM with accumulation: C = alpha * A @ B^T + beta * C
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn matmul_beta(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_gemm_beta(a, b, c, m, n, k, alpha, beta, stream.as_ptr()) })
    }

    /// Batched GEMM: C[b] = A[b] @ B[b]^T
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn batched_matmul(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_batched_gemm(a, b, c, batch, m, n, k, stream.as_ptr()) })
    }
}

/// Rotary Position Embedding operations
pub mod rope {
    use super::*;

    /// Precompute RoPE frequencies with NTK scaling
    ///
    /// # Safety
    /// - `freqs` must be valid device pointer of shape [max_seq_len, head_dim/2]
    pub unsafe fn compute_freqs(
        freqs: *mut f32,
        max_seq_len: i32,
        head_dim: i32,
        base: f32,
        alpha: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_rope_freqs(freqs, max_seq_len, head_dim, base, alpha, stream.as_ptr()) })
    }

    /// Apply RoPE to input tensor
    ///
    /// # Safety
    /// - `input` must be valid device pointer of shape [batch, seq_len, n_heads, head_dim]
    /// - `freqs` must be valid device pointer of shape [max_seq_len, head_dim/2]
    /// - `output` must be valid device pointer of shape [batch, seq_len, n_heads, head_dim]
    pub unsafe fn forward(
        input: *const f32,
        freqs: *const f32,
        output: *mut f32,
        batch: i32,
        seq_len: i32,
        n_heads: i32,
        head_dim: i32,
        offset: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_rope_forward(
                input, freqs, output, batch, seq_len, n_heads, head_dim, offset, stream.as_ptr(),
            )
        })
    }

    /// Apply RoPE to Q and K tensors
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn apply_qk(
        q: *const f32,
        k: *const f32,
        freqs: *const f32,
        q_out: *mut f32,
        k_out: *mut f32,
        batch: i32,
        seq_len: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        offset: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_rope_qk(
                q, k, freqs, q_out, k_out, batch, seq_len, n_q_heads, n_kv_heads, head_dim, offset, stream.as_ptr(),
            )
        })
    }
}

/// Attention operations
pub mod attention {
    use super::*;

    /// Compute attention scores: scores = Q @ K^T * scale
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn compute_scores(
        q: *const f32,
        k: *const f32,
        scores: *mut f32,
        batch: i32,
        n_heads: i32,
        seq_len: i32,
        head_dim: i32,
        scale: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_attention_scores(
                q, k, scores, batch, n_heads, seq_len, head_dim, scale, stream.as_ptr(),
            )
        })
    }

    /// Compute attention output: output = weights @ V
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn compute_output(
        weights: *const f32,
        v: *const f32,
        output: *mut f32,
        batch: i32,
        n_heads: i32,
        seq_len: i32,
        head_dim: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_attention_output(weights, v, output, batch, n_heads, seq_len, head_dim, stream.as_ptr()) })
    }

    /// FlashAttention-style fused attention
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn flash_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        batch: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        seq_len: i32,
        head_dim: i32,
        scale: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_flash_attention(
                q, k, v, output, batch, n_q_heads, n_kv_heads, seq_len, head_dim, scale, stream.as_ptr(),
            )
        })
    }
}

/// Loss functions
pub mod loss {
    use super::*;

    /// Cross entropy forward: computes loss and stores log_probs for backward
    ///
    /// # Safety
    /// - `logits` must be valid device pointer of shape [batch_seq, vocab_size]
    /// - `targets` must be valid device pointer of shape [batch_seq]
    /// - `loss` must be valid device pointer for scalar output
    /// - `log_probs` must be valid device pointer of shape [batch_seq, vocab_size]
    pub unsafe fn cross_entropy_forward(
        logits: *const f32,
        targets: *const i32,
        loss: *mut f32,
        log_probs: *mut f32,
        batch_seq: i32,
        vocab_size: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_cross_entropy_forward(logits, targets, loss, log_probs, batch_seq, vocab_size, stream.as_ptr())
        })
    }

    /// Cross entropy backward: d_logits = softmax(logits) - one_hot(target)
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn cross_entropy_backward(
        log_probs: *const f32,
        targets: *const i32,
        d_logits: *mut f32,
        batch_seq: i32,
        vocab_size: i32,
        scale: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_cross_entropy_backward(log_probs, targets, d_logits, batch_seq, vocab_size, scale, stream.as_ptr())
        })
    }

    /// MoE auxiliary loss for load balancing
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn aux_loss_forward(
        router_probs: *const f32,
        expert_indices: *const i32,
        aux_loss: *mut f32,
        f_counts: *mut f32,
        p_means: *mut f32,
        batch_seq: i32,
        n_experts: i32,
        top_k: i32,
        alpha: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_aux_loss_forward(
                router_probs, expert_indices, aux_loss, f_counts, p_means,
                batch_seq, n_experts, top_k, alpha, stream.as_ptr(),
            )
        })
    }
}

/// Optimizer operations
pub mod optimizer {
    use super::*;

    /// AdamW optimizer step
    ///
    /// # Safety
    /// - All pointers must be valid device pointers of size `size`
    pub unsafe fn adamw_step(
        param: *mut f32,
        grad: *const f32,
        m: *mut f32,
        v: *mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
        size: i64,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_adamw_step(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step, size, stream.as_ptr())
        })
    }

    /// Zero gradients
    ///
    /// # Safety
    /// - `grad` must be valid device pointer of size `size`
    pub unsafe fn zero_grad(grad: *mut f32, size: i64, stream: Stream) -> CudaResult<()> {
        check_result(unsafe { cuda_zero_grad(grad, size, stream.as_ptr()) })
    }

    /// Gradient clipping by global norm
    ///
    /// # Safety
    /// - All pointers must be valid device pointers
    /// - `partial_norms` must have enough space for (size + 255) / 256 floats
    pub unsafe fn grad_clip(
        grad: *mut f32,
        partial_norms: *mut f32,
        total_norm: *mut f32,
        clip_norm: f32,
        size: i64,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_grad_clip(grad, partial_norms, total_norm, clip_norm, size, stream.as_ptr())
        })
    }

    /// Scatter add for embedding backward
    ///
    /// # Safety
    /// - All pointers must be valid device pointers with appropriate shapes
    pub unsafe fn scatter_add(
        grad_output: *const f32,
        indices: *const i32,
        grad_weight: *mut f32,
        num_indices: i32,
        embedding_dim: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_scatter_add(grad_output, indices, grad_weight, num_indices, embedding_dim, stream.as_ptr())
        })
    }
}

/// GPU-side token generation (decoding)
pub mod decode {
    use super::*;

    /// Greedy decoding: select argmax token for each batch item
    ///
    /// # Safety
    /// - `logits` must be valid device pointer of shape [batch, vocab_size]
    /// - `output` must be valid device pointer of shape [batch]
    pub unsafe fn argmax(
        logits: *const f32,
        output: *mut i32,
        batch: i32,
        vocab_size: i32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe { cuda_argmax(logits, output, batch, vocab_size, stream.as_ptr()) })
    }

    /// Multinomial sampling with temperature
    ///
    /// # Safety
    /// - `logits` must be valid device pointer of shape [batch, vocab_size]
    /// - `output` must be valid device pointer of shape [batch]
    /// - `seeds` must be valid device pointer of shape [batch] containing RNG seeds
    pub unsafe fn sample(
        logits: *const f32,
        output: *mut i32,
        seeds: *const u64,
        batch: i32,
        vocab_size: i32,
        temperature: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_sample(logits, output, seeds, batch, vocab_size, temperature, stream.as_ptr())
        })
    }

    /// Top-k sampling: sample from top k tokens
    ///
    /// # Safety
    /// - `logits` must be valid device pointer of shape [batch, vocab_size]
    /// - `output` must be valid device pointer of shape [batch]
    /// - `seeds` must be valid device pointer of shape [batch] containing RNG seeds
    pub unsafe fn topk_sample(
        logits: *const f32,
        output: *mut i32,
        seeds: *const u64,
        batch: i32,
        vocab_size: i32,
        k: i32,
        temperature: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_topk_sample(logits, output, seeds, batch, vocab_size, k, temperature, stream.as_ptr())
        })
    }

    /// Nucleus (top-p) sampling: sample from smallest set with cumulative prob >= top_p
    ///
    /// # Safety
    /// - `logits` must be valid device pointer of shape [batch, vocab_size]
    /// - `output` must be valid device pointer of shape [batch]
    /// - `seeds` must be valid device pointer of shape [batch] containing RNG seeds
    pub unsafe fn topp_sample(
        logits: *const f32,
        output: *mut i32,
        seeds: *const u64,
        batch: i32,
        vocab_size: i32,
        top_p: f32,
        temperature: f32,
        stream: Stream,
    ) -> CudaResult<()> {
        check_result(unsafe {
            cuda_topp_sample(logits, output, seeds, batch, vocab_size, top_p, temperature, stream.as_ptr())
        })
    }
}

// Tests
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
    fn test_stream_default() {
        let stream = Stream::DEFAULT;
        assert!(stream.as_ptr().is_null());
    }
}
