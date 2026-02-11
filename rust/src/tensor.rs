// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Tensor primitives and CPU ops.
//!
//! Core tensor type for the MoE Transformer. All storage is f32, row-major,
//! flat `Vec<f32>`. Shape metadata is kept separately in [`Shape`].
//!
//! Key design decisions:
//! - `from_vec()` takes ownership (zero-copy), `from_slice()` copies -- prefer
//!   `from_vec` when the caller already owns the data.
//! - `softmax_into_slice` / `softmax_in_place` use the numerically stable
//!   max-subtraction trick to avoid exp() overflow.
//! - Batched matmul delegates to Apple Accelerate `cblas_sgemm` via the
//!   `accelerate` module. The Tensor layer handles batch iteration and shape
//!   bookkeeping; the hot inner product is always BLAS.

use std::fmt;

/// Tensor shape with row-major semantics.
///
/// Empty shape `[]` represents a scalar with `numel() = 1`.
/// Strides are implicit: `stride[d] = product(dims[d+1..])`.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        if self.0.is_empty() {
            1
        } else {
            self.0.iter().product()
        }
    }

    pub fn last_dim(&self) -> usize {
        self.0.last().copied().unwrap_or(1)
    }

    pub fn batch_dims(&self) -> &[usize] {
        if self.0.is_empty() {
            &[]
        } else {
            &self.0[..self.0.len() - 1]
        }
    }

    /// Total number of elements in the batch dimensions.
    /// Returns 1 (not 0) for scalars/1-D tensors so callers can always
    /// divide total numel by batch_size to get the last-dim width.
    pub fn batch_size(&self) -> usize {
        let prod = self.batch_dims().iter().product::<usize>();
        if prod == 0 { 1 } else { prod }
    }

    pub fn matrix_dims(&self) -> (usize, usize) {
        let n = self.0.len();
        assert!(n >= 2, "matrix_dims requires >=2D");
        (self.0[n - 2], self.0[n - 1])
    }

    pub fn without_last(&self, n: usize) -> Shape {
        let end = self.0.len().saturating_sub(n);
        Shape::new(&self.0[..end])
    }

    pub fn with_last_dim(&self, new_last: usize) -> Shape {
        let mut dims = self.0.clone();
        if let Some(last) = dims.last_mut() {
            *last = new_last;
        } else {
            dims.push(new_last);
        }
        Shape(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Data type for tensor storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit brain floating point
    BF16,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Tensor op errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    InvalidDimension {
        expected: usize,
        got: usize,
    },
    InvalidMatrixRank {
        shape: Vec<usize>,
    },
    MatmulBatchMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    MatmulInnerMismatch {
        lhs_k: usize,
        rhs_k: usize,
    },
    NumelMismatch {
        expected: usize,
        got: usize,
    },
    IndexRankMismatch {
        expected: usize,
        got: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        bound: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { lhs, rhs } => {
                write!(f, "shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::InvalidDimension { expected, got } => {
                write!(f, "invalid dimension: expected {expected}, got {got}")
            }
            Self::InvalidMatrixRank { shape } => {
                write!(f, "matrix op requires >=2D, got {shape:?}")
            }
            Self::MatmulBatchMismatch { lhs, rhs } => {
                write!(f, "matmul batch mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::MatmulInnerMismatch { lhs_k, rhs_k } => {
                write!(f, "matmul inner mismatch: lhs_k={lhs_k}, rhs_k={rhs_k}")
            }
            Self::NumelMismatch { expected, got } => {
                write!(f, "numel mismatch: expected {expected}, got {got}")
            }
            Self::IndexRankMismatch { expected, got } => {
                write!(f, "index rank mismatch: expected {expected}, got {got}")
            }
            Self::IndexOutOfBounds { axis, index, bound } => {
                write!(
                    f,
                    "index out of bounds at axis {axis}: index={index}, bound={bound}"
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

pub type TensorResult<T> = Result<T, TensorError>;

macro_rules! panic_wrapper {
    (pub fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty => $try_name:ident) => {
        pub fn $name(&self $(, $arg: $ty)*) -> $ret {
            match self.$try_name($($arg),*) {
                Ok(v) => v,
                Err(e) => panic!("{e}"),
            }
        }
    };
    (pub fn $name:ident(&mut self $(, $arg:ident : $ty:ty)*) -> $ret:ty => $try_name:ident) => {
        pub fn $name(&mut self $(, $arg: $ty)*) -> $ret {
            match self.$try_name($($arg),*) {
                Ok(v) => v,
                Err(e) => panic!("{e}"),
            }
        }
    };
}

/// CPU tensor with row-major f32 storage.
///
/// Memory layout: flat `Vec<f32>` with row-major (C-order) indexing.
/// Index calculation: `flat_idx = sum_d(index_d * stride_d)` where
/// `stride_d = product(dims[d+1..])`.
///
/// Gradient is stored as an optional boxed Tensor (same shape). This avoids
/// a separate gradient map and keeps param + grad colocated for the optimizer.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DType,
    // Boxed to keep Tensor itself small on the stack (Box = 1 pointer).
    grad: Option<Box<Tensor>>,
}

impl Tensor {
    fn from_parts(data: Vec<f32>, shape: Shape, dtype: DType) -> Tensor {
        Tensor {
            data,
            shape,
            dtype,
            grad: None,
        }
    }

    fn like_with_data(&self, data: Vec<f32>) -> Tensor {
        Self::from_parts(data, self.shape.clone(), self.dtype)
    }

    fn with_shape_like(&self, data: Vec<f32>, shape: Shape) -> Tensor {
        Self::from_parts(data, shape, self.dtype)
    }

    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        Self::from_parts(vec![0.0; shape.numel()], shape, dtype)
    }

    pub fn ones(shape: Shape, dtype: DType) -> Self {
        Self::from_parts(vec![1.0; shape.numel()], shape, dtype)
    }

    pub fn from_slice(data: &[f32], shape: Shape) -> Self {
        assert_eq!(data.len(), shape.numel(), "data length vs shape mismatch");
        Self::from_parts(data.to_vec(), shape, DType::F32)
    }

    /// Create tensor from an owned `Vec<f32>`, **zero-copy** (no allocation).
    /// Prefer this over `from_slice` whenever the caller already owns the buffer.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.numel(), "data length vs shape mismatch");
        Self::from_parts(data, shape, DType::F32)
    }

    pub fn scalar(value: f32) -> Self {
        Self::from_parts(vec![value], Shape::new(&[]), DType::F32)
    }

    /// Generate N(0,1) samples using LCG + Box-Muller transform.
    ///
    /// Box-Muller: given u1,u2 ~ Uniform(0,1):
    ///   z0 = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    ///   z1 = sqrt(-2*ln(u1)) * sin(2*pi*u2)
    /// Both z0,z1 are independent N(0,1).
    ///
    /// LCG is intentionally simple -- this is for reproducible weight init,
    /// not cryptographic randomness.
    pub fn randn(shape: Shape, dtype: DType, seed: u64) -> Self {
        let mut state = seed;
        let mut lcg_uniform = || -> f64 {
            // LCG with Knuth's constants (period 2^64)
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Clamp away from 0 to avoid ln(0) = -inf in Box-Muller
            (state as f64 / u64::MAX as f64).clamp(1e-10, 1.0)
        };
        let n = shape.numel();
        let mut data = Vec::with_capacity(n);
        // Box-Muller produces samples in pairs (cos, sin)
        let mut i = 0;
        while i < n {
            let u1 = lcg_uniform();
            let u2 = lcg_uniform();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data.push((r * theta.cos()) as f32);
            if i + 1 < n {
                data.push((r * theta.sin()) as f32);
            }
            i += 2;
        }
        Self::from_parts(data, shape, dtype)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume the tensor and return the underlying data Vec.
    /// Zero-copy: moves the Vec out without cloning.
    /// Useful for recovering pre-allocated buffers after a computation.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() requires scalar tensor");
        self.data[0]
    }

    pub fn try_dims_3d(&self) -> TensorResult<(usize, usize, usize)> {
        let d = self.shape.dims();
        if d.len() != 3 {
            return Err(TensorError::InvalidDimension {
                expected: 3,
                got: d.len(),
            });
        }
        Ok((d[0], d[1], d[2]))
    }

    panic_wrapper!(pub fn dims_3d(&self) -> (usize, usize, usize) => try_dims_3d);

    pub fn try_get(&self, indices: &[usize]) -> TensorResult<f32> {
        self.try_flat_index(indices).map(|idx| self.data[idx])
    }

    panic_wrapper!(pub fn get(&self, indices: &[usize]) -> f32 => try_get);

    pub fn try_set(&mut self, indices: &[usize], value: f32) -> TensorResult<()> {
        let idx = self.try_flat_index(indices)?;
        self.data[idx] = value;
        Ok(())
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        if let Err(e) = self.try_set(indices, value) {
            panic!("{e}");
        }
    }

    /// Convert multi-dim indices to a flat offset.
    /// Row-major formula: flat = ((idx[0]*dim[1] + idx[1])*dim[2] + idx[2])...
    /// Implemented as a fold: acc = acc*bound + idx, which naturally produces
    /// the row-major linear index.
    fn try_flat_index(&self, indices: &[usize]) -> TensorResult<usize> {
        let dims = self.shape.dims();
        if indices.len() != dims.len() {
            return Err(TensorError::IndexRankMismatch {
                expected: dims.len(),
                got: indices.len(),
            });
        }
        dims.iter()
            .zip(indices)
            .enumerate()
            .try_fold(0usize, |acc, (axis, (&bound, &i))| {
                if i >= bound {
                    Err(TensorError::IndexOutOfBounds {
                        axis,
                        index: i,
                        bound,
                    })
                } else {
                    Ok(acc * bound + i)
                }
            })
    }

    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_deref()
    }

    pub fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_deref_mut()
    }

    /// Apply AdamW update in-place, accessing data and grad without copying.
    ///
    /// Temporarily takes the grad out of self to avoid simultaneous mutable/immutable
    /// borrows on the same struct, then puts it back. This is zero-copy: no grad
    /// data is cloned.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_update(
        &mut self,
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        corr1: f32,
        corr2: f32,
    ) {
        // Take grad out of self to split the borrow (zero-copy move)
        let grad_box = self.grad.take();
        if let Some(ref g) = grad_box {
            crate::simd::adamw_step_simd(
                &mut self.data,
                m,
                v,
                g.data(),
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                corr1,
                corr2,
            );
        } else {
            let zeros = vec![0.0f32; self.data.len()];
            crate::simd::adamw_step_simd(
                &mut self.data,
                m,
                v,
                &zeros,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                corr1,
                corr2,
            );
        }
        // Put grad back (zero-copy move)
        self.grad = grad_box;
    }

    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(Box::new(grad));
    }

    /// Zero gradient in place if it exists, avoiding reallocation.
    /// If no grad exists yet, leave as None (accumulate_grad will create it on first use).
    pub fn clear_grad(&mut self) {
        if let Some(g) = &mut self.grad {
            for v in g.data_mut() {
                *v = 0.0;
            }
        }
    }

    fn map_unary(&self, op: impl Fn(f32) -> f32) -> Tensor {
        self.like_with_data(self.data.iter().copied().map(op).collect())
    }

    pub fn scale(&self, s: f32) -> Tensor {
        self.map_unary(|x| x * s)
    }

    // SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    pub fn silu(&self) -> Tensor {
        self.map_unary(|x| x / (1.0 + (-x).exp()))
    }

    pub fn relu(&self) -> Tensor {
        self.map_unary(|x| x.max(0.0))
    }

    fn ensure_same_shape(&self, other: &Tensor) -> TensorResult<()> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                lhs: self.shape.dims().to_vec(),
                rhs: other.shape.dims().to_vec(),
            });
        }
        Ok(())
    }

    fn map_binary(&self, other: &Tensor, op: impl Fn(f32, f32) -> f32) -> TensorResult<Tensor> {
        self.ensure_same_shape(other)?;
        let out = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| op(a, b))
            .collect();
        Ok(self.like_with_data(out))
    }

    pub fn try_add(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.map_binary(other, |a, b| a + b)
    }

    panic_wrapper!(pub fn add(&self, other: &Tensor) -> Tensor => try_add);

    pub fn try_sub(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.map_binary(other, |a, b| a - b)
    }

    panic_wrapper!(pub fn sub(&self, other: &Tensor) -> Tensor => try_sub);

    pub fn try_mul(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.map_binary(other, |a, b| a * b)
    }

    panic_wrapper!(pub fn mul(&self, other: &Tensor) -> Tensor => try_mul);

    pub fn try_div(&self, other: &Tensor) -> TensorResult<Tensor> {
        self.map_binary(other, |a, b| a / b)
    }

    panic_wrapper!(pub fn div(&self, other: &Tensor) -> Tensor => try_div);

    /// Element-wise add writing into a pre-allocated output (`_into` pattern).
    /// Avoids allocating a new Vec on every residual connection.
    pub fn add_into(&self, other: &Tensor, out: &mut Tensor) -> TensorResult<()> {
        self.ensure_same_shape(other)?;
        self.ensure_same_shape(out)?;
        for ((o, &a), &b) in out.data.iter_mut().zip(&self.data).zip(&other.data) {
            *o = a + b;
        }
        Ok(())
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        let n = self.numel();
        if n == 0 { 0.0 } else { self.sum() / n as f32 }
    }

    pub fn sum_last_dim(&self) -> Tensor {
        if self.shape.dims().is_empty() {
            return self.clone();
        }
        let w = self.shape.last_dim();
        let out: Vec<f32> = self.data.chunks_exact(w).map(|c| c.iter().sum()).collect();
        self.with_shape_like(out, self.shape.without_last(1))
    }

    pub fn mean_last_dim(&self) -> Tensor {
        if self.shape.dims().is_empty() {
            return self.clone();
        }
        self.sum_last_dim()
            .scale(1.0 / self.shape.last_dim() as f32)
    }

    /// Applies softmax along the last dimension, writing result into `out`.
    /// Softmax: p_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
    /// The `_into` pattern: caller provides a pre-allocated output buffer,
    /// avoiding allocation inside the hot path.
    pub fn softmax_into(&self, out: &mut Tensor) -> TensorResult<()> {
        self.ensure_same_shape(out)?;
        if self.shape.dims().is_empty() {
            out.fill_(1.0);
            return Ok(());
        }
        let w = self.shape.last_dim();
        for (src, dst) in self.data.chunks_exact(w).zip(out.data.chunks_exact_mut(w)) {
            softmax_into_slice(src, dst);
        }
        Ok(())
    }

    pub fn softmax(&self) -> Tensor {
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype);
        if let Err(e) = self.softmax_into(&mut out) {
            panic!("{e}");
        }
        out
    }

    fn try_matrix_dims(&self) -> TensorResult<(usize, usize)> {
        if self.shape.ndim() < 2 {
            return Err(TensorError::InvalidMatrixRank {
                shape: self.shape.dims().to_vec(),
            });
        }
        Ok(self.shape.matrix_dims())
    }

    /// Batched matrix multiplication: C = A @ B
    ///   A: [..., m, k], B: [..., k, n] -> C: [..., m, n]
    ///
    /// Iterates over batch dimensions, delegating each (m,k)x(k,n) slice
    /// to `cblas_sgemm` via `accelerate::sgemm`. Row-major throughout:
    /// A is contiguous in [m*k] chunks, B in [k*n] chunks.
    pub fn try_matmul(&self, other: &Tensor) -> TensorResult<Tensor> {
        if self.shape.batch_dims() != other.shape.batch_dims() {
            return Err(TensorError::MatmulBatchMismatch {
                lhs: self.shape.batch_dims().to_vec(),
                rhs: other.shape.batch_dims().to_vec(),
            });
        }

        let (m, k) = self.try_matrix_dims()?;
        let (k2, n) = other.try_matrix_dims()?;
        if k != k2 {
            return Err(TensorError::MatmulInnerMismatch {
                lhs_k: k,
                rhs_k: k2,
            });
        }

        let batch = self.shape.batch_size();
        let out_stride = m * n;
        let mut out = vec![0.0; batch * out_stride];

        // Each batch element is an independent matmul: C[b] = A[b] @ B[b]
        for (bt, (lhs_batch, rhs_batch)) in self
            .data
            .chunks_exact(m * k)
            .zip(other.data.chunks_exact(k * n))
            .enumerate()
        {
            let o = &mut out[bt * out_stride..(bt + 1) * out_stride];
            // Delegates to cblas_sgemm (Apple Accelerate / AMX)
            crate::accelerate::sgemm(m, n, k, 1.0, lhs_batch, rhs_batch, 0.0, o);
        }

        let batch_d = self.shape.batch_dims();
        let mut out_dims = Vec::with_capacity(batch_d.len() + 2);
        out_dims.extend_from_slice(batch_d);
        out_dims.push(m);
        out_dims.push(n);
        Ok(self.with_shape_like(out, Shape::from(out_dims)))
    }

    panic_wrapper!(pub fn matmul(&self, other: &Tensor) -> Tensor => try_matmul);

    /// Transpose the last two dimensions: [..., m, n] -> [..., n, m].
    /// Row-major: src[i,j] at offset i*n+j maps to dst[j,i] at offset j*m+i.
    pub fn try_transpose(&self) -> TensorResult<Tensor> {
        let (m, n) = self.try_matrix_dims()?;
        let stride = m * n;
        let mut out = vec![0.0; self.numel()];

        for (src_batch, dst_batch) in self
            .data
            .chunks_exact(stride)
            .zip(out.chunks_exact_mut(stride))
        {
            for i in 0..m {
                for j in 0..n {
                    // src row-major: [i,j] = i*n+j; dst row-major: [j,i] = j*m+i
                    dst_batch[j * m + i] = src_batch[i * n + j];
                }
            }
        }

        let batch_d = self.shape.batch_dims();
        let mut out_dims = Vec::with_capacity(batch_d.len() + 2);
        out_dims.extend_from_slice(batch_d);
        out_dims.push(n);
        out_dims.push(m);
        Ok(self.with_shape_like(out, Shape::from(out_dims)))
    }

    panic_wrapper!(pub fn transpose(&self) -> Tensor => try_transpose);

    pub fn try_reshape(&self, new_shape: Shape) -> TensorResult<Tensor> {
        let expected = self.numel();
        let got = new_shape.numel();
        if expected != got {
            return Err(TensorError::NumelMismatch { expected, got });
        }
        Ok(self.with_shape_like(self.data.clone(), new_shape))
    }

    panic_wrapper!(pub fn reshape(&self, new_shape: Shape) -> Tensor => try_reshape);

    /// Reshape that only changes shape metadata -- zero-copy.
    /// Consumes self so the original cannot be used (avoids aliasing).
    pub fn reshape_move(mut self, new_shape: Shape) -> Tensor {
        assert_eq!(
            self.shape.numel(),
            new_shape.numel(),
            "reshape_move: numel mismatch {} vs {}",
            self.shape.numel(),
            new_shape.numel()
        );
        self.shape = new_shape;
        self
    }

    pub fn view(&self, new_shape: Shape) -> Tensor {
        self.reshape(new_shape)
    }

    /// In-place SiLU: x[i] = x[i] / (1 + exp(-x[i])).
    /// Avoids allocating a new tensor (Julia's silu_in_place! equivalent).
    pub fn silu_in_place(&mut self) {
        for v in self.data.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    /// In-place element-wise multiply: self[i] *= other[i].
    /// Avoids allocating a new tensor (Julia's mul_in_place! equivalent).
    pub fn mul_in_place(&mut self, other: &Tensor) {
        debug_assert_eq!(self.numel(), other.numel());
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a *= b;
        }
    }

    /// In-place element-wise add: self[i] += other[i].
    /// Avoids allocating a new tensor for residual connections.
    pub fn add_in_place(&mut self, other: &Tensor) {
        debug_assert_eq!(self.numel(), other.numel());
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// Replace shape metadata in place (zero-copy).
    pub fn reshape_in_place(&mut self, new_shape: Shape) {
        assert_eq!(
            self.shape.numel(),
            new_shape.numel(),
            "reshape_in_place: numel mismatch {} vs {}",
            self.shape.numel(),
            new_shape.numel()
        );
        self.shape = new_shape;
    }

    pub fn fill_(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn zero_(&mut self) {
        self.fill_(0.0);
    }
}

/// Softmax: p_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
///
/// Numerically stable: subtracting max(x) before exp() prevents overflow.
/// The denominator is clamped to 1e-12 to avoid division by zero when all
/// inputs are -inf (degenerate case with causal masking).
pub(crate) fn softmax_into_slice(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    // Pass 1: find max for numerical stability
    let max_v = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Pass 2: exp(x - max) and accumulate sum
    let mut sum = 0.0f32;
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = (x - max_v).exp();
        sum += *d;
    }
    // Pass 3: normalize
    let inv = 1.0 / sum.max(1e-12);
    for d in dst.iter_mut() {
        *d *= inv;
    }
}

/// In-place softmax over a mutable slice. Same algorithm as `softmax_into_slice`
/// but overwrites the input buffer directly (used in attention score normalization
/// and sampling where we don't need the original logits).
pub fn softmax_in_place(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
    let max_v = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in xs.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-12);
    for v in xs.iter_mut() {
        *v *= inv;
    }
}

/// L1-normalize a slice in place: x_i = x_i / sum(x).
/// Used by top-p sampling to re-normalize after truncation.
pub(crate) fn normalize_in_place(xs: &mut [f32]) {
    let sum: f32 = xs.iter().sum();
    if sum <= 0.0 {
        return;
    }
    let inv = 1.0 / sum;
    for v in xs.iter_mut() {
        *v *= inv;
    }
}
