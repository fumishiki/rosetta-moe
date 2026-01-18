//! Tensor types with actual data storage and operations

use std::fmt;

// Error Types
/// Tensor operation error
#[derive(Debug, Clone)]
pub(crate) enum TensorError {
    /// Shape mismatch between tensors
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    /// Invalid dimension count
    InvalidDimension { expected: usize, got: usize },
    /// Index out of bounds
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    /// Inner dimensions don't match for matmul
    MatmulDimensionMismatch { k1: usize, k2: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidDimension { expected, got } => {
                write!(f, "Invalid dimension: expected {}, got {}", expected, got)
            }
            TensorError::IndexOutOfBounds { index, shape } => {
                write!(f, "Index {:?} out of bounds for shape {:?}", index, shape)
            }
            TensorError::MatmulDimensionMismatch { k1, k2 } => {
                write!(f, "Matmul dimension mismatch: {} vs {}", k1, k2)
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Result type for tensor operations
pub(crate) type TensorResult<T> = Result<T, TensorError>;

// Tensor
/// CPU上のTensorを表す（実データ保持）
#[derive(Debug)]
pub(crate) struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DType,
    requires_grad: bool,
    pub(crate) grad: Option<Box<Tensor>>,
}

/// Tensorの形状
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shape(Vec<usize>);

impl Shape {
    pub(crate) fn new(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    pub(crate) fn numel(&self) -> usize {
        if self.0.is_empty() {
            1
        } else {
            self.0.iter().product()
        }
    }

    pub(crate) fn ndim(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Get the last dimension size (e.g., hidden_dim for [batch, seq, hidden])
    pub(crate) fn last_dim(&self) -> usize {
        *self.0.last().unwrap_or(&1)
    }

    /// Get all dimensions except the last one
    pub(crate) fn batch_dims(&self) -> &[usize] {
        if self.0.is_empty() {
            &[]
        } else {
            &self.0[..self.0.len() - 1]
        }
    }

    /// Get product of all dimensions except the last (batch * seq for 3D tensors)
    pub(crate) fn batch_size(&self) -> usize {
        self.batch_dims().iter().product::<usize>().max(1)
    }

    /// Create new shape without the last N dimensions
    pub(crate) fn without_last(&self, n: usize) -> Shape {
        let end = self.0.len().saturating_sub(n);
        Shape::new(&self.0[..end])
    }

    /// Create new shape with modified last dimension
    pub(crate) fn with_last_dim(&self, new_last: usize) -> Shape {
        let mut dims = self.0.clone();
        if let Some(last) = dims.last_mut() {
            *last = new_last;
        } else {
            dims.push(new_last);
        }
        Shape(dims)
    }
}

/// データ型
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub(crate) fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }
}

impl Tensor {
    // ==========================================================================
    // Constructors
    // ==========================================================================

    /// Create tensor filled with zeros
    pub(crate) fn zeros(shape: Shape, dtype: DType) -> Self {
        let numel = shape.numel();
        Self {
            data: vec![0.0; numel],
            shape,
            dtype,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create tensor filled with ones
    pub(crate) fn ones(shape: Shape, dtype: DType) -> Self {
        let numel = shape.numel();
        Self {
            data: vec![1.0; numel],
            shape,
            dtype,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create tensor from slice
    pub(crate) fn from_slice(data: &[f32], shape: Shape) -> Self {
        assert_eq!(data.len(), shape.numel(), "Data length must match shape");
        Self {
            data: data.to_vec(),
            shape,
            dtype: DType::F32,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create tensor with random values (simple LCG for reproducibility)
    pub(crate) fn randn(shape: Shape, dtype: DType, seed: u64) -> Self {
        let numel = shape.numel();
        let mut data = Vec::with_capacity(numel);
        let mut state = seed;

        for _ in 0..numel {
            // Simple LCG: state = (a * state + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [-1, 1] range (approximately normal via Box-Muller would be better)
            let u = (state as f64) / (u64::MAX as f64);
            let v = (u * 2.0 - 1.0) as f32;
            data.push(v * 0.02); // Small init like Kaiming
        }

        Self {
            data,
            shape,
            dtype,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create scalar tensor
    pub(crate) fn scalar(value: f32) -> Self {
        Self {
            data: vec![value],
            shape: Shape::new(&[]),
            dtype: DType::F32,
            requires_grad: false,
            grad: None,
        }
    }

    // Internal Helpers
    /// Wrap computed data into a new Tensor with same shape/dtype
    fn wrap_result(&self, data: Vec<f32>) -> Tensor {
        Tensor {
            data,
            shape: self.shape.clone(),
            dtype: self.dtype,
            requires_grad: false,
            grad: None,
        }
    }

    /// Wrap computed data with a new shape
    fn wrap_with_shape(&self, data: Vec<f32>, shape: Shape) -> Tensor {
        Tensor {
            data,
            shape,
            dtype: self.dtype,
            requires_grad: false,
            grad: None,
        }
    }

    // Accessors
    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Extract 3D dimensions (batch, seq_len, hidden_dim)
    /// Panics if tensor is not 3D
    pub(crate) fn dims_3d(&self) -> (usize, usize, usize) {
        self.try_dims_3d().expect("Expected 3D tensor")
    }

    /// Try to extract 3D dimensions, returning error if not 3D
    pub(crate) fn try_dims_3d(&self) -> TensorResult<(usize, usize, usize)> {
        let dims = self.shape.dims();
        if dims.len() != 3 {
            return Err(TensorError::InvalidDimension { expected: 3, got: dims.len() });
        }
        Ok((dims[0], dims[1], dims[2]))
    }

    /// Get single element (for scalar tensors or by flat index)
    pub(crate) fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() only for scalar tensors");
        self.data[0]
    }

    /// Get element at index
    pub(crate) fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Set element at index
    pub(crate) fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.ndim());
        let dims = self.shape.dims();
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..dims.len()).rev() {
            assert!(indices[i] < dims[i], "Index out of bounds");
            idx += indices[i] * stride;
            stride *= dims[i];
        }
        idx
    }

    // Element-wise Operations
    /// Element-wise addition
    pub(crate) fn add(&self, other: &Tensor) -> Tensor {
        self.try_add(other).expect("Shapes must match for add")
    }

    /// Try element-wise addition, returning error on shape mismatch
    pub(crate) fn try_add(&self, other: &Tensor) -> TensorResult<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(self.wrap_result(data))
    }

    /// Element-wise subtraction
    pub(crate) fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for sub");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        self.wrap_result(data)
    }

    /// Element-wise multiplication
    pub(crate) fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for mul");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        self.wrap_result(data)
    }

    /// Scalar multiplication
    pub(crate) fn scale(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|x| x * s).collect();
        self.wrap_result(data)
    }

    /// Element-wise division
    pub(crate) fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must match for div");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a / b).collect();
        self.wrap_result(data)
    }

    // Activation Functions
    /// SiLU (Swish): x * sigmoid(x)
    pub(crate) fn silu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        self.wrap_result(data)
    }

    /// ReLU: max(0, x)
    pub(crate) fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        self.wrap_result(data)
    }

    /// Softmax along last dimension
    pub(crate) fn softmax(&self) -> Tensor {
        if self.shape.dims().is_empty() {
            return Tensor::ones(self.shape.clone(), self.dtype);
        }

        let last_dim = self.shape.last_dim();
        let outer_size = self.shape.batch_size();
        let mut data = vec![0.0; self.numel()];

        for i in 0..outer_size {
            let start = i * last_dim;
            let row = &self.data[start..start + last_dim];

            // Numerical stability: subtract max
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

            for (j, &x) in row.iter().enumerate() {
                data[start + j] = (x - max_val).exp() / exp_sum;
            }
        }

        self.wrap_result(data)
    }

    // ==========================================================================
    // Reduction Operations
    // ==========================================================================

    /// Sum all elements
    pub(crate) fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub(crate) fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    /// Sum along last dimension
    pub(crate) fn sum_last_dim(&self) -> Tensor {
        if self.shape.dims().is_empty() {
            return self.clone();
        }

        let last_dim = self.shape.last_dim();
        let outer_size = self.shape.batch_size();
        let new_shape = self.shape.without_last(1);
        let mut data = vec![0.0; outer_size];

        for i in 0..outer_size {
            let start = i * last_dim;
            data[i] = self.data[start..start + last_dim].iter().sum();
        }

        self.wrap_with_shape(data, new_shape)
    }

    /// Mean along last dimension
    pub(crate) fn mean_last_dim(&self) -> Tensor {
        if self.shape.dims().is_empty() {
            return self.clone();
        }

        let last_dim = self.shape.last_dim();
        let mut result = self.sum_last_dim();
        for x in result.data.iter_mut() {
            *x /= last_dim as f32;
        }
        result
    }

    /// Matrix multiplication: self @ other
    /// self: [..., M, K], other: [..., K, N] -> [..., M, N]
    pub(crate) fn matmul(&self, other: &Tensor) -> Tensor {
        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();

        assert!(self_dims.len() >= 2, "matmul requires at least 2D tensor");
        assert!(other_dims.len() >= 2, "matmul requires at least 2D tensor");

        let m = self_dims[self_dims.len() - 2];
        let k1 = self.shape.last_dim();
        let k2 = other_dims[other_dims.len() - 2];
        let n = other.shape.last_dim();

        assert_eq!(k1, k2, "Inner dimensions must match for matmul");

        // Simple 2D case
        if self_dims.len() == 2 && other_dims.len() == 2 {
            let mut data = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k1 {
                        sum += self.data[i * k1 + k] * other.data[k * n + j];
                    }
                    data[i * n + j] = sum;
                }
            }
            return self.wrap_with_shape(data, Shape::new(&[m, n]));
        }

        // Batched case (simplified: assumes same batch dims)
        let batch_dims = self.shape.without_last(2);
        let batch_size = batch_dims.numel().max(1);

        let mut out_dims = batch_dims.dims().to_vec();
        out_dims.push(m);
        out_dims.push(n);

        let mut data = vec![0.0; batch_size * m * n];
        let self_stride = m * k1;
        let other_stride = k2 * n;
        let out_stride = m * n;

        for b in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k1 {
                        sum += self.data[b * self_stride + i * k1 + k]
                            * other.data[b * other_stride + k * n + j];
                    }
                    data[b * out_stride + i * n + j] = sum;
                }
            }
        }

        self.wrap_with_shape(data, Shape::new(&out_dims))
    }

    /// Transpose last two dimensions
    pub(crate) fn transpose(&self) -> Tensor {
        let dims = self.shape.dims();
        assert!(dims.len() >= 2, "transpose requires at least 2D tensor");

        let n = self.shape.last_dim();
        let m = dims[dims.len() - 2];
        let batch_dims = self.shape.without_last(2);
        let batch_size = batch_dims.numel().max(1);

        let mut new_dims = batch_dims.dims().to_vec();
        new_dims.push(n);
        new_dims.push(m);

        let mut data = vec![0.0; self.numel()];
        let stride = m * n;

        for b in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    data[b * stride + j * m + i] = self.data[b * stride + i * n + j];
                }
            }
        }

        self.wrap_with_shape(data, Shape::new(&new_dims))
    }

    // Shape Operations
    /// Reshape tensor
    pub(crate) fn reshape(&self, new_shape: Shape) -> Tensor {
        assert_eq!(self.numel(), new_shape.numel(), "Reshape must preserve numel");
        self.wrap_with_shape(self.data.clone(), new_shape)
    }

    /// View as different shape (same data)
    pub(crate) fn view(&self, new_shape: Shape) -> Tensor {
        self.reshape(new_shape)
    }

    // Utility
    pub(crate) fn clone(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            requires_grad: self.requires_grad,
            grad: self.grad.as_ref().map(|g| Box::new(g.as_ref().clone())),
        }
    }

    /// Fill tensor with value
    pub(crate) fn fill_(&mut self, value: f32) {
        self.data.fill(value);
    }

    /// Zero the tensor
    pub(crate) fn zero_(&mut self) {
        self.fill_(0.0);
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(Shape::new(&[2, 3]), DType::F32);
        assert_eq!(t.numel(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(Shape::new(&[2, 3]), DType::F32);
        assert!(t.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_slice(&data, Shape::new(&[2, 2]));
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 1]), 4.0);
    }

    #[test]
    fn test_add() {
        let a = Tensor::ones(Shape::new(&[2, 2]), DType::F32);
        let b = Tensor::ones(Shape::new(&[2, 2]), DType::F32);
        let c = a.add(&b);
        assert!(c.data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_matmul() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(&[2, 3]));
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(&[3, 2]));
        let c = a.matmul(&b);

        assert_eq!(c.shape().dims(), &[2, 2]);
        // c[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        assert_eq!(c.get(&[0, 0]), 22.0);
        // c[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        assert_eq!(c.get(&[0, 1]), 28.0);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], Shape::new(&[1, 3]));
        let s = t.softmax();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu() {
        let t = Tensor::from_slice(&[0.0, 1.0, -1.0], Shape::new(&[3]));
        let s = t.silu();
        // silu(0) = 0 * 0.5 = 0
        assert!((s.data[0] - 0.0).abs() < 1e-6);
        // silu(1) ≈ 0.731
        assert!((s.data[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(&[2, 3]));
        let tr = t.transpose();
        assert_eq!(tr.shape().dims(), &[3, 2]);
        assert_eq!(tr.get(&[0, 0]), 1.0);
        assert_eq!(tr.get(&[0, 1]), 4.0);
        assert_eq!(tr.get(&[1, 0]), 2.0);
    }

    #[test]
    fn test_try_add_success() {
        let a = Tensor::ones(Shape::new(&[2, 2]), DType::F32);
        let b = Tensor::ones(Shape::new(&[2, 2]), DType::F32);
        let result = a.try_add(&b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_add_shape_mismatch() {
        let a = Tensor::ones(Shape::new(&[2, 2]), DType::F32);
        let b = Tensor::ones(Shape::new(&[3, 3]), DType::F32);
        let result = a.try_add(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_try_dims_3d() {
        let t = Tensor::zeros(Shape::new(&[2, 3, 4]), DType::F32);
        assert!(t.try_dims_3d().is_ok());
        assert_eq!(t.try_dims_3d().unwrap(), (2, 3, 4));

        let t2 = Tensor::zeros(Shape::new(&[2, 3]), DType::F32);
        assert!(t2.try_dims_3d().is_err());
    }

    #[test]
    fn test_shape_helpers() {
        let shape = Shape::new(&[2, 3, 4]);
        assert_eq!(shape.last_dim(), 4);
        assert_eq!(shape.batch_dims(), &[2, 3]);
        assert_eq!(shape.batch_size(), 6);
        assert_eq!(shape.without_last(1).dims(), &[2, 3]);
        assert_eq!(shape.without_last(2).dims(), &[2]);
        assert_eq!(shape.with_last_dim(8).dims(), &[2, 3, 8]);
    }
}
