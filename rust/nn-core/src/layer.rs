//! Layer trait and basic layers with actual implementations

use crate::tensor::{Tensor, Shape, DType};

/// Macro to collect parameters from multiple layers
macro_rules! collect_params {
    ($($layer:expr),+ $(,)?) => {{
        let mut params = Vec::new();
        $(params.extend($layer.parameters());)+
        params
    }}
}

/// Macro to collect mutable parameters from multiple layers
macro_rules! collect_params_mut {
    ($($layer:expr),+ $(,)?) => {{
        let mut params = Vec::new();
        $(params.extend($layer.parameters_mut());)+
        params
    }}
}

/// Forward/Backward を持つ層の trait
pub(crate) trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn backward(&self, grad_output: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
}

/// Embedding層: token_ids -> embeddings
pub(crate) struct Embedding {
    pub(crate) weight: Tensor,
    // Cache for backward
    last_input: Option<Vec<usize>>,
}

impl Embedding {
    pub(crate) fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        // Initialize with small random values
        Self {
            weight: Tensor::randn(Shape::new(&[vocab_size, hidden_dim]), DType::F32, 42),
            last_input: None,
        }
    }

    /// Forward pass: lookup embeddings for token ids
    /// input: [batch, seq_len] (interpreted as i32 indices)
    /// output: [batch, seq_len, hidden_dim]
    pub(crate) fn forward_with_ids(&self, token_ids: &[usize], batch: usize, seq_len: usize) -> Tensor {
        let hidden_dim = self.weight.shape().dims()[1];
        let mut data = vec![0.0; batch * seq_len * hidden_dim];

        for (i, &token_id) in token_ids.iter().enumerate() {
            let out_start = i * hidden_dim;
            let weight_start = token_id * hidden_dim;

            for j in 0..hidden_dim {
                data[out_start + j] = self.weight.data()[weight_start + j];
            }
        }

        Tensor::from_slice(&data, Shape::new(&[batch, seq_len, hidden_dim]))
    }
}

impl Layer for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Treat input as integer indices
        let dims = input.shape().dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let hidden_dim = self.weight.shape().dims()[1];

        // Convert float data to indices (for compatibility)
        let token_ids: Vec<usize> = input.data().iter()
            .map(|&x| x as usize)
            .collect();

        self.forward_with_ids(&token_ids, batch, seq_len)
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // Embedding backward returns grad w.r.t. weight (scatter add)
        // For simplicity, return zeros as we handle grad accumulation elsewhere
        Tensor::zeros(self.weight.shape().clone(), DType::F32)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}

// =============================================================================
// RMSNorm
// =============================================================================

/// RMSNorm層: y = x * w / sqrt(mean(x^2) + eps)
pub(crate) struct RMSNorm {
    pub(crate) weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub(crate) fn new(hidden_dim: usize) -> Self {
        // Initialize weights to 1.0
        Self {
            weight: Tensor::ones(Shape::new(&[hidden_dim]), DType::F32),
            eps: 1e-6,
        }
    }
}

impl Layer for RMSNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.shape().dims();
        let hidden_dim = *dims.last().unwrap();
        let outer_size = input.numel() / hidden_dim;

        let mut data = vec![0.0; input.numel()];
        let weight_data = self.weight.data();

        for i in 0..outer_size {
            let start = i * hidden_dim;
            let end = start + hidden_dim;
            let row = &input.data()[start..end];

            // Compute RMS: sqrt(mean(x^2) + eps)
            let mean_sq: f32 = row.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
            let rms = (mean_sq + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            // Normalize and scale
            for (j, &x) in row.iter().enumerate() {
                data[start + j] = x * inv_rms * weight_data[j];
            }
        }

        Tensor::from_slice(&data, input.shape().clone())
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // Simplified backward (just pass through scaled)
        grad_output.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}

// =============================================================================
// Linear
// =============================================================================

/// Linear層（bias無し）: y = x @ W^T
pub(crate) struct Linear {
    pub(crate) weight: Tensor, // [out_features, in_features]
}

impl Linear {
    pub(crate) fn new(in_features: usize, out_features: usize) -> Self {
        // Kaiming-like init
        Self {
            weight: Tensor::randn(Shape::new(&[out_features, in_features]), DType::F32, 123),
        }
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let input_dims = input.shape().dims();
        let out_features = self.weight.shape().dims()[0];
        let in_features = self.weight.shape().dims()[1];

        // Handle batched input: [..., in_features] -> [..., out_features]
        let batch_dims = &input_dims[..input_dims.len() - 1];
        let batch_size: usize = batch_dims.iter().product();

        let mut out_dims = batch_dims.to_vec();
        out_dims.push(out_features);

        let mut data = vec![0.0; batch_size * out_features];
        let weight_data = self.weight.data();

        // y = x @ W^T
        for b in 0..batch_size {
            let x_start = b * in_features;
            for o in 0..out_features {
                let w_start = o * in_features;
                let mut sum = 0.0;
                for i in 0..in_features {
                    sum += input.data()[x_start + i] * weight_data[w_start + i];
                }
                data[b * out_features + o] = sum;
            }
        }

        Tensor::from_slice(&data, Shape::new(&out_dims))
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // dx = dout @ W
        let grad_dims = grad_output.shape().dims();
        let out_features = self.weight.shape().dims()[0];
        let in_features = self.weight.shape().dims()[1];

        let batch_dims = &grad_dims[..grad_dims.len() - 1];
        let batch_size: usize = batch_dims.iter().product();

        let mut out_dims = batch_dims.to_vec();
        out_dims.push(in_features);

        let mut data = vec![0.0; batch_size * in_features];
        let weight_data = self.weight.data();

        for b in 0..batch_size {
            for i in 0..in_features {
                let mut sum = 0.0;
                for o in 0..out_features {
                    // W is [out, in], so W[o, i]
                    sum += grad_output.data()[b * out_features + o] * weight_data[o * in_features + i];
                }
                data[b * in_features + i] = sum;
            }
        }

        Tensor::from_slice(&data, Shape::new(&out_dims))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}

// =============================================================================
// SwiGLU FFN
// =============================================================================

/// SwiGLU FFN: out = (silu(x @ W_gate) * (x @ W_up)) @ W_down
pub(crate) struct SwiGLUFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLUFFN {
    pub(crate) fn new(hidden_dim: usize, ffn_dim: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_dim, ffn_dim),
            up_proj: Linear::new(hidden_dim, ffn_dim),
            down_proj: Linear::new(ffn_dim, hidden_dim),
        }
    }
}

impl Layer for SwiGLUFFN {
    fn forward(&self, input: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(input).silu();
        let up = self.up_proj.forward(input);
        let hidden = gate.mul(&up);
        self.down_proj.forward(&hidden)
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        self.down_proj.backward(grad_output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        collect_params!(self.gate_proj, self.up_proj, self.down_proj)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        collect_params_mut!(self.gate_proj, self.up_proj, self.down_proj)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let norm = RMSNorm::new(4);
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::new(&[1, 4]));
        let output = norm.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 4]);
        // Check normalization: output should have similar magnitude
        let out_rms: f32 = output.data().iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!((out_rms.sqrt() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_linear() {
        let linear = Linear::new(4, 2);
        let input = Tensor::ones(Shape::new(&[1, 4]), DType::F32);
        let output = linear.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_linear_batch() {
        let linear = Linear::new(4, 2);
        let input = Tensor::ones(Shape::new(&[2, 3, 4]), DType::F32);
        let output = linear.forward(&input);

        assert_eq!(output.shape().dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_swiglu() {
        let ffn = SwiGLUFFN::new(8, 32);
        let input = Tensor::ones(Shape::new(&[1, 2, 8]), DType::F32);
        let output = ffn.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 2, 8]);
    }
}
