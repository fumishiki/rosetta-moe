//! Multi-Query Attention (MQA) with actual implementation

use crate::ModelConfig;
use crate::tensor::{Tensor, Shape};
use crate::layer::{Layer, Linear};

/// Multi-Query Attention: 12 Q heads, 1 KV head
pub(crate) struct MQAAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MQAAttention {
    pub(crate) fn new(config: &ModelConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let head_dim = config.head_dim;
        let n_heads = config.n_heads;

        Self {
            q_proj: Linear::new(hidden_dim, n_heads * head_dim),  // [hidden, n_heads * head_dim]
            k_proj: Linear::new(hidden_dim, head_dim),            // [hidden, head_dim] - single KV head
            v_proj: Linear::new(hidden_dim, head_dim),            // [hidden, head_dim]
            o_proj: Linear::new(n_heads * head_dim, hidden_dim),  // [n_heads * head_dim, hidden]
            n_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

impl Layer for MQAAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.shape().dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let hidden_dim = dims[2];

        // Project to Q, K, V
        // Q: [batch * seq, n_heads * head_dim] -> [batch, seq, n_heads, head_dim]
        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);

        // Reshape Q: [batch, seq, n_heads * head_dim] -> [batch, n_heads, seq, head_dim]
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        // Compute attention for each head
        // For MQA, K and V are shared across all Q heads
        let mut attn_out = vec![0.0; batch * seq_len * self.n_heads * self.head_dim];

        for b in 0..batch {
            for h in 0..self.n_heads {
                // Compute attention scores: Q @ K^T * scale
                let mut scores = vec![0.0; seq_len * seq_len];

                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // Causal mask: only attend to previous positions
                        if j > i {
                            scores[i * seq_len + j] = f32::NEG_INFINITY;
                        } else {
                            let mut dot = 0.0;
                            for d in 0..self.head_dim {
                                // Q[b, i, h, d] @ K[b, j, d]
                                let q_idx = b * seq_len * self.n_heads * self.head_dim
                                    + i * self.n_heads * self.head_dim
                                    + h * self.head_dim
                                    + d;
                                let k_idx = b * seq_len * self.head_dim
                                    + j * self.head_dim
                                    + d;
                                dot += q_data[q_idx] * k_data[k_idx];
                            }
                            scores[i * seq_len + j] = dot * self.scale;
                        }
                    }
                }

                // Softmax over scores
                for i in 0..seq_len {
                    let row_start = i * seq_len;
                    let row_end = row_start + seq_len;
                    let row = &mut scores[row_start..row_end];

                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

                    for x in row.iter_mut() {
                        *x = (*x - max_val).exp() / exp_sum;
                    }
                }

                // Apply attention: scores @ V
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        let mut sum = 0.0;
                        for j in 0..seq_len {
                            let v_idx = b * seq_len * self.head_dim + j * self.head_dim + d;
                            sum += scores[i * seq_len + j] * v_data[v_idx];
                        }
                        let out_idx = b * seq_len * self.n_heads * self.head_dim
                            + i * self.n_heads * self.head_dim
                            + h * self.head_dim
                            + d;
                        attn_out[out_idx] = sum;
                    }
                }
            }
        }

        // Reshape and project output
        let attn_tensor = Tensor::from_slice(
            &attn_out,
            Shape::new(&[batch, seq_len, self.n_heads * self.head_dim])
        );

        self.o_proj.forward(&attn_tensor)
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // Simplified backward
        self.o_proj.backward(grad_output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.q_proj.weight,
            &self.k_proj.weight,
            &self.v_proj.weight,
            &self.o_proj.weight,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.q_proj.weight,
            &mut self.k_proj.weight,
            &mut self.v_proj.weight,
            &mut self.o_proj.weight,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn test_mqa_attention() {
        let config = ModelConfig::default_6_9b();
        let attn = MQAAttention::new(&config);

        // Small input for testing
        let input = Tensor::randn(Shape::new(&[1, 4, 768]), DType::F32, 42);
        let output = attn.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 4, 768]);
    }
}
