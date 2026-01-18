//! Mixture of Experts (MoE) with actual implementation

use crate::ModelConfig;
use crate::tensor::{Tensor, Shape};
use crate::layer::{Layer, Linear, SwiGLUFFN};

/// Expert FFN - type alias for SwiGLUFFN (identical architecture)
pub(crate) type ExpertFFN = SwiGLUFFN;

/// Router for expert selection with top-k
pub(crate) struct Router {
    gate: Linear,
    n_experts: usize,
}

impl Router {
    pub(crate) fn new(hidden_dim: usize, n_experts: usize) -> Self {
        Self {
            gate: Linear::new(hidden_dim, n_experts),
            n_experts,
        }
    }

    /// Top-k selection with softmax weights
    /// Returns: (expert_indices, expert_weights) both [batch * seq_len, top_k]
    pub(crate) fn route(&self, input: &Tensor, top_k: usize) -> (Vec<Vec<usize>>, Tensor) {
        // Compute router logits
        let logits = self.gate.forward(input);
        let probs = logits.softmax();

        let dims = input.shape().dims();
        let batch_seq = dims[0] * dims[1];
        let hidden_dim = dims[2];

        let probs_data = probs.data();
        let mut indices = Vec::with_capacity(batch_seq);
        let mut weights_data = vec![0.0; batch_seq * top_k];

        for i in 0..batch_seq {
            let row_start = i * self.n_experts;

            // Find top-k experts
            let mut expert_scores: Vec<(usize, f32)> = (0..self.n_experts)
                .map(|j| (j, probs_data[row_start + j]))
                .collect();
            expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_experts: Vec<usize> = expert_scores.iter()
                .take(top_k)
                .map(|(idx, _)| *idx)
                .collect();

            // Renormalize weights for selected experts
            let weight_sum: f32 = expert_scores.iter()
                .take(top_k)
                .map(|(_, w)| w)
                .sum();

            for (k, (_, w)) in expert_scores.iter().take(top_k).enumerate() {
                weights_data[i * top_k + k] = w / weight_sum;
            }

            indices.push(top_experts);
        }

        let weights = Tensor::from_slice(&weights_data, Shape::new(&[batch_seq, top_k]));
        (indices, weights)
    }
}

/// MoE Layer with multiple experts
pub(crate) struct MoELayer {
    router: Router,
    experts: Vec<ExpertFFN>,
    top_k: usize,
    n_experts: usize,
}

impl MoELayer {
    pub(crate) fn new(config: &ModelConfig) -> Self {
        let experts = (0..config.n_experts)
            .map(|_| ExpertFFN::new(config.hidden_dim, config.ffn_dim))
            .collect();

        Self {
            router: Router::new(config.hidden_dim, config.n_experts),
            experts,
            top_k: config.top_k_experts,
            n_experts: config.n_experts,
        }
    }
}

impl Layer for MoELayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let dims = input.shape().dims();
        let batch = dims[0];
        let seq_len = dims[1];
        let hidden_dim = dims[2];
        let batch_seq = batch * seq_len;

        // 1. Route tokens
        let (expert_indices, expert_weights) = self.router.route(input, self.top_k);
        let weights_data = expert_weights.data();

        // 2. Dispatch to experts and combine outputs
        let mut output = vec![0.0; batch * seq_len * hidden_dim];

        for i in 0..batch_seq {
            let token_indices = &expert_indices[i];
            let input_start = i * hidden_dim;

            // Get input token
            let token_data: Vec<f32> = input.data()[input_start..input_start + hidden_dim].to_vec();
            let token_tensor = Tensor::from_slice(&token_data, Shape::new(&[1, 1, hidden_dim]));

            // Process through selected experts
            for (k, &expert_idx) in token_indices.iter().enumerate() {
                let weight = weights_data[i * self.top_k + k];
                let expert_out = self.experts[expert_idx].forward(&token_tensor);

                // Weighted accumulation
                for j in 0..hidden_dim {
                    output[input_start + j] += weight * expert_out.data()[j];
                }
            }
        }

        Tensor::from_slice(&output, Shape::new(&[batch, seq_len, hidden_dim]))
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // Simplified: distribute gradient equally to all experts
        grad_output.scale(1.0 / self.top_k as f32)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.router.gate.weight];
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.router.gate.weight];
        for expert in &mut self.experts {
            params.extend(expert.parameters_mut());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    #[test]
    fn test_expert_ffn() {
        let expert = ExpertFFN::new(8, 32);
        let input = Tensor::ones(Shape::new(&[1, 1, 8]), DType::F32);
        let output = expert.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 1, 8]);
    }

    #[test]
    fn test_router() {
        let router = Router::new(8, 4);
        let input = Tensor::randn(Shape::new(&[1, 2, 8]), DType::F32, 42);
        let (indices, weights) = router.route(&input, 2);

        assert_eq!(indices.len(), 2);  // batch_seq = 1 * 2
        assert_eq!(indices[0].len(), 2);  // top_k = 2
        assert_eq!(weights.shape().dims(), &[2, 2]);

        // Weights should sum to 1
        let w_sum: f32 = weights.data()[0..2].iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_moe_layer() {
        // Smaller config for testing
        let config = ModelConfig {
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            n_experts: 4,
            top_k_experts: 2,
            vocab_size: 100,
            max_seq_len: 16,
            ffn_dim: 32,
            head_dim: 4,
            rope_base: 10000.0,
            rope_alpha: 1.0,
        };

        let moe = MoELayer::new(&config);
        let input = Tensor::randn(Shape::new(&[1, 2, 8]), DType::F32, 42);
        let output = moe.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 2, 8]);
    }
}
