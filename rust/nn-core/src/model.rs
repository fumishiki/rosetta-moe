//! Transformer model with actual forward pass

use crate::ModelConfig;
use crate::tensor::Tensor;
use crate::layer::{Layer, Embedding, RMSNorm, Linear};
use crate::attention::MQAAttention;
use crate::moe::MoELayer;

// Transformer Block
/// Single Transformer Block with MoE
/// Pre-norm architecture: x = x + attn(norm(x)), x = x + moe(norm(x))
pub(crate) struct TransformerBlock {
    attn_norm: RMSNorm,
    attention: MQAAttention,
    ffn_norm: RMSNorm,
    moe: MoELayer,
}

impl TransformerBlock {
    pub(crate) fn new(config: &ModelConfig) -> Self {
        Self {
            attn_norm: RMSNorm::new(config.hidden_dim),
            attention: MQAAttention::new(config),
            ffn_norm: RMSNorm::new(config.hidden_dim),
            moe: MoELayer::new(config),
        }
    }
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Pre-norm architecture
        // x = x + attn(norm(x))
        let normed = self.attn_norm.forward(input);
        let attn_out = self.attention.forward(&normed);
        let x = input.add(&attn_out);

        // x = x + moe(norm(x))
        let normed = self.ffn_norm.forward(&x);
        let moe_out = self.moe.forward(&normed);
        x.add(&moe_out)
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // Simplified backward
        let grad = self.moe.backward(grad_output);
        self.attention.backward(&grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.attn_norm.parameters();
        params.extend(self.attention.parameters());
        params.extend(self.ffn_norm.parameters());
        params.extend(self.moe.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.attn_norm.parameters_mut();
        params.extend(self.attention.parameters_mut());
        params.extend(self.ffn_norm.parameters_mut());
        params.extend(self.moe.parameters_mut());
        params
    }
}

// =============================================================================
// MoE Transformer
// =============================================================================

/// MoE Transformer (6.9B total, 1.8B active)
pub(crate) struct MoETransformer {
    config: ModelConfig,
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RMSNorm,
    lm_head: Linear,
}

impl MoETransformer {
    pub(crate) fn new(config: ModelConfig) -> Self {
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config))
            .collect();

        Self {
            embedding: Embedding::new(config.vocab_size, config.hidden_dim),
            blocks,
            final_norm: RMSNorm::new(config.hidden_dim),
            lm_head: Linear::new(config.hidden_dim, config.vocab_size),
            config,
        }
    }

    pub(crate) fn default() -> Self {
        Self::new(ModelConfig::default_6_9b())
    }

    /// Create a smaller model for testing
    pub(crate) fn tiny() -> Self {
        Self::new(ModelConfig {
            hidden_dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 1,
            n_experts: 4,
            top_k_experts: 2,
            vocab_size: 1000,
            max_seq_len: 128,
            ffn_dim: 256,
            head_dim: 16,
            rope_base: 10000.0,
            rope_alpha: 1.0,
        })
    }

    pub(crate) fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Forward pass with token IDs
    pub(crate) fn forward_ids(&self, token_ids: &[usize], batch: usize, seq_len: usize) -> Tensor {
        // Embedding
        let mut x = self.embedding.forward_with_ids(token_ids, batch, seq_len);

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Final norm + LM head
        let x = self.final_norm.forward(&x);
        self.lm_head.forward(&x)
    }
}

impl Layer for MoETransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Interpret input as token IDs
        let dims = input.shape().dims();
        let batch = dims[0];
        let seq_len = dims[1];

        let token_ids: Vec<usize> = input.data().iter()
            .map(|&x| x as usize)
            .collect();

        self.forward_ids(&token_ids, batch, seq_len)
    }

    fn backward(&self, grad_output: &Tensor) -> Tensor {
        // Backward through LM head and blocks
        let mut grad = self.lm_head.backward(grad_output);

        for block in self.blocks.iter().rev() {
            grad = block.backward(&grad);
        }

        grad
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.embedding.parameters();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.embedding.parameters_mut();
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.final_norm.parameters_mut());
        params.extend(self.lm_head.parameters_mut());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType};

    #[test]
    fn test_transformer_block() {
        let config = ModelConfig {
            hidden_dim: 32,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 1,
            n_experts: 2,
            top_k_experts: 1,
            vocab_size: 100,
            max_seq_len: 16,
            ffn_dim: 64,
            head_dim: 8,
            rope_base: 10000.0,
            rope_alpha: 1.0,
        };

        let block = TransformerBlock::new(&config);
        let input = Tensor::randn(Shape::new(&[1, 4, 32]), DType::F32, 42);
        let output = block.forward(&input);

        assert_eq!(output.shape().dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_tiny_model_forward() {
        let model = MoETransformer::tiny();

        // Token IDs: batch=1, seq_len=4
        let token_ids = vec![1, 2, 3, 4];
        let logits = model.forward_ids(&token_ids, 1, 4);

        assert_eq!(logits.shape().dims(), &[1, 4, 1000]);

        // Check that logits are not all zeros
        let sum: f32 = logits.data().iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }
}
