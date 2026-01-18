//! MoE Transformer (6.9B/1.8B) Core Implementation
//!
//! Rust型安全性を活用したMoE Transformerエンジン。
//! 全ての内部型は非公開で閉じた設計。

#![forbid(unsafe_code)]
#![allow(dead_code)] // Library crate with pub(crate) items
#![allow(unused_variables)] // Stub implementations have unused params
#![allow(clippy::needless_range_loop)] // Explicit indexing for clarity
#![allow(clippy::manual_memcpy)] // Explicit loops for educational clarity
#![allow(clippy::manual_is_multiple_of)] // Explicit modulo for clarity

mod tensor;
mod layer;
mod attention;
mod moe;
mod model;
mod train;
mod checkpoint;
mod mixed_precision;

// Model Configuration
/// モデル設定
pub(crate) struct ModelConfig {
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_experts: usize,
    top_k_experts: usize,
    vocab_size: usize,
    max_seq_len: usize,
    ffn_dim: usize,
    head_dim: usize,
    rope_base: f32,
    rope_alpha: f32,
}

impl ModelConfig {
    /// 6.9B MoE Transformer デフォルト設定
    fn default_6_9b() -> Self {
        Self {
            hidden_dim: 768,
            n_layers: 30,
            n_heads: 12,
            n_kv_heads: 1, // MQA
            n_experts: 16,
            top_k_experts: 4,
            vocab_size: 32000,
            max_seq_len: 32768,
            ffn_dim: 6144,
            head_dim: 64,
            rope_base: 10000.0,
            rope_alpha: 8.0, // NTK scaling for 256K inference
        }
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, Shape, DType};
    use crate::model::MoETransformer;

    #[test]
    fn test_config() {
        let config = ModelConfig::default_6_9b();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.n_layers, 30);
        assert_eq!(config.n_experts, 16);
        assert_eq!(config.top_k_experts, 4);
    }

    #[test]
    fn test_shape() {
        let shape = Shape::new(&[2, 1024, 768]);
        assert_eq!(shape.numel(), 2 * 1024 * 768);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(Shape::new(&[4, 512, 768]), DType::F32);
        assert_eq!(t.shape().numel(), 4 * 512 * 768);
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn test_model_creation() {
        let model = MoETransformer::default();
        assert_eq!(model.config().hidden_dim, 768);
        assert_eq!(model.num_layers(), 30);
    }

    #[test]
    fn test_training_pipeline() {
        use crate::train::{Trainer, TrainConfig};

        // Use tiny model for testing
        let model = MoETransformer::tiny();
        let train_config = TrainConfig::default();
        let mut trainer = Trainer::new(model, train_config);

        // Create valid token IDs (within vocab_size=1000)
        let batch_size = 2;
        let seq_len = 8;
        let mut token_data = vec![0.0f32; batch_size * seq_len];
        for (i, val) in token_data.iter_mut().enumerate() {
            *val = (i % 100) as f32; // Valid token IDs 0-99
        }
        let input = Tensor::from_slice(&token_data, Shape::new(&[batch_size, seq_len]));
        let targets = Tensor::zeros(Shape::new(&[batch_size, seq_len]), DType::F32);

        let loss = trainer.train_step(&input, &targets);
        assert!(loss >= 0.0);

        // Verify LR schedule
        let lr = trainer.get_lr();
        assert!(lr > 0.0);
        assert!(lr <= 1e-4);
    }

    #[test]
    fn test_model_forward_backward() {
        use crate::layer::Layer;

        // Use tiny model for testing
        let model = MoETransformer::tiny();

        // Forward pass with valid token IDs (within vocab_size=1000)
        let batch = 1;
        let seq_len = 4;
        let token_ids: Vec<usize> = vec![10, 20, 30, 40]; // Valid token IDs
        let logits = model.forward_ids(&token_ids, batch, seq_len);

        // Output should be [batch, seq_len, vocab_size=1000]
        assert_eq!(logits.shape().dims(), &[1, 4, 1000]);

        // Backward pass
        let grad = Tensor::ones(logits.shape().clone(), DType::F32);
        let input_grad = model.backward(&grad);

        // Backward produces grad w.r.t. hidden states
        assert_eq!(input_grad.shape().dims(), &[1, 4, 64]); // hidden_dim=64 for tiny

        // Verify parameters exist
        let params = model.parameters();
        assert!(!params.is_empty());
    }
}
