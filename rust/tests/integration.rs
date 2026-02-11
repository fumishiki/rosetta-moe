// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

use nn_core::{
    AuxLoss, CheckpointContext, CheckpointStorage, Config, CrossEntropyLoss, DType, Layer, Linear,
    LossScaler, MQAttention, MixedPrecisionConfig, MoELayer, MoETransformer, RMSNorm, Router,
    Shape, SwiGLU, Tensor, TrainConfig, Trainer, TransformerBlock,
};

// --- tensor ---

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(Shape::new(&[4, 512, 768]), DType::F32);
    assert_eq!(t.shape().numel(), 4 * 512 * 768);
    assert_eq!(t.dtype(), DType::F32);
}

// --- layer ---

#[test]
fn test_linear_batch() {
    let mut linear = Linear::new(4, 2);
    let input = Tensor::ones(Shape::new(&[2, 3, 4]), DType::F32);
    let output = linear.forward(&input);
    assert_eq!(output.shape().dims(), &[2, 3, 2]);
}

#[test]
fn test_rmsnorm() {
    let mut norm = RMSNorm::new(4);
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::new(&[1, 4]));
    let y = norm.forward(&x);
    assert_eq!(y.shape().dims(), &[1, 4]);
}

#[test]
fn test_swiglu() {
    let mut ffn = SwiGLU::new(8, 32);
    let x = Tensor::ones(Shape::new(&[1, 2, 8]), DType::F32);
    let y = ffn.forward(&x);
    assert_eq!(y.shape().dims(), &[1, 2, 8]);
}

// --- model ---

#[test]
fn test_config() {
    let config = Config::default_6_9b();
    assert_eq!(config.hidden_dim, 768);
    assert_eq!(config.n_layers, 30);
    assert_eq!(config.n_experts, 16);
}

#[test]
fn test_model_creation() {
    let model = MoETransformer::tiny();
    assert_eq!(model.config().hidden_dim, 64);
    assert_eq!(model.num_layers(), 2);
}

#[test]
fn test_tiny_model_forward() {
    let mut model = MoETransformer::tiny();
    let logits = model.forward_ids(&[1, 2, 3, 4], 1, 4);
    assert_eq!(logits.shape().dims(), &[1, 4, 1000]);
    assert!(logits.data().iter().any(|&x| x != 0.0));
}

#[test]
fn test_generation_interfaces() {
    let mut model = MoETransformer::tiny();
    let prompt = vec![1, 2, 3];

    assert_eq!(model.generate_greedy(&prompt, 8).len(), 8);
    assert_eq!(model.generate_sample(&prompt, 8, 1.0, 42).len(), 8);
    assert_eq!(model.generate_top_k(&prompt, 8, 10, 1.0, 42).len(), 8);
    assert_eq!(model.generate_top_p(&prompt, 8, 0.9, 1.0, 42).len(), 8);
}

#[test]
fn test_transformer_block_shape() {
    let config = Config::tiny();
    let mut block = TransformerBlock::new(&config);
    let x = Tensor::randn(Shape::new(&[1, 4, 64]), DType::F32, 7);
    let y = block.forward(&x);
    assert_eq!(y.shape().dims(), &[1, 4, 64]);
}

#[test]
fn test_model_forward_backward() {
    let mut model = MoETransformer::tiny();

    let token_ids: Vec<usize> = vec![10, 20, 30, 40];
    let logits = model.forward_ids(&token_ids, 1, 4);
    assert_eq!(logits.shape().dims(), &[1, 4, 1000]);

    let grad = Tensor::ones(logits.shape().clone(), DType::F32);
    let input_grad = model.backward(&grad);
    assert_eq!(input_grad.shape().dims(), &[1, 4, 64]);

    let params = model.parameters();
    assert!(!params.is_empty());
}

// --- attention ---

#[test]
fn test_mqa_attention() {
    let config = Config::default_6_9b();
    let mut attn = MQAttention::new(&config);
    let x = Tensor::randn(Shape::new(&[1, 4, 768]), DType::F32, 42);
    let y = attn.forward(&x);
    assert_eq!(y.shape().dims(), &[1, 4, 768]);
}

// --- train ---

#[test]
fn test_cross_entropy_forward_backward() {
    let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], Shape::new(&[1, 2, 3]));
    let targets = Tensor::from_slice(&[2.0, 0.0], Shape::new(&[1, 2]));

    let loss = CrossEntropyLoss::forward(&logits, &targets);
    assert!(loss.item().is_finite() && loss.item() > 0.0);

    let grad = CrossEntropyLoss::backward(&logits, &targets);
    assert_eq!(grad.shape().dims(), &[1, 2, 3]);
}

#[test]
fn test_aux_loss() {
    let router_probs = Tensor::from_slice(&[0.7, 0.2, 0.1, 0.1, 0.8, 0.1], Shape::new(&[2, 3]));
    let expert_indices = Tensor::from_slice(&[0.0, 1.0, 1.0, 2.0], Shape::new(&[2, 2]));
    let loss = AuxLoss::forward(&router_probs, &expert_indices, 3);
    assert!(loss.item().is_finite() && loss.item() >= 0.0);
}

#[test]
fn test_training_pipeline() {
    let model = MoETransformer::tiny();
    let mut trainer = Trainer::new(model, TrainConfig::default());

    let batch_size = 2;
    let seq_len = 8;
    let mut token_data = vec![0.0f32; batch_size * seq_len];
    for (i, v) in token_data.iter_mut().enumerate() {
        *v = (i % 100) as f32;
    }

    let input = Tensor::from_slice(&token_data, Shape::new(&[batch_size, seq_len]));
    let targets = Tensor::zeros(Shape::new(&[batch_size, seq_len]), DType::F32);

    let loss = trainer.train_step(&input, &targets);
    assert!(loss >= 0.0);

    let lr = trainer.get_lr();
    assert!(lr > 0.0 && lr <= 1e-4);
}

// --- checkpoint ---

#[test]
fn test_checkpoint_storage() {
    let mut storage = CheckpointStorage::new(true);
    assert!(storage.is_enabled());
    assert!(storage.is_empty());

    storage.save(0, Tensor::zeros(Shape::new(&[2, 3]), DType::F32));
    assert_eq!(storage.len(), 1);
    assert!(storage.get(0).is_some());

    storage.clear();
    assert!(storage.is_empty());
}

#[test]
fn test_checkpoint_context() {
    let ctx = CheckpointContext::new(2);
    assert!(ctx.should_checkpoint(0));
    assert!(!ctx.should_checkpoint(1));
    assert!(ctx.should_checkpoint(2));
}

// --- mixed_precision ---

#[test]
fn test_loss_scaler_static() {
    let scaler = LossScaler::static_scale(1024.0);
    assert_eq!(scaler.scale(), 1024.0);
    assert_eq!(scaler.scale_loss(1.0), 1024.0);
    assert_eq!(scaler.unscale_grads(1024.0), 1.0);
}

#[test]
fn test_loss_scaler_dynamic() {
    let mut scaler = LossScaler::dynamic();
    scaler.check_overflow(&[f32::INFINITY]);
    scaler.update();
    assert_eq!(scaler.scale(), 32768.0);
}

#[test]
fn test_mixed_precision_config() {
    let config = MixedPrecisionConfig::fp16();
    assert!(config.enabled);
    assert_eq!(config.compute_dtype, DType::F16);
    assert!(config.is_fp32_layer("final_norm"));
    assert!(!config.is_fp32_layer("attention"));
}

// --- moe ---

#[test]
fn test_router() {
    let mut router = Router::new(8, 4);
    let x = Tensor::randn(Shape::new(&[1, 2, 8]), DType::F32, 42);
    let (idx, w, _gate_probs) = router.route(&x, 2);
    assert_eq!(idx.len(), 2);
    assert_eq!(idx[0].len(), 2);
    assert_eq!(w.shape().dims(), &[2, 2]);
}

#[test]
fn test_moe_layer() {
    let config = Config {
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
    let mut moe = MoELayer::new(&config);
    let x = Tensor::randn(Shape::new(&[1, 2, 8]), DType::F32, 42);
    let y = moe.forward(&x);
    assert_eq!(y.shape().dims(), &[1, 2, 8]);
}
