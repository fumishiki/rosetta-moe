// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Training: losses, optimizer, trainer loop, checkpointing, mixed precision.
//!
//! Key components:
//! - **CrossEntropyLoss**: forward (loss value) and backward (gradient w.r.t. logits)
//! - **AdamW**: optimizer with decoupled weight decay
//! - **Trainer**: orchestrates forward, backward, gradient clipping, optimizer step
//! - **Checkpoint / LossScaler / MixedPrecision**: training infrastructure stubs

use std::collections::HashMap;

use crate::layers::Layer;
use crate::model::MoETransformer;
use crate::tensor::{DType, Tensor};

/// Cross-entropy loss for language modeling.
///
/// Forward:  L = -(1/N) * sum_i(log(softmax(logits_i)[target_i]))
/// Backward: dL/d(logits_i) = (1/N) * (softmax(logits_i) - one_hot(target_i))
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Cross-entropy: L = -(1/N) * sum_i(log(softmax(logits_i)[target_i]))
    ///
    /// Uses softmax_into_slice with a reusable probs buffer (one per vocab-row)
    /// to avoid N separate allocations. The log is clamped with max(p, 1e-12) to
    /// prevent log(0) = -inf when softmax rounds a probability to exactly 0.
    pub fn forward(logits: &Tensor, targets: &Tensor) -> Tensor {
        let (n, vocab) = logits_layout(logits, targets);
        let logits_data = logits.data();
        let targets_data = targets.data();

        // Reusable softmax buffer -- avoids allocating per token
        let mut probs = vec![0.0; vocab];
        let mut total = 0.0;

        for i in 0..n {
            let start = i * vocab;
            crate::tensor::softmax_into_slice(&logits_data[start..start + vocab], &mut probs);
            let target = checked_target(targets_data[i], vocab);
            // Clamp to 1e-12 to avoid log(0) = -inf
            total -= probs[target].max(1e-12).ln();
        }

        Tensor::scalar(total / n as f32)
    }

    /// Cross-entropy gradient: dL/d(logits_i) = (1/N) * (softmax(logits_i) - one_hot(target_i))
    ///
    /// For each position: grad = softmax(logits), then subtract 1 at the target index.
    /// This elegant formula comes from d/dx[-log(softmax(x)_c)] = softmax(x) - e_c.
    pub fn backward(logits: &Tensor, targets: &Tensor) -> Tensor {
        let (n, vocab) = logits_layout(logits, targets);
        let logits_data = logits.data();
        let targets_data = targets.data();

        let mut grad = vec![0.0; logits.numel()];
        let mut probs = vec![0.0; vocab];

        for i in 0..n {
            let start = i * vocab;
            let grad_row = &mut grad[start..start + vocab];
            crate::tensor::softmax_into_slice(&logits_data[start..start + vocab], &mut probs);
            // grad = softmax(logits)
            grad_row.copy_from_slice(&probs);
            // grad[target] -= 1 (the one-hot subtraction)
            let target = checked_target(targets_data[i], vocab);
            grad_row[target] -= 1.0;
        }

        // Average over N tokens
        let inv_n = 1.0 / n as f32;
        for g in grad.iter_mut() {
            *g *= inv_n;
        }

        Tensor::from_vec(grad, logits.shape().clone())
    }
}

/// Standalone auxiliary load-balancing loss (tensor-based interface).
///
/// L_aux = N * sum_e(f_e * p_e)
/// Same core formula as `MoELayer::aux_loss` but without the alpha multiplier,
/// and takes tensors instead of cached data. The caller is responsible for
/// scaling by alpha if needed.
pub struct AuxLoss;

impl AuxLoss {
    pub fn forward(
        router_probs: &Tensor,
        expert_indices: &Tensor,
        n_experts: usize,
    ) -> Tensor {
        let probs_dims = router_probs.shape().dims();
        assert_eq!(
            probs_dims.len(),
            2,
            "router_probs must be [batch_seq, n_experts]"
        );
        assert_eq!(
            probs_dims[1], n_experts,
            "router_probs second dim must equal n_experts"
        );

        let idx_dims = expert_indices.shape().dims();
        assert_eq!(
            idx_dims.len(),
            2,
            "expert_indices must be [batch_seq, top_k]"
        );
        assert_eq!(idx_dims[0], probs_dims[0], "batch_seq mismatch");

        let batch_seq = probs_dims[0];
        let top_k = idx_dims[1];
        let probs = router_probs.data();
        let idx = expert_indices.data();

        let mut f_counts = vec![0.0f32; n_experts];
        let mut p_means = vec![0.0f32; n_experts];

        for t in 0..batch_seq {
            for e in 0..n_experts {
                p_means[e] += probs[t * n_experts + e];
            }
            for k in 0..top_k {
                let e_idx = idx[t * top_k + k] as usize;
                if e_idx < n_experts {
                    f_counts[e_idx] += 1.0;
                }
            }
        }

        let denom_assign = (batch_seq * top_k) as f32;
        let denom_prob = batch_seq as f32;
        let loss: f32 = (0..n_experts)
            .map(|e| (f_counts[e] / denom_assign) * (p_means[e] / denom_prob))
            .sum();

        Tensor::scalar(loss * n_experts as f32)
    }
}

/// AdamW optimizer with decoupled weight decay (Loshchilov & Hutter 2017).
///
/// Update rule per parameter w:
///   m = beta1 * m + (1 - beta1) * g           (first moment)
///   v = beta2 * v + (1 - beta2) * g^2          (second moment)
///   m_hat = m / (1 - beta1^t)                  (bias correction)
///   v_hat = v / (1 - beta2^t)                  (bias correction)
///   w -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
///
/// The weight decay term is *decoupled* from the adaptive learning rate
/// (applied directly to w, not through the Adam update), which is the key
/// difference from L2 regularization in standard Adam.
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    /// First moment estimates (one per parameter tensor)
    m: Vec<Tensor>,
    /// Second moment estimates (one per parameter tensor)
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(params: &[&Tensor], lr: f32) -> Self {
        let m = params
            .iter()
            .map(|p| Tensor::zeros(p.shape().clone(), p.dtype()))
            .collect();
        let v = params
            .iter()
            .map(|p| Tensor::zeros(p.shape().clone(), p.dtype()))
            .collect();
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            step_count: 0,
            m,
            v,
        }
    }

    /// One optimizer step using SIMD-accelerated approximate rsqrt.
    ///
    /// Bias correction factors: corr = 1/(1 - beta^t), clamped to avoid
    /// division by zero when beta^t rounds to exactly 1.0 at early steps.
    ///
    /// Uses `Tensor::adamw_update()` which temporarily takes the grad out of
    /// the tensor to split borrows safely -- zero-copy, no grad cloning.
    ///
    /// Uses NEON `vrsqrteq_f32` + Newton-Raphson for approximate rsqrt,
    /// replacing the expensive scalar `sqrt()` call. This matches Julia's
    /// `@fastmath` strategy that uses ARM NEON `frsqrte`.
    pub fn step(&mut self, params: &mut [&mut Tensor]) {
        self.step_count += 1;
        // Bias correction: 1/(1 - beta^t)
        let corr1 = 1.0 / (1.0 - self.beta1.powi(self.step_count as i32)).max(1e-12);
        let corr2 = 1.0 / (1.0 - self.beta2.powi(self.step_count as i32)).max(1e-12);

        for (i, param) in params.iter_mut().enumerate() {
            param.adamw_update(
                self.m[i].data_mut(),
                self.v[i].data_mut(),
                self.lr, self.beta1, self.beta2, self.eps,
                self.weight_decay, corr1, corr2,
            );
        }
    }

    pub fn zero_grad(&mut self, params: &mut [&mut Tensor]) {
        for p in params {
            p.clear_grad();
        }
    }
}

pub struct TrainConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub grad_clip: f32,
    pub aux_loss_weight: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            seq_len: 2048,
            lr: 1e-4,
            warmup_steps: 1000,
            total_steps: 100_000,
            grad_clip: 1.0,
            aux_loss_weight: 0.01,
        }
    }
}

pub struct Trainer {
    model: MoETransformer,
    optimizer: AdamW,
    config: TrainConfig,
    current_step: usize,
}

impl Trainer {
    pub fn new(model: MoETransformer, config: TrainConfig) -> Self {
        let params = model.parameters();
        let optimizer = AdamW::new(&params, config.lr);
        Self {
            model,
            optimizer,
            config,
            current_step: 0,
        }
    }

    pub fn train_step(&mut self, input: &Tensor, targets: &Tensor) -> f32 {
        // Zero all parameter gradients before forward/backward
        {
            let mut params = self.model.parameters_mut();
            self.optimizer.zero_grad(&mut params);
        }

        let logits = self.model.forward(input);
        let ce_loss = CrossEntropyLoss::forward(&logits, targets).item();

        let grad = CrossEntropyLoss::backward(&logits, targets);
        let _model_grad = self.model.backward(&grad);

        let aux_loss = self.model.total_aux_loss(self.config.aux_loss_weight);
        let total_loss = ce_loss + aux_loss;

        // Global gradient norm clipping + optimizer step (single parameters_mut call)
        let mut params = self.model.parameters_mut();
        let clip_norm = self.config.grad_clip;
        if clip_norm > 0.0 {
            let mut total_norm_sq = 0.0f32;
            for p in params.iter() {
                if let Some(g) = p.grad() {
                    total_norm_sq += g.data().iter().map(|v| v * v).sum::<f32>();
                }
            }
            let global_norm = total_norm_sq.sqrt();
            if global_norm > clip_norm {
                let clip_coeff = clip_norm / (global_norm + 1e-12);
                for p in params.iter_mut() {
                    if let Some(g) = p.grad_mut() {
                        for v in g.data_mut().iter_mut() {
                            *v *= clip_coeff;
                        }
                    }
                }
            }
        }
        self.optimizer.step(&mut params);

        self.current_step += 1;
        total_loss
    }

    /// Learning rate schedule: linear warmup + cosine decay.
    ///
    /// Warmup (step < warmup_steps):
    ///   lr = base_lr * step / warmup_steps
    ///
    /// Cosine decay (step >= warmup_steps):
    ///   progress = (step - warmup) / (total - warmup)
    ///   lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    ///   where min_lr = 0.1 * base_lr
    pub fn get_lr(&self) -> f32 {
        if self.current_step < self.config.warmup_steps {
            // Linear warmup
            return self.config.lr * (self.current_step as f32) / (self.config.warmup_steps as f32);
        }

        // Cosine decay to 10% of base LR
        let progress = (self.current_step - self.config.warmup_steps) as f32
            / (self.config.total_steps - self.config.warmup_steps) as f32;
        let min_lr = self.config.lr * 0.1;
        min_lr + 0.5 * (self.config.lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// Activation checkpointing storage for memory-efficient backward pass.
/// Stores intermediate activations at checkpoint boundaries so they can be
/// recomputed during backward rather than kept in memory for all layers.
pub struct CheckpointStorage {
    checkpoints: HashMap<usize, Tensor>,
    enabled: bool,
}

impl CheckpointStorage {
    pub fn new(enabled: bool) -> Self {
        Self {
            checkpoints: HashMap::new(),
            enabled,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn save(&mut self, block_idx: usize, input: Tensor) {
        if self.enabled {
            self.checkpoints.insert(block_idx, input);
        }
    }

    pub fn get(&self, block_idx: usize) -> Option<&Tensor> {
        self.checkpoints.get(&block_idx)
    }

    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }

    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

/// Manages activation checkpointing with configurable segment granularity.
/// `segment_size` controls how often to checkpoint (every N transformer blocks).
/// Larger segments = less memory but more recomputation.
pub struct CheckpointContext {
    storage: CheckpointStorage,
    segment_size: usize,
}

impl CheckpointContext {
    pub fn new(segment_size: usize) -> Self {
        let size = segment_size.max(1);
        Self {
            storage: CheckpointStorage::new(segment_size > 0),
            segment_size: size,
        }
    }

    pub fn disabled() -> Self {
        Self {
            storage: CheckpointStorage::new(false),
            segment_size: 1,
        }
    }

    pub fn should_checkpoint(&self, block_idx: usize) -> bool {
        self.storage.is_enabled() && (block_idx % self.segment_size == 0)
    }

    pub fn maybe_save(&mut self, block_idx: usize, input: Tensor) {
        if self.should_checkpoint(block_idx) {
            self.storage.save(block_idx, input);
        }
    }

    pub fn get_checkpoint(&self, block_idx: usize) -> Option<&Tensor> {
        let checkpoint_idx = (block_idx / self.segment_size) * self.segment_size;
        self.storage.get(checkpoint_idx)
    }

    pub fn clear(&mut self) {
        self.storage.clear();
    }

    pub fn storage(&self) -> &CheckpointStorage {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut CheckpointStorage {
        &mut self.storage
    }
}

impl Default for CheckpointContext {
    fn default() -> Self {
        Self::new(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossScaleMode {
    Static(f32),
    Dynamic {
        init_scale: f32,
        scale_factor: f32,
        scale_window: usize,
    },
}

impl Default for LossScaleMode {
    fn default() -> Self {
        Self::Dynamic {
            init_scale: 65536.0,
            scale_factor: 2.0,
            scale_window: 2000,
        }
    }
}

/// Dynamic/static loss scaling for mixed-precision training.
///
/// Dynamic mode: doubles scale every `scale_window` steps without overflow;
/// halves it immediately when overflow (NaN/Inf) is detected.
/// This "try to grow, shrink on failure" approach finds the largest stable scale.
pub struct LossScaler {
    scale: f32,
    mode: LossScaleMode,
    growth_tracker: usize,
    overflow_detected: bool,
}

impl LossScaler {
    pub fn new(mode: LossScaleMode) -> Self {
        let scale = match mode {
            LossScaleMode::Static(s) => s,
            LossScaleMode::Dynamic { init_scale, .. } => init_scale,
        };
        Self {
            scale,
            mode,
            growth_tracker: 0,
            overflow_detected: false,
        }
    }

    pub fn static_scale(scale: f32) -> Self {
        Self::new(LossScaleMode::Static(scale))
    }

    pub fn dynamic() -> Self {
        Self::new(LossScaleMode::default())
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    pub fn unscale_grads(&self, grad: f32) -> f32 {
        grad / self.scale
    }

    pub fn check_overflow(&mut self, grads: &[f32]) -> bool {
        self.overflow_detected = grads.iter().any(|&g| !g.is_finite());
        self.overflow_detected
    }

    pub fn update(&mut self) {
        let LossScaleMode::Dynamic {
            scale_factor,
            scale_window,
            ..
        } = self.mode
        else {
            return;
        };

        if self.overflow_detected {
            self.scale /= scale_factor;
            self.growth_tracker = 0;
            self.overflow_detected = false;
            return;
        }

        self.growth_tracker += 1;
        if self.growth_tracker >= scale_window {
            self.scale *= scale_factor;
            self.growth_tracker = 0;
        }
    }

    pub fn should_skip_step(&self) -> bool {
        self.overflow_detected
    }
}

#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    pub enabled: bool,
    pub compute_dtype: DType,
    pub loss_scale_mode: LossScaleMode,
    pub fp32_layers: Vec<String>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            compute_dtype: DType::F16,
            loss_scale_mode: LossScaleMode::default(),
            fp32_layers: vec!["final_norm".to_string(), "lm_head".to_string()],
        }
    }
}

impl MixedPrecisionConfig {
    pub fn fp16() -> Self {
        Self {
            enabled: true,
            compute_dtype: DType::F16,
            ..Self::default()
        }
    }

    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn is_fp32_layer(&self, layer_name: &str) -> bool {
        self.fp32_layers.iter().any(|s| layer_name.contains(s))
    }
}

#[derive(Default)]
pub struct MasterWeights {
    weights: Vec<Tensor>,
}

impl MasterWeights {
    pub fn from_params(params: &[&Tensor]) -> Self {
        let weights = params
            .iter()
            .map(|p| Tensor::zeros(p.shape().clone(), DType::F32))
            .collect();
        Self { weights }
    }

    pub fn weights(&self) -> &[Tensor] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [Tensor] {
        &mut self.weights
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Gradient clipping by global L2 norm (in-place, consuming the input).
///
/// If ||grad||_2 > clip_norm:
///   grad_clipped = grad * (clip_norm / ||grad||_2)
///
/// Consumes the input tensor and scales in-place to avoid allocation.
/// Prevents exploding gradients. The epsilon (1e-12) guards against
/// division by zero when the gradient is exactly zero.
fn clip_grad_by_global_norm_inplace(mut grad: Tensor, clip_norm: f32) -> Tensor {
    if clip_norm <= 0.0 {
        return grad;
    }
    let norm = grad.data().iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > clip_norm {
        let scale = clip_norm / (norm + 1e-12);
        for v in grad.data_mut().iter_mut() {
            *v *= scale;
        }
    }
    grad
}

fn logits_layout(logits: &Tensor, targets: &Tensor) -> (usize, usize) {
    let dims = logits.shape().dims();
    assert_eq!(dims.len(), 3, "logits must be [batch, seq_len, vocab]");
    let n = dims[0] * dims[1];
    assert_eq!(targets.numel(), n, "targets must contain batch*seq indices");
    (n, dims[2])
}

fn checked_target(target: f32, vocab: usize) -> usize {
    let idx = target as usize;
    assert!(idx < vocab, "target index out of range");
    idx
}

