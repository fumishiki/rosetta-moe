//! GPU-resident training loop with minimal CPU transfers
//!
//! This module provides a high-level training API that:
//! - Keeps all tensors on GPU during training
//! - Only transfers minimal data to CPU (loss values, metrics)
//! - Supports GPU-side token generation for autoregressive training

use crate::{GpuTensor, Shape, DType, FfiResult};
use nn_cuda::{CudaResult, Stream};

/// Decoding strategy for token generation
#[derive(Debug, Clone, Copy)]
pub enum DecodingStrategy {
    /// Greedy decoding (argmax)
    Greedy,
    /// Multinomial sampling with temperature
    Sample { temperature: f32 },
    /// Top-k sampling
    TopK { k: i32, temperature: f32 },
    /// Nucleus (top-p) sampling
    TopP { top_p: f32, temperature: f32 },
}

impl Default for DecodingStrategy {
    fn default() -> Self {
        DecodingStrategy::Greedy
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Learning rate
    pub lr: f32,
    /// AdamW beta1
    pub beta1: f32,
    /// AdamW beta2
    pub beta2: f32,
    /// AdamW epsilon
    pub eps: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Gradient clipping norm (0 = disabled)
    pub grad_clip_norm: f32,
    /// Decoding strategy for autoregressive generation
    pub decoding: DecodingStrategy,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            grad_clip_norm: 1.0,
            decoding: DecodingStrategy::Greedy,
        }
    }
}

/// GPU-resident trainer state
pub struct GpuTrainer {
    config: TrainerConfig,
    step: i32,
    /// Optimizer momentum state (one per parameter)
    m_states: Vec<GpuTensor>,
    /// Optimizer velocity state (one per parameter)
    v_states: Vec<GpuTensor>,
    /// Random seeds for sampling (one per batch item)
    rng_seeds: Option<GpuTensor>,
    /// Scratch buffer for partial norms during gradient clipping
    partial_norms: Option<GpuTensor>,
    /// Scratch buffer for total norm
    total_norm: Option<GpuTensor>,
}

impl GpuTrainer {
    /// Create a new trainer with the given configuration
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            step: 0,
            m_states: Vec::new(),
            v_states: Vec::new(),
            rng_seeds: None,
            partial_norms: None,
            total_norm: None,
        }
    }

    /// Initialize optimizer states for parameters
    ///
    /// Call this once before training with the total number of parameters
    pub fn init_optimizer_states(&mut self, param_sizes: &[usize]) -> FfiResult<()> {
        self.m_states.clear();
        self.v_states.clear();

        for &size in param_sizes {
            self.m_states.push(GpuTensor::zeros(Shape::new(&[size]), DType::F32)?);
            self.v_states.push(GpuTensor::zeros(Shape::new(&[size]), DType::F32)?);
        }

        Ok(())
    }

    /// Initialize RNG seeds for sampling-based decoding
    pub fn init_rng(&mut self, batch_size: usize, initial_seed: u64) -> FfiResult<()> {
        // Create seeds array: each batch item gets a different seed
        let seeds: Vec<u64> = (0..batch_size)
            .map(|i| initial_seed.wrapping_add(i as u64))
            .collect();

        // Upload to GPU (as f32 for now, will be cast when used)
        // Note: This is a workaround since GpuTensor only supports f32
        // In a real implementation, we'd have a proper u64 tensor type
        let seeds_f32: Vec<f32> = seeds.iter().map(|&s| s as f32).collect();
        self.rng_seeds = Some(GpuTensor::from_slice(&seeds_f32, Shape::new(&[batch_size]))?);

        Ok(())
    }

    /// Get current training step
    pub fn step(&self) -> i32 {
        self.step
    }

    /// Run a single optimizer step on a parameter with its gradient
    ///
    /// Returns Ok if successful, Err if optimizer states not initialized
    pub fn optimizer_step(
        &mut self,
        param_idx: usize,
        param: &mut GpuTensor,
        grad: &GpuTensor,
        stream: Stream,
    ) -> CudaResult<()> {
        if param_idx >= self.m_states.len() {
            return Err(nn_cuda::CudaError::NOT_AVAILABLE);
        }

        crate::adamw_step(
            param,
            grad,
            &mut self.m_states[param_idx],
            &mut self.v_states[param_idx],
            self.config.lr,
            self.config.beta1,
            self.config.beta2,
            self.config.eps,
            self.config.weight_decay,
            self.step + 1, // AdamW uses 1-indexed step for bias correction
            stream,
        )
    }

    /// Increment the training step counter
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Decode tokens from logits using the configured strategy
    ///
    /// All operations happen on GPU - no CPU transfer needed
    pub fn decode(
        &self,
        logits: &GpuTensor,
        output: &mut GpuTensor,
        stream: Stream,
    ) -> CudaResult<()> {
        match self.config.decoding {
            DecodingStrategy::Greedy => {
                crate::argmax(logits, output, stream)
            }
            DecodingStrategy::Sample { temperature } => {
                if let Some(ref seeds) = self.rng_seeds {
                    crate::sample(logits, output, seeds, temperature, stream)
                } else {
                    // Fall back to greedy if no RNG initialized
                    crate::argmax(logits, output, stream)
                }
            }
            DecodingStrategy::TopK { k, temperature } => {
                if let Some(ref seeds) = self.rng_seeds {
                    crate::topk_sample(logits, output, seeds, k, temperature, stream)
                } else {
                    crate::argmax(logits, output, stream)
                }
            }
            DecodingStrategy::TopP { top_p, temperature } => {
                if let Some(ref seeds) = self.rng_seeds {
                    crate::topp_sample(logits, output, seeds, top_p, temperature, stream)
                } else {
                    crate::argmax(logits, output, stream)
                }
            }
        }
    }

    /// Get loss value from GPU (single f32 transfer)
    ///
    /// This is one of the few operations that transfers data to CPU
    pub fn get_loss(&self, loss_tensor: &GpuTensor) -> FfiResult<f32> {
        let loss_vec = loss_tensor.to_vec()?;
        Ok(loss_vec.get(0).copied().unwrap_or(0.0))
    }

    /// Update decoding strategy
    pub fn set_decoding_strategy(&mut self, strategy: DecodingStrategy) {
        self.config.decoding = strategy;
    }

    /// Update learning rate (for scheduling)
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

/// Training metrics collected during a step
#[derive(Debug, Clone, Default)]
pub struct StepMetrics {
    /// Cross-entropy loss
    pub loss: f32,
    /// Auxiliary (load balancing) loss for MoE
    pub aux_loss: Option<f32>,
    /// Gradient norm before clipping
    pub grad_norm: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert_eq!(config.lr, 1e-4);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.95);
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.1);
        assert_eq!(config.grad_clip_norm, 1.0);
    }

    #[test]
    fn test_decoding_strategy_default() {
        let strategy = DecodingStrategy::default();
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_decoding_strategy_variants() {
        let greedy = DecodingStrategy::Greedy;
        let sample = DecodingStrategy::Sample { temperature: 0.7 };
        let topk = DecodingStrategy::TopK { k: 50, temperature: 0.8 };
        let topp = DecodingStrategy::TopP { top_p: 0.9, temperature: 0.7 };

        assert!(matches!(greedy, DecodingStrategy::Greedy));
        assert!(matches!(sample, DecodingStrategy::Sample { temperature: 0.7 }));
        assert!(matches!(topk, DecodingStrategy::TopK { k: 50, .. }));
        assert!(matches!(topp, DecodingStrategy::TopP { top_p: 0.9, .. }));
    }

    #[test]
    fn test_trainer_step_counter() {
        let mut trainer = GpuTrainer::new(TrainerConfig::default());
        assert_eq!(trainer.step(), 0);
        trainer.increment_step();
        assert_eq!(trainer.step(), 1);
        trainer.increment_step();
        assert_eq!(trainer.step(), 2);
    }

    #[test]
    fn test_trainer_lr_update() {
        let mut trainer = GpuTrainer::new(TrainerConfig::default());
        assert_eq!(trainer.config.lr, 1e-4);

        trainer.set_learning_rate(5e-5);
        assert_eq!(trainer.config.lr, 5e-5);
    }

    #[test]
    fn test_trainer_decoding_strategy_update() {
        let mut trainer = GpuTrainer::new(TrainerConfig::default());
        assert!(matches!(trainer.config.decoding, DecodingStrategy::Greedy));

        trainer.set_decoding_strategy(DecodingStrategy::TopK { k: 40, temperature: 0.9 });
        assert!(matches!(trainer.config.decoding, DecodingStrategy::TopK { k: 40, .. }));
    }

    #[test]
    fn test_step_metrics() {
        let metrics = StepMetrics {
            loss: 2.5,
            aux_loss: Some(0.01),
            grad_norm: Some(1.2),
        };

        assert_eq!(metrics.loss, 2.5);
        assert_eq!(metrics.aux_loss, Some(0.01));
        assert_eq!(metrics.grad_norm, Some(1.2));
    }

    #[test]
    fn test_step_metrics_default() {
        let metrics = StepMetrics::default();
        assert_eq!(metrics.loss, 0.0);
        assert_eq!(metrics.aux_loss, None);
        assert_eq!(metrics.grad_norm, None);
    }

    #[test]
    fn test_trainer_init_optimizer_states() {
        let mut trainer = GpuTrainer::new(TrainerConfig::default());

        // Note: This will fail because GPU is not available (stub),
        // but it tests the API structure
        let param_sizes = vec![1024, 2048, 512];
        let result = trainer.init_optimizer_states(&param_sizes);

        // With stubs, allocation will fail
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_full_config() {
        let config = TrainerConfig {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.05,
            grad_clip_norm: 0.5,
            decoding: DecodingStrategy::TopP { top_p: 0.95, temperature: 0.8 },
        };

        let trainer = GpuTrainer::new(config);
        assert_eq!(trainer.config.lr, 3e-4);
        assert_eq!(trainer.config.weight_decay, 0.05);
        assert!(matches!(
            trainer.config.decoding,
            DecodingStrategy::TopP { top_p: 0.95, temperature: 0.8 }
        ));
    }
}
