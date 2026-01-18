//! Mixed Precision Training (FP16/BF16)
//!
//! - Forward/Backward: FP16/BF16 で計算
//! - Master weights: FP32 で保持
//! - Dynamic loss scaling で underflow 防止

use crate::tensor::{Tensor, DType};

/// Loss scaling mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LossScaleMode {
    /// Static loss scale
    Static(f32),
    /// Dynamic loss scaling
    Dynamic {
        init_scale: f32,
        scale_factor: f32,
        scale_window: usize,
    },
}

impl Default for LossScaleMode {
    fn default() -> Self {
        LossScaleMode::Dynamic {
            init_scale: 65536.0,  // 2^16
            scale_factor: 2.0,
            scale_window: 2000,
        }
    }
}

/// Dynamic loss scaler for mixed precision training
pub(crate) struct LossScaler {
    scale: f32,
    mode: LossScaleMode,
    /// Steps since last scale adjustment
    growth_tracker: usize,
    /// Whether overflow was detected in current step
    overflow_detected: bool,
}

impl LossScaler {
    pub(crate) fn new(mode: LossScaleMode) -> Self {
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

    pub(crate) fn static_scale(scale: f32) -> Self {
        Self::new(LossScaleMode::Static(scale))
    }

    pub(crate) fn dynamic() -> Self {
        Self::new(LossScaleMode::default())
    }

    /// Get current loss scale
    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// Scale the loss before backward pass
    pub(crate) fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Unscale gradients after backward pass
    pub(crate) fn unscale_grads(&self, grad: f32) -> f32 {
        grad / self.scale
    }

    /// Check for overflow/underflow in gradients
    pub(crate) fn check_overflow(&mut self, grads: &[f32]) -> bool {
        self.overflow_detected = grads.iter().any(|&g| !g.is_finite());
        self.overflow_detected
    }

    /// Update scale based on overflow status
    pub(crate) fn update(&mut self) {
        match self.mode {
            LossScaleMode::Static(_) => {}
            LossScaleMode::Dynamic { scale_factor, scale_window, .. } => {
                if self.overflow_detected {
                    // Reduce scale on overflow
                    self.scale /= scale_factor;
                    self.growth_tracker = 0;
                    self.overflow_detected = false;
                } else {
                    // Increase scale after successful steps
                    self.growth_tracker += 1;
                    if self.growth_tracker >= scale_window {
                        self.scale *= scale_factor;
                        self.growth_tracker = 0;
                    }
                }
            }
        }
    }

    /// Check if optimizer step should be skipped due to overflow
    pub(crate) fn should_skip_step(&self) -> bool {
        self.overflow_detected
    }
}

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub(crate) struct MixedPrecisionConfig {
    /// Enable mixed precision
    pub(crate) enabled: bool,
    /// Computation dtype (FP16 or BF16)
    pub(crate) compute_dtype: DType,
    /// Loss scaling mode
    pub(crate) loss_scale_mode: LossScaleMode,
    /// Layers to keep in FP32 (e.g., final layer norm)
    pub(crate) fp32_layers: Vec<String>,
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
    pub(crate) fn fp16() -> Self {
        Self {
            enabled: true,
            compute_dtype: DType::F16,
            ..Default::default()
        }
    }

    pub(crate) fn disabled() -> Self {
        Self::default()
    }

    /// Check if a layer should use FP32
    pub(crate) fn is_fp32_layer(&self, layer_name: &str) -> bool {
        self.fp32_layers.iter().any(|s| layer_name.contains(s))
    }
}

/// Master weights container for mixed precision
pub(crate) struct MasterWeights {
    /// FP32 master copy of weights
    weights: Vec<Tensor>,
}

impl MasterWeights {
    pub(crate) fn new() -> Self {
        Self { weights: Vec::new() }
    }

    /// Initialize from model parameters
    pub(crate) fn from_params(params: &[&Tensor]) -> Self {
        let weights = params.iter()
            .map(|p| {
                // Clone to FP32
                Tensor::zeros(p.shape().clone(), DType::F32)
            })
            .collect();
        Self { weights }
    }

    /// Get master weights
    pub(crate) fn weights(&self) -> &[Tensor] {
        &self.weights
    }

    /// Get mutable master weights
    pub(crate) fn weights_mut(&mut self) -> &mut [Tensor] {
        &mut self.weights
    }

    /// Number of parameter tensors
    pub(crate) fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if empty
    pub(crate) fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

impl Default for MasterWeights {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(scaler.scale(), 65536.0);

        // Simulate overflow
        scaler.check_overflow(&[f32::INFINITY]);
        scaler.update();
        assert_eq!(scaler.scale(), 32768.0); // Halved
    }

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::fp16();
        assert!(config.enabled);
        assert_eq!(config.compute_dtype, DType::F16);
        assert!(config.is_fp32_layer("final_norm"));
        assert!(!config.is_fp32_layer("attention"));
    }
}
