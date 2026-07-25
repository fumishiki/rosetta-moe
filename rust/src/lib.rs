// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Educational MoE Transformer core with CPU and GPU backends.
//!
//! Public API facade for the `nn-core` crate. All internal modules are private;
//! this file is the sole public boundary. Users interact with the re-exported
//! types only -- implementation details stay hidden.
//!
//! # Module organization
//! - `config`   -- Model hyperparameter configs (tiny / small / medium / 6.9B)
//! - `cpu/`     -- CPU backend: tensor, layers, attention, moe, model, train, generate
//! - `gpu/`     -- GPU backend (feature-gated): Metal tensor, layers, model, train
//!
//! # Safety
//! `#![deny(unsafe_code)]` at the crate root means only submodules with explicit
//! `#![allow(unsafe_code)]` (accelerate, simd, gpu) can contain unsafe blocks.

#![deny(unsafe_code)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_is_multiple_of)]

pub mod config;
pub(crate) mod cpu;

#[cfg(feature = "metal")]
pub(crate) mod gpu;

// ---- Public API re-exports (facade pattern) ----
// All internal types are re-exported flat from the crate root.
// Users write `nn_core::Tensor`, not `nn_core::cpu::tensor::Tensor`.

pub use cpu::attention::MQAttention;
pub use config::Config;
pub use cpu::generate::SamplingStrategy;
pub use cpu::layers::{Embedding, ExpertFFN, Layer, Linear, RMSNorm, SwiGLU};
pub use cpu::model::MoETransformer;
pub use cpu::moe::{MoELayer, Router, TransformerBlock};
pub use cpu::tensor::{DType, Shape, Tensor, TensorError, TensorResult, seed_rng, softmax_in_place};
pub use cpu::train::{
    AdamW, AuxLoss, CheckpointContext, CheckpointStorage, CrossEntropyLoss, LossScaleMode,
    LossScaler, MasterWeights, MixedPrecisionConfig, RoutingMode, TrainConfig, Trainer,
};

// Re-export BLAS wrappers for direct benchmarking (bypasses Tensor overhead).
pub use cpu::accelerate::{sgemm, sgemm_transa, sgemm_transb};

// Metal GPU backend (feature-gated).
#[cfg(feature = "metal")]
pub use gpu::metal_tensor::{MetalContext, MetalTensor, dispatch_kernel, mps_matmul};
#[cfg(feature = "metal")]
pub use gpu::metal_model::{GpuModel, gpu_forward};
#[cfg(feature = "metal")]
pub use gpu::metal_train::{gpu_train_step, gpu_train_step_no_readback};

// Convenience type aliases for cross-language naming consistency.
pub type ModelConfig = Config;
pub type MQAAttention = MQAttention;
pub type SwiGLUFFN = SwiGLU;
