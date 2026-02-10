// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Educational CPU-only MoE Transformer core.
//!
//! Public API facade for the `nn-core` crate. All internal modules are private;
//! this file is the sole public boundary. Users interact with the re-exported
//! types only -- implementation details stay hidden.
//!
//! # Module organization
//! - `tensor`     -- Tensor type, shape, elementwise ops, softmax, matmul
//! - `config`     -- Model hyperparameter configs (tiny / 6.9B)
//! - `layers`     -- Building blocks: Embedding, RMSNorm, Linear, SwiGLU
//! - `attention`  -- Multi-Query Attention with RoPE
//! - `moe`        -- Router, MoE dispatch, TransformerBlock
//! - `model`      -- Full MoETransformer (embed -> blocks -> lm_head)
//! - `generate`   -- Sampling strategies and autoregressive generation
//! - `train`      -- CrossEntropyLoss, AdamW, Trainer, checkpointing
//! - `accelerate` -- Apple Accelerate BLAS FFI (cblas_sgemm)
//!
//! # Safety
//! `#![deny(unsafe_code)]` at the crate root means only the `accelerate` module
//! (which has `#![allow(unsafe_code)]`) can contain unsafe blocks. All unsafe
//! is confined to BLAS FFI calls with documented SAFETY invariants.

#![deny(unsafe_code)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_is_multiple_of)]

mod accelerate;
mod attention;
mod config;
mod generate;
mod layers;
mod model;
mod moe;
mod simd;
mod tensor;
mod train;

// ---- Public API re-exports (facade pattern) ----
// All internal types are re-exported flat from the crate root.
// Users write `nn_core::Tensor`, not `nn_core::tensor::Tensor`.

pub use attention::MQAttention;
pub use config::Config;
pub use generate::SamplingStrategy;
pub use layers::{Embedding, ExpertFFN, Layer, Linear, RMSNorm, SwiGLU};
pub use model::MoETransformer;
pub use moe::{MoELayer, Router, TransformerBlock};
pub use tensor::{DType, Shape, Tensor, TensorError, TensorResult, softmax_in_place};
pub use train::{
    AdamW, AuxLoss, CheckpointContext, CheckpointStorage, CrossEntropyLoss, LossScaleMode,
    LossScaler, MasterWeights, MixedPrecisionConfig, TrainConfig, Trainer,
};

// Re-export BLAS wrappers for direct benchmarking (bypasses Tensor overhead).
pub use accelerate::{sgemm, sgemm_transa, sgemm_transb};

// Convenience type aliases for cross-language naming consistency.
pub type ModelConfig = Config;
pub type MQAAttention = MQAttention;
pub type SwiGLUFFN = SwiGLU;
