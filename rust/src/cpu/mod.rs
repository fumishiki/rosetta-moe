// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! CPU implementation of the MoE Transformer.
//!
//! All computation uses f32 `Vec<f32>` storage with Apple Accelerate BLAS
//! (cblas_sgemm) for matrix multiplication. No GPU/Metal dependencies.

pub(crate) mod accelerate;
pub(crate) mod attention;
pub(crate) mod generate;
pub(crate) mod layers;
pub(crate) mod model;
pub(crate) mod moe;
pub(crate) mod simd;
pub(crate) mod tensor;
pub(crate) mod train;
