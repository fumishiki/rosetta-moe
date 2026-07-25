// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Metal GPU backend for the MoE Transformer.
//!
//! All GPU operations use Metal Performance Shaders (MPS) for matmul and
//! custom MSL compute kernels for element-wise ops (RMSNorm, SiLU, add, mul).
//! MetalTensor backs data with MTLBuffer (unified memory on Apple Silicon).
//!
//! No CPU Tensor (`Vec<f32>`) is used in the GPU compute path.
//! CPU types are only referenced for weight access during initial upload.

#![allow(unsafe_code)]

pub mod metal_tensor;
pub mod metal_layers;
pub mod metal_model;
pub mod metal_train;
