// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! GPU training step: forward + cross-entropy loss on GPU.
//!
//! Uses GpuModel for pure GPU forward, then computes loss on GPU.
//! Only the final scalar loss value is read back to CPU.

use super::metal_tensor::{MetalContext, MetalTensor};
use super::metal_model::{GpuModel, gpu_forward};
use super::metal_layers::{gpu_cross_entropy_forward, gpu_reduce_sum};

/// GPU train step without CPU readback.
///
/// Returns a GPU scalar tensor (loss sum). Caller may keep it on device.
pub fn gpu_train_step_no_readback(
    ctx: &MetalContext,
    gpu_model: &GpuModel,
    input: &MetalTensor,
    targets: &MetalTensor,
) -> Result<MetalTensor, String> {
    let batch = input.shape()[0];
    let seq_len = input.shape()[1];
    let batch_seq = batch * seq_len;
    let vocab = gpu_model.config().vocab_size;

    let logits = gpu_forward(ctx, gpu_model, input)?;
    let per_token_loss = gpu_cross_entropy_forward(ctx, &logits, targets, batch_seq, vocab)?;
    gpu_reduce_sum(ctx, &per_token_loss, batch_seq)
}

/// GPU train step: forward + cross-entropy loss, 100% on Metal.
///
/// Returns the scalar loss value (the only CPU readback).
/// input: [batch, seq_len] token IDs as f32
/// targets: [batch, seq_len] target token IDs as f32
pub fn gpu_train_step(
    ctx: &MetalContext,
    gpu_model: &GpuModel,
    input: &MetalTensor,
    targets: &MetalTensor,
) -> Result<f32, String> {
    let batch = input.shape()[0];
    let seq_len = input.shape()[1];
    let batch_seq = batch * seq_len;
    let loss_sum = gpu_train_step_no_readback(ctx, gpu_model, input, targets)?;

    // Read back single scalar (the ONLY CPU readback)
    let loss = loss_sum.read_f32(0) / batch_seq as f32;

    Ok(loss)
}
