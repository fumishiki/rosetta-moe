// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! GPU layer operations: Linear, RMSNorm, SiLU, Softmax, elementwise ops.
//!
//! All ops take MetalTensor inputs and produce MetalTensor outputs.
//! NO CPU Tensor types are referenced. Weights are MetalTensor (pre-uploaded).

use super::metal_tensor::{MetalContext, MetalTensor, dispatch_kernel, mps_matmul};

/// GPU linear layer: Y = X @ W_T using MPS matmul.
/// weight_t: [in_features, out_features] — already transposed on GPU.
pub fn gpu_linear(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight_t: &MetalTensor,
    in_features: usize,
    out_features: usize,
) -> Result<MetalTensor, String> {
    let batch_seq = input.len() / in_features;
    let mut out = MetalTensor::zeros(ctx, vec![batch_seq, out_features])?;
    mps_matmul(ctx, input, weight_t, &mut out, batch_seq, out_features, in_features)?;
    Ok(out)
}

/// GPU RMSNorm layer: dispatch rmsnorm MSL kernel.
pub fn gpu_rmsnorm(
    ctx: &MetalContext,
    input: &MetalTensor,
    weight: &MetalTensor,
    n_rows: usize,
    hidden_dim: usize,
    eps: f32,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![n_rows, hidden_dim])?;
    let dim_buf = MetalTensor::from_u32_scalar(ctx, hidden_dim as u32)?;
    let eps_buf = MetalTensor::from_f32_scalar(ctx, eps)?;

    let threadgroup_size = 256;
    dispatch_kernel(
        ctx,
        "rmsnorm",
        &[input, &out, weight, &dim_buf, &eps_buf],
        &[],
        n_rows * threadgroup_size,
        threadgroup_size,
    )?;
    Ok(out)
}

/// GPU SiLU activation: dispatch silu MSL kernel.
pub fn gpu_silu(ctx: &MetalContext, input: &MetalTensor) -> Result<MetalTensor, String> {
    let n = input.len();
    let out = MetalTensor::zeros(ctx, input.shape().to_owned())?;
    dispatch_kernel(ctx, "silu", &[input, &out], &[], n, 256)?;
    Ok(out)
}

/// GPU elementwise add: dispatch add MSL kernel.
pub fn gpu_add(ctx: &MetalContext, a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor, String> {
    let n = a.len();
    let out = MetalTensor::zeros(ctx, a.shape().to_owned())?;
    dispatch_kernel(ctx, "add", &[a, b, &out], &[], n, 256)?;
    Ok(out)
}

/// GPU elementwise multiply: dispatch mul MSL kernel.
pub fn gpu_mul(ctx: &MetalContext, a: &MetalTensor, b: &MetalTensor) -> Result<MetalTensor, String> {
    let n = a.len();
    let out = MetalTensor::zeros(ctx, a.shape().to_owned())?;
    dispatch_kernel(ctx, "mul", &[a, b, &out], &[], n, 256)?;
    Ok(out)
}

/// GPU softmax: per-row softmax using threadgroup reduction.
/// input: [n_rows, row_len], output: [n_rows, row_len]
pub fn gpu_softmax(
    ctx: &MetalContext,
    input: &MetalTensor,
    n_rows: usize,
    row_len: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![n_rows, row_len])?;
    let n_buf = MetalTensor::from_u32_scalar(ctx, row_len as u32)?;
    let threadgroup_size = 256;
    dispatch_kernel(
        ctx,
        "softmax",
        &[input, &out, &n_buf],
        &[],
        n_rows * threadgroup_size,
        threadgroup_size,
    )?;
    Ok(out)
}

/// GPU scale: out = input * scalar
pub fn gpu_scale(
    ctx: &MetalContext,
    input: &MetalTensor,
    scalar: f32,
) -> Result<MetalTensor, String> {
    let n = input.len();
    let out = MetalTensor::zeros(ctx, input.shape().to_owned())?;
    let scalar_buf = MetalTensor::from_f32_scalar(ctx, scalar)?;
    dispatch_kernel(ctx, "scale", &[input, &out, &scalar_buf], &[], n, 256)?;
    Ok(out)
}

/// GPU embedding gather: out[i] = weight[token_ids[i]]
pub fn gpu_embedding(
    ctx: &MetalContext,
    token_ids: &MetalTensor,
    weight: &MetalTensor,
    batch_seq: usize,
    hidden_dim: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![batch_seq, hidden_dim])?;
    let dim_buf = MetalTensor::from_u32_scalar(ctx, hidden_dim as u32)?;
    let n = batch_seq * hidden_dim;
    dispatch_kernel(
        ctx,
        "embedding_gather",
        &[token_ids, weight, &out, &dim_buf],
        &[],
        n,
        256,
    )?;
    Ok(out)
}

/// GPU RoPE in-place rotation.
pub fn gpu_rope_inplace(
    ctx: &MetalContext,
    x: &MetalTensor,
    cos_cache: &MetalTensor,
    sin_cache: &MetalTensor,
    batch_seq: usize,
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<(), String> {
    let half_dim = head_dim / 2;
    let total_pairs = batch_seq * n_heads * half_dim;
    let n_heads_buf = MetalTensor::from_u32_scalar(ctx, n_heads as u32)?;
    let head_dim_buf = MetalTensor::from_u32_scalar(ctx, head_dim as u32)?;
    let seq_len_buf = MetalTensor::from_u32_scalar(ctx, seq_len as u32)?;
    dispatch_kernel(
        ctx,
        "rope_inplace",
        &[x, cos_cache, sin_cache, &n_heads_buf, &head_dim_buf, &seq_len_buf],
        &[],
        total_pairs,
        256,
    )?;
    Ok(())
}

/// GPU causal mask fill: scores[qi, ki] = -inf for ki > qi.
pub fn gpu_causal_mask(
    ctx: &MetalContext,
    scores: &MetalTensor,
    batch_heads: usize,
    seq_len: usize,
) -> Result<(), String> {
    let total = batch_heads * seq_len * seq_len;
    let seq_len_buf = MetalTensor::from_u32_scalar(ctx, seq_len as u32)?;
    dispatch_kernel(
        ctx,
        "causal_mask_fill",
        &[scores, &seq_len_buf],
        &[],
        total,
        256,
    )?;
    Ok(())
}

/// GPU attention score computation:
/// scores[bh, qi, ki] = scale * dot(q[b,qi,h,:], k[b,ki,kv_h,:]).
pub fn gpu_attention_scores(
    ctx: &MetalContext,
    q: &MetalTensor,
    k: &MetalTensor,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![batch * n_heads, seq_len, seq_len])?;
    let n_heads_buf = MetalTensor::from_u32_scalar(ctx, n_heads as u32)?;
    let n_kv_heads_buf = MetalTensor::from_u32_scalar(ctx, n_kv_heads as u32)?;
    let head_dim_buf = MetalTensor::from_u32_scalar(ctx, head_dim as u32)?;
    let seq_len_buf = MetalTensor::from_u32_scalar(ctx, seq_len as u32)?;
    let scale_buf = MetalTensor::from_f32_scalar(ctx, scale)?;
    let total = batch * n_heads * seq_len * seq_len;
    dispatch_kernel(
        ctx,
        "attention_scores",
        &[
            q,
            k,
            &out,
            &n_heads_buf,
            &n_kv_heads_buf,
            &head_dim_buf,
            &seq_len_buf,
            &scale_buf,
        ],
        &[],
        total,
        256,
    )?;
    Ok(out)
}

/// GPU attention value aggregation:
/// out[b, qi, h, d] = sum_ki weights[b,h,qi,ki] * v[b,ki,kv_h,d].
pub fn gpu_attention_weighted_sum(
    ctx: &MetalContext,
    attn_weights: &MetalTensor,
    v: &MetalTensor,
    batch_seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![batch_seq, n_heads * head_dim])?;
    let n_heads_buf = MetalTensor::from_u32_scalar(ctx, n_heads as u32)?;
    let n_kv_heads_buf = MetalTensor::from_u32_scalar(ctx, n_kv_heads as u32)?;
    let head_dim_buf = MetalTensor::from_u32_scalar(ctx, head_dim as u32)?;
    let seq_len_buf = MetalTensor::from_u32_scalar(ctx, seq_len as u32)?;
    let total = batch_seq * n_heads * head_dim;
    dispatch_kernel(
        ctx,
        "attention_weighted_sum",
        &[
            attn_weights,
            v,
            &out,
            &n_heads_buf,
            &n_kv_heads_buf,
            &head_dim_buf,
            &seq_len_buf,
        ],
        &[],
        total,
        256,
    )?;
    Ok(out)
}

/// GPU transpose: [rows, cols] -> [cols, rows]
pub fn gpu_transpose(
    ctx: &MetalContext,
    input: &MetalTensor,
    rows: usize,
    cols: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![cols, rows])?;
    let rows_buf = MetalTensor::from_u32_scalar(ctx, rows as u32)?;
    let cols_buf = MetalTensor::from_u32_scalar(ctx, cols as u32)?;
    let n = rows * cols;
    dispatch_kernel(
        ctx,
        "transpose_2d",
        &[input, &out, &rows_buf, &cols_buf],
        &[],
        n,
        256,
    )?;
    Ok(out)
}

/// GPU MoE top-k routing.
/// gate_probs: [batch_seq, n_experts] post-softmax
/// Returns (indices: [batch_seq, top_k], weights: [batch_seq, top_k])
pub fn gpu_moe_topk(
    ctx: &MetalContext,
    gate_probs: &MetalTensor,
    batch_seq: usize,
    n_experts: usize,
    top_k: usize,
) -> Result<(MetalTensor, MetalTensor), String> {
    let out_indices = MetalTensor::zeros(ctx, vec![batch_seq, top_k])?;
    let out_weights = MetalTensor::zeros(ctx, vec![batch_seq, top_k])?;
    let n_experts_buf = MetalTensor::from_u32_scalar(ctx, n_experts as u32)?;
    let top_k_buf = MetalTensor::from_u32_scalar(ctx, top_k as u32)?;
    // One threadgroup per token, single thread per threadgroup (serial top-k is fine for small n_experts)
    dispatch_kernel(
        ctx,
        "moe_topk",
        &[gate_probs, &out_indices, &out_weights, &n_experts_buf, &top_k_buf],
        &[],
        batch_seq,
        1,
    )?;
    Ok((out_indices, out_weights))
}

/// GPU MoE expert weight extraction.
/// Returns per-token weight for a single expert from top-k routing.
pub fn gpu_moe_expert_weight(
    ctx: &MetalContext,
    indices: &MetalTensor,
    weights: &MetalTensor,
    batch_seq: usize,
    top_k: usize,
    expert_idx: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![batch_seq])?;
    let top_k_buf = MetalTensor::from_u32_scalar(ctx, top_k as u32)?;
    let expert_buf = MetalTensor::from_u32_scalar(ctx, expert_idx as u32)?;
    dispatch_kernel(
        ctx,
        "moe_topk_extract_weight",
        &[indices, weights, &out, &top_k_buf, &expert_buf],
        &[],
        batch_seq,
        256,
    )?;
    Ok(out)
}

/// GPU row-wise scaling: output[row, col] = input[row, col] * row_weights[row].
pub fn gpu_row_scale(
    ctx: &MetalContext,
    input: &MetalTensor,
    row_weights: &MetalTensor,
    n_rows: usize,
    n_cols: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![n_rows, n_cols])?;
    let cols_buf = MetalTensor::from_u32_scalar(ctx, n_cols as u32)?;
    dispatch_kernel(
        ctx,
        "row_scale",
        &[input, row_weights, &out, &cols_buf],
        &[],
        n_rows * n_cols,
        256,
    )?;
    Ok(out)
}

/// GPU MoE gather: gather tokens assigned to an expert.
pub fn gpu_moe_gather(
    ctx: &MetalContext,
    input: &MetalTensor,
    token_map: &MetalTensor,
    n_assigned: usize,
    hidden_dim: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![n_assigned, hidden_dim])?;
    let dim_buf = MetalTensor::from_u32_scalar(ctx, hidden_dim as u32)?;
    let n = n_assigned * hidden_dim;
    dispatch_kernel(
        ctx,
        "moe_gather",
        &[input, token_map, &out, &dim_buf],
        &[],
        n,
        256,
    )?;
    Ok(out)
}

/// GPU MoE weighted scatter-add: output[token] += weight * expert_out[assigned]
pub fn gpu_moe_scatter_add(
    ctx: &MetalContext,
    expert_out: &MetalTensor,
    token_map: &MetalTensor,
    weights: &MetalTensor,
    output: &MetalTensor,
    n_assigned: usize,
    hidden_dim: usize,
) -> Result<(), String> {
    let dim_buf = MetalTensor::from_u32_scalar(ctx, hidden_dim as u32)?;
    let n = n_assigned * hidden_dim;
    dispatch_kernel(
        ctx,
        "moe_scatter_add",
        &[expert_out, token_map, weights, output, &dim_buf],
        &[],
        n,
        256,
    )?;
    Ok(())
}

/// GPU cross-entropy forward: per-token losses.
pub fn gpu_cross_entropy_forward(
    ctx: &MetalContext,
    logits: &MetalTensor,
    targets: &MetalTensor,
    batch_seq: usize,
    vocab_size: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![batch_seq])?;
    let vocab_buf = MetalTensor::from_u32_scalar(ctx, vocab_size as u32)?;
    let threadgroup_size = 256;
    dispatch_kernel(
        ctx,
        "cross_entropy_forward",
        &[logits, targets, &out, &vocab_buf],
        &[],
        batch_seq * threadgroup_size,
        threadgroup_size,
    )?;
    Ok(out)
}

/// GPU cross-entropy backward: grad = scale * (softmax(logits) - one_hot(targets))
pub fn gpu_cross_entropy_backward(
    ctx: &MetalContext,
    logits: &MetalTensor,
    targets: &MetalTensor,
    batch_seq: usize,
    vocab_size: usize,
    scale: f32,
) -> Result<MetalTensor, String> {
    let grad = MetalTensor::zeros(ctx, vec![batch_seq, vocab_size])?;
    let vocab_buf = MetalTensor::from_u32_scalar(ctx, vocab_size as u32)?;
    let scale_buf = MetalTensor::from_f32_scalar(ctx, scale)?;
    let threadgroup_size = 256;
    dispatch_kernel(
        ctx,
        "cross_entropy_backward",
        &[logits, targets, &grad, &vocab_buf, &scale_buf],
        &[],
        batch_seq * threadgroup_size,
        threadgroup_size,
    )?;
    Ok(grad)
}

/// GPU reduce sum: sum all elements to scalar.
pub fn gpu_reduce_sum(
    ctx: &MetalContext,
    input: &MetalTensor,
    n: usize,
) -> Result<MetalTensor, String> {
    let out = MetalTensor::zeros(ctx, vec![1])?;
    let n_buf = MetalTensor::from_u32_scalar(ctx, n as u32)?;
    let threadgroup_size = 256;
    dispatch_kernel(
        ctx,
        "reduce_sum",
        &[input, &out, &n_buf],
        &[],
        threadgroup_size,
        threadgroup_size,
    )?;
    Ok(out)
}
