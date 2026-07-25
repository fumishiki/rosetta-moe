// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! GPU model: all weights on Metal, pure GPU forward pass.
//!
//! GpuModel holds all transformer weights as MetalTensors.
//! Constructed once from a CPU model via `GpuModel::from_cpu()`, then runs
//! entirely on GPU with zero CPU↔GPU transfers during forward/backward.
//!
//! Pipeline (100% Metal):
//! 1. Embedding gather (MSL kernel)
//! 2. Per TransformerBlock:
//!    a. RMSNorm (MSL) -> Q/K/V linear (MPS) -> RoPE (MSL) -> Attention (MPS+MSL) -> O linear (MPS) -> residual (MSL)
//!    b. RMSNorm (MSL) -> Router (MPS+MSL softmax+topk) -> Expert SwiGLU (MPS+MSL) -> scatter-add (MSL) -> residual (MSL)
//! 3. Final RMSNorm (MSL) -> LM head (MPS) -> logits

use super::metal_tensor::{MetalContext, MetalTensor};
use super::metal_layers::*;
use crate::config::Config;

/// Pre-uploaded attention weights for one layer.
struct GpuAttentionWeights {
    q_weight_t: MetalTensor,   // [in, n_heads*head_dim] transposed
    k_weight_t: MetalTensor,   // [in, n_kv_heads*head_dim] transposed
    v_weight_t: MetalTensor,   // [in, n_kv_heads*head_dim] transposed
    o_weight_t: MetalTensor,   // [n_heads*head_dim, hidden] transposed
}

/// Pre-uploaded expert FFN weights for one expert.
struct GpuExpertWeights {
    gate_weight_t: MetalTensor, // [in, ffn_dim] transposed
    up_weight_t: MetalTensor,   // [in, ffn_dim] transposed
    down_weight_t: MetalTensor, // [ffn_dim, hidden] transposed
}

/// Pre-uploaded weights for one transformer block.
struct GpuBlockWeights {
    attn_norm_weight: MetalTensor,
    attn: GpuAttentionWeights,
    ffn_norm_weight: MetalTensor,
    router_gate_weight_t: MetalTensor, // [hidden, n_experts] transposed
    experts: Vec<GpuExpertWeights>,
}

/// Full model with all weights on GPU.
pub struct GpuModel {
    config: Config,
    embedding_weight: MetalTensor,
    blocks: Vec<GpuBlockWeights>,
    final_norm_weight: MetalTensor,
    lm_head_weight_t: MetalTensor,
    /// Precomputed RoPE cos/sin caches: [max_seq, half_dim]
    rope_cos: MetalTensor,
    rope_sin: MetalTensor,
    eps: f32,
}

impl GpuModel {
    /// Upload all weights from a CPU MoETransformer to GPU.
    /// This is the ONLY point where CPU data is read.
    /// After construction, forward/backward runs 100% on Metal.
    pub fn from_cpu(
        ctx: &MetalContext,
        model: &crate::cpu::model::MoETransformer,
    ) -> Result<Self, String> {
        let cfg = model.config().clone();
        let hidden = cfg.hidden_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let ffn_dim = cfg.ffn_dim;
        let n_experts = cfg.n_experts;
        let vocab = cfg.vocab_size;

        // Helper: upload weight [out, in] and transpose to [in, out] on GPU
        let upload_transposed = |data: &[f32], out_dim: usize, in_dim: usize| -> Result<MetalTensor, String> {
            let raw = MetalTensor::upload(ctx, data, vec![out_dim, in_dim])?;
            gpu_transpose(ctx, &raw, out_dim, in_dim)
        };

        // Embedding weight (no transpose needed — used by gather kernel)
        let embedding_weight = MetalTensor::upload(
            ctx,
            model.embedding().weight.data(),
            vec![vocab, hidden],
        )?;

        // Transformer blocks
        let blocks_cpu = unsafe {
            // SAFETY: We only read weights (immutable access). The &mut requirement
            // on blocks_mut is for the forward/backward cache, not weight data.
            // We cast away mutability only to access the block accessor.
            let model_ptr = model as *const crate::cpu::model::MoETransformer
                as *mut crate::cpu::model::MoETransformer;
            (*model_ptr).blocks_mut()
        };

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for block in &mut *blocks_cpu {
            // Attention norm
            let attn_norm_weight = MetalTensor::upload(
                ctx,
                block.attn_norm().weight.data(),
                vec![hidden],
            )?;

            // Attention projections (direct accessor — no Layer trait needed)
            let attn = block.attention();
            let q_wt = upload_transposed(attn.q_weight_data(), n_heads * head_dim, hidden)?;
            let k_wt = upload_transposed(attn.k_weight_data(), n_kv_heads * head_dim, hidden)?;
            let v_wt = upload_transposed(attn.v_weight_data(), n_kv_heads * head_dim, hidden)?;
            let o_wt = upload_transposed(attn.o_weight_data(), hidden, n_heads * head_dim)?;

            // FFN norm
            let ffn_norm_weight = MetalTensor::upload(
                ctx,
                block.ffn_norm().weight.data(),
                vec![hidden],
            )?;

            // Router gate
            let router_gate_data = block.moe().router().gate_weight_data();
            let router_gate_wt = upload_transposed(router_gate_data, n_experts, hidden)?;

            // Experts
            let mut gpu_experts = Vec::with_capacity(n_experts);
            for expert in block.moe().experts() {
                let gate_wt = upload_transposed(expert.gate_proj().weight.data(), ffn_dim, hidden)?;
                let up_wt = upload_transposed(expert.up_proj().weight.data(), ffn_dim, hidden)?;
                let down_wt = upload_transposed(expert.down_proj().weight.data(), hidden, ffn_dim)?;
                gpu_experts.push(GpuExpertWeights {
                    gate_weight_t: gate_wt,
                    up_weight_t: up_wt,
                    down_weight_t: down_wt,
                });
            }

            blocks.push(GpuBlockWeights {
                attn_norm_weight,
                attn: GpuAttentionWeights {
                    q_weight_t: q_wt,
                    k_weight_t: k_wt,
                    v_weight_t: v_wt,
                    o_weight_t: o_wt,
                },
                ffn_norm_weight,
                router_gate_weight_t: router_gate_wt,
                experts: gpu_experts,
            });
        }

        // Final norm
        let final_norm_weight = MetalTensor::upload(
            ctx,
            model.final_norm().weight.data(),
            vec![hidden],
        )?;

        // LM head
        let lm_head_wt = upload_transposed(model.lm_head().weight.data(), vocab, hidden)?;

        // Precompute RoPE cos/sin cache
        let base = if cfg.rope_alpha > 1.0 {
            cfg.rope_base * cfg.rope_alpha.powf(head_dim as f32 / (head_dim as f32 - 2.0))
        } else {
            cfg.rope_base
        };
        let half_dim = head_dim / 2;
        let max_seq = cfg.max_seq_len;
        let mut cos_data = vec![0.0f32; max_seq * half_dim];
        let mut sin_data = vec![0.0f32; max_seq * half_dim];
        for pos in 0..max_seq {
            for i in 0..half_dim {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                cos_data[pos * half_dim + i] = c;
                sin_data[pos * half_dim + i] = s;
            }
        }
        let rope_cos = MetalTensor::upload(ctx, &cos_data, vec![max_seq, half_dim])?;
        let rope_sin = MetalTensor::upload(ctx, &sin_data, vec![max_seq, half_dim])?;

        let eps = block_eps(blocks_cpu);

        Ok(Self {
            config: cfg,
            embedding_weight,
            blocks,
            final_norm_weight,
            lm_head_weight_t: lm_head_wt,
            rope_cos,
            rope_sin,
            eps,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// Extract eps from first block's attn_norm (all blocks use the same eps).
fn block_eps(blocks: &[crate::cpu::moe::TransformerBlock]) -> f32 {
    if blocks.is_empty() {
        1e-6
    } else {
        blocks[0].attn_norm().eps()
    }
}

/// GPU forward pass: 100% Metal, zero CPU↔GPU transfers.
///
/// input: MetalTensor [batch, seq_len] containing token IDs as f32.
/// Returns: MetalTensor [batch_seq, vocab_size] logits.
pub fn gpu_forward(
    ctx: &MetalContext,
    gpu_model: &GpuModel,
    input: &MetalTensor,
) -> Result<MetalTensor, String> {
    let cfg = &gpu_model.config;
    let batch = input.shape()[0];
    let seq_len = input.shape()[1];
    let hidden = cfg.hidden_dim;
    let batch_seq = batch * seq_len;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let n_experts = cfg.n_experts;
    let top_k = cfg.top_k_experts;
    let eps = gpu_model.eps;

    // 1. Embedding gather (MSL kernel)
    let mut x = gpu_embedding(ctx, input, &gpu_model.embedding_weight, batch_seq, hidden)?;

    // 2. Transformer blocks
    for block in &gpu_model.blocks {
        // --- Attention sub-block ---
        // RMSNorm
        let normed = gpu_rmsnorm(ctx, &x, &block.attn_norm_weight, batch_seq, hidden, eps)?;

        // Q/K/V projections (MPS matmul)
        let q = gpu_linear(ctx, &normed, &block.attn.q_weight_t, hidden, n_heads * head_dim)?;
        let k = gpu_linear(ctx, &normed, &block.attn.k_weight_t, hidden, n_kv_heads * head_dim)?;
        let v = gpu_linear(ctx, &normed, &block.attn.v_weight_t, hidden, n_kv_heads * head_dim)?;

        // RoPE (in-place MSL kernel)
        gpu_rope_inplace(ctx, &q, &gpu_model.rope_cos, &gpu_model.rope_sin, batch_seq, n_heads, head_dim, seq_len)?;
        gpu_rope_inplace(ctx, &k, &gpu_model.rope_cos, &gpu_model.rope_sin, batch_seq, n_kv_heads, head_dim, seq_len)?;

        // Attention: scores = scale * Q @ K^T, mask, softmax, output = weights @ V
        // All steps stay on GPU kernels (no host-side head extraction/upload).
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_heads = batch * n_heads;
        let all_scores = gpu_attention_scores(
            ctx,
            &q,
            &k,
            batch,
            n_heads,
            n_kv_heads,
            seq_len,
            head_dim,
            scale,
        )?;

        // Causal mask (MSL kernel)
        gpu_causal_mask(ctx, &all_scores, total_heads, seq_len)?;

        // Softmax per row (MSL kernel)
        let attn_weights = gpu_softmax(ctx, &all_scores, total_heads * seq_len, seq_len)?;

        let attn_out = gpu_attention_weighted_sum(
            ctx,
            &attn_weights,
            &v,
            batch_seq,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
        )?;
        let o_out = gpu_linear(ctx, &attn_out, &block.attn.o_weight_t, n_heads * head_dim, hidden)?;

        // Residual add
        x = gpu_add(ctx, &x, &o_out)?;

        // --- MoE sub-block ---
        // RMSNorm
        let normed2 = gpu_rmsnorm(ctx, &x, &block.ffn_norm_weight, batch_seq, hidden, eps)?;

        // Router: gate_logits = normed2 @ gate_weight_t
        let gate_logits = gpu_linear(ctx, &normed2, &block.router_gate_weight_t, hidden, n_experts)?;

        // Softmax -> probabilities
        let gate_probs = gpu_softmax(ctx, &gate_logits, batch_seq, n_experts)?;

        // Top-k selection (MSL kernel)
        let (expert_indices, expert_weights) = gpu_moe_topk(ctx, &gate_probs, batch_seq, n_experts, top_k)?;

        // Per-expert weighted accumulation (fully GPU):
        // 1) compute per-token routing weight for expert e from top-k outputs
        // 2) run expert on all tokens
        // 3) row-scale by routing weight and add to moe_out
        let mut moe_out = MetalTensor::zeros(ctx, vec![batch_seq, hidden])?;

        for e_idx in 0..n_experts {
            let expert_row_w = gpu_moe_expert_weight(
                ctx,
                &expert_indices,
                &expert_weights,
                batch_seq,
                top_k,
                e_idx,
            )?;

            let expert = &block.experts[e_idx];
            let gate = gpu_linear(ctx, &normed2, &expert.gate_weight_t, hidden, cfg.ffn_dim)?;
            let gate_silu = gpu_silu(ctx, &gate)?;
            let up = gpu_linear(ctx, &normed2, &expert.up_weight_t, hidden, cfg.ffn_dim)?;
            let fused = gpu_mul(ctx, &gate_silu, &up)?;
            let expert_out = gpu_linear(ctx, &fused, &expert.down_weight_t, cfg.ffn_dim, hidden)?;

            let weighted = gpu_row_scale(ctx, &expert_out, &expert_row_w, batch_seq, hidden)?;
            moe_out = gpu_add(ctx, &moe_out, &weighted)?;
        }

        // Residual add
        x = gpu_add(ctx, &x, &moe_out)?;
    }

    // 3. Final RMSNorm
    let normed_final = gpu_rmsnorm(ctx, &x, &gpu_model.final_norm_weight, batch_seq, hidden, gpu_model.eps)?;

    // 4. LM Head
    let logits = gpu_linear(ctx, &normed_final, &gpu_model.lm_head_weight_t, hidden, cfg.vocab_size)?;

    Ok(logits)
}
