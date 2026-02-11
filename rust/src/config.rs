// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Model configuration profiles.
//!
//! All four language implementations (Rust, Julia, Go, Python) share the same
//! hyperparameters so benchmark results are directly comparable.

/// Shared model configuration for local CPU baseline.
///
/// Two presets:
/// - `default_6_9b()`: full-scale config modelled after a 6.9B-param MoE architecture
///   (768 hidden, 30 layers, 16 experts with top-4 routing).
/// - `tiny()`: minimal config for unit tests and benchmarks (64 hidden, 2 layers).
///
/// Key relationships:
/// - `n_heads * head_dim` = total Q projection width (must equal `hidden_dim` for residual add).
/// - `n_kv_heads` < `n_heads` enables Multi-Query / Grouped-Query Attention.
/// - `ffn_dim` is the SwiGLU intermediate dimension (typically 8x hidden).
/// - `rope_alpha` > 1.0 enables YaRN-style base frequency scaling for long context.
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Number of key/value heads (n_kv_heads < n_heads = grouped-query attention)
    pub n_kv_heads: usize,
    pub n_experts: usize,
    /// Top-k experts selected per token by the router
    pub top_k_experts: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    /// SwiGLU intermediate (up/gate projection) dimension
    pub ffn_dim: usize,
    pub head_dim: usize,
    /// RoPE base frequency (default 10000.0 per Vaswani et al.)
    pub rope_base: f32,
    /// RoPE frequency scaling factor (alpha > 1.0 = YaRN extended context)
    pub rope_alpha: f32,
}

impl Config {
    pub fn default_6_9b() -> Self {
        Self {
            hidden_dim: 768,
            n_layers: 30,
            n_heads: 12,
            n_kv_heads: 1,
            n_experts: 16,
            top_k_experts: 4,
            vocab_size: 32000,
            max_seq_len: 32768,
            ffn_dim: 6144,
            head_dim: 64,
            rope_base: 10000.0,
            rope_alpha: 8.0,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hidden_dim: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 1,
            n_experts: 4,
            top_k_experts: 2,
            vocab_size: 1000,
            max_seq_len: 512,
            ffn_dim: 256,
            head_dim: 16,
            rope_base: 10000.0,
            rope_alpha: 1.0,
        }
    }

    pub fn small() -> Self {
        Self {
            hidden_dim: 256,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 1,
            n_experts: 4,
            top_k_experts: 2,
            vocab_size: 1000,
            max_seq_len: 512,
            ffn_dim: 1024,
            head_dim: 64,
            rope_base: 10000.0,
            rope_alpha: 1.0,
        }
    }
}
