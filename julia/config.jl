# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

# config.jl — Model hyperparameter configuration
#
# Defines the Config struct that parameterizes the entire MoE Transformer.
# Two presets are provided: tiny() for testing and default_6_9b() for a
# realistic 6.9B-parameter model.

struct Config
    hidden_dim::Int;    n_layers::Int;    n_heads::Int;     n_kv_heads::Int
    n_experts::Int;     top_k_experts::Int; vocab_size::Int; max_seq_len::Int
    ffn_dim::Int;       head_dim::Int;    rope_base::Float32; rope_alpha::Float32
end

# Tiny config for unit tests and benchmarks (64-dim, 2 layers, 4 heads)
tiny() = Config(64, 2, 4, 1, 4, 2, 1000, 512, 256, 16, 10000f0, 1f0)

# Small config for scale comparison benchmarks (256-dim, 2 layers, 4 heads)
small() = Config(256, 2, 4, 1, 4, 2, 1000, 512, 1024, 64, 10000f0, 1f0)

# Realistic 6.9B config (768-dim, 30 layers, 12 heads, 16 experts top-4)
default_6_9b() = Config(768, 30, 12, 1, 16, 4, 32000, 32768, 6144, 64, 10000f0, 8f0)

# Total parameters (all experts counted):
#   embedding + per_layer * n_layers + lm_head
#   per_layer = attention_params + router + expert_ffn * n_experts + 2 * rmsnorm
function total_params(c::Config)
    emb = c.vocab_size * c.hidden_dim
    attn = c.hidden_dim * c.hidden_dim * 2 + c.hidden_dim * c.head_dim * 2
    per_layer = attn + c.hidden_dim * c.n_experts + c.hidden_dim * c.ffn_dim * 3 * c.n_experts + c.hidden_dim * 2
    emb + per_layer * c.n_layers + c.hidden_dim * c.vocab_size
end

# Active parameters per forward pass (only top_k experts activated):
#   Same as total_params but expert FFN uses top_k instead of n_experts
function active_params(c::Config)
    emb = c.vocab_size * c.hidden_dim
    attn = c.hidden_dim * c.hidden_dim * 2 + c.hidden_dim * c.head_dim * 2
    per_layer = attn + c.hidden_dim * c.ffn_dim * 3 * c.top_k_experts + c.hidden_dim * 2
    emb + per_layer * c.n_layers + c.hidden_dim * c.vocab_size
end
