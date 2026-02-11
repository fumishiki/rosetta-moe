# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

# model.jl — Full MoE Transformer model assembly
#
# Assembles the complete model: Embedding -> N x TransformerBlock -> RMSNorm -> Linear (lm_head).
# The forward pass uses foldl to chain blocks, and backward uses foldr for reverse traversal.
#
# Shape convention (column-major, Julia style):
#   Input:  (batch, seq_len) — token IDs as Float32
#   Hidden: (batch, seq_len, hidden_dim)
#   Output: (batch, seq_len, vocab_size) — logits

mutable struct MoETransformer
    config::Config
    embedding::Embedding
    blocks::Vector{TransformerBlock}
    final_norm::RMSNorm
    lm_head::Linear            # ties could share weights with embedding, but kept separate here
end

function MoETransformer(cfg::Config)
    MoETransformer(
        cfg,
        Embedding(cfg.vocab_size, cfg.hidden_dim),
        [TransformerBlock(cfg) for _ in 1:cfg.n_layers],
        RMSNorm(cfg.hidden_dim),
        Linear(cfg.hidden_dim, cfg.vocab_size, false)
    )
end

tiny_model() = MoETransformer(tiny())
small_model() = MoETransformer(small())
default_model() = MoETransformer(default_6_9b())

# Forward: Embed -> Blocks -> Norm -> LM Head
# foldl chains blocks left-to-right: block_1(block_0(embed(x)))
function forward(m::MoETransformer, input::Tensor)
    x = foldl((x, blk) -> forward(blk, x), m.blocks; init=forward(m.embedding, input))
    forward(m.lm_head, forward(m.final_norm, x))
end

# Convenience: accept raw token ID vector and reshape to (batch, seq_len)
function forward_ids(m::MoETransformer, token_ids::Vector{Int}, batch::Int, seq_len::Int)
    data = Array{Float32}(undef, batch, seq_len)
    for i in eachindex(token_ids)
        data[i] = Float32(token_ids[i])
    end
    forward(m, Tensor(data, F32))
end

# Backward: reverse order through blocks using foldr
function backward(m::MoETransformer, grad_output::Tensor)
    grad = backward(m.final_norm, backward(m.lm_head, grad_output))
    backward(m.embedding, foldr((blk, g) -> backward(blk, g), m.blocks; init=grad))
end

function parameters(m::MoETransformer)
    Iterators.flatten((parameters(m.embedding), Iterators.flatten(parameters.(m.blocks)),
                       parameters(m.final_norm), parameters(m.lm_head)))
end

# Sum auxiliary losses across all transformer blocks
total_aux_loss(m::MoETransformer, alpha::Float32) = sum(blk -> aux_loss(blk, alpha), m.blocks)
