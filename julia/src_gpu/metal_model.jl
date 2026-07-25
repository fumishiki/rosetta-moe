# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal_model.jl — Full MoE Transformer model on GPU
#
# All layers use MetalTensor (MtlArray). Forward and backward run on GPU.

mutable struct MetalMoETransformer
    config::Config
    embedding::MetalEmbedding
    blocks::Vector{MetalTransformerBlock}
    final_norm::MetalRMSNorm
    lm_head::MetalLinear
end

function MetalMoETransformer(cfg::Config)
    MetalMoETransformer(
        cfg,
        MetalEmbedding(cfg.vocab_size, cfg.hidden_dim),
        [MetalTransformerBlock(cfg) for _ in 1:cfg.n_layers],
        MetalRMSNorm(cfg.hidden_dim),
        MetalLinear(cfg.hidden_dim, cfg.vocab_size, false)
    )
end

tiny_metal_model() = MetalMoETransformer(tiny())
small_metal_model() = MetalMoETransformer(small())
medium_metal_model() = MetalMoETransformer(medium())

function gpu_forward(m::MetalMoETransformer, input::MetalTensor)
    x = gpu_forward(m.embedding, input)
    for blk in m.blocks
        x = gpu_forward(blk, x)
    end
    x = gpu_forward(m.final_norm, x)
    gpu_forward(m.lm_head, x)
end

function gpu_backward(m::MetalMoETransformer, grad_output::MetalTensor)
    grad = gpu_backward(m.final_norm, gpu_backward(m.lm_head, grad_output))
    for i in length(m.blocks):-1:1
        grad = gpu_backward(m.blocks[i], grad)
    end
    gpu_backward(m.embedding, grad)
end

function gpu_parameters(m::MetalMoETransformer)
    params = gpu_parameters(m.embedding)
    for blk in m.blocks
        append!(params, gpu_parameters(blk))
    end
    append!(params, gpu_parameters(m.final_norm))
    append!(params, gpu_parameters(m.lm_head))
    params
end

total_aux_loss_mtl(m::MetalMoETransformer, alpha::Float32) = sum(blk -> aux_loss_mtl(blk, alpha), m.blocks)

function apply_z_loss_mtl!(m::MetalMoETransformer, z_weight::Float32)::Float32
    total = 0f0
    for blk in m.blocks
        total += compute_z_loss_with_grad_mtl!(blk.moe.router, z_weight)
    end
    total
end

function apply_aux_loss_mtl!(m::MetalMoETransformer, alpha::Float32)::Float32
    total = 0f0
    for blk in m.blocks
        total += compute_aux_loss_with_grad_mtl!(blk.moe.router, alpha)
    end
    total
end
