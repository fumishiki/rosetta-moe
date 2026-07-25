# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal_layers.jl — GPU layer primitives using MtlArray
#
# All operations use MtlArray directly. No Array() conversions.
# Forward AND backward implemented on GPU.

abstract type AbstractMetalLayer end

# =============================================================================
# MetalEmbedding — lookup table on GPU via one-hot matmul
# =============================================================================

mutable struct MetalEmbedding <: AbstractMetalLayer
    weight::MetalTensor          # (vocab_size, embed_dim) on GPU
    vocab_size::Int
    embed_dim::Int
    vocab_col::MtlMatrix{Float32}  # (vocab_size, 1) cached index column on GPU
    last_one_hot::Union{MtlMatrix{Float32}, Nothing}  # (num_tokens, vocab_size) for backward
    last_batch::Int
    last_seq_len::Int
end

function MetalEmbedding(vocab_size::Int, embed_dim::Int)
    std = sqrt(2f0 / Float32(embed_dim))
    vocab_cpu = reshape(Float32[i for i in 0:vocab_size-1], vocab_size, 1)
    MetalEmbedding(
        randn_std_mtensor(vocab_size, embed_dim; std=std),
        vocab_size,
        embed_dim,
        _upload_to_mtl(vocab_cpu),
        nothing,
        0,
        0,
    )
end

# Forward: gather rows via one-hot matmul on GPU
# token_ids (batch, seq) -> one_hot (num_tokens, vocab) @ weight (vocab, dim) -> (num_tokens, dim)
function gpu_forward(e::MetalEmbedding, input::MetalTensor)
    dims = size(input.data)
    batch, seq_len = dims[1], dims[2]
    if !inference_mode()
        e.last_batch = batch
        e.last_seq_len = seq_len
    end
    num_tokens = batch * seq_len

    # Build one-hot on GPU: (num_tokens, vocab_size)
    td_flat = reshape(input.data, 1, num_tokens)  # (1, num_tokens)
    one_hot_t = Float32.(e.vocab_col .== td_flat)  # (vocab, num_tokens) broadcast
    one_hot = permutedims(one_hot_t, (2, 1))  # (num_tokens, vocab)

    if !inference_mode()
        e.last_one_hot = one_hot
    end

    # Matmul: (num_tokens, vocab) @ (vocab, dim) = (num_tokens, dim)
    out = one_hot * e.weight.data  # MPS matmul on GPU
    MetalTensor(reshape(out, batch, seq_len, e.embed_dim), F32)
end

function gpu_backward(e::MetalEmbedding, grad_output::MetalTensor)
    dims = size(grad_output.data)
    batch, seq_len, embed_dim = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len

    flat_grad = reshape(grad_output.data, num_tokens, embed_dim)

    # Weight gradient: one_hot^T @ flat_grad = (vocab, num_tokens) @ (num_tokens, dim) = (vocab, dim)
    one_hot = e.last_one_hot
    w_grad = transpose(one_hot) * flat_grad  # (vocab, dim) — scatter-add via matmul

    if e.weight.grad === nothing
        e.weight.grad = w_grad
    else
        e.weight.grad .= w_grad
    end

    # Embedding backward returns zero grad for input (not trainable)
    zeros_mtensor(dims...)
end

gpu_parameters(e::MetalEmbedding) = MetalTensor[e.weight]

# =============================================================================
# MetalLinear — fully connected layer on GPU
# =============================================================================

mutable struct MetalLinear <: AbstractMetalLayer
    weight::MetalTensor                # (out_feat, in_feat) on GPU
    bias::Union{MetalTensor,Nothing}
    in_feat::Int
    out_feat::Int
    use_bias::Bool
    last_input::Union{MetalTensor,Nothing}
    infer_out_cache::Union{MtlMatrix{Float32},Nothing}
end

function MetalLinear(in_features::Int, out_features::Int, use_bias::Bool)
    std = sqrt(2f0 / Float32(in_features))
    w = randn_std_mtensor(out_features, in_features; std=std)
    b = use_bias ? zeros_mtensor(out_features) : nothing
    MetalLinear(w, b, in_features, out_features, use_bias, nothing, nothing)
end

function gpu_forward(l::MetalLinear, input::MetalTensor)
    if !inference_mode()
        l.last_input = input
    end
    dims = size(input.data)
    leading = dims[1:end-1]
    batch_size = prod(leading)
    flat = reshape(input.data, batch_size, l.in_feat)

    if inference_mode()
        if l.infer_out_cache === nothing || size(l.infer_out_cache) != (batch_size, l.out_feat)
            l.infer_out_cache = Metal.zeros(Float32, batch_size, l.out_feat)
        end
        out = l.infer_out_cache
        mul!(out, flat, transpose(l.weight.data))
        if l.use_bias
            out .+= reshape(l.bias.data, 1, :)
        end
        return MetalTensor(reshape(out, leading..., l.out_feat), input.dtype)
    else
        out = flat * transpose(l.weight.data)
        if l.use_bias
            out .+= reshape(l.bias.data, 1, :)
        end
        return MetalTensor(reshape(out, leading..., l.out_feat), input.dtype)
    end
end

function gpu_backward(l::MetalLinear, grad_output::MetalTensor)
    l.last_input === nothing && error("backward called before forward")
    input_shape = size(l.last_input.data)
    dims = size(grad_output.data)
    leading = dims[1:end-1]
    batch_size = prod(leading)

    flat_grad = reshape(grad_output.data, batch_size, l.out_feat)

    # grad_input = grad_output @ W
    grad_input = flat_grad * l.weight.data  # (batch, out) @ (out, in) = (batch, in)

    # weight gradient: grad_W = flat_grad^T @ flat_input
    flat_input = reshape(l.last_input.data, batch_size, l.in_feat)
    w_grad = transpose(flat_grad) * flat_input  # (out, batch) @ (batch, in) = (out, in)

    if l.weight.grad === nothing
        l.weight.grad = w_grad
    else
        l.weight.grad .= w_grad
    end

    # bias gradient
    if l.use_bias
        b_grad = sum(flat_grad; dims=1)  # (1, out)
        bg = reshape(b_grad, l.out_feat)
        if l.bias.grad === nothing
            l.bias.grad = bg
        else
            l.bias.grad .= bg
        end
    end

    MetalTensor(reshape(grad_input, input_shape...), grad_output.dtype)
end

function gpu_parameters(l::MetalLinear)
    l.use_bias ? MetalTensor[l.weight, l.bias] : MetalTensor[l.weight]
end

# =============================================================================
# MetalRMSNorm — RMS normalization on GPU
# =============================================================================

mutable struct MetalRMSNorm <: AbstractMetalLayer
    weight::MetalTensor       # gamma: learnable scale, (dim,) on GPU
    eps::Float32
    dim::Int
    last_input::Union{MetalTensor,Nothing}
    last_inv_rms::Union{MtlArray{Float32},Nothing}  # cached (num_vectors,1) for backward
end

function MetalRMSNorm(dim::Int, eps::Float32=1f-6)
    MetalRMSNorm(ones_mtensor(dim), eps, dim, nothing, nothing)
end

function gpu_forward(r::MetalRMSNorm, input::MetalTensor)
    if !inference_mode()
        r.last_input = input
    end
    d = input.data
    sz = size(d)
    total = length(d)
    dim = r.dim
    num_vectors = div(total, dim)

    flat = reshape(d, num_vectors, dim)
    # RMS = sqrt(mean(x^2) + eps)
    sum_sq = sum(flat .* flat; dims=2)  # (num_vectors, 1)
    inv_rms = 1f0 ./ sqrt.(sum_sq ./ Float32(dim) .+ r.eps)  # (num_vectors, 1)
    if !inference_mode()
        r.last_inv_rms = inv_rms
    end

    w = reshape(r.weight.data, 1, dim)  # broadcast-ready
    out_flat = flat .* inv_rms .* w
    MetalTensor(reshape(out_flat, sz), input.dtype)
end

function gpu_backward(r::MetalRMSNorm, grad_output::MetalTensor)
    r.last_input === nothing && error("backward called before forward")
    god = grad_output.data
    lid = r.last_input.data
    sz = size(god)
    total = length(god)
    dim = r.dim
    num_vectors = div(total, dim)

    flat_god = reshape(god, num_vectors, dim)
    flat_lid = reshape(lid, num_vectors, dim)
    inv_rms = r.last_inv_rms  # (num_vectors, 1)
    w = reshape(r.weight.data, 1, dim)

    # gamma gradient: sum_v(grad_output * x * inv_rms)
    w_grad = sum(flat_god .* flat_lid .* inv_rms; dims=1)  # (1, dim)
    wg = reshape(w_grad, dim)
    if r.weight.grad === nothing
        r.weight.grad = wg
    else
        r.weight.grad .= wg
    end

    # input gradient: d_x = (d_y * gamma / rms) - x * dot_sum / (dim * rms^3)
    scaled_god = flat_god .* w  # d_y * gamma
    dot_sum = sum(scaled_god .* flat_lid .* inv_rms; dims=2)  # (num_vectors, 1)
    inv_rms3 = inv_rms .* inv_rms .* inv_rms
    grad_input = scaled_god .* inv_rms .- flat_lid .* dot_sum .* inv_rms3 ./ Float32(dim)

    MetalTensor(reshape(grad_input, sz), grad_output.dtype)
end

gpu_parameters(r::MetalRMSNorm) = MetalTensor[r.weight]

# =============================================================================
# MetalSwiGLU — Gated feed-forward on GPU
# =============================================================================

mutable struct MetalSwiGLU <: AbstractMetalLayer
    w_gate::MetalLinear
    w_up::MetalLinear
    w_down::MetalLinear
    hidden_dim::Int
    ffn_dim::Int
    last_gate_pre_silu::Union{MtlArray{Float32}, Nothing}
    last_silu_gate::Union{MtlArray{Float32}, Nothing}
    last_up::Union{MetalTensor, Nothing}
    infer_fused_gate_up_weight::Union{MtlMatrix{Float32}, Nothing}
    infer_proj_cache::Union{MtlMatrix{Float32}, Nothing}
    infer_out_cache::Union{MtlMatrix{Float32}, Nothing}
end

function MetalSwiGLU(hidden_dim::Int, ffn_dim::Int)
    MetalSwiGLU(
        MetalLinear(hidden_dim, ffn_dim, false),
        MetalLinear(hidden_dim, ffn_dim, false),
        MetalLinear(ffn_dim, hidden_dim, false),
        hidden_dim, ffn_dim, nothing, nothing, nothing, nothing, nothing, nothing
    )
end

function _ensure_infer_fused_gate_up_weight!(s::MetalSwiGLU)::MtlMatrix{Float32}
    if s.infer_fused_gate_up_weight === nothing ||
       size(s.infer_fused_gate_up_weight) != (2 * s.ffn_dim, s.hidden_dim)
        fused = Metal.zeros(Float32, 2 * s.ffn_dim, s.hidden_dim)
        fused[1:s.ffn_dim, :] .= s.w_gate.weight.data
        fused[s.ffn_dim + 1:2 * s.ffn_dim, :] .= s.w_up.weight.data
        s.infer_fused_gate_up_weight = fused
    end
    return s.infer_fused_gate_up_weight
end

function gpu_forward(s::MetalSwiGLU, input::MetalTensor)
    if inference_mode()
        dims = size(input.data)
        leading = dims[1:end-1]
        batch_size = prod(leading)
        flat = reshape(input.data, batch_size, s.hidden_dim)

        fused_w = _ensure_infer_fused_gate_up_weight!(s)
        if s.infer_proj_cache === nothing || size(s.infer_proj_cache) != (batch_size, 2 * s.ffn_dim)
            s.infer_proj_cache = Metal.zeros(Float32, batch_size, 2 * s.ffn_dim)
        end
        proj = s.infer_proj_cache
        mul!(proj, flat, transpose(fused_w))

        gate = @view proj[:, 1:s.ffn_dim]
        up = @view proj[:, s.ffn_dim + 1:2 * s.ffn_dim]
        gate .= gate ./ (1f0 .+ exp.(.-gate))
        gate .*= up

        if s.infer_out_cache === nothing || size(s.infer_out_cache) != (batch_size, s.hidden_dim)
            s.infer_out_cache = Metal.zeros(Float32, batch_size, s.hidden_dim)
        end
        out = s.infer_out_cache
        mul!(out, gate, transpose(s.w_down.weight.data))
        return MetalTensor(reshape(out, leading..., s.hidden_dim), input.dtype)
    end

    # Keep inference cache coherent with train-time weight updates.
    s.infer_fused_gate_up_weight = nothing
    s.infer_proj_cache = nothing
    s.infer_out_cache = nothing

    gate = gpu_forward(s.w_gate, input)
    if !inference_mode()
        s.last_gate_pre_silu = copy(gate.data)
    end
    silu_in_place_mtl!(gate)
    if !inference_mode()
        s.last_silu_gate = copy(gate.data)
    end
    up = gpu_forward(s.w_up, input)
    if !inference_mode()
        s.last_up = up
    end
    mul_in_place_mtl!(gate, up)
    gpu_forward(s.w_down, gate)
end

function gpu_backward(s::MetalSwiGLU, grad_output::MetalTensor)
    grad_hidden = gpu_backward(s.w_down, grad_output)
    ghd = grad_hidden.data

    # silu derivative: sig = sigmoid(z), dsilu = sig * (1 + z*(1-sig))
    gps = s.last_gate_pre_silu
    sig = 1f0 ./ (1f0 .+ exp.(.-gps))
    dsilu = sig .* (1f0 .+ gps .* (1f0 .- sig))

    # grad_gate = grad_hidden * up * dsilu
    lu = s.last_up.data
    grad_gate = ghd .* lu .* dsilu

    # grad_up = grad_hidden * silu(gate)
    sg = s.last_silu_gate
    grad_up = ghd .* sg

    g1 = gpu_backward(s.w_gate, MetalTensor(grad_gate, grad_output.dtype))
    g2 = gpu_backward(s.w_up, MetalTensor(grad_up, grad_output.dtype))
    add_in_place_mtl!(g1, g2)
    return g1
end

function gpu_parameters(s::MetalSwiGLU)
    vcat(gpu_parameters(s.w_gate), gpu_parameters(s.w_up), gpu_parameters(s.w_down))
end
