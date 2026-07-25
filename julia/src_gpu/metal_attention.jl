# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal_attention.jl — Multi-Query Attention with RoPE on GPU
#
# All matrix operations use MtlArray. Attention score computation uses
# per-head GPU matmul via MPS. No Array() conversions in forward/backward.
#
# RoPE applied via GPU broadcast (cos/sin tables uploaded once).
# Causal masking via GPU broadcast. Softmax via GPU broadcast.

mutable struct MetalMQAttention <: AbstractMetalLayer
    wQ::MetalLinear; wK::MetalLinear; wV::MetalLinear; wO::MetalLinear
    n_heads::Int; n_kv_heads::Int; head_dim::Int
    hidden_dim::Int; scale::Float32
    freqs::Vector{Float32}  # CPU-side precomputed RoPE frequencies (small, read-only)
    rope_cos_cache::Union{MtlArray{Float32,4}, Nothing}   # (1, seq, 1, head_dim/2)
    rope_sin_cache::Union{MtlArray{Float32,4}, Nothing}   # (1, seq, 1, head_dim/2)
    causal_mask_cache::Union{MtlMatrix{Float32}, Nothing} # (seq, seq), 0 or -inf
    causal_binary_cache::Union{MtlMatrix{Float32}, Nothing} # (seq, seq), 1 or 0
    cache_seq_len::Int
    # Cached for backward
    last_q_perm::Union{MtlArray{Float32,4}, Nothing}
    last_k_perm::Union{MtlArray{Float32,4}, Nothing}
    last_v_perm::Union{MtlArray{Float32,4}, Nothing}
    last_attn_weights::Union{MtlArray{Float32,4}, Nothing}  # (seq, seq, heads, batch) on GPU
    last_batch::Int
    last_seq_len::Int
    infer_scores_cache::Union{MtlArray{Float32,4}, Nothing}
    infer_weights_cache::Union{MtlArray{Float32,4}, Nothing}
    infer_out_cache::Union{MtlArray{Float32,4}, Nothing}
end

function MetalMQAttention(hidden_dim::Int, n_heads::Int, n_kv_heads::Int, head_dim::Int,
                          rope_base::Float32, rope_alpha::Float32)
    base = rope_base
    if rope_alpha > 1f0
        base = rope_base * (rope_alpha ^ (Float32(head_dim) / Float32(head_dim - 2)))
    end
    half_dim = head_dim ÷ 2
    freqs = Float32[1f0 / (base ^ (Float32(2 * i) / Float32(head_dim))) for i in 0:half_dim-1]
    MetalMQAttention(
        MetalLinear(hidden_dim, n_heads * head_dim, false),
        MetalLinear(hidden_dim, n_kv_heads * head_dim, false),
        MetalLinear(hidden_dim, n_kv_heads * head_dim, false),
        MetalLinear(n_heads * head_dim, hidden_dim, false),
        n_heads, n_kv_heads, head_dim, hidden_dim,
        1f0 / sqrt(Float32(head_dim)), freqs,
        nothing, nothing, nothing, nothing, 0,
        nothing, nothing, nothing, nothing, 0, 0,
        nothing, nothing, nothing
    )
end

# Precompute and cache RoPE tables + causal masks for current sequence length.
function _ensure_attention_cache!(a::MetalMQAttention, seq_len::Int)
    if a.cache_seq_len == seq_len &&
       a.rope_cos_cache !== nothing &&
       a.rope_sin_cache !== nothing &&
       a.causal_mask_cache !== nothing &&
       a.causal_binary_cache !== nothing
        return nothing
    end

    half_dim = a.head_dim ÷ 2
    pos_cpu = reshape(Float32[(s - 1) for s in 1:seq_len], seq_len, 1)
    freq_cpu = reshape(a.freqs, 1, half_dim)
    angles_cpu = pos_cpu * freq_cpu

    a.rope_cos_cache = _upload_to_mtl(reshape(cos.(angles_cpu), 1, seq_len, 1, half_dim))
    a.rope_sin_cache = _upload_to_mtl(reshape(sin.(angles_cpu), 1, seq_len, 1, half_dim))

    rows_cpu = reshape(Float32[qi for qi in 1:seq_len], seq_len, 1)
    cols_cpu = reshape(Float32[ki for ki in 1:seq_len], 1, seq_len)
    rows_mtl = _upload_to_mtl(rows_cpu)
    cols_mtl = _upload_to_mtl(cols_cpu)

    causal_binary = Float32.(cols_mtl .<= rows_mtl)
    a.causal_binary_cache = causal_binary
    a.causal_mask_cache = ifelse.(causal_binary .== 1f0, 0f0, -3.4028235f38)
    a.cache_seq_len = seq_len
    return nothing
end

# RoPE on GPU via cached cos/sin tables.
function _gpu_apply_rope!(
    data::MtlArray{Float32,4},
    cos_table::MtlArray{Float32,4},
    sin_table::MtlArray{Float32,4},
    head_dim::Int,
)
    half_dim = head_dim ÷ 2

    # Extract even/odd slices: data[:, :, :, 1:2:end] and data[:, :, :, 2:2:end]
    x_even = data[:, :, :, 1:2:head_dim]  # (batch, seq, heads, half_dim)
    x_odd  = data[:, :, :, 2:2:head_dim]

    # Rotated: [x0*cos - x1*sin, x0*sin + x1*cos]
    new_even = x_even .* cos_table .- x_odd .* sin_table
    new_odd  = x_even .* sin_table .+ x_odd .* cos_table

    # Write back interleaved
    data[:, :, :, 1:2:head_dim] .= new_even
    data[:, :, :, 2:2:head_dim] .= new_odd
end

function _gpu_attention_inference(
    a::MetalMQAttention,
    q_4d::MtlArray{Float32,4},
    k_4d::MtlArray{Float32,4},
    v_4d::MtlArray{Float32,4},
    causal_mask::MtlMatrix{Float32},
    batch::Int,
    seq_len::Int,
)
    n_heads = a.n_heads
    n_kv = a.n_kv_heads
    head_dim = a.head_dim

    k_rep = n_kv == n_heads ? k_4d : repeat(k_4d, 1, 1, div(n_heads, n_kv), 1)
    v_rep = n_kv == n_heads ? v_4d : repeat(v_4d, 1, 1, div(n_heads, n_kv), 1)

    if a.infer_scores_cache === nothing || size(a.infer_scores_cache) != (batch, seq_len, seq_len, n_heads)
        a.infer_scores_cache = Metal.zeros(Float32, batch, seq_len, seq_len, n_heads)
    end
    scores = a.infer_scores_cache

    # NOTE: only singleton insertion via reshape (no axis reordering).
    # q5: (batch, query_seq, 1, heads, head_dim)
    # k5: (batch, 1, key_seq, heads, head_dim)
    q5 = reshape(q_4d, batch, seq_len, 1, n_heads, head_dim)
    k5 = reshape(k_rep, batch, 1, seq_len, n_heads, head_dim)
    scores .= reshape(sum(q5 .* k5; dims=5), batch, seq_len, seq_len, n_heads)
    scores .*= a.scale
    scores .+= reshape(causal_mask, 1, seq_len, seq_len, 1)

    if a.infer_weights_cache === nothing || size(a.infer_weights_cache) != (batch, seq_len, seq_len, n_heads)
        a.infer_weights_cache = Metal.zeros(Float32, batch, seq_len, seq_len, n_heads)
    end
    weights = a.infer_weights_cache
    mx = maximum(scores; dims=3)
    weights .= exp.(scores .- mx)
    weights ./= sum(weights; dims=3)

    if a.infer_out_cache === nothing || size(a.infer_out_cache) != (batch, seq_len, n_heads, head_dim)
        a.infer_out_cache = Metal.zeros(Float32, batch, seq_len, n_heads, head_dim)
    end
    out = a.infer_out_cache
    w5 = reshape(weights, batch, seq_len, seq_len, n_heads, 1)
    v5 = reshape(v_rep, batch, 1, seq_len, n_heads, head_dim)
    out .= reshape(sum(w5 .* v5; dims=3), batch, seq_len, n_heads, head_dim)
    out
end

function gpu_forward(a::MetalMQAttention, input::MetalTensor)
    dims = size(input.data)
    batch = dims[1]
    seq_len = dims[2]
    if !inference_mode()
        a.last_batch = batch
        a.last_seq_len = seq_len
    end

    q_out = gpu_forward(a.wQ, input)
    k_out = gpu_forward(a.wK, input)
    v_out = gpu_forward(a.wV, input)

    q_4d = reshape(q_out.data, batch, seq_len, a.n_heads, a.head_dim)
    k_4d = reshape(k_out.data, batch, seq_len, a.n_kv_heads, a.head_dim)
    v_4d = reshape(v_out.data, batch, seq_len, a.n_kv_heads, a.head_dim)

    _ensure_attention_cache!(a, seq_len)

    # Apply RoPE on GPU
    _gpu_apply_rope!(q_4d, a.rope_cos_cache, a.rope_sin_cache, a.head_dim)
    _gpu_apply_rope!(k_4d, a.rope_cos_cache, a.rope_sin_cache, a.head_dim)

    if inference_mode()
        out_4d = _gpu_attention_inference(a, q_4d, k_4d, v_4d, a.causal_mask_cache, batch, seq_len)
        reshaped = reshape(out_4d, batch, seq_len, a.n_heads * a.head_dim)
        return gpu_forward(a.wO, MetalTensor(reshaped, q_out.dtype))
    end

    # Permute to (head_dim, seq, heads, batch) for per-head matmul
    q_perm = permutedims(q_4d, (4, 2, 3, 1))  # (head_dim, seq, heads, batch)
    k_perm = permutedims(k_4d, (4, 2, 3, 1))
    v_perm = permutedims(v_4d, (4, 2, 3, 1))

    if !inference_mode()
        a.last_q_perm = q_perm
        a.last_k_perm = k_perm
        a.last_v_perm = v_perm
    end

    head_dim = a.head_dim
    sc = a.scale
    n_heads = a.n_heads
    n_kv = a.n_kv_heads
    # Causal mask (seq, seq) on GPU
    causal_mask = a.causal_mask_cache

    # Per-head attention on GPU using 2D matmul slices
    # Accumulate output in (head_dim, seq, heads, batch) format
    out_perm = Metal.zeros(Float32, head_dim, seq_len, n_heads, batch)

    attn_weights_all = inference_mode() ? nothing : Metal.zeros(Float32, seq_len, seq_len, n_heads, batch)

    for b in 1:batch
        for h in 1:n_heads
            kv_h = ((h - 1) % n_kv) + 1

            q_h = @view q_perm[:, :, h, b]
            k_h = @view k_perm[:, :, kv_h, b]
            v_h = @view v_perm[:, :, kv_h, b]

            # scores = Q^T @ K * scale -> (seq_q, seq_k)
            scores = transpose(q_h) * k_h  # (seq, head_dim) @ (head_dim, seq) = (seq, seq) — MPS matmul
            scores = scores .* sc .+ causal_mask  # apply scale + causal mask

            # Softmax along dim 2 (key dimension) on GPU
            mx = maximum(scores; dims=2)  # (seq, 1)
            e = exp.(scores .- mx)
            s = sum(e; dims=2)  # (seq, 1)
            weights = e ./ s  # (seq, seq)

            if !inference_mode()
                attn_weights_all[:, :, h, b] .= weights
            end

            # Weighted sum: V @ W^T -> (head_dim, seq)
            # out = V_h @ weights^T = (head_dim, seq_k) @ (seq_q, seq_k)^T = (head_dim, seq_q)
            out_h = v_h * transpose(weights)  # (head_dim, seq)
            out_perm[:, :, h, b] .= out_h
        end
    end

    if !inference_mode()
        a.last_attn_weights = attn_weights_all
    end

    # Permute back to (batch, seq, heads, head_dim) and reshape
    out_4d = permutedims(out_perm, (4, 2, 3, 1))  # (batch, seq, heads, head_dim)
    reshaped = reshape(out_4d, batch, seq_len, n_heads * head_dim)
    gpu_forward(a.wO, MetalTensor(reshaped, q_out.dtype))
end

function gpu_backward(a::MetalMQAttention, grad_output::MetalTensor)
    batch = a.last_batch
    seq_len = a.last_seq_len
    head_dim = a.head_dim
    n_heads = a.n_heads
    n_kv = a.n_kv_heads
    sc = a.scale
    _ensure_attention_cache!(a, seq_len)
    causal = a.causal_binary_cache

    # Backward through W_o
    grad_o_input = gpu_backward(a.wO, grad_output)

    go_4d = reshape(grad_o_input.data, batch, seq_len, n_heads, head_dim)
    # Permute to (head_dim, seq, heads, batch)
    go_perm = permutedims(go_4d, (4, 2, 3, 1))

    # Gradient accumulators on GPU
    grad_q_perm = Metal.zeros(Float32, head_dim, seq_len, n_heads, batch)
    grad_k_perm = Metal.zeros(Float32, head_dim, seq_len, n_kv, batch)
    grad_v_perm = Metal.zeros(Float32, head_dim, seq_len, n_kv, batch)

    # Per-head backward on GPU
    for b in 1:batch
        for h in 1:n_heads
            kv_h = ((h - 1) % n_kv) + 1

            dO_h = @view go_perm[:, :, h, b]    # (head_dim, seq)
            Q_h = @view a.last_q_perm[:, :, h, b]   # (head_dim, seq)
            K_h = @view a.last_k_perm[:, :, kv_h, b] # (head_dim, seq)
            V_h = @view a.last_v_perm[:, :, kv_h, b] # (head_dim, seq)
            W_h = @view a.last_attn_weights[:, :, h, b]  # (seq, seq)

            # grad_V += dO @ W  (head_dim, seq) @ (seq, seq) = (head_dim, seq)
            gV_add = dO_h * W_h
            grad_v_perm[:, :, kv_h, b] .+= gV_add

            # grad_scores = dO^T . V = V @ dO^T => but we need (seq_q, seq_k)
            # grad_scores[qi, ki] = sum_d(dO[d, qi] * V[d, ki])
            grad_scores = transpose(dO_h) * V_h  # (seq, head_dim) @ (head_dim, seq) = (seq, seq)

            # Softmax backward: dS[qi,ki] = W[qi,ki] * (grad_scores[qi,ki] - sum_k(grad_scores[qi,k]*W[qi,k]))
            sum_gw = sum(grad_scores .* W_h; dims=2)  # (seq, 1)
            dS = W_h .* (grad_scores .- sum_gw)  # (seq, seq)

            # Apply causal mask to dS (zero out upper triangle)
            dS = dS .* causal

            # grad_Q: K @ dS^T * scale -> (head_dim, seq_k) @ (seq_q, seq_k)^T = (head_dim, seq_q)
            gQ_add = K_h * transpose(dS) .* sc
            grad_q_perm[:, :, h, b] .= gQ_add

            # grad_K: Q @ dS * scale -> (head_dim, seq_q) @ (seq_q, seq_k) = (head_dim, seq_k)
            gK_add = Q_h * dS .* sc
            grad_k_perm[:, :, kv_h, b] .+= gK_add
        end
    end

    # Permute gradients back to (batch, seq, heads, head_dim) then reshape
    grad_q_out = reshape(permutedims(grad_q_perm, (4, 2, 3, 1)), batch, seq_len, n_heads * head_dim)
    grad_k_out = reshape(permutedims(grad_k_perm, (4, 2, 3, 1)), batch, seq_len, n_kv * head_dim)
    grad_v_out = reshape(permutedims(grad_v_perm, (4, 2, 3, 1)), batch, seq_len, n_kv * head_dim)

    # Backward through Q, K, V projections
    grad_x_q = gpu_backward(a.wQ, MetalTensor(grad_q_out, grad_output.dtype))
    grad_x_k = gpu_backward(a.wK, MetalTensor(grad_k_out, grad_output.dtype))
    grad_x_v = gpu_backward(a.wV, MetalTensor(grad_v_out, grad_output.dtype))

    add_in_place_mtl!(grad_x_q, grad_x_k)
    add_in_place_mtl!(grad_x_q, grad_x_v)
    return grad_x_q
end

function gpu_parameters(a::MetalMQAttention)
    vcat(gpu_parameters(a.wQ), gpu_parameters(a.wK), gpu_parameters(a.wV), gpu_parameters(a.wO))
end
