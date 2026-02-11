# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# moe.jl — Mixture of Experts (MoE) routing and dispatch
#
# MoE: output = sum_k(gate_k * Expert_k(x)) for top-k experts
#
# The Router selects the top-k experts per token via softmax gating.
# The MoELayer dispatches tokens to their selected experts, runs each
# expert's SwiGLU, and accumulates weighted results.
#
# Auxiliary load-balancing loss encourages uniform expert utilization:
#   L_aux = alpha * N * sum_e(f_e * P_e)
#   where f_e = fraction of tokens routed to expert e,
#         P_e = mean gate probability for expert e,
#         N   = number of experts.

mutable struct Router <: AbstractLayer
    gate::Linear              # (hidden_dim -> n_experts) scoring projection
    n_experts::Int
    top_k::Int
    last_input::Union{Tensor,Nothing}
    last_weights::Union{Tensor,Nothing}
    last_indices::Vector{Vector{Int}}
    last_gate_prob::Union{Tensor,Nothing}
    # Pre-allocated buffers
    weights_buf::Matrix{Float32}                   # (num_tokens, top_k)
    softmax_buf::Matrix{Float32}                   # (num_tokens, n_experts)
    indices_buf::Vector{Vector{Int}}
    perm_buf::Vector{Int}
    grad_buf::Vector{Float32}                      # pre-allocated gradient buffer for backward
    # Pre-allocated buffers for compute_aux_loss
    aux_counts_buf::Vector{Float32}
    aux_probs_buf::Vector{Float32}
end

function Router(hidden_dim::Int, n_experts::Int, top_k::Int)
    @assert 1 <= top_k <= n_experts "invalid topK for router"
    Router(Linear(hidden_dim, n_experts, false), n_experts, top_k,
           nothing, nothing, Vector{Int}[], nothing,
           Matrix{Float32}(undef, 0, 0), Matrix{Float32}(undef, 0, 0),
           Vector{Int}[], Int[], Float32[],
           Float32[], Float32[])
end

# Router forward: gate_probs = softmax(W_gate @ x), select top-k experts per token
function forward(r::Router, input::Tensor)
    r.last_input = input
    dims = size(input.data)
    _, num_tokens, feat_dim = _split_last(dims)
    flat_data = reshape(input.data, num_tokens, feat_dim)
    flat_input = Tensor(flat_data, input.dtype)

    # Gate forward (Linear already uses buffer)
    gate_out = forward(r.gate, flat_input)

    # In-place softmax using pre-allocated buffer
    _router_softmax!(r, gate_out)
end

function _router_softmax!(r::Router, gate_out::Tensor)
    _router_softmax_inner!(r, gate_out.data, gate_out.dtype)
end

# Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# Then select top-k experts per token and normalize their weights to sum to 1.
function _router_softmax_inner!(r::Router, gd::Array{Float32,N}, dtype::DType) where {N}
    gate_sz = size(gd)
    num_tokens = gate_sz[1]
    n_exp = gate_sz[2]
    if size(r.softmax_buf) != (num_tokens, n_exp)
        r.softmax_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    sm_buf = r.softmax_buf
    gd_flat = reshape(gd, num_tokens, n_exp)
    softmax!(sm_buf, gd_flat)
    r.last_gate_prob = Tensor(sm_buf, dtype)

    # Weights buffer for top-k selected expert weights
    w_sz = (num_tokens, r.top_k)
    if size(r.weights_buf) != w_sz
        r.weights_buf = Matrix{Float32}(undef, w_sz...)
    end
    wb = r.weights_buf
    fill!(wb, 0f0)

    # Indices buffer — resize outer vector, reuse inner vectors
    top_k = r.top_k
    resize!(r.indices_buf, num_tokens)
    for i in 1:num_tokens
        if !isassigned(r.indices_buf, i)
            r.indices_buf[i] = Vector{Int}(undef, top_k)
        elseif length(r.indices_buf[i]) != top_k
            resize!(r.indices_buf[i], top_k)
        end
    end

    # Reuse permutation buffer for top-k selection (avoids partialsortperm allocation)
    if length(r.perm_buf) < n_exp
        r.perm_buf = Vector{Int}(undef, n_exp)
    end
    perm = r.perm_buf

    for t in 1:num_tokens
        # Manual top-k selection: insertion sort of top-k indices by descending value.
        # For small n_exp (4-16) and top_k (2-4), this is faster than partialsortperm
        # because it avoids all allocation.
        @inbounds for e in 1:n_exp
            perm[e] = e
        end
        # Partial selection sort: find top-k largest values
        @inbounds for k in 1:top_k
            best = k
            for j in k+1:n_exp
                if sm_buf[t, perm[j]] > sm_buf[t, perm[best]]
                    best = j
                end
            end
            if best != k
                perm[k], perm[best] = perm[best], perm[k]
            end
            idx_0based = perm[k] - 1  # store as 0-based for consistency with other langs
            r.indices_buf[t][k] = idx_0based
            wb[t, k] = sm_buf[t, perm[k]]
        end
        # Normalize top-k weights to sum to 1 (L1 normalization)
        token_weights = @view wb[t, :]
        normalize_in_place!(token_weights)
    end

    r.last_indices = r.indices_buf
    weights = Tensor(wb, F32)
    r.last_weights = weights
    weights, r.last_indices
end

function backward(r::Router, grad_output::Tensor)
    sz = size(r.last_input.data)
    total = prod(sz)
    if length(r.grad_buf) != total
        r.grad_buf = Vector{Float32}(undef, total)
    end
    fill!(r.grad_buf, 0f0)
    Tensor(reshape(r.grad_buf, sz), grad_output.dtype)
end

parameters(r::Router) = parameters(r.gate)

# Auxiliary load-balancing loss:
#   L_aux = alpha * N * sum_e(f_e * P_e)
#   f_e = (tokens routed to expert e) / (total token-expert assignments)
#   P_e = (sum of gate probs for expert e) / num_tokens
# Encourages uniform expert utilization; minimized when all experts receive
# equal traffic and equal probability mass.
function compute_aux_loss(r::Router, alpha::Float32)
    r.last_gate_prob === nothing && return 0f0
    num_tokens = size(r.last_gate_prob.data, 1)
    n_exp = r.n_experts
    top_k = r.top_k
    # Reuse pre-allocated buffers for expert counts and probs
    if length(r.aux_counts_buf) != n_exp
        r.aux_counts_buf = Vector{Float32}(undef, n_exp)
    end
    if length(r.aux_probs_buf) != n_exp
        r.aux_probs_buf = Vector{Float32}(undef, n_exp)
    end
    expert_counts = r.aux_counts_buf
    expert_probs = r.aux_probs_buf
    fill!(expert_counts, 0f0)
    fill!(expert_probs, 0f0)
    gp = r.last_gate_prob.data
    @inbounds for t in 1:num_tokens
        for k in 1:top_k
            expert_counts[r.last_indices[t][k] + 1] += 1f0  # 0-based -> 1-based
        end
        @simd for e in 1:n_exp
            expert_probs[e] += gp[t, e]
        end
    end
    total_assign = Float32(num_tokens * top_k)
    inv_total = 1f0 / total_assign
    inv_nt = 1f0 / Float32(num_tokens)
    aux_loss = 0f0
    @inbounds @simd for e in 1:n_exp
        aux_loss += (expert_counts[e] * inv_total) * (expert_probs[e] * inv_nt)
    end
    aux_loss * alpha * Float32(n_exp)
end

# =============================================================================
# MoELayer — Full Mixture of Experts layer
# =============================================================================

mutable struct MoELayer <: AbstractLayer
    router::Router
    experts::Vector{SwiGLU}
    hidden_dim::Int
    n_experts::Int
    top_k::Int
    # Pre-allocated buffers
    output_buf::Matrix{Float32}
    expert_batch_bufs::Vector{Matrix{Float32}}  # per-expert batch buffers
    expert_tokens::Vector{Vector{Int}}       # which tokens go to each expert
    expert_weight_idx::Vector{Vector{Int}}   # which top-k slot each token uses
    # Backward: cached data from forward
    last_flat_data::Matrix{Float32}  # (num_tokens, hidden_dim) saved for backward
    last_num_tokens::Int
    last_leading::Tuple  # leading dims for output reshape
    # Backward pre-allocated buffers
    grad_input_buf::Matrix{Float32}       # (num_tokens, hidden_dim) — reused each backward
    bwd_expert_grad_buf::Matrix{Float32}  # (max_tok_per_expert, hidden_dim)
    bwd_expert_input_buf::Matrix{Float32} # (max_tok_per_expert, hidden_dim)
end

function MoELayer(hidden_dim::Int, ffn_dim::Int, n_experts::Int, top_k::Int)
    experts = [SwiGLU(hidden_dim, ffn_dim) for _ in 1:n_experts]
    et = [Int[] for _ in 1:n_experts]
    ewi = [Int[] for _ in 1:n_experts]
    ebb = [Matrix{Float32}(undef, 0, 0) for _ in 1:n_experts]
    MoELayer(Router(hidden_dim, n_experts, top_k), experts, hidden_dim, n_experts, top_k,
             Matrix{Float32}(undef, 0, 0), ebb, et, ewi,
             Matrix{Float32}(undef, 0, 0), 0, (),  # last_flat_data, last_num_tokens, last_leading
             Matrix{Float32}(undef, 0, hidden_dim),   # grad_input_buf
             Matrix{Float32}(undef, 0, hidden_dim),   # bwd_expert_grad_buf
             Matrix{Float32}(undef, 0, hidden_dim))   # bwd_expert_input_buf
end

# MoE forward: output = sum_k(gate_k * Expert_k(x)) for top-k experts per token
# Steps:
#   1. Router selects top-k experts + weights per token
#   2. Group tokens by expert (build per-expert batch)
#   3. Run each expert's SwiGLU on its batch
#   4. Accumulate weighted expert outputs into final result
function forward(m::MoELayer, input::Tensor)
    _moelayer_forward!(m, input.data, input.dtype)
end

function _moelayer_forward!(m::MoELayer, id::Array{Float32,N}, dtype::DType) where {N}
    dims = size(id)
    leading, num_tokens, _ = _split_last(dims)
    input_tensor = Tensor(id, dtype)

    weights, indices = forward(m.router, input_tensor)
    # Extract concrete types from Router output for type stability
    wd = m.router.weights_buf
    flat_data = reshape(id, num_tokens, m.hidden_dim)

    # Save flat input data for backward (copy because input view may be mutated)
    if size(m.last_flat_data) != (num_tokens, m.hidden_dim)
        m.last_flat_data = Matrix{Float32}(undef, num_tokens, m.hidden_dim)
    end
    copyto!(m.last_flat_data, flat_data)
    m.last_num_tokens = num_tokens
    m.last_leading = leading

    # Ensure output buffer
    hidden_dim = m.hidden_dim
    out_sz = (num_tokens, hidden_dim)
    if size(m.output_buf) != out_sz
        m.output_buf = Matrix{Float32}(undef, out_sz...)
    end
    od = m.output_buf
    fill!(od, 0f0)
    fid = flat_data
    n_experts = m.n_experts
    top_k = m.top_k

    # Build per-expert token lists (which tokens -> which expert)
    for e in 1:n_experts
        empty!(m.expert_tokens[e])
        empty!(m.expert_weight_idx[e])
    end

    @inbounds for t in 1:num_tokens
        for k in 1:top_k
            e_idx = indices[t][k]             # 0-based expert index
            push!(m.expert_tokens[e_idx + 1], t)
            push!(m.expert_weight_idx[e_idx + 1], k)
        end
    end

    # Run each expert on its assigned tokens
    for e_idx in 1:n_experts
        tokens = m.expert_tokens[e_idx]
        isempty(tokens) && continue
        n_tok = length(tokens)

        # Per-expert batch buffer (grow-only: never shrinks to avoid reallocation)
        eb = m.expert_batch_bufs[e_idx]
        if size(eb, 1) < n_tok || size(eb, 2) != hidden_dim
            m.expert_batch_bufs[e_idx] = Matrix{Float32}(undef, max(n_tok, size(eb, 1)), hidden_dim)
        end
        bb = m.expert_batch_bufs[e_idx]

        # Gather: copy selected token vectors into contiguous expert batch
        @inbounds for (i, t) in enumerate(tokens)
            @simd for d in 1:hidden_dim
                bb[i, d] = fid[t, d]
            end
        end

        # Wrap buffer as Tensor — if buffer matches exactly, wrap directly (zero alloc).
        # Otherwise need a slice copy (this alloc is O(n_tok*hidden_dim)).
        if size(bb, 1) == n_tok
            batch_input = Tensor(bb, F32)
        else
            batch_data = bb[1:n_tok, :]  # unavoidable copy for mismatched size
            batch_input = Tensor(batch_data, F32)
        end

        expert_out = forward(m.experts[e_idx], batch_input)
        # Scatter-add: accumulate weighted expert output back to each token's position
        _moe_accumulate!(od, expert_out.data, tokens, m.expert_weight_idx[e_idx], wd, hidden_dim)
    end

    Tensor(reshape(od, leading..., hidden_dim), dtype)
end

# Scatter-add: output[t, :] += weight * expert_output[i, :] for each assigned token.
# Type-stable: dispatches on concrete Matrix{Float32}.
function _moe_accumulate!(od::Matrix{Float32}, eod::Array{Float32,N}, tokens::Vector{Int},
                          weight_idx::Vector{Int}, wd::Matrix{Float32}, hidden_dim::Int) where {N}
    eod_flat = reshape(eod, length(tokens), hidden_dim)
    @inbounds for (i, t) in enumerate(tokens)
        k = weight_idx[i]
        w = wd[t, k]
        @views @fastmath od[t, :] .+= w .* eod_flat[i, :]
    end
end

function backward(m::MoELayer, grad_output::Tensor)
    num_tokens = m.last_num_tokens
    hidden_dim = m.hidden_dim
    flat_grad = reshape(grad_output.data, num_tokens, hidden_dim)
    flat_x = m.last_flat_data
    wd = m.router.weights_buf  # (num_tokens, top_k)

    # Pre-allocate / reuse grad_input buffer
    if size(m.grad_input_buf) != (num_tokens, hidden_dim)
        m.grad_input_buf = Matrix{Float32}(undef, num_tokens, hidden_dim)
    end
    grad_input = m.grad_input_buf
    fill!(grad_input, 0f0)

    for e_idx in 1:m.n_experts
        tokens = m.expert_tokens[e_idx]
        isempty(tokens) && continue
        n_tok = length(tokens)
        weight_idx = m.expert_weight_idx[e_idx]

        # Reuse expert_grad buffer (grow-only)
        if size(m.bwd_expert_grad_buf, 1) < n_tok
            m.bwd_expert_grad_buf = Matrix{Float32}(undef, n_tok, hidden_dim)
        end
        expert_grad_data = m.bwd_expert_grad_buf

        @inbounds for (i, t) in enumerate(tokens)
            k = weight_idx[i]
            w = wd[t, k]
            @views @fastmath expert_grad_data[i, :] .= w .* flat_grad[t, :]
        end
        # Slice if buffer is larger than needed
        if size(expert_grad_data, 1) == n_tok
            expert_grad = Tensor(expert_grad_data, grad_output.dtype)
        else
            expert_grad = Tensor(expert_grad_data[1:n_tok, :], grad_output.dtype)
        end

        # Reuse expert_input buffer (grow-only)
        if size(m.bwd_expert_input_buf, 1) < n_tok
            m.bwd_expert_input_buf = Matrix{Float32}(undef, n_tok, hidden_dim)
        end
        expert_input_data = m.bwd_expert_input_buf

        @inbounds for (i, t) in enumerate(tokens)
            @views expert_input_data[i, :] .= flat_x[t, :]
        end
        if size(expert_input_data, 1) == n_tok
            expert_input = Tensor(expert_input_data, grad_output.dtype)
        else
            expert_input = Tensor(expert_input_data[1:n_tok, :], grad_output.dtype)
        end

        # Set cached input for expert sub-layers (gate, up need the original input)
        m.experts[e_idx].w_gate.last_input = expert_input
        m.experts[e_idx].w_up.last_input = expert_input

        # Backward through expert (accumulates weight gradients in SwiGLU sub-layers)
        grad_expert_input = backward(m.experts[e_idx], expert_grad)
        ge_data = grad_expert_input.data

        # Scatter-add: accumulate expert input gradient back to token positions
        ge_flat = reshape(ge_data, n_tok, hidden_dim)
        @inbounds for (i, t) in enumerate(tokens)
            @views @fastmath grad_input[t, :] .+= ge_flat[i, :]
        end
    end

    Tensor(reshape(grad_input, m.last_leading..., hidden_dim), grad_output.dtype)
end

function parameters(m::MoELayer)
    Iterators.flatten((parameters(m.router), Iterators.flatten(parameters.(m.experts))))
end

aux_loss(m::MoELayer, alpha::Float32) = compute_aux_loss(m.router, alpha)

# =============================================================================
# TransformerBlock — one transformer layer: Attn + MoE with residual connections
# =============================================================================
# Block structure (Pre-Norm):
#   x1 = x + Attention(RMSNorm(x))
#   x2 = x1 + MoE(RMSNorm(x1))
# Residual connections are computed in-place to avoid allocation.

mutable struct TransformerBlock <: AbstractLayer
    attn_norm::RMSNorm
    attention::MQAttention
    ffn_norm::RMSNorm
    moe::MoELayer
    residual_buf::Vector{Float32}  # shared buffer for both residual adds
end

function TransformerBlock(cfg::Config)
    TransformerBlock(
        RMSNorm(cfg.hidden_dim),
        MQAttention(cfg.hidden_dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                    cfg.rope_base, cfg.rope_alpha),
        RMSNorm(cfg.hidden_dim),
        MoELayer(cfg.hidden_dim, cfg.ffn_dim, cfg.n_experts, cfg.top_k_experts),
        Float32[]
    )
end

# Forward: x1 = x + Attn(Norm(x)), then x2 = x1 + MoE(Norm(x1))
function forward(blk::TransformerBlock, input::Tensor)
    attn_out = forward(blk.attention, forward(blk.attn_norm, input))
    _tfblock_residual!(blk, input.data, attn_out.data, input.dtype)
end

# First residual: x1 = input + attn_out, stored in residual_buf.
# Then MoE output is added in-place to the same buffer.
function _tfblock_residual!(blk::TransformerBlock, id::Array{Float32,N}, ad::Array{Float32,M}, dtype::DType) where {N,M}
    sz = size(id)
    total = length(id)
    if length(blk.residual_buf) != total
        blk.residual_buf = Vector{Float32}(undef, total)
    end
    rb = blk.residual_buf
    # Flatten for linear indexing — reshape of concrete Array is zero-copy
    id_flat = reshape(id, total)
    ad_flat = reshape(ad, total)
    @inbounds @simd for i in 1:total
        rb[i] = id_flat[i] + ad_flat[i]
    end
    x = Tensor(reshape(rb, sz), dtype)
    moe_out = forward(blk.moe, forward(blk.ffn_norm, x))
    # Second residual: x2 = x1 + moe_out — must create NEW tensor (not in-place)
    # because ffn_norm.last_input.data shares rb via x, and backward needs the
    # original x1 values to compute correct RMSNorm gradients.
    x + moe_out
end

function _tfblock_add_moe!(rb::Vector{Float32}, md::Array{Float32,N}, total::Int) where {N}
    md_flat = reshape(md, total)
    @inbounds @simd for i in 1:total
        rb[i] += md_flat[i]
    end
end

function backward(blk::TransformerBlock, grad_output::Tensor)
    # MoE residual path: ffn_norm → MoE → residual add
    grad_moe_input = backward(blk.moe, backward(blk.ffn_norm, grad_output))
    # Residual: grad_h = grad_output + grad_moe_input (in-place to avoid allocation)
    # Safe: RMSNorm backward has already finished reading grad_output.data
    add_in_place!(grad_output, grad_moe_input)

    # Attention residual path: attn_norm → Attention → residual add
    grad_attn_input = backward(blk.attention, backward(blk.attn_norm, grad_output))
    # Residual: grad_x = grad_output + grad_attn_input (in-place)
    add_in_place!(grad_output, grad_attn_input)

    return grad_output
end

function parameters(blk::TransformerBlock)
    Iterators.flatten((parameters(blk.attn_norm), parameters(blk.attention),
                       parameters(blk.ffn_norm), parameters(blk.moe)))
end

aux_loss(blk::TransformerBlock, alpha::Float32) = aux_loss(blk.moe, alpha)
