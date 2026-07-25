# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# attention.jl — Multi-Query Attention with Rotary Position Embeddings (RoPE)
#
# Implements Multi-Query Attention (MQA), a memory-efficient variant where
# K and V have fewer heads than Q (n_kv_heads < n_heads). Multiple Q heads
# share a single KV head via mod1(h, n_kv_heads).
#
# Attention: scores = Q @ K^T / sqrt(d_k), weights = softmax(scores + mask), output = weights @ V
#
# RoPE: Rotary Position Embeddings encode position by rotating pairs of
# dimensions using sinusoidal functions, enabling relative position awareness
# without explicit position embeddings.
#
# Zero-alloc strategy:
#   All intermediate buffers (q_perm_buf, k_perm_buf, v_perm_buf, out_perm_buf,
#   output_perm_buf, scores_buf) are stored as mutable struct fields.
#   They are allocated on first use and reused on subsequent calls, giving
#   0 GC pauses in steady-state forward passes.

mutable struct MQAttention <: AbstractLayer
    wQ::Linear; wK::Linear; wV::Linear; wO::Linear
    n_heads::Int; n_kv_heads::Int; head_dim::Int
    hidden_dim::Int; scale::Float32     # scale = 1/sqrt(head_dim)
    freqs::Vector{Float32}              # pre-computed RoPE frequency table
    # Pre-allocated workspace buffers
    scores_buf::Vector{Float32}         # flattened (seq_len * seq_len) for attention scores
    scores_buf_len::Int
    # Permuted dimension buffers for zero-alloc forward
    q_perm_buf::Array{Float32,4}
    k_perm_buf::Array{Float32,4}
    v_perm_buf::Array{Float32,4}
    out_perm_buf::Array{Float32,4}
    output_perm_buf::Array{Float32,4}
    # Backward: cached data from forward
    last_attn_weights::Union{Vector{Float32}, Nothing}  # all heads' weights (batch*n_heads*seq*seq)
    last_batch::Int
    last_seq_len::Int
    # Backward pre-allocated buffers
    grad_q_buf::Array{Float32,4}       # (head_dim, n_heads, seq, batch)
    grad_k_buf::Array{Float32,4}       # (head_dim, n_kv, seq, batch)
    grad_v_buf::Array{Float32,4}       # (head_dim, n_kv, seq, batch)
    go_perm_buf::Array{Float32,4}      # (head_dim, n_heads, seq, batch)
    grad_w_row_buf::Vector{Float32}    # (seq_len,)
end

function MQAttention(hidden_dim::Int, n_heads::Int, n_kv_heads::Int, head_dim::Int,
                     rope_base::Float32, rope_alpha::Float32)
    # RoPE frequency computation with optional NTK-aware scaling:
    # base' = base * alpha^(d / (d-2))  when alpha > 1
    # freq_i = 1 / base'^(2i / d)  for i = 0, 1, ..., d/2-1
    base = rope_base
    if rope_alpha > 1f0
        base = rope_base * (rope_alpha ^ (Float32(head_dim) / Float32(head_dim - 2)))
    end
    half_dim = head_dim ÷ 2
    freqs = Float32[1f0 / (base ^ (Float32(2 * i) / Float32(head_dim))) for i in 0:half_dim-1]
    MQAttention(
        Linear(hidden_dim, n_heads * head_dim, false),
        Linear(hidden_dim, n_kv_heads * head_dim, false),
        Linear(hidden_dim, n_kv_heads * head_dim, false),
        Linear(n_heads * head_dim, hidden_dim, false),
        n_heads, n_kv_heads, head_dim, hidden_dim,
        1f0 / sqrt(Float32(head_dim)), freqs,
        Float32[], 0,
        Array{Float32}(undef, 0, 0, 0, 0), Array{Float32}(undef, 0, 0, 0, 0),
        Array{Float32}(undef, 0, 0, 0, 0), Array{Float32}(undef, 0, 0, 0, 0),
        Array{Float32}(undef, 0, 0, 0, 0),
        nothing, 0, 0,  # last_attn_weights, last_batch, last_seq_len
        Array{Float32}(undef, 0, 0, 0, 0),  # grad_q_buf
        Array{Float32}(undef, 0, 0, 0, 0),  # grad_k_buf
        Array{Float32}(undef, 0, 0, 0, 0),  # grad_v_buf
        Array{Float32}(undef, 0, 0, 0, 0),  # go_perm_buf
        Float32[]                             # grad_w_row_buf
    )
end

# RoPE: [x0, x1] -> [x0*cos(theta) - x1*sin(theta), x0*sin(theta) + x1*cos(theta)]
# Applied in-place to pairs of dimensions (2i, 2i+1) for each position.
# theta = position * freq_i, where freq_i = 1 / base^(2i/d)
function _rotate_half!(data::Array{Float32,4}, freqs::Vector{Float32},
                       batch::Int, seq_len::Int, n_heads::Int, head_dim::Int)
    half_dim = head_dim ÷ 2
    @inbounds for b in 1:batch, s in 1:seq_len
        pos = Float32(s - 1)  # 0-indexed position
        for h in 1:n_heads
            for i in 0:half_dim-1
                angle = pos * freqs[i + 1]
                c, sn = cos(angle), sin(angle)
                x0 = data[b, s, h, 2*i + 1]
                x1 = data[b, s, h, 2*i + 2]
                data[b, s, h, 2*i + 1] = x0 * c - x1 * sn
                data[b, s, h, 2*i + 2] = x0 * sn + x1 * c
            end
        end
    end
end

# Apply RoPE to both Q and K tensors in-place.
function _apply_rope!(q_data::Array{Float32,4}, k_data::Array{Float32,4},
                      freqs, batch, seq_len, q_heads, k_heads, head_dim)
    _rotate_half!(q_data, freqs, batch, seq_len, q_heads, head_dim)
    _rotate_half!(k_data, freqs, batch, seq_len, k_heads, head_dim)
end

# Multi-Query Attention forward pass:
#   1. Project input to Q, K, V via Linear layers
#   2. Reshape to (batch, seq, heads, head_dim) and apply RoPE
#   3. Permute to (head_dim, heads, seq, batch) for cache-friendly dot products
#   4. Compute causal attention: scores = Q @ K^T / sqrt(d_k)
#   5. Apply causal mask (upper triangle = -inf)
#   6. Softmax over valid positions (row-wise, only up to current position)
#   7. Weighted sum: output = weights @ V
#   8. Permute back and project through wO
function forward(a::MQAttention, input::Tensor)
    dims = size(input.data)
    batch = dims[1]
    seq_len = dims[2]
    a.last_batch = batch
    a.last_seq_len = seq_len
    q_out = forward(a.wQ, input)
    k_out = forward(a.wK, input)
    v_out = forward(a.wV, input)
    q = Tensor(reshape(q_out.data, batch, seq_len, a.n_heads, a.head_dim), q_out.dtype)
    k = Tensor(reshape(k_out.data, batch, seq_len, a.n_kv_heads, a.head_dim), k_out.dtype)
    v = Tensor(reshape(v_out.data, batch, seq_len, a.n_kv_heads, a.head_dim), v_out.dtype)

    _apply_rope!(q.data, k.data, a.freqs, batch, seq_len, a.n_heads, a.n_kv_heads, a.head_dim)

    # Permute to (head_dim, heads, seq, batch) — puts head_dim as the innermost
    # (fastest-varying) dimension in column-major layout, making the dot product
    # over d a contiguous memory access pattern.
    perm = (4, 3, 2, 1)
    q_perm_sz = (a.head_dim, a.n_heads, seq_len, batch)
    k_perm_sz = (a.head_dim, a.n_kv_heads, seq_len, batch)
    v_perm_sz = k_perm_sz

    # Reuse permuted buffers (allocate only on shape change)
    if size(a.q_perm_buf) != q_perm_sz
        a.q_perm_buf = Array{Float32}(undef, q_perm_sz...)
    end
    if size(a.k_perm_buf) != k_perm_sz
        a.k_perm_buf = Array{Float32}(undef, k_perm_sz...)
    end
    if size(a.v_perm_buf) != v_perm_sz
        a.v_perm_buf = Array{Float32}(undef, v_perm_sz...)
    end
    out_perm_sz = (a.head_dim, a.n_heads, seq_len, batch)
    if size(a.out_perm_buf) != out_perm_sz
        a.out_perm_buf = Array{Float32}(undef, out_perm_sz...)
    end

    permutedims!(a.q_perm_buf, q.data, perm)
    permutedims!(a.k_perm_buf, k.data, perm)
    permutedims!(a.v_perm_buf, v.data, perm)
    fill!(a.out_perm_buf, 0f0)

    q_p = a.q_perm_buf
    k_p = a.k_perm_buf
    v_p = a.v_perm_buf
    out_p = a.out_perm_buf

    # Reuse or resize flat scores buffer (seq_len^2 per head per batch)
    needed = seq_len * seq_len
    if a.scores_buf_len < needed
        resize!(a.scores_buf, needed)
        a.scores_buf_len = needed
    end
    scores = a.scores_buf

    head_dim = a.head_dim
    sc = a.scale               # 1/sqrt(d_k)
    n_kv = a.n_kv_heads
    neg_inf = -Float32(3.4028235e38)

    # Pre-allocate buffer for saving attention weights (needed for backward)
    aw_needed = batch * a.n_heads * seq_len * seq_len
    if a.last_attn_weights === nothing || length(a.last_attn_weights) != aw_needed
        a.last_attn_weights = Vector{Float32}(undef, aw_needed)
    end
    aw_buf = a.last_attn_weights::Vector{Float32}

    # Attention: scores_ij = (Q_i . K_j) / sqrt(d_k)
    # Causal mask: scores_ij = -inf for j > i (future positions)
    # MQ: K/V head index = mod1(q_head, n_kv_heads) — multiple Q heads share one KV head
    @inbounds for b in 1:batch, h in 1:a.n_heads
        kv_h = mod1(h, n_kv)  # map Q head -> shared KV head
        # Compute dot products for each (query_pos, key_pos) pair
        for qi in 1:seq_len
            for ki in 1:qi     # causal: only attend to positions <= qi
                dot = 0f0
                @simd for d in 1:head_dim
                    dot += q_p[d, h, qi, b] * k_p[d, kv_h, ki, b]
                end
                scores[(qi - 1) * seq_len + ki] = dot * sc
            end
            # Mask future positions with -inf (excluded from softmax)
            for ki in qi+1:seq_len
                scores[(qi - 1) * seq_len + ki] = neg_inf
            end
        end
        # Softmax over valid (causal) positions for each query
        for qi in 1:seq_len
            s_start = (qi - 1) * seq_len + 1
            s_end = (qi - 1) * seq_len + qi
            softmax_in_place!(@view scores[s_start:s_end])
        end
        # Save attention weights for backward pass
        aw_off = ((b-1)*a.n_heads + (h-1)) * seq_len * seq_len
        for qi in 1:seq_len
            for ki in 1:qi
                aw_buf[aw_off + (qi-1)*seq_len + ki] = scores[(qi - 1) * seq_len + ki]
            end
            for ki in qi+1:seq_len
                aw_buf[aw_off + (qi-1)*seq_len + ki] = 0f0
            end
        end
        # Weighted sum: output_d = sum_k(weight_k * V_k_d)
        for qi in 1:seq_len
            for ki in 1:qi
                w = scores[(qi - 1) * seq_len + ki]
                @simd for d in 1:head_dim
                    out_p[d, h, qi, b] += w * v_p[d, kv_h, ki, b]
                end
            end
        end
    end

    # Permute back to (batch, seq, heads, head_dim) — inverse of perm (4,3,2,1) is itself
    inv_perm = (4, 3, 2, 1)
    out_sz = (batch, seq_len, a.n_heads, a.head_dim)
    if size(a.output_perm_buf) != out_sz
        a.output_perm_buf = Array{Float32}(undef, out_sz...)
    end
    permutedims!(a.output_perm_buf, out_p, inv_perm)
    reshaped = reshape(a.output_perm_buf, batch, seq_len, a.n_heads * a.head_dim)
    output = Tensor(reshaped, q.dtype)
    forward(a.wO, output)  # final linear projection back to hidden_dim
end

# Full attention backward: propagates through wO, attention weights, softmax,
# score scaling, and Q/K/V projections. RoPE backward is omitted (frozen embeddings).
# Uses BLAS mul! for per-head matrix operations.
function backward(a::MQAttention, grad_output::Tensor)
    batch = a.last_batch
    seq_len = a.last_seq_len
    head_dim = a.head_dim
    n_heads = a.n_heads
    n_kv = a.n_kv_heads
    sc = a.scale

    # 1. Backward through W_o (accumulates wO weight gradients)
    grad_o_input = backward(a.wO, grad_output)

    # Reshape grad_o to (batch, seq, n_heads, head_dim) then permute to
    # (head_dim, n_heads, seq, batch) — matches perm_buf layout for contiguous d-axis access
    go_4d = reshape(grad_o_input.data, batch, seq_len, n_heads, head_dim)
    go_sz = (head_dim, n_heads, seq_len, batch)
    if size(a.go_perm_buf) != go_sz
        a.go_perm_buf = Array{Float32}(undef, go_sz...)
    end
    permutedims!(a.go_perm_buf, go_4d, (4, 3, 2, 1))
    go_perm = a.go_perm_buf

    # Cached data from forward (still valid since backward is called before next forward)
    q_p = a.q_perm_buf::Array{Float32,4}   # (head_dim, n_heads, seq, batch)
    k_p = a.k_perm_buf::Array{Float32,4}   # (head_dim, n_kv, seq, batch)
    v_p = a.v_perm_buf::Array{Float32,4}   # (head_dim, n_kv, seq, batch)
    aw = a.last_attn_weights::Vector{Float32}

    # Gradients in permuted layout — reuse pre-allocated buffers
    q_sz = (head_dim, n_heads, seq_len, batch)
    if size(a.grad_q_buf) != q_sz
        a.grad_q_buf = Array{Float32}(undef, q_sz...)
    end
    fill!(a.grad_q_buf, 0f0)
    grad_q = a.grad_q_buf

    kv_sz = (head_dim, n_kv, seq_len, batch)
    if size(a.grad_k_buf) != kv_sz
        a.grad_k_buf = Array{Float32}(undef, kv_sz...)
    end
    fill!(a.grad_k_buf, 0f0)
    grad_k = a.grad_k_buf

    if size(a.grad_v_buf) != kv_sz
        a.grad_v_buf = Array{Float32}(undef, kv_sz...)
    end
    fill!(a.grad_v_buf, 0f0)
    grad_v = a.grad_v_buf

    # Scratch buffers for BLAS operations
    grad_scores_mat = Matrix{Float32}(undef, seq_len, seq_len)
    w_mat = Matrix{Float32}(undef, seq_len, seq_len)

    @inbounds for b in 1:batch, h in 1:n_heads
        kv_h = mod1(h, n_kv)
        w_off = ((b - 1) * n_heads + (h - 1)) * seq_len * seq_len

        # Extract contiguous W matrix from flat aw buffer (row-major -> column-major)
        for qi in 1:seq_len
            for ki in 1:seq_len
                w_mat[qi, ki] = aw[w_off + (qi-1)*seq_len + ki]
            end
        end

        # Per-head views (column-major contiguous on first dim)
        dO_h = @view go_perm[:, h, :, b]    # [D, S]
        Q_h = @view q_p[:, h, :, b]          # [D, S]
        K_h = @view k_p[:, kv_h, :, b]       # [D, S]
        V_h = @view v_p[:, kv_h, :, b]       # [D, S]
        gQ_h = @view grad_q[:, h, :, b]      # [D, S]
        gK_h = @view grad_k[:, kv_h, :, b]   # [D, S]
        gV_h = @view grad_v[:, kv_h, :, b]   # [D, S]

        # 2. grad_V += dO @ W  (col-major: [D,S] += [D,S] * [S,S])
        mul!(gV_h, dO_h, w_mat, 1f0, 1f0)

        # 3. grad_scores = transpose(dO) * V  → [S,S] = [S,D] * [D,S]
        mul!(grad_scores_mat, transpose(dO_h), V_h)

        # 4. Softmax backward (element-wise with causal mask)
        for qi in 1:seq_len
            sum_gw_w = 0f0
            for ki in 1:qi
                sum_gw_w += grad_scores_mat[qi, ki] * w_mat[qi, ki]
            end
            for ki in 1:qi
                grad_scores_mat[qi, ki] = w_mat[qi, ki] * (grad_scores_mat[qi, ki] - sum_gw_w)
            end
            for ki in qi+1:seq_len
                grad_scores_mat[qi, ki] = 0f0
            end
        end

        # 5. grad_Q = K @ transpose(grad_scores) * scale → [D,S] = [D,S] * [S,S]
        mul!(gQ_h, K_h, transpose(grad_scores_mat), sc, 0f0)

        # 6. grad_K += Q @ grad_scores * scale → [D,S] += [D,S] * [S,S]
        mul!(gK_h, Q_h, grad_scores_mat, sc, 1f0)
    end

    # Permute gradients back to (batch, seq, heads, head_dim) and reshape to (batch, seq, heads*head_dim)
    grad_q_out = reshape(permutedims(grad_q, (4, 3, 2, 1)), batch, seq_len, n_heads * head_dim)
    grad_k_out = reshape(permutedims(grad_k, (4, 3, 2, 1)), batch, seq_len, n_kv * head_dim)
    grad_v_out = reshape(permutedims(grad_v, (4, 3, 2, 1)), batch, seq_len, n_kv * head_dim)

    # 7. Backward through Q, K, V projections (accumulates weight gradients)
    # wQ/wK/wV.last_input is already set from forward — no need to re-set
    grad_x_q = backward(a.wQ, Tensor(grad_q_out, grad_output.dtype))
    grad_x_k = backward(a.wK, Tensor(grad_k_out, grad_output.dtype))
    grad_x_v = backward(a.wV, Tensor(grad_v_out, grad_output.dtype))

    # Sum gradients from all projection paths (in-place to avoid allocation)
    add_in_place!(grad_x_q, grad_x_k)
    add_in_place!(grad_x_q, grad_x_v)
    return grad_x_q
end

function parameters(a::MQAttention)
    Iterators.flatten((parameters(a.wQ), parameters(a.wK), parameters(a.wV), parameters(a.wO)))
end
