# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

# layers.jl — Neural network layer primitives
#
# Implements Embedding, Linear, RMSNorm, and SwiGLU layers.
# All layers follow the AbstractLayer interface: forward(), backward(), parameters().
#
# Zero-alloc forward pass pattern:
#   Each layer struct has mutable buffer fields (e.g., output_buf, matmul_buf).
#   On first call, buffers are allocated; on subsequent calls, they are reused
#   if the shape matches. This means the steady-state forward pass allocates 0
#   bytes and triggers 0 GC pauses.
#
# Multiple dispatch:
#   Public forward(layer, input::Tensor) delegates to an inner _*_forward!()
#   function that dispatches on Array{Float32,N}. This gives the compiler a
#   concrete type (not abstract Array{Float32}) for better type inference.

abstract type AbstractLayer end

# =============================================================================
# Embedding — lookup table: token_id -> dense vector
# =============================================================================

mutable struct Embedding <: AbstractLayer
    weight::Tensor          # (vocab_size, embed_dim)
    vocab_size::Int
    embed_dim::Int
    last_input::Vector{Int} # cached token IDs for backward pass
    output_buf::Array{Float32,3}          # pre-allocated (batch, seq, dim)
    grad_buf::Vector{Float32}              # pre-allocated gradient buffer for backward
end

# Xavier-like init: std = sqrt(2 / embed_dim)
function Embedding(vocab_size::Int, embed_dim::Int)
    std = sqrt(2f0 / Float32(embed_dim))
    Embedding(randn_std(vocab_size, embed_dim; std=std), vocab_size, embed_dim, Int[], Array{Float32}(undef, 0, 0, 0), Float32[])
end

function forward(e::Embedding, input::Tensor)
    _embedding_forward!(e, input.data)
end

function _embedding_forward!(e::Embedding, input_data::Array{Float32,N}) where {N}
    dims = size(input_data)
    batch, seq_len = dims[1], dims[2]
    n = batch * seq_len
    # Reuse last_input vector (resize! avoids reallocation if capacity suffices)
    resize!(e.last_input, n)
    idx = 0
    for b in 1:batch, s in 1:seq_len
        idx += 1
        @inbounds e.last_input[idx] = Int(input_data[b, s])
    end
    sz = (batch, seq_len, e.embed_dim)
    # Reuse output buffer if shape matches; allocate only on first call or shape change
    if size(e.output_buf) != sz
        e.output_buf = Array{Float32}(undef, sz...)
    end
    od = e.output_buf::Array{Float32, 3}
    w = e.weight.data::Matrix{Float32}
    idx = 0
    for b in 1:batch, s in 1:seq_len
        idx += 1
        tid = e.last_input[idx] + 1  # 0-based token ID -> 1-based Julia index
        @assert 1 <= tid <= e.vocab_size "token ID out of range"
        # Column-major: (vocab, dim) weight has stride=vocab_size along dim axis.
        # This loop reads w[tid, 1], w[tid, 2], ... with stride=vocab_size (not stride-1).
        @inbounds @simd for d in 1:e.embed_dim
            od[b, s, d] = w[tid, d]
        end
    end
    Tensor(od, F32)
end

function backward(e::Embedding, grad_output::Tensor)
    sz = size(grad_output.data)
    n = prod(sz)
    if length(e.grad_buf) != n
        e.grad_buf = zeros(Float32, n)
    else
        fill!(e.grad_buf, 0f0)
    end

    # Store weight gradient: scatter-add grad_output into weight.grad
    god = grad_output.data
    w_sz = size(e.weight.data)
    if e.weight.grad === nothing || size(e.weight.grad) != w_sz
        e.weight.grad = zeros(Float32, w_sz...)
    end
    wg = e.weight.grad::Matrix{Float32}
    dims = size(god)
    batch, seq_len, embed_dim = dims[1], dims[2], dims[3]
    idx = 0
    for b in 1:batch, s in 1:seq_len
        idx += 1
        tid = e.last_input[idx] + 1  # 0-based -> 1-based
        @inbounds @fastmath @simd for d in 1:embed_dim
            wg[tid, d] += god[b, s, d]
        end
    end

    Tensor(reshape(e.grad_buf, sz), grad_output.dtype)
end

parameters(e::Embedding) = Tensor[e.weight]

# =============================================================================
# Linear — fully connected layer: y = x @ W^T + b
# =============================================================================

mutable struct Linear <: AbstractLayer
    weight::Tensor                  # (out_feat, in_feat) — stored transposed for matmul_transposed_b
    bias::Union{Tensor,Nothing}
    in_feat::Int
    out_feat::Int
    use_bias::Bool
    last_input::Union{Tensor,Nothing}
    matmul_buf::Matrix{Float32}             # pre-allocated (batch, out_feat)
    grad_matmul_buf::Matrix{Float32}        # pre-allocated for backward matmul
end

# Xavier-like init: std = sqrt(2 / in_features)
function Linear(in_features::Int, out_features::Int, use_bias::Bool)
    std = sqrt(2f0 / Float32(in_features))
    w = randn_std(out_features, in_features; std=std)
    b = use_bias ? zeros_tensor(out_features) : nothing
    Linear(w, b, in_features, out_features, use_bias, nothing, Matrix{Float32}(undef, 0, 0), Matrix{Float32}(undef, 0, 0))
end

# Split N-D tensor dims into (leading_dims, leading_product, last_dim)
function _split_last(dims::NTuple{N,Int}) where {N}
    leading = ntuple(i -> dims[i], Val(N-1))
    leading_size = prod(leading)
    leading, leading_size, dims[N]
end

# Linear forward: y = x @ W^T (+ bias)
# Weight is (out, in), input is (..., in) -> output is (..., out).
# matmul_transposed_b! computes x @ W^T using sgemm with TRANSB='T'.
function forward(l::Linear, input::Tensor)
    l.last_input = input
    _linear_forward!(l, input.data, input.dtype)
end

function _linear_forward!(l::Linear, id::Array{Float32,N}, dtype::DType) where {N}
    dims = size(id)
    leading, batch_size, _ = _split_last(dims)
    flat = reshape(id, batch_size, l.in_feat)
    # Reuse matmul output buffer
    out_sz = (batch_size, l.out_feat)
    if size(l.matmul_buf) != out_sz
        l.matmul_buf = Matrix{Float32}(undef, out_sz...)
    end
    buf = l.matmul_buf::Matrix{Float32}
    w = l.weight.data::Matrix{Float32}
    matmul_transposed_b!(buf, flat, w)  # buf = flat @ W^T, dispatches to BLAS sgemm
    if l.use_bias
        b_d = l.bias.data::Vector{Float32}
        # Broadcast bias across batch dimension
        @inbounds for i in 1:batch_size
            @simd for j in 1:l.out_feat
                buf[i, j] += b_d[j]
            end
        end
    end
    Tensor(reshape(buf, leading..., l.out_feat), dtype)
end

function backward(l::Linear, grad_output::Tensor)
    l.last_input === nothing && error("backward called before forward")
    input_shape = size(l.last_input.data)
    dims = size(grad_output.data)
    _, batch_size, _ = _split_last(dims)
    flat_grad = reshape(grad_output.data, batch_size, l.out_feat)
    w = l.weight.data::Matrix{Float32}
    # grad_input = grad_output @ W  (undo the transpose from forward)
    # Reuse backward matmul buffer
    grad_sz = (batch_size, l.in_feat)
    if size(l.grad_matmul_buf) != grad_sz
        l.grad_matmul_buf = Matrix{Float32}(undef, grad_sz...)
    end
    buf = l.grad_matmul_buf::Matrix{Float32}
    tensor_matmul!(buf, flat_grad, w)  # buf = flat_grad @ W

    # Store weight gradient: grad_W = flat_grad^T @ flat_input
    # flat_grad is (batch_size, out_feat), flat_input is (batch_size, in_feat)
    # grad_W = flat_grad^T @ flat_input -> (out_feat, in_feat)
    flat_input = reshape(l.last_input.data, batch_size, l.in_feat)
    w_sz = size(w)
    if l.weight.grad === nothing || size(l.weight.grad) != w_sz
        l.weight.grad = Matrix{Float32}(undef, w_sz...)
    end
    wg = l.weight.grad::Matrix{Float32}
    mul!(wg, transpose(flat_grad), flat_input)  # BLAS: wg = flat_grad^T @ flat_input

    # Store bias gradient: grad_b = sum(flat_grad, dims=1), shape (out_feat,)
    if l.use_bias
        b_sz = size(l.bias.data)
        if l.bias.grad === nothing || size(l.bias.grad) != b_sz
            l.bias.grad = Vector{Float32}(undef, l.out_feat)
        end
        bg = l.bias.grad::Vector{Float32}
        fill!(bg, 0f0)
        @inbounds @fastmath for i in 1:batch_size
            @simd for j in 1:l.out_feat
                bg[j] += flat_grad[i, j]
            end
        end
    end

    Tensor(reshape(buf, input_shape...), grad_output.dtype)
end

function parameters(l::Linear)
    l.use_bias ? Tensor[l.weight, l.bias] : Tensor[l.weight]
end

# =============================================================================
# RMSNorm — Root Mean Square Layer Normalization
# RMSNorm: y = x * (1 / sqrt(mean(x^2) + eps)) * gamma
# Unlike LayerNorm, RMSNorm omits the mean-centering step, making it cheaper.
# =============================================================================

mutable struct RMSNorm <: AbstractLayer
    weight::Tensor            # gamma: learnable scale, (dim,)
    eps::Float32
    dim::Int
    last_input::Union{Tensor,Nothing}
    last_rms::Vector{Float32}  # cached RMS values for backward pass
    output_flat::Vector{Float32}           # flat 1D buffer, reshaped as needed
    output_len::Int
    grad_buf::Vector{Float32}              # pre-allocated gradient buffer for backward
end

function RMSNorm(dim::Int, eps::Float32=1f-6)
    RMSNorm(ones_tensor(dim), eps, dim, nothing, Float32[], Float32[], 0, Float32[])
end

function forward(r::RMSNorm, input::Tensor)
    r.last_input = input
    _rmsnorm_forward!(r, input.data, input.dtype)
end

# RMSNorm: y_i = x_i / rms(x) * gamma_i
#   where rms(x) = sqrt(mean(x^2) + eps)
#
# Type-stable inner function: dispatches on concrete Array{Float32,N} to avoid
# the performance penalty of abstract Array{Float32} in the method signature.
function _rmsnorm_forward!(r::RMSNorm, id::Array{Float32,N}, dtype::DType) where {N}
    sz = size(id)
    total = length(id)
    dim = r.dim
    eps = r.eps
    num_vectors = div(total, dim)

    # Reuse flat output buffer (reshape to match input shape at the end)
    if r.output_len != total
        r.output_flat = Vector{Float32}(undef, total)
        r.output_len = total
    end
    out_flat = r.output_flat::Vector{Float32}
    w_d = r.weight.data::Vector{Float32}

    resize!(r.last_rms, num_vectors)
    flat_in = reshape(id, num_vectors, dim)
    flat_out = reshape(out_flat, num_vectors, dim)
    for v in 1:num_vectors
        # Compute sum of squares
        sum_sq = 0f0
        @inbounds @simd for i in 1:dim
            x = flat_in[v, i]
            sum_sq += x * x
        end
        rms = sqrt(sum_sq / Float32(dim) + eps)
        r.last_rms[v] = rms  # cache for backward
        inv_rms = 1f0 / rms
        # y_i = x_i * (1/rms) * gamma_i
        @inbounds @simd for i in 1:dim
            flat_out[v, i] = flat_in[v, i] * inv_rms * w_d[i]
        end
    end
    # reshape is a view (zero-copy) — the flat buffer is shared
    Tensor(reshape(out_flat, sz), dtype)
end

# RMSNorm backward:
#   d_x_i = (d_y_i * gamma_i / rms) - x_i * (sum_j(d_y_j * gamma_j * x_j / rms)) / (dim * rms^3)
# This is the chain rule through the normalization + scale.
#
# Performance: Reshapes to (num_vectors, dim) so the inner loop over dim accesses
# contiguous memory (stride-1 in column-major). The original CartesianIndices
# approach had stride=batch*seq_len for the dim axis (non-contiguous).
function backward(r::RMSNorm, grad_output::Tensor)
    r.last_input === nothing && error("backward called before forward")
    _rmsnorm_backward!(r, grad_output.data, r.last_input.data, grad_output.dtype)
end

function _rmsnorm_backward!(r::RMSNorm, god::Array{Float32,N1}, lid::Array{Float32,N2}, dtype::DType) where {N1,N2}
    sz = size(god)
    total = length(god)
    dim = r.dim
    num_vectors = div(total, dim)

    if length(r.grad_buf) != total
        r.grad_buf = Vector{Float32}(undef, total)
    end
    gid_vec = r.grad_buf
    w_d = r.weight.data::Vector{Float32}

    # Reshape to (num_vectors, dim) — produces concrete Matrix{Float32}
    flat_god = reshape(god, num_vectors, dim)
    flat_lid = reshape(lid, num_vectors, dim)
    flat_gid = reshape(gid_vec, num_vectors, dim)

    # Store gamma gradient: grad_gamma_i = sum_v(grad_output[v,i] * x[v,i] / rms[v])
    w_sz = size(r.weight.data)
    if r.weight.grad === nothing || size(r.weight.grad) != w_sz
        r.weight.grad = zeros(Float32, w_sz...)
    end
    wg = r.weight.grad::Vector{Float32}

    for v in 1:num_vectors
        rms = r.last_rms[v]
        inv_rms = 1f0 / rms
        rms3 = rms * rms * rms
        # dot_sum = sum(d_y * gamma * x)
        dot_sum = 0f0
        @inbounds @fastmath @simd for i in 1:dim
            dot_sum += flat_god[v, i] * w_d[i] * flat_lid[v, i]
        end
        inv_dim_rms3 = 1f0 / (Float32(dim) * rms3)
        @inbounds @fastmath @simd for i in 1:dim
            flat_gid[v, i] = flat_god[v, i] * w_d[i] * inv_rms - flat_lid[v, i] * dot_sum * inv_dim_rms3
        end
        # Accumulate gamma gradient
        @inbounds @fastmath @simd for i in 1:dim
            wg[i] += flat_god[v, i] * flat_lid[v, i] * inv_rms
        end
    end
    Tensor(reshape(gid_vec, sz), dtype)
end

parameters(r::RMSNorm) = Tensor[r.weight]

# =============================================================================
# SwiGLU — Gated feed-forward with SiLU activation
# SwiGLU: output = down(silu(gate(x)) * up(x))
#   where silu(z) = z * sigmoid(z)
#
# Three separate Linear projections: gate, up, down.
# gate and up project from hidden_dim -> ffn_dim,
# down projects from ffn_dim -> hidden_dim.
# =============================================================================

mutable struct SwiGLU <: AbstractLayer
    w_gate::Linear
    w_up::Linear
    w_down::Linear
    hidden_dim::Int
    ffn_dim::Int
    last_gate::Union{Tensor,Nothing}
    last_up::Union{Tensor,Nothing}
    last_gate_pre_silu::Union{Array{Float32}, Nothing}  # gate output before silu (for silu derivative)
    last_silu_gate::Union{Array{Float32}, Nothing}      # silu(gate) before mul (for grad_up)
    grad_gate_buf::Vector{Float32}          # pre-allocated for grad w.r.t. gate
    grad_up_buf::Vector{Float32}            # pre-allocated for grad w.r.t. up
    grad_sum_buf::Vector{Float32}           # pre-allocated for accumulating gate + up grads
end

function SwiGLU(hidden_dim::Int, ffn_dim::Int)
    SwiGLU(
        Linear(hidden_dim, ffn_dim, false),
        Linear(hidden_dim, ffn_dim, false),
        Linear(ffn_dim, hidden_dim, false),
        hidden_dim, ffn_dim, nothing, nothing, nothing, nothing,
        Float32[], Float32[], Float32[]
    )
end

# SwiGLU forward: y = W_down @ (silu(W_gate @ x) * (W_up @ x))
# In-place ops (silu_in_place!, mul_in_place!) avoid 2 temporary allocations.
function forward(s::SwiGLU, input::Tensor)
    gate = forward(s.w_gate, input)
    # Cache pre-silu gate values for backward (silu derivative needs raw gate output)
    s.last_gate_pre_silu = copy(gate.data)
    silu_in_place!(gate)           # in-place silu avoids 1 allocation
    # Cache post-silu gate values before mul corrupts them (grad_up needs silu(gate))
    s.last_silu_gate = copy(gate.data)
    up = forward(s.w_up, input)
    s.last_up = up
    mul_in_place!(gate, up)        # gate *= up, in-place avoids 1 allocation
    forward(s.w_down, gate)        # down-project back to hidden_dim
end

function backward(s::SwiGLU, grad_output::Tensor)
    grad_hidden = backward(s.w_down, grad_output)
    ghd = grad_hidden.data
    sz = size(ghd)
    # grad w.r.t. silu(gate) output: d(silu(g)*u)/d(silu(g)) = u
    # Then apply silu derivative: grad_gate = (grad_hidden * up) * silu'(gate_pre_silu)
    #   silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    n = length(ghd)
    if length(s.grad_gate_buf) != n
        s.grad_gate_buf = Vector{Float32}(undef, n)
    end
    gg = s.grad_gate_buf
    lu = s.last_up.data
    gps = s.last_gate_pre_silu::Array{Float32}
    # Broadcast: fused silu derivative (zero-alloc, auto-vectorized)
    # Two-pass: first compute sigmoid, then full derivative expression
    # gg = sigmoid(gps)
    # gg = (ghd * lu) * gg * (1 + gps * (1 - gg))
    _swiglu_silu_grad!(gg, ghd, lu, gps)
    # grad_up = grad_hidden * silu(gate) (element-wise)
    # Use last_silu_gate (saved before mul_in_place! corrupted gate.data)
    if length(s.grad_up_buf) != n
        s.grad_up_buf = Vector{Float32}(undef, n)
    end
    gu = s.grad_up_buf
    sg = s.last_silu_gate::Array{Float32}
    _swiglu_grad_up!(gu, ghd, sg)
    grad_gate_t = Tensor(reshape(gg, sz), grad_output.dtype)
    grad_up_t = Tensor(reshape(gu, sz), grad_output.dtype)
    g1 = backward(s.w_gate, grad_gate_t)
    g2 = backward(s.w_up, grad_up_t)
    # Sum gate and up gradients in-place using broadcast (zero-alloc)
    g1d = g1.data
    g2d = g2.data
    sz2 = size(g1d)
    n2 = length(g1d)
    if length(s.grad_sum_buf) != n2
        s.grad_sum_buf = Vector{Float32}(undef, n2)
    end
    gs = s.grad_sum_buf
    _swiglu_grad_sum!(gs, g1d, g2d)
    Tensor(reshape(gs, sz2), grad_output.dtype)
end

# Function barriers for SwiGLU backward: dispatch on concrete Array{Float32,N}
# to enable fused broadcast with zero allocations.
function _swiglu_silu_grad!(gg::Vector{Float32}, ghd::Array{Float32,N1},
                            lu::Array{Float32,N2}, gps::Array{Float32,N3}) where {N1,N2,N3}
    ghd_flat = vec(ghd)
    lu_flat = vec(lu)
    gps_flat = vec(gps)
    # Two-pass silu derivative: sig = sigmoid(z), then dsilu = sig * (1 + z*(1-sig))
    @fastmath @. gg = 1f0 / (1f0 + exp(-gps_flat))
    @fastmath @. gg = (ghd_flat * lu_flat) * gg * (1f0 + gps_flat * (1f0 - gg))
end

function _swiglu_grad_up!(gu::Vector{Float32}, ghd::Array{Float32,N1},
                          sg::Array{Float32,N2}) where {N1,N2}
    ghd_flat = vec(ghd)
    sg_flat = vec(sg)
    @fastmath @. gu = ghd_flat * sg_flat
end

function _swiglu_grad_sum!(gs::Vector{Float32}, g1d::Array{Float32,N1},
                           g2d::Array{Float32,N2}) where {N1,N2}
    g1d_flat = vec(g1d)
    g2d_flat = vec(g2d)
    @fastmath @. gs = g1d_flat + g2d_flat
end

function parameters(s::SwiGLU)
    Tensor[p for p in Iterators.flatten((parameters(s.w_gate), parameters(s.w_up), parameters(s.w_down)))]
end
