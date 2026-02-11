# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

# tensor.jl — Core tensor type and element-wise / BLAS operations
#
# All data is stored as Array{Float32} regardless of the logical DType tag.
# The DType enum exists for future mixed-precision support; currently F32 only.
#
# Key design decisions:
#   - Julia is column-major: a (M, K) matrix has M rows and K columns, with
#     each column of M elements stored contiguously (first index varies fastest).
#     This is the natural layout for BLAS (Fortran-order), so mul! on 2D
#     matrices dispatches directly to sgemm with no copy.
#   - All mul! calls go through LinearAlgebra, which dispatches via LBT
#     (libblastrampoline). On macOS with AppleAccelerate.jl loaded, this
#     routes to Apple's Accelerate framework (AMX hardware), giving 7-14x
#     speedup over NEON-only OpenBLAS.
#   - In-place (!-suffix) functions mutate their first argument to avoid
#     allocation. The forward pass pre-allocates all buffers so the hot
#     path triggers 0 GC pauses.

@enum DType F32 F16 BF16 I32 I64

mutable struct Tensor
    data::Array{Float32}
    dtype::DType
    grad::Union{Array{Float32}, Nothing}
end

# Convenience constructor: default grad to nothing (backward-compatible)
Tensor(data::Array{Float32}, dtype::DType) = Tensor(data, dtype, nothing)

# --- Constructors -----------------------------------------------------------

zeros_tensor(dims::Int...; dtype::DType=F32) = Tensor(zeros(Float32, dims...), dtype)
ones_tensor(dims::Int...; dtype::DType=F32) = Tensor(ones(Float32, dims...), dtype)
randn_tensor(dims::Int...; dtype::DType=F32) = Tensor(randn(Float32, dims...), dtype)
randn_std(dims::Int...; std::Float32=1f0, dtype::DType=F32) = Tensor(randn(Float32, dims...) .* std, dtype)
from_array(arr::AbstractArray; dtype::DType=F32) = Tensor(Float32.(arr), dtype)

Base.size(t::Tensor) = size(t.data)
numel(t::Tensor) = length(t.data)
clone(t::Tensor) = Tensor(copy(t.data), t.dtype, t.grad === nothing ? nothing : copy(t.grad))

# --- Element-wise binary ops (allocating) ------------------------------------

Base.:+(a::Tensor, b::Tensor) = _tensor_binop_add(a.data, b.data, a.dtype)
Base.:-(a::Tensor, b::Tensor) = _tensor_binop_sub(a.data, b.data, a.dtype)
Base.:*(a::Tensor, b::Tensor) = _tensor_binop_mul(a.data, b.data, a.dtype)

function _tensor_binop_add(a::Array{Float32,N}, b::Array{Float32,M}, dtype::DType) where {N,M}
    Tensor(a .+ b, dtype)
end
function _tensor_binop_sub(a::Array{Float32,N}, b::Array{Float32,M}, dtype::DType) where {N,M}
    Tensor(a .- b, dtype)
end
function _tensor_binop_mul(a::Array{Float32,N}, b::Array{Float32,M}, dtype::DType) where {N,M}
    Tensor(a .* b, dtype)
end

scale(t::Tensor, s) = _tensor_scale(t.data, Float32(s), t.dtype)
function _tensor_scale(d::Array{Float32,N}, s::Float32, dtype::DType) where {N}
    Tensor(d .* s, dtype)
end

# SiLU (Swish): f(x) = x * sigmoid(x) = x / (1 + exp(-x))
silu(t::Tensor) = _tensor_silu(t.data, t.dtype)
function _tensor_silu(d::Array{Float32,N}, dtype::DType) where {N}
    Tensor(@.(d / (1f0 + exp(-d))), dtype)
end

# SiLU: f(x) = x / (1 + exp(-x))
# In-place variant: mutates data array directly, avoids allocation.
# @inbounds is safe here: eachindex always yields valid indices.
# @simd enables SIMD vectorization for the simple element-wise loop.
function silu_in_place!(t::Tensor)
    _silu_in_place_arr!(t.data)
    return t
end

function _silu_in_place_arr!(d::Array{Float32,N}) where {N}
    @inbounds @simd for i in eachindex(d)
        x = d[i]
        d[i] = x / (1f0 + exp(-x))
    end
end

# --- In-place element-wise ops (zero-alloc) ----------------------------------

function add_in_place!(a::Tensor, b::Tensor)
    _add_in_place_arr!(a.data, b.data)
    return a
end

function _add_in_place_arr!(ad::Array{Float32,N}, bd::Array{Float32,M}) where {N,M}
    ad .+= bd
end

function scale_in_place!(t::Tensor, s)
    _scale_in_place_arr!(t.data, Float32(s))
    return t
end

function _scale_in_place_arr!(d::Array{Float32,N}, s::Float32) where {N}
    d .*= s
end

# In-place element-wise multiply: a .*= b
function mul_in_place!(a::Tensor, b::Tensor)
    _mul_in_place_arr!(a.data, b.data)
    return a
end

function _mul_in_place_arr!(ad::Array{Float32,N}, bd::Array{Float32,M}) where {N,M}
    ad .*= bd
end

tensor_sum(t::Tensor)::Float32 = _tensor_sum(t.data)
tensor_mean(t::Tensor)::Float32 = _tensor_sum(t.data) / Float32(length(t.data))

_tensor_sum(d::Array{Float32,N}) where {N} = sum(d)

# --- Matrix multiplication ---------------------------------------------------
# 2D matmul: C = A * B  where A is (M, K), B is (K, N), C is (M, N)
# Julia column-major: columns are contiguous, so (M, K) has stride-1 along M.
# mul! dispatches to BLAS sgemm via LBT -> AppleAccelerate (AMX) on macOS.

function tensor_matmul(a::Tensor, b::Tensor, ::Val{2})::Tensor
    _tensor_matmul_2d(a.data, b.data, a.dtype)
end

function _tensor_matmul_2d(ad::Array{Float32,N1}, bd::Array{Float32,N2}, dtype::DType) where {N1,N2}
    M, K = size(ad)
    K2, N = size(bd)
    K == K2 || error("matmul dimension mismatch: $K vs $K2")
    ad2 = reshape(ad, M, K)
    bd2 = reshape(bd, K, N)
    out = Matrix{Float32}(undef, M, N)
    mul!(out, ad2, bd2)  # BLAS sgemm via LBT
    return Tensor(out, dtype)
end

# 3D batched matmul: C[b,:,:] = A[b,:,:] * B[b,:,:]
# Julia stores 3D arrays in column-major order: the first index varies fastest.
# For a (B, M, K) array, @view arr[batch,:,:] yields a non-contiguous slice
# with stride[1]=B (not 1). BLAS sgemm requires stride[1]=1 (contiguous columns),
# so we must copy each batch slice into contiguous 2D buffers before calling mul!.
# Without this copy, Julia would fall back to generic matmul (~10x slower).
function tensor_matmul(a::Tensor, b::Tensor, ::Val{3})::Tensor
    _tensor_matmul_3d(a.data, b.data, a.dtype)
end

function _tensor_matmul_3d(ad::Array{Float32,N1}, bd::Array{Float32,N2}, dtype::DType) where {N1,N2}
    B, M, K = size(ad)
    B2, K2, N = size(bd)
    B == B2 || error("batch size mismatch: $B vs $B2")
    K == K2 || error("matmul dimension mismatch: $K vs $K2")
    out = Array{Float32}(undef, B, M, N)
    # Pre-allocate contiguous 2D buffers for BLAS-compatible slices.
    # @view ad[batch,:,:] has stride[1]=B (non-contiguous) -- BLAS sgemm requires
    # stride[1]=1. Copy each batch slice into contiguous buffers for real BLAS dispatch.
    a_buf = Matrix{Float32}(undef, M, K)
    b_buf = Matrix{Float32}(undef, K, N)
    o_buf = Matrix{Float32}(undef, M, N)
    for batch in 1:B
        # Manual copy: column-major iteration (inner=rows, outer=cols) for cache locality
        @inbounds for j in 1:K, i in 1:M
            a_buf[i, j] = ad[batch, i, j]
        end
        @inbounds for j in 1:N, i in 1:K
            b_buf[i, j] = bd[batch, i, j]
        end
        mul!(o_buf, a_buf, b_buf)  # BLAS sgemm on contiguous data
        @inbounds for j in 1:N, i in 1:M
            out[batch, i, j] = o_buf[i, j]
        end
    end
    return Tensor(out, dtype)
end

tensor_matmul(::Tensor, ::Tensor, ::Val{N}) where {N} = error("matmul not implemented for $(N)D tensors")
tensor_matmul(a::Tensor, b::Tensor) = tensor_matmul(a, b, Val(ndims(a.data)))

# C = A * B^T  where A is (M, K), B is (N, K), C is (M, N)
# transpose(bd2) creates a lazy wrapper; mul! fuses the transpose into sgemm (TRANSA/TRANSB flags).
function matmul_transposed_b(a::Tensor, b::Tensor)::Tensor
    _matmul_transposed_b_arr(a.data, b.data, a.dtype)
end

function _matmul_transposed_b_arr(ad::Array{Float32,N1}, bd::Array{Float32,N2}, dtype::DType) where {N1,N2}
    M, K = size(ad)
    N, K2 = size(bd)
    K == K2 || error("matmulT dimension mismatch: $K vs $K2")
    ad2 = reshape(ad, M, K)
    bd2 = reshape(bd, N, K)
    out = Matrix{Float32}(undef, M, N)
    mul!(out, ad2, transpose(bd2))  # sgemm with TRANSB='T'
    Tensor(out, dtype)
end

# In-place matmul_transposed_b: writes result into pre-allocated output (Tensor args)
function matmul_transposed_b!(out::Matrix{Float32}, a::Tensor, b::Tensor)
    mul!(out, a.data, transpose(b.data))
    return out
end

# In-place matmul_transposed_b on raw Arrays
function matmul_transposed_b!(out::Matrix{Float32}, a::Matrix{Float32}, b::Matrix{Float32})
    mul!(out, a, transpose(b))
    return out
end

# In-place 2D matmul: out = a * b, writing into pre-allocated out
function tensor_matmul!(out::Matrix{Float32}, a::Matrix{Float32}, b::Matrix{Float32})
    mul!(out, a, b)
    return out
end

# --- Softmax -----------------------------------------------------------------
# Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# The max-subtraction trick prevents overflow in exp().
# Three-pass algorithm: (1) find max, (2) exp + sum, (3) normalize.

# In-place softmax for 2D arrays: operates on reshaped views.
# Each row is treated as an independent probability vector.
function softmax!(out::AbstractMatrix{Float32}, src::AbstractMatrix{Float32})
    num_vectors, last_dim = size(src)
    for v in 1:num_vectors
        mx = -Inf32
        @inbounds for i in 1:last_dim
            val = src[v, i]
            mx = ifelse(val > mx, val, mx)
        end
        s = 0f0
        @inbounds for i in 1:last_dim
            e = exp(src[v, i] - mx)
            out[v, i] = e
            s += e
        end
        inv_s = 1f0 / s
        @inbounds @simd for i in 1:last_dim
            out[v, i] *= inv_s
        end
    end
    out
end

# In-place element-wise add on raw arrays: a .+= b
function add!(a::Array{Float32}, b::Array{Float32})
    a .+= b
    return a
end

# In-place element-wise subtract on raw arrays: a .-= b
function sub!(a::Array{Float32}, b::Array{Float32})
    a .-= b
    return a
end

# Transpose: swaps the last two dimensions via permutedims.
# permutedims allocates a new array (not a view).
function transpose_tensor(t::Tensor)::Tensor
    _transpose_arr(t.data, t.dtype)
end

function _transpose_arr(d::Array{Float32,N}, dtype::DType) where {N}
    N >= 2 || error("transpose requires at least 2D tensor")
    perm = collect(1:N)
    perm[end], perm[end-1] = perm[end-1], perm[end]
    Tensor(permutedims(d, perm), dtype)
end

# reshape is a zero-copy view in Julia (shares underlying memory).
reshape_tensor(t::Tensor, dims::Int...) = Tensor(reshape(t.data, dims...), t.dtype)

# Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# Allocating version: creates new output array.
# Flattens to (num_vectors, last_dim) for uniform iteration.
function softmax(t::Tensor)::Tensor
    _softmax_arr(t.data, t.dtype)
end

function _softmax_arr(src::Array{Float32,N}, dtype::DType)::Tensor where {N}
    nd = N
    nd >= 1 || error("softmax requires at least 1 dimension")
    out = similar(src)
    last_dim = size(src, nd)
    num_vectors = div(length(src), last_dim)
    flat_src = reshape(src, num_vectors, last_dim)
    flat_dst = reshape(out, num_vectors, last_dim)
    for v in 1:num_vectors
        mx = -Inf32
        @inbounds for i in 1:last_dim
            val = flat_src[v, i]
            mx = ifelse(val > mx, val, mx)
        end
        s = 0f0
        @inbounds for i in 1:last_dim
            e = exp(flat_src[v, i] - mx)
            flat_dst[v, i] = e
            s += e
        end
        inv_s = 1f0 / s
        @inbounds @simd for i in 1:last_dim
            flat_dst[v, i] *= inv_s
        end
    end
    Tensor(out, dtype)
end

# Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# In-place into a pre-allocated output buffer (same shape as input).
# Concrete dispatch on Array{Float32,N} avoids abstract-type penalty.
function softmax_into!(out::Array{Float32,N}, src::Array{Float32,M}, dtype::DType)::Tensor where {N,M}
    last_dim = size(src, M)
    num_vectors = div(length(src), last_dim)
    flat_src = reshape(src, num_vectors, last_dim)
    flat_dst = reshape(out, num_vectors, last_dim)
    for v in 1:num_vectors
        mx = -Inf32
        @inbounds for i in 1:last_dim
            val = flat_src[v, i]
            mx = ifelse(val > mx, val, mx)
        end
        s = 0f0
        @inbounds for i in 1:last_dim
            e = exp(flat_src[v, i] - mx)
            flat_dst[v, i] = e
            s += e
        end
        inv_s = 1f0 / s
        @inbounds @simd for i in 1:last_dim
            flat_dst[v, i] *= inv_s
        end
    end
    Tensor(out, dtype)
end

# Softmax: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# In-place on a 1D vector. Used for attention score rows.
function softmax_in_place!(xs::AbstractVector{Float32})
    mx = -Inf32
    @inbounds for i in eachindex(xs)
        val = xs[i]
        mx = ifelse(val > mx, val, mx)
    end
    s = 0f0
    @inbounds for i in eachindex(xs)
        e = exp(xs[i] - mx)
        xs[i] = e
        s += e
    end
    inv_s = 1f0 / s
    @inbounds @simd for i in eachindex(xs)
        xs[i] *= inv_s
    end
    return xs
end

argmax_f32(xs::AbstractVector{Float32})::Tuple{Int,Float32} = reverse(findmax(xs))

# L1 normalization: x_i = x_i / sum(x)
function normalize_in_place!(xs::AbstractVector{Float32})
    s = 0f0
    @inbounds for i in eachindex(xs)
        s += xs[i]
    end
    s == 0f0 && return xs
    inv_s = 1f0 / s
    @inbounds @simd for i in eachindex(xs)
        xs[i] *= inv_s
    end
    return xs
end
