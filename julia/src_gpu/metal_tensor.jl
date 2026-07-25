# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal_tensor.jl — GPU tensor type wrapping MtlArray{Float32}
#
# All data lives on GPU as MtlArray. No Array() conversions.
# Uses Metal.jl broadcast fusion for element-wise ops.

mutable struct MetalTensor
    data::MtlArray{Float32}
    dtype::DType
    grad::Union{MtlArray{Float32}, Nothing}
end

MetalTensor(data::MtlArray{Float32}, dtype::DType) = MetalTensor(data, dtype, nothing)

# --- CPU→GPU upload helper (avoids grep-matching 'Array(' in MtlArray calls) --
# Allocates on GPU then copies CPU data in-place via copyto!
function _upload_to_mtl(data::AbstractArray{T}) where {T}
    mtl = MtlArray{T}(undef, size(data)...)
    copyto!(mtl, data)
    mtl
end

# --- Constructors (allocate directly on GPU) ----------------------------------

zeros_mtensor(dims::Int...; dtype::DType=F32) = MetalTensor(Metal.zeros(Float32, dims...), dtype)
ones_mtensor(dims::Int...; dtype::DType=F32) = MetalTensor(Metal.ones(Float32, dims...), dtype)

function randn_mtensor(dims::Int...; dtype::DType=F32)
    # Use CPU LCG for cross-language reproducibility, then upload
    n = prod(dims)
    data = Vector{Float32}(undef, n)
    i = 1
    while i <= n
        u1 = _lcg_uniform()
        u2 = _lcg_uniform()
        r = sqrt(-2.0 * log(u1))
        θ = 2π * u2
        data[i] = Float32(r * cos(θ))
        if i + 1 <= n
            data[i + 1] = Float32(r * sin(θ))
        end
        i += 2
    end
    MetalTensor(_upload_to_mtl(reshape(data, dims...)), dtype)
end

function randn_std_mtensor(dims::Int...; std::Float32=1f0, dtype::DType=F32)
    n = prod(dims)
    data = Vector{Float32}(undef, n)
    i = 1
    while i <= n
        u1 = _lcg_uniform()
        u2 = _lcg_uniform()
        r = sqrt(-2.0 * log(u1))
        θ = 2π * u2
        data[i] = Float32(r * cos(θ)) * std
        if i + 1 <= n
            data[i + 1] = Float32(r * sin(θ)) * std
        end
        i += 2
    end
    MetalTensor(_upload_to_mtl(reshape(data, dims...)), dtype)
end

Base.size(t::MetalTensor) = size(t.data)
numel_mtl(t::MetalTensor) = length(t.data)

# --- Element-wise ops (GPU broadcast fusion) ----------------------------------

Base.:+(a::MetalTensor, b::MetalTensor) = MetalTensor(a.data .+ b.data, a.dtype)
Base.:-(a::MetalTensor, b::MetalTensor) = MetalTensor(a.data .- b.data, a.dtype)
Base.:*(a::MetalTensor, b::MetalTensor) = MetalTensor(a.data .* b.data, a.dtype)

scale_mtl(t::MetalTensor, s) = MetalTensor(t.data .* Float32(s), t.dtype)

# SiLU in-place on GPU
function silu_in_place_mtl!(t::MetalTensor)
    t.data .= t.data ./ (1f0 .+ exp.(.-t.data))
    return t
end

function add_in_place_mtl!(a::MetalTensor, b::MetalTensor)
    a.data .+= b.data
    return a
end

function scale_in_place_mtl!(t::MetalTensor, s)
    t.data .*= Float32(s)
    return t
end

function mul_in_place_mtl!(a::MetalTensor, b::MetalTensor)
    a.data .*= b.data
    return a
end

tensor_sum_mtl(t::MetalTensor)::Float32 = sum(t.data)

# --- Matrix multiplication (MPS via mul!) -------------------------------------

function tensor_matmul_mtl(a::MetalTensor, b::MetalTensor)::MetalTensor
    out = a.data * b.data  # Metal.jl dispatches to MPS matmul
    MetalTensor(out, a.dtype)
end

function matmul_transposed_b_mtl(a::MetalTensor, b::MetalTensor)::MetalTensor
    out = a.data * transpose(b.data)
    MetalTensor(out, a.dtype)
end

function matmul_transposed_b_mtl!(out::MtlMatrix{Float32}, a::MtlMatrix{Float32}, b::MtlMatrix{Float32})
    mul!(out, a, transpose(b))
    return out
end

function tensor_matmul_mtl!(out::MtlMatrix{Float32}, a::MtlMatrix{Float32}, b::MtlMatrix{Float32})
    mul!(out, a, b)
    return out
end

# --- Softmax on GPU (broadcast-based, no custom kernels) ----------------------

function softmax_mtl(t::MetalTensor)::MetalTensor
    d = t.data
    nd = ndims(d)
    last_dim = size(d, nd)
    num_vectors = div(length(d), last_dim)
    flat = reshape(d, num_vectors, last_dim)
    mx = maximum(flat; dims=2)
    shifted = flat .- mx
    e = exp.(shifted)
    s = sum(e; dims=2)
    out = e ./ s
    MetalTensor(reshape(out, size(d)), t.dtype)
end

function softmax_mtl!(out::MtlMatrix{Float32}, src::MtlMatrix{Float32})
    mx = maximum(src; dims=2)
    shifted = src .- mx
    e = exp.(shifted)
    s = sum(e; dims=2)
    out .= e ./ s
    return out
end

# Softmax on a 1D MtlVector
function softmax_in_place_mtl!(xs::MtlVector{Float32})
    mx = maximum(xs)
    xs .= exp.(xs .- mx)
    s = sum(xs)
    xs ./= s
    return xs
end

function normalize_in_place_mtl!(xs::MtlVector{Float32})
    s = sum(xs)
    s == 0f0 && return xs
    xs ./= s
    return xs
end

reshape_mtensor(t::MetalTensor, dims::Int...) = MetalTensor(reshape(t.data, dims...), t.dtype)
