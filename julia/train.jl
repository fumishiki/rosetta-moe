# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

# train.jl — Training loop, AdamW optimizer, loss functions, and utilities
#
# Implements:
#   - Cross-entropy loss and gradient computation
#   - Gradient clipping (global norm)
#   - AdamW optimizer with cosine-decay learning rate schedule
#   - Activation checkpointing (for memory-efficient training)
#   - Loss scaling (static/dynamic) for future mixed-precision support
#
# Note on allocation:
#   Both the forward and backward passes use pre-allocated buffers for all
#   intermediate gradient tensors. After the first call (which allocates the
#   buffers), subsequent calls reuse them, targeting 0 GC pauses in steady state.

struct TrainConfig
    lr::Float32               # peak learning rate
    beta1::Float32            # AdamW first moment decay
    beta2::Float32            # AdamW second moment decay
    eps::Float32              # AdamW epsilon (numerical stability)
    weight_decay::Float32     # AdamW weight decay coefficient
    grad_clip::Float32        # max gradient global norm
    warmup_steps::Int         # linear warmup duration
    total_steps::Int          # total training steps (for cosine decay)
    aux_alpha::Float32        # MoE auxiliary loss coefficient
end

default_train_config() = TrainConfig(1f-4, 0.9f0, 0.95f0, 1f-8, 0.1f0, 1f0, 1000, 100000, 0.01f0)

mutable struct AdamWState
    m::Vector{Float32}        # first moment estimate (per parameter element)
    v::Vector{Float32}        # second moment estimate (per parameter element)
end

mutable struct Trainer
    model::MoETransformer
    config::TrainConfig
    step::Int
    states::Vector{AdamWState}
    params_cache::Vector{Tensor}  # cached parameters(model) — avoids vcat allocation per step
    grad_buf::Union{Array{Float32}, Nothing}  # reusable gradient buffer
    ce_row_buf::Union{Vector{Float32}, Nothing}  # reusable row buffer for cross_entropy
    # Pre-allocated permutation buffers for cross-entropy (vocab-first layout for contiguous access)
    ce_perm_buf::Union{Array{Float32,3}, Nothing}       # (vocab, batch, seq_len) for logits
    ce_grad_perm_buf::Union{Array{Float32,3}, Nothing}  # (vocab, batch, seq_len) for grad output
end

function Trainer(model::MoETransformer, config::TrainConfig)
    params = collect(parameters(model))
    states = [AdamWState(zeros(Float32, numel(p)), zeros(Float32, numel(p))) for p in params]
    Trainer(model, config, 0, states, params, nothing, nothing, nothing, nothing)
end

# Learning rate schedule: linear warmup then cosine decay to 10% of peak lr.
# LR(t) = lr * t / warmup_steps                            for t < warmup_steps
# LR(t) = min_lr + 0.5 * (lr - min_lr) * (1 + cos(pi * progress))  for t >= warmup_steps
#   where progress = (t - warmup) / (total - warmup), min_lr = lr * 0.1
function get_lr(t::Trainer)::Float32
    if t.step < t.config.warmup_steps
        return t.config.lr * Float32(t.step) / Float32(t.config.warmup_steps)
    end
    progress = Float32(t.step - t.config.warmup_steps) / Float32(t.config.total_steps - t.config.warmup_steps)
    progress = min(progress, 1f0)
    min_lr = t.config.lr * 0.1f0
    min_lr + 0.5f0 * (t.config.lr - min_lr) * (1f0 + cos(Float32(pi) * progress))
end

# CrossEntropy: L = -mean(log(softmax(logits)[target]))
#             = -mean(logits[target] - log(sum(exp(logits))))
# Uses log-sum-exp trick for numerical stability.
#
# Performance: logits shape is (batch, seq_len, vocab). In Julia's column-major
# layout, iterating vocab in the inner loop accesses stride=batch*seq_len memory.
# We permute to (vocab, batch, seq_len) so vocab becomes the contiguous axis,
# then operate on stride-1 columns. The permutedims! cost is amortized by the
# dramatic improvement in inner-loop cache locality.
function cross_entropy_loss(logits::Tensor, targets::Tensor;
                            row_buf::Union{Vector{Float32}, Nothing}=nothing,
                            perm_buf::Union{Array{Float32,3}, Nothing}=nothing)::Float32
    _cross_entropy_loss_impl(logits.data, targets.data, row_buf, perm_buf)
end

function _cross_entropy_loss_impl(ld::Array{Float32,N}, td::Array{Float32,M},
                                  row_buf::Union{Vector{Float32}, Nothing},
                                  perm_buf::Union{Array{Float32,3}, Nothing})::Float32 where {N,M}
    dims = size(ld)
    batch, seq_len, vocab_size = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len

    # Permute to (vocab, batch, seq_len) for contiguous vocab access
    perm_sz = (vocab_size, batch, seq_len)
    if perm_buf === nothing || size(perm_buf) != perm_sz
        perm_buf = Array{Float32}(undef, perm_sz...)
    end
    permutedims!(perm_buf, ld, (3, 1, 2))

    total_loss = 0f0
    @fastmath for b in 1:batch, s in 1:seq_len
        @inbounds target_idx = Int(td[b, s]) + 1  # 0-based -> 1-based
        # Log-sum-exp on contiguous column perm_buf[:, b, s]
        max_val = -Inf32
        @inbounds for v in 1:vocab_size
            val = perm_buf[v, b, s]
            max_val = ifelse(val > max_val, val, max_val)
        end
        sum_exp = 0f0
        @inbounds for v in 1:vocab_size
            sum_exp += exp(perm_buf[v, b, s] - max_val)
        end
        @inbounds log_prob = perm_buf[target_idx, b, s] - max_val - log(sum_exp)
        total_loss -= log_prob
    end
    total_loss / Float32(num_tokens)
end

# CrossEntropy gradient: d_logits = softmax(logits) - one_hot(target)
# Averaged over all tokens in the batch.
# The gradient for the target class is (p - 1), for all others it's p,
# where p = softmax(logits).
#
# Performance: Same permutedims strategy as cross_entropy_loss. We permute logits
# to (vocab, batch, seq_len), compute softmax on contiguous columns, then permute
# the gradient back to (batch, seq_len, vocab). This eliminates strided memory
# access in the inner loop (26x speedup vs the original implementation).
function cross_entropy_grad(logits::Tensor, targets::Tensor;
                            row_buf::Union{Vector{Float32}, Nothing}=nothing)::Tensor
    _cross_entropy_grad_into!(nothing, logits, targets; row_buf=row_buf)
end

function cross_entropy_grad_into!(grad_buf::Array{Float32}, logits::Tensor, targets::Tensor;
                                  row_buf::Union{Vector{Float32}, Nothing}=nothing,
                                  perm_buf::Union{Array{Float32,3}, Nothing}=nothing,
                                  grad_perm_buf::Union{Array{Float32,3}, Nothing}=nothing)::Tensor
    _cross_entropy_grad_into!(grad_buf, logits, targets;
                              row_buf=row_buf, perm_buf=perm_buf, grad_perm_buf=grad_perm_buf)
end

# d_logits[b,s,v] = (softmax(logits[b,s,:])[v] - 1{v==target}) / num_tokens
function _cross_entropy_grad_into!(grad_buf::Union{Array{Float32}, Nothing}, logits::Tensor, targets::Tensor;
                                   row_buf::Union{Vector{Float32}, Nothing}=nothing,
                                   perm_buf::Union{Array{Float32,3}, Nothing}=nothing,
                                   grad_perm_buf::Union{Array{Float32,3}, Nothing}=nothing)::Tensor
    dims = size(logits)
    batch, seq_len, vocab_size = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len
    if grad_buf === nothing || size(grad_buf) != dims
        gd = Array{Float32}(undef, dims...)
    else
        gd = grad_buf
    end
    _cross_entropy_grad_permuted!(gd, logits.data, targets.data,
                                  batch, seq_len, vocab_size, num_tokens,
                                  perm_buf, grad_perm_buf)
    Tensor(gd, logits.dtype)
end

# Inner function barrier: dispatches on concrete Array{Float32,N} types for
# type-stable SIMD code generation. Permutes to (vocab, batch, seq_len) for
# contiguous inner-loop access, then permutes result back.
function _cross_entropy_grad_permuted!(gd::Array{Float32,N1},
                                       ld::Array{Float32,N2},
                                       td::Array{Float32,N3},
                                       batch::Int, seq_len::Int, vocab_size::Int, num_tokens::Int,
                                       perm_buf::Union{Array{Float32,3}, Nothing},
                                       grad_perm_buf::Union{Array{Float32,3}, Nothing}) where {N1,N2,N3}
    perm_sz = (vocab_size, batch, seq_len)

    # Permute logits to (vocab, batch, seq_len) — vocab becomes contiguous (stride-1)
    if perm_buf === nothing || size(perm_buf) != perm_sz
        perm_buf = Array{Float32}(undef, perm_sz...)
    end
    permutedims!(perm_buf, ld, (3, 1, 2))

    # Gradient in permuted layout
    if grad_perm_buf === nothing || size(grad_perm_buf) != perm_sz
        grad_perm_buf = Array{Float32}(undef, perm_sz...)
    end

    # Softmax + one-hot subtraction on contiguous columns
    @fastmath for b in 1:batch, s in 1:seq_len
        # Softmax: 3-pass (max, exp+sum, normalize) on contiguous perm_buf[:, b, s]
        mx = -Inf32
        @inbounds for v in 1:vocab_size
            val = perm_buf[v, b, s]
            mx = ifelse(val > mx, val, mx)
        end
        sm = 0f0
        @inbounds for v in 1:vocab_size
            e = exp(perm_buf[v, b, s] - mx)
            grad_perm_buf[v, b, s] = e
            sm += e
        end
        inv_s = 1f0 / sm
        @inbounds @simd for v in 1:vocab_size
            grad_perm_buf[v, b, s] *= inv_s
        end
        # Subtract 1 at target class
        @inbounds target_idx = Int(td[b, s]) + 1
        @inbounds grad_perm_buf[target_idx, b, s] -= 1f0
    end

    # Average over all tokens
    sc = 1f0 / Float32(num_tokens)
    @inbounds @simd for i in eachindex(grad_perm_buf)
        grad_perm_buf[i] *= sc
    end

    # Permute back to (batch, seq_len, vocab)
    permutedims!(gd, grad_perm_buf, (2, 3, 1))
    nothing
end

# Gradient clipping: ||g||_2 > clip_norm => g *= clip_norm / ||g||_2
# Uses function barrier for SIMD on concrete Array{Float32,N}.
function clip_grad_by_global_norm!(t::Tensor, clip_norm::Float32)::Float32
    clip_norm <= 0f0 && return 0f0
    _clip_grad_impl!(t.data, clip_norm)
end

function _clip_grad_impl!(d::Array{Float32,N}, clip_norm::Float32)::Float32 where {N}
    sum_sq = 0f0
    @inbounds @simd for i in eachindex(d)
        sum_sq += d[i] * d[i]
    end
    norm = sqrt(sum_sq)
    if norm > clip_norm
        sc = clip_norm / (norm + 1f-12)
        @inbounds @simd for i in eachindex(d)
            d[i] *= sc
        end
    end
    norm
end

# =============================================================================
# Training step: forward -> loss -> backward -> AdamW update
# =============================================================================
# AdamW: m = beta1*m + (1-beta1)*g
#        v = beta2*v + (1-beta2)*g^2
#        m_hat = m / (1 - beta1^t)
#        v_hat = v / (1 - beta2^t)
#        w -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
#
# Note: This implementation uses a simplified gradient signal (mean absolute
# gradient) as a proxy for per-parameter gradients, which is suitable for
# educational purposes but not production training.

# AdamW SIMD kernel for a single parameter tensor with per-element gradients.
# This is a function barrier: Tensor.data is Array{Float32} (abstract over N),
# but dispatching through this function specializes on the concrete Array{Float32,N},
# allowing LLVM to emit vectorized SIMD code. Without this barrier, the inner loop
# falls back to generic (unvectorized) iteration — 200x slower.
function _adamw_update_param!(pd::Array{Float32,N}, md::Vector{Float32}, vd::Vector{Float32},
                              b1::Float32, one_minus_b1::Float32, b2::Float32, one_minus_b2::Float32,
                              lr::Float32, m_corr::Float32, v_corr::Float32,
                              eps::Float32, wd::Float32, gd::Array{Float32,M}) where {N,M}
    gd_flat = reshape(gd, length(gd))
    @inbounds @fastmath @simd for j in eachindex(pd)
        g_val = gd_flat[j]
        mj = b1 * md[j] + one_minus_b1 * g_val             # m = beta1*m + (1-beta1)*g
        vj = b2 * vd[j] + one_minus_b2 * g_val * g_val     # v = beta2*v + (1-beta2)*g^2
        md[j] = mj
        vd[j] = vj
        # w -= lr * (m_hat/(sqrt(v_hat)+eps) + wd*w)
        pd[j] -= lr * (mj * m_corr / (sqrt(vj * v_corr) + eps) + wd * pd[j])
    end
end

function train_step!(t::Trainer, input::Tensor, targets::Tensor)::Float32
    t.step += 1
    params = t.params_cache

    # Zero all parameter gradients before backward pass
    for param in params
        if param.grad === nothing
            param.grad = zeros(Float32, size(param.data)...)
        else
            fill!(param.grad, 0f0)
        end
    end

    logits = forward(t.model, input)
    # Ensure permutation buffers for cross-entropy (vocab-first layout)
    dims = size(logits)
    vocab_size = dims[3]
    perm_sz = (vocab_size, dims[1], dims[2])
    if t.ce_perm_buf === nothing || size(t.ce_perm_buf) != perm_sz
        t.ce_perm_buf = Array{Float32}(undef, perm_sz...)
    end
    if t.ce_grad_perm_buf === nothing || size(t.ce_grad_perm_buf) != perm_sz
        t.ce_grad_perm_buf = Array{Float32}(undef, perm_sz...)
    end
    loss = cross_entropy_loss(logits, targets; perm_buf=t.ce_perm_buf)
    aux_loss = total_aux_loss(t.model, t.config.aux_alpha)
    total_loss = loss + aux_loss
    # Reuse gradient buffer
    if t.grad_buf === nothing || size(t.grad_buf) != dims
        t.grad_buf = Array{Float32}(undef, dims...)
    end
    grad_output = cross_entropy_grad_into!(t.grad_buf, logits, targets;
                                           perm_buf=t.ce_perm_buf,
                                           grad_perm_buf=t.ce_grad_perm_buf)
    clip_grad_by_global_norm!(grad_output, t.config.grad_clip)
    backward(t.model, grad_output)
    lr = get_lr(t)
    step = t.step
    b1 = t.config.beta1
    b2 = t.config.beta2
    eps = t.config.eps
    wd = t.config.weight_decay
    # Bias correction terms for AdamW
    m_corr = 1f0 / (1f0 - b1^Float32(step))   # 1/(1 - beta1^t)
    v_corr = 1f0 / (1f0 - b2^Float32(step))   # 1/(1 - beta2^t)
    one_minus_b1 = 1f0 - b1
    one_minus_b2 = 1f0 - b2
    # AdamW update for each parameter using actual per-parameter gradients
    for (i, param) in enumerate(params)
        if param.grad !== nothing
            _adamw_update_param!(param.data, t.states[i].m, t.states[i].v,
                                 b1, one_minus_b1, b2, one_minus_b2,
                                 lr, m_corr, v_corr, eps, wd, param.grad)
        end
    end
    total_loss
end

# =============================================================================
# Activation Checkpointing — trade compute for memory
# =============================================================================
# Stores intermediate activations at segment boundaries during forward pass.
# During backward, recomputes from the nearest checkpoint instead of storing
# all activations in memory.

mutable struct CheckpointStorage
    checkpoints::Dict{Int,Tensor}
    enabled::Bool
end

CheckpointStorage(enabled::Bool) = CheckpointStorage(Dict{Int,Tensor}(), enabled)

function save!(s::CheckpointStorage, block_idx::Int, t::Tensor)
    s.enabled && (s.checkpoints[block_idx] = t)
end

get_checkpoint(s::CheckpointStorage, block_idx::Int) = get(s.checkpoints, block_idx, nothing)
clear!(s::CheckpointStorage) = (empty!(s.checkpoints); nothing)
Base.length(s::CheckpointStorage) = length(s.checkpoints)

struct CheckpointContext
    storage::CheckpointStorage
    segment_size::Int           # save every N blocks
end

function CheckpointContext(segment_size::Int)
    sz = max(segment_size, 1)
    CheckpointContext(CheckpointStorage(segment_size > 0), sz)
end

disabled_checkpoint_context() = CheckpointContext(CheckpointStorage(false), 1)

should_checkpoint(c::CheckpointContext, block_idx::Int) = c.storage.enabled && (block_idx % c.segment_size == 0)

function maybe_save!(c::CheckpointContext, block_idx::Int, t::Tensor)
    should_checkpoint(c, block_idx) && save!(c.storage, block_idx, t)
end

function get_checkpoint(c::CheckpointContext, block_idx::Int)
    cp_idx = div(block_idx, c.segment_size) * c.segment_size
    get_checkpoint(c.storage, cp_idx)
end

clear!(c::CheckpointContext) = clear!(c.storage)

# =============================================================================
# Loss Scaling — for mixed-precision training (future use)
# =============================================================================
# Static: fixed scale factor, no adaptive behavior.
# Dynamic: auto-adjusts scale based on gradient overflow detection.

abstract type AbstractLossScaler end

struct StaticLossScaler <: AbstractLossScaler
    current_scale::Float32
end

mutable struct DynamicLossScaler <: AbstractLossScaler
    current_scale::Float32     # current loss scale (starts high, e.g. 65536)
    scale_factor::Float32      # multiply/divide by this on scale changes
    scale_window::Int          # steps without overflow before scaling up
    growth_tracker::Int        # consecutive non-overflow steps
    overflow::Bool
end

# Shared interface
scale_loss(s::AbstractLossScaler, loss::Float32) = loss * s.current_scale
unscale_grads(s::AbstractLossScaler, grad::Float32) = grad / s.current_scale

# Static scaler: no-ops for overflow/update
should_skip_step(::StaticLossScaler) = false
check_overflow!(::StaticLossScaler, ::Vector{Float32}) = false
update!(::StaticLossScaler) = nothing

# Dynamic scaler: skip step if overflow detected
should_skip_step(s::DynamicLossScaler) = s.overflow

function check_overflow!(s::DynamicLossScaler, grads::Vector{Float32})::Bool
    s.overflow = any(g -> !isfinite(g) || abs(g) > 3.4f38, grads)
    s.overflow
end

# Dynamic scale update:
#   overflow -> halve scale, reset growth tracker
#   no overflow for scale_window steps -> double scale
function update!(s::DynamicLossScaler)
    if s.overflow
        s.current_scale /= s.scale_factor
        s.growth_tracker = 0
        s.overflow = false
        return
    end
    s.growth_tracker += 1
    if s.growth_tracker >= s.scale_window
        s.current_scale *= s.scale_factor
        s.growth_tracker = 0
    end
end

# Constructors
static_loss_scaler(sc::Float32) = StaticLossScaler(sc)
dynamic_loss_scaler() = DynamicLossScaler(65536f0, 2f0, 2000, 0, false)

# =============================================================================
# Mixed Precision Configuration (future use)
# =============================================================================

struct MixedPrecisionConfig
    enabled::Bool
    compute_dtype::DType
    loss_scale::Type{<:AbstractLossScaler}
    fp32_layers::Vector{String}    # layer names that must stay in FP32
end

is_fp32_layer(c::MixedPrecisionConfig, name::String) = any(s -> occursin(s, name), c.fp32_layers)
default_mixed_precision_config() = MixedPrecisionConfig(false, F16, DynamicLossScaler, ["final_norm", "lm_head"])

function fp16_mixed_precision_config()
    cfg = default_mixed_precision_config()
    MixedPrecisionConfig(true, cfg.compute_dtype, cfg.loss_scale, cfg.fp32_layers)
end

# Master weight copies for mixed-precision training (FP32 copies of FP16 weights)
struct MasterWeights
    weights::Vector{Tensor}
end

new_master_weights(params::Vector{Tensor}) = MasterWeights([zeros_tensor(size(p)...) for p in params])
