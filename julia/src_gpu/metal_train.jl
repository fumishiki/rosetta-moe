# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal_train.jl — GPU training loop with AdamW optimizer
#
# Cross-entropy loss and gradient computed on GPU.
# AdamW updates run on GPU (MtlArray broadcast).
# NO Array() conversions — all computation stays on GPU.

struct MetalTrainConfig
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32
    grad_clip::Float32
    warmup_steps::Int
    total_steps::Int
    aux_alpha::Float32
    z_loss_weight::Float32
    routing_mode::RoutingMode
end

default_metal_train_config() = MetalTrainConfig(1f-4, 0.9f0, 0.95f0, 1f-8, 0.1f0, 0.5f0, 1000, 100000, 0.01f0, 0.05f0, TopKMode)

const _CE_VOCAB_RANGE_CACHE = Dict{Int, MtlVector{Float32}}()

function _get_ce_vocab_range(vocab_size::Int)::MtlVector{Float32}
    cached = get(_CE_VOCAB_RANGE_CACHE, vocab_size, nothing)
    if cached === nothing
        cached = _upload_to_mtl(Float32[i for i in 0:vocab_size-1])
        _CE_VOCAB_RANGE_CACHE[vocab_size] = cached
    end
    return cached
end

mutable struct MetalAdamWState
    m::MtlVector{Float32}
    v::MtlVector{Float32}
end

mutable struct MetalTrainer
    model::MetalMoETransformer
    config::MetalTrainConfig
    step::Int
    states::Vector{MetalAdamWState}
    params_cache::Vector{MetalTensor}
end

function MetalTrainer(model::MetalMoETransformer, config::MetalTrainConfig)
    params = gpu_parameters(model)
    states = [MetalAdamWState(Metal.zeros(Float32, numel_mtl(p)), Metal.zeros(Float32, numel_mtl(p))) for p in params]
    MetalTrainer(model, config, 0, states, params)
end

function get_lr_mtl(t::MetalTrainer)::Float32
    if t.step < t.config.warmup_steps
        return t.config.lr * Float32(t.step) / Float32(t.config.warmup_steps)
    end
    progress = Float32(t.step - t.config.warmup_steps) / Float32(t.config.total_steps - t.config.warmup_steps)
    progress = min(progress, 1f0)
    min_lr = t.config.lr * 0.1f0
    min_lr + 0.5f0 * (t.config.lr - min_lr) * (1f0 + cos(Float32(pi) * progress))
end

# Cross-entropy loss on GPU — fully on-device
function cross_entropy_loss_mtl(logits::MetalTensor, targets::MetalTensor)::Float32
    ld = logits.data
    td = targets.data  # stays on GPU
    dims = size(ld)
    batch, seq_len, vocab_size = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len

    # Permute to (vocab, batch, seq) for contiguous vocab access on GPU
    perm = permutedims(ld, (3, 1, 2))

    # GPU softmax along vocab dimension (dim 1)
    flat = reshape(perm, vocab_size, num_tokens)
    mx = maximum(flat; dims=1)  # (1, num_tokens)
    shifted = flat .- mx
    log_sum_exp = log.(sum(exp.(shifted); dims=1))  # (1, num_tokens)

    # Gather target logits using one-hot on GPU
    # targets is (batch, seq) with 0-based indices
    td_flat = reshape(td, num_tokens)  # (num_tokens,)
    # Build one-hot mask on GPU: (vocab, num_tokens)
    indices_gpu = reshape(td_flat, 1, num_tokens)  # (1, num_tokens)
    vocab_range = _get_ce_vocab_range(vocab_size)  # (vocab,)
    vocab_mat = reshape(vocab_range, vocab_size, 1)  # (vocab, 1)
    one_hot = Float32.(vocab_mat .== indices_gpu)  # (vocab, num_tokens) broadcast comparison

    # target_logits = sum(one_hot .* shifted_logits) per token
    target_shifted = sum(one_hot .* shifted; dims=1)  # (1, num_tokens)

    # loss = -mean(target_shifted - log_sum_exp)
    per_token_loss = .-target_shifted .+ log_sum_exp  # (1, num_tokens)
    Float32(sum(per_token_loss)) / Float32(num_tokens)
end

# Cross-entropy loss sum tensor on GPU (no scalar host readback).
function cross_entropy_loss_sum_tensor_mtl(logits::MetalTensor, targets::MetalTensor)::MetalTensor
    ld = logits.data
    td = targets.data
    dims = size(ld)
    batch, seq_len, vocab_size = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len

    perm = permutedims(ld, (3, 1, 2))
    flat = reshape(perm, vocab_size, num_tokens)
    mx = maximum(flat; dims=1)
    shifted = flat .- mx
    log_sum_exp = log.(sum(exp.(shifted); dims=1))

    td_flat = reshape(td, num_tokens)
    indices_gpu = reshape(td_flat, 1, num_tokens)
    vocab_range = _get_ce_vocab_range(vocab_size)
    vocab_mat = reshape(vocab_range, vocab_size, 1)
    one_hot = Float32.(vocab_mat .== indices_gpu)

    target_shifted = sum(one_hot .* shifted; dims=1)
    per_token_loss = .-target_shifted .+ log_sum_exp
    loss_sum = sum(per_token_loss; dims=2)
    MetalTensor(loss_sum, logits.dtype)
end

# Fused cross-entropy loss + gradient on GPU — fully on-device
function cross_entropy_loss_grad_mtl!(logits::MetalTensor, targets::MetalTensor)::Tuple{Float32, MetalTensor}
    ld = logits.data
    td = targets.data
    dims = size(ld)
    batch, seq_len, vocab_size = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len

    # Permute to (vocab, batch, seq)
    perm = permutedims(ld, (3, 1, 2))
    flat = reshape(perm, vocab_size, num_tokens)

    # GPU softmax
    mx = maximum(flat; dims=1)
    shifted = flat .- mx
    e = exp.(shifted)
    s = sum(e; dims=1)
    probs = e ./ s  # (vocab, num_tokens) — this IS the gradient (softmax output)

    # One-hot target mask on GPU
    td_flat = reshape(td, 1, num_tokens)
    vocab_range = _get_ce_vocab_range(vocab_size)
    vocab_mat = reshape(vocab_range, vocab_size, 1)
    one_hot = Float32.(vocab_mat .== td_flat)  # (vocab, num_tokens)

    # Loss: -log(prob[target])
    target_probs = sum(one_hot .* probs; dims=1)  # (1, num_tokens)
    loss = Float32(-sum(log.(max.(target_probs, 1f-12)))) / Float32(num_tokens)

    # Gradient: probs - one_hot, scaled by 1/num_tokens
    sc = 1f0 / Float32(num_tokens)
    grad_flat = (probs .- one_hot) .* sc  # (vocab, num_tokens)

    # Reshape back to (vocab, batch, seq) then permute to (batch, seq, vocab)
    grad_perm = reshape(grad_flat, vocab_size, batch, seq_len)
    gd = permutedims(grad_perm, (2, 3, 1))
    loss, MetalTensor(gd, logits.dtype)
end

# Cross-entropy gradient only (no host scalar loss readback).
function cross_entropy_grad_only_mtl!(logits::MetalTensor, targets::MetalTensor)::MetalTensor
    ld = logits.data
    td = targets.data
    dims = size(ld)
    batch, seq_len, vocab_size = dims[1], dims[2], dims[3]
    num_tokens = batch * seq_len

    perm = permutedims(ld, (3, 1, 2))
    flat = reshape(perm, vocab_size, num_tokens)

    mx = maximum(flat; dims=1)
    shifted = flat .- mx
    e = exp.(shifted)
    s = sum(e; dims=1)
    probs = e ./ s

    td_flat = reshape(td, 1, num_tokens)
    vocab_range = _get_ce_vocab_range(vocab_size)
    vocab_mat = reshape(vocab_range, vocab_size, 1)
    one_hot = Float32.(vocab_mat .== td_flat)

    sc = 1f0 / Float32(num_tokens)
    grad_flat = (probs .- one_hot) .* sc
    grad_perm = reshape(grad_flat, vocab_size, batch, seq_len)
    gd = permutedims(grad_perm, (2, 3, 1))
    MetalTensor(gd, logits.dtype)
end

# Gradient clipping on GPU
function clip_grad_mtl!(t::MetalTensor, clip_norm::Float32)::Float32
    clip_norm <= 0f0 && return 0f0
    sum_sq = sum(t.data .* t.data)
    norm = sqrt(sum_sq)
    if norm > clip_norm
        sc = clip_norm / (norm + 1f-12)
        t.data .*= sc
    end
    norm
end

# AdamW update on GPU using broadcast
function _adamw_update_mtl!(pd::MtlArray{Float32}, md::MtlVector{Float32}, vd::MtlVector{Float32},
                            b1::Float32, one_minus_b1::Float32, b2::Float32, one_minus_b2::Float32,
                            lr::Float32, m_corr::Float32, v_corr::Float32,
                            eps::Float32, wd::Float32, gd::MtlArray{Float32})
    gd_flat = reshape(gd, length(gd))
    # m = beta1*m + (1-beta1)*g
    md .= b1 .* md .+ one_minus_b1 .* gd_flat
    # v = beta2*v + (1-beta2)*g^2
    vd .= b2 .* vd .+ one_minus_b2 .* gd_flat .* gd_flat
    # w -= lr * (m_hat/(sqrt(v_hat)+eps) + wd*w)
    pd_flat = reshape(pd, length(pd))
    pd_flat .-= lr .* (md .* m_corr ./ (sqrt.(vd .* v_corr) .+ eps) .+ wd .* pd_flat)
end

function gpu_train_step!(t::MetalTrainer, input::MetalTensor, targets::MetalTensor)::Float32
    inference_mode() && error("gpu_train_step! requires inference_mode=false")
    t.step += 1
    params = t.params_cache

    # Zero gradients
    for param in params
        if param.grad === nothing
            param.grad = Metal.zeros(Float32, size(param.data)...)
        else
            param.grad .= 0f0
        end
    end

    # Forward
    logits = gpu_forward(t.model, input)

    # Loss + grad
    loss, grad_output = cross_entropy_loss_grad_mtl!(logits, targets)
    total_loss = loss

    # Clip and backward
    clip_grad_mtl!(grad_output, t.config.grad_clip)
    gpu_backward(t.model, grad_output)

    # Auxiliary losses
    if t.config.routing_mode == TopKMode
        aux_loss = apply_aux_loss_mtl!(t.model, t.config.aux_alpha)
        z_loss = apply_z_loss_mtl!(t.model, t.config.z_loss_weight)
        total_loss += aux_loss + z_loss
    end

    # AdamW update
    lr = get_lr_mtl(t)
    step = t.step
    b1 = t.config.beta1
    b2 = t.config.beta2
    eps = t.config.eps
    wd = t.config.weight_decay
    m_corr = 1f0 / (1f0 - b1^Float32(step))
    v_corr = 1f0 / (1f0 - b2^Float32(step))
    one_minus_b1 = 1f0 - b1
    one_minus_b2 = 1f0 - b2

    for (i, param) in enumerate(params)
        if param.grad !== nothing
            _adamw_update_mtl!(param.data, t.states[i].m, t.states[i].v,
                               b1, one_minus_b1, b2, one_minus_b2,
                               lr, m_corr, v_corr, eps, wd, param.grad)
        end
    end

    total_loss
end

# Benchmark path: GPU train step without any host scalar readback.
function gpu_train_step_no_readback!(t::MetalTrainer, input::MetalTensor, targets::MetalTensor)
    inference_mode() && error("gpu_train_step_no_readback! requires inference_mode=false")
    t.step += 1
    params = t.params_cache

    for param in params
        if param.grad === nothing
            param.grad = Metal.zeros(Float32, size(param.data)...)
        else
            param.grad .= 0f0
        end
    end

    logits = gpu_forward(t.model, input)
    grad_output = cross_entropy_grad_only_mtl!(logits, targets)
    gpu_backward(t.model, grad_output)

    lr = get_lr_mtl(t)
    step = t.step
    b1 = t.config.beta1
    b2 = t.config.beta2
    eps = t.config.eps
    wd = t.config.weight_decay
    m_corr = 1f0 / (1f0 - b1^Float32(step))
    v_corr = 1f0 / (1f0 - b2^Float32(step))
    one_minus_b1 = 1f0 - b1
    one_minus_b2 = 1f0 - b2

    for (i, param) in enumerate(params)
        if param.grad !== nothing
            _adamw_update_mtl!(
                param.data,
                t.states[i].m,
                t.states[i].v,
                b1,
                one_minus_b1,
                b2,
                one_minus_b2,
                lr,
                m_corr,
                v_corr,
                eps,
                wd,
                param.grad,
            )
        end
    end

    nothing
end

# Benchmark path: full-model forward + CE loss sum on GPU, no readback.
function gpu_forward_ce_no_readback!(t::MetalTrainer, input::MetalTensor, targets::MetalTensor)
    logits = gpu_forward(t.model, input)
    _ = cross_entropy_loss_sum_tensor_mtl(logits, targets)
    nothing
end
