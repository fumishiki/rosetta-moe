# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal_moe.jl — Mixture of Experts on GPU
#
# Router uses GPU broadcast for softmax and top-k selection.
# Expert dispatch uses one-hot gather/scatter on GPU (matmul-based).
# NO Array() conversions — all data stays on GPU.

# =============================================================================
# MetalRouter — GPU-native routing with top-k via broadcast
# =============================================================================

mutable struct MetalRouter <: AbstractMetalLayer
    gate::MetalLinear
    n_experts::Int
    top_k::Int
    last_input::Union{MetalTensor,Nothing}
    # All routing data stays on GPU
    last_topk_weights::Union{MtlMatrix{Float32}, Nothing}  # (num_tokens, top_k) GPU
    last_topk_onehot::Union{Vector{MtlMatrix{Float32}}, Nothing}  # top_k x (num_tokens, n_experts) GPU
    last_gate_probs::Union{MtlMatrix{Float32}, Nothing}    # (num_tokens, n_experts) GPU
    last_logits_mtl::Union{MtlMatrix{Float32}, Nothing}    # (num_tokens, n_experts) GPU
    routing_mode::RoutingMode
    expert_bias::Vector{Float32}
    relu_lambda_l1::Float32
    last_avg_active::Float32
    last_expert_counts_mtl::Union{MtlVector{Float32}, Nothing}
    last_relu_sum::Float32
    infer_remaining_cache::Union{MtlMatrix{Float32}, Nothing}
    infer_expert_weights_cache::Union{MtlMatrix{Float32}, Nothing}
    infer_selected_sum_cache::Union{MtlMatrix{Float32}, Nothing}
end

function MetalRouter(hidden_dim::Int, n_experts::Int, top_k::Int;
                     routing_mode::RoutingMode=TopKMode, relu_lambda_l1::Float32=0.01f0)
    MetalRouter(MetalLinear(hidden_dim, n_experts, false), n_experts, top_k,
                nothing, nothing, nothing, nothing, nothing,
                routing_mode, zeros(Float32, n_experts), relu_lambda_l1,
                0f0, nothing, 0f0, nothing, nothing, nothing)
end

function gpu_forward(r::MetalRouter, input::MetalTensor)
    if !inference_mode()
        r.last_input = input
    end
    dims = size(input.data)
    leading = dims[1:end-1]
    num_tokens = prod(leading)
    feat_dim = dims[end]

    flat_input = MetalTensor(reshape(input.data, num_tokens, feat_dim), input.dtype)
    gate_out = gpu_forward(r.gate, flat_input)

    # gate_out.data: (num_tokens, n_experts) on GPU
    logits_mtl = reshape(gate_out.data, num_tokens, r.n_experts)
    if !inference_mode()
        r.last_logits_mtl = copy(logits_mtl)
    end

    # GPU softmax along expert dimension (dim 2)
    mx = maximum(logits_mtl; dims=2)  # (num_tokens, 1)
    shifted = logits_mtl .- mx
    e = exp.(shifted)
    s = sum(e; dims=2)  # (num_tokens, 1)
    probs = e ./ s  # (num_tokens, n_experts)
    if !inference_mode()
        r.last_gate_probs = probs
    end

    n_exp = r.n_experts
    top_k = r.top_k

    # GPU top-k via iterative argmax
    # For each k, find the max of remaining probabilities, build one-hot mask
    remaining = copy(probs)  # (num_tokens, n_experts) — working copy
    topk_weights = Metal.zeros(Float32, num_tokens, top_k)
    topk_onehots = Vector{MtlMatrix{Float32}}(undef, top_k)

    for k in 1:top_k
        # Find max value per token
        max_vals = maximum(remaining; dims=2)  # (num_tokens, 1)
        # Build one-hot: which expert has the max
        # Handle ties by taking first match (sufficient for routing)
        is_max = Float32.(remaining .== max_vals)  # (num_tokens, n_experts)
        # If multiple tied, zero out all but the first using cumsum trick
        cummax = cumsum(is_max; dims=2)
        one_hot_k = is_max .* Float32.(cummax .== 1f0)  # first match only

        topk_onehots[k] = one_hot_k  # (num_tokens, n_experts)

        # Extract weight for this k
        weight_k = sum(one_hot_k .* probs; dims=2)  # (num_tokens, 1)
        topk_weights[:, k:k] .= weight_k

        # Zero out selected expert for next iteration
        remaining = remaining .* (1f0 .- one_hot_k)
    end

    if !inference_mode()
        r.last_topk_onehot = topk_onehots
    end

    # Normalize weights per token
    w_sum = sum(topk_weights; dims=2)  # (num_tokens, 1)
    w_sum_safe = max.(w_sum, 1f-12)
    topk_weights = topk_weights ./ w_sum_safe
    if !inference_mode()
        r.last_topk_weights = topk_weights
    end

    # Expert counts on GPU
    # Sum one-hot per expert across tokens for all k slots
    total_onehot = Metal.zeros(Float32, num_tokens, n_exp)
    for k in 1:top_k
        total_onehot .+= topk_onehots[k]
    end
    if !inference_mode()
        r.last_expert_counts_mtl = reshape(sum(total_onehot; dims=1), n_exp)
    end

    return topk_weights, topk_onehots
end

function _router_expert_weights_inference(r::MetalRouter, input::MetalTensor)
    dims = size(input.data)
    leading = dims[1:end-1]
    num_tokens = prod(leading)
    feat_dim = dims[end]

    flat_input = MetalTensor(reshape(input.data, num_tokens, feat_dim), input.dtype)
    gate_out = gpu_forward(r.gate, flat_input)
    logits_mtl = reshape(gate_out.data, num_tokens, r.n_experts)

    mx = maximum(logits_mtl; dims=2)
    shifted = logits_mtl .- mx
    e = exp.(shifted)
    s = sum(e; dims=2)
    probs = e ./ s

    if r.infer_remaining_cache === nothing || size(r.infer_remaining_cache) != (num_tokens, r.n_experts)
        r.infer_remaining_cache = Metal.zeros(Float32, num_tokens, r.n_experts)
    end
    if r.infer_expert_weights_cache === nothing || size(r.infer_expert_weights_cache) != (num_tokens, r.n_experts)
        r.infer_expert_weights_cache = Metal.zeros(Float32, num_tokens, r.n_experts)
    end
    if r.infer_selected_sum_cache === nothing || size(r.infer_selected_sum_cache) != (num_tokens, 1)
        r.infer_selected_sum_cache = Metal.zeros(Float32, num_tokens, 1)
    end

    remaining = r.infer_remaining_cache
    expert_weights = r.infer_expert_weights_cache
    selected_sum = r.infer_selected_sum_cache
    remaining .= probs
    expert_weights .= 0f0
    selected_sum .= 0f0

    for _ in 1:r.top_k
        max_vals = maximum(remaining; dims=2)
        one_hot = Float32.(remaining .== max_vals)
        w = sum(one_hot .* probs; dims=2)
        expert_weights .+= one_hot .* w
        selected_sum .+= w
        remaining .*= (1f0 .- one_hot)
    end

    expert_weights ./= max.(selected_sum, 1f-12)
    expert_weights
end

function gpu_backward(r::MetalRouter, grad_output::MetalTensor)
    sz = size(r.last_input.data)
    zeros_mtensor(sz...)
end

gpu_parameters(r::MetalRouter) = gpu_parameters(r.gate)

# Aux loss (computed on GPU from cached values)
function compute_aux_loss_mtl(r::MetalRouter, alpha::Float32)
    r.last_gate_probs === nothing && return 0f0
    r.last_expert_counts_mtl === nothing && return 0f0

    probs = r.last_gate_probs  # (num_tokens, n_experts) GPU
    expert_counts = r.last_expert_counts_mtl  # (n_experts,) GPU
    n_exp = r.n_experts
    num_tokens_f = Float32(size(probs, 1))
    top_k = r.top_k

    total_assign = sum(expert_counts)
    total_assign_s = Float32(total_assign)
    total_assign_s == 0f0 && return 0f0

    f = expert_counts ./ total_assign_s  # (n_experts,)
    P = reshape(sum(probs; dims=1), n_exp) ./ num_tokens_f  # (n_experts,)

    Float32(sum(f .* P)) * alpha * Float32(n_exp)
end

# Z-loss with gradient (writes to gate weight grad on GPU)
function compute_z_loss_with_grad_mtl!(r::MetalRouter, z_weight::Float32)::Float32
    r.last_logits_mtl === nothing && return 0f0
    r.last_gate_probs === nothing && return 0f0
    r.last_input === nothing && return 0f0

    logits = r.last_logits_mtl  # (num_tokens, n_experts) GPU
    gate_probs = r.last_gate_probs
    num_tokens = size(logits, 1)
    n_exp = r.n_experts

    # logsumexp per token on GPU
    mx = maximum(logits; dims=2)  # (num_tokens, 1)
    lse = mx .+ log.(sum(exp.(logits .- mx); dims=2))  # (num_tokens, 1)

    # z_loss = z_weight * mean(lse^2)
    z_loss = Float32(z_weight * sum(lse .* lse) / Float32(num_tokens))

    # Gradient w.r.t. logits: d(z_loss)/d(logits) = z_weight * 2 * lse * softmax(logits) / num_tokens
    coeff = z_weight * 2f0 / Float32(num_tokens)
    grad_logits = coeff .* lse .* gate_probs  # (num_tokens, n_experts) broadcast

    # Backprop to gate weights on GPU: W_grad = grad_logits^T @ input
    input_data = r.last_input.data
    dims_in = size(input_data)
    hidden_dim = dims_in[end]
    flat_input = reshape(input_data, num_tokens, hidden_dim)  # stays on GPU

    w_grad = transpose(grad_logits) * flat_input  # (n_exp, hidden_dim) — MPS matmul on GPU
    gate_weight = r.gate.weight
    if gate_weight.grad === nothing
        gate_weight.grad = w_grad
    else
        gate_weight.grad .+= w_grad
    end

    z_loss
end

# Aux loss with gradient (on GPU)
function compute_aux_loss_with_grad_mtl!(r::MetalRouter, alpha::Float32)::Float32
    r.last_expert_counts_mtl === nothing && return 0f0
    r.last_gate_probs === nothing && return 0f0
    r.last_input === nothing && return 0f0

    gate_probs = r.last_gate_probs  # (num_tokens, n_experts) GPU
    n_exp = r.n_experts
    num_tokens = size(gate_probs, 1)
    expert_counts = r.last_expert_counts_mtl  # (n_experts,) GPU

    total_assign = sum(expert_counts)
    Float32(total_assign) == 0f0 && return 0f0

    inv_total = 1f0 / Float32(total_assign)
    f = expert_counts .* inv_total  # (n_experts,)

    inv_nt = 1f0 / Float32(num_tokens)
    P = reshape(sum(gate_probs; dims=1), n_exp) .* inv_nt  # (n_experts,)

    aux_loss = Float32(sum(f .* P)) * alpha * Float32(n_exp)

    # Gradient w.r.t. logits (through softmax)
    # d(aux)/d(probs[t,e]) = alpha * n_exp * f[e] / num_tokens
    # d(probs)/d(logits) = probs * (delta - probs)  (softmax Jacobian)
    # Combined: grad_logits[t,e] = coeff * probs[t,e] * (f[e] - dot(f, probs[t,:]))
    coeff = alpha * Float32(n_exp) / Float32(num_tokens)
    f_row = reshape(f, 1, n_exp)  # (1, n_experts) for broadcast

    # dot_fp[t] = sum_e(f[e] * probs[t, e]) per token
    dot_fp = sum(gate_probs .* f_row; dims=2)  # (num_tokens, 1)
    grad_logits = coeff .* gate_probs .* (f_row .- dot_fp)  # (num_tokens, n_experts) GPU

    # Backprop to gate weights on GPU
    input_data = r.last_input.data
    dims_in = size(input_data)
    hidden_dim = dims_in[end]
    flat_input = reshape(input_data, num_tokens, hidden_dim)

    w_grad = transpose(grad_logits) * flat_input  # (n_exp, hidden_dim) — MPS matmul
    gate_weight = r.gate.weight
    if gate_weight.grad === nothing
        gate_weight.grad = w_grad
    else
        gate_weight.grad .+= w_grad
    end

    aux_loss
end

# =============================================================================
# MetalMoELayer — GPU-native expert dispatch
# =============================================================================

mutable struct MetalMoELayer <: AbstractMetalLayer
    router::MetalRouter
    experts::Vector{MetalSwiGLU}
    hidden_dim::Int
    n_experts::Int
    top_k::Int
    # Cached for backward
    last_num_tokens::Int
    last_leading::Tuple
    last_expert_weights::Union{MtlMatrix{Float32}, Nothing}  # (num_tokens, n_experts)
    infer_out_cache::Union{MtlMatrix{Float32}, Nothing}
    infer_fused_gate_up_weight::Union{MtlMatrix{Float32}, Nothing}
    infer_proj_all_cache::Union{MtlMatrix{Float32}, Nothing}
    infer_tmp_out_cache::Union{MtlMatrix{Float32}, Nothing}
end

function MetalMoELayer(hidden_dim::Int, ffn_dim::Int, n_experts::Int, top_k::Int;
                       routing_mode::RoutingMode=TopKMode, relu_lambda_l1::Float32=0.01f0)
    experts = [MetalSwiGLU(hidden_dim, ffn_dim) for _ in 1:n_experts]
    MetalMoELayer(MetalRouter(hidden_dim, n_experts, top_k; routing_mode=routing_mode, relu_lambda_l1=relu_lambda_l1),
                  experts, hidden_dim, n_experts, top_k, 0, (), nothing, nothing, nothing, nothing, nothing)
end

function _ensure_moe_infer_fused_gate_up_weight!(m::MetalMoELayer)::MtlMatrix{Float32}
    ffn_dim = m.experts[1].ffn_dim
    rows = m.n_experts * 2 * ffn_dim
    if m.infer_fused_gate_up_weight === nothing || size(m.infer_fused_gate_up_weight) != (rows, m.hidden_dim)
        fused = Metal.zeros(Float32, rows, m.hidden_dim)
        for e_idx in 1:m.n_experts
            base = (e_idx - 1) * 2 * ffn_dim
            fused[base + 1:base + ffn_dim, :] .= m.experts[e_idx].w_gate.weight.data
            fused[base + ffn_dim + 1:base + 2 * ffn_dim, :] .= m.experts[e_idx].w_up.weight.data
        end
        m.infer_fused_gate_up_weight = fused
    end
    return m.infer_fused_gate_up_weight
end

function _moe_experts_fused_inference!(
    m::MetalMoELayer,
    flat_data::MtlMatrix{Float32},
    expert_weights::MtlMatrix{Float32},
    out_gpu::MtlMatrix{Float32},
    num_tokens::Int,
)
    ffn_dim = m.experts[1].ffn_dim
    fused_w = _ensure_moe_infer_fused_gate_up_weight!(m)
    proj_cols = m.n_experts * 2 * ffn_dim
    if m.infer_proj_all_cache === nothing || size(m.infer_proj_all_cache) != (num_tokens, proj_cols)
        m.infer_proj_all_cache = Metal.zeros(Float32, num_tokens, proj_cols)
    end
    proj_all = m.infer_proj_all_cache
    mul!(proj_all, flat_data, transpose(fused_w))

    if m.infer_tmp_out_cache === nothing || size(m.infer_tmp_out_cache) != (num_tokens, m.hidden_dim)
        m.infer_tmp_out_cache = Metal.zeros(Float32, num_tokens, m.hidden_dim)
    end
    tmp_out = m.infer_tmp_out_cache

    for e_idx in 1:m.n_experts
        base = (e_idx - 1) * 2 * ffn_dim
        gate = @view proj_all[:, base + 1:base + ffn_dim]
        up = @view proj_all[:, base + ffn_dim + 1:base + 2 * ffn_dim]
        gate .= gate ./ (1f0 .+ exp.(.-gate))
        gate .*= up

        mul!(tmp_out, gate, transpose(m.experts[e_idx].w_down.weight.data))
        w_e = expert_weights[:, e_idx]
        tmp_out .*= reshape(w_e, num_tokens, 1)
        out_gpu .+= tmp_out
    end

    return nothing
end

function gpu_forward(m::MetalMoELayer, input::MetalTensor)
    dims = size(input.data)
    leading = dims[1:end-1]
    num_tokens = prod(leading)
    hidden_dim = m.hidden_dim

    expert_weights = inference_mode() ? _router_expert_weights_inference(m.router, input) : begin
        topk_weights, topk_onehots = gpu_forward(m.router, input)

        # Per-token expert weights: (num_tokens, n_experts)
        # weight[t, e] = sum_k topk_weights[t, k] * onehot_k[t, e]
        # Dense weighted dispatch avoids variable-length gather/scatter and host sync.
        ew = Metal.zeros(Float32, num_tokens, m.n_experts)
        for k in 1:m.top_k
            ew .+= topk_onehots[k] .* reshape(topk_weights[:, k], num_tokens, 1)
        end
        ew
    end
    flat_data = reshape(input.data, num_tokens, hidden_dim)

    if !inference_mode()
        m.last_num_tokens = num_tokens
        m.last_leading = leading
    end

    if !inference_mode()
        m.last_expert_weights = expert_weights
    end

    # Output accumulator on GPU
    out_gpu = if inference_mode()
        if m.infer_out_cache === nothing || size(m.infer_out_cache) != (num_tokens, hidden_dim)
            m.infer_out_cache = Metal.zeros(Float32, num_tokens, hidden_dim)
        end
        m.infer_out_cache .= 0f0
        m.infer_out_cache
    else
        m.infer_fused_gate_up_weight = nothing
        m.infer_proj_all_cache = nothing
        m.infer_tmp_out_cache = nothing
        m.infer_out_cache = nothing
        Metal.zeros(Float32, num_tokens, hidden_dim)
    end
    if inference_mode()
        _moe_experts_fused_inference!(m, flat_data, expert_weights, out_gpu, num_tokens)
    else
        flat_input = MetalTensor(flat_data, input.dtype)
        for e_idx in 1:m.n_experts
            expert_out = gpu_forward(m.experts[e_idx], flat_input)
            w_e = expert_weights[:, e_idx]
            out_gpu .+= expert_out.data .* reshape(w_e, num_tokens, 1)
        end
    end

    MetalTensor(reshape(out_gpu, leading..., hidden_dim), input.dtype)
end

function gpu_backward(m::MetalMoELayer, grad_output::MetalTensor)
    num_tokens = m.last_num_tokens
    hidden_dim = m.hidden_dim
    flat_grad = reshape(grad_output.data, num_tokens, hidden_dim)
    expert_weights = m.last_expert_weights
    expert_weights === nothing && error("MoE backward called before forward")

    grad_input = Metal.zeros(Float32, num_tokens, hidden_dim)

    for e_idx in 1:m.n_experts
        w_e = expert_weights[:, e_idx]
        expert_grad = MetalTensor(flat_grad .* reshape(w_e, num_tokens, 1), grad_output.dtype)
        grad_expert_input = gpu_backward(m.experts[e_idx], expert_grad)
        grad_input .+= grad_expert_input.data
    end

    MetalTensor(reshape(grad_input, m.last_leading..., hidden_dim), grad_output.dtype)
end

function gpu_parameters(m::MetalMoELayer)
    params = gpu_parameters(m.router)
    for e in m.experts
        append!(params, gpu_parameters(e))
    end
    params
end

aux_loss_mtl(m::MetalMoELayer, alpha::Float32) = compute_aux_loss_mtl(m.router, alpha)

# =============================================================================
# MetalTransformerBlock
# =============================================================================

mutable struct MetalTransformerBlock <: AbstractMetalLayer
    attn_norm::MetalRMSNorm
    attention::MetalMQAttention
    ffn_norm::MetalRMSNorm
    moe::MetalMoELayer
end

function MetalTransformerBlock(cfg::Config)
    MetalTransformerBlock(
        MetalRMSNorm(cfg.hidden_dim),
        MetalMQAttention(cfg.hidden_dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                         cfg.rope_base, cfg.rope_alpha),
        MetalRMSNorm(cfg.hidden_dim),
        MetalMoELayer(cfg.hidden_dim, cfg.ffn_dim, cfg.n_experts, cfg.top_k_experts)
    )
end

function gpu_forward(blk::MetalTransformerBlock, input::MetalTensor)
    normed = gpu_forward(blk.attn_norm, input)
    attn_out = gpu_forward(blk.attention, normed)
    x1 = input + attn_out  # residual

    normed2 = gpu_forward(blk.ffn_norm, x1)
    moe_out = gpu_forward(blk.moe, normed2)
    x1 + moe_out  # residual
end

function gpu_backward(blk::MetalTransformerBlock, grad_output::MetalTensor)
    # MoE path
    grad_moe_input = gpu_backward(blk.moe, gpu_backward(blk.ffn_norm, grad_output))
    grad_h = grad_output + grad_moe_input  # residual gradient

    # Attention path
    grad_attn_input = gpu_backward(blk.attention, gpu_backward(blk.attn_norm, grad_h))
    grad_h + grad_attn_input  # residual gradient
end

function gpu_parameters(blk::MetalTransformerBlock)
    vcat(gpu_parameters(blk.attn_norm), gpu_parameters(blk.attention),
         gpu_parameters(blk.ffn_norm), gpu_parameters(blk.moe))
end

aux_loss_mtl(blk::MetalTransformerBlock, alpha::Float32) = aux_loss_mtl(blk.moe, alpha)
