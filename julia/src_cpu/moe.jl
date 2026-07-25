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
    last_logits::Union{Matrix{Float32},Nothing}   # pre-softmax gate logits for z-loss
    # Pre-allocated buffers
    weights_buf::Matrix{Float32}                   # (num_tokens, top_k)
    softmax_buf::Matrix{Float32}                   # (num_tokens, n_experts)
    indices_buf::Vector{Vector{Int}}
    perm_buf::Vector{Int}
    grad_buf::Vector{Float32}                      # pre-allocated gradient buffer for backward
    # Pre-allocated buffers for compute_aux_loss
    aux_counts_buf::Vector{Float32}
    aux_probs_buf::Vector{Float32}
    # Pre-allocated buffers for z-loss
    z_loss_lse_buf::Vector{Float32}               # logsumexp values per token
    z_loss_grad_buf::Matrix{Float32}              # gradient w.r.t. logits
    # Routing mode fields
    routing_mode::RoutingMode
    expert_bias::Vector{Float32}         # BiasFree, all 0 init
    relu_lambda_l1::Float32
    last_avg_active::Float32             # ReMoE diagnostic
    last_expert_counts::Vector{Float32}  # BiasFree bias update
    last_relu_sum::Float32               # ReMoE L1 loss
end

function Router(hidden_dim::Int, n_experts::Int, top_k::Int; routing_mode::RoutingMode=TopKMode, relu_lambda_l1::Float32=0.01f0)
    @assert 1 <= top_k <= n_experts "invalid topK for router"
    Router(Linear(hidden_dim, n_experts, false), n_experts, top_k,
           nothing, nothing, Vector{Int}[], nothing, nothing,
           Matrix{Float32}(undef, 0, 0), Matrix{Float32}(undef, 0, 0),
           Vector{Int}[], Int[], Float32[],
           Float32[], Float32[],
           Float32[], Matrix{Float32}(undef, 0, 0),
           routing_mode, zeros(Float32, n_experts), relu_lambda_l1,
           0f0, zeros(Float32, n_experts), 0f0)
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
    # Store pre-softmax logits for z-loss
    if r.last_logits === nothing || size(r.last_logits) != (num_tokens, n_exp)
        r.last_logits = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    gd_flat = reshape(gd, num_tokens, n_exp)
    copyto!(r.last_logits, gd_flat)

    # Branch on routing mode
    if r.routing_mode == TopKMode
        _router_topk_impl!(r, gd_flat, num_tokens, n_exp, dtype)
    elseif r.routing_mode == BiasFreeMode
        _router_biasfree_impl!(r, gd_flat, num_tokens, n_exp, dtype)
    elseif r.routing_mode == ReLUMode
        _router_relu_impl!(r, gd_flat, num_tokens, n_exp, dtype)
    else
        error("Unknown routing mode: $(r.routing_mode)")
    end
end

# TopK routing: standard softmax + top-k selection
function _router_topk_impl!(r::Router, gd_flat::Matrix{Float32}, num_tokens::Int, n_exp::Int, dtype::DType)
    if size(r.softmax_buf) != (num_tokens, n_exp)
        r.softmax_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    sm_buf = r.softmax_buf
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

# BiasFree routing: softmax + bias-augmented selection
function _router_biasfree_impl!(r::Router, gd_flat::Matrix{Float32}, num_tokens::Int, n_exp::Int, dtype::DType)
    if size(r.softmax_buf) != (num_tokens, n_exp)
        r.softmax_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    sm_buf = r.softmax_buf
    softmax!(sm_buf, gd_flat)
    r.last_gate_prob = Tensor(sm_buf, dtype)

    # Selection score buffer: softmax + expert_bias
    selection_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    @inbounds for t in 1:num_tokens
        for e in 1:n_exp
            selection_buf[t, e] = sm_buf[t, e] + r.expert_bias[e]
        end
    end

    # Reset expert counts for this forward pass
    fill!(r.last_expert_counts, 0f0)

    # Weights buffer
    w_sz = (num_tokens, r.top_k)
    if size(r.weights_buf) != w_sz
        r.weights_buf = Matrix{Float32}(undef, w_sz...)
    end
    wb = r.weights_buf
    fill!(wb, 0f0)

    # Indices buffer
    top_k = r.top_k
    resize!(r.indices_buf, num_tokens)
    for i in 1:num_tokens
        if !isassigned(r.indices_buf, i)
            r.indices_buf[i] = Vector{Int}(undef, top_k)
        elseif length(r.indices_buf[i]) != top_k
            resize!(r.indices_buf[i], top_k)
        end
    end

    # Permutation buffer
    if length(r.perm_buf) < n_exp
        r.perm_buf = Vector{Int}(undef, n_exp)
    end
    perm = r.perm_buf

    # Select top-k by selection_score, but use original softmax for weights
    for t in 1:num_tokens
        @inbounds for e in 1:n_exp
            perm[e] = e
        end
        # Select by selection_buf (augmented)
        @inbounds for k in 1:top_k
            best = k
            for j in k+1:n_exp
                if selection_buf[t, perm[j]] > selection_buf[t, perm[best]]
                    best = j
                end
            end
            if best != k
                perm[k], perm[best] = perm[best], perm[k]
            end
            e_1based = perm[k]
            idx_0based = e_1based - 1
            r.indices_buf[t][k] = idx_0based
            # Use ORIGINAL softmax prob, NOT augmented score
            wb[t, k] = sm_buf[t, e_1based]
            # Track expert usage
            r.last_expert_counts[e_1based] += 1f0
        end
        # Normalize weights
        token_weights = @view wb[t, :]
        normalize_in_place!(token_weights)
    end

    r.last_indices = r.indices_buf
    weights = Tensor(wb, F32)
    r.last_weights = weights
    weights, r.last_indices
end

# ReLU routing: ReLU on raw logits, select active experts
function _router_relu_impl!(r::Router, gd_flat::Matrix{Float32}, num_tokens::Int, n_exp::Int, dtype::DType)
    # No softmax — use ReLU on raw logits
    # For each token: gate[e] = max(0, logits[e])
    # Select top_k among active experts (gate > 0)
    # If n_active == 0, fallback to best logit with weight 1.0
    # If n_active > top_k, select top_k by value
    # Renormalize, pad to top_k

    # Weights buffer
    w_sz = (num_tokens, r.top_k)
    if size(r.weights_buf) != w_sz
        r.weights_buf = Matrix{Float32}(undef, w_sz...)
    end
    wb = r.weights_buf
    fill!(wb, 0f0)

    # Indices buffer
    top_k = r.top_k
    resize!(r.indices_buf, num_tokens)
    for i in 1:num_tokens
        if !isassigned(r.indices_buf, i)
            r.indices_buf[i] = Vector{Int}(undef, top_k)
        elseif length(r.indices_buf[i]) != top_k
            resize!(r.indices_buf[i], top_k)
        end
        fill!(r.indices_buf[i], 0)  # default 0-based index
    end

    # Temp buffer for active expert indices per token
    active_buf = Vector{Tuple{Int,Float32}}(undef, n_exp)
    total_active_sum = 0f0

    for t in 1:num_tokens
        # Collect active experts (ReLU > 0)
        n_active = 0
        @inbounds for e in 1:n_exp
            val = max(0f0, gd_flat[t, e])
            if val > 0f0
                n_active += 1
                active_buf[n_active] = (e, val)
            end
        end
        total_active_sum += Float32(n_active)

        if n_active == 0
            # Fallback: select best expert by raw logit value
            best_e = 1
            best_val = gd_flat[t, 1]
            @inbounds for e in 2:n_exp
                if gd_flat[t, e] > best_val
                    best_e = e
                    best_val = gd_flat[t, e]
                end
            end
            r.indices_buf[t][1] = best_e - 1  # 0-based
            wb[t, 1] = 1f0
            # Rest are already 0
        elseif n_active <= top_k
            # Use all active experts
            @inbounds for k in 1:n_active
                e_1based, val = active_buf[k]
                r.indices_buf[t][k] = e_1based - 1
                wb[t, k] = val
            end
            # Normalize
            token_weights = @view wb[t, 1:n_active]
            normalize_in_place!(token_weights)
            # Pad rest with 0 (already filled)
        else
            # n_active > top_k: select top_k by value
            # Partial sort: find top_k largest values
            @inbounds for k in 1:top_k
                best = k
                for j in k+1:n_active
                    if active_buf[j][2] > active_buf[best][2]
                        best = j
                    end
                end
                if best != k
                    active_buf[k], active_buf[best] = active_buf[best], active_buf[k]
                end
                e_1based, val = active_buf[k]
                r.indices_buf[t][k] = e_1based - 1
                wb[t, k] = val
            end
            # Normalize
            token_weights = @view wb[t, :]
            normalize_in_place!(token_weights)
        end
    end

    # Store average active experts for diagnostics
    r.last_avg_active = total_active_sum / Float32(num_tokens)

    # Store sum of ReLU values for L1 loss computation
    relu_sum = 0f0
    @inbounds for t in 1:num_tokens
        for e in 1:n_exp
            relu_sum += max(0f0, gd_flat[t, e])
        end
    end
    r.last_relu_sum = relu_sum

    # No last_gate_prob for ReLU mode (not used)
    r.last_gate_prob = nothing

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

# Update expert bias for BiasFree routing
function update_expert_bias!(r::Router, gamma::Float32)
    r.routing_mode != BiasFreeMode && return
    total_assignments = sum(r.last_expert_counts)
    total_assignments <= 0f0 && return

    n_exp = r.n_experts
    target = 1f0 / Float32(n_exp)

    @inbounds for e in 1:n_exp
        f_e = r.last_expert_counts[e] / total_assignments
        r.expert_bias[e] += gamma * sign(target - f_e)
    end
end

# Set routing mode
function set_routing_mode!(r::Router, mode::RoutingMode)
    r.routing_mode = mode
end

# Compute ReLU L1 loss and gradient for ReMoE routing
function compute_relu_l1_loss_with_grad!(r::Router)::Float32
    r.routing_mode != ReLUMode && return 0f0
    r.last_logits === nothing && return 0f0
    r.last_input === nothing && return 0f0

    logits = r.last_logits
    num_tokens = size(logits, 1)
    n_exp = r.n_experts
    lambda = r.relu_lambda_l1

    # L_l1 = lambda * mean_t(sum_e relu(logits[t,e]))
    # Already computed in _router_relu_impl! and stored in r.last_relu_sum
    l1_loss = lambda * r.last_relu_sum / Float32(num_tokens)

    # Gradient: grad_logits[t,e] = lambda * (1/B) * (logits[t,e] > 0 ? 1 : 0)
    if size(r.z_loss_grad_buf) != (num_tokens, n_exp)
        r.z_loss_grad_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    grad_logits = r.z_loss_grad_buf
    coeff = lambda / Float32(num_tokens)

    @inbounds for t in 1:num_tokens
        for e in 1:n_exp
            grad_logits[t, e] = logits[t, e] > 0f0 ? coeff : 0f0
        end
    end

    # Backprop to gate weights: gate_weight_grad = grad_logits.T @ last_input
    input_data = r.last_input.data
    dims = size(input_data)
    leading, _, hidden_dim = _split_last(dims)
    flat_input = reshape(input_data, num_tokens, hidden_dim)

    gate_weight = r.gate.weight
    if gate_weight.grad === nothing
        gate_weight.grad = zeros(Float32, size(gate_weight.data)...)
    end

    # Manual transpose matmul: result[e, h] = sum_t(grad_logits[t, e] * flat_input[t, h])
    @inbounds for e in 1:n_exp
        for h in 1:hidden_dim
            acc = 0f0
            @simd for t in 1:num_tokens
                acc += grad_logits[t, e] * flat_input[t, h]
            end
            gate_weight.grad[e, h] += acc
        end
    end

    # If gate has bias, accumulate gradient
    if r.gate.bias !== nothing
        if r.gate.bias.grad === nothing
            r.gate.bias.grad = zeros(Float32, size(r.gate.bias.data)...)
        end
        @inbounds for e in 1:n_exp
            acc = 0f0
            @simd for t in 1:num_tokens
                acc += grad_logits[t, e]
            end
            r.gate.bias.grad[e] += acc
        end
    end

    l1_loss
end

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

# Router Z-loss (ST-MoE): L_z = z_weight * (1/B) * sum_i(logsumexp(logits_i)^2)
# Gradient: dL_z/d(logits[i,j]) = z_weight * (2/B) * logsumexp(logits_i) * gate_probs[i,j]
# Backprop to gate weights: gate_weight_grad += grad_logits.T @ last_input
function compute_z_loss_with_grad!(r::Router, z_weight::Float32)::Float32
    r.last_logits === nothing && return 0f0
    r.last_gate_prob === nothing && return 0f0
    r.last_input === nothing && return 0f0

    logits = r.last_logits
    gate_probs = r.last_gate_prob.data
    num_tokens = size(logits, 1)
    n_exp = r.n_experts

    # Ensure logsumexp buffer
    if length(r.z_loss_lse_buf) != num_tokens
        r.z_loss_lse_buf = Vector{Float32}(undef, num_tokens)
    end
    lse = r.z_loss_lse_buf

    # Compute logsumexp for each token using max-subtract trick
    @inbounds for t in 1:num_tokens
        mx = -Inf32
        for e in 1:n_exp
            val = logits[t, e]
            mx = ifelse(val > mx, val, mx)
        end
        sum_exp = 0f0
        for e in 1:n_exp
            sum_exp += exp(logits[t, e] - mx)
        end
        lse[t] = mx + log(sum_exp)
    end

    # Z-loss: (1/B) * sum(lse^2)
    z_loss_sum = 0f0
    @inbounds @simd for t in 1:num_tokens
        z_loss_sum += lse[t] * lse[t]
    end
    z_loss = z_weight * z_loss_sum / Float32(num_tokens)

    # Gradient w.r.t. logits: (2/B) * lse * gate_probs
    if size(r.z_loss_grad_buf) != (num_tokens, n_exp)
        r.z_loss_grad_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    grad_logits = r.z_loss_grad_buf
    coeff = z_weight * 2f0 / Float32(num_tokens)
    @inbounds for t in 1:num_tokens
        lse_t = lse[t]
        for e in 1:n_exp
            grad_logits[t, e] = coeff * lse_t * gate_probs[t, e]
        end
    end

    # Backprop to gate weights: gate_weight_grad = grad_logits.T @ last_input
    # last_input shape: (batch, seq_len, hidden_dim) -> flatten to (num_tokens, hidden_dim)
    input_data = r.last_input.data
    dims = size(input_data)
    leading, _, hidden_dim = _split_last(dims)
    flat_input = reshape(input_data, num_tokens, hidden_dim)

    # gate_weight_grad[e, h] = sum_t(grad_logits[t, e] * flat_input[t, h])
    # Transpose: grad_logits is (num_tokens, n_exp), want (n_exp, num_tokens) @ (num_tokens, hidden_dim)
    # => (n_exp, hidden_dim)
    # Using transpose: grad_logits' @ flat_input
    gate_weight = r.gate.weight
    if gate_weight.grad === nothing
        gate_weight.grad = zeros(Float32, size(gate_weight.data)...)
    end
    # Accumulate: weight.grad += grad_logits' * flat_input
    # Manual transpose matmul: result[e, h] = sum_t(grad_logits[t, e] * flat_input[t, h])
    @inbounds for e in 1:n_exp
        for h in 1:hidden_dim
            acc = 0f0
            @simd for t in 1:num_tokens
                acc += grad_logits[t, e] * flat_input[t, h]
            end
            gate_weight.grad[e, h] += acc
        end
    end

    # If gate has bias, accumulate gradient to bias
    if r.gate.bias !== nothing
        if r.gate.bias.grad === nothing
            r.gate.bias.grad = zeros(Float32, size(r.gate.bias.data)...)
        end
        @inbounds for e in 1:n_exp
            acc = 0f0
            @simd for t in 1:num_tokens
                acc += grad_logits[t, e]
            end
            r.gate.bias.grad[e] += acc
        end
    end

    z_loss
end

# Aux-loss with gradient backprop (TopK mode)
# L_aux = alpha * N * sum_e(f_e * P_e)
# where f_e = fraction of tokens routed to expert e,
#       P_e = mean gate probability for expert e,
#       N   = number of experts.
# Gradient: dL_aux/d(logits[t,e]) = alpha * N / num_tokens * softmax(logits[t,:])[e] * (f_e - dot(f, softmax(logits[t,:])))
# Backprop to gate weights: gate_weight_grad += grad_logits.T @ last_input
function compute_aux_loss_with_grad!(r::Router, alpha::Float32)::Float32
    r.last_expert_counts === nothing && return 0f0
    r.last_gate_prob === nothing && return 0f0
    r.last_input === nothing && return 0f0

    expert_counts = r.last_expert_counts
    gate_probs = r.last_gate_prob.data
    n_exp = r.n_experts

    # Compute total assignments and f_e values
    total_assign = sum(expert_counts)
    total_assign == 0f0 && return 0f0

    num_tokens = size(gate_probs, 1)
    batch_seq = num_tokens

    # Normalize expert counts to get f_e
    inv_total = 1f0 / total_assign
    # Create normalized f array
    if length(r.aux_counts_buf) != n_exp
        r.aux_counts_buf = Vector{Float32}(undef, n_exp)
    end
    f = r.aux_counts_buf
    @inbounds @simd for e in 1:n_exp
        f[e] = expert_counts[e] * inv_total
    end

    # Compute P_e (mean gate probability per expert)
    if length(r.aux_probs_buf) != n_exp
        r.aux_probs_buf = Vector{Float32}(undef, n_exp)
    end
    P = r.aux_probs_buf
    fill!(P, 0f0)
    @inbounds for t in 1:num_tokens
        @simd for e in 1:n_exp
            P[e] += gate_probs[t, e]
        end
    end
    inv_nt = 1f0 / Float32(num_tokens)
    @inbounds @simd for e in 1:n_exp
        P[e] *= inv_nt
    end

    # Compute aux_loss scalar: alpha * N * sum_e(f_e * P_e)
    aux_loss = 0f0
    @inbounds @simd for e in 1:n_exp
        aux_loss += f[e] * P[e]
    end
    aux_loss_value = aux_loss * alpha * Float32(n_exp)

    # Gradient w.r.t. logits: alpha * N / num_tokens * softmax[t,e] * (f[e] - dot(f, softmax[t,:]))
    if size(r.z_loss_grad_buf) != (num_tokens, n_exp)
        r.z_loss_grad_buf = Matrix{Float32}(undef, num_tokens, n_exp)
    end
    grad_logits = r.z_loss_grad_buf
    coeff = alpha * Float32(n_exp) / Float32(batch_seq)
    @inbounds for t in 1:num_tokens
        # Compute dot(f, probs[t,:])
        dot_fp_t = 0f0
        @simd for e in 1:n_exp
            dot_fp_t += f[e] * gate_probs[t, e]
        end
        # grad_logits[t,e] = coeff * probs[t,e] * (f[e] - dot_fp_t)
        @simd for e in 1:n_exp
            grad_logits[t, e] = coeff * gate_probs[t, e] * (f[e] - dot_fp_t)
        end
    end

    # Backprop to gate weights: gate_weight_grad = grad_logits.T @ last_input
    # last_input shape: (batch, seq_len, hidden_dim) -> flatten to (num_tokens, hidden_dim)
    input_data = r.last_input.data
    dims = size(input_data)
    leading, _, hidden_dim = _split_last(dims)
    flat_input = reshape(input_data, num_tokens, hidden_dim)

    # gate_weight_grad[e, h] = sum_t(grad_logits[t, e] * flat_input[t, h])
    gate_weight = r.gate.weight
    if gate_weight.grad === nothing
        gate_weight.grad = zeros(Float32, size(gate_weight.data)...)
    end
    # Accumulate: weight.grad += grad_logits' * flat_input
    @inbounds for e in 1:n_exp
        for h in 1:hidden_dim
            acc = 0f0
            @simd for t in 1:num_tokens
                acc += grad_logits[t, e] * flat_input[t, h]
            end
            gate_weight.grad[e, h] += acc
        end
    end

    # If gate has bias, accumulate gradient to bias
    if r.gate.bias !== nothing
        if r.gate.bias.grad === nothing
            r.gate.bias.grad = zeros(Float32, size(r.gate.bias.data)...)
        end
        @inbounds for e in 1:n_exp
            acc = 0f0
            @simd for t in 1:num_tokens
                acc += grad_logits[t, e]
            end
            r.gate.bias.grad[e] += acc
        end
    end

    aux_loss_value
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
    bwd_expert_grad_bufs::Vector{Matrix{Float32}}   # per-expert grad batch buffers
    bwd_expert_input_bufs::Vector{Matrix{Float32}}  # per-expert input batch buffers
end

function MoELayer(hidden_dim::Int, ffn_dim::Int, n_experts::Int, top_k::Int; routing_mode::RoutingMode=TopKMode, relu_lambda_l1::Float32=0.01f0)
    experts = [SwiGLU(hidden_dim, ffn_dim) for _ in 1:n_experts]
    et = [Int[] for _ in 1:n_experts]
    ewi = [Int[] for _ in 1:n_experts]
    ebb = [Matrix{Float32}(undef, 0, 0) for _ in 1:n_experts]
    beg = [Matrix{Float32}(undef, 0, 0) for _ in 1:n_experts]
    bei = [Matrix{Float32}(undef, 0, 0) for _ in 1:n_experts]
    MoELayer(Router(hidden_dim, n_experts, top_k; routing_mode=routing_mode, relu_lambda_l1=relu_lambda_l1),
             experts, hidden_dim, n_experts, top_k,
             Matrix{Float32}(undef, 0, 0), ebb, et, ewi,
             Matrix{Float32}(undef, 0, 0), 0, (),  # last_flat_data, last_num_tokens, last_leading
             Matrix{Float32}(undef, 0, hidden_dim),   # grad_input_buf
             beg,                                      # bwd_expert_grad_bufs
             bei)                                      # bwd_expert_input_bufs
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

    _, indices = forward(m.router, input_tensor)
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

        # Per-expert batch buffer: keep exact shape to avoid slice-copy fallback.
        eb = m.expert_batch_bufs[e_idx]
        if size(eb, 1) != n_tok || size(eb, 2) != hidden_dim
            m.expert_batch_bufs[e_idx] = Matrix{Float32}(undef, n_tok, hidden_dim)
        end
        bb = m.expert_batch_bufs[e_idx]

        # Gather: copy selected token vectors into contiguous expert batch
        @inbounds for (i, t) in enumerate(tokens)
            @simd for d in 1:hidden_dim
                bb[i, d] = fid[t, d]
            end
        end

        batch_input = Tensor(bb, F32)

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
        @simd for d in 1:hidden_dim
            od[t, d] += w * eod_flat[i, d]
        end
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

        # Per-expert grad buffer: exact shape avoids row-slice copies.
        if size(m.bwd_expert_grad_bufs[e_idx], 1) != n_tok || size(m.bwd_expert_grad_bufs[e_idx], 2) != hidden_dim
            m.bwd_expert_grad_bufs[e_idx] = Matrix{Float32}(undef, n_tok, hidden_dim)
        end
        expert_grad_data = m.bwd_expert_grad_bufs[e_idx]

        @inbounds for (i, t) in enumerate(tokens)
            k = weight_idx[i]
            w = wd[t, k]
            @simd for d in 1:hidden_dim
                expert_grad_data[i, d] = w * flat_grad[t, d]
            end
        end
        expert_grad = Tensor(expert_grad_data, grad_output.dtype)

        # Per-expert input buffer: exact shape avoids row-slice copies.
        if size(m.bwd_expert_input_bufs[e_idx], 1) != n_tok || size(m.bwd_expert_input_bufs[e_idx], 2) != hidden_dim
            m.bwd_expert_input_bufs[e_idx] = Matrix{Float32}(undef, n_tok, hidden_dim)
        end
        expert_input_data = m.bwd_expert_input_bufs[e_idx]

        @inbounds for (i, t) in enumerate(tokens)
            @simd for d in 1:hidden_dim
                expert_input_data[i, d] = flat_x[t, d]
            end
        end
        expert_input = Tensor(expert_input_data, grad_output.dtype)

        # Set cached input for expert sub-layers (gate, up need the original input)
        m.experts[e_idx].w_gate.last_input = expert_input
        m.experts[e_idx].w_up.last_input = expert_input

        # Backward through expert (accumulates weight gradients in SwiGLU sub-layers)
        grad_expert_input = backward(m.experts[e_idx], expert_grad)
        ge_data = grad_expert_input.data

        # Scatter-add: accumulate expert input gradient back to token positions
        ge_flat = reshape(ge_data, n_tok, hidden_dim)
        @inbounds for (i, t) in enumerate(tokens)
            @simd for d in 1:hidden_dim
                grad_input[t, d] += ge_flat[i, d]
            end
        end
    end

    Tensor(reshape(grad_input, m.last_leading..., hidden_dim), grad_output.dtype)
end

function parameters(m::MoELayer)
    Iterators.flatten((parameters(m.router), Iterators.flatten(parameters.(m.experts))))
end

aux_loss(m::MoELayer, alpha::Float32) = compute_aux_loss(m.router, alpha)

# Propagate routing mode updates
set_routing_mode!(m::MoELayer, mode::RoutingMode) = set_routing_mode!(m.router, mode)
update_expert_bias!(m::MoELayer, gamma::Float32) = update_expert_bias!(m.router, gamma)
compute_relu_l1_loss_with_grad!(m::MoELayer) = compute_relu_l1_loss_with_grad!(m.router)

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

# Propagate routing mode updates
set_routing_mode!(blk::TransformerBlock, mode::RoutingMode) = set_routing_mode!(blk.moe, mode)
update_expert_bias!(blk::TransformerBlock, gamma::Float32) = update_expert_bias!(blk.moe, gamma)
compute_relu_l1_loss_with_grad!(blk::TransformerBlock) = compute_relu_l1_loss_with_grad!(blk.moe)
