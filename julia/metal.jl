# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# metal.jl — Metal GPU backend for MoE Transformer
#
# Julia's Metal.jl provides:
# 1. MtlArray — GPU buffers on M1 unified memory
# 2. mul! on MtlArray → MPS matmul (automatic dispatch)
# 3. @metal macro — compiles Julia functions to MSL shaders
#
# Key design:
# - Forward pass runs on GPU (MtlArrays)
# - Backward pass runs on CPU (existing implementation)
# - Metal.@sync ensures GPU ops complete before timing

using Metal
using LinearAlgebra

# Check if Metal.jl is functional on this system
function metal_available()
    try
        Metal.functional()
    catch
        false
    end
end

# GPU softmax kernel (Julia -> MSL via @metal)
# Multi-pass reduction: max, then exp+sum, then normalize
# Note: Simplified reduction (serial) — production would use parallel reduction
function gpu_softmax_kernel!(output, input, max_val, sum_val, n, pass)
    tid = thread_position_in_threadgroup_1d()
    gid = threadgroup_position_in_grid_1d()
    tg_size = threads_per_threadgroup_1d()

    row_offset = (gid - 1) * n

    if pass == 1
        # Find max (simplified: thread 1 scans entire row)
        if tid == 1
            local_max = Float32(-Inf)
            for i in 1:n
                @inbounds local_max = max(local_max, input[row_offset + i])
            end
            @inbounds max_val[gid] = local_max
        end
    elseif pass == 2
        # Exp and sum (simplified: thread 1 scans entire row)
        if tid == 1
            row_max = @inbounds max_val[gid]
            local_sum = 0f0
            for i in 1:n
                @inbounds val = exp(input[row_offset + i] - row_max)
                @inbounds output[row_offset + i] = val
                local_sum += val
            end
            @inbounds sum_val[gid] = local_sum
        end
    elseif pass == 3
        # Normalize (parallel across threads)
        total = @inbounds sum_val[gid]
        i = tid
        while i <= n
            @inbounds output[row_offset + i] /= total
            i += tg_size
        end
    end

    return nothing
end

# GPU RMSNorm kernel
# Note: Simplified reduction (serial) — production would use parallel reduction
function gpu_rmsnorm_kernel!(output, input, weight, rms_val, hidden_dim, eps, pass)
    tid = thread_position_in_threadgroup_1d()
    gid = threadgroup_position_in_grid_1d()
    tg_size = threads_per_threadgroup_1d()

    row_offset = (gid - 1) * hidden_dim

    if pass == 1
        # Sum of squares (simplified: thread 1 scans entire row)
        if tid == 1
            local_sum_sq = 0f0
            for i in 1:hidden_dim
                @inbounds val = input[row_offset + i]
                local_sum_sq += val * val
            end
            @inbounds rms_val[gid] = 1f0 / sqrt(local_sum_sq / Float32(hidden_dim) + eps)
        end
    elseif pass == 2
        # Scale by weight (parallel across threads)
        rms = @inbounds rms_val[gid]
        i = tid
        while i <= hidden_dim
            @inbounds output[row_offset + i] = input[row_offset + i] * rms * weight[i]
            i += tg_size
        end
    end

    return nothing
end

# GPU SiLU kernel
function gpu_silu_kernel!(output, input)
    idx = thread_position_in_grid_1d()
    @inbounds x = input[idx]
    @inbounds output[idx] = x / (1f0 + exp(-x))
    return nothing
end

# GPU softmax (host function)
function gpu_softmax!(output::MtlArray{Float32}, input::MtlArray{Float32}, n_rows::Int, n_cols::Int)
    max_val = MtlArray{Float32}(undef, n_rows)
    sum_val = MtlArray{Float32}(undef, n_rows)

    # Pass 1: find max
    @metal threads=256 groups=n_rows gpu_softmax_kernel!(output, input, max_val, sum_val, n_cols, 1)

    # Pass 2: exp and sum
    @metal threads=256 groups=n_rows gpu_softmax_kernel!(output, input, max_val, sum_val, n_cols, 2)

    # Pass 3: normalize
    @metal threads=256 groups=n_rows gpu_softmax_kernel!(output, input, max_val, sum_val, n_cols, 3)

    return output
end

# GPU RMSNorm (host function)
function gpu_rmsnorm!(output::MtlArray{Float32}, input::MtlArray{Float32}, weight::MtlArray{Float32}, n_rows::Int, hidden_dim::Int, eps::Float32)
    rms_val = MtlArray{Float32}(undef, n_rows)

    # Pass 1: compute RMS
    @metal threads=256 groups=n_rows gpu_rmsnorm_kernel!(output, input, weight, rms_val, hidden_dim, eps, 1)

    # Pass 2: scale
    @metal threads=256 groups=n_rows gpu_rmsnorm_kernel!(output, input, weight, rms_val, hidden_dim, eps, 2)

    return output
end

# GPU SiLU (host function)
function gpu_silu!(output::MtlArray{Float32}, input::MtlArray{Float32})
    n = length(input)
    threads = 256
    groups = cld(n, threads)
    @metal threads=threads groups=groups gpu_silu_kernel!(output, input)
    return output
end

# GPU matmul wrapper (MPS backend via Metal.jl)
function gpu_matmul!(C::MtlMatrix{Float32}, A::MtlMatrix{Float32}, B::MtlMatrix{Float32})
    mul!(C, A, B)
end

# Convert Tensor to MtlArray
function to_mtl(t::Tensor)
    MtlArray{Float32}(t.data)
end

# Convert MtlArray back to Tensor
function from_mtl(mtl::MtlArray{Float32}, dtype::DType=F32)
    Tensor(Array(mtl), dtype)
end

# GPU forward pass for Linear layer
function gpu_forward_linear(layer::Linear, x_mtl::MtlArray{Float32})
    # x: (batch*seq, in_features)
    # weight: (out_features, in_features) — stored transposed
    # Need to compute: x @ W^T where W is (out, in)
    # Result: (batch*seq, out_features)

    weight_mtl = to_mtl(layer.weight)  # (out_features, in_features)
    # x_mtl is (batch*seq, in_features), weight_mtl is (out_features, in_features)
    # We need: x @ W^T = (batch*seq, in) @ (in, out) = (batch*seq, out)
    # Metal's * operator: A * B = (m,k) * (k,n) = (m,n)
    # So we need: x_mtl * transpose(weight_mtl) = (batch*seq, in) * (in, out)
    out_mtl = x_mtl * transpose(weight_mtl)

    if layer.bias !== nothing
        bias_mtl = to_mtl(layer.bias)
        # Broadcast add
        out_mtl .+= reshape(bias_mtl, 1, :)
    end

    return out_mtl
end

# GPU forward pass for RMSNorm layer
function gpu_forward_rmsnorm(layer::RMSNorm, x_mtl::MtlArray{Float32}, batch_seq::Int, hidden_dim::Int)
    weight_mtl = to_mtl(layer.weight)
    out_mtl = MtlArray{Float32}(undef, size(x_mtl))

    gpu_rmsnorm!(out_mtl, x_mtl, weight_mtl, batch_seq, hidden_dim, layer.eps)

    return out_mtl
end

# GPU forward pass for SwiGLU
function gpu_forward_swiglu(layer::SwiGLU, x_mtl::MtlArray{Float32})
    # gate = w_gate(x)
    gate_mtl = gpu_forward_linear(layer.w_gate, x_mtl)

    # up = w_up(x)
    up_mtl = gpu_forward_linear(layer.w_up, x_mtl)

    # silu(gate)
    gate_silu_mtl = MtlArray{Float32}(undef, size(gate_mtl))
    gpu_silu!(gate_silu_mtl, gate_mtl)

    # gate_silu .* up
    fused_mtl = gate_silu_mtl .* up_mtl

    # w_down(fused)
    gpu_forward_linear(layer.w_down, fused_mtl)
end

# GPU forward pass for MQAttention (simplified — no custom kernels for attention yet)
function gpu_forward_mqattention(layer::MQAttention, x_mtl::MtlArray{Float32}, batch::Int, seq_len::Int)
    # For now, fall back to CPU for attention (complex kernel)
    x_cpu = from_mtl(x_mtl, F32)
    out_cpu = forward(layer, x_cpu)
    to_mtl(out_cpu)
end

# GPU forward pass for Router (simplified)
function gpu_forward_router(layer::Router, x_mtl::MtlArray{Float32})
    # w(x) -> softmax
    logits_mtl = gpu_forward_linear(layer.w, x_mtl)

    batch_seq = size(logits_mtl, 1)
    n_experts = size(logits_mtl, 2)

    # Softmax
    probs_mtl = MtlArray{Float32}(undef, size(logits_mtl))
    gpu_softmax!(probs_mtl, logits_mtl, batch_seq, n_experts)

    # For simplicity, return to CPU for routing logic
    from_mtl(probs_mtl, F32)
end

# GPU forward pass for MoE layer (hybrid: GPU SwiGLU, CPU routing)
function gpu_forward_moe(moe::MoELayer, x_mtl::MtlArray{Float32}, batch::Int, seq_len::Int)
    hidden = size(x_mtl, 2)
    batch_seq = batch * seq_len

    # Router: GPU linear + softmax → CPU top-k routing
    x_cpu = from_mtl(x_mtl, F32)
    _, indices = forward(moe.router, x_cpu)
    wd = moe.router.weights_buf

    # Build inverted index: expert → tokens
    n_experts = length(moe.experts)
    expert_tokens = [Int[] for _ in 1:n_experts]
    expert_weight_idx = [Int[] for _ in 1:n_experts]
    for t in 1:batch_seq
        for (k, eidx) in enumerate(indices[t])
            push!(expert_tokens[eidx + 1], t)  # 0-based -> 1-based
            push!(expert_weight_idx[eidx + 1], k)
        end
    end

    # Per-expert GPU SwiGLU
    x_data = Array(x_mtl)  # read back for gathering
    out_data = zeros(Float32, batch_seq, hidden)

    for e_idx in 1:n_experts
        tokens = expert_tokens[e_idx]
        isempty(tokens) && continue
        n_tok = length(tokens)

        # Gather token vectors
        batch_data = zeros(Float32, n_tok, hidden)
        for (i, t) in enumerate(tokens)
            @views batch_data[i, :] .= x_data[t, :]
        end
        batch_mtl = MtlArray(batch_data)

        # GPU SwiGLU for this expert
        expert_out_mtl = gpu_forward_swiglu(moe.experts[e_idx], batch_mtl)
        expert_out = Array(expert_out_mtl)

        # Weighted scatter-add
        for (i, t) in enumerate(tokens)
            k = expert_weight_idx[e_idx][i]
            alpha = wd[t, k]  # weights is (batch_seq, top_k)
            @views out_data[t, :] .+= alpha .* expert_out[i, :]
        end
    end

    MtlArray(out_data)
end

# GPU forward pass for a single TransformerBlock
function gpu_forward_block(block::TransformerBlock, x_mtl::MtlArray{Float32}, batch::Int, seq_len::Int)
    hidden = size(x_mtl, 2)
    batch_seq = batch * seq_len

    # RMSNorm (attn)
    normed = gpu_forward_rmsnorm(block.attn_norm, x_mtl, batch_seq, hidden)

    # Attention (CPU fallback — convert to 3D tensor)
    normed_cpu = from_mtl(normed, F32)
    # Reshape to (batch, seq, hidden) for attention
    normed_3d = Tensor(reshape(normed_cpu.data, batch, seq_len, hidden), F32)
    attn_out = forward(block.attention, normed_3d)
    # Flatten back to 2D and convert to GPU
    attn_out_2d = reshape(attn_out.data, batch_seq, hidden)
    attn_out_mtl = MtlArray(attn_out_2d)

    # Residual add
    x_mtl = x_mtl .+ attn_out_mtl

    # RMSNorm (ffn)
    normed2 = gpu_forward_rmsnorm(block.ffn_norm, x_mtl, batch_seq, hidden)

    # MoE (hybrid)
    moe_out = gpu_forward_moe(block.moe, normed2, batch, seq_len)

    # Residual add
    x_mtl .+ moe_out
end

# GPU forward pass for full model (hybrid: matmuls on GPU, complex logic on CPU)
function gpu_forward(model::MoETransformer, input::Tensor)
    dims = size(input.data)
    batch = dims[1]
    seq_len = dims[2]
    hidden = model.config.hidden_dim
    batch_seq = batch * seq_len

    # 1. Embedding (CPU → GPU)
    x_cpu = forward(model.embedding, input)  # CPU embedding lookup
    # Reshape to (batch_seq, hidden) for 2D operations
    x_data_2d = reshape(x_cpu.data, batch_seq, hidden)
    x_mtl = MtlArray(x_data_2d)

    # 2. Transformer blocks
    for block in model.blocks
        x_mtl = gpu_forward_block(block, x_mtl, batch, seq_len)
    end

    # 3. Final RMSNorm
    x_mtl = gpu_forward_rmsnorm(model.final_norm, x_mtl, batch_seq, hidden)

    # 4. LM Head
    logits_mtl = gpu_forward_linear(model.lm_head, x_mtl)

    # 5. Read back to CPU
    vocab = model.config.vocab_size
    logits_data = reshape(Array(logits_mtl), batch, seq_len, vocab)
    Tensor(logits_data, F32)
end

# GPU train step: forward on GPU, backward + optimizer on CPU.
# M1 unified memory makes CPU↔GPU readback effectively free.
#
# Strategy: Run CPU forward first to populate all layer caches (needed for backward),
# then run GPU forward to get accelerated logits, then do CPU backward + optimizer.
function gpu_train_step!(trainer::Trainer, input::Tensor, targets::Tensor)
    # CPU forward to populate caches (needed for backward)
    _ = forward(trainer.model, input)

    # GPU forward for accelerated logits computation
    logits = gpu_forward(trainer.model, input)

    # CPU loss + backward + optimizer (reuse existing train infrastructure)
    train_step_from_logits!(trainer, logits, targets)
end

# Benchmark kernel: GPU matmul only
function gpu_kernel_matmul(m::Int, n::Int, k::Int)
    a_mtl = MtlArray(randn(Float32, m, k))
    b_mtl = MtlArray(randn(Float32, k, n))
    c_mtl = MtlArray{Float32}(undef, m, n)

    Metal.@sync gpu_matmul!(c_mtl, a_mtl, b_mtl)

    return from_mtl(c_mtl, F32)
end

# Benchmark kernel: GPU softmax only
function gpu_kernel_softmax(n_rows::Int, n_cols::Int)
    input_mtl = MtlArray(randn(Float32, n_rows, n_cols))
    output_mtl = MtlArray{Float32}(undef, n_rows, n_cols)

    Metal.@sync gpu_softmax!(output_mtl, input_mtl, n_rows, n_cols)

    return from_mtl(output_mtl, F32)
end

# Benchmark kernel: GPU RMSNorm only
function gpu_kernel_rmsnorm(n_rows::Int, hidden_dim::Int)
    input_mtl = MtlArray(randn(Float32, n_rows, hidden_dim))
    weight_mtl = MtlArray(randn(Float32, hidden_dim))
    output_mtl = MtlArray{Float32}(undef, n_rows, hidden_dim)

    Metal.@sync gpu_rmsnorm!(output_mtl, input_mtl, weight_mtl, n_rows, hidden_dim, 1f-5)

    return from_mtl(output_mtl, F32)
end
