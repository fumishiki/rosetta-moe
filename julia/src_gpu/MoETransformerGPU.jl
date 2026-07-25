# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# MoETransformerGPU.jl — Metal GPU module definition
#
# Entry point for the GPU Mixture-of-Experts Transformer implementation.
# All layers use MetalTensor (MtlArray{Float32}).
# Forward AND backward run on GPU.
# NO Array() conversions — all compute stays on GPU.

module MoETransformerGPU

using Metal
using LinearAlgebra

# Include shared config from parent directory
include(joinpath(@__DIR__, "..", "config.jl"))

# Shared LCG for reproducible init (same as CPU)
const LCG_MULT = UInt64(6364136223846793005)
const LCG_U64MAX = Float64(typemax(UInt64))
const _lcg_state = Ref{UInt64}(UInt64(42))
const _inference_mode = Ref(false)

function seed_rng!(seed::Integer)
    _lcg_state[] = UInt64(seed)
    return nothing
end

function set_inference_mode!(enabled::Bool)
    _inference_mode[] = enabled
    return nothing
end

inference_mode() = _inference_mode[]

function _lcg_uniform()::Float64
    _lcg_state[] = _lcg_state[] * LCG_MULT + UInt64(1)
    u = Float64(_lcg_state[]) / LCG_U64MAX
    return max(u, 1e-10)
end

# Include order matters
include("metal_tensor.jl")
include("metal_layers.jl")
include("metal_attention.jl")
include("metal_moe.jl")
include("metal_model.jl")
include("metal_train.jl")

# Check if Metal.jl is functional
function metal_available()
    try
        Metal.functional()
    catch
        false
    end
end

export DType, F32, F16, BF16, I32, I64
export Config, tiny, small, medium, default_6_9b, total_params, active_params
export RoutingMode, TopKMode, BiasFreeMode, ReLUMode
export seed_rng!
export set_inference_mode!, inference_mode
export metal_available

# Metal tensor types
export MetalTensor, zeros_mtensor, ones_mtensor, randn_mtensor, randn_std_mtensor
export numel_mtl, scale_mtl, silu_in_place_mtl!, add_in_place_mtl!, mul_in_place_mtl!
export tensor_sum_mtl, tensor_matmul_mtl, matmul_transposed_b_mtl
export softmax_mtl, reshape_mtensor

# Metal layer types
export AbstractMetalLayer
export MetalEmbedding, MetalLinear, MetalRMSNorm, MetalSwiGLU
export MetalMQAttention, MetalRouter, MetalMoELayer, MetalTransformerBlock
export MetalMoETransformer, tiny_metal_model, small_metal_model, medium_metal_model

# GPU operations
export gpu_forward, gpu_backward, gpu_parameters
export total_aux_loss_mtl, apply_z_loss_mtl!, apply_aux_loss_mtl!

# Training
export MetalTrainConfig, default_metal_train_config
export MetalAdamWState, MetalTrainer
export get_lr_mtl, gpu_train_step!, gpu_train_step_no_readback!, gpu_forward_ce_no_readback!
export cross_entropy_loss_mtl, cross_entropy_loss_sum_tensor_mtl, cross_entropy_loss_grad_mtl!, clip_grad_mtl!

end # module
