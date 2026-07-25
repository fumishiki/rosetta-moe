# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# MoETransformer.jl — Top-level module definition
#
# Entry point for the Mixture-of-Experts Transformer implementation.
# All sub-files are included in dependency order; the module re-exports
# the full public API so callers only need `using .MoETransformer`.

module MoETransformer

using LinearAlgebra
# LBT (libblastrampoline) backend selection:
# On macOS, AppleAccelerate.jl replaces the default OpenBLAS backend via LBT,
# routing all LinearAlgebra.mul! calls to Apple's Accelerate framework (AMX).
# This is a load-time side-effect — no code changes needed downstream.
@static if Sys.isapple()
    try
        using AppleAccelerate
    catch e
        @warn "AppleAccelerate not available, falling back to OpenBLAS" exception=(e, catch_backtrace())
    end
end

# Include order matters: each file may depend on types from prior files.
include("tensor.jl")
include("config.jl")
include("layers.jl")
include("attention.jl")
include("moe.jl")
include("model.jl")
include("generate.jl")
include("train.jl")

# Conditionally include Metal GPU backend
@static if Sys.isapple()
    try
        include("metal.jl")
    catch e
        @warn "Metal GPU backend not available" exception=(e, catch_backtrace())
    end
end

export DType, F32, F16, BF16, I32, I64
export Tensor, zeros_tensor, ones_tensor, randn_tensor, randn_std, from_array
export numel, clone, scale, silu, add_in_place!, scale_in_place!
export tensor_sum, tensor_mean
export seed_rng!
export tensor_matmul, matmul_transposed_b, transpose_tensor, reshape_tensor
export softmax, softmax_in_place!, argmax_f32, normalize_in_place!
export AbstractLayer, forward, backward, parameters
export Config, tiny, small, default_6_9b, total_params, active_params
export RoutingMode, TopKMode, BiasFreeMode, ReLUMode
export Embedding, Linear, RMSNorm, SwiGLU
export MQAttention, Router, MoELayer, TransformerBlock
export MoETransformer, tiny_model, small_model, medium_model, default_model
export forward_ids, total_aux_loss, compute_aux_loss
export apply_aux_loss!, apply_z_loss!, compute_aux_loss_with_grad!, compute_z_loss_with_grad!
export set_routing_mode!, update_routing_biases!, apply_relu_l1_loss!, avg_active_experts
export SamplingStrategy, GreedySampling, TemperatureSampling, TopKSampling, TopPSampling
export generate, pick_token
export generate_greedy, generate_sample, generate_topk, generate_topp
export TrainConfig, default_train_config, AdamWState, Trainer
export get_lr, train_step!, train_step_from_logits!, cross_entropy_loss, cross_entropy_grad, cross_entropy_grad_into!, clip_grad_by_global_norm!
export CheckpointStorage, CheckpointContext, disabled_checkpoint_context
export save!, should_checkpoint, maybe_save!, get_checkpoint, clear!
export AbstractLossScaler, StaticLossScaler, DynamicLossScaler
export static_loss_scaler, dynamic_loss_scaler
export scale_loss, unscale_grads, check_overflow!, update!, should_skip_step
export MixedPrecisionConfig, default_mixed_precision_config, fp16_mixed_precision_config, is_fp32_layer
export MasterWeights

# Metal GPU backend exports (if available)
@static if Sys.isapple()
    if @isdefined(metal_available)
        export metal_available
        export gpu_forward, gpu_train_step!
        export gpu_forward_moe, gpu_forward_block
        export gpu_kernel_matmul, gpu_kernel_softmax, gpu_kernel_rmsnorm
    end
end

end # module
