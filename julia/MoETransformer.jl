# SPDX-License-Identifier: CC-BY-NC-4.0
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

export DType, F32, F16, BF16, I32, I64
export Tensor, zeros_tensor, ones_tensor, randn_tensor, randn_std, from_array
export numel, clone, scale, silu, add_in_place!, scale_in_place!
export tensor_sum, tensor_mean
export tensor_matmul, matmul_transposed_b, transpose_tensor, reshape_tensor
export softmax, softmax_in_place!, argmax_f32, normalize_in_place!
export AbstractLayer, forward, backward, parameters
export Config, tiny, small, default_6_9b, total_params, active_params
export Embedding, Linear, RMSNorm, SwiGLU
export MQAttention, Router, MoELayer, TransformerBlock
export MoETransformer, tiny_model, small_model, default_model
export forward_ids, total_aux_loss, compute_aux_loss
export SamplingStrategy, GreedySampling, TemperatureSampling, TopKSampling, TopPSampling
export generate, pick_token
export generate_greedy, generate_sample, generate_topk, generate_topp
export TrainConfig, default_train_config, AdamWState, Trainer
export get_lr, train_step!, cross_entropy_loss, cross_entropy_grad, cross_entropy_grad_into!, clip_grad_by_global_norm!
export CheckpointStorage, CheckpointContext, disabled_checkpoint_context
export save!, should_checkpoint, maybe_save!, get_checkpoint, clear!
export AbstractLossScaler, StaticLossScaler, DynamicLossScaler
export static_loss_scaler, dynamic_loss_scaler
export scale_loss, unscale_grads, check_overflow!, update!, should_skip_step
export MixedPrecisionConfig, default_mixed_precision_config, fp16_mixed_precision_config, is_fp32_layer
export MasterWeights

end # module
