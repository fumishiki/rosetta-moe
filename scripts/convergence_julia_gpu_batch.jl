#!/usr/bin/env julia
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# Batch GPU convergence verification for Julia MoE Transformer.
#
# Runs N=30 trials in a SINGLE Julia process to amortize JIT compilation.
# A warmup trial (seed=0) ensures all code paths are compiled before measurement.
# Output: single JSON with all trial loss curves to stdout.
# Progress: printed to stderr.

include(joinpath(@__DIR__, "..", "julia", "MoETransformer.jl"))
using .MoETransformer
using Random
using Printf

function run_trial(seed::Int, input::Tensor, targets::Tensor, n_steps::Int)
    Random.seed!(seed)
    seed_rng!(seed)
    model = tiny_model()
    cfg = TrainConfig(1f-3, 0.9f0, 0.95f0, 1f-8, 0.1f0, 0.5f0, 50, 600, 0.01f0, 0.05f0, TopKMode, 0.001f0, 0.01f0, 2)
    trainer = Trainer(model, cfg)

    losses = Vector{Float32}(undef, n_steps)
    for i in 1:n_steps
        losses[i] = gpu_train_step!(trainer, input, targets)
    end
    losses
end

function main()
    # Check Metal availability
    if !@isdefined(metal_available) || !metal_available()
        println(stderr, "Metal GPU backend not available")
        exit(1)
    end

    n_trials = 30
    n_steps = 500

    # Parse optional --trials and --steps from CLI
    for i in 1:length(ARGS)
        if ARGS[i] == "--trials" && i < length(ARGS)
            n_trials = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--steps" && i < length(ARGS)
            n_steps = parse(Int, ARGS[i+1])
        end
    end

    # Prepare input/target data (same as single-seed version)
    batch, seq = 2, 8
    input_data = Float32[Float32(mod(i, 1000)) for i in 0:batch*seq-1]
    target_data = Float32[Float32(mod(i + 1, 1000)) for i in 0:batch*seq-1]
    input = from_array(reshape(input_data, batch, seq))
    targets = from_array(reshape(target_data, batch, seq))

    # Warmup run (seed=0) — compile all code paths before measurement
    println(stderr, "[julia/gpu] warmup (seed=0)...")
    run_trial(0, input, targets, n_steps)
    GC.gc(true)  # release warmup GPU buffers before measurement
    println(stderr, "[julia/gpu] warmup complete")

    # Measurement runs
    all_trials = Vector{Vector{Float32}}(undef, n_trials)
    for trial in 1:n_trials
        println(stderr, "[julia/gpu] trial $trial/$n_trials...")
        all_trials[trial] = run_trial(trial, input, targets, n_steps)
        GC.gc(true)  # force full GC to release Metal GPU buffers
    end

    # Output JSON
    print("{\"language\":\"julia_gpu\",\"n_trials\":$n_trials,\"trials\":[")
    for (t, losses) in enumerate(all_trials)
        t > 1 && print(",")
        loss_strs = join([@sprintf("%.6f", l) for l in losses], ",")
        print("[$loss_strs]")
    end
    println("]}")
end

main()
