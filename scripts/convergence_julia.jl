#!/usr/bin/env julia
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# Loss convergence verification for Julia MoE Transformer.

include(joinpath(@__DIR__, "..", "julia", "MoETransformer.jl"))
using .MoETransformer
using Random
using Printf

function main()
    seed_val = 42
    for i in 1:length(ARGS)
        if ARGS[i] == "--seed" && i < length(ARGS)
            seed_val = parse(Int, ARGS[i+1])
        end
    end
    Random.seed!(seed_val)
    seed_rng!(seed_val)
    model = tiny_model()
    cfg = TrainConfig(1f-3, 0.9f0, 0.95f0, 1f-8, 0.1f0, 0.5f0, 50, 600, 0.01f0, 0.05f0, TopKMode, 0.001f0, 0.01f0, 2)
    trainer = Trainer(model, cfg)

    batch, seq = 2, 8
    input_data = Float32[Float32(mod(i, 1000)) for i in 0:batch*seq-1]
    target_data = Float32[Float32(mod(i + 1, 1000)) for i in 0:batch*seq-1]
    input = from_array(reshape(input_data, batch, seq))
    targets = from_array(reshape(target_data, batch, seq))

    n_steps = 500
    losses = Float32[]
    for _ in 1:n_steps
        push!(losses, train_step!(trainer, input, targets))
    end

    # Output JSON
    loss_strs = join([@sprintf("%.6f", l) for l in losses], ",")
    println("{\"language\":\"julia\",\"steps\":$n_steps,\"losses\":[$loss_strs]}")
end

main()
