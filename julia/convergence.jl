# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# Loss convergence verification.
# Trains the tiny MoE Transformer for 500 steps and outputs loss per step.
#
# Usage: julia --project=. convergence.jl
#
# The portable Knuth LCG ensures that weight initialization produces identical
# random values across Rust/Go/Python/Julia. This allows direct comparison of
# optimization and numerical stability across implementations.

include("MoETransformer.jl")
using .MoETransformer

seed_rng!(42)
model = tiny_model()
train_cfg = TrainConfig(
    1e-3,       # lr
    0.9,        # beta1
    0.95,       # beta2
    1e-8,       # eps
    0.0,        # weight_decay (Rust doesn't have this, set to 0)
    0.5,        # grad_clip
    50,         # warmup_steps
    600,        # total_steps
    0.01,       # aux_loss_weight
    0.05,       # z_loss_weight (Phase 0: 0.01 -> 0.05)
    TopKMode,   # routing_mode
    0.001,      # bias_gamma
    0.01,       # relu_lambda_l1
    2           # relu_target_k
)
trainer = Trainer(model, train_cfg)

batch = 2
seq = 8
vocab = 1000

# Fixed deterministic input (same across all 4 languages)
input_data = Float32[(i % vocab) for i in 0:batch*seq-1]
input = from_array(reshape(input_data, batch, seq))

target_data = Float32[((i + 1) % vocab) for i in 0:batch*seq-1]
targets = from_array(reshape(target_data, batch, seq))

n_steps = 500
losses = Float32[]
sizehint!(losses, n_steps)

for _ in 1:n_steps
    loss = train_step!(trainer, input, targets)
    push!(losses, loss)
end

# Output JSON
losses_str = join([string(round(l; digits=6)) for l in losses], ",")
println("{\"language\":\"julia\",\"steps\":$n_steps,\"losses\":[$losses_str]}")
