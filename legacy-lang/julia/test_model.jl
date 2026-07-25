using Pkg
Pkg.activate("lang/julia")

using MLCore
using Random, Printf

println("=== Testing 1.9B BLT Recurrent MoE ===\n")

rng = Random.default_rng()
config = C1_9B
model = BLTRecurrentMoE(config)

println("Model configuration:")
println("  vocab: $(config.vocab), hidden: $(config.hidden), ffn: $(config.ffn)")
println("  heads: $(config.heads), kv_heads: $(config.kv_heads), head_dim: $(config.head_dim)")
println("  experts: $(config.experts), topk: $(config.topk), loops: $(config.loops)")
println("  low_rank_r: $(config.r)\n")

ps, st = Lux.setup(rng, model)
println("Parameters initialized\n")

batch, seq = 2, 4
ids = rand(0:255, batch, seq)
println("Input shape: (batch=$batch, seq=$seq)")
println("Sample IDs: $(ids[1, :])\n")

println("Running forward pass...")
@time y, st_new = model(ids, ps, st)
println("Output shape: $(size(y))")
println("Output sample: $(y[1:5, 1, 1])\n")

println("Testing individual layers:")

println("\n1. BLT Input Layer")
input_layer = BLTInput(config)
ps_input, st_input = Lux.setup(rng, input_layer)
@time h, _ = input_layer(ids, ps_input, st_input)
println("   Hidden shape: $(size(h))")

println("\n2. MQA Attention")
attn = MQAAttention(config)
ps_attn, st_attn = Lux.setup(rng, attn)
@time h_attn, _ = attn(h, ps_attn, st_attn)
println("   Output shape: $(size(h_attn))")

println("\n3. Router")
router = Router(config)
ps_router, st_router = Lux.setup(rng, router)
@time (weights, indices), _ = router(h, ps_router, st_router)
println("   Weights shape: $(size(weights)), Indices shape: $(size(indices))")
println("   Top experts for token 1: $(indices[:, 1])")
println("   Weights for token 1: $(weights[:, 1])")

println("\n4. Shared Expert")
shared = SharedExpert(config)
ps_shared, st_shared = Lux.setup(rng, shared)
@time h_expert, _ = shared(h, ps_shared, st_shared)
println("   Output shape: $(size(h_expert))")

println("\n5. Low-Rank Delta Expert")
delta = LowRankDelta(config)
ps_delta, st_delta = Lux.setup(rng, delta)
test_x = h[:, 1:1, 1:1]
@time h_delta, _ = delta(test_x, 1, ps_delta, st_delta)
println("   Output shape: $(size(h_delta))")

println("\n6. Complete MoE Layer")
moe = MoE(config)
ps_moe, st_moe = Lux.setup(rng, moe)
@time h_moe, _ = moe(h, ps_moe, st_moe)
println("   Output shape: $(size(h_moe))")

println("\n=== All tests passed! ===")
