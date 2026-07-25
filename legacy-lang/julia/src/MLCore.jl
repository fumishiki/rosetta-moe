module MLCore

using CUDA, LinearAlgebra, Statistics, Printf, Random
using Lux, Zygote, Optimisers, NNlib, ChainRulesCore, SpecialFunctions
using YAML

# --- Config ---
abstract type ModelSize end
struct M1_9B <: ModelSize end
struct M8B <: ModelSize end

# Dispatch-based config generation
config(::M1_9B) = (vocab=256, hidden=2048, ffn=16384, heads=32, kv_heads=1, head_dim=64,
                   experts=128, topk=8, loops=4, ctx_len=4096, rope_base=1e4, r=256)
config(::M8B) = (vocab=256, hidden=4096, ffn=32768, heads=64, kv_heads=8, head_dim=64,
                 experts=128, topk=8, loops=4, ctx_len=8192, rope_base=1e4, r=512)

@kwdef struct Config
    vocab::Int; hidden::Int; ffn::Int; heads::Int; kv_heads::Int; head_dim::Int
    experts::Int; topk::Int; loops::Int; ctx_len::Int; rope_base::Float64; r::Int
end

# Dispatch-based config initialization
Config(size::ModelSize) = Config(; config(size)...)

# YAML-based config loading
load_config(model_name::String, yaml_path::String=joinpath(@__DIR__, "..", "config.yaml")) = begin
    cfg = YAML.load_file(yaml_path)
    model_cfg = cfg["models"][model_name]
    Config(;
        vocab=model_cfg["vocab"],
        hidden=model_cfg["hidden"],
        ffn=model_cfg["ffn"],
        heads=model_cfg["heads"],
        kv_heads=model_cfg["kv_heads"],
        head_dim=model_cfg["head_dim"],
        experts=model_cfg["experts"],
        topk=model_cfg["topk"],
        loops=model_cfg["loops"],
        ctx_len=model_cfg["ctx_len"],
        rope_base=Float64(model_cfg["rope_base"]),
        r=model_cfg["r"]
    )
end

# Convenience function: load with training config
load_full_config(yaml_path::String=joinpath(@__DIR__, "..", "config.yaml")) = YAML.load_file(yaml_path)

# Default constants (backward compatibility)
const C1_9B = Config(M1_9B())
const C8B = Config(M8B())

# --- Primitives ---
abstract type AbstractActivation end
struct Erf <: AbstractActivation end
struct SiLU <: AbstractActivation end
struct GELU <: AbstractActivation end

# Dispatch-based activation
activate(::Erf, x) = erf.(x)
activate(::SiLU, x) = NNlib.swish.(x)
activate(::GELU, x) = NNlib.gelu.(x)

struct Derf <: Lux.AbstractLuxLayer end
(::Derf)(x, _, st) = (erf.(x), st)

struct RoPE <: Lux.AbstractLuxLayer dim::Int; base::Float64 end
(r::RoPE)(x, _, st) = (x, st)

# --- Dynamic Patching ---
abstract type AbstractPatcher end
struct NoPatcher <: AbstractPatcher end
struct EntropyPatcher <: AbstractPatcher vocab::Int; hidden::Int; threshold::Float32 end
EntropyPatcher(vocab::Int=256, hidden::Int=128, threshold::Float32=2.5f0) = EntropyPatcher(vocab, hidden, threshold)

# Dispatch-based initialization
Lux.initialparameters(::AbstractRNG, ::NoPatcher) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::NoPatcher) = NamedTuple()

Lux.initialparameters(rng::AbstractRNG, l::EntropyPatcher) = (
    embed=Lux.glorot_uniform(rng, l.hidden, l.vocab),
    lm_head=Lux.glorot_uniform(rng, l.vocab, l.hidden),
    threshold=Float32[l.threshold]
)
Lux.initialstates(::AbstractRNG, ::EntropyPatcher) = NamedTuple()

# Entropy calculation via broadcast and pipe
softmax_entropy(logits) = logits |>
    x -> (mx = maximum(x); exp.(x .- mx)) |>
    probs -> probs ./ (sum(probs) + eps(Float32)) |>
    p -> -sum(p .* log.(p .+ eps(Float32)))

embed_bytes(byte_seq, embed_mat, vocab) = begin
    batch, seq = size(byte_seq)
    stack([embed_mat[:, clamp(Int(byte_seq[b, s]) + 1, 1, vocab)]
           for b in 1:batch, s in 1:seq], dims=3) |>
    x -> reshape(x, size(embed_mat, 1), batch, seq)
end

# Dispatch-based patching
(::NoPatcher)(byte_seq, ps, st) = ((nothing, nothing), st)

function (l::EntropyPatcher)(byte_seq, ps, st)
    batch, seq = size(byte_seq)

    embedded = embed_bytes(byte_seq, ps.embed, l.vocab)
    logits = ps.lm_head * reshape(embedded, l.hidden, :) |> x -> reshape(x, l.vocab, batch, seq)

    entropies = mapslices(softmax_entropy, logits, dims=1) |> x -> dropdims(x, dims=1)
    boundaries = @. Float32(entropies > ps.threshold[1])
    patch_ids = mapslices(x -> cumsum(x) .+ 1, boundaries, dims=2) .|> Int

    ((patch_ids, entropies), st)
end

struct EMASmoothing <: Lux.AbstractLuxLayer hidden::Int; momentum::Float32 end
EMASmoothing(hidden::Int, momentum::Float32=0.95f0) = EMASmoothing(hidden, momentum)
Lux.initialparameters(::AbstractRNG, ::EMASmoothing) = NamedTuple()
Lux.initialstates(::AbstractRNG, l::EMASmoothing) = (ema_state=nothing,)

(l::EMASmoothing)(x, boundary_probs, ps, st) = begin
    ema = isnothing(st.ema_state) ? x : st.ema_state
    p = reshape(boundary_probs, 1, size(boundary_probs)...)
    x_smooth = @. p * x + (1 - p) * ema
    (x_smooth, (ema_state=copy(x_smooth),))
end

# Dispatch-based aggregation
aggregate_patches(::NoPatcher, embedded, ::Nothing, ::Nothing, hidden) = embedded

function aggregate_patches(::EntropyPatcher, embedded, patch_ids, max_patches, hidden)
    batch, seq = size(patch_ids)
    patch_output = zeros(Float32, hidden, batch, max_patches)
    patch_counts = zeros(Int, batch, max_patches)

    @inbounds for b in 1:batch, s in 1:seq
        p_id = patch_ids[b, s]
        p_id <= max_patches && (patch_output[:, b, p_id] .+= @views embedded[:, b, s]; patch_counts[b, p_id] += 1)
    end

    @inbounds @. patch_output /= max(patch_counts, 1)
    patch_output
end

# --- Layers ---
struct BLTInput <: Lux.AbstractLuxLayer c::Config end
Lux.initialparameters(rng::AbstractRNG, l::BLTInput) = (
    embed=Lux.glorot_uniform(rng, l.c.hidden, l.c.vocab),
    proj=Lux.glorot_uniform(rng, l.c.hidden, l.c.hidden)
)
Lux.initialstates(::AbstractRNG, ::BLTInput) = NamedTuple()

function (l::BLTInput)(ids, ps, st)
    batch, seq = size(ids)
    embedded = zeros(Float32, l.c.hidden, batch, seq)

    for b in 1:batch, s in 1:seq
        idx = clamp(Int(ids[b, s]) + 1, 1, l.c.vocab)
        embedded[:, b, s] = ps.embed[:, idx]
    end

    sz = size(embedded)
    projected = ps.proj * reshape(embedded, l.c.hidden, batch * seq)
    (reshape(projected, l.c.hidden, batch, seq), st)
end

struct BLTInputWithPatching{P<:AbstractPatcher} <: Lux.AbstractLuxLayer
    c::Config
    patcher::P
    ema::EMASmoothing
end

BLTInputWithPatching(c::Config; use_patching::Bool=false, patcher_hidden::Int=128,
                     threshold::Float32=2.5f0, ema_momentum::Float32=0.95f0) =
    use_patching ?
    BLTInputWithPatching(c, EntropyPatcher(c.vocab, patcher_hidden, threshold),
                        EMASmoothing(c.hidden, ema_momentum)) :
    BLTInputWithPatching(c, NoPatcher(), EMASmoothing(c.hidden, ema_momentum))

function Lux.initialparameters(rng::AbstractRNG, l::BLTInputWithPatching)
    (
        embed=Lux.glorot_uniform(rng, l.c.hidden, l.c.vocab),
        proj=Lux.glorot_uniform(rng, l.c.hidden, l.c.hidden),
        patcher=Lux.initialparameters(rng, l.patcher),
        ema=Lux.initialparameters(rng, l.ema)
    )
end

function Lux.initialstates(rng::AbstractRNG, l::BLTInputWithPatching)
    (
        patcher=Lux.initialstates(rng, l.patcher),
        ema=Lux.initialstates(rng, l.ema)
    )
end

# Dispatch on patcher type: NoPatcher
(l::BLTInputWithPatching{NoPatcher})(ids, ps, st) = begin
    embedded = embed_bytes(ids, ps.embed, l.c.vocab)
    output = ps.proj * reshape(embedded, l.c.hidden, :) |>
             x -> reshape(x, l.c.hidden, size(ids)...)
    (output, st)
end

# Dispatch on patcher type: EntropyPatcher
(l::BLTInputWithPatching{EntropyPatcher})(ids, ps, st) = begin
    batch, seq = size(ids)
    (patch_ids, entropies), st_p = l.patcher(ids, ps.patcher, st.patcher)

    embedded = embed_bytes(ids, ps.embed, l.c.vocab)
    max_patches = maximum(patch_ids)
    patch_embeds = aggregate_patches(l.patcher, embedded, patch_ids, max_patches, l.c.hidden)

    boundary_probs = @. Float32(entropies > ps.patcher.threshold[1])
    smoothed, st_e = l.ema(embedded, boundary_probs, ps.ema, st.ema)

    output = ps.proj * reshape(patch_embeds, l.c.hidden, :) |>
             x -> reshape(x, l.c.hidden, batch, max_patches)

    (output, (patcher=st_p, ema=st_e))
end

struct MQAAttention <: Lux.AbstractLuxLayer c::Config end
Lux.initialparameters(rng::AbstractRNG, l::MQAAttention) = (
    q=Lux.glorot_uniform(rng, l.c.heads * l.c.head_dim, l.c.hidden),
    k=Lux.glorot_uniform(rng, l.c.kv_heads * l.c.head_dim, l.c.hidden),
    v=Lux.glorot_uniform(rng, l.c.kv_heads * l.c.head_dim, l.c.hidden),
    o=Lux.glorot_uniform(rng, l.c.hidden, l.c.heads * l.c.head_dim)
)
Lux.initialstates(::AbstractRNG, ::MQAAttention) = NamedTuple()

apply_rope!(q, k, freqs, positions) = begin
    _, seq, heads, hdim = size(q)
    @inbounds for s in 1:seq, h in 1:heads, i in 1:(hdim÷2)
        angle = positions[s] * freqs[i]
        c, sn = cos(angle), sin(angle)
        q_re, q_im = q[1, s, h, 2i-1], q[1, s, h, 2i]
        q[1, s, h, 2i-1] = q_re * c - q_im * sn
        q[1, s, h, 2i] = q_re * sn + q_im * c
        k_re, k_im = k[1, s, h, 2i-1], k[1, s, h, 2i]
        k[1, s, h, 2i-1] = k_re * c - k_im * sn
        k[1, s, h, 2i] = k_re * sn + k_im * c
    end
end

compute_qkv(x, ps, hidden, nh, nkv, hd) = begin
    xr = reshape(x, hidden, :)
    q = ps.q * xr |> q -> reshape(q, 1, size(x, 2), nh, hd)
    k = ps.k * xr |> k -> reshape(k, 1, size(x, 2), nkv, hd)
    v = ps.v * xr |> v -> reshape(v, 1, size(x, 2), nkv, hd)
    (q, k, v)
end

causal_mask!(scores, seq) = @inbounds for h in 1:size(scores, 2), qi in 1:seq, ki in qi+1:seq
    scores[1, h, qi, ki] = -Inf32
end

(l::MQAAttention)(x, ps, st) = begin
    _, seq, hidden = size(x)
    nh, nkv, hd = l.c.heads, l.c.kv_heads, l.c.head_dim
    scale = Float32(1 / sqrt(hd))

    q, k, v = compute_qkv(x, ps, hidden, nh, nkv, hd)

    freqs = [Float32(1.0 / (l.c.rope_base ^ (2i / hd))) for i in 0:(hd÷2-1)]
    apply_rope!(q, k, freqs, Float32.(0:seq-1))

    k_exp, v_exp = repeat(k, outer=(1, 1, nh÷nkv, 1)), repeat(v, outer=(1, 1, nh÷nkv, 1))

    scores = zeros(Float32, 1, nh, seq, seq)
    @inbounds for h in 1:nh, qi in 1:seq, ki in 1:seq
        scores[1, h, qi, ki] = sum(@views(q[1, qi, h, :]) .* @views(k_exp[1, ki, h, :])) * scale
    end
    causal_mask!(scores, seq)

    attn = mapslices(col_softmax, scores, dims=4)

    out = zeros(Float32, 1, seq, nh, hd)
    @inbounds for h in 1:nh, qi in 1:seq, d in 1:hd
        out[1, qi, h, d] = sum(@views(attn[1, h, qi, :]) .* @views(v_exp[1, :, h, d]))
    end

    ps.o * reshape(out, nh * hd, seq) |> y -> (reshape(y, hidden, 1, seq), st)
end

struct SharedExpert <: Lux.AbstractLuxLayer c::Config end
Lux.initialparameters(rng::AbstractRNG, l::SharedExpert) = (
    gate=Lux.glorot_uniform(rng, l.c.ffn, l.c.hidden),
    up=Lux.glorot_uniform(rng, l.c.ffn, l.c.hidden),
    down=Lux.glorot_uniform(rng, l.c.hidden, l.c.ffn)
)
(l::SharedExpert)(x, ps, st) = begin
    sz = size(x)
    x |> x -> reshape(x, sz[1], :) |>
         xr -> ps.down * @.(NNlib.swish(ps.gate * xr) * (ps.up * xr)) |>
         y -> (reshape(y, sz[1], sz[2:end]...), st)
end

struct Router <: Lux.AbstractLuxLayer c::Config end
Lux.initialparameters(rng::AbstractRNG, l::Router) = (gate=Lux.glorot_uniform(rng, l.c.experts, l.c.hidden),)
Lux.initialstates(::AbstractRNG, ::Router) = NamedTuple()

col_softmax(col) = col |> c -> (mx = maximum(c); exp.(c .- mx)) |> e -> e ./ sum(e)

topk_select(probs_col, k) = begin
    sorted_idx = sortperm(probs_col, rev=true)
    top_idx = @views sorted_idx[1:k]
    top_probs = @views probs_col[top_idx]
    (top_probs ./ sum(top_probs), top_idx)
end

(l::Router)(x, ps, st) = begin
    hidden, batch, seq = size(x)
    logits = ps.gate * reshape(x, hidden, :)
    probs = mapslices(col_softmax, logits, dims=1)

    n_tokens = batch * seq
    weights = zeros(Float32, l.c.topk, n_tokens)
    indices = zeros(Int, l.c.topk, n_tokens)

    @inbounds for t in 1:n_tokens
        weights[:, t], indices[:, t] = topk_select(@views(probs[:, t]), l.c.topk)
    end

    ((weights, indices), st)
end

# Abstract low-rank strategies
abstract type AbstractLowRank end
struct LoRA <: AbstractLowRank end  # Standard LoRA
struct DoRA <: AbstractLowRank end  # Weight-decomposed LoRA (future)

# Dispatch-based low-rank factorization
lowrank_scale(::LoRA, r) = Float32(sqrt(1 / r))
lowrank_scale(::DoRA, r) = Float32(1 / r)

struct LowRankDelta{LR<:AbstractLowRank} <: Lux.AbstractLuxLayer
    c::Config
    strategy::LR
end
LowRankDelta(c::Config, strategy::LR=LoRA()) where LR = LowRankDelta(c, strategy)

Lux.initialparameters(rng::AbstractRNG, l::LowRankDelta) = begin
    ne, r, h, f = l.c.experts, l.c.r, l.c.hidden, l.c.ffn
    scale = lowrank_scale(l.strategy, r)
    (
        a_gate=[Lux.randn32(rng, h, r) .* scale for _ in 1:ne],
        b_gate=[Lux.randn32(rng, r, f) .* scale for _ in 1:ne],
        a_up=[Lux.randn32(rng, h, r) .* scale for _ in 1:ne],
        b_up=[Lux.randn32(rng, r, f) .* scale for _ in 1:ne],
        a_down=[Lux.randn32(rng, f, r) .* scale for _ in 1:ne],
        b_down=[Lux.randn32(rng, r, h) .* scale for _ in 1:ne]
    )
end
Lux.initialstates(::AbstractRNG, ::LowRankDelta) = NamedTuple()

(l::LowRankDelta{LoRA})(x, idx, ps, st) = begin
    gate, up, down = ps.a_gate[idx] * ps.b_gate[idx], ps.a_up[idx] * ps.b_up[idx], ps.a_down[idx] * ps.b_down[idx]
    down * @.(NNlib.swish(gate * x) * (up * x)) |> y -> (y, st)
end

struct MoE <: Lux.AbstractLuxLayer c::Config; shared::SharedExpert; router::Router; delta::LowRankDelta end
MoE(c::Config) = MoE(c, SharedExpert(c), Router(c), LowRankDelta(c))
function Lux.initialparameters(rng::AbstractRNG, l::MoE)
    (shared=Lux.initialparameters(rng, l.shared),
     router=Lux.initialparameters(rng, l.router),
     delta=Lux.initialparameters(rng, l.delta))
end
function Lux.initialstates(rng::AbstractRNG, l::MoE)
    (shared=Lux.initialstates(rng, l.shared),
     router=Lux.initialstates(rng, l.router),
     delta=Lux.initialstates(rng, l.delta))
end

function (l::MoE)(x, ps, st)
    base, st_s = l.shared(x, ps.shared, st.shared)
    (weights, indices), st_r = l.router(x, ps.router, st.router)

    hidden, batch, seq = size(x)
    xr = reshape(x, hidden, batch * seq)
    delta_out = zeros(Float32, hidden, batch * seq)

    for t in 1:batch*seq
        for k in 1:l.c.topk
            expert_idx = indices[k, t]
            expert_x = xr[:, t:t]
            expert_out, _ = l.delta(expert_x, expert_idx, ps.delta, st.delta)
            delta_out[:, t] .+= weights[k, t] .* expert_out[:, 1]
        end
    end

    final = base + reshape(delta_out, hidden, batch, seq)
    (final, (shared=st_s, router=st_r, delta=st.delta))
end

struct RecurrentBlock <: Lux.AbstractLuxLayer c::Config; attn::MQAAttention; moe::MoE end
RecurrentBlock(c::Config) = RecurrentBlock(c, MQAAttention(c), MoE(c))
Lux.initialparameters(rng::AbstractRNG, l::RecurrentBlock) = (attn=Lux.initialparameters(rng, l.attn), moe=Lux.initialparameters(rng, l.moe), t_emb=Lux.randn32(rng, l.c.hidden, l.c.loops))
Lux.initialstates(rng::AbstractRNG, l::RecurrentBlock) = (attn=Lux.initialstates(rng, l.attn), moe=Lux.initialstates(rng, l.moe))

function (l::RecurrentBlock)(x, ps, st)
    h, st_a, st_m = x, st.attn, st.moe
    for i in 1:l.c.loops
        h_t = h .+ ps.t_emb[:, i]
        h_a, st_a = l.attn(erf.(h_t), ps.attn, st_a)
        h_m, st_m = l.moe(erf.(h .+ h_a), ps.moe, st_m)
        h = h .+ h_a .+ h_m
    end
    h, (attn=st_a, moe=st_m)
end

struct BLTRecurrentMoE <: Lux.AbstractLuxLayer c::Config; input::BLTInput; blk::RecurrentBlock; output::Lux.Dense end
BLTRecurrentMoE(c::Config=C1_9B) = BLTRecurrentMoE(c, BLTInput(c), RecurrentBlock(c), Lux.Dense(c.hidden => c.vocab, use_bias=false))
Lux.initialparameters(rng::AbstractRNG, l::BLTRecurrentMoE) = (input=Lux.initialparameters(rng, l.input), blk=Lux.initialparameters(rng, l.blk), output=Lux.initialparameters(rng, l.output))
Lux.initialstates(rng::AbstractRNG, l::BLTRecurrentMoE) = (input=Lux.initialstates(rng, l.input), blk=Lux.initialstates(rng, l.blk), output=Lux.initialstates(rng, l.output))
(m::BLTRecurrentMoE)(x, ps, st) = (
    (y, st_i) = m.input(x, ps.input, st.input);
    (y, st_b) = m.blk(y, ps.blk, st.blk);
    (y, st_o) = m.output(y, ps.output, st.output);
    (y, (input=st_i, blk=st_b, output=st_o))
)

struct BLTRecurrentMoEWithPatching <: Lux.AbstractLuxLayer
    c::Config
    input::BLTInputWithPatching
    blk::RecurrentBlock
    output::Lux.Dense
end
BLTRecurrentMoEWithPatching(c::Config=C1_9B; kwargs...) =
    BLTRecurrentMoEWithPatching(c, BLTInputWithPatching(c; kwargs...),
                                RecurrentBlock(c), Lux.Dense(c.hidden => c.vocab, use_bias=false))

Lux.initialparameters(rng::AbstractRNG, l::BLTRecurrentMoEWithPatching) =
    (input=Lux.initialparameters(rng, l.input), blk=Lux.initialparameters(rng, l.blk),
     output=Lux.initialparameters(rng, l.output))

Lux.initialstates(rng::AbstractRNG, l::BLTRecurrentMoEWithPatching) =
    (input=Lux.initialstates(rng, l.input), blk=Lux.initialstates(rng, l.blk),
     output=Lux.initialstates(rng, l.output))

function (m::BLTRecurrentMoEWithPatching)(x, ps, st)
    (y, st_i) = m.input(x, ps.input, st.input)
    (y, st_b) = m.blk(y, ps.blk, st.blk)
    (y, st_o) = m.output(y, ps.output, st.output)
    (y, (input=st_i, blk=st_b, output=st_o))
end

# --- Quantization ---
abstract type QScheme end
struct INT4 <: QScheme end
struct NF4 <: QScheme end
const NF4_CB = Float32[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]

# Dispatch-based quantization
scale(::INT4, x) = 15f0 / (2maximum(abs, x) + eps(Float32))
quant(::INT4, x, s) = (clamp.(round.(x .* s), -8, 7) .|> Int8, s)
dequant(::INT4, q, s) = Float32.(q) ./ s

quant_block(block, i, bs, n) = begin
    r = (i-1)*bs+1:min(i*bs,n)
    absmax = maximum(abs, @views block[r])
    normalized = @views(block[r]) ./ (absmax + eps(Float32))
    (map(y -> argmin(abs.(NF4_CB .- y)) - 1, normalized), absmax)
end

quant(::NF4, x, bs=64) = begin
    n, nb = length(x), cld(length(x), bs)
    q, am = zeros(UInt8, n), zeros(Float32, nb)
    @inbounds for i in 1:nb
        r = (i-1)*bs+1:min(i*bs,n)
        q[r], am[i] = quant_block(x, i, bs, n)
    end
    (q, am)
end

dequant(::NF4, q, am, bs=64) = begin
    r = zeros(Float32, length(q))
    @inbounds for i in 1:length(am)
        rng = (i-1)*bs+1:min(i*bs,length(q))
        r[rng] = @views(NF4_CB[q[rng].+1]) .* am[i]
    end
    r
end

struct QLinear{S<:QScheme} <: Lux.AbstractLuxLayer dims_in::Int; dims_out::Int; s::S end
QLinear(din::Int, dout::Int, s::S=INT4()) where S = QLinear(din, dout, s)
Lux.initialparameters(rng::AbstractRNG, l::QLinear{INT4}) = (w=rand(rng, Int8, l.dims_out, l.dims_in), s=rand(rng, Float32))
Lux.initialparameters(rng::AbstractRNG, l::QLinear{NF4}) = (w=rand(rng, UInt8, l.dims_out*l.dims_in), am=rand(rng, Float32, cld(l.dims_out*l.dims_in, 64))) 
Lux.initialstates(::AbstractRNG, ::QLinear) = NamedTuple()
(l::QLinear{INT4})(x, ps, st) = (dequant(INT4(), ps.w, ps.s) * x, st)
(l::QLinear{NF4})(x, ps, st) = (reshape(dequant(NF4(), ps.w, ps.am), l.dims_out, l.dims_in) * x, st)

# --- Training ---
struct Muon <: Optimisers.AbstractRule lr::Float64; mom::Float64; nesterov::Bool end
Muon(; lr=1e-3, mom=0.95, nesterov=true) = Muon(lr, mom, nesterov)
Optimisers.init(::Muon, x::AbstractArray) = zeros(eltype(x), size(x))

# Dispatch-based orthogonalization
orth_update!(upd::AbstractVector) = upd
orth_update!(upd::AbstractMatrix) = (m, n = size(upd); m > 1 && orth!(upd); upd)
orth!(W::AbstractMatrix) = begin
    m, n = size(W)
    m >= n ? (W .= W * (3I - W'W) / 2) : (W .= (3I - W*W') * W / 2)
end

Optimisers.apply!(o::Muon, state, x, dx) = begin
    b = @. o.mom * state + dx
    upd = o.nesterov ? @.(o.mom * b + dx) : b
    (b, orth_update!(upd) .* o.lr)
end

train!(model, data; epochs=1, gpu=false) = begin
    rng = Random.default_rng(); ps, st = Lux.setup(rng, model)
    if gpu; ps, st = (ps, st) .|> Lux.gpu; end
    opt_st = Optimisers.setup(AdamW(eta=1e-3), ps)
    for ep in 1:epochs, (x, y) in data
        grads = Zygote.gradient(p -> sum(first(model(x, p, st))), ps)[1]
        opt_st, ps = Optimisers.update(opt_st, ps, grads)
        @printf "Epoch %d: Loss %.4f\n" ep 0.0
    end
    ps, st
end

export Config, ModelSize, M1_9B, M8B, C1_9B, C8B
export load_config, load_full_config
export AbstractActivation, Erf, SiLU, GELU, activate
export Derf, RoPE, MQAAttention, Router, SharedExpert, LowRankDelta, MoE
export RecurrentBlock, BLTInput, BLTInputWithPatching
export BLTRecurrentMoE, BLTRecurrentMoEWithPatching
export AbstractPatcher, NoPatcher, EntropyPatcher, EMASmoothing, aggregate_patches
export AbstractLowRank, LoRA, DoRA
export QScheme, INT4, NF4, QLinear, quant, dequant, scale
export Muon, orth_update!, train!

end
