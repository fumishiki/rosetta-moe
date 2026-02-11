# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

# generate.jl — Text generation with multiple sampling strategies
#
# Implements autoregressive token generation using the MoE Transformer.
# Sampling strategies use Julia's multiple dispatch: pick_token() dispatches
# on the strategy type, so adding a new strategy requires only a new struct
# and a new method — no if/else chains.
#
# PRNG: Uses a simple LCG (Linear Congruential Generator) for deterministic
# reproducibility across languages (same seed -> same sequence in all 4 impls).

abstract type SamplingStrategy end
struct GreedySampling <: SamplingStrategy end
struct TemperatureSampling <: SamplingStrategy
    temperature::Float32
    state::Ref{UInt64}       # mutable PRNG state (LCG)
end
struct TopKSampling <: SamplingStrategy
    k::Int
    temperature::Float32
    state::Ref{UInt64}
end
struct TopPSampling <: SamplingStrategy
    top_p::Float32
    temperature::Float32
    state::Ref{UInt64}
end

# LCG PRNG: state = state * 6364136223846793005 + 1, return upper 32 bits as [0,1)
# Matches the same constants used in the Rust/Go/Python implementations
# for cross-language reproducibility.
function next_rand01(state::Ref{UInt64})
    state[] = state[] * UInt64(6364136223846793005) + UInt64(1)
    Float32(UInt32(state[] >> 32)) / 4294967296f0
end

# Sample from a discrete probability distribution using inverse CDF
function _sample_from_probs(probs::AbstractVector{Float32}, state::Ref{UInt64})
    r = next_rand01(state)
    cum = 0f0
    @inbounds for i in eachindex(probs)
        cum += probs[i]
        r <= cum && return i
    end
    length(probs)
end

# Temperature sampling: scale logits by 1/T, then softmax, then sample
# Temperature=0 falls back to greedy (argmax).
function _sample_from_logits(logits::AbstractVector{Float32}, temperature::Float32, state::Ref{UInt64})
    if temperature <= 0f0
        idx, _ = argmax_f32(logits)
        return idx
    end
    scaled = logits .* (1f0 / temperature)
    softmax_in_place!(scaled)
    _sample_from_probs(scaled, state)
end

# Top-K sampling: keep only the K highest-scoring tokens, mask rest to -inf,
# then apply temperature sampling.
function _sample_topk(logits::AbstractVector{Float32}, k::Int, temperature::Float32, state::Ref{UInt64})
    n = length(logits)
    (k <= 0 || k >= n) && return _sample_from_logits(logits, temperature, state)
    top_idx = partialsortperm(logits, 1:k; rev=true)
    filtered = fill(-Float32(3.4028235e38), n)
    @inbounds for i in top_idx
        filtered[i] = logits[i]
    end
    _sample_from_logits(filtered, temperature, state)
end

# Top-P (nucleus) sampling: keep tokens whose cumulative probability >= top_p,
# then sample from this nucleus set.
function _sample_topp(logits::AbstractVector{Float32}, top_p::Float32, temperature::Float32, state::Ref{UInt64})
    (top_p <= 0f0 || top_p >= 1f0) && return _sample_from_logits(logits, temperature, state)
    if temperature <= 0f0
        idx, _ = argmax_f32(logits)
        return idx
    end
    scaled = logits .* (1f0 / temperature)
    softmax_in_place!(scaled)
    indices = sortperm(scaled; rev=true)
    selected = Int[]
    s = 0f0
    for idx in indices
        push!(selected, idx)
        s += scaled[idx]
        s >= top_p && break
    end
    trunc = Float32[scaled[i] for i in selected]
    normalize_in_place!(trunc)
    chosen = _sample_from_probs(trunc, state)
    selected[chosen]
end

# --- Multiple dispatch: pick_token selects sampling strategy at the type level ---

function pick_token(::GreedySampling, logits::AbstractVector{Float32})
    idx, _ = argmax_f32(logits)
    idx - 1  # 0-based token ID
end

function pick_token(s::TemperatureSampling, logits::AbstractVector{Float32})
    _sample_from_logits(logits, s.temperature, s.state) - 1
end

function pick_token(s::TopKSampling, logits::AbstractVector{Float32})
    _sample_topk(logits, s.k, s.temperature, s.state) - 1
end

function pick_token(s::TopPSampling, logits::AbstractVector{Float32})
    _sample_topp(logits, s.top_p, s.temperature, s.state) - 1
end

# Autoregressive generation loop:
#   1. Forward pass on current token sequence
#   2. Extract logits for the last position
#   3. Sample next token using the chosen strategy
#   4. Append and repeat
# Note: This is a naive implementation that re-computes the full sequence
# each step (no KV cache). Suitable for educational purposes.
function generate(m::MoETransformer, prompt::Vector{Int}, max_len::Int, strategy::SamplingStrategy)
    tokens = copy(prompt)
    isempty(tokens) && push!(tokens, 0)
    while length(tokens) < max_len
        logits = forward_ids(m, tokens, 1, length(tokens))
        last = @view logits.data[1, length(tokens), :]  # logits for last position
        push!(tokens, pick_token(strategy, last))
    end
    tokens
end

# --- Convenience wrappers for common strategies ---

generate_greedy(m::MoETransformer, prompt::Vector{Int}, max_len::Int) =
    generate(m, prompt, max_len, GreedySampling())

generate_sample(m::MoETransformer, prompt::Vector{Int}, max_len::Int,
                temperature::Float32, seed::UInt64) =
    generate(m, prompt, max_len, TemperatureSampling(temperature, Ref(seed)))

generate_topk(m::MoETransformer, prompt::Vector{Int}, max_len::Int,
              k::Int, temperature::Float32, seed::UInt64) =
    generate(m, prompt, max_len, TopKSampling(k, temperature, Ref(seed)))

generate_topp(m::MoETransformer, prompt::Vector{Int}, max_len::Int,
              top_p::Float32, temperature::Float32, seed::UInt64) =
    generate(m, prompt, max_len, TopPSampling(top_p, temperature, Ref(seed)))
