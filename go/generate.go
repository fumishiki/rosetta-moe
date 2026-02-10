// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

import "sort"

// SamplingStrategy defines how to pick the next token from logit scores.
// Implementations: GreedySampling, TemperatureSampling, TopKSampling, TopPSampling.
type SamplingStrategy interface {
	PickToken(logits []float32) int
}

// GreedySampling always picks the token with the highest logit (argmax).
type GreedySampling struct{}

// PickToken returns the index of the maximum logit.
func (g GreedySampling) PickToken(logits []float32) int {
	idx, _ := argmax(logits)
	return idx
}

// TemperatureSampling scales logits by 1/T then samples from the softmax
// distribution. Higher T = more random, lower T = more greedy.
type TemperatureSampling struct {
	Temperature float32
	State       *uint64 // PRNG state (LCG)
}

// PickToken samples a token after temperature scaling.
func (s TemperatureSampling) PickToken(logits []float32) int {
	return sampleFromLogits(logits, s.Temperature, s.State)
}

// TopKSampling restricts sampling to the K highest-probability tokens,
// then applies temperature sampling within that subset.
type TopKSampling struct {
	K           int
	Temperature float32
	State       *uint64
}

// PickToken samples from the top-K logits.
func (s TopKSampling) PickToken(logits []float32) int {
	return sampleTopKFromLogits(logits, s.K, s.Temperature, s.State)
}

// TopPSampling (nucleus sampling) includes the smallest set of tokens whose
// cumulative probability exceeds TopP, then samples from that subset.
type TopPSampling struct {
	TopP        float32
	Temperature float32
	State       *uint64
}

// PickToken samples from the nucleus (top-p) of the distribution.
func (s TopPSampling) PickToken(logits []float32) int {
	return sampleTopPFromLogits(logits, s.TopP, s.Temperature, s.State)
}

// Generate produces tokens auto-regressively using the given sampling strategy.
// At each step, the full sequence is fed through the model (no KV cache),
// the last position's logits are sampled, and the new token is appended.
func Generate(m *MoETransformer, prompt []int, maxLen int, strategy SamplingStrategy) []int {
	tokens := cloneInts(prompt)
	if len(tokens) == 0 {
		tokens = append(tokens, 0)
	}
	for len(tokens) < maxLen {
		logits := m.ForwardIDs(tokens, 1, len(tokens))
		// Extract logits for the last position only.
		// Flat offset: (seq_len - 1) * vocab_size
		last := logits.DataPtr()[(len(tokens)-1)*m.config.VocabSize : len(tokens)*m.config.VocabSize]
		tokens = append(tokens, strategy.PickToken(last))
	}
	return tokens
}

// GenerateGreedy is a convenience method for greedy (argmax) decoding.
func (m *MoETransformer) GenerateGreedy(prompt []int, maxLen int) []int {
	return Generate(m, prompt, maxLen, GreedySampling{})
}

// GenerateSample generates with temperature sampling.
func (m *MoETransformer) GenerateSample(prompt []int, maxLen int, temperature float32, seed uint64) []int {
	return Generate(m, prompt, maxLen, TemperatureSampling{Temperature: temperature, State: &seed})
}

// GenerateTopK generates with top-K sampling.
func (m *MoETransformer) GenerateTopK(prompt []int, maxLen, k int, temperature float32, seed uint64) []int {
	return Generate(m, prompt, maxLen, TopKSampling{K: k, Temperature: temperature, State: &seed})
}

// GenerateTopP generates with nucleus (top-P) sampling.
func (m *MoETransformer) GenerateTopP(prompt []int, maxLen int, topP, temperature float32, seed uint64) []int {
	return Generate(m, prompt, maxLen, TopPSampling{TopP: topP, Temperature: temperature, State: &seed})
}

// softmaxInPlace applies softmax to xs in-place with numerical stability.
//   p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
func softmaxInPlace(xs []float32) {
	if len(xs) == 0 {
		return
	}
	_, maxVal := argmax(xs)
	for i := range xs {
		xs[i] = ExpF32(xs[i] - maxVal)
	}
	normalizeInPlace(xs)
}

// nextRand01 returns a pseudo-random float32 in [0, 1) using a 64-bit LCG.
// The multiplier is from Knuth's MMIX. Using a simple LCG avoids importing
// math/rand, keeping sampling deterministic and reproducible from a seed.
func nextRand01(state *uint64) float32 {
	*state = *state*6364136223846793005 + 1
	return float32(uint32(*state>>32)) / 4294967296.0
}

// sampleFromProbs samples an index from a discrete probability distribution
// using the inverse CDF method.
func sampleFromProbs(probs []float32, state *uint64) int {
	r := nextRand01(state)
	cum := float32(0)
	for i, p := range probs {
		cum += p
		if r <= cum {
			return i
		}
	}
	return len(probs) - 1
}

// sampleFromLogits applies temperature scaling, softmax, then samples.
//   scaled_i = logit_i / temperature
//   p = softmax(scaled)
//   sample from p
func sampleFromLogits(logits []float32, temperature float32, state *uint64) int {
	if temperature <= 0 {
		idx, _ := argmax(logits)
		return idx
	}
	scaled := make([]float32, len(logits))
	invTemp := float32(1.0) / temperature
	for i, v := range logits {
		scaled[i] = v * invTemp
	}
	softmaxInPlace(scaled)
	return sampleFromProbs(scaled, state)
}

// sampleTopKFromLogits keeps only the K highest logits (setting the rest to -inf),
// then delegates to temperature sampling on the filtered distribution.
func sampleTopKFromLogits(logits []float32, k int, temperature float32, state *uint64) int {
	n := len(logits)
	if k <= 0 || k >= n {
		return sampleFromLogits(logits, temperature, state)
	}

	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return logits[indices[i]] > logits[indices[j]]
	})

	filtered := make([]float32, n)
	for i := range filtered {
		filtered[i] = NegInf
	}
	for i := 0; i < k; i++ {
		filtered[indices[i]] = logits[indices[i]]
	}
	return sampleFromLogits(filtered, temperature, state)
}

// sampleTopPFromLogits implements nucleus sampling: accumulates tokens from
// highest to lowest probability until the cumulative probability exceeds topP,
// then samples from that nucleus.
func sampleTopPFromLogits(logits []float32, topP, temperature float32, state *uint64) int {
	if topP <= 0 || topP >= 1 {
		return sampleFromLogits(logits, temperature, state)
	}
	if temperature <= 0 {
		idx, _ := argmax(logits)
		return idx
	}

	scaled := make([]float32, len(logits))
	invTemp := float32(1.0) / temperature
	for i, v := range logits {
		scaled[i] = v * invTemp
	}
	softmaxInPlace(scaled)

	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return scaled[indices[i]] > scaled[indices[j]]
	})

	// Accumulate highest-probability tokens until sum >= topP
	selected := make([]int, 0, len(indices))
	sum := float32(0)
	for _, idx := range indices {
		selected = append(selected, idx)
		sum += scaled[idx]
		if sum >= topP {
			break
		}
	}

	trunc := make([]float32, len(selected))
	for i, idx := range selected {
		trunc[i] = scaled[idx]
	}
	normalizeInPlace(trunc)

	chosen := sampleFromProbs(trunc, state)
	return selected[chosen]
}
