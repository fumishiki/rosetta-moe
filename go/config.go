// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// Config holds the hyperparameters defining a MoE Transformer architecture.
// Two presets are provided: Default6_9B (full-scale, 6.9B total / 1.8B active)
// and Tiny (for tests and benchmarks).
type Config struct {
	HiddenDim, NLayers, NHeads, NKVHeads        int
	NExperts, TopKExperts, VocabSize, MaxSeqLen int
	FFNDim, HeadDim                             int
	RoPEBase, RoPEAlpha                         float32
}

// Default6_9B returns a full-scale MoE config: 768 hidden, 30 layers, 12 heads,
// 16 experts (top-4), 32K vocab, 32K context. Approximately 6.9B total parameters
// with 1.8B active per token.
func Default6_9B() Config {
	return Config{768, 30, 12, 1, 16, 4, 32000, 32768, 6144, 64, 10000, 8}
}

// Tiny returns a minimal config for testing: 64 hidden, 2 layers, 4 heads,
// 4 experts (top-2), 1K vocab. Small enough for fast unit tests.
func Tiny() Config {
	return Config{64, 2, 4, 1, 4, 2, 1000, 512, 256, 16, 10000, 1}
}

// Small returns a config with hidden=256 for scale comparison benchmarks.
// Same structure as Tiny but 4x larger hidden dimension.
func Small() Config {
	return Config{256, 2, 4, 1, 4, 2, 1000, 512, 1024, 64, 10000, 1}
}

// TotalParams estimates the total parameter count across ALL experts.
//   total = embedding + N_layers * (attention + router + N_experts * FFN + 2*norm) + lm_head
func (c Config) TotalParams() int {
	emb := c.VocabSize * c.HiddenDim
	attn := c.HiddenDim*c.HiddenDim*2 + c.HiddenDim*c.HeadDim*2
	perLayer := attn + c.HiddenDim*c.NExperts + c.HiddenDim*c.FFNDim*3*c.NExperts + c.HiddenDim*2
	return emb + perLayer*c.NLayers + c.HiddenDim*c.VocabSize
}

// ActiveParams estimates the parameter count actually used per token
// (only top-K experts contribute, not all N_experts).
func (c Config) ActiveParams() int {
	emb := c.VocabSize * c.HiddenDim
	attn := c.HiddenDim*c.HiddenDim*2 + c.HiddenDim*c.HeadDim*2
	perLayer := attn + c.HiddenDim*c.FFNDim*3*c.TopKExperts + c.HiddenDim*2
	return emb + perLayer*c.NLayers + c.HiddenDim*c.VocabSize
}
