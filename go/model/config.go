// Package model provides the MoE Transformer implementation.
package model

// Config holds the model configuration.
type Config struct {
	HiddenDim   int     // Hidden dimension (768)
	NLayers     int     // Number of layers (30)
	NHeads      int     // Number of attention heads (12)
	NKVHeads    int     // Number of KV heads for MQA (1)
	NExperts    int     // Number of experts (16)
	TopKExperts int     // Number of active experts (4)
	VocabSize   int     // Vocabulary size (32000)
	MaxSeqLen   int     // Maximum sequence length (32768)
	FFNDim      int     // FFN intermediate dimension (6144)
	HeadDim     int     // Head dimension (64)
	RoPEBase    float32 // RoPE base frequency (10000)
	RoPEAlpha   float32 // NTK scaling factor (8)
}

// Default6_9B returns the default 6.9B model configuration.
func Default6_9B() Config {
	return Config{
		HiddenDim:   768,
		NLayers:     30,
		NHeads:      12,
		NKVHeads:    1, // MQA
		NExperts:    16,
		TopKExperts: 4,
		VocabSize:   32000,
		MaxSeqLen:   32768,
		FFNDim:      6144,
		HeadDim:     64,
		RoPEBase:    10000.0,
		RoPEAlpha:   8.0, // NTK scaling for 256K inference
	}
}

// Tiny returns a tiny model configuration for testing.
func Tiny() Config {
	return Config{
		HiddenDim:   64,
		NLayers:     2,
		NHeads:      4,
		NKVHeads:    1,
		NExperts:    4,
		TopKExperts: 2,
		VocabSize:   1000,
		MaxSeqLen:   512,
		FFNDim:      256,
		HeadDim:     16,
		RoPEBase:    10000.0,
		RoPEAlpha:   1.0,
	}
}

// TotalParams estimates total parameters.
func (c Config) TotalParams() int {
	// Embedding
	embedding := c.VocabSize * c.HiddenDim

	// Per layer
	attention := c.HiddenDim*c.HiddenDim*2 + c.HiddenDim*c.HeadDim*2 // Q,O + K,V MQA
	router := c.HiddenDim * c.NExperts
	expertFFN := c.HiddenDim * c.FFNDim * 3 * c.NExperts // gate, up, down × experts
	norms := c.HiddenDim * 2
	perLayer := attention + router + expertFFN + norms

	// LM head
	lmHead := c.HiddenDim * c.VocabSize

	return embedding + perLayer*c.NLayers + lmHead
}

// ActiveParams estimates active parameters per token.
func (c Config) ActiveParams() int {
	embedding := c.VocabSize * c.HiddenDim

	attention := c.HiddenDim*c.HiddenDim*2 + c.HiddenDim*c.HeadDim*2
	// Only top-k experts active
	activeFFN := c.HiddenDim * c.FFNDim * 3 * c.TopKExperts
	norms := c.HiddenDim * 2
	perLayer := attention + activeFFN + norms

	lmHead := c.HiddenDim * c.VocabSize

	return embedding + perLayer*c.NLayers + lmHead
}
