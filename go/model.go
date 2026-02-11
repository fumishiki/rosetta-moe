// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// MoETransformer is the complete Mixture-of-Experts Transformer language model.
//
// Architecture (pre-norm, decoder-only):
//   embedding -> [TransformerBlock x N_layers] -> RMSNorm -> Linear(lm_head) -> logits
//
// Each TransformerBlock contains MQAttention + MoE(SwiGLU) with pre-norm residuals.
type MoETransformer struct {
	config    Config
	embedding *Embedding
	blocks    []*TransformerBlock
	finalNorm *RMSNorm
	lmHead    *Linear
}

// NewMoETransformer constructs the full model from a Config.
func NewMoETransformer(cfg Config) *MoETransformer {
	blocks := make([]*TransformerBlock, cfg.NLayers)
	for i := range blocks {
		blocks[i] = NewTransformerBlock(cfg)
	}
	return &MoETransformer{
		config:    cfg,
		embedding: NewEmbedding(cfg.VocabSize, cfg.HiddenDim),
		blocks:    blocks,
		finalNorm: NewRMSNorm(cfg.HiddenDim, 1e-6),
		lmHead:    NewLinear(cfg.HiddenDim, cfg.VocabSize, false),
	}
}

// NewDefault creates a full-scale 6.9B MoE model.
func NewDefault() *MoETransformer { return NewMoETransformer(Default6_9B()) }

// NewTiny creates a minimal model for testing.
func NewTiny() *MoETransformer { return NewMoETransformer(Tiny()) }

// NewSmall creates a small model (hidden=256) for scale comparison benchmarks.
func NewSmall() *MoETransformer { return NewMoETransformer(Small()) }

// TinyModel is an alias for NewTiny (convenience).
func TinyModel() *MoETransformer { return NewTiny() }

// DefaultModel is an alias for NewDefault (convenience).
func DefaultModel() *MoETransformer { return NewDefault() }

// Config returns the model's configuration.
func (m *MoETransformer) Config() Config { return m.config }

// NumLayers returns the number of Transformer blocks.
func (m *MoETransformer) NumLayers() int { return m.config.NLayers }

// Forward runs the complete model: embedding -> blocks -> norm -> lm_head.
// Input: [batch, seq_len] of token IDs (as float32). Output: [batch, seq_len, vocab_size] logits.
func (m *MoETransformer) Forward(input *Tensor) *Tensor {
	x := m.embedding.Forward(input)
	for _, blk := range m.blocks {
		x = blk.Forward(x)
	}
	return m.lmHead.Forward(m.finalNorm.Forward(x))
}

// ForwardIDs is a convenience wrapper that converts int token IDs to a tensor.
func (m *MoETransformer) ForwardIDs(tokenIDs []int, batch, seqLen int) *Tensor {
	return m.Forward(FromSlice(idsToF32(tokenIDs), NewShape(batch, seqLen)))
}

// Backward propagates gradients through the model in reverse layer order.
func (m *MoETransformer) Backward(gradOutput *Tensor) *Tensor {
	grad := m.finalNorm.Backward(m.lmHead.Backward(gradOutput))
	for i := len(m.blocks) - 1; i >= 0; i-- {
		grad = m.blocks[i].Backward(grad)
	}
	return m.embedding.Backward(grad)
}

// Parameters returns all trainable parameters in the model.
func (m *MoETransformer) Parameters() []*Tensor {
	p := append([]*Tensor(nil), m.embedding.Parameters()...)
	for _, blk := range m.blocks {
		p = append(p, blk.Parameters()...)
	}
	return concatParams(p, m.finalNorm.Parameters(), m.lmHead.Parameters())
}

// TotalAuxLoss sums the auxiliary load-balancing losses from all MoE layers.
func (m *MoETransformer) TotalAuxLoss(alpha float32) float32 {
	total := float32(0)
	for _, blk := range m.blocks {
		total += blk.AuxLoss(alpha)
	}
	return total
}

// Generate produces tokens using greedy decoding (default strategy).
func (m *MoETransformer) Generate(prompt []int, maxLen int) []int {
	return m.GenerateGreedy(prompt, maxLen)
}

// idsToF32 converts int token IDs to float32 for use as tensor data.
// Token IDs are stored as float32 in the embedding input tensor because
// the Tensor type only supports []float32 storage.
func idsToF32(ids []int) []float32 {
	out := make([]float32, len(ids))
	for i, id := range ids {
		out[i] = float32(id)
	}
	return out
}
