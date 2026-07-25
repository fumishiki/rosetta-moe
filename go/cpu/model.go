// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package cpu

// MoETransformer is the complete Mixture-of-Experts Transformer language model.
type MoETransformer struct {
	Cfg       Config
	Embedding *Embedding
	Blocks    []*TransformerBlock
	FinalNorm *RMSNorm
	LmHead    *Linear
}

// NewMoETransformer constructs the full model from a Config.
func NewMoETransformer(cfg Config) *MoETransformer {
	blocks := make([]*TransformerBlock, cfg.NLayers)
	for i := range blocks {
		blocks[i] = NewTransformerBlock(cfg)
	}
	return &MoETransformer{
		Cfg:       cfg,
		Embedding: NewEmbedding(cfg.VocabSize, cfg.HiddenDim),
		Blocks:    blocks,
		FinalNorm: NewRMSNorm(cfg.HiddenDim, 1e-6),
		LmHead:    NewLinear(cfg.HiddenDim, cfg.VocabSize, false),
	}
}

// NewDefault creates a full-scale 6.9B MoE model.
func NewDefault() *MoETransformer { return NewMoETransformer(Default6_9B()) }

// NewTiny creates a minimal model for testing.
func NewTiny() *MoETransformer { return NewMoETransformer(Tiny()) }

// NewSmall creates a small model (hidden=256) for scale comparison benchmarks.
func NewSmall() *MoETransformer { return NewMoETransformer(Small()) }

// TinyModel is an alias for NewTiny.
func TinyModel() *MoETransformer { return NewTiny() }

// DefaultModel is an alias for NewDefault.
func DefaultModel() *MoETransformer { return NewDefault() }

// Config returns the model's configuration.
func (m *MoETransformer) Config() Config { return m.Cfg }

// NumLayers returns the number of Transformer blocks.
func (m *MoETransformer) NumLayers() int { return m.Cfg.NLayers }

// Forward runs the complete model: embedding -> blocks -> norm -> lm_head.
func (m *MoETransformer) Forward(input *Tensor) *Tensor {
	x := m.Embedding.Forward(input)
	for _, blk := range m.Blocks {
		x = blk.Forward(x)
	}
	return m.LmHead.Forward(m.FinalNorm.Forward(x))
}

// ForwardIDs is a convenience wrapper that converts int token IDs to a tensor.
func (m *MoETransformer) ForwardIDs(tokenIDs []int, batch, seqLen int) *Tensor {
	return m.Forward(FromSlice(IdsToF32(tokenIDs), NewShape(batch, seqLen)))
}

// Backward propagates gradients through the model in reverse layer order.
func (m *MoETransformer) Backward(gradOutput *Tensor) *Tensor {
	grad := m.FinalNorm.Backward(m.LmHead.Backward(gradOutput))
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		grad = m.Blocks[i].Backward(grad)
	}
	return m.Embedding.Backward(grad)
}

// Parameters returns all trainable parameters in the model.
func (m *MoETransformer) Parameters() []*Tensor {
	p := append([]*Tensor(nil), m.Embedding.Parameters()...)
	for _, blk := range m.Blocks {
		p = append(p, blk.Parameters()...)
	}
	return concatParams(p, m.FinalNorm.Parameters(), m.LmHead.Parameters())
}

// TotalAuxLoss sums the auxiliary load-balancing losses from all MoE layers.
func (m *MoETransformer) TotalAuxLoss(alpha float32) float32 {
	total := float32(0)
	for _, blk := range m.Blocks {
		total += blk.AuxLoss(alpha)
	}
	return total
}

// ApplyZLoss computes router z-loss across all layers and backprops gradients.
func (m *MoETransformer) ApplyZLoss(zWeight float32) float32 {
	total := float32(0)
	for _, blk := range m.Blocks {
		total += blk.Moe.Router.ComputeZLossWithGrad(zWeight)
	}
	return total
}

// ApplyAuxLoss computes auxiliary load-balancing loss across all layers and backprops.
func (m *MoETransformer) ApplyAuxLoss(alpha float32) float32 {
	total := float32(0)
	for _, blk := range m.Blocks {
		total += blk.Moe.Router.ComputeAuxLossWithGrad(alpha)
	}
	return total
}

// SetRoutingMode configures the routing strategy for all layers.
func (m *MoETransformer) SetRoutingMode(mode RoutingMode, lambda float32) {
	for _, blk := range m.Blocks {
		blk.SetRoutingMode(mode, lambda)
	}
}

// UpdateRoutingBiases updates BiasFree expert biases across all layers.
func (m *MoETransformer) UpdateRoutingBiases(gamma float32) {
	for _, blk := range m.Blocks {
		blk.UpdateRoutingBiases(gamma)
	}
}

// ApplyReLUL1Loss computes and backprops ReLU L1 regularization loss.
func (m *MoETransformer) ApplyReLUL1Loss() float32 {
	total := float32(0)
	for _, blk := range m.Blocks {
		total += blk.ApplyReLUL1Loss()
	}
	return total
}

// AvgActiveExperts returns the average number of active experts across all layers.
func (m *MoETransformer) AvgActiveExperts() float32 {
	if len(m.Blocks) == 0 {
		return 0
	}
	total := float32(0)
	for _, blk := range m.Blocks {
		total += blk.AvgActiveExperts()
	}
	return total / float32(len(m.Blocks))
}

// Generate produces tokens using greedy decoding.
func (m *MoETransformer) Generate(prompt []int, maxLen int) []int {
	return m.GenerateGreedy(prompt, maxLen)
}

// IdsToF32 converts int token IDs to float32 for use as tensor data.
func IdsToF32(ids []int) []float32 {
	out := make([]float32, len(ids))
	for i, id := range ids {
		out[i] = float32(id)
	}
	return out
}
