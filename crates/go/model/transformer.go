package model

import (
	"github.com/pikafumi/machine_learning/crates/go/layer"
	"github.com/pikafumi/machine_learning/crates/go/tensor"
)

// MoETransformer implements the full MoE Transformer model.
type MoETransformer struct {
	config    Config
	embedding *layer.Embedding
	blocks    []*TransformerBlock
	finalNorm *layer.RMSNorm
	lmHead    *layer.Linear
}

// NewMoETransformer creates a new MoE Transformer.
func NewMoETransformer(cfg Config) *MoETransformer {
	blocks := make([]*TransformerBlock, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		blocks[i] = NewTransformerBlock(cfg)
	}

	return &MoETransformer{
		config:    cfg,
		embedding: layer.NewEmbedding(cfg.VocabSize, cfg.HiddenDim),
		blocks:    blocks,
		finalNorm: layer.NewRMSNorm(cfg.HiddenDim, 1e-6),
		lmHead:    layer.NewLinear(cfg.HiddenDim, cfg.VocabSize, false),
	}
}

// NewDefault creates a default 6.9B model.
func NewDefault() *MoETransformer {
	return NewMoETransformer(Default6_9B())
}

// NewTiny creates a tiny model for testing.
func NewTiny() *MoETransformer {
	return NewMoETransformer(Tiny())
}

// Config returns the model configuration.
func (m *MoETransformer) Config() Config {
	return m.config
}

// NumLayers returns the number of layers.
func (m *MoETransformer) NumLayers() int {
	return m.config.NLayers
}

// Forward performs forward pass.
// Input: [batch, seq_len] token IDs (as float32)
// Output: [batch, seq_len, vocab_size] logits
func (m *MoETransformer) Forward(input *tensor.Tensor) *tensor.Tensor {
	// Embedding lookup
	x := m.embedding.Forward(input)

	// Transformer blocks
	for _, block := range m.blocks {
		x = block.Forward(x)
	}

	// Final norm
	x = m.finalNorm.Forward(x)

	// LM head
	logits := m.lmHead.Forward(x)

	return logits
}

// ForwardIDs performs forward pass with token IDs.
func (m *MoETransformer) ForwardIDs(tokenIDs []int, batch, seqLen int) *tensor.Tensor {
	// Convert to tensor
	data := make([]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		data[i] = float32(id)
	}
	input := tensor.FromSlice(data, tensor.NewShape(batch, seqLen))
	return m.Forward(input)
}

// Backward performs backward pass.
func (m *MoETransformer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Backward through LM head
	grad := m.lmHead.Backward(gradOutput)

	// Backward through final norm
	grad = m.finalNorm.Backward(grad)

	// Backward through blocks (reverse order)
	for i := len(m.blocks) - 1; i >= 0; i-- {
		grad = m.blocks[i].Backward(grad)
	}

	// Backward through embedding
	grad = m.embedding.Backward(grad)

	return grad
}

// Parameters returns all model parameters.
func (m *MoETransformer) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	params = append(params, m.embedding.Parameters()...)
	for _, block := range m.blocks {
		params = append(params, block.Parameters()...)
	}
	params = append(params, m.finalNorm.Parameters()...)
	params = append(params, m.lmHead.Parameters()...)
	return params
}

// TotalAuxLoss returns the sum of all MoE auxiliary losses.
func (m *MoETransformer) TotalAuxLoss(alpha float32) float32 {
	total := float32(0.0)
	for _, block := range m.blocks {
		total += block.AuxLoss(alpha)
	}
	return total
}

// Generate performs autoregressive generation.
func (m *MoETransformer) Generate(prompt []int, maxLen int) []int {
	tokens := make([]int, len(prompt))
	copy(tokens, prompt)

	for len(tokens) < maxLen {
		// Forward pass
		logits := m.ForwardIDs(tokens, 1, len(tokens))

		// Get last token logits
		lastIdx := len(tokens) - 1
		vocabSize := m.config.VocabSize
		logitsData := logits.DataPtr()
		lastLogits := logitsData[lastIdx*vocabSize : (lastIdx+1)*vocabSize]

		// Argmax
		maxIdx := 0
		maxVal := lastLogits[0]
		for i := 1; i < vocabSize; i++ {
			if lastLogits[i] > maxVal {
				maxVal = lastLogits[i]
				maxIdx = i
			}
		}

		tokens = append(tokens, maxIdx)
	}

	return tokens
}
