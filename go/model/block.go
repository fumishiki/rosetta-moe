package model

import (
	"github.com/fumi-engineer/machine_learning/go/layer"
	"github.com/fumi-engineer/machine_learning/go/tensor"
)

// TransformerBlock implements a single transformer block.
type TransformerBlock struct {
	attnNorm *layer.RMSNorm
	attention *MQAttention
	ffnNorm  *layer.RMSNorm
	moe      *MoELayer
}

// NewTransformerBlock creates a new transformer block.
func NewTransformerBlock(cfg Config) *TransformerBlock {
	return &TransformerBlock{
		attnNorm:  layer.NewRMSNorm(cfg.HiddenDim, 1e-6),
		attention: NewMQAttention(cfg.HiddenDim, cfg.NHeads, cfg.NKVHeads, cfg.HeadDim, cfg.RoPEBase, cfg.RoPEAlpha),
		ffnNorm:   layer.NewRMSNorm(cfg.HiddenDim, 1e-6),
		moe:       NewMoELayer(cfg.HiddenDim, cfg.FFNDim, cfg.NExperts, cfg.TopKExperts),
	}
}

// Forward performs transformer block forward pass.
// Input: [batch, seq_len, hidden_dim]
// Output: [batch, seq_len, hidden_dim]
func (b *TransformerBlock) Forward(input *tensor.Tensor) *tensor.Tensor {
	// Pre-norm attention
	normed := b.attnNorm.Forward(input)
	attnOut := b.attention.Forward(normed)
	x := input.Add(attnOut) // Residual

	// Pre-norm FFN (MoE)
	normed = b.ffnNorm.Forward(x)
	moeOut := b.moe.Forward(normed)
	x = x.Add(moeOut) // Residual

	return x
}

// Backward computes gradients for transformer block.
func (b *TransformerBlock) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Simplified backward
	gradMoe := b.moe.Backward(gradOutput)
	gradAttn := b.attention.Backward(gradMoe)
	return gradAttn
}

// Parameters returns block parameters.
func (b *TransformerBlock) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	params = append(params, b.attnNorm.Parameters()...)
	params = append(params, b.attention.Parameters()...)
	params = append(params, b.ffnNorm.Parameters()...)
	params = append(params, b.moe.Parameters()...)
	return params
}

// AuxLoss returns the MoE auxiliary loss.
func (b *TransformerBlock) AuxLoss(alpha float32) float32 {
	return b.moe.AuxLoss(alpha)
}
