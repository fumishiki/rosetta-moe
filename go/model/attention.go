package model

import (
	"math"

	"github.com/fumi-engineer/machine_learning/go/layer"
	"github.com/fumi-engineer/machine_learning/go/tensor"
)

// MQAttention implements Multi-Query Attention.
type MQAttention struct {
	wQ    *layer.Linear // [hiddenDim, hiddenDim]
	wK    *layer.Linear // [hiddenDim, headDim]
	wV    *layer.Linear // [hiddenDim, headDim]
	wO    *layer.Linear // [hiddenDim, hiddenDim]

	nHeads    int
	nKVHeads  int
	headDim   int
	hiddenDim int
	scale     float32

	// RoPE frequencies
	ropeBase  float32
	ropeAlpha float32
}

// NewMQAttention creates a new multi-query attention layer.
func NewMQAttention(hiddenDim, nHeads, nKVHeads, headDim int, ropeBase, ropeAlpha float32) *MQAttention {
	return &MQAttention{
		wQ:        layer.NewLinear(hiddenDim, nHeads*headDim, false),
		wK:        layer.NewLinear(hiddenDim, nKVHeads*headDim, false),
		wV:        layer.NewLinear(hiddenDim, nKVHeads*headDim, false),
		wO:        layer.NewLinear(nHeads*headDim, hiddenDim, false),
		nHeads:    nHeads,
		nKVHeads:  nKVHeads,
		headDim:   headDim,
		hiddenDim: hiddenDim,
		scale:     float32(1.0 / math.Sqrt(float64(headDim))),
		ropeBase:  ropeBase,
		ropeAlpha: ropeAlpha,
	}
}

// applyRoPE applies Rotary Position Embedding.
func (a *MQAttention) applyRoPE(q, k *tensor.Tensor, seqLen int) (*tensor.Tensor, *tensor.Tensor) {
	// Compute frequencies
	freqs := make([]float32, a.headDim/2)
	base := a.ropeBase
	if a.ropeAlpha > 1.0 {
		// NTK scaling
		base = a.ropeBase * float32(math.Pow(float64(a.ropeAlpha), float64(a.headDim)/float64(a.headDim-2)))
	}

	for i := 0; i < a.headDim/2; i++ {
		freqs[i] = float32(1.0 / math.Pow(float64(base), float64(2*i)/float64(a.headDim)))
	}

	// Apply RoPE (simplified - in-place would be more efficient)
	qData := q.DataPtr()
	kData := k.DataPtr()

	qOut := tensor.New(q.Shape(), tensor.F32)
	kOut := tensor.New(k.Shape(), tensor.F32)
	qOutData := qOut.DataPtr()
	kOutData := kOut.DataPtr()

	qDims := q.Shape().Dims()
	batch := qDims[0]
	qHeads := qDims[2]

	kDims := k.Shape().Dims()
	kHeads := kDims[2]

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			pos := float32(s)

			// Apply to Q
			for h := 0; h < qHeads; h++ {
				offset := ((b*seqLen+s)*qHeads + h) * a.headDim
				for i := 0; i < a.headDim/2; i++ {
					angle := pos * freqs[i]
					cos := float32(math.Cos(float64(angle)))
					sin := float32(math.Sin(float64(angle)))

					x0 := qData[offset+2*i]
					x1 := qData[offset+2*i+1]
					qOutData[offset+2*i] = x0*cos - x1*sin
					qOutData[offset+2*i+1] = x0*sin + x1*cos
				}
			}

			// Apply to K
			for h := 0; h < kHeads; h++ {
				offset := ((b*seqLen+s)*kHeads + h) * a.headDim
				for i := 0; i < a.headDim/2; i++ {
					angle := pos * freqs[i]
					cos := float32(math.Cos(float64(angle)))
					sin := float32(math.Sin(float64(angle)))

					x0 := kData[offset+2*i]
					x1 := kData[offset+2*i+1]
					kOutData[offset+2*i] = x0*cos - x1*sin
					kOutData[offset+2*i+1] = x0*sin + x1*cos
				}
			}
		}
	}

	return qOut, kOut
}

// Forward performs multi-query attention.
// Input: [batch, seq_len, hidden_dim]
// Output: [batch, seq_len, hidden_dim]
func (a *MQAttention) Forward(input *tensor.Tensor) *tensor.Tensor {
	dims := input.Shape().Dims()
	batch := dims[0]
	seqLen := dims[1]

	// Project Q, K, V
	q := a.wQ.Forward(input) // [batch, seq_len, nHeads * headDim]
	k := a.wK.Forward(input) // [batch, seq_len, nKVHeads * headDim]
	v := a.wV.Forward(input) // [batch, seq_len, nKVHeads * headDim]

	// Reshape to [batch, seq_len, nHeads, headDim]
	q = q.Reshape(tensor.NewShape(batch, seqLen, a.nHeads, a.headDim))
	k = k.Reshape(tensor.NewShape(batch, seqLen, a.nKVHeads, a.headDim))
	v = v.Reshape(tensor.NewShape(batch, seqLen, a.nKVHeads, a.headDim))

	// Apply RoPE
	q, k = a.applyRoPE(q, k, seqLen)

	// Compute attention scores
	// For MQA: broadcast K,V across Q heads
	output := tensor.New(tensor.NewShape(batch, seqLen, a.nHeads, a.headDim), tensor.F32)
	outputData := output.DataPtr()

	qData := q.DataPtr()
	kData := k.DataPtr()
	vData := v.DataPtr()

	for b := 0; b < batch; b++ {
		for h := 0; h < a.nHeads; h++ {
			kvHead := h % a.nKVHeads // MQA: share KV head

			// Compute attention for this head
			scores := make([]float32, seqLen*seqLen)

			for qi := 0; qi < seqLen; qi++ {
				for ki := 0; ki < seqLen; ki++ {
					// Causal mask
					if ki > qi {
						scores[qi*seqLen+ki] = float32(math.Inf(-1))
						continue
					}

					// Dot product
					dot := float32(0.0)
					qOffset := ((b*seqLen+qi)*a.nHeads + h) * a.headDim
					kOffset := ((b*seqLen+ki)*a.nKVHeads + kvHead) * a.headDim
					for d := 0; d < a.headDim; d++ {
						dot += qData[qOffset+d] * kData[kOffset+d]
					}
					scores[qi*seqLen+ki] = dot * a.scale
				}
			}

			// Softmax per query
			for qi := 0; qi < seqLen; qi++ {
				offset := qi * seqLen
				maxVal := scores[offset]
				for ki := 1; ki <= qi; ki++ {
					if scores[offset+ki] > maxVal {
						maxVal = scores[offset+ki]
					}
				}

				sum := float32(0.0)
				for ki := 0; ki <= qi; ki++ {
					scores[offset+ki] = float32(math.Exp(float64(scores[offset+ki] - maxVal)))
					sum += scores[offset+ki]
				}
				for ki := 0; ki <= qi; ki++ {
					scores[offset+ki] /= sum
				}
			}

			// Compute output
			for qi := 0; qi < seqLen; qi++ {
				outOffset := ((b*seqLen+qi)*a.nHeads + h) * a.headDim
				for d := 0; d < a.headDim; d++ {
					sum := float32(0.0)
					for ki := 0; ki <= qi; ki++ {
						vOffset := ((b*seqLen+ki)*a.nKVHeads + kvHead) * a.headDim
						sum += scores[qi*seqLen+ki] * vData[vOffset+d]
					}
					outputData[outOffset+d] = sum
				}
			}
		}
	}

	// Reshape and project output
	output = output.Reshape(tensor.NewShape(batch, seqLen, a.nHeads*a.headDim))
	return a.wO.Forward(output)
}

// Backward computes gradients for attention.
func (a *MQAttention) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Simplified backward
	return a.wO.Backward(gradOutput)
}

// Parameters returns attention parameters.
func (a *MQAttention) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	params = append(params, a.wQ.Parameters()...)
	params = append(params, a.wK.Parameters()...)
	params = append(params, a.wV.Parameters()...)
	params = append(params, a.wO.Parameters()...)
	return params
}
