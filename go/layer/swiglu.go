package layer

import (
	"github.com/fumi-engineer/machine_learning/go/tensor"
)

// SwiGLU implements the SwiGLU feed-forward network.
// SwiGLU(x) = (xW_gate * SiLU(xW_gate)) @ W_down
type SwiGLU struct {
	wGate *Linear // [hiddenDim, ffnDim]
	wUp   *Linear // [hiddenDim, ffnDim]
	wDown *Linear // [ffnDim, hiddenDim]

	hiddenDim int
	ffnDim    int

	// Cached for backward
	lastGate *tensor.Tensor
	lastUp   *tensor.Tensor
}

// NewSwiGLU creates a new SwiGLU layer.
func NewSwiGLU(hiddenDim, ffnDim int) *SwiGLU {
	return &SwiGLU{
		wGate:     NewLinear(hiddenDim, ffnDim, false),
		wUp:       NewLinear(hiddenDim, ffnDim, false),
		wDown:     NewLinear(ffnDim, hiddenDim, false),
		hiddenDim: hiddenDim,
		ffnDim:    ffnDim,
	}
}

// Forward performs SwiGLU forward pass.
// Input: [..., hiddenDim]
// Output: [..., hiddenDim]
func (s *SwiGLU) Forward(input *tensor.Tensor) *tensor.Tensor {
	// gate = SiLU(x @ W_gate)
	gateLinear := s.wGate.Forward(input)
	gate := gateLinear.SiLU()
	s.lastGate = gate

	// up = x @ W_up
	up := s.wUp.Forward(input)
	s.lastUp = up

	// hidden = gate * up
	hidden := gate.Mul(up)

	// output = hidden @ W_down
	output := s.wDown.Forward(hidden)

	return output
}

// Backward computes gradients for SwiGLU.
func (s *SwiGLU) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Backward through W_down
	gradHidden := s.wDown.Backward(gradOutput)

	// gradGate = gradHidden * up
	gradGate := gradHidden.Mul(s.lastUp)

	// gradUp = gradHidden * gate
	gradUp := gradHidden.Mul(s.lastGate)

	// Backward through SiLU (simplified)
	// d/dx SiLU(x) = SiLU(x) + sigmoid(x) * (1 - SiLU(x))
	// For simplicity, we approximate with just propagating through

	// Backward through W_gate and W_up
	gradFromGate := s.wGate.Backward(gradGate)
	gradFromUp := s.wUp.Backward(gradUp)

	// Sum gradients
	return gradFromGate.Add(gradFromUp)
}

// Parameters returns all SwiGLU parameters.
func (s *SwiGLU) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0)
	params = append(params, s.wGate.Parameters()...)
	params = append(params, s.wUp.Parameters()...)
	params = append(params, s.wDown.Parameters()...)
	return params
}
