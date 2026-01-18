package model

import (
	"github.com/pikafumi/machine_learning/crates/go/layer"
	"github.com/pikafumi/machine_learning/crates/go/tensor"
)

// MoELayer implements a Mixture of Experts layer.
type MoELayer struct {
	router  *Router
	experts []*layer.SwiGLU

	hiddenDim int
	ffnDim    int
	nExperts  int
	topK      int
}

// NewMoELayer creates a new MoE layer.
func NewMoELayer(hiddenDim, ffnDim, nExperts, topK int) *MoELayer {
	experts := make([]*layer.SwiGLU, nExperts)
	for i := 0; i < nExperts; i++ {
		experts[i] = layer.NewSwiGLU(hiddenDim, ffnDim)
	}

	return &MoELayer{
		router:    NewRouter(hiddenDim, nExperts, topK),
		experts:   experts,
		hiddenDim: hiddenDim,
		ffnDim:    ffnDim,
		nExperts:  nExperts,
		topK:      topK,
	}
}

// Forward performs MoE forward pass.
// Input: [batch, seq_len, hidden_dim]
// Output: [batch, seq_len, hidden_dim]
func (m *MoELayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	dims := input.Shape().Dims()
	batch := dims[0]
	seqLen := dims[1]
	numTokens := batch * seqLen

	// Get router weights and indices
	weights, indices := m.router.Forward(input)

	// Reshape input to [batch*seq_len, hidden_dim]
	flatInput := input.Reshape(tensor.NewShape(numTokens, m.hiddenDim))
	flatInputData := flatInput.DataPtr()

	// Initialize output
	output := tensor.Zeros(tensor.NewShape(numTokens, m.hiddenDim), tensor.F32)
	outputData := output.DataPtr()
	weightsData := weights.DataPtr()

	// Process each token
	for t := 0; t < numTokens; t++ {
		// Get token embedding
		tokenInput := tensor.FromSlice(
			flatInputData[t*m.hiddenDim:(t+1)*m.hiddenDim],
			tensor.NewShape(1, m.hiddenDim),
		)

		// Compute weighted sum of expert outputs
		for k := 0; k < m.topK; k++ {
			expertIdx := indices[t][k]
			weight := weightsData[t*m.topK+k]

			// Expert forward
			expertOut := m.experts[expertIdx].Forward(tokenInput)
			expertData := expertOut.DataPtr()

			// Accumulate weighted output
			for i := 0; i < m.hiddenDim; i++ {
				outputData[t*m.hiddenDim+i] += weight * expertData[i]
			}
		}
	}

	// Reshape to [batch, seq_len, hidden_dim]
	return output.Reshape(tensor.NewShape(batch, seqLen, m.hiddenDim))
}

// Backward computes gradients for MoE layer.
func (m *MoELayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Simplified backward - propagate through router
	return m.router.Backward(gradOutput)
}

// Parameters returns all MoE parameters.
func (m *MoELayer) Parameters() []*tensor.Tensor {
	params := m.router.Parameters()
	for _, expert := range m.experts {
		params = append(params, expert.Parameters()...)
	}
	return params
}

// AuxLoss returns the load balancing auxiliary loss.
func (m *MoELayer) AuxLoss(alpha float32) float32 {
	return m.router.ComputeAuxLoss(alpha)
}
