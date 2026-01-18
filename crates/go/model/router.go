package model

import (
	"sort"

	"github.com/pikafumi/machine_learning/crates/go/layer"
	"github.com/pikafumi/machine_learning/crates/go/tensor"
)

// Router implements the MoE router with top-k selection.
type Router struct {
	gate     *layer.Linear
	nExperts int
	topK     int

	// Cached for backward
	lastInput    *tensor.Tensor
	lastWeights  *tensor.Tensor
	lastIndices  [][]int
	lastGateProb *tensor.Tensor
}

// NewRouter creates a new MoE router.
func NewRouter(hiddenDim, nExperts, topK int) *Router {
	return &Router{
		gate:     layer.NewLinear(hiddenDim, nExperts, false),
		nExperts: nExperts,
		topK:     topK,
	}
}

// indexValue is used for sorting.
type indexValue struct {
	index int
	value float32
}

// Forward computes router weights and expert indices.
// Input: [batch, seq_len, hidden_dim]
// Returns: weights [batch*seq_len, topK], indices [batch*seq_len][topK]
func (r *Router) Forward(input *tensor.Tensor) (*tensor.Tensor, [][]int) {
	r.lastInput = input.Clone()

	dims := input.Shape().Dims()
	batch := dims[0]
	seqLen := dims[1]
	hiddenDim := dims[2]
	numTokens := batch * seqLen

	// Reshape to [batch*seq_len, hidden_dim]
	flatInput := input.Reshape(tensor.NewShape(numTokens, hiddenDim))

	// Gate logits: [batch*seq_len, nExperts]
	gateLogits := r.gate.Forward(flatInput)

	// Softmax
	gateProbs := gateLogits.Softmax()
	r.lastGateProb = gateProbs

	// Top-k selection
	weights := tensor.New(tensor.NewShape(numTokens, r.topK), tensor.F32)
	weightsData := weights.DataPtr()
	probsData := gateProbs.DataPtr()
	r.lastIndices = make([][]int, numTokens)

	for t := 0; t < numTokens; t++ {
		offset := t * r.nExperts

		// Create index-value pairs
		pairs := make([]indexValue, r.nExperts)
		for e := 0; e < r.nExperts; e++ {
			pairs[e] = indexValue{index: e, value: probsData[offset+e]}
		}

		// Sort by value descending
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].value > pairs[j].value
		})

		// Select top-k
		r.lastIndices[t] = make([]int, r.topK)
		sum := float32(0.0)
		for k := 0; k < r.topK; k++ {
			r.lastIndices[t][k] = pairs[k].index
			sum += pairs[k].value
		}

		// Normalize weights
		for k := 0; k < r.topK; k++ {
			weightsData[t*r.topK+k] = pairs[k].value / sum
		}
	}

	r.lastWeights = weights
	return weights, r.lastIndices
}

// Backward computes gradients for router.
// Note: This is a simplified backward that returns gradient of same shape as input.
// Full implementation would backprop through softmax and top-k selection.
func (r *Router) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Return gradient of same shape as original input
	return tensor.Zeros(r.lastInput.Shape(), tensor.F32)
}

// Parameters returns router parameters.
func (r *Router) Parameters() []*tensor.Tensor {
	return r.gate.Parameters()
}

// ComputeAuxLoss computes load balancing auxiliary loss.
func (r *Router) ComputeAuxLoss(alpha float32) float32 {
	if r.lastGateProb == nil {
		return 0.0
	}

	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastGateProb.Shape().At(0)

	// Compute fraction of tokens routed to each expert
	expertCounts := make([]float32, r.nExperts)
	expertProbs := make([]float32, r.nExperts)

	for t := 0; t < numTokens; t++ {
		for k := 0; k < r.topK; k++ {
			expertIdx := r.lastIndices[t][k]
			expertCounts[expertIdx] += 1.0
		}
		for e := 0; e < r.nExperts; e++ {
			expertProbs[e] += probsData[t*r.nExperts+e]
		}
	}

	// Normalize
	totalAssignments := float32(numTokens * r.topK)
	for e := 0; e < r.nExperts; e++ {
		expertCounts[e] /= totalAssignments
		expertProbs[e] /= float32(numTokens)
	}

	// Aux loss = alpha * N * sum(f_i * P_i)
	auxLoss := float32(0.0)
	for e := 0; e < r.nExperts; e++ {
		auxLoss += expertCounts[e] * expertProbs[e]
	}
	auxLoss *= alpha * float32(r.nExperts)

	return auxLoss
}
