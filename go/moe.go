// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// ---------------------------------------------------------------------------
// Router (Gating Network)
// ---------------------------------------------------------------------------

// Router selects the top-K experts for each token via a learned gate.
//
// Gating:
//
//	gate_probs = softmax(W_gate @ x)    -- probability over all experts
//	top_k_indices = argmax_k(gate_probs)
//	top_k_weights = normalize(gate_probs[top_k_indices])
//
// The top-k weights are renormalized to sum to 1 so the MoE output is a
// proper weighted average of expert outputs.
type Router struct {
	gate         *Linear
	nExperts     int
	topK         int
	lastInput    *Tensor
	lastWeights  *Tensor
	lastIndices  [][]int
	lastGateProb *Tensor // cached for auxiliary load-balancing loss
	// Reusable buffers to reduce per-forward allocation pressure.
	softmaxBuf  *Tensor // reusable softmax output
	selected    []bool  // reusable per-token selection flags
	flatIdx     []int   // flat index backing store for lastIndices
	auxCountBuf []float32
	auxProbBuf  []float32
	// Z-loss buffers
	lastLogits *Tensor // pre-softmax gate logits for z-loss
	// Routing mode and BiasFree state
	routingMode      RoutingMode
	expertBias       []float32 // BiasFree expert bias (all 0 init)
	lastExpertCounts []float32 // BiasFree: expert assignment counts
	// ReLU mode state
	reluLambdaL1  float32 // ReMoE L1 regularization coefficient
	lastAvgActive float32 // ReMoE diagnostic: average active expert count
	lastReLUSum   float32 // ReMoE: sum of ReLU activations for L1 loss
}

// NewRouter creates a top-K expert router.
func NewRouter(hiddenDim, nExperts, topK int) *Router {
	if topK < 1 || topK > nExperts {
		panic("invalid topK for router")
	}
	return &Router{
		gate:             NewLinear(hiddenDim, nExperts, false),
		nExperts:         nExperts,
		topK:             topK,
		selected:         make([]bool, nExperts),
		routingMode:      TopKMode,
		expertBias:       make([]float32, nExperts), // all 0 init
		lastExpertCounts: make([]float32, nExperts),
		reluLambdaL1:     0.01,
	}
}

// SetRoutingMode configures the routing strategy and lambda for this router.
func (r *Router) SetRoutingMode(mode RoutingMode, lambda float32) {
	r.routingMode = mode
	r.reluLambdaL1 = lambda
}

// Forward computes expert selection for every token.
// Returns (weights [numTokens, topK], indices [numTokens][topK]).
func (r *Router) Forward(input *Tensor) (*Tensor, [][]int) {
	r.lastInput = input

	_, numTokens, featDim := splitLast(input.Shape().DimsRef())
	flatInput := input.Reshape(NewShape(numTokens, featDim))

	// Gate: linear projection
	gateLogits := r.gate.Forward(flatInput)
	r.lastLogits = gateLogits

	// Reset expert counts for all softmax-based routing modes
	if r.routingMode == TopKMode || r.routingMode == BiasFreeMode {
		for i := range r.lastExpertCounts {
			r.lastExpertCounts[i] = 0
		}
	}

	// Route based on mode
	switch r.routingMode {
	case TopKMode, BiasFreeMode:
		return r.forwardSoftmaxBased(gateLogits, numTokens)
	case ReLUMode:
		return r.forwardReLU(gateLogits, numTokens)
	default:
		panic("unknown routing mode")
	}
}

// forwardSoftmaxBased handles TopK and BiasFree routing (both use softmax).
func (r *Router) forwardSoftmaxBased(gateLogits *Tensor, numTokens int) (*Tensor, [][]int) {
	probShape := gateLogits.Shape()
	if r.softmaxBuf == nil || !r.softmaxBuf.Shape().Equal(probShape) {
		r.softmaxBuf = New(probShape, F32)
	}
	gateLogits.SoftmaxInto(r.softmaxBuf)
	r.lastGateProb = r.softmaxBuf
	probsData := r.softmaxBuf.DataPtr()

	weights := New(NewShape(numTokens, r.topK), F32)
	wData := weights.DataPtr()

	// Allocate flat index backing store
	totalIdx := numTokens * r.topK
	if cap(r.flatIdx) >= totalIdx {
		r.flatIdx = r.flatIdx[:totalIdx]
	} else {
		r.flatIdx = make([]int, totalIdx)
	}
	if cap(r.lastIndices) >= numTokens {
		r.lastIndices = r.lastIndices[:numTokens]
	} else {
		r.lastIndices = make([][]int, numTokens)
	}
	for t := 0; t < numTokens; t++ {
		r.lastIndices[t] = r.flatIdx[t*r.topK : (t+1)*r.topK]
	}

	selected := r.selected
	for t := 0; t < numTokens; t++ {
		row := probsData[t*r.nExperts : (t+1)*r.nExperts]
		indices := r.lastIndices[t]
		tokenWeights := wData[t*r.topK : (t+1)*r.topK]
		resetBools(selected)

		if r.routingMode == TopKMode {
			// TopK: select by probability directly
			for k := 0; k < r.topK; k++ {
				bestIdx, bestVal := -1, float32(-1)
				for e := 0; e < r.nExperts; e++ {
					if !selected[e] && row[e] > bestVal {
						bestVal = row[e]
						bestIdx = e
					}
				}
				selected[bestIdx] = true
				indices[k] = bestIdx
				tokenWeights[k] = bestVal
				r.lastExpertCounts[bestIdx]++
			}
		} else {
			// BiasFree: select by prob + bias, but weight by original prob
			for k := 0; k < r.topK; k++ {
				bestIdx, bestScore := -1, float32(-1e38)
				for e := 0; e < r.nExperts; e++ {
					if !selected[e] {
						score := row[e] + r.expertBias[e]
						if score > bestScore {
							bestScore = score
							bestIdx = e
						}
					}
				}
				selected[bestIdx] = true
				indices[k] = bestIdx
				tokenWeights[k] = row[bestIdx] // Use original prob, not augmented score
				r.lastExpertCounts[bestIdx]++
			}
		}

		// Renormalize so top-K weights sum to 1
		normalizeInPlace(tokenWeights)
	}

	r.lastWeights = weights
	return weights, r.lastIndices
}

// forwardReLU handles ReLU-based routing (ReMoE).
func (r *Router) forwardReLU(gateLogits *Tensor, numTokens int) (*Tensor, [][]int) {
	logitsData := gateLogits.DataPtr()

	weights := New(NewShape(numTokens, r.topK), F32)
	wData := weights.DataPtr()

	totalIdx := numTokens * r.topK
	if cap(r.flatIdx) >= totalIdx {
		r.flatIdx = r.flatIdx[:totalIdx]
	} else {
		r.flatIdx = make([]int, totalIdx)
	}
	if cap(r.lastIndices) >= numTokens {
		r.lastIndices = r.lastIndices[:numTokens]
	} else {
		r.lastIndices = make([][]int, numTokens)
	}
	for t := 0; t < numTokens; t++ {
		r.lastIndices[t] = r.flatIdx[t*r.topK : (t+1)*r.topK]
	}

	// Track statistics for ReLU mode
	totalActive := 0
	reluSum := float32(0)

	for t := 0; t < numTokens; t++ {
		row := logitsData[t*r.nExperts : (t+1)*r.nExperts]
		indices := r.lastIndices[t]
		tokenWeights := wData[t*r.topK : (t+1)*r.topK]

		// Apply ReLU and collect active experts
		type expertWeight struct {
			idx    int
			weight float32
		}
		active := make([]expertWeight, 0, r.nExperts)
		for e := 0; e < r.nExperts; e++ {
			w := row[e]
			if w > 0 {
				active = append(active, expertWeight{e, w})
				reluSum += w
			}
		}

		// If no active experts, force activate argmax with weight 1.0
		if len(active) == 0 {
			maxIdx, maxVal := 0, row[0]
			for e := 1; e < r.nExperts; e++ {
				if row[e] > maxVal {
					maxVal = row[e]
					maxIdx = e
				}
			}
			active = append(active, expertWeight{maxIdx, 1.0})
		}

		// If too many active, keep only top-k by weight
		if len(active) > r.topK {
			// Simple selection sort for top-k
			for k := 0; k < r.topK; k++ {
				maxPos := k
				for i := k + 1; i < len(active); i++ {
					if active[i].weight > active[maxPos].weight {
						maxPos = i
					}
				}
				active[k], active[maxPos] = active[maxPos], active[k]
			}
			active = active[:r.topK]
		}

		totalActive += len(active)

		// Renormalize active weights
		sum := float32(0)
		for _, ew := range active {
			sum += ew.weight
		}
		if sum < 1e-12 {
			sum = 1e-12
		}

		// Pad to topK with (expert=0, weight=0)
		for k := 0; k < r.topK; k++ {
			if k < len(active) {
				indices[k] = active[k].idx
				tokenWeights[k] = active[k].weight / sum
			} else {
				indices[k] = 0
				tokenWeights[k] = 0
			}
		}
	}

	r.lastAvgActive = float32(totalActive) / float32(numTokens)
	r.lastReLUSum = reluSum

	r.lastWeights = weights
	r.lastGateProb = nil // ReLU mode doesn't use softmax probs
	return weights, r.lastIndices
}

// Backward returns zeros (router gradients are not propagated in this implementation).
func (r *Router) Backward(gradOutput *Tensor) *Tensor {
	return Zeros(r.lastInput.Shape(), F32)
}

// Parameters returns the gate linear projection weight.
func (r *Router) Parameters() []*Tensor { return r.gate.Parameters() }

// ComputeAuxLoss computes the load-balancing auxiliary loss (Switch Transformer).
//
//	aux_loss = alpha * N_experts * sum_e(f_e * P_e)
//
// where:
//
//	f_e = fraction of tokens routed to expert e
//	P_e = mean gate probability for expert e
//
// This encourages uniform expert utilization. Multiplied by alpha and N_experts
// so the loss scale is independent of expert count.
func (r *Router) ComputeAuxLoss(alpha float32) float32 {
	if r.lastGateProb == nil {
		return 0
	}
	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastGateProb.Shape().At(0)

	if cap(r.auxCountBuf) >= r.nExperts {
		r.auxCountBuf = r.auxCountBuf[:r.nExperts]
		for i := range r.auxCountBuf {
			r.auxCountBuf[i] = 0
		}
	} else {
		r.auxCountBuf = make([]float32, r.nExperts)
	}
	if cap(r.auxProbBuf) >= r.nExperts {
		r.auxProbBuf = r.auxProbBuf[:r.nExperts]
		for i := range r.auxProbBuf {
			r.auxProbBuf[i] = 0
		}
	} else {
		r.auxProbBuf = make([]float32, r.nExperts)
	}
	expertCounts := r.auxCountBuf
	expertProbs := r.auxProbBuf

	for t := 0; t < numTokens; t++ {
		for k := 0; k < r.topK; k++ {
			expertCounts[r.lastIndices[t][k]] += 1
		}
		for e := 0; e < r.nExperts; e++ {
			expertProbs[e] += probsData[t*r.nExperts+e]
		}
	}

	totalAssign := float32(numTokens * r.topK)
	auxLoss := float32(0)
	for e := 0; e < r.nExperts; e++ {
		auxLoss += (expertCounts[e] / totalAssign) * (expertProbs[e] / float32(numTokens))
	}
	return auxLoss * alpha * float32(r.nExperts)
}

// UpdateExpertBias updates BiasFree expert biases based on load imbalance.
func (r *Router) UpdateExpertBias(gamma float32) {
	if r.routingMode != BiasFreeMode {
		return
	}

	totalAssignments := float32(0)
	for _, count := range r.lastExpertCounts {
		totalAssignments += count
	}
	if totalAssignments < 1 {
		return
	}

	target := 1.0 / float32(r.nExperts)
	for e := 0; e < r.nExperts; e++ {
		f_e := r.lastExpertCounts[e] / totalAssignments
		delta := target - f_e
		sign := float32(1)
		if delta < 0 {
			sign = -1
		}
		r.expertBias[e] += gamma * sign
	}
}

// ComputeReLUL1LossWithGrad computes ReLU L1 regularization loss and backprops gradients.
func (r *Router) ComputeReLUL1LossWithGrad() float32 {
	if r.routingMode != ReLUMode || r.lastLogits == nil {
		return 0
	}

	logitsData := r.lastLogits.DataPtr()
	numTokens := r.lastLogits.Shape().At(0)
	batchSeq := float32(numTokens)

	// L_l1 = lambda * mean_t(sum_e relu(logits[t,e]))
	// Already computed r.lastReLUSum in forwardReLU
	l1Loss := r.reluLambdaL1 * (r.lastReLUSum / batchSeq)

	// grad_logits[t,e] = lambda * (1/batchSeq) * (1 if logits[t,e] > 0 else 0)
	gradScale := r.reluLambdaL1 / batchSeq
	gradLogits := make([]float32, numTokens*r.nExperts)
	for t := 0; t < numTokens; t++ {
		for e := 0; e < r.nExperts; e++ {
			idx := t*r.nExperts + e
			if logitsData[idx] > 0 {
				gradLogits[idx] = gradScale
			}
		}
	}

	// Backprop to gate.weight: gate_weight_grad += grad_logits.T @ input
	if r.gate.lastInput != nil {
		hiddenDim := r.gate.weight.Shape().At(1)
		inputData := r.gate.lastInput.DataPtr()

		if r.gate.weight.Grad == nil {
			r.gate.weight.Grad = make([]float32, r.nExperts*hiddenDim)
		}
		gradWeight := r.gate.weight.Grad

		for t := 0; t < numTokens; t++ {
			for e := 0; e < r.nExperts; e++ {
				gradL := gradLogits[t*r.nExperts+e]
				for h := 0; h < hiddenDim; h++ {
					gradWeight[e*hiddenDim+h] += gradL * inputData[t*hiddenDim+h]
				}
			}
		}
	}

	return l1Loss
}

// ComputeZLossWithGrad computes router z-loss (ST-MoE) and backprops to gate weights.
//
//	L_z = z_weight * (1/B) * Σ_i logsumexp(logits_i)²
//
// Gradient: dL_z/d(logits[i,j]) = z_weight * (2/B) * lse_i * softmax(logits_i)_j
// Backprops to gate.weight via: gate_weight_grad += Σ_t grad_logits[t,e] * last_input[t,h]
func (r *Router) ComputeZLossWithGrad(zWeight float32) float32 {
	if r.lastLogits == nil || r.lastGateProb == nil {
		return 0
	}

	logitsData := r.lastLogits.DataPtr()
	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastLogits.Shape().At(0)

	zLossSum := float32(0)
	gradLogits := make([]float32, numTokens*r.nExperts)

	// Compute z-loss and gradients w.r.t. logits
	for t := 0; t < numTokens; t++ {
		offset := t * r.nExperts
		logitsRow := logitsData[offset : offset+r.nExperts]
		probsRow := probsData[offset : offset+r.nExperts]

		// logsumexp with max-subtract trick for numerical stability
		maxLogit := float32(-1e38)
		for _, l := range logitsRow {
			if l > maxLogit {
				maxLogit = l
			}
		}
		sumExp := float32(0)
		for _, l := range logitsRow {
			sumExp += ExpF32(l - maxLogit)
		}
		lse := maxLogit + LogF32(sumExp)

		zLossSum += lse * lse

		// Gradient: dL_z/d(logits[t,e]) = z_weight * (2/B) * lse * probs[t,e]
		gradScale := zWeight * (2.0 / float32(numTokens)) * lse
		for e := 0; e < r.nExperts; e++ {
			gradLogits[offset+e] = gradScale * probsRow[e]
		}
	}

	zLoss := zWeight * zLossSum / float32(numTokens)

	// Backprop grad_logits to gate.weight
	// gate.weight shape: [nExperts, hiddenDim]
	// grad_logits shape: [numTokens, nExperts]
	// lastInput shape: [numTokens, hiddenDim]
	if r.gate.lastInput != nil {
		hiddenDim := r.gate.weight.Shape().At(1)
		inputData := r.gate.lastInput.DataPtr()

		// Ensure gate.weight.Grad is allocated
		if r.gate.weight.Grad == nil {
			r.gate.weight.Grad = make([]float32, r.nExperts*hiddenDim)
		}
		gradWeight := r.gate.weight.Grad

		for t := 0; t < numTokens; t++ {
			for e := 0; e < r.nExperts; e++ {
				gradL := gradLogits[t*r.nExperts+e]
				for h := 0; h < hiddenDim; h++ {
					gradWeight[e*hiddenDim+h] += gradL * inputData[t*hiddenDim+h]
				}
			}
		}
	}

	return zLoss
}

// ComputeAuxLossWithGrad computes the load-balancing auxiliary loss and backprops to gate weights.
//
//	aux_loss = alpha * N_experts * sum_e(f_e * P_e)
//
// where:
//
//	f_e = fraction of tokens routed to expert e
//	P_e = mean gate probability for expert e
//
// Gradient: dL_aux/d(logits[t,e]) = alpha * N_experts / numTokens * probs[t,e] * (f[e] - dot_fp_t)
// where dot_fp_t = Σ_{e'} f[e'] * probs[t,e']
// Backprops to gate.weight via: gate_weight_grad += Σ_t grad_logits[t,e] * last_input[t,h]
func (r *Router) ComputeAuxLossWithGrad(alpha float32) float32 {
	if r.lastGateProb == nil {
		return 0
	}

	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastGateProb.Shape().At(0)

	// Compute f_e (fraction of tokens routed to each expert)
	totalAssign := float32(0)
	for _, count := range r.lastExpertCounts {
		totalAssign += count
	}
	if totalAssign < 1 {
		return 0
	}

	f_e := make([]float32, r.nExperts)
	for e := 0; e < r.nExperts; e++ {
		f_e[e] = r.lastExpertCounts[e] / totalAssign
	}

	// Compute P_e (mean gate probability per expert)
	P_e := make([]float32, r.nExperts)
	for t := 0; t < numTokens; t++ {
		for e := 0; e < r.nExperts; e++ {
			P_e[e] += probsData[t*r.nExperts+e]
		}
	}
	for e := 0; e < r.nExperts; e++ {
		P_e[e] /= float32(numTokens)
	}

	// Compute scalar aux loss
	auxLoss := float32(0)
	for e := 0; e < r.nExperts; e++ {
		auxLoss += f_e[e] * P_e[e]
	}
	auxLoss *= alpha * float32(r.nExperts)

	// Compute gradient w.r.t. logits
	gradLogits := make([]float32, numTokens*r.nExperts)
	for t := 0; t < numTokens; t++ {
		// dot_fp_t = Σ_{e'} f[e'] * probs[t,e']
		dot_fp_t := float32(0)
		for e := 0; e < r.nExperts; e++ {
			dot_fp_t += f_e[e] * probsData[t*r.nExperts+e]
		}

		// grad_logits[t,e] = alpha * nExperts / numTokens * probs[t,e] * (f[e] - dot_fp_t)
		gradScale := alpha * float32(r.nExperts) / float32(numTokens)
		for e := 0; e < r.nExperts; e++ {
			gradLogits[t*r.nExperts+e] = gradScale * probsData[t*r.nExperts+e] * (f_e[e] - dot_fp_t)
		}
	}

	// Backprop to gate.weight: gate_weight_grad += grad_logits.T @ lastInput
	if r.gate.lastInput != nil {
		hiddenDim := r.gate.weight.Shape().At(1)
		inputData := r.gate.lastInput.DataPtr()

		// Ensure gate.weight.Grad is allocated
		if r.gate.weight.Grad == nil {
			r.gate.weight.Grad = make([]float32, r.nExperts*hiddenDim)
		}
		gradWeight := r.gate.weight.Grad

		for t := 0; t < numTokens; t++ {
			for e := 0; e < r.nExperts; e++ {
				gradL := gradLogits[t*r.nExperts+e]
				for h := 0; h < hiddenDim; h++ {
					gradWeight[e*hiddenDim+h] += gradL * inputData[t*hiddenDim+h]
				}
			}
		}
	}

	return auxLoss
}

// ---------------------------------------------------------------------------
// MoELayer (Mixture of Experts)
// ---------------------------------------------------------------------------

// MoELayer implements token-level Mixture of Experts.
//
//	output = sum_k(weight_k * Expert_k(x))   for top-k selected experts
//
// Each expert is an independent SwiGLU FFN. The router selects which experts
// process each token; only the top-K experts are evaluated (sparse activation).
type MoELayer struct {
	router    *Router
	experts   []*SwiGLU
	hiddenDim int
	nExperts  int
	topK      int
	outBuf    []float32 // reusable output buffer for forward pass
	gradInBuf []float32 // reusable input-gradient buffer for backward
	// Per-expert reusable buffers. Each expert keeps its own storage because
	// expert.Forward caches input tensors for backward.
	expertBatchBuf [][]float32
	expertGradBuf  [][]float32
	// Cached from forward for backward
	lastWeights         []float32 // [numTokens * topK]
	lastIndices         [][]int
	lastLeadingDims     []int
	lastNumTokens       int
	lastExpertTokens    [][]int
	lastExpertWeightIdx [][]int
}

// NewMoELayer creates a MoE layer with nExperts independent SwiGLU experts.
func NewMoELayer(hiddenDim, ffnDim, nExperts, topK int) *MoELayer {
	experts := make([]*SwiGLU, nExperts)
	for i := range experts {
		experts[i] = NewSwiGLU(hiddenDim, ffnDim)
	}
	return &MoELayer{
		router:         NewRouter(hiddenDim, nExperts, topK),
		experts:        experts,
		hiddenDim:      hiddenDim,
		nExperts:       nExperts,
		topK:           topK,
		expertBatchBuf: make([][]float32, nExperts),
		expertGradBuf:  make([][]float32, nExperts),
	}
}

// Forward routes each token to its top-K experts and combines their outputs.
//
// Implementation:
//  1. Router selects top-K experts per token with normalized weights
//  2. Tokens are grouped by expert (inverted index: expert -> token list)
//  3. Each expert processes its batch of assigned tokens
//  4. Results are scattered back and weighted-summed into the output
//
// The expert grouping avoids redundant computation: each expert runs a single
// batched forward pass over all tokens assigned to it.
func (m *MoELayer) Forward(input *Tensor) *Tensor {
	leadingDims, numTokens, _ := splitLast(input.Shape().DimsRef())

	weights, indices := m.router.Forward(input)
	flatInput := input.Reshape(NewShape(numTokens, m.hiddenDim))
	flatData := flatInput.DataPtr()

	// Cache for backward (referencing forward-time buffers; valid until next forward).
	m.lastWeights = weights.DataPtr()
	m.lastIndices = indices
	m.lastLeadingDims = cloneInts(leadingDims)
	m.lastNumTokens = numTokens

	// Reuse output buffer to reduce GC pressure across forward calls.
	outLen := numTokens * m.hiddenDim
	if cap(m.outBuf) >= outLen {
		m.outBuf = m.outBuf[:outLen]
		for i := range m.outBuf {
			m.outBuf[i] = 0
		}
	} else {
		m.outBuf = make([]float32, outLen)
	}
	output := FromSliceNoCopy(m.outBuf, NewShape(numTokens, m.hiddenDim))
	outData := output.DataPtr()
	wData := weights.DataPtr()

	// Build inverted index: expert_id -> list of (token_index, weight_slot)
	// Pre-allocate with estimated capacity to avoid repeated grow+copy.
	avgTokensPerExpert := (numTokens*m.topK)/m.nExperts + 1
	if len(m.lastExpertTokens) != m.nExperts {
		m.lastExpertTokens = make([][]int, m.nExperts)
	}
	if len(m.lastExpertWeightIdx) != m.nExperts {
		m.lastExpertWeightIdx = make([][]int, m.nExperts)
	}
	expertTokens := m.lastExpertTokens
	expertWeightIdx := m.lastExpertWeightIdx
	for i := 0; i < m.nExperts; i++ {
		if cap(expertTokens[i]) >= avgTokensPerExpert {
			expertTokens[i] = expertTokens[i][:0]
		} else {
			expertTokens[i] = make([]int, 0, avgTokensPerExpert)
		}
		if cap(expertWeightIdx[i]) >= avgTokensPerExpert {
			expertWeightIdx[i] = expertWeightIdx[i][:0]
		} else {
			expertWeightIdx[i] = make([]int, 0, avgTokensPerExpert)
		}
	}
	for t := 0; t < numTokens; t++ {
		for k := 0; k < m.topK; k++ {
			eIdx := indices[t][k]
			expertTokens[eIdx] = append(expertTokens[eIdx], t)
			expertWeightIdx[eIdx] = append(expertWeightIdx[eIdx], k)
		}
	}
	m.lastExpertTokens = expertTokens
	m.lastExpertWeightIdx = expertWeightIdx

	// Process each expert's assigned tokens as a single batch.
	for eIdx := 0; eIdx < m.nExperts; eIdx++ {
		tokens := expertTokens[eIdx]
		if len(tokens) == 0 {
			continue
		}

		// Gather: collect assigned token vectors into a contiguous batch.
		need := len(tokens) * m.hiddenDim
		batchData := m.expertBatchBuf[eIdx]
		if cap(batchData) >= need {
			batchData = batchData[:need]
		} else {
			batchData = make([]float32, need)
		}
		m.expertBatchBuf[eIdx] = batchData
		for i, t := range tokens {
			copy(batchData[i*m.hiddenDim:], flatData[t*m.hiddenDim:(t+1)*m.hiddenDim])
		}
		batchInput := FromSliceNoCopy(batchData, NewShape(len(tokens), m.hiddenDim))
		expertOut := m.experts[eIdx].Forward(batchInput)
		eOutData := expertOut.DataPtr()

		// Scatter-add: weighted expert output back to each token's position
		for i, t := range tokens {
			k := expertWeightIdx[eIdx][i]
			w := wData[t*m.topK+k]
			tOff := t * m.hiddenDim
			eOff := i * m.hiddenDim
			oRow := outData[tOff : tOff+m.hiddenDim]
			eRow := eOutData[eOff : eOff+m.hiddenDim]
			for d := range oRow {
				oRow[d] += w * eRow[d]
			}
		}
	}

	return output.Reshape(withLastDim(leadingDims, m.hiddenDim))
}

// Backward propagates gradients through the MoE layer.
// For each expert, computes weighted gradients for its assigned tokens
// and accumulates expert weight gradients via SwiGLU.Backward.
func (m *MoELayer) Backward(gradOutput *Tensor) *Tensor {
	numTokens := m.lastNumTokens
	_, _, _ = splitLast(gradOutput.Shape().DimsRef())
	flatGrad := gradOutput.Reshape(NewShape(numTokens, m.hiddenDim)).DataPtr()

	gradLen := numTokens * m.hiddenDim
	if cap(m.gradInBuf) >= gradLen {
		m.gradInBuf = m.gradInBuf[:gradLen]
		for i := range m.gradInBuf {
			m.gradInBuf[i] = 0
		}
	} else {
		m.gradInBuf = make([]float32, gradLen)
	}
	gradInput := m.gradInBuf
	expertTokens := m.lastExpertTokens
	expertWeightIdx := m.lastExpertWeightIdx
	for eIdx := 0; eIdx < m.nExperts; eIdx++ {
		tokens := expertTokens[eIdx]
		if len(tokens) == 0 {
			continue
		}

		// Compute weighted grad for this expert: flat_grad[token] * weight
		need := len(tokens) * m.hiddenDim
		expertGradData := m.expertGradBuf[eIdx]
		if cap(expertGradData) >= need {
			expertGradData = expertGradData[:need]
		} else {
			expertGradData = make([]float32, need)
		}
		m.expertGradBuf[eIdx] = expertGradData
		for i, t := range tokens {
			k := expertWeightIdx[eIdx][i]
			w := m.lastWeights[t*m.topK+k]
			gOff := t * m.hiddenDim
			eOff := i * m.hiddenDim
			for d := 0; d < m.hiddenDim; d++ {
				expertGradData[eOff+d] = flatGrad[gOff+d] * w
			}
		}
		expertGrad := FromSliceNoCopy(expertGradData, NewShape(len(tokens), m.hiddenDim))

		// Backward through expert (accumulates weight gradients in SwiGLU sub-layers)
		// Expert forward already cached its own lastInput/lastGate/lastUp.
		gradExpertInput := m.experts[eIdx].Backward(expertGrad)
		geData := gradExpertInput.DataPtr()

		// Scatter-add expert input gradient back to token positions
		for i, t := range tokens {
			tOff := t * m.hiddenDim
			eOff := i * m.hiddenDim
			for d := 0; d < m.hiddenDim; d++ {
				gradInput[tOff+d] += geData[eOff+d]
			}
		}
	}

	return FromSliceNoCopy(gradInput, NewShape(numTokens, m.hiddenDim)).
		Reshape(withLastDim(m.lastLeadingDims, m.hiddenDim))
}

// Parameters returns all parameters: router gate + all expert weights.
func (m *MoELayer) Parameters() []*Tensor {
	p := append([]*Tensor(nil), m.router.Parameters()...)
	for _, e := range m.experts {
		p = append(p, e.Parameters()...)
	}
	return p
}

// AuxLoss returns the load-balancing auxiliary loss for this layer.
func (m *MoELayer) AuxLoss(alpha float32) float32 { return m.router.ComputeAuxLoss(alpha) }

// SetRoutingMode configures the routing strategy for this MoE layer.
func (m *MoELayer) SetRoutingMode(mode RoutingMode, lambda float32) {
	m.router.SetRoutingMode(mode, lambda)
}

// UpdateRoutingBiases updates BiasFree expert biases.
func (m *MoELayer) UpdateRoutingBiases(gamma float32) {
	m.router.UpdateExpertBias(gamma)
}

// ApplyReLUL1Loss computes and backprops ReLU L1 regularization loss.
func (m *MoELayer) ApplyReLUL1Loss() float32 {
	return m.router.ComputeReLUL1LossWithGrad()
}

// AvgActiveExperts returns the average number of active experts (ReLU mode).
func (m *MoELayer) AvgActiveExperts() float32 {
	return m.router.lastAvgActive
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

// TransformerBlock is a single Transformer layer with pre-norm residual connections:
//
//	x = x + Attention(RMSNorm(x))     -- self-attention with pre-norm
//	x = x + MoE(RMSNorm(x))           -- MoE feed-forward with pre-norm
type TransformerBlock struct {
	attnNorm  *RMSNorm
	attention *MQAttention
	ffnNorm   *RMSNorm
	moe       *MoELayer
	// Cached intermediate for backward residual
	lastH1 *Tensor // h1 = input + attnOut
}

// NewTransformerBlock creates a Transformer block from the model config.
func NewTransformerBlock(cfg Config) *TransformerBlock {
	return &TransformerBlock{
		attnNorm:  NewRMSNorm(cfg.HiddenDim, 1e-6),
		attention: NewMQAttention(cfg.HiddenDim, cfg.NHeads, cfg.NKVHeads, cfg.HeadDim, cfg.RoPEBase, cfg.RoPEAlpha),
		ffnNorm:   NewRMSNorm(cfg.HiddenDim, 1e-6),
		moe:       NewMoELayer(cfg.HiddenDim, cfg.FFNDim, cfg.NExperts, cfg.TopKExperts),
	}
}

// Forward applies the pre-norm Transformer block with residual connections.
//
//	h1 = input + Attention(RMSNorm(input))
//	output = h1 + MoE(RMSNorm(h1))
func (blk *TransformerBlock) Forward(input *Tensor) *Tensor {
	attnOut := blk.attention.Forward(blk.attnNorm.Forward(input))
	h1 := input.Add(attnOut)
	blk.lastH1 = h1 // cache for backward (not strictly needed but matches Python)
	moeOut := blk.moe.Forward(blk.ffnNorm.Forward(h1))
	return h1.Add(moeOut)
}

// Backward propagates gradients through the block with proper residual connections.
//
// Forward: h1 = x + attn(norm1(x)), out = h1 + moe(norm2(h1))
// Backward:
//
//	d_moe_input = moe.Backward(gradOutput)
//	d_h1_from_moe = norm2.Backward(d_moe_input)
//	d_h1 = gradOutput + d_h1_from_moe          (residual 2)
//	d_attn_input = attn.Backward(d_h1)
//	d_x_from_attn = norm1.Backward(d_attn_input)
//	d_x = d_h1 + d_x_from_attn                 (residual 1)
func (blk *TransformerBlock) Backward(gradOutput *Tensor) *Tensor {
	// Backward through MoE path
	gradMoeInput := blk.moe.Backward(gradOutput)
	gradH1FromMoe := blk.ffnNorm.Backward(gradMoeInput)
	// Residual 2: gradOutput flows directly + through MoE path
	gradH1 := gradOutput.Add(gradH1FromMoe)

	// Backward through attention path
	gradAttnInput := blk.attention.Backward(gradH1)
	gradXFromAttn := blk.attnNorm.Backward(gradAttnInput)
	// Residual 1: gradH1 flows directly + through attention path
	gradX := gradH1.Add(gradXFromAttn)

	return gradX
}

// Parameters returns all parameters from norms, attention, and MoE.
func (blk *TransformerBlock) Parameters() []*Tensor {
	return concatParams(
		blk.attnNorm.Parameters(),
		blk.attention.Parameters(),
		blk.ffnNorm.Parameters(),
		blk.moe.Parameters(),
	)
}

// AuxLoss returns the MoE auxiliary loss for this block.
func (blk *TransformerBlock) AuxLoss(alpha float32) float32 { return blk.moe.AuxLoss(alpha) }

// SetRoutingMode configures the routing strategy for this block.
func (blk *TransformerBlock) SetRoutingMode(mode RoutingMode, lambda float32) {
	blk.moe.SetRoutingMode(mode, lambda)
}

// UpdateRoutingBiases updates BiasFree expert biases.
func (blk *TransformerBlock) UpdateRoutingBiases(gamma float32) {
	blk.moe.UpdateRoutingBiases(gamma)
}

// ApplyReLUL1Loss computes and backprops ReLU L1 regularization loss.
func (blk *TransformerBlock) ApplyReLUL1Loss() float32 {
	return blk.moe.ApplyReLUL1Loss()
}

// AvgActiveExperts returns the average number of active experts.
func (blk *TransformerBlock) AvgActiveExperts() float32 {
	return blk.moe.AvgActiveExperts()
}
