// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// ---------------------------------------------------------------------------
// Router (Gating Network)
// ---------------------------------------------------------------------------

// Router selects the top-K experts for each token via a learned gate.
//
// Gating:
//   gate_probs = softmax(W_gate @ x)    -- probability over all experts
//   top_k_indices = argmax_k(gate_probs)
//   top_k_weights = normalize(gate_probs[top_k_indices])
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
	softmaxBuf *Tensor  // reusable softmax output
	selected   []bool   // reusable per-token selection flags
	flatIdx    []int    // flat index backing store for lastIndices
}

// NewRouter creates a top-K expert router.
func NewRouter(hiddenDim, nExperts, topK int) *Router {
	if topK < 1 || topK > nExperts {
		panic("invalid topK for router")
	}
	return &Router{
		gate:     NewLinear(hiddenDim, nExperts, false),
		nExperts: nExperts,
		topK:     topK,
		selected: make([]bool, nExperts),
	}
}

// Forward computes expert selection for every token.
// Returns (weights [numTokens, topK], indices [numTokens][topK]).
func (r *Router) Forward(input *Tensor) (*Tensor, [][]int) {
	r.lastInput = input

	_, numTokens, featDim := splitLast(input.Shape().DimsRef())
	flatInput := input.Reshape(NewShape(numTokens, featDim))

	// Gate: linear projection + softmax -> per-expert probabilities.
	// Reuse softmax output buffer to avoid allocation every call.
	gateLogits := r.gate.Forward(flatInput)
	probShape := gateLogits.Shape()
	if r.softmaxBuf == nil || !r.softmaxBuf.Shape().Equal(probShape) {
		r.softmaxBuf = New(probShape, F32)
	}
	gateLogits.SoftmaxInto(r.softmaxBuf)
	r.lastGateProb = r.softmaxBuf
	probsData := r.softmaxBuf.DataPtr()

	weights := New(NewShape(numTokens, r.topK), F32)
	wData := weights.DataPtr()

	// Allocate flat index backing store: one contiguous array for all tokens'
	// top-K indices, sliced into per-token views. This replaces numTokens
	// separate make([]int, topK) calls with a single allocation.
	totalIdx := numTokens * r.topK
	if cap(r.flatIdx) >= totalIdx {
		r.flatIdx = r.flatIdx[:totalIdx]
		for i := range r.flatIdx {
			r.flatIdx[i] = 0
		}
	} else {
		r.flatIdx = make([]int, totalIdx)
	}
	r.lastIndices = make([][]int, numTokens)
	for t := 0; t < numTokens; t++ {
		r.lastIndices[t] = r.flatIdx[t*r.topK : (t+1)*r.topK]
	}

	// Greedy top-K selection per token: pick the highest probability expert,
	// mark it selected, repeat K times. O(K*E) per token.
	selected := r.selected
	for t := 0; t < numTokens; t++ {
		row := probsData[t*r.nExperts : (t+1)*r.nExperts]
		indices := r.lastIndices[t]
		tokenWeights := wData[t*r.topK : (t+1)*r.topK]
		resetBools(selected)

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
		}
		// Renormalize so top-K weights sum to 1.
		normalizeInPlace(tokenWeights)
	}

	r.lastWeights = weights
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
//   aux_loss = alpha * N_experts * sum_e(f_e * P_e)
//
// where:
//   f_e = fraction of tokens routed to expert e
//   P_e = mean gate probability for expert e
//
// This encourages uniform expert utilization. Multiplied by alpha and N_experts
// so the loss scale is independent of expert count.
func (r *Router) ComputeAuxLoss(alpha float32) float32 {
	if r.lastGateProb == nil {
		return 0
	}
	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastGateProb.Shape().At(0)

	expertCounts := make([]float32, r.nExperts)
	expertProbs := make([]float32, r.nExperts)

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

// ---------------------------------------------------------------------------
// MoELayer (Mixture of Experts)
// ---------------------------------------------------------------------------

// MoELayer implements token-level Mixture of Experts.
//
//   output = sum_k(weight_k * Expert_k(x))   for top-k selected experts
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
	// Cached from forward for backward
	lastFlatX        []float32
	lastWeights      []float32 // [numTokens * topK]
	lastIndices      [][]int
	lastLeadingDims  []int
	lastNumTokens    int
}

// NewMoELayer creates a MoE layer with nExperts independent SwiGLU experts.
func NewMoELayer(hiddenDim, ffnDim, nExperts, topK int) *MoELayer {
	experts := make([]*SwiGLU, nExperts)
	for i := range experts {
		experts[i] = NewSwiGLU(hiddenDim, ffnDim)
	}
	return &MoELayer{
		router:    NewRouter(hiddenDim, nExperts, topK),
		experts:   experts,
		hiddenDim: hiddenDim,
		nExperts:  nExperts,
		topK:      topK,
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

	// Cache for backward
	m.lastFlatX = make([]float32, len(flatData))
	copy(m.lastFlatX, flatData)
	m.lastWeights = make([]float32, len(weights.DataPtr()))
	copy(m.lastWeights, weights.DataPtr())
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
	expertTokens := make([][]int, m.nExperts)
	expertWeightIdx := make([][]int, m.nExperts)
	for i := range expertTokens {
		expertTokens[i] = make([]int, 0, avgTokensPerExpert)
		expertWeightIdx[i] = make([]int, 0, avgTokensPerExpert)
	}
	for t := 0; t < numTokens; t++ {
		for k := 0; k < m.topK; k++ {
			eIdx := indices[t][k]
			expertTokens[eIdx] = append(expertTokens[eIdx], t)
			expertWeightIdx[eIdx] = append(expertWeightIdx[eIdx], k)
		}
	}

	// Process each expert's assigned tokens as a single batch.
	for eIdx := 0; eIdx < m.nExperts; eIdx++ {
		tokens := expertTokens[eIdx]
		if len(tokens) == 0 {
			continue
		}

		// Gather: collect assigned token vectors into a contiguous batch
		batchData := make([]float32, len(tokens)*m.hiddenDim)
		for i, t := range tokens {
			copy(batchData[i*m.hiddenDim:], flatData[t*m.hiddenDim:(t+1)*m.hiddenDim])
		}
		// Use FromSliceNoCopy: batchData is freshly allocated, no need to copy again.
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

	gradInput := make([]float32, numTokens*m.hiddenDim)

	// Rebuild inverted index (same as forward)
	avgTokensPerExpert := (numTokens*m.topK)/m.nExperts + 1
	expertTokens := make([][]int, m.nExperts)
	expertWeightIdx := make([][]int, m.nExperts)
	for i := range expertTokens {
		expertTokens[i] = make([]int, 0, avgTokensPerExpert)
		expertWeightIdx[i] = make([]int, 0, avgTokensPerExpert)
	}
	for t := 0; t < numTokens; t++ {
		for k := 0; k < m.topK; k++ {
			eIdx := m.lastIndices[t][k]
			expertTokens[eIdx] = append(expertTokens[eIdx], t)
			expertWeightIdx[eIdx] = append(expertWeightIdx[eIdx], k)
		}
	}

	for eIdx := 0; eIdx < m.nExperts; eIdx++ {
		tokens := expertTokens[eIdx]
		if len(tokens) == 0 {
			continue
		}

		// Compute weighted grad for this expert: flat_grad[token] * weight
		expertGradData := make([]float32, len(tokens)*m.hiddenDim)
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

		// Set expert's cached input for backward
		expertInputData := make([]float32, len(tokens)*m.hiddenDim)
		for i, t := range tokens {
			copy(expertInputData[i*m.hiddenDim:], m.lastFlatX[t*m.hiddenDim:(t+1)*m.hiddenDim])
		}
		expertInput := FromSliceNoCopy(expertInputData, NewShape(len(tokens), m.hiddenDim))

		// Restore expert's lastInput for backward pass
		m.experts[eIdx].lastInput = expertInput
		m.experts[eIdx].wGate.lastInput = expertInput
		m.experts[eIdx].wUp.lastInput = expertInput

		// Backward through expert (accumulates weight gradients in SwiGLU sub-layers)
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
