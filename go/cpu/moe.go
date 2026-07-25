// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package cpu

// RoutingMode defines the MoE routing strategy.
type RoutingMode int

const (
	TopKMode RoutingMode = iota
	BiasFreeMode
	ReLUMode
)

// Router selects the top-K experts for each token via a learned gate.
type Router struct {
	gate         *Linear
	nExperts     int
	topK         int
	lastInput    *Tensor
	lastWeights  *Tensor
	lastIndices  [][]int
	lastGateProb *Tensor
	softmaxBuf   *Tensor
	selected     []bool
	flatIdx      []int
	auxCountBuf  []float32
	auxProbBuf   []float32
	lastLogits   *Tensor
	routingMode      RoutingMode
	expertBias       []float32
	lastExpertCounts []float32
	reluLambdaL1     float32
	lastAvgActive    float32
	lastReLUSum      float32
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
		expertBias:       make([]float32, nExperts),
		lastExpertCounts: make([]float32, nExperts),
		reluLambdaL1:     0.01,
	}
}

// SetRoutingMode configures the routing strategy.
func (r *Router) SetRoutingMode(mode RoutingMode, lambda float32) {
	r.routingMode = mode
	r.reluLambdaL1 = lambda
}

// Forward computes expert selection for every token.
func (r *Router) Forward(input *Tensor) (*Tensor, [][]int) {
	r.lastInput = input
	_, numTokens, featDim := splitLast(input.Shape().DimsRef())
	flatInput := input.Reshape(NewShape(numTokens, featDim))
	gateLogits := r.gate.Forward(flatInput)
	r.lastLogits = gateLogits

	if r.routingMode == TopKMode || r.routingMode == BiasFreeMode {
		for i := range r.lastExpertCounts {
			r.lastExpertCounts[i] = 0
		}
	}

	switch r.routingMode {
	case TopKMode, BiasFreeMode:
		return r.forwardSoftmaxBased(gateLogits, numTokens)
	case ReLUMode:
		return r.forwardReLU(gateLogits, numTokens)
	default:
		panic("unknown routing mode")
	}
}

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
				tokenWeights[k] = row[bestIdx]
				r.lastExpertCounts[bestIdx]++
			}
		}

		normalizeInPlace(tokenWeights)
	}

	r.lastWeights = weights
	return weights, r.lastIndices
}

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

	totalActive := 0
	reluSum := float32(0)

	for t := 0; t < numTokens; t++ {
		row := logitsData[t*r.nExperts : (t+1)*r.nExperts]
		indices := r.lastIndices[t]
		tokenWeights := wData[t*r.topK : (t+1)*r.topK]

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

		if len(active) > r.topK {
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

		sum := float32(0)
		for _, ew := range active {
			sum += ew.weight
		}
		if sum < 1e-12 {
			sum = 1e-12
		}

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
	r.lastGateProb = nil
	return weights, r.lastIndices
}

// Backward returns zeros.
func (r *Router) Backward(gradOutput *Tensor) *Tensor {
	return Zeros(r.lastInput.Shape(), F32)
}

// Parameters returns the gate linear projection weight.
func (r *Router) Parameters() []*Tensor { return r.gate.Parameters() }

// ComputeAuxLoss computes the load-balancing auxiliary loss.
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

// UpdateExpertBias updates BiasFree expert biases.
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

// ComputeReLUL1LossWithGrad computes ReLU L1 regularization loss and backprops.
func (r *Router) ComputeReLUL1LossWithGrad() float32 {
	if r.routingMode != ReLUMode || r.lastLogits == nil {
		return 0
	}
	logitsData := r.lastLogits.DataPtr()
	numTokens := r.lastLogits.Shape().At(0)
	batchSeq := float32(numTokens)

	l1Loss := r.reluLambdaL1 * (r.lastReLUSum / batchSeq)

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

	if r.gate.LastInput != nil {
		hiddenDim := r.gate.Weight.Shape().At(1)
		inputData := r.gate.LastInput.DataPtr()
		if r.gate.Weight.Grad == nil {
			r.gate.Weight.Grad = make([]float32, r.nExperts*hiddenDim)
		}
		gradWeight := r.gate.Weight.Grad
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

// ComputeZLossWithGrad computes router z-loss and backprops.
func (r *Router) ComputeZLossWithGrad(zWeight float32) float32 {
	if r.lastLogits == nil || r.lastGateProb == nil {
		return 0
	}
	logitsData := r.lastLogits.DataPtr()
	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastLogits.Shape().At(0)

	zLossSum := float32(0)
	gradLogits := make([]float32, numTokens*r.nExperts)

	for t := 0; t < numTokens; t++ {
		offset := t * r.nExperts
		logitsRow := logitsData[offset : offset+r.nExperts]
		probsRow := probsData[offset : offset+r.nExperts]

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

		gradScale := zWeight * (2.0 / float32(numTokens)) * lse
		for e := 0; e < r.nExperts; e++ {
			gradLogits[offset+e] = gradScale * probsRow[e]
		}
	}

	zLoss := zWeight * zLossSum / float32(numTokens)

	if r.gate.LastInput != nil {
		hiddenDim := r.gate.Weight.Shape().At(1)
		inputData := r.gate.LastInput.DataPtr()
		if r.gate.Weight.Grad == nil {
			r.gate.Weight.Grad = make([]float32, r.nExperts*hiddenDim)
		}
		gradWeight := r.gate.Weight.Grad
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

// ComputeAuxLossWithGrad computes aux loss and backprops.
func (r *Router) ComputeAuxLossWithGrad(alpha float32) float32 {
	if r.lastGateProb == nil {
		return 0
	}
	probsData := r.lastGateProb.DataPtr()
	numTokens := r.lastGateProb.Shape().At(0)

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

	P_e := make([]float32, r.nExperts)
	for t := 0; t < numTokens; t++ {
		for e := 0; e < r.nExperts; e++ {
			P_e[e] += probsData[t*r.nExperts+e]
		}
	}
	for e := 0; e < r.nExperts; e++ {
		P_e[e] /= float32(numTokens)
	}

	auxLoss := float32(0)
	for e := 0; e < r.nExperts; e++ {
		auxLoss += f_e[e] * P_e[e]
	}
	auxLoss *= alpha * float32(r.nExperts)

	gradLogits := make([]float32, numTokens*r.nExperts)
	for t := 0; t < numTokens; t++ {
		dot_fp_t := float32(0)
		for e := 0; e < r.nExperts; e++ {
			dot_fp_t += f_e[e] * probsData[t*r.nExperts+e]
		}
		gradScale := alpha * float32(r.nExperts) / float32(numTokens)
		for e := 0; e < r.nExperts; e++ {
			gradLogits[t*r.nExperts+e] = gradScale * probsData[t*r.nExperts+e] * (f_e[e] - dot_fp_t)
		}
	}

	if r.gate.LastInput != nil {
		hiddenDim := r.gate.Weight.Shape().At(1)
		inputData := r.gate.LastInput.DataPtr()
		if r.gate.Weight.Grad == nil {
			r.gate.Weight.Grad = make([]float32, r.nExperts*hiddenDim)
		}
		gradWeight := r.gate.Weight.Grad
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
// MoELayer
// ---------------------------------------------------------------------------

// MoELayer implements token-level Mixture of Experts.
type MoELayer struct {
	Router    *Router
	Experts   []*SwiGLU
	hiddenDim int
	nExperts  int
	TopK      int
	outBuf    []float32
	gradInBuf []float32
	expertBatchBuf [][]float32
	expertGradBuf  [][]float32
	lastWeights         []float32
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
		Router:         NewRouter(hiddenDim, nExperts, topK),
		Experts:        experts,
		hiddenDim:      hiddenDim,
		nExperts:       nExperts,
		TopK:           topK,
		expertBatchBuf: make([][]float32, nExperts),
		expertGradBuf:  make([][]float32, nExperts),
	}
}

// Forward routes each token to its top-K experts and combines their outputs.
func (m *MoELayer) Forward(input *Tensor) *Tensor {
	leadingDims, numTokens, _ := splitLast(input.Shape().DimsRef())

	weights, indices := m.Router.Forward(input)
	flatInput := input.Reshape(NewShape(numTokens, m.hiddenDim))
	flatData := flatInput.DataPtr()

	m.lastWeights = weights.DataPtr()
	m.lastIndices = indices
	m.lastLeadingDims = cloneInts(leadingDims)
	m.lastNumTokens = numTokens

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

	avgTokensPerExpert := (numTokens*m.TopK)/m.nExperts + 1
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
		for k := 0; k < m.TopK; k++ {
			eIdx := indices[t][k]
			expertTokens[eIdx] = append(expertTokens[eIdx], t)
			expertWeightIdx[eIdx] = append(expertWeightIdx[eIdx], k)
		}
	}
	m.lastExpertTokens = expertTokens
	m.lastExpertWeightIdx = expertWeightIdx

	for eIdx := 0; eIdx < m.nExperts; eIdx++ {
		tokens := expertTokens[eIdx]
		if len(tokens) == 0 {
			continue
		}

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
		expertOut := m.Experts[eIdx].Forward(batchInput)
		eOutData := expertOut.DataPtr()

		for i, t := range tokens {
			k := expertWeightIdx[eIdx][i]
			w := wData[t*m.TopK+k]
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
			w := m.lastWeights[t*m.TopK+k]
			gOff := t * m.hiddenDim
			eOff := i * m.hiddenDim
			for d := 0; d < m.hiddenDim; d++ {
				expertGradData[eOff+d] = flatGrad[gOff+d] * w
			}
		}
		expertGrad := FromSliceNoCopy(expertGradData, NewShape(len(tokens), m.hiddenDim))

		gradExpertInput := m.Experts[eIdx].Backward(expertGrad)
		geData := gradExpertInput.DataPtr()

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
	p := append([]*Tensor(nil), m.Router.Parameters()...)
	for _, e := range m.Experts {
		p = append(p, e.Parameters()...)
	}
	return p
}

// AuxLoss returns the load-balancing auxiliary loss for this layer.
func (m *MoELayer) AuxLoss(alpha float32) float32 { return m.Router.ComputeAuxLoss(alpha) }

// SetRoutingMode configures the routing strategy for this MoE layer.
func (m *MoELayer) SetRoutingMode(mode RoutingMode, lambda float32) {
	m.Router.SetRoutingMode(mode, lambda)
}

// UpdateRoutingBiases updates BiasFree expert biases.
func (m *MoELayer) UpdateRoutingBiases(gamma float32) {
	m.Router.UpdateExpertBias(gamma)
}

// ApplyReLUL1Loss computes and backprops ReLU L1 regularization loss.
func (m *MoELayer) ApplyReLUL1Loss() float32 {
	return m.Router.ComputeReLUL1LossWithGrad()
}

// AvgActiveExperts returns the average number of active experts.
func (m *MoELayer) AvgActiveExperts() float32 {
	return m.Router.lastAvgActive
}

// ---------------------------------------------------------------------------
// TransformerBlock
// ---------------------------------------------------------------------------

// TransformerBlock is a single Transformer layer with pre-norm residual connections.
type TransformerBlock struct {
	AttnNorm  *RMSNorm
	Attention *MQAttention
	FfnNorm   *RMSNorm
	Moe       *MoELayer
	lastH1    *Tensor
}

// NewTransformerBlock creates a Transformer block from the model config.
func NewTransformerBlock(cfg Config) *TransformerBlock {
	return &TransformerBlock{
		AttnNorm:  NewRMSNorm(cfg.HiddenDim, 1e-6),
		Attention: NewMQAttention(cfg.HiddenDim, cfg.NHeads, cfg.NKVHeads, cfg.HeadDim, cfg.RoPEBase, cfg.RoPEAlpha),
		FfnNorm:   NewRMSNorm(cfg.HiddenDim, 1e-6),
		Moe:       NewMoELayer(cfg.HiddenDim, cfg.FFNDim, cfg.NExperts, cfg.TopKExperts),
	}
}

// Forward applies the pre-norm Transformer block with residual connections.
func (blk *TransformerBlock) Forward(input *Tensor) *Tensor {
	attnOut := blk.Attention.Forward(blk.AttnNorm.Forward(input))
	h1 := input.Add(attnOut)
	blk.lastH1 = h1
	moeOut := blk.Moe.Forward(blk.FfnNorm.Forward(h1))
	return h1.Add(moeOut)
}

// Backward propagates gradients through the block.
func (blk *TransformerBlock) Backward(gradOutput *Tensor) *Tensor {
	gradMoeInput := blk.Moe.Backward(gradOutput)
	gradH1FromMoe := blk.FfnNorm.Backward(gradMoeInput)
	gradH1 := gradOutput.Add(gradH1FromMoe)

	gradAttnInput := blk.Attention.Backward(gradH1)
	gradXFromAttn := blk.AttnNorm.Backward(gradAttnInput)
	gradX := gradH1.Add(gradXFromAttn)

	return gradX
}

// Parameters returns all parameters from norms, attention, and MoE.
func (blk *TransformerBlock) Parameters() []*Tensor {
	return concatParams(
		blk.AttnNorm.Parameters(),
		blk.Attention.Parameters(),
		blk.FfnNorm.Parameters(),
		blk.Moe.Parameters(),
	)
}

// AuxLoss returns the MoE auxiliary loss for this block.
func (blk *TransformerBlock) AuxLoss(alpha float32) float32 { return blk.Moe.AuxLoss(alpha) }

// SetRoutingMode configures the routing strategy.
func (blk *TransformerBlock) SetRoutingMode(mode RoutingMode, lambda float32) {
	blk.Moe.SetRoutingMode(mode, lambda)
}

// UpdateRoutingBiases updates BiasFree expert biases.
func (blk *TransformerBlock) UpdateRoutingBiases(gamma float32) {
	blk.Moe.UpdateRoutingBiases(gamma)
}

// ApplyReLUL1Loss computes and backprops ReLU L1 regularization loss.
func (blk *TransformerBlock) ApplyReLUL1Loss() float32 {
	return blk.Moe.ApplyReLUL1Loss()
}

// AvgActiveExperts returns the average number of active experts.
func (blk *TransformerBlock) AvgActiveExperts() float32 {
	return blk.Moe.AvgActiveExperts()
}
