// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// MQAttention implements Multi-Query Attention (MQA) with Rotary Position
// Embeddings (RoPE). In MQA, query heads outnumber key/value heads
// (nHeads > nKVHeads); each KV head is shared by (nHeads / nKVHeads) query heads,
// reducing KV cache memory in inference.
//
// Full attention equation:
//   scores = (Q @ K^T) / sqrt(d_k)
//   weights = softmax(scores + causal_mask)
//   output = weights @ V
//
// RoPE applies position-dependent rotation to Q and K before the dot product:
//   [x0, x1] -> [x0*cos(theta) - x1*sin(theta), x0*sin(theta) + x1*cos(theta)]
//   where theta_i = pos / base^(2i/d_head)
type MQAttention struct {
	wQ, wK, wV, wO            *Linear
	nHeads, nKVHeads, headDim int
	hiddenDim                 int
	scale                     float32   // 1 / sqrt(head_dim)
	freqs                     []float32 // precomputed RoPE frequency bands
	scoresBuf                 []float32 // reusable buffer for attention scores
	attnOutBuf                []float32 // reusable buffer for attention output
	// Cached from forward pass for backward
	lastInput       *Tensor     // input to attention
	lastQ           *Tensor     // Q after RoPE [batch, seq, nHeads, headDim]
	lastK           *Tensor     // K after RoPE [batch, seq, nKVHeads, headDim]
	lastV           *Tensor     // V [batch, seq, nKVHeads, headDim]
	lastAttnWeights []float32   // softmax weights [batch * nHeads * seq * seq]
	lastBatch       int
	lastSeqLen      int
}

// NewMQAttention creates a Multi-Query Attention layer.
//
// RoPE frequency computation:
//   base' = base * alpha^(d_head / (d_head - 2))   (NTK-aware scaling if alpha > 1)
//   freq_i = 1 / base'^(2i / d_head)               for i in [0, d_head/2)
func NewMQAttention(hiddenDim, nHeads, nKVHeads, headDim int, ropeBase, ropeAlpha float32) *MQAttention {
	base := ropeBase
	if ropeAlpha > 1.0 {
		// NTK-aware RoPE scaling: increase the base to extend context length
		// without fine-tuning. The exponent d/(d-2) is the NTK-aware formula.
		base = ropeBase * PowF32(ropeAlpha, float32(headDim)/float32(headDim-2))
	}
	freqs := make([]float32, headDim/2)
	for i := range freqs {
		freqs[i] = 1.0 / PowF32(base, float32(2*i)/float32(headDim))
	}

	return &MQAttention{
		wQ:     NewLinear(hiddenDim, nHeads*headDim, false),
		wK:     NewLinear(hiddenDim, nKVHeads*headDim, false),
		wV:     NewLinear(hiddenDim, nKVHeads*headDim, false),
		wO:     NewLinear(nHeads*headDim, hiddenDim, false),
		nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
		hiddenDim: hiddenDim,
		scale:     1.0 / SqrtF32(float32(headDim)),
		freqs:     freqs,
	}
}

// Forward computes Multi-Query Attention with causal masking.
//
// Steps:
//  1. Project: Q = W_Q @ x, K = W_K @ x, V = W_V @ x
//  2. Reshape to [batch, seq, heads, head_dim]
//  3. Apply RoPE to Q and K
//  4. Compute scores = Q @ K^T / sqrt(d_k) with causal mask (upper triangle = -inf)
//  5. Softmax over valid positions (up to and including current position)
//  6. Weighted sum: output = softmax_weights @ V
//  7. Reshape and project output through W_O
//
// The causal mask is implemented by only iterating ki <= qi in the score loop,
// filling future positions with NegInf before softmax.
func (a *MQAttention) Forward(input *Tensor) *Tensor {
	dims := input.Shape().DimsRef()
	batch, seqLen := dims[0], dims[1]
	a.lastInput = input
	a.lastBatch = batch
	a.lastSeqLen = seqLen

	q := a.wQ.Forward(input).Reshape(NewShape(batch, seqLen, a.nHeads, a.headDim))
	k := a.wK.Forward(input).Reshape(NewShape(batch, seqLen, a.nKVHeads, a.headDim))
	v := a.wV.Forward(input).Reshape(NewShape(batch, seqLen, a.nKVHeads, a.headDim))

	a.applyRoPE(q.DataPtr(), k.DataPtr(), batch, seqLen, a.nHeads, a.nKVHeads)

	// Cache Q, K, V after RoPE for backward pass
	a.lastQ = q.Clone()
	a.lastK = k.Clone()
	a.lastV = v.Clone()

	// Reuse attention output buffer to avoid allocation per forward pass.
	outLen := batch * seqLen * a.nHeads * a.headDim
	if cap(a.attnOutBuf) >= outLen {
		a.attnOutBuf = a.attnOutBuf[:outLen]
		for i := range a.attnOutBuf {
			a.attnOutBuf[i] = 0
		}
	} else {
		a.attnOutBuf = make([]float32, outLen)
	}
	output := FromSliceNoCopy(a.attnOutBuf, NewShape(batch, seqLen, a.nHeads, a.headDim))
	outData, qData, kData, vData := output.DataPtr(), q.DataPtr(), k.DataPtr(), v.DataPtr()

	// Allocate attention weights storage: [batch * nHeads * seqLen * seqLen]
	attnWeightsLen := batch * a.nHeads * seqLen * seqLen
	if len(a.lastAttnWeights) < attnWeightsLen {
		a.lastAttnWeights = make([]float32, attnWeightsLen)
	} else {
		a.lastAttnWeights = a.lastAttnWeights[:attnWeightsLen]
		for i := range a.lastAttnWeights {
			a.lastAttnWeights[i] = 0
		}
	}

	// Reuse scores buffer to avoid allocation per forward pass.
	scoresLen := seqLen * seqLen
	if cap(a.scoresBuf) >= scoresLen {
		a.scoresBuf = a.scoresBuf[:scoresLen]
	} else {
		a.scoresBuf = make([]float32, scoresLen)
	}
	scores := a.scoresBuf
	for b := 0; b < batch; b++ {
		for h := 0; h < a.nHeads; h++ {
			// Multi-Query: map query head h to KV head (h % nKVHeads)
			kvH := h % a.nKVHeads

			// Compute Q @ K^T / sqrt(d_k) with causal mask.
			for qi := 0; qi < seqLen; qi++ {
				qOff := ((b*seqLen+qi)*a.nHeads + h) * a.headDim
				qRow := qData[qOff : qOff+a.headDim]
				sRow := scores[qi*seqLen : (qi+1)*seqLen]

				for ki := 0; ki <= qi; ki++ {
					kOff := ((b*seqLen+ki)*a.nKVHeads + kvH) * a.headDim
					kRow := kData[kOff : kOff+a.headDim]
					dot := float32(0)
					for d := range qRow {
						dot += qRow[d] * kRow[d]
					}
					sRow[ki] = dot * a.scale
				}
				// Causal mask: future positions get -inf so softmax zeroes them.
				for ki := qi + 1; ki < seqLen; ki++ {
					sRow[ki] = NegInf
				}
			}

			// Softmax each row up to the causal boundary.
			// Zero masked positions so backward BLAS sees 0, not -inf.
			for qi := 0; qi < seqLen; qi++ {
				softmaxInPlace(scores[qi*seqLen : qi*seqLen+qi+1])
				for ki := qi + 1; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = 0
				}
			}

			// Save attention weights for backward pass
			// Layout: [batch, nHeads, seqLen, seqLen]
			awOff := (b*a.nHeads + h) * seqLen * seqLen
			copy(a.lastAttnWeights[awOff:awOff+seqLen*seqLen], scores[:seqLen*seqLen])

			// Weighted sum: output = attention_weights @ V
			for qi := 0; qi < seqLen; qi++ {
				outOff := ((b*seqLen+qi)*a.nHeads + h) * a.headDim
				oRow := outData[outOff : outOff+a.headDim]
				for ki := 0; ki <= qi; ki++ {
					w := scores[qi*seqLen+ki]
					vOff := ((b*seqLen+ki)*a.nKVHeads + kvH) * a.headDim
					vRow := vData[vOff : vOff+a.headDim]
					for d := range oRow {
						oRow[d] += w * vRow[d]
					}
				}
			}
		}
	}

	// Concatenate heads: [batch, seq, nHeads, headDim] -> [batch, seq, nHeads*headDim]
	output = output.Reshape(NewShape(batch, seqLen, a.nHeads*a.headDim))
	return a.wO.Forward(output)
}

// Backward computes the full attention backward pass.
// Propagates gradients through: W_o -> attention (V, weights, softmax, scores) -> W_q, W_k, W_v.
//
// Uses per-head BLAS sgemm calls with strided access to [batch, seq, nHeads, headDim] tensors.
func (a *MQAttention) Backward(gradOutput *Tensor) *Tensor {
	batch, seqLen := a.lastBatch, a.lastSeqLen

	// 1. Backward through W_o: gradOInput shape [batch, seq, nHeads*headDim]
	gradOInput := a.wO.Backward(gradOutput)
	goData := gradOInput.DataPtr()

	// Q, K, V data from cached tensors [batch, seq, heads, headDim]
	qData := a.lastQ.DataPtr()
	kData := a.lastK.DataPtr()
	vData := a.lastV.DataPtr()

	// Allocate gradient tensors for Q, K, V (zero-initialized)
	gradQ := make([]float32, batch*seqLen*a.nHeads*a.headDim)
	gradK := make([]float32, batch*seqLen*a.nKVHeads*a.headDim)
	gradV := make([]float32, batch*seqLen*a.nKVHeads*a.headDim)

	// Scratch buffer for grad_scores per head [seqLen * seqLen]
	gradScores := make([]float32, seqLen*seqLen)

	hd := a.headDim
	qStride := a.nHeads * hd
	kvStride := a.nKVHeads * hd

	for b := 0; b < batch; b++ {
		for h := 0; h < a.nHeads; h++ {
			kvH := h % a.nKVHeads
			awOff := (b*a.nHeads + h) * seqLen * seqLen

			// Base offsets for this (batch, head) into the [batch, seq, heads, headDim] tensors
			qBase := b*seqLen*qStride + h*hd
			kvBase := b*seqLen*kvStride + kvH*hd
			goBase := qBase
			gqBase := qBase
			gkBase := kvBase
			gvBase := kvBase

			// 2. grad_V += W^T @ dO
			// W is [seqLen, seqLen], dO is [seqLen, headDim]
			// gradV[seqLen, headDim] += W^T[seqLen, seqLen] @ dO[seqLen, headDim]
			sgemmRaw(true, false,
				seqLen, hd, seqLen,
				1.0,
				a.lastAttnWeights[awOff:], seqLen,
				goData[goBase:], qStride,
				1.0, // beta=1.0: accumulate into gradV (multiple Q heads may share this KV head)
				gradV[gvBase:], kvStride)

			// 3. grad_W = dO @ V^T -> gradScores
			// dO[seqLen, headDim] @ V^T[headDim, seqLen] = gradScores[seqLen, seqLen]
			sgemmRaw(false, true,
				seqLen, seqLen, hd,
				1.0,
				goData[goBase:], qStride,
				vData[kvBase:], kvStride,
				0.0, // beta=0.0: overwrite gradScores
				gradScores, seqLen)

			// 4. Softmax backward (element-wise)
			// gradScores = weights * (gradScores - sum(gradScores * weights))
			for qi := 0; qi < seqLen; qi++ {
				row := qi * seqLen
				sumTerm := float32(0)
				for ki := 0; ki <= qi; ki++ {
					sumTerm += gradScores[row+ki] * a.lastAttnWeights[awOff+row+ki]
				}
				for ki := 0; ki <= qi; ki++ {
					w := a.lastAttnWeights[awOff+row+ki]
					gradScores[row+ki] = w * (gradScores[row+ki] - sumTerm)
				}
				// Zero future positions (sgemm wrote values there from the full V^T multiply)
				for ki := qi + 1; ki < seqLen; ki++ {
					gradScores[row+ki] = 0
				}
			}

			// 5. grad_Q = scale * grad_scores @ K
			// gradScores[seqLen, seqLen] @ K[seqLen, headDim] = gradQ[seqLen, headDim]
			sgemmRaw(false, false,
				seqLen, hd, seqLen,
				a.scale,
				gradScores, seqLen,
				kData[kvBase:], kvStride,
				0.0, // beta=0.0: each Q head is unique, no accumulation needed
				gradQ[gqBase:], qStride)

			// 6. grad_K += scale * grad_scores^T @ Q
			// gradScores^T[seqLen, seqLen] @ Q[seqLen, headDim] = gradK[seqLen, headDim]
			sgemmRaw(true, false,
				seqLen, hd, seqLen,
				a.scale,
				gradScores, seqLen,
				qData[qBase:], qStride,
				1.0, // beta=1.0: accumulate into gradK (multiple Q heads may share this KV head)
				gradK[gkBase:], kvStride)
		}
	}

	// 7. Backward through Q, K, V projection layers
	// Flatten gradQ from [batch, seq, nHeads, headDim] to [batch, seq, nHeads*headDim]
	gradQTensor := FromSliceNoCopy(gradQ, NewShape(batch, seqLen, a.nHeads*a.headDim))
	gradKTensor := FromSliceNoCopy(gradK, NewShape(batch, seqLen, a.nKVHeads*a.headDim))
	gradVTensor := FromSliceNoCopy(gradV, NewShape(batch, seqLen, a.nKVHeads*a.headDim))

	// Set lastInput for Q/K/V projections (they all used the same input x)
	a.wQ.lastInput = a.lastInput
	a.wK.lastInput = a.lastInput
	a.wV.lastInput = a.lastInput

	gradXQ := a.wQ.Backward(gradQTensor)
	gradXK := a.wK.Backward(gradKTensor)
	gradXV := a.wV.Backward(gradVTensor)

	// Sum gradients from all projection paths
	return gradXQ.Add(gradXK).Add(gradXV)
}

// Parameters returns all projection weights: Q, K, V, and O.
func (a *MQAttention) Parameters() []*Tensor {
	return concatParams(
		a.wQ.Parameters(),
		a.wK.Parameters(),
		a.wV.Parameters(),
		a.wO.Parameters(),
	)
}

// applyRoPE applies Rotary Position Embeddings in-place to Q and K tensors.
//
// RoPE rotates consecutive pairs of dimensions by a position-dependent angle:
//   theta_i = position * freq_i
//   [x_{2i}, x_{2i+1}] -> [x_{2i}*cos(theta) - x_{2i+1}*sin(theta),
//                           x_{2i}*sin(theta) + x_{2i+1}*cos(theta)]
//
// This encodes relative position information directly into the Q/K vectors
// so that the dot product Q_i . K_j depends on (i - j), not absolute positions.
func (a *MQAttention) applyRoPE(qData, kData []float32, batch, seqLen, qHeads, kHeads int) {
	halfDim := a.headDim / 2
	rotate := func(data []float32, heads int, b, s int) {
		base := (b*seqLen + s) * heads * a.headDim
		pos := float32(s)
		for h := 0; h < heads; h++ {
			off := base + h*a.headDim
			row := data[off : off+a.headDim]
			for i := 0; i < halfDim; i++ {
				angle := pos * a.freqs[i]
				cos, sin := CosF32(angle), SinF32(angle)
				x0, x1 := row[2*i], row[2*i+1]
				row[2*i] = x0*cos - x1*sin
				row[2*i+1] = x0*sin + x1*cos
			}
		}
	}

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			rotate(qData, qHeads, b, s)
			rotate(kData, kHeads, b, s)
		}
	}
}
