// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package cpu

// MQAttention implements Multi-Query Attention (MQA) with Rotary Position
// Embeddings (RoPE).
type MQAttention struct {
	WQ, WK, WV, WO            *Linear
	nHeads, nKVHeads, headDim int
	hiddenDim                 int
	scale                     float32
	freqs                     []float32
	scoresBuf                 []float32
	attnOutBuf                []float32
	gradQBuf                  []float32
	gradKBuf                  []float32
	gradVBuf                  []float32
	gradScoresBuf             []float32
	lastInput       *Tensor
	lastQ           *Tensor
	lastK           *Tensor
	lastV           *Tensor
	lastAttnWeights []float32
	lastBatch       int
	lastSeqLen      int
}

// NewMQAttention creates a Multi-Query Attention layer.
func NewMQAttention(hiddenDim, nHeads, nKVHeads, headDim int, ropeBase, ropeAlpha float32) *MQAttention {
	base := ropeBase
	if ropeAlpha > 1.0 {
		base = ropeBase * PowF32(ropeAlpha, float32(headDim)/float32(headDim-2))
	}
	freqs := make([]float32, headDim/2)
	for i := range freqs {
		freqs[i] = 1.0 / PowF32(base, float32(2*i)/float32(headDim))
	}

	return &MQAttention{
		WQ:     NewLinear(hiddenDim, nHeads*headDim, false),
		WK:     NewLinear(hiddenDim, nKVHeads*headDim, false),
		WV:     NewLinear(hiddenDim, nKVHeads*headDim, false),
		WO:     NewLinear(nHeads*headDim, hiddenDim, false),
		nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
		hiddenDim: hiddenDim,
		scale:     1.0 / SqrtF32(float32(headDim)),
		freqs:     freqs,
	}
}

// Forward computes Multi-Query Attention with causal masking.
func (a *MQAttention) Forward(input *Tensor) *Tensor {
	dims := input.Shape().DimsRef()
	batch, seqLen := dims[0], dims[1]
	a.lastInput = input
	a.lastBatch = batch
	a.lastSeqLen = seqLen

	q := a.WQ.Forward(input).Reshape(NewShape(batch, seqLen, a.nHeads, a.headDim))
	k := a.WK.Forward(input).Reshape(NewShape(batch, seqLen, a.nKVHeads, a.headDim))
	v := a.WV.Forward(input).Reshape(NewShape(batch, seqLen, a.nKVHeads, a.headDim))

	a.applyRoPE(q.DataPtr(), k.DataPtr(), batch, seqLen, a.nHeads, a.nKVHeads)

	a.lastQ = q
	a.lastK = k
	a.lastV = v

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

	attnWeightsLen := batch * a.nHeads * seqLen * seqLen
	if len(a.lastAttnWeights) < attnWeightsLen {
		a.lastAttnWeights = make([]float32, attnWeightsLen)
	} else {
		a.lastAttnWeights = a.lastAttnWeights[:attnWeightsLen]
		for i := range a.lastAttnWeights {
			a.lastAttnWeights[i] = 0
		}
	}

	scoresLen := seqLen * seqLen
	if cap(a.scoresBuf) >= scoresLen {
		a.scoresBuf = a.scoresBuf[:scoresLen]
	} else {
		a.scoresBuf = make([]float32, scoresLen)
	}
	scores := a.scoresBuf
	for b := 0; b < batch; b++ {
		for h := 0; h < a.nHeads; h++ {
			kvH := h % a.nKVHeads

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
				for ki := qi + 1; ki < seqLen; ki++ {
					sRow[ki] = NegInf
				}
			}

			for qi := 0; qi < seqLen; qi++ {
				softmaxInPlace(scores[qi*seqLen : qi*seqLen+qi+1])
				for ki := qi + 1; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = 0
				}
			}

			awOff := (b*a.nHeads + h) * seqLen * seqLen
			copy(a.lastAttnWeights[awOff:awOff+seqLen*seqLen], scores[:seqLen*seqLen])

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

	output = output.Reshape(NewShape(batch, seqLen, a.nHeads*a.headDim))
	return a.WO.Forward(output)
}

// Backward computes the full attention backward pass.
func (a *MQAttention) Backward(gradOutput *Tensor) *Tensor {
	batch, seqLen := a.lastBatch, a.lastSeqLen

	gradOInput := a.WO.Backward(gradOutput)
	goData := gradOInput.DataPtr()

	qData := a.lastQ.DataPtr()
	kData := a.lastK.DataPtr()
	vData := a.lastV.DataPtr()

	gradQLen := batch * seqLen * a.nHeads * a.headDim
	if cap(a.gradQBuf) >= gradQLen {
		a.gradQBuf = a.gradQBuf[:gradQLen]
	} else {
		a.gradQBuf = make([]float32, gradQLen)
	}
	gradQ := a.gradQBuf

	gradKVLen := batch * seqLen * a.nKVHeads * a.headDim
	if cap(a.gradKBuf) >= gradKVLen {
		a.gradKBuf = a.gradKBuf[:gradKVLen]
		for i := range a.gradKBuf {
			a.gradKBuf[i] = 0
		}
	} else {
		a.gradKBuf = make([]float32, gradKVLen)
	}
	gradK := a.gradKBuf
	if cap(a.gradVBuf) >= gradKVLen {
		a.gradVBuf = a.gradVBuf[:gradKVLen]
		for i := range a.gradVBuf {
			a.gradVBuf[i] = 0
		}
	} else {
		a.gradVBuf = make([]float32, gradKVLen)
	}
	gradV := a.gradVBuf

	gradScoresLen := seqLen * seqLen
	if cap(a.gradScoresBuf) >= gradScoresLen {
		a.gradScoresBuf = a.gradScoresBuf[:gradScoresLen]
	} else {
		a.gradScoresBuf = make([]float32, gradScoresLen)
	}
	gradScores := a.gradScoresBuf

	hd := a.headDim
	qStride := a.nHeads * hd
	kvStride := a.nKVHeads * hd

	for b := 0; b < batch; b++ {
		for h := 0; h < a.nHeads; h++ {
			kvH := h % a.nKVHeads
			awOff := (b*a.nHeads + h) * seqLen * seqLen

			qBase := b*seqLen*qStride + h*hd
			kvBase := b*seqLen*kvStride + kvH*hd
			goBase := qBase
			gqBase := qBase
			gkBase := kvBase
			gvBase := kvBase

			SgemmRaw(true, false,
				seqLen, hd, seqLen,
				1.0,
				a.lastAttnWeights[awOff:], seqLen,
				goData[goBase:], qStride,
				1.0,
				gradV[gvBase:], kvStride)

			SgemmRaw(false, true,
				seqLen, seqLen, hd,
				1.0,
				goData[goBase:], qStride,
				vData[kvBase:], kvStride,
				0.0,
				gradScores, seqLen)

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
				for ki := qi + 1; ki < seqLen; ki++ {
					gradScores[row+ki] = 0
				}
			}

			SgemmRaw(false, false,
				seqLen, hd, seqLen,
				a.scale,
				gradScores, seqLen,
				kData[kvBase:], kvStride,
				0.0,
				gradQ[gqBase:], qStride)

			SgemmRaw(true, false,
				seqLen, hd, seqLen,
				a.scale,
				gradScores, seqLen,
				qData[qBase:], qStride,
				1.0,
				gradK[gkBase:], kvStride)
		}
	}

	gradQTensor := FromSliceNoCopy(gradQ, NewShape(batch, seqLen, a.nHeads*a.headDim))
	gradKTensor := FromSliceNoCopy(gradK, NewShape(batch, seqLen, a.nKVHeads*a.headDim))
	gradVTensor := FromSliceNoCopy(gradV, NewShape(batch, seqLen, a.nKVHeads*a.headDim))

	a.WQ.LastInput = a.lastInput
	a.WK.LastInput = a.lastInput
	a.WV.LastInput = a.lastInput

	gradXQ := a.WQ.Backward(gradQTensor)
	gradXK := a.WK.Backward(gradKTensor)
	gradXV := a.WV.Backward(gradVTensor)

	gradXQ.AddInPlace(gradXK)
	gradXQ.AddInPlace(gradXV)
	return gradXQ
}

// Parameters returns all projection weights: Q, K, V, and O.
func (a *MQAttention) Parameters() []*Tensor {
	return concatParams(
		a.WQ.Parameters(),
		a.WK.Parameters(),
		a.WV.Parameters(),
		a.WO.Parameters(),
	)
}

// applyRoPE applies Rotary Position Embeddings in-place to Q and K tensors.
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
