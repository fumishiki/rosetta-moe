// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package cpu

// Layer is the common interface for neural network layers with forward/backward
// passes and parameter access (for the optimizer).
type Layer interface {
	Forward(input *Tensor) *Tensor
	Backward(gradOutput *Tensor) *Tensor
	Parameters() []*Tensor
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

// Embedding is a lookup table: token ID -> dense vector.
type Embedding struct {
	Weight    *Tensor
	vocabSize int
	embedDim  int
	lastInput []int
}

// NewEmbedding creates an embedding table with Kaiming-style initialization.
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	std := SqrtF32(2.0 / float32(embedDim))
	return &Embedding{
		Weight:    RandnWithStd(NewShape(vocabSize, embedDim), F32, std),
		vocabSize: vocabSize,
		embedDim:  embedDim,
	}
}

// Forward looks up embeddings for each token ID in the input tensor.
func (e *Embedding) Forward(input *Tensor) *Tensor {
	dims := input.Shape().DimsRef()
	batch, seqLen := dims[0], dims[1]

	e.lastInput = make([]int, batch*seqLen)
	inputData := input.DataPtr()
	for i := range e.lastInput {
		e.lastInput[i] = int(inputData[i])
	}

	output := New(NewShape(batch, seqLen, e.embedDim), F32)
	out, w := output.DataPtr(), e.Weight.DataPtr()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			tid := e.lastInput[b*seqLen+s]
			if tid < 0 || tid >= e.vocabSize {
				panic("token ID out of range")
			}
			copy(out[(b*seqLen+s)*e.embedDim:], w[tid*e.embedDim:(tid+1)*e.embedDim])
		}
	}
	return output
}

// Backward accumulates weight gradients via scatter-add.
func (e *Embedding) Backward(gradOutput *Tensor) *Tensor {
	dims := gradOutput.Shape().DimsRef()
	batch, seqLen := dims[0], dims[1]
	gData := gradOutput.DataPtr()

	if e.Weight.Grad == nil {
		e.Weight.Grad = make([]float32, len(e.Weight.data))
	}
	wGrad := e.Weight.Grad
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			tid := e.lastInput[b*seqLen+s]
			gOff := (b*seqLen + s) * e.embedDim
			wOff := tid * e.embedDim
			for d := 0; d < e.embedDim; d++ {
				wGrad[wOff+d] += gData[gOff+d]
			}
		}
	}
	return Zeros(gradOutput.Shape(), F32)
}

// Parameters returns the embedding weight table.
func (e *Embedding) Parameters() []*Tensor { return []*Tensor{e.Weight} }

// VocabSize returns the vocabulary size.
func (e *Embedding) VocabSize() int { return e.vocabSize }

// EmbedDim returns the embedding dimension.
func (e *Embedding) EmbedDim() int { return e.embedDim }

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

// Linear computes y = x @ W^T + b (optional bias).
type Linear struct {
	Weight    *Tensor
	Bias      *Tensor
	inFeat    int
	outFeat   int
	useBias   bool
	LastInput *Tensor
}

// NewLinear creates a linear layer with Kaiming initialization.
func NewLinear(inFeatures, outFeatures int, useBias bool) *Linear {
	std := SqrtF32(2.0 / float32(inFeatures))
	l := &Linear{
		Weight:  RandnWithStd(NewShape(outFeatures, inFeatures), F32, std),
		inFeat:  inFeatures,
		outFeat: outFeatures,
		useBias: useBias,
	}
	if useBias {
		l.Bias = Zeros(NewShape(outFeatures), F32)
	}
	return l
}

// Forward computes y = x @ W^T (+ bias).
func (l *Linear) Forward(input *Tensor) *Tensor {
	l.LastInput = input
	batchDims, batchSize, _ := splitLast(input.Shape().DimsRef())
	flatInput := input.Reshape(NewShape(batchSize, l.inFeat))
	output := MatmulTransposedB(flatInput, l.Weight)

	if l.useBias {
		out, b := output.DataPtr(), l.Bias.DataPtr()
		for i := 0; i < batchSize; i++ {
			row := out[i*l.outFeat : (i+1)*l.outFeat]
			for j := range row {
				row[j] += b[j]
			}
		}
	}

	return output.Reshape(withLastDim(batchDims, l.outFeat))
}

// Backward computes dL/dx = dL/dy @ W and accumulates weight and bias gradients.
func (l *Linear) Backward(gradOutput *Tensor) *Tensor {
	if l.LastInput == nil {
		panic("backward called before forward")
	}
	inputShape := l.LastInput.Shape()
	_, batchSize, _ := splitLast(gradOutput.Shape().DimsRef())
	flatGrad := gradOutput.Reshape(NewShape(batchSize, l.outFeat))
	flatInput := l.LastInput.Reshape(NewShape(batchSize, l.inFeat))

	gradInput := Matmul(flatGrad, l.Weight)

	dW := make([]float32, l.outFeat*l.inFeat)
	fgData := flatGrad.DataPtr()
	fiData := flatInput.DataPtr()
	if batchSize > 0 && l.outFeat > 0 && l.inFeat > 0 {
		SgemmTransA(l.outFeat, l.inFeat, batchSize,
			1.0, fgData, l.outFeat,
			fiData, l.inFeat,
			0.0, dW, l.inFeat)
	}
	l.Weight.AccumulateGrad(dW)

	if l.useBias && l.Bias != nil {
		db := make([]float32, l.outFeat)
		for i := 0; i < batchSize; i++ {
			row := fgData[i*l.outFeat : (i+1)*l.outFeat]
			for j := range row {
				db[j] += row[j]
			}
		}
		l.Bias.AccumulateGrad(db)
	}

	return gradInput.Reshape(inputShape)
}

// Parameters returns the weight (and bias, if present).
func (l *Linear) Parameters() []*Tensor {
	if l.useBias {
		return []*Tensor{l.Weight, l.Bias}
	}
	return []*Tensor{l.Weight}
}

// InFeatures returns the input dimension.
func (l *Linear) InFeatures() int { return l.inFeat }

// OutFeatures returns the output dimension.
func (l *Linear) OutFeatures() int { return l.outFeat }

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

// RMSNorm implements Root Mean Square Layer Normalization.
type RMSNorm struct {
	Weight    *Tensor
	Eps       float32
	dim       int
	lastInput *Tensor
	lastRMS   []float32
}

// NewRMSNorm creates an RMSNorm layer with gamma initialized to 1.
func NewRMSNorm(dim int, eps float32) *RMSNorm {
	return &RMSNorm{
		Weight:  Ones(NewShape(dim), F32),
		Eps:     eps,
		dim:     dim,
		lastRMS: make([]float32, 0, 512),
	}
}

// Forward applies RMSNorm along the last dimension.
func (r *RMSNorm) Forward(input *Tensor) *Tensor {
	r.lastInput = input

	shape := input.Shape()
	numVectors := shape.Numel() / r.dim
	if cap(r.lastRMS) >= numVectors {
		r.lastRMS = r.lastRMS[:numVectors]
	} else {
		r.lastRMS = make([]float32, numVectors)
	}

	output := New(shape, F32)
	in, out, w := input.DataPtr(), output.DataPtr(), r.Weight.DataPtr()
	for v := 0; v < numVectors; v++ {
		off := v * r.dim
		row := in[off : off+r.dim]

		sumSq := float32(0)
		for _, x := range row {
			sumSq += x * x
		}

		rms := SqrtF32(sumSq/float32(r.dim) + r.Eps)
		r.lastRMS[v] = rms
		invRms := 1.0 / rms

		oRow := out[off : off+r.dim]
		for i := range oRow {
			oRow[i] = row[i] * invRms * w[i]
		}
	}
	return output
}

// Backward computes the input gradient for RMSNorm and accumulates weight gradient.
func (r *RMSNorm) Backward(gradOutput *Tensor) *Tensor {
	if r.lastInput == nil {
		panic("backward called before forward")
	}
	shape := gradOutput.Shape()
	numVectors := shape.Numel() / r.dim

	gradInput := New(shape, F32)
	gOut, gIn := gradOutput.DataPtr(), gradInput.DataPtr()
	in, w := r.lastInput.DataPtr(), r.Weight.DataPtr()

	dGamma := make([]float32, r.dim)

	for v := 0; v < numVectors; v++ {
		off := v * r.dim
		rms := r.lastRMS[v]
		rms3 := rms * rms * rms
		invRms := 1.0 / rms

		for i := 0; i < r.dim; i++ {
			dGamma[i] += gOut[off+i] * in[off+i] * invRms
		}

		dotSum := float32(0)
		for i := 0; i < r.dim; i++ {
			dotSum += gOut[off+i] * w[i] * in[off+i]
		}
		for i := 0; i < r.dim; i++ {
			gIn[off+i] = gOut[off+i]*w[i]/rms - in[off+i]*dotSum/(float32(r.dim)*rms3)
		}
	}

	r.Weight.AccumulateGrad(dGamma)
	return gradInput
}

// Parameters returns the learnable gamma scale vector.
func (r *RMSNorm) Parameters() []*Tensor { return []*Tensor{r.Weight} }

// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------

// SwiGLU implements the SwiGLU feed-forward network.
type SwiGLU struct {
	WGate, WUp, WDown *Linear
	hiddenDim, ffnDim int
	lastUp            *Tensor
	lastGatePreSiLU   []float32
}

// NewSwiGLU creates a SwiGLU FFN block.
func NewSwiGLU(hiddenDim, ffnDim int) *SwiGLU {
	return &SwiGLU{
		WGate:     NewLinear(hiddenDim, ffnDim, false),
		WUp:       NewLinear(hiddenDim, ffnDim, false),
		WDown:     NewLinear(ffnDim, hiddenDim, false),
		hiddenDim: hiddenDim,
		ffnDim:    ffnDim,
	}
}

// Forward computes SwiGLU(x) = W_down @ (SiLU(W_gate @ x) * W_up @ x).
func (s *SwiGLU) Forward(input *Tensor) *Tensor {
	gate := s.WGate.Forward(input)
	gateData := gate.DataPtr()
	if cap(s.lastGatePreSiLU) >= len(gateData) {
		s.lastGatePreSiLU = s.lastGatePreSiLU[:len(gateData)]
	} else {
		s.lastGatePreSiLU = make([]float32, len(gateData))
	}
	copy(s.lastGatePreSiLU, gateData)
	gate.SiLUInPlace()
	up := s.WUp.Forward(input)
	s.lastUp = up
	gate.MulInPlace(up)
	return s.WDown.Forward(gate)
}

// Backward propagates gradients through the SwiGLU block.
func (s *SwiGLU) Backward(gradOutput *Tensor) *Tensor {
	gradHidden := s.WDown.Backward(gradOutput)

	gradSiluGate := gradHidden.Mul(s.lastUp)
	gradUp := New(gradHidden.Shape(), F32)

	preSilu := s.lastGatePreSiLU
	gHidden := gradHidden.DataPtr()
	gSilu := gradSiluGate.DataPtr()
	gUp := gradUp.DataPtr()
	for i := range gSilu {
		z := preSilu[i]
		sig := 1.0 / (1.0 + ExpF32(-z))
		silu := z * sig
		dSilu := sig * (1.0 + z*(1.0-sig))
		gUp[i] = gHidden[i] * silu
		gSilu[i] *= dSilu
	}

	gradIn := s.WGate.Backward(gradSiluGate)
	gradIn.AddInPlace(s.WUp.Backward(gradUp))
	return gradIn
}

// Parameters returns all weights from gate, up, and down projections.
func (s *SwiGLU) Parameters() []*Tensor {
	return concatParams(
		s.WGate.Parameters(),
		s.WUp.Parameters(),
		s.WDown.Parameters(),
	)
}
