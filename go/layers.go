// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

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
//
//	output[b, s, :] = weight[token_ids[b, s], :]
//
// Weight shape: [vocab_size, embed_dim]. Initialized with N(0, sqrt(2/d)).
type Embedding struct {
	weight    *Tensor
	vocabSize int
	embedDim  int
	lastInput []int // cached token IDs for backward pass
}

// NewEmbedding creates an embedding table with Kaiming-style initialization.
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	std := SqrtF32(2.0 / float32(embedDim))
	return &Embedding{
		weight:    RandnWithStd(NewShape(vocabSize, embedDim), F32, std),
		vocabSize: vocabSize,
		embedDim:  embedDim,
	}
}

// Forward looks up embeddings for each token ID in the input tensor.
// Input: [batch, seq_len] of float32-encoded token IDs.
// Output: [batch, seq_len, embed_dim].
func (e *Embedding) Forward(input *Tensor) *Tensor {
	dims := input.Shape().DimsRef()
	batch, seqLen := dims[0], dims[1]

	e.lastInput = make([]int, batch*seqLen)
	inputData := input.DataPtr()
	for i := range e.lastInput {
		e.lastInput[i] = int(inputData[i])
	}

	output := New(NewShape(batch, seqLen, e.embedDim), F32)
	out, w := output.DataPtr(), e.weight.DataPtr()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			tid := e.lastInput[b*seqLen+s]
			if tid < 0 || tid >= e.vocabSize {
				panic("token ID out of range")
			}
			// Copy one embedding vector: flat offset = tid * embed_dim
			copy(out[(b*seqLen+s)*e.embedDim:], w[tid*e.embedDim:(tid+1)*e.embedDim])
		}
	}
	return output
}

// Backward accumulates weight gradients via scatter-add and returns zeros
// (no meaningful gradient w.r.t. discrete token IDs).
func (e *Embedding) Backward(gradOutput *Tensor) *Tensor {
	dims := gradOutput.Shape().DimsRef()
	batch, seqLen := dims[0], dims[1]
	gData := gradOutput.DataPtr()

	// Scatter-add: weight.Grad[tokenID] += gradOutput[b, s, :]
	if e.weight.Grad == nil {
		e.weight.Grad = make([]float32, len(e.weight.data))
	}
	wGrad := e.weight.Grad
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
func (e *Embedding) Parameters() []*Tensor { return []*Tensor{e.weight} }

// VocabSize returns the vocabulary size.
func (e *Embedding) VocabSize() int { return e.vocabSize }

// EmbedDim returns the embedding dimension.
func (e *Embedding) EmbedDim() int { return e.embedDim }

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

// Linear computes y = x @ W^T + b (optional bias).
//
// Weight shape: [out_features, in_features] (transposed layout so that
// MatmulTransposedB can be used, avoiding a separate transpose allocation).
type Linear struct {
	weight    *Tensor
	bias      *Tensor
	inFeat    int
	outFeat   int
	useBias   bool
	lastInput *Tensor // cached for backward pass
}

// NewLinear creates a linear layer with Kaiming initialization: N(0, sqrt(2/in)).
func NewLinear(inFeatures, outFeatures int, useBias bool) *Linear {
	std := SqrtF32(2.0 / float32(inFeatures))
	l := &Linear{
		weight:  RandnWithStd(NewShape(outFeatures, inFeatures), F32, std),
		inFeat:  inFeatures,
		outFeat: outFeatures,
		useBias: useBias,
	}
	if useBias {
		l.bias = Zeros(NewShape(outFeatures), F32)
	}
	return l
}

// Forward computes y = x @ W^T (+ bias). Input may be any shape where the
// last dim is in_features; leading dims are treated as a flat batch.
//
// The leading dims are peeled off, matmul runs on [batch, in_features],
// then the output is reshaped back to [...leading, out_features].
func (l *Linear) Forward(input *Tensor) *Tensor {
	l.lastInput = input
	batchDims, batchSize, _ := splitLast(input.Shape().DimsRef())
	flatInput := input.Reshape(NewShape(batchSize, l.inFeat))
	// Uses CblasTrans on weight to avoid materializing W^T.
	output := MatmulTransposedB(flatInput, l.weight)

	if l.useBias {
		out, b := output.DataPtr(), l.bias.DataPtr()
		for i := 0; i < batchSize; i++ {
			row := out[i*l.outFeat : (i+1)*l.outFeat]
			for j := range row {
				row[j] += b[j]
			}
		}
	}

	return output.Reshape(withLastDim(batchDims, l.outFeat))
}

// Backward computes dL/dx = dL/dy @ W (the input gradient) and accumulates
// weight and bias gradients: dW = gradOutput^T @ input, db = sum(gradOutput).
func (l *Linear) Backward(gradOutput *Tensor) *Tensor {
	if l.lastInput == nil {
		panic("backward called before forward")
	}
	inputShape := l.lastInput.Shape()
	_, batchSize, _ := splitLast(gradOutput.Shape().DimsRef())
	flatGrad := gradOutput.Reshape(NewShape(batchSize, l.outFeat))
	flatInput := l.lastInput.Reshape(NewShape(batchSize, l.inFeat))

	// dX = gradOutput @ W -> [batchSize, inFeat]
	gradInput := Matmul(flatGrad, l.weight)

	// dW = gradOutput^T @ input -> [outFeat, inFeat]
	// Uses BLAS: C = A^T @ B where A=[batchSize, outFeat], B=[batchSize, inFeat]
	dW := make([]float32, l.outFeat*l.inFeat)
	fgData := flatGrad.DataPtr()
	fiData := flatInput.DataPtr()
	if batchSize > 0 && l.outFeat > 0 && l.inFeat > 0 {
		sgemmTransA(l.outFeat, l.inFeat, batchSize,
			1.0, fgData, l.outFeat,
			fiData, l.inFeat,
			0.0, dW, l.inFeat)
	}
	l.weight.AccumulateGrad(dW)

	// db = sum(gradOutput, axis=0) -> [outFeat]
	if l.useBias && l.bias != nil {
		db := make([]float32, l.outFeat)
		for i := 0; i < batchSize; i++ {
			row := fgData[i*l.outFeat : (i+1)*l.outFeat]
			for j := range row {
				db[j] += row[j]
			}
		}
		l.bias.AccumulateGrad(db)
	}

	return gradInput.Reshape(inputShape)
}

// Parameters returns the weight (and bias, if present).
func (l *Linear) Parameters() []*Tensor {
	if l.useBias {
		return []*Tensor{l.weight, l.bias}
	}
	return []*Tensor{l.weight}
}

// InFeatures returns the input dimension.
func (l *Linear) InFeatures() int { return l.inFeat }

// OutFeatures returns the output dimension.
func (l *Linear) OutFeatures() int { return l.outFeat }

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

// RMSNorm implements Root Mean Square Layer Normalization.
//
//	RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
//
// Unlike LayerNorm, RMSNorm has no mean-centering (no beta), making it
// cheaper and empirically as effective for Transformer pre-norm.
type RMSNorm struct {
	weight    *Tensor   // gamma (learnable scale), shape [dim]
	eps       float32   // numerical stability constant
	dim       int       // normalization dimension
	lastInput *Tensor   // cached for backward
	lastRMS   []float32 // cached rms values per vector for backward
}

// NewRMSNorm creates an RMSNorm layer with gamma initialized to 1.
func NewRMSNorm(dim int, eps float32) *RMSNorm {
	return &RMSNorm{
		weight:  Ones(NewShape(dim), F32),
		eps:     eps,
		dim:     dim,
		lastRMS: make([]float32, 0, 512),
	}
}

// Forward applies RMSNorm along the last dimension.
//
//	rms = sqrt(mean(x^2) + eps)
//	y_i = x_i / rms * gamma_i
func (r *RMSNorm) Forward(input *Tensor) *Tensor {
	r.lastInput = input

	shape := input.Shape()
	numVectors := shape.Numel() / r.dim
	// Reuse the lastRMS buffer to reduce GC pressure across calls.
	if cap(r.lastRMS) >= numVectors {
		r.lastRMS = r.lastRMS[:numVectors]
	} else {
		r.lastRMS = make([]float32, numVectors)
	}

	output := New(shape, F32)
	in, out, w := input.DataPtr(), output.DataPtr(), r.weight.DataPtr()
	for v := 0; v < numVectors; v++ {
		off := v * r.dim
		row := in[off : off+r.dim]

		sumSq := float32(0)
		for _, x := range row {
			sumSq += x * x
		}

		rms := SqrtF32(sumSq/float32(r.dim) + r.eps)
		r.lastRMS[v] = rms
		invRms := 1.0 / rms

		oRow := out[off : off+r.dim]
		for i := range oRow {
			oRow[i] = row[i] * invRms * w[i]
		}
	}
	return output
}

// Backward computes the input gradient for RMSNorm and accumulates the
// weight gradient. Uses the chain rule through the normalization: the gradient
// depends on the cached rms value and the dot product of grad_output with the
// normalized input, producing a correction term.
//
//	d_gamma[i] = sum over all vectors v of (gradOutput[v,i] * input[v,i] / rms[v])
//	d_input = gradOutput * gamma / rms - input * dot(gradOutput*gamma, input) / (dim * rms^3)
func (r *RMSNorm) Backward(gradOutput *Tensor) *Tensor {
	if r.lastInput == nil {
		panic("backward called before forward")
	}
	shape := gradOutput.Shape()
	numVectors := shape.Numel() / r.dim

	gradInput := New(shape, F32)
	gOut, gIn := gradOutput.DataPtr(), gradInput.DataPtr()
	in, w := r.lastInput.DataPtr(), r.weight.DataPtr()

	// Accumulate weight gradient: d_gamma[i] = sum_v(gradOutput[v,i] * x[v,i] / rms[v])
	dGamma := make([]float32, r.dim)

	for v := 0; v < numVectors; v++ {
		off := v * r.dim
		rms := r.lastRMS[v]
		rms3 := rms * rms * rms
		invRms := 1.0 / rms

		// Accumulate d_gamma for this vector
		for i := 0; i < r.dim; i++ {
			dGamma[i] += gOut[off+i] * in[off+i] * invRms
		}

		// dot = sum(grad_out * gamma * input) -- needed for the correction term
		dotSum := float32(0)
		for i := 0; i < r.dim; i++ {
			dotSum += gOut[off+i] * w[i] * in[off+i]
		}
		for i := 0; i < r.dim; i++ {
			gIn[off+i] = gOut[off+i]*w[i]/rms - in[off+i]*dotSum/(float32(r.dim)*rms3)
		}
	}

	r.weight.AccumulateGrad(dGamma)
	return gradInput
}

// Parameters returns the learnable gamma scale vector.
func (r *RMSNorm) Parameters() []*Tensor { return []*Tensor{r.weight} }

// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------

// SwiGLU implements the SwiGLU feed-forward network used in modern Transformers.
//
//	SwiGLU(x) = (SiLU(W_gate @ x) * (W_up @ x)) @ W_down
//
// Expanded:
//   gate = W_gate @ x             -- project to FFN dim
//   gate = gate * sigmoid(gate)   -- SiLU activation
//   up   = W_up @ x               -- parallel up-projection
//   out  = W_down @ (gate * up)   -- down-project back to hidden dim
//
// Three linear projections: gate [hidden -> ffn], up [hidden -> ffn],
// down [ffn -> hidden]. No bias in any of them.
type SwiGLU struct {
	wGate, wUp, wDown       *Linear
	hiddenDim, ffnDim       int
	lastGate, lastUp        *Tensor // cached for backward (silu(gate), up)
	lastGatePreSiLU         *Tensor // cached pre-SiLU gate output for derivative
	lastInput               *Tensor // cached input for backward
}

// NewSwiGLU creates a SwiGLU FFN block.
func NewSwiGLU(hiddenDim, ffnDim int) *SwiGLU {
	return &SwiGLU{
		wGate:     NewLinear(hiddenDim, ffnDim, false),
		wUp:       NewLinear(hiddenDim, ffnDim, false),
		wDown:     NewLinear(ffnDim, hiddenDim, false),
		hiddenDim: hiddenDim,
		ffnDim:    ffnDim,
	}
}

// Forward computes SwiGLU(x) = W_down @ (SiLU(W_gate @ x) * W_up @ x).
// Caches pre-SiLU gate output for the backward pass SiLU derivative.
func (s *SwiGLU) Forward(input *Tensor) *Tensor {
	s.lastInput = input
	gate := s.wGate.Forward(input)
	s.lastGatePreSiLU = gate.Clone() // cache pre-SiLU for backward derivative
	gate.SiLUInPlace()
	s.lastGate = gate.Clone() // cache silu(gate) before MulInPlace mutates it
	up := s.wUp.Forward(input)
	s.lastUp = up
	gate.MulInPlace(up) // gate is now silu(gate) * up
	return s.wDown.Forward(gate)
}

// Backward propagates gradients through the SwiGLU block including the
// SiLU derivative. Forward: hidden = silu(gate(x)) * up(x), output = down(hidden).
//
// Chain rule through SiLU: silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
func (s *SwiGLU) Backward(gradOutput *Tensor) *Tensor {
	// Backward through down projection
	gradHidden := s.wDown.Backward(gradOutput)

	// d(silu(gate) * up) / d(up) = silu(gate)
	gradUp := gradHidden.Mul(s.lastGate)

	// d(silu(gate) * up) / d(silu(gate)) = up
	gradSiluGate := gradHidden.Mul(s.lastUp)

	// Apply SiLU derivative: silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
	preSilu := s.lastGatePreSiLU.DataPtr()
	gSilu := gradSiluGate.DataPtr()
	for i := range gSilu {
		z := preSilu[i]
		sig := 1.0 / (1.0 + ExpF32(-z))
		dSilu := sig * (1.0 + z*(1.0-sig))
		gSilu[i] *= dSilu
	}

	// Ensure lastInput is set for gate and up projections
	s.wGate.lastInput = s.lastInput
	s.wUp.lastInput = s.lastInput

	return s.wGate.Backward(gradSiluGate).Add(s.wUp.Backward(gradUp))
}

// Parameters returns all weights from gate, up, and down projections.
func (s *SwiGLU) Parameters() []*Tensor {
	return concatParams(
		s.wGate.Parameters(),
		s.wUp.Parameters(),
		s.wDown.Parameters(),
	)
}
