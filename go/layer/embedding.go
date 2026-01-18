package layer

import (
	"math"

	"github.com/fumi-engineer/machine_learning/go/tensor"
)

// Embedding implements token embedding lookup.
type Embedding struct {
	weight    *tensor.Tensor
	vocabSize int
	embedDim  int

	// Cached for backward
	lastInput []int
}

// NewEmbedding creates a new embedding layer.
func NewEmbedding(vocabSize, embedDim int) *Embedding {
	// Initialize with normal distribution scaled by 1/sqrt(embedDim)
	std := float32(1.0 / math.Sqrt(float64(embedDim)))
	weight := tensor.RandnWithStd(tensor.NewShape(vocabSize, embedDim), tensor.F32, std)
	return &Embedding{
		weight:    weight,
		vocabSize: vocabSize,
		embedDim:  embedDim,
	}
}

// Forward performs embedding lookup.
// Input: [batch, seq_len] token IDs (stored as float32)
// Output: [batch, seq_len, embed_dim]
func (e *Embedding) Forward(input *tensor.Tensor) *tensor.Tensor {
	dims := input.Shape().Dims()
	batch := dims[0]
	seqLen := dims[1]

	// Store input for backward
	e.lastInput = make([]int, batch*seqLen)
	for i := 0; i < batch*seqLen; i++ {
		e.lastInput[i] = int(input.DataPtr()[i])
	}

	// Output shape: [batch, seq_len, embed_dim]
	output := tensor.New(tensor.NewShape(batch, seqLen, e.embedDim), tensor.F32)
	outputData := output.DataPtr()
	weightData := e.weight.DataPtr()

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			tokenID := e.lastInput[b*seqLen+s]
			if tokenID < 0 || tokenID >= e.vocabSize {
				panic("token ID out of range")
			}

			srcOffset := tokenID * e.embedDim
			dstOffset := (b*seqLen + s) * e.embedDim
			copy(outputData[dstOffset:dstOffset+e.embedDim], weightData[srcOffset:srcOffset+e.embedDim])
		}
	}

	return output
}

// Backward computes gradients for embedding.
func (e *Embedding) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// Embedding backward uses scatter_add
	// For now, return zero gradient (embedding backward is handled specially)
	return tensor.Zeros(gradOutput.Shape(), tensor.F32)
}

// Parameters returns the embedding weight.
func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.weight}
}

// VocabSize returns the vocabulary size.
func (e *Embedding) VocabSize() int {
	return e.vocabSize
}

// EmbedDim returns the embedding dimension.
func (e *Embedding) EmbedDim() int {
	return e.embedDim
}
