package layer

import (
	"math"

	"github.com/pikafumi/machine_learning/crates/go/tensor"
)

// Linear implements a fully connected layer.
type Linear struct {
	weight  *tensor.Tensor // [outFeatures, inFeatures]
	bias    *tensor.Tensor // [outFeatures] or nil
	inFeat  int
	outFeat int
	useBias bool

	// Cached for backward
	lastInput *tensor.Tensor
}

// NewLinear creates a new linear layer.
func NewLinear(inFeatures, outFeatures int, useBias bool) *Linear {
	// Kaiming initialization
	std := float32(math.Sqrt(2.0 / float64(inFeatures)))
	weight := tensor.RandnWithStd(tensor.NewShape(outFeatures, inFeatures), tensor.F32, std)

	var bias *tensor.Tensor
	if useBias {
		bias = tensor.Zeros(tensor.NewShape(outFeatures), tensor.F32)
	}

	return &Linear{
		weight:  weight,
		bias:    bias,
		inFeat:  inFeatures,
		outFeat: outFeatures,
		useBias: useBias,
	}
}

// Forward performs linear transformation: y = xW^T + b
// Input: [..., inFeatures]
// Output: [..., outFeatures]
func (l *Linear) Forward(input *tensor.Tensor) *tensor.Tensor {
	l.lastInput = input.Clone()

	shape := input.Shape()
	dims := shape.Dims()
	batchDims := dims[:len(dims)-1]

	// Compute batch size
	batchSize := 1
	for _, d := range batchDims {
		batchSize *= d
	}

	// Reshape input to [batchSize, inFeat]
	flatInput := input.Reshape(tensor.NewShape(batchSize, l.inFeat))

	// Compute y = x @ W^T
	wT := l.weight.Transpose() // [inFeatures, outFeatures]
	output := tensor.Matmul(flatInput, wT)

	// Add bias if present
	if l.useBias {
		outputData := output.DataPtr()
		biasData := l.bias.DataPtr()
		for b := 0; b < batchSize; b++ {
			offset := b * l.outFeat
			for i := 0; i < l.outFeat; i++ {
				outputData[offset+i] += biasData[i]
			}
		}
	}

	// Reshape output to [..., outFeatures]
	outDims := append(batchDims, l.outFeat)
	return output.Reshape(tensor.NewShape(outDims...))
}

// Backward computes gradients for linear layer.
func (l *Linear) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	if l.lastInput == nil {
		panic("backward called before forward")
	}

	// Get original input shape for final reshape
	inputShape := l.lastInput.Shape()

	shape := gradOutput.Shape()
	dims := shape.Dims()
	batchDims := dims[:len(dims)-1]

	batchSize := 1
	for _, d := range batchDims {
		batchSize *= d
	}

	// Flatten gradOutput to [batchSize, outFeat]
	flatGrad := gradOutput.Reshape(tensor.NewShape(batchSize, l.outFeat))

	// gradInput = gradOutput @ W
	gradInput := tensor.Matmul(flatGrad, l.weight)

	// Reshape to original input shape (use saved shape, not computed from gradOutput)
	return gradInput.Reshape(inputShape)
}

// Parameters returns weight and optionally bias.
func (l *Linear) Parameters() []*tensor.Tensor {
	if l.useBias {
		return []*tensor.Tensor{l.weight, l.bias}
	}
	return []*tensor.Tensor{l.weight}
}

// InFeatures returns input features.
func (l *Linear) InFeatures() int {
	return l.inFeat
}

// OutFeatures returns output features.
func (l *Linear) OutFeatures() int {
	return l.outFeat
}
