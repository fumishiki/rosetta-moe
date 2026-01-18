package layer

import (
	"math"

	"github.com/fumi-engineer/machine_learning/go/tensor"
)

// RMSNorm implements Root Mean Square Layer Normalization.
type RMSNorm struct {
	weight *tensor.Tensor
	eps    float32
	dim    int

	// Cached for backward
	lastInput *tensor.Tensor
	lastRMS   []float32
}

// NewRMSNorm creates a new RMSNorm layer.
func NewRMSNorm(dim int, eps float32) *RMSNorm {
	weight := tensor.Ones(tensor.NewShape(dim), tensor.F32)
	return &RMSNorm{
		weight: weight,
		eps:    eps,
		dim:    dim,
	}
}

// Forward applies RMS normalization.
// Input: [..., dim]
// Output: [..., dim]
func (r *RMSNorm) Forward(input *tensor.Tensor) *tensor.Tensor {
	r.lastInput = input.Clone()

	shape := input.Shape()
	numel := shape.Numel()
	numVectors := numel / r.dim

	r.lastRMS = make([]float32, numVectors)
	output := tensor.New(shape, tensor.F32)
	inputData := input.DataPtr()
	outputData := output.DataPtr()
	weightData := r.weight.DataPtr()

	for v := 0; v < numVectors; v++ {
		offset := v * r.dim

		// Compute RMS
		sumSq := float32(0.0)
		for i := 0; i < r.dim; i++ {
			x := inputData[offset+i]
			sumSq += x * x
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(r.dim) + r.eps)))
		r.lastRMS[v] = rms

		// Normalize and scale
		for i := 0; i < r.dim; i++ {
			outputData[offset+i] = (inputData[offset+i] / rms) * weightData[i]
		}
	}

	return output
}

// Backward computes gradients for RMSNorm.
func (r *RMSNorm) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	if r.lastInput == nil {
		panic("backward called before forward")
	}

	shape := gradOutput.Shape()
	numel := shape.Numel()
	numVectors := numel / r.dim

	gradInput := tensor.New(shape, tensor.F32)
	gradOutputData := gradOutput.DataPtr()
	gradInputData := gradInput.DataPtr()
	inputData := r.lastInput.DataPtr()
	weightData := r.weight.DataPtr()

	for v := 0; v < numVectors; v++ {
		offset := v * r.dim
		rms := r.lastRMS[v]
		rms3 := rms * rms * rms

		// Compute dot product of grad_output * (x/rms) * weight
		dotSum := float32(0.0)
		for i := 0; i < r.dim; i++ {
			x := inputData[offset+i]
			dotSum += gradOutputData[offset+i] * weightData[i] * x / rms
		}

		// Compute gradient
		for i := 0; i < r.dim; i++ {
			x := inputData[offset+i]
			gradInputData[offset+i] = (gradOutputData[offset+i]*weightData[i]/rms -
				x*dotSum/(float32(r.dim)*rms3))
		}
	}

	return gradInput
}

// Parameters returns the layer's weight.
func (r *RMSNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{r.weight}
}
