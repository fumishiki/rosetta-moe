// Package layer provides neural network layers for the MoE Transformer.
package layer

import (
	"github.com/fumi-engineer/machine_learning/go/tensor"
)

// Layer is the interface for all neural network layers.
type Layer interface {
	// Forward performs forward pass.
	Forward(input *tensor.Tensor) *tensor.Tensor
	// Backward performs backward pass.
	Backward(gradOutput *tensor.Tensor) *tensor.Tensor
	// Parameters returns the layer's trainable parameters.
	Parameters() []*tensor.Tensor
}
