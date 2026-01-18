package tensor

import (
	"fmt"
	"strings"
)

// Shape represents the dimensions of a tensor.
type Shape struct {
	dims []int
}

// NewShape creates a new Shape from dimensions.
func NewShape(dims ...int) Shape {
	d := make([]int, len(dims))
	copy(d, dims)
	return Shape{dims: d}
}

// Dims returns a copy of the dimensions.
func (s Shape) Dims() []int {
	d := make([]int, len(s.dims))
	copy(d, s.dims)
	return d
}

// NDim returns the number of dimensions.
func (s Shape) NDim() int {
	return len(s.dims)
}

// Numel returns the total number of elements.
func (s Shape) Numel() int {
	if len(s.dims) == 0 {
		return 0
	}
	n := 1
	for _, d := range s.dims {
		n *= d
	}
	return n
}

// At returns the size at the given dimension.
func (s Shape) At(dim int) int {
	if dim < 0 {
		dim = len(s.dims) + dim
	}
	if dim < 0 || dim >= len(s.dims) {
		return 0
	}
	return s.dims[dim]
}

// Strides returns the strides for row-major layout.
func (s Shape) Strides() []int {
	if len(s.dims) == 0 {
		return nil
	}
	strides := make([]int, len(s.dims))
	strides[len(s.dims)-1] = 1
	for i := len(s.dims) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * s.dims[i+1]
	}
	return strides
}

// Equal checks if two shapes are equal.
func (s Shape) Equal(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i := range s.dims {
		if s.dims[i] != other.dims[i] {
			return false
		}
	}
	return true
}

// String returns a string representation.
func (s Shape) String() string {
	parts := make([]string, len(s.dims))
	for i, d := range s.dims {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Broadcast computes the broadcast shape of two shapes.
func Broadcast(a, b Shape) (Shape, error) {
	maxLen := len(a.dims)
	if len(b.dims) > maxLen {
		maxLen = len(b.dims)
	}

	result := make([]int, maxLen)
	for i := range result {
		dimA := 1
		dimB := 1
		if i < len(a.dims) {
			dimA = a.dims[len(a.dims)-1-i]
		}
		if i < len(b.dims) {
			dimB = b.dims[len(b.dims)-1-i]
		}

		if dimA != dimB && dimA != 1 && dimB != 1 {
			return Shape{}, fmt.Errorf("cannot broadcast shapes %v and %v", a, b)
		}

		if dimA > dimB {
			result[maxLen-1-i] = dimA
		} else {
			result[maxLen-1-i] = dimB
		}
	}

	return Shape{dims: result}, nil
}
