package tensor

import (
	"fmt"
	"math"
	"math/rand"
)

// Tensor represents a multi-dimensional array.
type Tensor struct {
	data  []float32
	shape Shape
	dtype DType
}

// New creates a new tensor with the given shape and dtype.
func New(shape Shape, dtype DType) *Tensor {
	return &Tensor{
		data:  make([]float32, shape.Numel()),
		shape: shape,
		dtype: dtype,
	}
}

// Zeros creates a zero-filled tensor.
func Zeros(shape Shape, dtype DType) *Tensor {
	return New(shape, dtype)
}

// Ones creates a ones-filled tensor.
func Ones(shape Shape, dtype DType) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = 1.0
	}
	return t
}

// FromSlice creates a tensor from a slice.
func FromSlice(data []float32, shape Shape) *Tensor {
	if len(data) != shape.Numel() {
		panic(fmt.Sprintf("data length %d != shape numel %d", len(data), shape.Numel()))
	}
	d := make([]float32, len(data))
	copy(d, data)
	return &Tensor{
		data:  d,
		shape: shape,
		dtype: F32,
	}
}

// Randn creates a tensor with random normal values.
func Randn(shape Shape, dtype DType) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = float32(rand.NormFloat64())
	}
	return t
}

// RandnWithStd creates a tensor with random normal values with given std.
func RandnWithStd(shape Shape, dtype DType, std float32) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = float32(rand.NormFloat64()) * std
	}
	return t
}

// Shape returns the tensor's shape.
func (t *Tensor) Shape() Shape {
	return t.shape
}

// DType returns the tensor's dtype.
func (t *Tensor) DType() DType {
	return t.dtype
}

// Data returns a copy of the underlying data.
func (t *Tensor) Data() []float32 {
	d := make([]float32, len(t.data))
	copy(d, t.data)
	return d
}

// DataPtr returns the underlying data pointer (use with caution).
func (t *Tensor) DataPtr() []float32 {
	return t.data
}

// At returns the value at the given indices.
func (t *Tensor) At(indices ...int) float32 {
	if len(indices) != t.shape.NDim() {
		panic(fmt.Sprintf("expected %d indices, got %d", t.shape.NDim(), len(indices)))
	}
	idx := 0
	strides := t.shape.Strides()
	for i, index := range indices {
		if index < 0 || index >= t.shape.At(i) {
			panic(fmt.Sprintf("index %d out of bounds for dim %d with size %d", index, i, t.shape.At(i)))
		}
		idx += index * strides[i]
	}
	return t.data[idx]
}

// Set sets the value at the given indices.
func (t *Tensor) Set(value float32, indices ...int) {
	if len(indices) != t.shape.NDim() {
		panic(fmt.Sprintf("expected %d indices, got %d", t.shape.NDim(), len(indices)))
	}
	idx := 0
	strides := t.shape.Strides()
	for i, index := range indices {
		if index < 0 || index >= t.shape.At(i) {
			panic(fmt.Sprintf("index %d out of bounds for dim %d with size %d", index, i, t.shape.At(i)))
		}
		idx += index * strides[i]
	}
	t.data[idx] = value
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	return FromSlice(t.data, t.shape)
}

// Reshape returns a reshaped view (must have same numel).
func (t *Tensor) Reshape(newShape Shape) *Tensor {
	if t.shape.Numel() != newShape.Numel() {
		panic(fmt.Sprintf("cannot reshape %v to %v: different numel", t.shape, newShape))
	}
	return &Tensor{
		data:  t.data, // shared data
		shape: newShape,
		dtype: t.dtype,
	}
}

// Add performs element-wise addition.
func (t *Tensor) Add(other *Tensor) *Tensor {
	if !t.shape.Equal(other.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", t.shape, other.shape))
	}
	result := New(t.shape, t.dtype)
	for i := range t.data {
		result.data[i] = t.data[i] + other.data[i]
	}
	return result
}

// Sub performs element-wise subtraction.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if !t.shape.Equal(other.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", t.shape, other.shape))
	}
	result := New(t.shape, t.dtype)
	for i := range t.data {
		result.data[i] = t.data[i] - other.data[i]
	}
	return result
}

// Mul performs element-wise multiplication.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if !t.shape.Equal(other.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", t.shape, other.shape))
	}
	result := New(t.shape, t.dtype)
	for i := range t.data {
		result.data[i] = t.data[i] * other.data[i]
	}
	return result
}

// Scale multiplies by a scalar.
func (t *Tensor) Scale(s float32) *Tensor {
	result := New(t.shape, t.dtype)
	for i := range t.data {
		result.data[i] = t.data[i] * s
	}
	return result
}

// SiLU applies SiLU activation (x * sigmoid(x)).
func (t *Tensor) SiLU() *Tensor {
	result := New(t.shape, t.dtype)
	for i := range t.data {
		x := t.data[i]
		result.data[i] = x / (1.0 + float32(math.Exp(float64(-x))))
	}
	return result
}

// Softmax applies softmax along the last dimension.
func (t *Tensor) Softmax() *Tensor {
	if t.shape.NDim() < 1 {
		panic("softmax requires at least 1 dimension")
	}

	result := New(t.shape, t.dtype)
	lastDim := t.shape.At(-1)
	numVectors := t.shape.Numel() / lastDim

	for v := 0; v < numVectors; v++ {
		offset := v * lastDim

		// Find max for numerical stability
		maxVal := t.data[offset]
		for i := 1; i < lastDim; i++ {
			if t.data[offset+i] > maxVal {
				maxVal = t.data[offset+i]
			}
		}

		// Compute exp and sum
		sum := float32(0.0)
		for i := 0; i < lastDim; i++ {
			result.data[offset+i] = float32(math.Exp(float64(t.data[offset+i] - maxVal)))
			sum += result.data[offset+i]
		}

		// Normalize
		for i := 0; i < lastDim; i++ {
			result.data[offset+i] /= sum
		}
	}

	return result
}

// Matmul performs matrix multiplication.
// For 2D: [M, K] x [K, N] -> [M, N]
// For batched: [..., M, K] x [..., K, N] -> [..., M, N]
func Matmul(a, b *Tensor) *Tensor {
	if a.shape.NDim() < 2 || b.shape.NDim() < 2 {
		panic("matmul requires at least 2D tensors")
	}

	// Get dimensions
	aM := a.shape.At(-2)
	aK := a.shape.At(-1)
	bK := b.shape.At(-2)
	bN := b.shape.At(-1)

	if aK != bK {
		panic(fmt.Sprintf("matmul dimension mismatch: %d vs %d", aK, bK))
	}

	// Handle batched case (simplified: assuming same batch dims)
	var batchSize int
	var resultShape Shape
	if a.shape.NDim() == 2 && b.shape.NDim() == 2 {
		batchSize = 1
		resultShape = NewShape(aM, bN)
	} else {
		// For simplicity, assume 3D [batch, M, K] x [batch, K, N]
		if a.shape.NDim() == 3 && b.shape.NDim() == 3 {
			batchSize = a.shape.At(0)
			resultShape = NewShape(batchSize, aM, bN)
		} else {
			panic("unsupported batch dimensions")
		}
	}

	result := New(resultShape, a.dtype)

	// Naive matmul (CPU)
	for batch := 0; batch < batchSize; batch++ {
		aOffset := batch * aM * aK
		bOffset := batch * bK * bN
		cOffset := batch * aM * bN

		for i := 0; i < aM; i++ {
			for j := 0; j < bN; j++ {
				sum := float32(0.0)
				for k := 0; k < aK; k++ {
					sum += a.data[aOffset+i*aK+k] * b.data[bOffset+k*bN+j]
				}
				result.data[cOffset+i*bN+j] = sum
			}
		}
	}

	return result
}

// Transpose transposes the last two dimensions.
func (t *Tensor) Transpose() *Tensor {
	if t.shape.NDim() < 2 {
		panic("transpose requires at least 2D tensor")
	}

	dims := t.shape.Dims()
	dims[len(dims)-1], dims[len(dims)-2] = dims[len(dims)-2], dims[len(dims)-1]
	resultShape := NewShape(dims...)
	result := New(resultShape, t.dtype)

	rows := t.shape.At(-2)
	cols := t.shape.At(-1)
	batchSize := t.shape.Numel() / (rows * cols)

	for batch := 0; batch < batchSize; batch++ {
		srcOffset := batch * rows * cols
		dstOffset := batch * cols * rows
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result.data[dstOffset+j*rows+i] = t.data[srcOffset+i*cols+j]
			}
		}
	}

	return result
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() float32 {
	sum := float32(0.0)
	for _, v := range t.data {
		sum += v
	}
	return sum
}

// Mean returns the mean of all elements.
func (t *Tensor) Mean() float32 {
	return t.Sum() / float32(len(t.data))
}
