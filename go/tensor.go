// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

// Package nn implements a Mixture-of-Experts Transformer from scratch in Go.
//
// All tensor storage uses flat []float32 slices in row-major order.
// Matrix multiplication is delegated to Apple Accelerate (cblas_sgemm) via CGO.
// No external Go dependencies beyond the standard library.
package nn

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// DType enumerates supported data types. Only F32 is used at runtime;
// the others exist for future mixed-precision support.
type DType uint8

const (
	F32 DType = iota
	F16
	BF16
	I32
	I64
)

// Size returns the byte width of the data type.
func (d DType) Size() int {
	switch d {
	case F32, I32:
		return 4
	case F16, BF16:
		return 2
	case I64:
		return 8
	default:
		return 4
	}
}

// String returns a human-readable name for the data type.
func (d DType) String() string {
	names := [...]string{"f32", "f16", "bf16", "i32", "i64"}
	if int(d) < len(names) {
		return names[d]
	}
	return "unknown"
}

// Shape represents the dimensions of a tensor. Internally stored as a
// private slice to prevent external mutation.
type Shape struct{ dims []int }

// NewShape creates a Shape from variadic dimension sizes.
func NewShape(dims ...int) Shape {
	d := make([]int, len(dims))
	copy(d, dims)
	return Shape{dims: d}
}

// Dims returns a copy of the dimension sizes.
func (s Shape) Dims() []int {
	d := make([]int, len(s.dims))
	copy(d, s.dims)
	return d
}

// DimsRef returns a direct reference to the internal dimension slice.
// The caller must NOT mutate the returned slice. Used in hot paths to
// avoid the allocation that Dims() incurs.
func (s Shape) DimsRef() []int {
	return s.dims
}

// NDim returns the number of dimensions.
func (s Shape) NDim() int { return len(s.dims) }

// Numel returns the total number of elements (product of all dimensions).
func (s Shape) Numel() int {
	if len(s.dims) == 0 {
		return 0
	}
	return prod(s.dims)
}

// At returns the size of dimension dim. Negative indices count from the end
// (e.g., At(-1) returns the last dimension), matching NumPy convention.
func (s Shape) At(dim int) int {
	if dim < 0 {
		dim += len(s.dims)
	}
	if dim < 0 || dim >= len(s.dims) {
		return 0
	}
	return s.dims[dim]
}

// Strides returns row-major strides for the shape.
// For shape [2, 3, 4] the strides are [12, 4, 1], meaning moving
// one step along dim 0 advances 12 elements in flat storage.
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

// Equal returns true if two shapes have identical dimensions.
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

// String formats the shape as "[d0, d1, ...]".
func (s Shape) String() string {
	parts := make([]string, len(s.dims))
	for i, d := range s.dims {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Broadcast computes the broadcast-compatible output shape for two inputs,
// following NumPy broadcasting rules: dimensions are compared right-to-left,
// and each pair must either be equal or one of them must be 1.
func Broadcast(a, b Shape) (Shape, error) {
	maxLen := len(a.dims)
	if len(b.dims) > maxLen {
		maxLen = len(b.dims)
	}
	result := make([]int, maxLen)
	for i := range result {
		dimA, dimB := 1, 1
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

// NegInf is the most negative finite float32, used as -infinity for masking
// in attention and softmax (avoids importing math.Inf which returns float64).
const NegInf = -float32(math.MaxFloat32)

// ---------------------------------------------------------------------------
// Pure-float32 math functions
//
// These avoid float64 casts to keep the entire compute path in float32,
// matching what the Accelerate sgemm operates on. Each uses standard
// numerical techniques: range reduction, Horner polynomials, and
// fast inverse sqrt.
// ---------------------------------------------------------------------------

// ExpF32 computes exp(x) in pure float32.
//
// Algorithm: range reduction x = k*ln2 + r, then Horner polynomial on r.
//   exp(x) = 2^k * (1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5!)
//
// Clamps to 0 / +Inf outside the representable range of float32.
func ExpF32(x float32) float32 {
	if x > 88.72 {
		return float32(math.Inf(1))
	}
	if x < -87.33 {
		return 0
	}
	const (
		invLn2 = float32(1.4426950)
		ln2Hi  = float32(0.6931458)
		ln2Lo  = float32(1.4286068e-06)
	)
	var k int32
	if x >= 0 {
		k = int32(x*invLn2 + 0.5)
	} else {
		k = int32(x*invLn2 - 0.5)
	}
	kf := float32(k)
	r := x - kf*ln2Hi - kf*ln2Lo
	r2 := r * r
	p := float32(1) + r + r2*(0.5+r*(0.16666667+r*(0.04166668+r*0.008333334)))
	// Reconstruct 2^k by shifting into the IEEE 754 exponent field.
	return p * math.Float32frombits(uint32(127+k)<<23)
}

// SqrtF32 computes sqrt(x) via the fast inverse square root trick
// (Quake III style) followed by two Newton-Raphson refinement steps.
//
//	y_0 = magic(x)          -- initial estimate of 1/sqrt(x)
//	y_{n+1} = y_n * (1.5 - 0.5*x*y_n^2)   -- Newton step
//	sqrt(x) = x * y_final                  -- invert
func SqrtF32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	bits := math.Float32bits(x)
	bits = 0x5f3759df - (bits >> 1)
	y := math.Float32frombits(bits)
	half := 0.5 * x
	y = y * (1.5 - half*y*y)
	y = y * (1.5 - half*y*y)
	return x * y
}

// LogF32 computes ln(x) via IEEE 754 decomposition: x = 2^e * m,
// then atanh-series polynomial on s = (m-1)/(m+1).
//
//	ln(x) = e*ln(2) + 2*s*(1 + s^2/3 + s^4/5 + s^6/7)
func LogF32(x float32) float32 {
	if x <= 0 {
		return NegInf
	}
	bits := math.Float32bits(x)
	e := int32((bits>>23)&0xFF) - 127
	bits = (bits & 0x007FFFFF) | 0x3F800000
	m := math.Float32frombits(bits)
	s := (m - 1) / (m + 1)
	s2 := s * s
	p := 2.0 * s * (1 + s2*(0.33333334+s2*(0.2+s2*0.14285715)))
	return float32(e)*0.6931472 + p
}

// PowF32 computes base^exp in float32 via exp(exp * ln(base)).
func PowF32(base, exp float32) float32 {
	if base <= 0 {
		return 0
	}
	return ExpF32(exp * LogF32(base))
}

// SinF32 computes sin(x) via range reduction to [0, pi/2]
// then a Horner polynomial approximation.
//
//	sin(x) ~ x * (1 - x^2/3! + x^4/5! - x^6/7!)
func SinF32(x float32) float32 {
	const (
		twoPi  = float32(6.2831855)
		pi     = float32(3.1415927)
		halfPi = float32(1.5707964)
	)
	x -= float32(int32(x/twoPi)) * twoPi
	if x < 0 {
		x += twoPi
	}
	sign := float32(1)
	if x > pi {
		sign = -1
		x -= pi
	}
	if x > halfPi {
		x = pi - x
	}
	x2 := x * x
	return sign * x * (1 - x2*(0.16666667-x2*(0.008333334-x2*0.00019841270)))
}

// CosF32 computes cos(x) = sin(x + pi/2).
func CosF32(x float32) float32 { return SinF32(x + 1.5707964) }

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

// Tensor stores multi-dimensional float32 data in a contiguous flat slice.
// Row-major layout: the last dimension varies fastest. All operations
// allocate new tensors unless suffixed with "InPlace".
type Tensor struct {
	data  []float32
	shape Shape
	dtype DType
	Grad  []float32 // per-element gradient, nil until allocated
}

// ZeroGrad resets the gradient. If Grad exists and matches the data length,
// it is zeroed in place to avoid reallocation. Otherwise Grad is set to nil
// so that only parameters that actually receive gradients via AccumulateGrad
// will have a non-nil Grad after the backward pass.
func (t *Tensor) ZeroGrad() {
	n := len(t.data)
	if t.Grad != nil && len(t.Grad) == n {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	} else {
		t.Grad = nil
	}
}

// AccumulateGrad adds grad element-wise into t.Grad, allocating if nil.
func (t *Tensor) AccumulateGrad(grad []float32) {
	if t.Grad == nil {
		t.Grad = make([]float32, len(t.data))
	}
	for i, g := range grad {
		t.Grad[i] += g
	}
}

// New allocates a zero-filled tensor of the given shape and dtype.
func New(shape Shape, dtype DType) *Tensor {
	return &Tensor{data: make([]float32, shape.Numel()), shape: shape, dtype: dtype}
}

// Zeros is an alias for New (zero-filled tensor).
func Zeros(shape Shape, dtype DType) *Tensor { return New(shape, dtype) }

// Ones allocates a tensor filled with 1.0.
func Ones(shape Shape, dtype DType) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = 1
	}
	return t
}

// FromSlice creates a tensor by copying the provided data.
// Panics if len(data) != shape.Numel().
func FromSlice(data []float32, shape Shape) *Tensor {
	if len(data) != shape.Numel() {
		panic(fmt.Sprintf("data length %d != shape numel %d", len(data), shape.Numel()))
	}
	d := make([]float32, len(data))
	copy(d, data)
	return &Tensor{data: d, shape: shape, dtype: F32}
}

// FromSliceNoCopy creates a tensor that directly owns the provided slice
// (no copy). The caller must NOT retain or mutate the slice after this call.
// Used in performance-critical paths where the data is freshly allocated.
func FromSliceNoCopy(data []float32, shape Shape) *Tensor {
	if len(data) != shape.Numel() {
		panic(fmt.Sprintf("data length %d != shape numel %d", len(data), shape.Numel()))
	}
	return &Tensor{data: data, shape: shape, dtype: F32}
}

// Randn allocates a tensor filled with standard normal random values (mean=0, std=1).
func Randn(shape Shape, dtype DType) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = float32(rand.NormFloat64())
	}
	return t
}

// RandnWithStd allocates a tensor filled with normal random values scaled by std.
func RandnWithStd(shape Shape, dtype DType, std float32) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = float32(rand.NormFloat64()) * std
	}
	return t
}

// Shape returns the tensor's shape.
func (t *Tensor) Shape() Shape { return t.shape }

// DType returns the tensor's data type tag.
func (t *Tensor) DType() DType { return t.dtype }

// DataPtr returns the underlying storage slice directly (no copy).
// Callers may mutate elements in-place; use Data() for a safe copy.
func (t *Tensor) DataPtr() []float32 { return t.data }

// Data returns a copy of the underlying storage.
func (t *Tensor) Data() []float32 {
	d := make([]float32, len(t.data))
	copy(d, t.data)
	return d
}

// flatIndex converts multi-dimensional indices to a flat offset using
// row-major strides. Panics on out-of-bounds access.
func (t *Tensor) flatIndex(indices []int) int {
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
	return idx
}

// At reads a single element by multi-dimensional index.
func (t *Tensor) At(indices ...int) float32 { return t.data[t.flatIndex(indices)] }

// Set writes a single element by multi-dimensional index.
func (t *Tensor) Set(value float32, indices ...int) { t.data[t.flatIndex(indices)] = value }

// Clone returns a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor { return FromSlice(t.data, t.shape) }

// Reshape returns a new tensor sharing the same backing data but with a
// different shape. The total number of elements must be unchanged.
// WARNING: because data is shared, mutations to one affect the other.
func (t *Tensor) Reshape(s Shape) *Tensor {
	if t.shape.Numel() != s.Numel() {
		panic(fmt.Sprintf("cannot reshape %v to %v: different numel", t.shape, s))
	}
	return &Tensor{data: t.data, shape: s, dtype: t.dtype}
}

func (t *Tensor) assertShape(other *Tensor) {
	if !t.shape.Equal(other.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", t.shape, other.shape))
	}
}

func (t *Tensor) unaryOp(f func(float32) float32) *Tensor {
	r := New(t.shape, t.dtype)
	src, dst := t.data, r.data
	for i := range dst {
		dst[i] = f(src[i])
	}
	return r
}

func (t *Tensor) binaryOp(other *Tensor, f func(a, b float32) float32) *Tensor {
	t.assertShape(other)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, other.data, r.data
	for i := range dst {
		dst[i] = f(a[i], b[i])
	}
	return r
}

// Add returns element-wise t + o. Direct loop avoids closure dispatch overhead.
func (t *Tensor) Add(o *Tensor) *Tensor {
	t.assertShape(o)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, o.data, r.data
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
	return r
}

// Sub returns element-wise t - o.
func (t *Tensor) Sub(o *Tensor) *Tensor {
	t.assertShape(o)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, o.data, r.data
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
	return r
}

// Mul returns element-wise t * o (Hadamard product).
func (t *Tensor) Mul(o *Tensor) *Tensor {
	t.assertShape(o)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, o.data, r.data
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
	return r
}

// Scale returns t * s (scalar multiplication). Direct loop avoids closure overhead.
func (t *Tensor) Scale(s float32) *Tensor {
	r := New(t.shape, t.dtype)
	src, dst := t.data, r.data
	for i := range dst {
		dst[i] = src[i] * s
	}
	return r
}

// SiLU returns the SiLU (Swish) activation applied element-wise.
//   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
func (t *Tensor) SiLU() *Tensor {
	r := New(t.shape, t.dtype)
	src, dst := t.data, r.data
	for i, x := range src {
		dst[i] = x / (1 + ExpF32(-x))
	}
	return r
}

// AddInPlace adds other to t element-wise, mutating t.
func (t *Tensor) AddInPlace(other *Tensor) {
	t.assertShape(other)
	a, b := t.data, other.data
	for i := range a {
		a[i] += b[i]
	}
}

// SiLUInPlace applies SiLU activation in-place, avoiding a temporary allocation.
//   SiLU(x) = x / (1 + exp(-x))
func (t *Tensor) SiLUInPlace() {
	for i, x := range t.data {
		t.data[i] = x / (1 + ExpF32(-x))
	}
}

// MulInPlace multiplies t by other element-wise, mutating t.
func (t *Tensor) MulInPlace(other *Tensor) {
	t.assertShape(other)
	a, b := t.data, other.data
	for i := range a {
		a[i] *= b[i]
	}
}

// ScaleInPlace multiplies every element of t by s, mutating t.
func (t *Tensor) ScaleInPlace(s float32) {
	for i := range t.data {
		t.data[i] *= s
	}
}

// softmaxCore computes row-wise softmax from src into dst along the last dimension.
// Shared implementation for both allocating and in-place variants.
func softmaxCore(src, dst []float32, lastDim, numVectors int) {
	for v := 0; v < numVectors; v++ {
		off := v * lastDim
		sRow := src[off : off+lastDim]
		dRow := dst[off : off+lastDim]

		maxVal := sRow[0]
		for i := 1; i < lastDim; i++ {
			if sRow[i] > maxVal {
				maxVal = sRow[i]
			}
		}
		sum := float32(0)
		for i := 0; i < lastDim; i++ {
			e := ExpF32(sRow[i] - maxVal)
			dRow[i] = e
			sum += e
		}
		invSum := 1.0 / sum
		for i := 0; i < lastDim; i++ {
			dRow[i] *= invSum
		}
	}
}

// Softmax computes row-wise softmax along the last dimension.
//   p_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
//
// The max-subtraction provides numerical stability by preventing overflow
// in the exponential. Applied independently to each row (last-dim vector).
func (t *Tensor) Softmax() *Tensor {
	if t.shape.NDim() < 1 {
		panic("softmax requires at least 1 dimension")
	}
	result := New(t.shape, t.dtype)
	lastDim := t.shape.At(-1)
	numVectors := t.shape.Numel() / lastDim
	softmaxCore(t.data, result.data, lastDim, numVectors)
	return result
}

// SoftmaxInto computes row-wise softmax into a pre-allocated output tensor,
// avoiding allocation. The output must have the same shape as the input.
func (t *Tensor) SoftmaxInto(out *Tensor) {
	t.assertShape(out)
	lastDim := t.shape.At(-1)
	numVectors := t.shape.Numel() / lastDim
	softmaxCore(t.data, out.data, lastDim, numVectors)
}

// Matmul computes matrix multiplication C = A @ B.
//
//	C[i,j] = sum_k A[i,k] * B[k,j]
//
// Supports 2D [M,K] x [K,N] -> [M,N] and batched 3D [B,M,K] x [B,K,N] -> [B,M,N].
// Delegates to Apple Accelerate cblas_sgemm for hardware-accelerated BLAS
// (routes through AMX on Apple Silicon).
func Matmul(a, b *Tensor) *Tensor {
	if a.shape.NDim() < 2 || b.shape.NDim() < 2 {
		panic("matmul requires at least 2D tensors")
	}
	aM, aK := a.shape.At(-2), a.shape.At(-1)
	bK, bN := b.shape.At(-2), b.shape.At(-1)
	if aK != bK {
		panic(fmt.Sprintf("matmul dimension mismatch: %d vs %d", aK, bK))
	}

	var batchSize int
	var resultShape Shape
	switch {
	case a.shape.NDim() == 2 && b.shape.NDim() == 2:
		batchSize = 1
		resultShape = NewShape(aM, bN)
	case a.shape.NDim() == 3 && b.shape.NDim() == 3:
		if a.shape.At(0) != b.shape.At(0) {
			panic(fmt.Sprintf("matmul batch mismatch: %d vs %d", a.shape.At(0), b.shape.At(0)))
		}
		batchSize = a.shape.At(0)
		resultShape = NewShape(batchSize, aM, bN)
	default:
		panic("unsupported batch dimensions")
	}

	result := New(resultShape, a.dtype)
	aStride, bStride, cStride := aM*aK, bK*bN, aM*bN

	for batch := 0; batch < batchSize; batch++ {
		aOff, bOff, cOff := batch*aStride, batch*bStride, batch*cStride
		// sgemm wraps cblas_sgemm from Apple Accelerate. See sgemm.go.
		sgemm(aM, bN, aK,
			1.0, a.data[aOff:aOff+aStride], aK,
			b.data[bOff:bOff+bStride], bN,
			0.0, result.data[cOff:cOff+cStride], bN)
	}
	return result
}

// MatmulTransposedB computes C = A @ B^T without materializing the transpose.
// A: [M, K], B: [N, K] -> C: [M, N]. Uses CblasTrans flag on B in sgemm,
// saving a full transpose allocation. This is the hot path for Linear.Forward.
func MatmulTransposedB(a, b *Tensor) *Tensor {
	if a.shape.NDim() != 2 || b.shape.NDim() != 2 {
		panic("MatmulTransposedB requires 2D tensors")
	}
	aM, aK := a.shape.At(-2), a.shape.At(-1)
	bN, bK := b.shape.At(-2), b.shape.At(-1)
	if aK != bK {
		panic(fmt.Sprintf("matmulT dimension mismatch: %d vs %d", aK, bK))
	}
	result := New(NewShape(aM, bN), a.dtype)
	sgemmTransB(aM, bN, aK,
		1.0, a.data, aK,
		b.data, bK,
		0.0, result.data, bN)
	return result
}

// Transpose swaps the last two dimensions.
// For a [B, M, N] tensor, produces [B, N, M] by explicit element copy.
// Flat index mapping: dst[j*rows + i] = src[i*cols + j].
func (t *Tensor) Transpose() *Tensor {
	if t.shape.NDim() < 2 {
		panic("transpose requires at least 2D tensor")
	}
	dims := t.shape.Dims()
	dims[len(dims)-1], dims[len(dims)-2] = dims[len(dims)-2], dims[len(dims)-1]
	result := New(NewShape(dims...), t.dtype)
	rows, cols := t.shape.At(-2), t.shape.At(-1)
	batchSize := t.shape.Numel() / (rows * cols)
	for batch := 0; batch < batchSize; batch++ {
		srcOff, dstOff := batch*rows*cols, batch*cols*rows
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result.data[dstOff+j*rows+i] = t.data[srcOff+i*cols+j]
			}
		}
	}
	return result
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() float32 {
	sum := float32(0)
	for _, v := range t.data {
		sum += v
	}
	return sum
}

// Mean returns the arithmetic mean of all elements.
func (t *Tensor) Mean() float32 { return t.Sum() / float32(len(t.data)) }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// prod returns the product of all integers in xs.
func prod(xs []int) int {
	n := 1
	for _, x := range xs {
		n *= x
	}
	return n
}

// splitLast splits dims into (leading dims, product of leading dims, last dim).
// Used throughout the codebase to treat [batch, seq, hidden] as (batch*seq, hidden)
// for 2D matmul operations.
func splitLast(dims []int) (leading []int, leadingSize int, last int) {
	if len(dims) == 0 {
		panic("shape must have at least one dimension")
	}
	last = dims[len(dims)-1]
	leading = dims[:len(dims)-1]
	leadingSize = prod(leading)
	return leading, leadingSize, last
}

// withLastDim creates a new shape by appending last to the leading dimensions.
// Restores the original batch dims after a flattened matmul.
func withLastDim(dims []int, last int) Shape {
	out := append(append([]int(nil), dims...), last)
	return NewShape(out...)
}

// concatParams concatenates multiple parameter slices into one.
// Used by composite layers to aggregate their sub-layer parameters.
func concatParams(groups ...[]*Tensor) []*Tensor {
	total := 0
	for _, g := range groups {
		total += len(g)
	}
	out := make([]*Tensor, 0, total)
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

func cloneInts(src []int) []int {
	dst := make([]int, len(src))
	copy(dst, src)
	return dst
}

// argmax returns the index and value of the maximum element.
func argmax(xs []float32) (int, float32) {
	bestIdx, bestVal := 0, xs[0]
	for i := 1; i < len(xs); i++ {
		if xs[i] > bestVal {
			bestIdx, bestVal = i, xs[i]
		}
	}
	return bestIdx, bestVal
}

// normalizeInPlace divides every element by the sum so they sum to 1.
// Used to renormalize top-k expert gate weights and sampling probabilities.
func normalizeInPlace(xs []float32) {
	sum := float32(0)
	for _, v := range xs {
		sum += v
	}
	if sum == 0 {
		return
	}
	invSum := 1.0 / sum
	for i := range xs {
		xs[i] *= invSum
	}
}

func resetBools(xs []bool) {
	for i := range xs {
		xs[i] = false
	}
}
